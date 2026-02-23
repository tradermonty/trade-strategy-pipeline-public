"""Backtest engine — custom event-driven simulation.

Implements next-day-open execution, fill rules from Section 4.4,
and integrates with CostModel and FillSimulator.

Supports two entry modes:
  - next_day_open (default): signal at T → enter at T+1 Open
  - same_bar_open: signal with entry_at_open=True → enter at T Open

Supports two trailing modes:
  - daily (default): trailing stop from 21-EMA / 10-day low, checked every bar
  - weekly: weekly trend break checked at week end → pending_exit → exit next bar Open

Portfolio mode (run_portfolio):
  - Shared capital across all symbols on a daily timeline
  - Score-based entry priority when max_positions constrained
  - Missing-bar handling with last_known_close evaluation
  - Force exit after max_missing_bars consecutive missing days
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pipeline.backtest.cost_model import CostModel, FillSimulator, PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    equity_curve: pd.Series
    trades: list[dict[str, Any]]
    initial_capital: float
    final_capital: float
    max_concurrent_positions: int = 0


class BacktestEngine:
    """Custom backtest engine with next-day-open execution and fill rules."""

    def __init__(
        self,
        initial_capital: float = 25_000.0,
        commission_per_share: float = 0.0,
        slippage_bps: int = 5,
        max_positions: int = 5,
        risk_per_trade: float = 0.01,
        stop_loss_pct: float = 0.07,
        take_profit_rr: float = 3.0,
        breakeven_at_rr: float | None = None,
        trailing_start_bars: int = 0,
        trailing_ema_window: int = 21,
        trailing_low_window: int = 10,
        entry_mode: str = "next_day_open",
        trailing_mode: str = "daily",
        weekly_ema_period: int = 10,
        weekly_nweek_low_period: int = 4,
        weekly_transition_weeks: int = 3,
        weekly_trailing_type: str = "weekly_ema",
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_model = CostModel(commission_per_share, slippage_bps)
        self.fill_sim = FillSimulator()
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_rr = take_profit_rr
        self.breakeven_at_rr = breakeven_at_rr
        self.trailing_start_bars = trailing_start_bars
        self.trailing_ema_window = trailing_ema_window
        self.trailing_low_window = trailing_low_window
        self.entry_mode = entry_mode
        self.trailing_mode = trailing_mode
        self.weekly_ema_period = weekly_ema_period
        self.weekly_nweek_low_period = weekly_nweek_low_period
        self.weekly_transition_weeks = weekly_transition_weeks
        self.weekly_trailing_type = weekly_trailing_type

    def _process_entry(
        self,
        bar_idx: int,
        bar: pd.Series,
        open_price: float,
        close_price: float,
        stop_price_override: float | None,
        cash: float,
        positions: list[dict[str, Any]],
    ) -> float:
        """Process a single entry. Returns updated cash. Mutates positions list."""
        if len(positions) >= self.max_positions:
            return cash

        entry_price = self.cost_model.apply_slippage(open_price, side="buy")
        if stop_price_override is not None:
            stop_price = stop_price_override
        else:
            stop_price = entry_price * (1 - self.stop_loss_pct)
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return cash

        target_price = entry_price + (risk_per_share * self.take_profit_rr)

        sizer = PositionSizer(
            portfolio_value=cash + self._position_value(positions, close_price),
            risk_per_trade=self.risk_per_trade,
            max_positions=self.max_positions,
        )
        shares = sizer.calculate(entry_price, stop_price)

        if shares > 0 and shares * entry_price <= cash:
            cost = self.cost_model.commission(shares, entry_price)
            cash -= shares * entry_price + cost
            positions.append(
                {
                    "entry_idx": bar_idx,
                    "entry_date": bar.get("timestamp"),
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "initial_stop": stop_price,
                    "target_price": target_price,
                    "trailing_stop_price": None,
                    "shares": shares,
                    "pending_exit": None,
                }
            )

        return cash

    def run(self, ohlcv: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """Run backtest on OHLCV data with entry signals.

        Args:
            ohlcv: DataFrame with open, high, low, close, volume columns.
            signals: DataFrame with 'entry' boolean column indicating signal bars.
                     Optional 'entry_at_open' boolean for same-bar-open entries.
                     Optional 'stop_price' float for signal-level stop override.

        Returns:
            BacktestResult with equity curve, trade list, and stats.
        """
        cash = self.initial_capital
        positions: list[dict[str, Any]] = []
        completed_trades: list[dict[str, Any]] = []
        equity_values: list[float] = []
        max_concurrent = 0
        pending_entry = False
        pending_stop_price: float | None = None

        # Pre-compute weekly data if needed
        weekly_df = None
        if self.trailing_mode == "weekly":
            from pipeline.backtest.weekly_trailing import aggregate_to_weekly

            weekly_df = aggregate_to_weekly(ohlcv)

        n = len(ohlcv)

        for i in range(n):
            bar = ohlcv.iloc[i]
            open_price = float(bar["open"])
            high_price = float(bar["high"])
            low_price = float(bar["low"])
            close_price = float(bar["close"])

            # Step 1: Process pending_exit (weekly trailing break → exit at this bar's Open)
            exit_indices: list[int] = []
            for j, pos in enumerate(positions):
                if pos.get("pending_exit") is not None:
                    exit_price = self.cost_model.apply_slippage(open_price, side="sell")
                    shares = pos["shares"]
                    cost = self.cost_model.commission(shares, exit_price)
                    pnl = (exit_price - pos["entry_price"]) * shares - cost
                    cash += shares * exit_price - cost
                    completed_trades.append(
                        {
                            "entry_date": pos["entry_date"],
                            "exit_date": bar.get("timestamp"),
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "exit_reason": pos["pending_exit"],
                            "bars_held": self._bars_held(pos.get("entry_idx"), i),
                        }
                    )
                    exit_indices.append(j)

            for j in sorted(exit_indices, reverse=True):
                positions.pop(j)

            # Step 2: Process pending entry (next-day-open from previous bar signal)
            if pending_entry:
                cash = self._process_entry(
                    i, bar, open_price, close_price, pending_stop_price, cash, positions
                )

            pending_entry = False
            pending_stop_price = None

            # Step 3: Same-bar-open entry (only when entry_mode allows it)
            if self.entry_mode == "same_bar_open" and i < len(signals):
                sig_row = signals.iloc[i]
                if sig_row.get("entry", False) and sig_row.get("entry_at_open", False):
                    sig_stop = (
                        float(sig_row["stop_price"])
                        if "stop_price" in sig_row.index and pd.notna(sig_row.get("stop_price"))
                        else None
                    )
                    cash = self._process_entry(
                        i, bar, open_price, close_price, sig_stop, cash, positions
                    )

            # Step 4: Check exits for all active positions (skip entry-bar positions)
            closed_indices: list[int] = []
            for j, pos in enumerate(positions):
                if pos["entry_idx"] == i:
                    continue

                bars_held = self._bars_held(pos.get("entry_idx"), i)

                # Breakeven stop
                if self.breakeven_at_rr is not None:
                    risk_per_share = pos["entry_price"] - pos["initial_stop"]
                    be_trigger = pos["entry_price"] + risk_per_share * self.breakeven_at_rr
                    if high_price >= be_trigger:
                        pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])

                # Daily trailing stop (only in daily mode)
                if (
                    self.trailing_mode == "daily"
                    and i > 0
                    and bars_held >= self.trailing_start_bars
                ):
                    ema_w = self.trailing_ema_window
                    low_w = self.trailing_low_window
                    lookback_start = max(0, i - ema_w)
                    closes = ohlcv["close"].iloc[lookback_start : i + 1].astype(float)
                    ema_val = closes.ewm(span=ema_w, min_periods=1).mean().iloc[-1]
                    low_start = max(0, i - low_w)
                    low_val = float(ohlcv["low"].iloc[low_start : i + 1].min())
                    pos["trailing_stop_price"] = max(ema_val, low_val)

                exit_result = self.fill_sim.resolve_exit(
                    stop_price=pos["stop_price"],
                    target_price=pos["target_price"],
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    trailing_stop_price=pos["trailing_stop_price"],
                )

                if exit_result["exit_reason"] is not None:
                    exit_price = self.cost_model.apply_slippage(
                        exit_result["fill_price"], side="sell"
                    )
                    shares = pos["shares"]
                    cost = self.cost_model.commission(shares, exit_price)
                    pnl = (exit_price - pos["entry_price"]) * shares - cost

                    cash += shares * exit_price - cost
                    completed_trades.append(
                        {
                            "entry_date": pos["entry_date"],
                            "exit_date": bar.get("timestamp"),
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "exit_reason": exit_result["exit_reason"],
                            "bars_held": self._bars_held(pos.get("entry_idx"), i),
                        }
                    )
                    closed_indices.append(j)

            for j in sorted(closed_indices, reverse=True):
                positions.pop(j)

            # Step 4b: Weekly trailing check (at week-end bars only)
            if self.trailing_mode == "weekly" and weekly_df is not None:
                ts = bar.get("timestamp")
                if ts is not None and _is_week_end_bar(ohlcv, i):
                    from pipeline.backtest.weekly_trailing import check_weekly_trailing_stop

                    for pos in positions:
                        if pos["entry_idx"] == i:
                            continue
                        if pos.get("pending_exit") is not None:
                            continue
                        triggered = check_weekly_trailing_stop(
                            close_price=close_price,
                            weekly_df=weekly_df,
                            entry_date=pd.Timestamp(pos["entry_date"]),
                            current_date=pd.Timestamp(ts),
                            trailing_type=self.weekly_trailing_type,
                            ema_period=self.weekly_ema_period,
                            nweek_low_period=self.weekly_nweek_low_period,
                            transition_weeks=self.weekly_transition_weeks,
                        )
                        if triggered:
                            pos["pending_exit"] = "weekly_trailing"

            # Step 5: Check if today's close generates entry signal for tomorrow
            if i < len(signals):
                sig_row = signals.iloc[i]
                is_entry = sig_row.get("entry", False)
                is_at_open = sig_row.get("entry_at_open", False)
                # In same_bar_open mode, entry_at_open signals were handled in Step 3.
                # In next_day_open mode, ALL entries (including entry_at_open) become T+1.
                should_pend = is_entry and (self.entry_mode == "next_day_open" or not is_at_open)
                if should_pend:
                    pending_entry = True
                    pending_stop_price = (
                        float(sig_row["stop_price"])
                        if "stop_price" in sig_row.index and pd.notna(sig_row.get("stop_price"))
                        else None
                    )

            # Record equity
            portfolio_value = cash + self._position_value(positions, close_price)
            equity_values.append(portfolio_value)
            max_concurrent = max(max_concurrent, len(positions))

        equity_curve = pd.Series(equity_values, name="equity")

        return BacktestResult(
            equity_curve=equity_curve,
            trades=completed_trades,
            initial_capital=self.initial_capital,
            final_capital=equity_values[-1] if equity_values else self.initial_capital,
            max_concurrent_positions=max_concurrent,
        )

    @staticmethod
    def _position_value(positions: list[dict[str, Any]], current_price: float) -> float:
        """Calculate total market value of open positions."""
        return float(sum(pos["shares"] * current_price for pos in positions))

    @staticmethod
    def _bars_held(entry_idx: int | None, current_idx: int | None) -> int:
        """Count held bars using per-symbol trading-bar indexes."""
        if isinstance(entry_idx, int) and current_idx is not None:
            return max(0, current_idx - entry_idx)
        return 0

    @staticmethod
    def _portfolio_value(
        positions: list[dict[str, Any]],
        close_prices: dict[str, float],
    ) -> float:
        """Calculate portfolio value using per-symbol close prices."""
        return float(sum(pos["shares"] * close_prices.get(pos["symbol"], 0.0) for pos in positions))

    def _process_portfolio_entry(
        self,
        symbol: str,
        date: pd.Timestamp,
        open_price: float,
        stop_price_override: float | None,
        entry_bar_idx: int | None,
        cash: float,
        positions: list[dict[str, Any]],
        close_prices: dict[str, float],
    ) -> float:
        """Process entry using portfolio-level valuation. Returns updated cash."""
        if len(positions) >= self.max_positions:
            return cash
        entry_price = self.cost_model.apply_slippage(open_price, side="buy")
        stop_price = stop_price_override or entry_price * (1 - self.stop_loss_pct)
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return cash
        target_price = entry_price + risk_per_share * self.take_profit_rr
        portfolio_val = cash + self._portfolio_value(positions, close_prices)
        sizer = PositionSizer(portfolio_val, self.risk_per_trade, self.max_positions)
        shares = sizer.calculate(entry_price, stop_price)
        if shares > 0 and shares * entry_price <= cash:
            cost = self.cost_model.commission(shares, entry_price)
            cash -= shares * entry_price + cost
            positions.append(
                {
                    "symbol": symbol,
                    "entry_idx": entry_bar_idx,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "initial_stop": stop_price,
                    "target_price": target_price,
                    "trailing_stop_price": None,
                    "shares": shares,
                    "pending_exit": None,
                }
            )
        return cash

    def run_portfolio(
        self,
        ohlcv: pd.DataFrame,
        signals: pd.DataFrame,
        max_missing_bars: int = 10,
    ) -> BacktestResult:
        """Run portfolio-level backtest with shared capital across symbols.

        Args:
            ohlcv: Multi-symbol DataFrame with symbol, timestamp, OHLCV columns.
            signals: Multi-symbol DataFrame with symbol, timestamp, entry, stop_price,
                     entry_at_open, and optional score columns.
            max_missing_bars: Force-exit positions after this many consecutive
                              days without a bar for the symbol.

        Returns:
            BacktestResult with DatetimeIndex equity curve.
        """
        cash = self.initial_capital
        positions: list[dict[str, Any]] = []
        completed_trades: list[dict[str, Any]] = []
        max_concurrent = 0

        # Build per-date, per-symbol lookup tables
        ohlcv = ohlcv.copy()
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
        signals = signals.copy()
        signals["timestamp"] = pd.to_datetime(signals["timestamp"])

        all_dates = sorted(pd.Timestamp(d) for d in ohlcv["timestamp"].unique())

        # Per-symbol trading-day index for bars_held calculation
        symbol_bar_idx: dict[tuple[str, pd.Timestamp], int] = {}
        for symbol in ohlcv["symbol"].unique():
            sym_ts = ohlcv[ohlcv["symbol"] == symbol]["timestamp"].unique()
            sym_dates = sorted(pd.Timestamp(d) for d in sym_ts)
            for idx, d in enumerate(sym_dates):
                symbol_bar_idx[(symbol, d)] = idx

        # Pre-index bars and signals by (date, symbol)
        bars_lookup: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
        for _, row in ohlcv.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            symbol = str(row["symbol"])
            key = (ts, symbol)
            bars_lookup[key] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "bar_idx": symbol_bar_idx.get((symbol, ts)),
            }

        sig_lookup: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
        for _, row in signals.iterrows():
            if row.get("entry", False):
                key = (row["timestamp"], row["symbol"])
                sig_lookup[key] = {
                    "stop_price": (
                        float(row["stop_price"]) if pd.notna(row.get("stop_price")) else None
                    ),
                    "entry_at_open": bool(row.get("entry_at_open", False)),
                    "score": float(row.get("score", 0.0)),
                }

        # Pre-compute week-end dates per symbol
        week_end_dates_by_symbol: dict[str, set[pd.Timestamp]] = {}
        for symbol in ohlcv["symbol"].unique():
            sym_dates = sorted(ohlcv[ohlcv["symbol"] == symbol]["timestamp"].unique())
            ends: set[pd.Timestamp] = set()
            for i, d in enumerate(sym_dates):
                if i == len(sym_dates) - 1 or (
                    pd.Timestamp(d).isocalendar()[1]
                    != pd.Timestamp(sym_dates[i + 1]).isocalendar()[1]
                ):
                    ends.add(d)
            week_end_dates_by_symbol[symbol] = ends

        # Pre-compute weekly data per symbol for weekly trailing
        weekly_dfs: dict[str, pd.DataFrame] = {}
        if self.trailing_mode == "weekly":
            from pipeline.backtest.weekly_trailing import aggregate_to_weekly

            for symbol in ohlcv["symbol"].unique():
                sym_ohlcv = ohlcv[ohlcv["symbol"] == symbol].sort_values("timestamp")
                if len(sym_ohlcv) > 0:
                    weekly_dfs[symbol] = aggregate_to_weekly(sym_ohlcv)

        # Pending entries: symbol → {stop_price, score, entry_at_open}
        pending_entries: dict[str, dict[str, Any]] = {}
        # Last known close per symbol
        last_known_close: dict[str, float] = {}
        # Last known per-symbol trading-day index
        last_known_bar_idx: dict[str, int] = {}
        # Missing bar counter per symbol
        missing_bar_count: dict[str, int] = defaultdict(int)

        equity_values: list[float] = []
        equity_dates: list[pd.Timestamp] = []

        for date in all_dates:
            # Determine which symbols have bars today
            today_bars: dict[str, dict[str, Any]] = {}
            for symbol in ohlcv["symbol"].unique():
                key = (date, symbol)
                if key in bars_lookup:
                    today_bars[symbol] = bars_lookup[key]
                    last_known_close[symbol] = bars_lookup[key]["close"]
                    bar_idx = bars_lookup[key]["bar_idx"]
                    if isinstance(bar_idx, int):
                        last_known_bar_idx[symbol] = bar_idx

            # Update missing bar counts for active positions
            active_symbols = {pos["symbol"] for pos in positions}
            for sym in active_symbols:
                if sym not in today_bars:
                    missing_bar_count[sym] += 1
                else:
                    missing_bar_count[sym] = 0

            # Force-exit positions with too many consecutive missing bars
            force_exit_indices: list[int] = []
            for j, pos in enumerate(positions):
                sym = pos["symbol"]
                if missing_bar_count[sym] >= max_missing_bars:
                    exit_price = last_known_close.get(sym, pos["entry_price"])
                    shares = pos["shares"]
                    cost = self.cost_model.commission(shares, exit_price)
                    pnl = (exit_price - pos["entry_price"]) * shares - cost
                    held = self._bars_held(pos.get("entry_idx"), last_known_bar_idx.get(sym))
                    cash += shares * exit_price - cost
                    completed_trades.append(
                        {
                            "symbol": sym,
                            "entry_date": pos["entry_date"],
                            "exit_date": date,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "exit_reason": "force_exit_missing",
                            "bars_held": held,
                        }
                    )
                    force_exit_indices.append(j)
            for j in sorted(force_exit_indices, reverse=True):
                positions.pop(j)

            # Step 1: Process pending_exit → cash recovery + slot release
            exit_indices: list[int] = []
            for j, pos in enumerate(positions):
                if pos.get("pending_exit") is None:
                    continue
                sym = pos["symbol"]
                if sym not in today_bars:
                    continue  # Hold pending_exit until bar available
                bar = today_bars[sym]
                bar_idx = bar.get("bar_idx")
                held = self._bars_held(
                    pos.get("entry_idx"), bar_idx if isinstance(bar_idx, int) else None
                )
                exit_price = self.cost_model.apply_slippage(bar["open"], side="sell")
                shares = pos["shares"]
                cost = self.cost_model.commission(shares, exit_price)
                pnl = (exit_price - pos["entry_price"]) * shares - cost
                cash += shares * exit_price - cost
                completed_trades.append(
                    {
                        "symbol": sym,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": shares,
                        "pnl": pnl,
                        "exit_reason": pos["pending_exit"],
                        "bars_held": held,
                    }
                )
                exit_indices.append(j)
            for j in sorted(exit_indices, reverse=True):
                positions.pop(j)

            # Step 2: Process pending entries (previous day signals, score order)
            sorted_pending = sorted(
                pending_entries.values(), key=lambda x: (-x["score"], x["symbol"])
            )
            remaining_pending: dict[str, dict[str, Any]] = {}
            for pend in sorted_pending:
                sym = pend["symbol"]
                if sym not in today_bars:
                    remaining_pending[sym] = pend  # Carry over
                    continue
                bar = today_bars[sym]
                cash = self._process_portfolio_entry(
                    sym,
                    date,
                    bar["open"],
                    pend["stop_price"],
                    bar["bar_idx"] if isinstance(bar.get("bar_idx"), int) else None,
                    cash,
                    positions,
                    {s: last_known_close.get(s, 0.0) for s in last_known_close},
                )
            pending_entries = remaining_pending

            # Step 3: Same-bar-open entry (today's signals with entry_at_open)
            same_bar_signals: list[dict[str, Any]] = []
            for sym in today_bars:
                key = (date, sym)
                if key in sig_lookup and sig_lookup[key]["entry_at_open"]:
                    same_bar_signals.append({"symbol": sym, **sig_lookup[key]})
            same_bar_signals.sort(key=lambda x: (-x["score"], x["symbol"]))
            for sig in same_bar_signals:
                sym = sig["symbol"]
                bar = today_bars[sym]
                cash = self._process_portfolio_entry(
                    sym,
                    date,
                    bar["open"],
                    sig["stop_price"],
                    bar["bar_idx"] if isinstance(bar.get("bar_idx"), int) else None,
                    cash,
                    positions,
                    {s: last_known_close.get(s, 0.0) for s in last_known_close},
                )

            # Step 4: Exit checks (SL/TP) for positions with bars today
            closed_indices: list[int] = []
            for j, pos in enumerate(positions):
                sym = pos["symbol"]
                if sym not in today_bars:
                    continue  # No bar → skip exit check
                if pos["entry_date"] == date:
                    continue  # Skip same-bar exit
                bar = today_bars[sym]
                exit_result = self.fill_sim.resolve_exit(
                    stop_price=pos["stop_price"],
                    target_price=pos["target_price"],
                    open_price=bar["open"],
                    high_price=bar["high"],
                    low_price=bar["low"],
                    trailing_stop_price=pos["trailing_stop_price"],
                )
                if exit_result["exit_reason"] is not None:
                    bar_idx = bar.get("bar_idx")
                    held = self._bars_held(
                        pos.get("entry_idx"), bar_idx if isinstance(bar_idx, int) else None
                    )
                    exit_price = self.cost_model.apply_slippage(
                        exit_result["fill_price"], side="sell"
                    )
                    shares = pos["shares"]
                    cost = self.cost_model.commission(shares, exit_price)
                    pnl = (exit_price - pos["entry_price"]) * shares - cost
                    cash += shares * exit_price - cost
                    completed_trades.append(
                        {
                            "symbol": sym,
                            "entry_date": pos["entry_date"],
                            "exit_date": date,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "exit_reason": exit_result["exit_reason"],
                            "bars_held": held,
                        }
                    )
                    closed_indices.append(j)
            for j in sorted(closed_indices, reverse=True):
                positions.pop(j)

            # Step 5: Weekly trailing check (week-end bars per symbol)
            if self.trailing_mode == "weekly":
                from pipeline.backtest.weekly_trailing import (
                    check_weekly_trailing_stop,
                )

                for pos in positions:
                    sym = pos["symbol"]
                    if sym not in today_bars:
                        continue
                    if pos["entry_date"] == date:
                        continue
                    if pos.get("pending_exit") is not None:
                        continue
                    if date not in week_end_dates_by_symbol.get(sym, set()):
                        continue
                    wdf = weekly_dfs.get(sym)
                    if wdf is None:
                        continue
                    close_price = today_bars[sym]["close"]
                    triggered = check_weekly_trailing_stop(
                        close_price=close_price,
                        weekly_df=wdf,
                        entry_date=pd.Timestamp(pos["entry_date"]),
                        current_date=pd.Timestamp(date),
                        trailing_type=self.weekly_trailing_type,
                        ema_period=self.weekly_ema_period,
                        nweek_low_period=self.weekly_nweek_low_period,
                        transition_weeks=self.weekly_transition_weeks,
                    )
                    if triggered:
                        pos["pending_exit"] = "weekly_trailing"

            # Step 6: Register next-day signals (pending entries)
            for sym in today_bars:
                key = (date, sym)
                if key not in sig_lookup:
                    continue
                sig = sig_lookup[key]
                if sig["entry_at_open"]:
                    continue  # Already handled in Step 3
                # next_day_open signal → register as pending
                pending_entries[sym] = {
                    "symbol": sym,
                    "stop_price": sig["stop_price"],
                    "score": sig["score"],
                }

            # Step 7: Record equity
            close_prices = {s: last_known_close.get(s, 0.0) for s in last_known_close}
            portfolio_val = cash + self._portfolio_value(positions, close_prices)
            equity_values.append(portfolio_val)
            equity_dates.append(date)
            max_concurrent = max(max_concurrent, len(positions))

        equity_curve = pd.Series(
            equity_values,
            index=pd.DatetimeIndex(equity_dates),
            name="equity",
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=completed_trades,
            initial_capital=self.initial_capital,
            final_capital=equity_values[-1] if equity_values else self.initial_capital,
            max_concurrent_positions=max_concurrent,
        )


def _is_week_end_bar(ohlcv: pd.DataFrame, idx: int) -> bool:
    """Check if bar at idx is the last trading day of its ISO week."""
    ts_val = ohlcv.iloc[idx].get("timestamp")
    ts = pd.Timestamp(ts_val)  # type: ignore[arg-type]
    if idx >= len(ohlcv) - 1:
        return True
    ts_next_val = ohlcv.iloc[idx + 1].get("timestamp")
    ts_next = pd.Timestamp(ts_next_val)  # type: ignore[arg-type]
    return bool(ts.isocalendar()[1] != ts_next.isocalendar()[1])
