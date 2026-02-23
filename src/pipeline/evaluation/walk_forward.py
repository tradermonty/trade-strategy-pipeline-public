"""Walk-Forward (anchored expanding) OOS validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pipeline.evaluation.metrics import compute_metrics, compute_profit_factor, compute_sharpe

logger = logging.getLogger(__name__)

MIN_WARMUP_BARS = 200  # SMA200 + trend template minimum requirement

if TYPE_CHECKING:
    from pipeline.runner.pipeline import Pipeline


@dataclass
class WalkForwardSplit:
    """Result for a single Walk-Forward split."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_metrics: dict[str, float] = field(default_factory=dict)
    oos_metrics: dict[str, float] = field(default_factory=dict)
    oos_trade_count: int = 0
    oos_trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated Walk-Forward results across all splits."""

    splits: list[WalkForwardSplit]
    oos_aggregate_pf: float = 0.0
    oos_aggregate_sharpe: float = 0.0
    degradation_ratio_pf: float = 0.0
    degradation_ratio_sharpe: float = 0.0
    passed: bool = False
    all_oos_trades: list[dict[str, Any]] = field(default_factory=list)


def create_anchored_splits(
    start_date: str,
    end_date: str,
    n_splits: int = 5,
    oos_ratio: float = 0.30,
    trading_dates: pd.DatetimeIndex | None = None,
) -> list[tuple[str, str, str, str]]:
    """Generate anchored expanding Walk-Forward splits (inclusive dates).

    The last ``oos_ratio`` fraction of the total period is divided into
    ``n_splits`` equal OOS windows.  Each split's IS period starts at
    ``start_date`` and expands to the day before the OOS window.

    All returned dates are **inclusive boundaries** — both IS and OOS
    ranges include their start and end dates.

    Parameters
    ----------
    trading_dates:
        If provided, split boundaries snap to actual trading days.
        This avoids splits landing on weekends/holidays.

    Returns list of ``(is_start, is_end, oos_start, oos_end)`` date strings.
    """
    if trading_dates is not None:
        return _anchored_splits_trading(start_date, end_date, n_splits, oos_ratio, trading_dates)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days

    oos_total_days = int(total_days * oos_ratio)
    if oos_total_days < n_splits:
        raise ValueError(
            f"OOS period ({oos_total_days} days) too short for {n_splits} splits. "
            f"Need at least {n_splits} days."
        )

    first_oos_start = start + pd.Timedelta(days=total_days - oos_total_days)

    # Integer boundaries: [0, b1, b2, ..., oos_total_days]
    boundaries = [round(k * oos_total_days / n_splits) for k in range(n_splits + 1)]

    splits: list[tuple[str, str, str, str]] = []
    for k in range(n_splits):
        oos_s = first_oos_start + pd.Timedelta(days=boundaries[k])
        if k < n_splits - 1:
            oos_e = first_oos_start + pd.Timedelta(days=boundaries[k + 1] - 1)
        else:
            oos_e = end  # Last split includes the final date
        is_e = oos_s - pd.Timedelta(days=1)
        splits.append(
            (
                start.strftime("%Y-%m-%d"),
                is_e.strftime("%Y-%m-%d"),
                oos_s.strftime("%Y-%m-%d"),
                oos_e.strftime("%Y-%m-%d"),
            )
        )

    return splits


def _anchored_splits_trading(
    start_date: str,
    end_date: str,
    n_splits: int,
    oos_ratio: float,
    trading_dates: pd.DatetimeIndex,
) -> list[tuple[str, str, str, str]]:
    """Trading-day-based anchored split generation."""
    td = trading_dates.sort_values()
    # Filter to [start_date, end_date]
    start_ts = pd.Timestamp(start_date, tz=td.tz)
    end_ts = pd.Timestamp(end_date, tz=td.tz)
    td = td[(td >= start_ts) & (td <= end_ts)]

    n = len(td)
    oos_start_idx = int(n * (1 - oos_ratio))
    oos_dates = td[oos_start_idx:]
    window = len(oos_dates) // n_splits

    if window < 1:
        msg = (
            f"Too few trading dates ({len(oos_dates)}) for {n_splits} OOS splits. "
            f"Need at least {n_splits} OOS trading days."
        )
        raise ValueError(msg)

    splits: list[tuple[str, str, str, str]] = []
    fmt = "%Y-%m-%d"
    for k in range(n_splits):
        s = k * window
        e = (k + 1) * window - 1 if k < n_splits - 1 else len(oos_dates) - 1
        oos_s = oos_dates[s]
        oos_e = oos_dates[e]
        # IS end = one trading day before OOS start
        is_end_idx = oos_start_idx + s - 1
        is_end_td = td[max(0, is_end_idx)]
        splits.append(
            (
                td[0].strftime(fmt),
                is_end_td.strftime(fmt),
                oos_s.strftime(fmt),
                oos_e.strftime(fmt),
            )
        )

    return splits


def _slice_ohlcv(ohlcv: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice OHLCV DataFrame by timestamp range [start, end] (inclusive)."""
    ts = pd.to_datetime(ohlcv["timestamp"])
    tz = ts.dt.tz
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)
    mask = (ts >= start_ts) & (ts <= end_ts)
    return ohlcv[mask].reset_index(drop=True)


def _filter_signals_to_period(signals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep only signals whose timestamp falls within [start, end]."""
    ts = pd.to_datetime(signals["timestamp"])
    tz = ts.dt.tz
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)
    mask = (ts >= start_ts) & (ts <= end_ts)
    return signals[mask].reset_index(drop=True)


def _warmup_guidance(
    ohlcv: pd.DataFrame,
    oos_ratio: float,
    n_splits: int,
    split_max_bars: int,
) -> str:
    """Return actionable guidance for insufficient warmup."""
    per_symbol_total = ohlcv.groupby("symbol").size()
    max_total = int(per_symbol_total.max()) if not per_symbol_total.empty else 0
    if max_total < MIN_WARMUP_BARS:
        return (
            f"Dataset max bars per symbol is {max_total}, below required {MIN_WARMUP_BARS}; "
            "extend data history."
        )

    suggestions: list[str] = []

    # Conservative upper bound for OOS ratio based on max per-symbol history.
    max_oos_ratio = 1.0 - (MIN_WARMUP_BARS / max_total)
    max_oos_ratio = max(0.0, min(0.99, max_oos_ratio))
    if oos_ratio > max_oos_ratio:
        suggestions.append(f"Try oos_ratio <= {max_oos_ratio:.2f} (current {oos_ratio:.2f})")
    if n_splits > 1 and split_max_bars < MIN_WARMUP_BARS:
        suggestions.append(f"Reduce n_splits below {n_splits} to enlarge early IS windows")

    if suggestions:
        return f"{'; '.join(suggestions)} or extend data history."
    return "Adjust oos_ratio/n_splits or extend data history."


def run_walk_forward(
    pipeline: Pipeline,
    ohlcv: pd.DataFrame,
    n_splits: int = 5,
    oos_ratio: float = 0.30,
    fail_on_short_warmup: bool = False,
) -> WalkForwardResult:
    """Run Walk-Forward analysis across all splits.

    For each split:
    - IS: generate signals on IS data, backtest on IS data, compute metrics.
    - OOS: generate signals on IS+OOS combined data (for indicator warmup),
      filter signals to OOS period, backtest on OOS data only (no carry-over).

    Aggregate OOS metrics use concatenated daily returns (Sharpe) and
    pooled trades (Profit Factor) to avoid RangeIndex overlap issues.
    """
    dates = pd.to_datetime(ohlcv["timestamp"])
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")

    # Use unique trading dates for split boundary alignment
    unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    splits_dates = create_anchored_splits(
        start_date, end_date, n_splits, oos_ratio, trading_dates=unique_dates
    )

    split_results: list[WalkForwardSplit] = []
    all_oos_trades: list[dict[str, Any]] = []
    all_oos_returns: list[pd.Series] = []
    all_is_pf: list[float] = []
    all_is_sharpe: list[float] = []

    for split_idx, (is_start, is_end, oos_start, oos_end) in enumerate(splits_dates):
        is_ohlcv = _slice_ohlcv(ohlcv, is_start, is_end)

        # Warmup check: optional fail-fast when IS period is too short
        if len(is_ohlcv) > 0:
            per_symbol_bars = is_ohlcv.groupby("symbol").size()
            max_symbol_bars = int(per_symbol_bars.max()) if not per_symbol_bars.empty else 0
            if max_symbol_bars < MIN_WARMUP_BARS:
                guidance = _warmup_guidance(ohlcv, oos_ratio, n_splits, max_symbol_bars)
                msg = (
                    f"Split {is_start}~{is_end}: max bars per symbol is {max_symbol_bars} "
                    f"(need {MIN_WARMUP_BARS} for indicator warmup)."
                )
                if fail_on_short_warmup:
                    raise ValueError(f"{msg} {guidance}")
                logger.warning("%s IS metrics may be biased.", msg)

        # IS: generate signals + backtest (IS data only)
        if len(is_ohlcv) > 0:
            is_signals = pipeline._generate_signals(is_ohlcv)
            is_bt = pipeline._run_backtest(is_ohlcv, is_signals)
            is_metrics = compute_metrics(is_bt.equity_curve, is_bt.trades)
        else:
            is_metrics = {"sharpe": 0.0, "profit_factor": 0.0}

        # OOS: generate signals with IS warmup, backtest OOS only
        warmup_ohlcv = _slice_ohlcv(ohlcv, is_start, oos_end)
        oos_ohlcv = _slice_ohlcv(ohlcv, oos_start, oos_end)

        if len(warmup_ohlcv) > 0 and len(oos_ohlcv) > 0:
            # Generate signals on IS+OOS (indicators see full history)
            warmup_signals = pipeline._generate_signals(warmup_ohlcv)
            # Filter signals to OOS period only
            oos_signals = _filter_signals_to_period(warmup_signals, oos_start, oos_end)
            # Run fresh backtest on OOS data only (no carry-over)
            oos_bt = pipeline._run_backtest(oos_ohlcv, oos_signals)
            oos_trades = oos_bt.trades

            # Equity curve is OOS-only → use directly
            if len(oos_bt.equity_curve) > 1:
                oos_returns = oos_bt.equity_curve.pct_change().dropna()
                oos_returns = oos_returns.reset_index(drop=True)
                all_oos_returns.append(oos_returns)

            oos_metrics = compute_metrics(oos_bt.equity_curve, oos_trades)
        else:
            oos_metrics = {"sharpe": 0.0, "profit_factor": 0.0}
            oos_trades = []

        # Stamp split_index and create independent copies to prevent reference sharing
        split_trades = [dict(t, split_index=split_idx) for t in oos_trades]
        all_oos_trades.extend([dict(t) for t in split_trades])

        all_is_pf.append(is_metrics.get("profit_factor", 0.0))
        all_is_sharpe.append(is_metrics.get("sharpe", 0.0))

        split_results.append(
            WalkForwardSplit(
                train_start=is_start,
                train_end=is_end,
                test_start=oos_start,
                test_end=oos_end,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                oos_trade_count=len(oos_trades),
                oos_trades=split_trades,
            )
        )

    # --- Aggregate OOS metrics ---

    # PF: from pooled OOS trades
    if all_oos_trades:
        oos_pnls = [t["pnl"] for t in all_oos_trades]
        oos_pf = compute_profit_factor(oos_pnls)
        if not np.isfinite(oos_pf):
            oos_pf = 0.0
    else:
        oos_pf = 0.0

    # Sharpe: from concatenated daily returns (sequential, no index overlap)
    if all_oos_returns:
        combined_returns = pd.concat(all_oos_returns, ignore_index=True)
        oos_sharpe = compute_sharpe(combined_returns)
    else:
        oos_sharpe = 0.0

    # IS aggregates (mean across splits, excluding inf)
    finite_is_pf = [x for x in all_is_pf if np.isfinite(x) and x > 0]
    is_pf_mean = sum(finite_is_pf) / len(finite_is_pf) if finite_is_pf else 0.0
    is_sharpe_mean = sum(all_is_sharpe) / len(all_is_sharpe) if all_is_sharpe else 0.0

    # Degradation ratios
    deg_pf = oos_pf / is_pf_mean if is_pf_mean > 0 else 0.0
    deg_sharpe = oos_sharpe / is_sharpe_mean if is_sharpe_mean != 0 else 0.0

    return WalkForwardResult(
        splits=split_results,
        oos_aggregate_pf=oos_pf,
        oos_aggregate_sharpe=oos_sharpe,
        degradation_ratio_pf=deg_pf,
        degradation_ratio_sharpe=deg_sharpe,
        passed=deg_pf > 0.5,
        all_oos_trades=all_oos_trades,
    )
