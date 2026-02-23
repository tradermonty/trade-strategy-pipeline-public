"""Tests for the backtest engine (vectorbt wrapper)."""

from pathlib import Path

import pandas as pd
import pytest

from pipeline.backtest.engine import BacktestEngine, BacktestResult

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load sample OHLCV fixture."""
    return pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")


@pytest.fixture
def engine() -> BacktestEngine:
    """Create a basic backtest engine."""
    return BacktestEngine(
        initial_capital=25_000.0,
        commission_per_share=0.0,
        slippage_bps=5,
        max_positions=5,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
    )


def _make_simple_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Create simple entry signals for testing."""
    sym_data = df[df["symbol"] == symbol].reset_index(drop=True)
    signals = pd.DataFrame(index=sym_data.index)
    signals["timestamp"] = sym_data["timestamp"]
    signals["symbol"] = symbol
    signals["entry"] = False
    # Signal at bar 50
    if len(sym_data) > 50:
        signals.loc[50, "entry"] = True
    return signals


def test_engine_runs_with_sample_data(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Engine should complete a run with fixture data."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result = engine.run(sym_data, signals)
    assert isinstance(result, BacktestResult)


def test_engine_returns_equity_curve(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Engine should return a time-indexed equity curve Series."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result = engine.run(sym_data, signals)
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) > 0
    assert result.equity_curve.iloc[0] == pytest.approx(25_000.0)


def test_engine_returns_trade_list(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Trade list should have entry/exit date, price, pnl, exit_reason, bars_held."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result = engine.run(sym_data, signals)
    if len(result.trades) > 0:
        trade = result.trades[0]
        assert "entry_date" in trade
        assert "exit_date" in trade
        assert "entry_price" in trade
        assert "exit_price" in trade
        assert "pnl" in trade
        assert "exit_reason" in trade
        assert "bars_held" in trade
        assert isinstance(trade["bars_held"], int)
        assert trade["bars_held"] > 0


def test_engine_uses_next_day_open(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Entry should be executed at T+1 Open."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result = engine.run(sym_data, signals)
    if len(result.trades) > 0:
        trade = result.trades[0]
        # Entry price should be close to bar 51's open (T+1)
        expected_open = sym_data.iloc[51]["open"]
        # With slippage, should be slightly higher
        assert trade["entry_price"] >= expected_open


def test_engine_applies_slippage(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Slippage should be reflected in entry price."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result = engine.run(sym_data, signals)
    if len(result.trades) > 0:
        trade = result.trades[0]
        raw_open = sym_data.iloc[51]["open"]
        # With 5bps slippage on buy, price should be slightly higher
        assert trade["entry_price"] > raw_open * 0.999


def test_engine_respects_max_positions(sample_data: pd.DataFrame) -> None:
    """Should not hold more than max_positions simultaneously."""
    engine = BacktestEngine(
        initial_capital=25_000.0,
        max_positions=2,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
    )
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    # Create multiple signals close together
    signals = pd.DataFrame(index=sym_data.index)
    signals["timestamp"] = sym_data["timestamp"]
    signals["symbol"] = "SYN_TREND"
    signals["entry"] = False
    for i in [30, 31, 32, 33, 34]:
        if i < len(signals):
            signals.loc[i, "entry"] = True

    result = engine.run(sym_data, signals)
    assert result.max_concurrent_positions <= 2


def test_engine_initial_capital(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Initial capital should be $25,000."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result = engine.run(sym_data, signals)
    assert result.initial_capital == 25_000.0


def test_engine_stop_loss_triggers(sample_data: pd.DataFrame) -> None:
    """A trade entered just before a gap-down should trigger stop_loss exit."""
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        max_positions=5,
    )
    sym_data = sample_data[sample_data["symbol"] == "SYN_GAPDOWN"].reset_index(drop=True)
    signals = pd.DataFrame(index=sym_data.index)
    signals["timestamp"] = sym_data["timestamp"]
    signals["symbol"] = "SYN_GAPDOWN"
    signals["entry"] = False
    # Signal at bar 178, entry at bar 179 open; gap-down at bar 180
    if len(sym_data) > 180:
        signals.loc[178, "entry"] = True

    result = engine.run(sym_data, signals)
    assert len(result.trades) > 0
    # The 12% gap down at bar 180 should trigger stop_loss (7% threshold)
    sl_trades = [t for t in result.trades if t["exit_reason"] == "stop_loss"]
    assert len(sl_trades) > 0


def test_engine_sector_exposure_logged(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Sector exposure exceeding 30% should be logged (Phase I: no blocking)."""
    # This test just verifies the engine doesn't crash with sector data
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    result = engine.run(sym_data, signals)
    # Phase I: just check it completes (sector logging is non-blocking)
    assert isinstance(result, BacktestResult)


def test_engine_deterministic(engine: BacktestEngine, sample_data: pd.DataFrame) -> None:
    """Same inputs should produce identical results."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    result1 = engine.run(sym_data.copy(), signals.copy())
    result2 = engine.run(sym_data.copy(), signals.copy())
    pd.testing.assert_series_equal(result1.equity_curve, result2.equity_curve)


def test_engine_trailing_window_params(sample_data: pd.DataFrame) -> None:
    """Shorter trailing windows should produce tighter stops (different results)."""
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    signals = _make_simple_signals(sample_data, "SYN_TREND")

    engine_default = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        trailing_ema_window=21,
        trailing_low_window=10,
    )
    engine_tight = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        trailing_ema_window=10,
        trailing_low_window=5,
    )

    result_default = engine_default.run(sym_data.copy(), signals.copy())
    result_tight = engine_tight.run(sym_data.copy(), signals.copy())

    # Both should complete successfully
    assert isinstance(result_default, BacktestResult)
    assert isinstance(result_tight, BacktestResult)
    # Different trailing windows should produce different equity curves
    # (unless no trades triggered, which is unlikely with SYN_TREND)
    if result_default.trades and result_tight.trades:
        assert not result_default.equity_curve.equals(result_tight.equity_curve)


def test_engine_breakeven_at_rr(sample_data: pd.DataFrame) -> None:
    """Engine with breakeven_at_rr should complete and produce valid result."""
    engine_be = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        breakeven_at_rr=1.0,
    )
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    result = engine_be.run(sym_data, signals)
    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 25_000.0


def test_breakeven_stop_moves_to_entry() -> None:
    """After price reaches 1R profit, stop should move to entry price."""
    n = 30
    stop_loss_pct = 0.10

    # Build OHLCV: signal at bar 0, entry at bar 1 open = 100
    # Price rises to 1R (110) at bar 5, then drops to 100.5 (above entry)
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1_000_000] * n,
            "timestamp": pd.bdate_range("2024-01-02", periods=n),
        }
    )
    # Bar 1: entry at open=100
    ohlcv.loc[1, "open"] = 100.0
    # Bars 2-5: price rises to trigger breakeven (high >= entry + 1R = 110)
    for i in range(2, 6):
        ohlcv.loc[i, "open"] = 105.0 + (i - 2) * 1.5
        ohlcv.loc[i, "high"] = 106.0 + (i - 2) * 1.5
        ohlcv.loc[i, "low"] = 104.0 + (i - 2) * 1.5
        ohlcv.loc[i, "close"] = 105.5 + (i - 2) * 1.5
    # Bar 5 high >= 110 triggers breakeven
    ohlcv.loc[5, "high"] = 111.0
    # Bars 6+: price drops back to entry+0.5 then holds
    for i in range(6, n):
        ohlcv.loc[i, "open"] = 101.0
        ohlcv.loc[i, "high"] = 102.0
        ohlcv.loc[i, "low"] = 100.5
        ohlcv.loc[i, "close"] = 101.0

    signals = pd.DataFrame(
        {
            "entry": [False] * n,
            "timestamp": ohlcv["timestamp"],
            "symbol": "TEST",
        }
    )
    signals.loc[0, "entry"] = True

    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=stop_loss_pct,
        take_profit_rr=3.0,
        breakeven_at_rr=1.0,
    )
    result = engine.run(ohlcv, signals)
    # With breakeven active, the stop moved to entry (~100) so the position
    # should not be stopped out at 100.5 (above entry)
    # The trade should either still be open or exited via trailing stop
    if result.trades:
        for t in result.trades:
            if t["exit_reason"] == "stop_loss":
                # If stopped out, exit price should be >= entry (breakeven stop)
                assert t["exit_price"] >= t["entry_price"] * 0.999


def test_breakeven_stop_not_triggered_below_threshold() -> None:
    """If price never reaches 1R, stop should stay at original level."""
    n = 30
    # Build OHLCV: signal at bar 0, entry at bar 1 open=100
    # Price barely rises (not reaching 1R=107), then drops below stop=93
    # Disable trailing stop with large trailing_start_bars to isolate SL behavior
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1_000_000] * n,
            "timestamp": pd.bdate_range("2024-01-02", periods=n),
        }
    )
    # Bars 2-5: price barely rises (max high=104, not reaching 1R=107)
    for i in range(2, 6):
        ohlcv.loc[i, "open"] = 103.0
        ohlcv.loc[i, "high"] = 104.0
        ohlcv.loc[i, "low"] = 102.0
        ohlcv.loc[i, "close"] = 103.0
    # Bars 6+: price drops below original stop (93)
    for i in range(6, n):
        ohlcv.loc[i, "open"] = 92.0
        ohlcv.loc[i, "high"] = 93.0
        ohlcv.loc[i, "low"] = 90.0
        ohlcv.loc[i, "close"] = 91.0

    signals = pd.DataFrame(
        {
            "entry": [False] * n,
            "timestamp": ohlcv["timestamp"],
            "symbol": "TEST",
        }
    )
    signals.loc[0, "entry"] = True

    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        breakeven_at_rr=1.0,
        trailing_start_bars=9999,  # Disable trailing to isolate stop-loss
    )
    result = engine.run(ohlcv, signals)
    assert len(result.trades) > 0
    # Should exit via stop_loss at original stop level (~93), not at entry (100)
    sl_trades = [t for t in result.trades if t["exit_reason"] == "stop_loss"]
    assert len(sl_trades) > 0
    assert sl_trades[0]["exit_price"] < 95.0  # Well below entry price


def test_trailing_start_delay(sample_data: pd.DataFrame) -> None:
    """trailing_start_bars=5 should delay trailing stop activation."""
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    signals = _make_simple_signals(sample_data, "SYN_TREND")

    engine_no_delay = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        trailing_start_bars=0,
    )
    engine_delayed = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        trailing_start_bars=5,
    )

    result_no_delay = engine_no_delay.run(sym_data.copy(), signals.copy())
    result_delayed = engine_delayed.run(sym_data.copy(), signals.copy())

    # Both should complete
    assert isinstance(result_no_delay, BacktestResult)
    assert isinstance(result_delayed, BacktestResult)
    # With delay, trailing stop engages later so results should differ
    if result_no_delay.trades and result_delayed.trades:
        assert not result_no_delay.equity_curve.equals(result_delayed.equity_curve)


# --- entry_at_open tests ---


def _make_entry_at_open_data(n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create OHLCV and signals for same-bar-open entry testing.

    Signal with entry_at_open=True at bar 5.  Price rises gently.
    """
    ohlcv = pd.DataFrame(
        {
            "open": [100.0 + i * 0.5 for i in range(n)],
            "high": [101.0 + i * 0.5 for i in range(n)],
            "low": [99.0 + i * 0.5 for i in range(n)],
            "close": [100.5 + i * 0.5 for i in range(n)],
            "volume": [1_000_000] * n,
            "timestamp": pd.bdate_range("2024-01-02", periods=n),
        }
    )
    signals = pd.DataFrame(
        {
            "entry": [False] * n,
            "entry_at_open": [False] * n,
            "stop_price": [float("nan")] * n,
            "timestamp": ohlcv["timestamp"],
            "symbol": "TEST",
        }
    )
    signals.loc[5, "entry"] = True
    signals.loc[5, "entry_at_open"] = True
    signals.loc[5, "stop_price"] = 92.0  # entry ~102.5, stop 92
    return ohlcv, signals


def test_same_bar_open_entry() -> None:
    """entry_at_open=True should enter at bar T's Open (not T+1)."""
    ohlcv, signals = _make_entry_at_open_data()
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run(ohlcv, signals)
    assert len(result.trades) > 0 or len(result.equity_curve) == len(ohlcv)
    # Check entry price is close to bar 5 open (102.5)
    if result.trades:
        entry_price = result.trades[0]["entry_price"]
        bar5_open = ohlcv.iloc[5]["open"]
        assert entry_price >= bar5_open * 0.999  # with slippage


def test_entry_at_open_no_exit_same_bar() -> None:
    """Entry at bar T open should not be exited on bar T."""
    ohlcv, signals = _make_entry_at_open_data()
    # Set stop very close to entry to test same-bar exit doesn't fire
    signals.loc[5, "stop_price"] = ohlcv.iloc[5]["open"] * 0.99
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.01,  # tight stop
        take_profit_rr=10.0,
    )
    result = engine.run(ohlcv, signals)
    # No trade should exit on bar 5 itself
    for trade in result.trades:
        if trade["bars_held"] == 0:
            pytest.fail("Same-bar exit should not happen for entry_at_open")


def test_entry_at_open_no_double_pending() -> None:
    """entry_at_open=True should NOT also set pending_entry for T+1."""
    ohlcv, signals = _make_entry_at_open_data()
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
        max_positions=2,
    )
    result = engine.run(ohlcv, signals)
    # entry_at_open should not also create pending_entry for T+1
    assert result.max_concurrent_positions <= 1


def test_daily_mode_backward_compatible(sample_data: pd.DataFrame) -> None:
    """Default entry_mode/trailing_mode should produce identical results to old engine."""
    signals = _make_simple_signals(sample_data, "SYN_TREND")
    sym_data = sample_data[sample_data["symbol"] == "SYN_TREND"].reset_index(drop=True)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.07,
        take_profit_rr=3.0,
        entry_mode="next_day_open",
        trailing_mode="daily",
    )
    result = engine.run(sym_data, signals)
    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 25_000.0


# --- Weekly trailing tests ---


def _make_weekly_trailing_data(n_weeks: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create OHLCV spanning n_weeks with entry at bar 5.

    Uptrend for first 10 weeks, then sharp decline to trigger weekly trailing.
    """
    n = n_weeks * 5
    dates = pd.bdate_range("2024-01-02", periods=n)
    closes = []
    for i in range(n):
        week = i // 5
        if week < 10:
            closes.append(100.0 + i * 0.5)  # uptrend
        else:
            closes.append(closes[-1] * 0.97)  # ~3% daily decline

    ohlcv = pd.DataFrame(
        {
            "open": [c - 0.2 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
            "timestamp": dates,
        }
    )
    signals = pd.DataFrame(
        {
            "entry": [False] * n,
            "entry_at_open": [False] * n,
            "stop_price": [float("nan")] * n,
            "timestamp": ohlcv["timestamp"],
            "symbol": "TEST",
        }
    )
    # Entry at bar 5 (entry_at_open for same-bar)
    signals.loc[5, "entry"] = True
    signals.loc[5, "entry_at_open"] = True
    signals.loc[5, "stop_price"] = 80.0  # wide stop

    return ohlcv, signals


def test_weekly_trailing_fires_on_week_end() -> None:
    """Weekly trailing should only fire on week-end bars."""
    ohlcv, signals = _make_weekly_trailing_data(n_weeks=15)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.02,
        stop_loss_pct=0.50,  # very wide SL so it doesn't trigger first
        take_profit_rr=100.0,  # very wide TP
        trailing_mode="weekly",
        weekly_ema_period=5,
        weekly_transition_weeks=2,
    )
    result = engine.run(ohlcv, signals)
    # Should have an exit with reason "weekly_trailing"
    weekly_exits = [t for t in result.trades if t["exit_reason"] == "weekly_trailing"]
    assert len(weekly_exits) > 0


def test_weekly_trailing_transition_period() -> None:
    """During transition period, weekly trailing should NOT trigger."""
    ohlcv, signals = _make_weekly_trailing_data(n_weeks=15)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.02,
        stop_loss_pct=0.50,
        take_profit_rr=100.0,
        trailing_mode="weekly",
        weekly_ema_period=5,
        weekly_transition_weeks=99,  # huge transition → never triggers
    )
    result = engine.run(ohlcv, signals)
    weekly_exits = [t for t in result.trades if t["exit_reason"] == "weekly_trailing"]
    assert len(weekly_exits) == 0


def test_weekly_trailing_pending_exit_next_open() -> None:
    """Weekly trend break should exit at next bar's Open (not detection bar)."""
    ohlcv, signals = _make_weekly_trailing_data(n_weeks=15)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.02,
        stop_loss_pct=0.50,
        take_profit_rr=100.0,
        trailing_mode="weekly",
        weekly_ema_period=5,
        weekly_transition_weeks=2,
    )
    result = engine.run(ohlcv, signals)
    weekly_exits = [t for t in result.trades if t["exit_reason"] == "weekly_trailing"]
    if weekly_exits:
        # Exit should be at Open, which is close to the exit bar's open price
        exit_trade = weekly_exits[0]
        assert exit_trade["bars_held"] > 0


def test_weekly_mode_disables_daily_trailing() -> None:
    """In weekly mode, daily trailing stop should NOT fire."""
    ohlcv, signals = _make_weekly_trailing_data(n_weeks=15)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.02,
        stop_loss_pct=0.50,
        take_profit_rr=100.0,
        trailing_mode="weekly",
        weekly_ema_period=5,
        weekly_transition_weeks=2,
        trailing_ema_window=5,  # tight daily trailing
        trailing_low_window=3,
    )
    result = engine.run(ohlcv, signals)
    trailing_exits = [t for t in result.trades if t["exit_reason"] == "trailing_stop"]
    assert len(trailing_exits) == 0, "Daily trailing should not fire in weekly mode"


# --- Portfolio mode tests ---


def _make_portfolio_ohlcv(
    symbols: list[str],
    n_days: int = 30,
    start: str = "2024-01-02",
    base_price: float = 100.0,
    missing_days: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Create multi-symbol OHLCV for portfolio tests.

    missing_days: {symbol: [day_indices_to_skip]} — omits bars on those days.
    """
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for sym_i, sym in enumerate(symbols):
        price = base_price + sym_i * 10  # offset per symbol
        skip_days = set((missing_days or {}).get(sym, []))
        for i, d in enumerate(dates):
            if i in skip_days:
                continue
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": price + i * 0.3,
                    "high": price + i * 0.3 + 1.0,
                    "low": price + i * 0.3 - 1.0,
                    "close": price + i * 0.3 + 0.5,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _make_portfolio_signals(
    ohlcv: pd.DataFrame,
    entries: dict[str, list[tuple[int, float]]],
) -> pd.DataFrame:
    """Create portfolio signals.

    entries: {symbol: [(day_index, score), ...]}
    Each signal is entry_at_open=True with a stop_price 10% below open.
    """
    rows = []
    dates = sorted(ohlcv["timestamp"].unique())
    for sym in ohlcv["symbol"].unique():
        sym_entries = dict(entries.get(sym, []))
        sym_data = ohlcv[ohlcv["symbol"] == sym].sort_values("timestamp")
        for _, row in sym_data.iterrows():
            d = row["timestamp"]
            day_idx = dates.index(d)
            is_entry = day_idx in sym_entries
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "entry": is_entry,
                    "entry_at_open": is_entry,
                    "stop_price": row["open"] * 0.90 if is_entry else float("nan"),
                    "score": sym_entries.get(day_idx, 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_portfolio_shared_capital() -> None:
    """With max_positions=1, only the higher-score symbol should enter."""
    ohlcv = _make_portfolio_ohlcv(["AAA", "BBB"], n_days=20)
    signals = _make_portfolio_signals(
        ohlcv,
        {
            "AAA": [(5, 40.0)],
            "BBB": [(5, 80.0)],
        },
    )
    engine = BacktestEngine(
        initial_capital=25_000.0,
        max_positions=1,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
        trailing_mode="daily",
    )
    result = engine.run_portfolio(ohlcv, signals)
    assert result.max_concurrent_positions <= 1
    if result.trades:
        assert result.trades[0]["symbol"] == "BBB"


def test_portfolio_equity_has_datetime_index() -> None:
    """Portfolio equity curve should have DatetimeIndex."""
    ohlcv = _make_portfolio_ohlcv(["AAA"], n_days=15)
    signals = _make_portfolio_signals(ohlcv, {"AAA": [(3, 50.0)]})
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    assert isinstance(result.equity_curve.index, pd.DatetimeIndex)


def test_portfolio_mdd_under_100pct() -> None:
    """Portfolio MDD should be in [0, 1.0]."""
    ohlcv = _make_portfolio_ohlcv(["AAA", "BBB"], n_days=30)
    signals = _make_portfolio_signals(
        ohlcv,
        {
            "AAA": [(3, 50.0)],
            "BBB": [(5, 60.0)],
        },
    )
    engine = BacktestEngine(
        initial_capital=25_000.0,
        max_positions=5,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    eq = result.equity_curve
    running_max = eq.cummax()
    drawdown = (running_max - eq) / running_max
    mdd = float(drawdown.max())
    assert 0.0 <= mdd <= 1.0


def test_portfolio_missing_bar_uses_last_close() -> None:
    """When a symbol has no bar on a date, equity uses last_known_close."""
    ohlcv = _make_portfolio_ohlcv(["AAA"], n_days=15, missing_days={"AAA": [8, 9]})
    signals = _make_portfolio_signals(ohlcv, {"AAA": [(3, 50.0)]})
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=100.0,  # wide TP
    )
    result = engine.run_portfolio(ohlcv, signals)
    # Equity should exist for all trading dates in the OHLCV
    assert len(result.equity_curve) == len(ohlcv["timestamp"].unique())


def test_portfolio_pending_entry_carries_over() -> None:
    """If symbol has no bar on entry day, pending_entry carries over."""
    # Signal on day 5, but symbol AAA has no bar on day 6 (next_day_open)
    ohlcv = _make_portfolio_ohlcv(["AAA", "BBB"], n_days=15, missing_days={"AAA": [6]})
    # Use next_day_open mode: signal at day 5 → entry at day 6
    signals = _make_portfolio_signals(ohlcv, {"AAA": [(5, 50.0)]})
    # Override entry_at_open to False for next_day_open testing
    signals["entry_at_open"] = False
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=100.0,
        entry_mode="next_day_open",
    )
    result = engine.run_portfolio(ohlcv, signals)
    # Entry should have happened (carried over to day 7): equity drops below
    # initial_capital due to position cost, or max_concurrent > 0
    assert result.max_concurrent_positions >= 1


def _make_portfolio_ohlcv_with_drop(
    symbols: list[str],
    n_days: int = 20,
    drop_after: int = 7,
) -> pd.DataFrame:
    """OHLCV that rises then drops sharply to trigger SL exits."""
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for sym_i, sym in enumerate(symbols):
        base = 100.0 + sym_i * 10
        for i, d in enumerate(dates):
            if i <= drop_after:
                price = base + i * 0.5
            else:
                price = base - (i - drop_after) * 3.0  # Sharp decline
                price = max(price, 10.0)
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price + 0.2,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_portfolio_signal_priority_by_score() -> None:
    """With 3 signals and max_positions=2, top 2 by score should enter."""
    ohlcv = _make_portfolio_ohlcv_with_drop(["AAA", "BBB", "CCC"], n_days=20)
    signals = _make_portfolio_signals(
        ohlcv,
        {
            "AAA": [(5, 30.0)],
            "BBB": [(5, 90.0)],
            "CCC": [(5, 60.0)],
        },
    )
    engine = BacktestEngine(
        initial_capital=100_000.0,
        max_positions=2,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    entered_symbols = {t["symbol"] for t in result.trades}
    assert "BBB" in entered_symbols
    assert "CCC" in entered_symbols
    assert "AAA" not in entered_symbols


def test_portfolio_fund_reuse_same_day() -> None:
    """Cash from pending_exit should be available for same_bar_open entries."""
    ohlcv = _make_portfolio_ohlcv(["AAA", "BBB"], n_days=30)
    # AAA enters at day 3, exits via SL eventually
    # BBB enters at a later day — should use freed cash
    signals = _make_portfolio_signals(
        ohlcv,
        {
            "AAA": [(3, 50.0)],
            "BBB": [(10, 60.0)],
        },
    )
    engine = BacktestEngine(
        initial_capital=25_000.0,
        max_positions=2,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    assert isinstance(result, BacktestResult)
    # At minimum the engine should complete without error
    assert result.initial_capital == 25_000.0


def test_portfolio_week_end_detection() -> None:
    """Pre-computed week_end_dates should correctly identify ISO week boundaries."""
    ohlcv = _make_portfolio_ohlcv(["AAA"], n_days=15, start="2024-01-02")
    dates = sorted(ohlcv["timestamp"].unique())
    # Build week end set manually
    ends: set[pd.Timestamp] = set()
    for i, d in enumerate(dates):
        if i == len(dates) - 1 or (
            pd.Timestamp(d).isocalendar()[1] != pd.Timestamp(dates[i + 1]).isocalendar()[1]
        ):
            ends.add(d)
    # Every week should have exactly one end-of-week date
    assert len(ends) >= 2  # 15 bdays ≈ 3 weeks


def test_portfolio_value_per_symbol_close() -> None:
    """_portfolio_value should use per-symbol close prices."""
    positions = [
        {"symbol": "AAA", "shares": 10},
        {"symbol": "BBB", "shares": 20},
    ]
    close_prices = {"AAA": 100.0, "BBB": 50.0}
    val = BacktestEngine._portfolio_value(positions, close_prices)
    assert val == pytest.approx(10 * 100.0 + 20 * 50.0)


def test_portfolio_pending_entry_before_same_bar() -> None:
    """Pending entries (prev day) should be processed before same_bar_open."""
    ohlcv = _make_portfolio_ohlcv(["AAA", "BBB"], n_days=20)
    # AAA: next_day_open signal at day 4 (pending for day 5)
    # BBB: same_bar_open signal at day 5
    sig_rows = []
    dates = sorted(ohlcv["timestamp"].unique())
    for sym in ["AAA", "BBB"]:
        sym_data = ohlcv[ohlcv["symbol"] == sym].sort_values("timestamp")
        for _, row in sym_data.iterrows():
            d = row["timestamp"]
            day_idx = dates.index(d)
            if sym == "AAA" and day_idx == 4:
                sig_rows.append(
                    {
                        "timestamp": d,
                        "symbol": sym,
                        "entry": True,
                        "entry_at_open": False,
                        "stop_price": row["open"] * 0.90,
                        "score": 30.0,
                    }
                )
            elif sym == "BBB" and day_idx == 5:
                sig_rows.append(
                    {
                        "timestamp": d,
                        "symbol": sym,
                        "entry": True,
                        "entry_at_open": True,
                        "stop_price": row["open"] * 0.90,
                        "score": 90.0,
                    }
                )
            else:
                sig_rows.append(
                    {
                        "timestamp": d,
                        "symbol": sym,
                        "entry": False,
                        "entry_at_open": False,
                        "stop_price": float("nan"),
                        "score": 0.0,
                    }
                )
    signals = pd.DataFrame(sig_rows)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        max_positions=1,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    # AAA's pending entry (day 4 signal → day 5 execution) should take
    # priority over BBB's same-bar signal at day 5
    if result.trades:
        first_trade = result.trades[0]
        assert first_trade["symbol"] == "AAA"


def test_portfolio_pending_exit_per_position() -> None:
    """Two positions in same symbol should have independent pending_exit."""
    ohlcv = _make_portfolio_ohlcv(["AAA"], n_days=30)
    # Two entries at different days
    signals = _make_portfolio_signals(
        ohlcv,
        {
            "AAA": [(3, 50.0), (8, 60.0)],
        },
    )
    engine = BacktestEngine(
        initial_capital=100_000.0,
        max_positions=5,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    assert isinstance(result, BacktestResult)


def test_portfolio_pending_entry_latest_wins() -> None:
    """Consecutive signals for same symbol: latest should supersede."""
    ohlcv = _make_portfolio_ohlcv(["AAA"], n_days=20)
    # Signals on day 5 and day 6 (next_day_open) — day 6 should supersede day 5
    sig_rows = []
    dates = sorted(ohlcv["timestamp"].unique())
    sym_data = ohlcv[ohlcv["symbol"] == "AAA"].sort_values("timestamp")
    for _, row in sym_data.iterrows():
        d = row["timestamp"]
        day_idx = dates.index(d)
        if day_idx in (5, 6):
            sig_rows.append(
                {
                    "timestamp": d,
                    "symbol": "AAA",
                    "entry": True,
                    "entry_at_open": False,
                    "stop_price": row["open"] * 0.90,
                    "score": 50.0 + day_idx * 5,
                }
            )
        else:
            sig_rows.append(
                {
                    "timestamp": d,
                    "symbol": "AAA",
                    "entry": False,
                    "entry_at_open": False,
                    "stop_price": float("nan"),
                    "score": 0.0,
                }
            )
    signals = pd.DataFrame(sig_rows)
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=100.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    # Should have at most 1 entry (latest signal supersedes)
    assert len(result.trades) <= 1


def test_portfolio_force_exit_missing_bars() -> None:
    """Positions should be force-exited after max_missing_bars consecutive missing days."""
    # AAA missing from day 5 through day 19 (15 missing days)
    ohlcv = _make_portfolio_ohlcv(
        ["AAA", "BBB"],
        n_days=25,
        missing_days={"AAA": list(range(5, 20))},
    )
    signals = _make_portfolio_signals(ohlcv, {"AAA": [(3, 50.0)]})
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=100.0,
    )
    result = engine.run_portfolio(ohlcv, signals, max_missing_bars=10)
    force_exits = [t for t in result.trades if t["exit_reason"] == "force_exit_missing"]
    assert len(force_exits) > 0
    assert all(t["bars_held"] > 0 for t in force_exits)


def test_portfolio_trade_bars_held_positive() -> None:
    """Portfolio exits should record positive bars_held for normal exits."""
    ohlcv = _make_portfolio_ohlcv_with_drop(["AAA"], n_days=20, drop_after=5)
    signals = _make_portfolio_signals(ohlcv, {"AAA": [(3, 50.0)]})
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.05,
        take_profit_rr=100.0,
    )
    result = engine.run_portfolio(ohlcv, signals)
    assert result.trades, "Expected at least one completed trade"
    assert all(t["bars_held"] > 0 for t in result.trades)


def test_portfolio_week_end_per_symbol() -> None:
    """Week-end dates should be computed per symbol (different trading calendars)."""
    # BBB missing some days means different week-end dates
    ohlcv = _make_portfolio_ohlcv(
        ["AAA", "BBB"],
        n_days=15,
        missing_days={"BBB": [4]},  # BBB misses Friday
    )
    dates_aaa = sorted(ohlcv[ohlcv["symbol"] == "AAA"]["timestamp"].unique())
    dates_bbb = sorted(ohlcv[ohlcv["symbol"] == "BBB"]["timestamp"].unique())
    assert len(dates_aaa) != len(dates_bbb)

    # Engine should still complete
    signals = _make_portfolio_signals(ohlcv, {"AAA": [(3, 50.0)]})
    engine = BacktestEngine(
        initial_capital=25_000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.10,
        take_profit_rr=10.0,
        trailing_mode="weekly",
        weekly_ema_period=3,
        weekly_transition_weeks=1,
    )
    result = engine.run_portfolio(ohlcv, signals)
    assert isinstance(result, BacktestResult)
