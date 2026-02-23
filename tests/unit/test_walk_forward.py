"""Tests for Walk-Forward (anchored expanding) OOS validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pipeline.evaluation.walk_forward import (
    MIN_WARMUP_BARS,
    WalkForwardResult,
    _filter_signals_to_period,
    _slice_ohlcv,
    create_anchored_splits,
    run_walk_forward,
)
from pipeline.runner.pipeline import Pipeline
from pipeline.spec.schema import ValidationConfig

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies" / "examples"


# --- create_anchored_splits tests ---


def test_create_anchored_splits_count() -> None:
    """create_anchored_splits should return exactly n_splits tuples."""
    splits = create_anchored_splits("2016-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30)
    assert len(splits) == 5
    for s in splits:
        assert len(s) == 4  # (is_start, is_end, oos_start, oos_end)


def test_splits_contiguous() -> None:
    """OOS windows should be contiguous: oos_end[k]+1day == oos_start[k+1]."""
    splits = create_anchored_splits("2016-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30)
    for i in range(len(splits) - 1):
        oos_end_i = pd.Timestamp(splits[i][3])
        oos_start_next = pd.Timestamp(splits[i + 1][2])
        gap = (oos_start_next - oos_end_i).days
        assert gap == 1, (
            f"Split {i} OOS end ({oos_end_i.date()}) to "
            f"Split {i + 1} OOS start ({oos_start_next.date()}) "
            f"should be 1 day apart, got {gap}"
        )


def test_splits_non_overlapping() -> None:
    """IS end should be the day before OOS start (no overlap)."""
    splits = create_anchored_splits("2016-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30)
    for i, (_, is_end, oos_start, _) in enumerate(splits):
        is_end_ts = pd.Timestamp(is_end)
        oos_start_ts = pd.Timestamp(oos_start)
        gap = (oos_start_ts - is_end_ts).days
        assert gap == 1, f"Split {i}: IS end to OOS start should be 1 day, got {gap}"


def test_splits_cover_oos_region() -> None:
    """OOS region should start at ~(1-oos_ratio) and end at end_date."""
    splits = create_anchored_splits("2016-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30)
    first_oos_start = pd.Timestamp(splits[0][2])
    last_oos_end = pd.Timestamp(splits[-1][3])

    start = pd.Timestamp("2016-01-01")
    end = pd.Timestamp("2024-01-01")
    expected_oos_start = start + pd.Timedelta(days=(end - start).days * 0.70)
    assert abs((first_oos_start - expected_oos_start).days) <= 1

    # Last OOS end should be exactly end_date (inclusive)
    assert last_oos_end == end


def test_splits_is_anchored() -> None:
    """All splits should have the same IS start date."""
    splits = create_anchored_splits("2016-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30)
    starts = {s[0] for s in splits}
    assert len(starts) == 1
    assert starts.pop() == "2016-01-01"


# --- _slice_ohlcv tests ---


def test_slice_ohlcv_inclusive() -> None:
    """_slice_ohlcv should include both start and end dates."""
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "symbol": "TEST",
            "close": [100, 101, 102, 103, 104],
        }
    )
    sliced = _slice_ohlcv(ohlcv, "2024-01-02", "2024-01-04")
    assert len(sliced) == 3
    assert sliced["close"].tolist() == [101, 102, 103]


# --- run_walk_forward integration tests ---


def test_walk_forward_with_fixtures() -> None:
    """Walk-Forward should complete with fixture data."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    assert isinstance(result, WalkForwardResult)
    assert len(result.splits) == 2
    assert isinstance(result.degradation_ratio_pf, float)
    # all_oos_trades should be a list (may be empty with fixture data)
    assert isinstance(result.all_oos_trades, list)


def test_degradation_ratio_calculation() -> None:
    """Degradation ratios should be finite float values."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    import numpy as np

    assert np.isfinite(result.degradation_ratio_pf)
    assert np.isfinite(result.degradation_ratio_sharpe)
    assert np.isfinite(result.oos_aggregate_pf)
    assert np.isfinite(result.oos_aggregate_sharpe)


# --- ValidationConfig constraint tests ---


def test_validation_config_n_splits_rejects_zero() -> None:
    """n_splits=0 should be rejected."""
    with pytest.raises(ValidationError):
        ValidationConfig(n_splits=0)


def test_validation_config_n_splits_rejects_one() -> None:
    """n_splits=1 should be rejected (ge=2)."""
    with pytest.raises(ValidationError):
        ValidationConfig(n_splits=1)


def test_validation_config_oos_ratio_rejects_zero() -> None:
    """oos_ratio=0 should be rejected (gt=0)."""
    with pytest.raises(ValidationError):
        ValidationConfig(oos_ratio=0.0)


def test_validation_config_oos_ratio_rejects_one() -> None:
    """oos_ratio=1.0 should be rejected (lt=1)."""
    with pytest.raises(ValidationError):
        ValidationConfig(oos_ratio=1.0)


def test_validation_config_valid_walk_forward() -> None:
    """Valid walk-forward config should parse."""
    config = ValidationConfig(method="walk_forward", n_splits=5, oos_ratio=0.30)
    assert config.n_splits == 5
    assert config.oos_ratio == 0.30


# --- Trading-date-based splits tests ---


def test_create_anchored_splits_with_trading_dates() -> None:
    """When trading_dates provided, splits should snap to actual trading days."""
    # Generate business days (Mon-Fri), skipping weekends
    td = pd.bdate_range("2020-01-01", "2024-01-01")
    splits = create_anchored_splits(
        "2020-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30, trading_dates=td
    )
    assert len(splits) == 5

    # All boundaries must be actual trading days
    td_set = set(td.strftime("%Y-%m-%d"))
    for is_start, is_end, oos_start, oos_end in splits:
        assert is_start in td_set, f"IS start {is_start} not a trading day"
        assert is_end in td_set, f"IS end {is_end} not a trading day"
        assert oos_start in td_set, f"OOS start {oos_start} not a trading day"
        assert oos_end in td_set, f"OOS end {oos_end} not a trading day"


def test_splits_trading_dates_no_gap() -> None:
    """IS end should be exactly one trading day before OOS start."""
    td = pd.bdate_range("2020-01-01", "2024-01-01")
    splits = create_anchored_splits(
        "2020-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30, trading_dates=td
    )
    td_list = list(td.strftime("%Y-%m-%d"))
    for i, (_, is_end, oos_start, _) in enumerate(splits):
        is_end_pos = td_list.index(is_end)
        oos_start_pos = td_list.index(oos_start)
        assert oos_start_pos == is_end_pos + 1, (
            f"Split {i}: IS end ({is_end}) and OOS start ({oos_start}) "
            f"should be adjacent trading days"
        )


def test_splits_trading_dates_contiguous() -> None:
    """OOS windows should be contiguous in trading-day mode."""
    td = pd.bdate_range("2020-01-01", "2024-01-01")
    splits = create_anchored_splits(
        "2020-01-01", "2024-01-01", n_splits=5, oos_ratio=0.30, trading_dates=td
    )
    td_list = list(td.strftime("%Y-%m-%d"))
    for i in range(len(splits) - 1):
        oos_end_i = splits[i][3]
        oos_start_next = splits[i + 1][2]
        end_pos = td_list.index(oos_end_i)
        start_pos = td_list.index(oos_start_next)
        assert start_pos == end_pos + 1, (
            f"Split {i} OOS end ({oos_end_i}) to Split {i + 1} OOS start ({oos_start_next}) "
            f"should be adjacent trading days"
        )


def test_splits_trading_dates_is_anchored() -> None:
    """All splits should share the same IS start (trading-day mode)."""
    td = pd.bdate_range("2020-01-01", "2024-01-01")
    splits = create_anchored_splits(
        "2020-01-01", "2024-01-01", n_splits=3, oos_ratio=0.30, trading_dates=td
    )
    starts = {s[0] for s in splits}
    assert len(starts) == 1
    # First trading day in the range
    assert starts.pop() == td[0].strftime("%Y-%m-%d")


# --- OOS equity slice fix test ---


def test_oos_equity_slice_multi_symbol() -> None:
    """OOS equity slice should use per-symbol bar count, not multi-symbol row count."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    # Verify multi-symbol data
    n_symbols = ohlcv["symbol"].nunique()
    assert n_symbols > 1, "Fixture must have multiple symbols for this test"

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)

    # With the fix, OOS metrics should be computed (not all zeros from empty slice)
    assert isinstance(result, WalkForwardResult)
    for split in result.splits:
        # oos_metrics should have been computed (even if sharpe=0 due to no trades)
        assert "sharpe" in split.oos_metrics
        assert "profit_factor" in split.oos_metrics
    # Aggregate values should be finite
    assert np.isfinite(result.oos_aggregate_pf)
    assert np.isfinite(result.oos_aggregate_sharpe)


# --- Warmup warning test ---


def test_warmup_insufficient_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Warning should be logged when IS period has fewer bars than MIN_WARMUP_BARS."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None
    # Build a deterministic short-history dataset (single symbol) for this test.
    one_symbol = str(ohlcv["symbol"].iloc[0])
    short_ohlcv = ohlcv[ohlcv["symbol"] == one_symbol].head(180).reset_index(drop=True)

    # With oos_ratio=0.30 and n_splits=2, the first IS period covers ~70% of bars.
    # 180 bars * 0.70 = 126 < MIN_WARMUP_BARS(200), so warning should appear.
    per_symbol = short_ohlcv.groupby("symbol").size()
    max_symbol_bars = int(per_symbol.max())
    expected_is_bars = int(max_symbol_bars * 0.70)
    assert expected_is_bars < MIN_WARMUP_BARS, (
        f"Test needs first IS split shorter than {MIN_WARMUP_BARS}, "
        f"got ~{expected_is_bars} (max symbol bars {max_symbol_bars} * 0.70)"
    )

    with caplog.at_level(logging.WARNING, logger="pipeline.evaluation.walk_forward"):
        run_walk_forward(pipeline, short_ohlcv, n_splits=2, oos_ratio=0.30)

    warmup_msgs = [r for r in caplog.records if "indicator warmup" in r.message]
    assert len(warmup_msgs) > 0, "Expected warmup warning but got none"


def test_warmup_insufficient_raises_when_fail_fast() -> None:
    """fail_on_short_warmup=True should raise when IS warmup is insufficient."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None
    one_symbol = str(ohlcv["symbol"].iloc[0])
    short_ohlcv = ohlcv[ohlcv["symbol"] == one_symbol].head(180).reset_index(drop=True)

    with pytest.raises(ValueError, match="indicator warmup"):
        run_walk_forward(
            pipeline,
            short_ohlcv,
            n_splits=2,
            oos_ratio=0.30,
            fail_on_short_warmup=True,
        )


def test_warmup_fail_fast_message_suggests_oos_ratio() -> None:
    """When feasible, fail-fast message should include a concrete oos_ratio suggestion."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    # Use the symbol with the longest history; warmup is feasible if oos_ratio is reduced.
    longest_symbol = str(ohlcv.groupby("symbol").size().idxmax())
    long_ohlcv = ohlcv[ohlcv["symbol"] == longest_symbol].reset_index(drop=True)

    with pytest.raises(ValueError, match=r"Try oos_ratio <= .*current 0.80"):
        run_walk_forward(
            pipeline,
            long_ohlcv,
            n_splits=2,
            oos_ratio=0.80,
            fail_on_short_warmup=True,
        )


# --- Finding 1: OOS carry-over fix tests ---


def test_oos_no_carryover_trades() -> None:
    """When OOS trade_count == 0, OOS Sharpe must also be 0 (no IS carry-over)."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    for split in result.splits:
        if split.oos_trade_count == 0:
            # No trades → sharpe should be 0 (not contaminated by IS carry-over)
            assert split.oos_metrics.get("sharpe", 0.0) == 0.0, (
                f"Split {split.test_start}~{split.test_end}: "
                f"oos_trade_count=0 but oos_sharpe={split.oos_metrics.get('sharpe')}"
            )


def test_filter_signals_to_period() -> None:
    """_filter_signals_to_period should keep only signals in [start, end]."""
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15", "2024-01-20"]
            ),
            "symbol": "TEST",
            "entry": [True, False, True, False, True],
        }
    )
    filtered = _filter_signals_to_period(signals, "2024-01-05", "2024-01-15")
    assert len(filtered) == 3
    assert filtered["timestamp"].iloc[0] == pd.Timestamp("2024-01-05")
    assert filtered["timestamp"].iloc[-1] == pd.Timestamp("2024-01-15")
    # Index should be reset
    assert list(filtered.index) == [0, 1, 2]


# --- Finding 2: Integer boundary split tests ---


def test_short_data_splits_no_inversion() -> None:
    """With short data (60 days), oos_end >= oos_start for all splits."""
    splits = create_anchored_splits("2024-01-01", "2024-03-01", n_splits=5, oos_ratio=0.30)
    assert len(splits) == 5
    for i, (_is_start, is_end, oos_start, oos_end) in enumerate(splits):
        assert pd.Timestamp(oos_end) >= pd.Timestamp(oos_start), (
            f"Split {i}: oos_end ({oos_end}) < oos_start ({oos_start}) — date inversion!"
        )
        assert pd.Timestamp(is_end) < pd.Timestamp(oos_start), (
            f"Split {i}: is_end ({is_end}) >= oos_start ({oos_start}) — overlap!"
        )


def test_splits_too_short_raises() -> None:
    """When OOS period is too short for n_splits, ValueError should be raised."""
    # 10 days total, oos_ratio=0.30 → oos_total_days=3, n_splits=5 → 3 < 5
    with pytest.raises(ValueError, match="too short"):
        create_anchored_splits("2024-01-01", "2024-01-11", n_splits=5, oos_ratio=0.30)


# --- all_oos_trades field tests ---


def test_all_oos_trades_returned() -> None:
    """WalkForwardResult should include all_oos_trades list."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    assert isinstance(result.all_oos_trades, list)
    # Trade count should match sum of per-split counts
    expected_count = sum(s.oos_trade_count for s in result.splits)
    assert len(result.all_oos_trades) == expected_count
    # Each trade should be a dict with pnl key
    for trade in result.all_oos_trades:
        assert isinstance(trade, dict)
        assert "pnl" in trade


# --- oos_trades per-split field tests ---


def test_split_has_oos_trades() -> None:
    """Each WalkForwardSplit should have oos_trades list matching oos_trade_count."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    for split in result.splits:
        assert isinstance(split.oos_trades, list)
        assert len(split.oos_trades) == split.oos_trade_count


def test_split_trades_have_split_index() -> None:
    """Each trade in split.oos_trades should have a split_index (0-based)."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    for i, split in enumerate(result.splits):
        for trade in split.oos_trades:
            assert "split_index" in trade, "Trade missing split_index"
            assert trade["split_index"] == i, (
                f"Expected split_index={i}, got {trade['split_index']}"
            )


def test_split_trades_independent_copies() -> None:
    """Trades in split.oos_trades must be independent of all_oos_trades (no ref sharing)."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None

    result = run_walk_forward(pipeline, ohlcv, n_splits=2, oos_ratio=0.30)
    # If there are trades, mutating a split trade should not affect all_oos_trades
    for split in result.splits:
        if split.oos_trades:
            split.oos_trades[0]["_test_marker"] = True
            break

    for trade in result.all_oos_trades:
        assert "_test_marker" not in trade, "Reference sharing between split and all_oos_trades"
