"""Tests for scored gap-up open signal detection and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.signals.gap_open_scored import (
    _score_gap_size,
    _score_ma_position,
    _score_pre_trend,
    _score_volume_trend,
    compute_element_scores,
    compute_weighted_score,
    detect_gap_open_scored,
    find_gap_open_candidates,
    score_to_grade,
    should_skip_entry,
)
from pipeline.spec.schema import EntryQualityFilter, GapOpenScoredConfig

# --- Scoring element thresholds ---


class TestScoreGapSize:
    def test_negative_gap(self) -> None:
        assert _score_gap_size(-0.05) == 1

    def test_small_gap(self) -> None:
        assert _score_gap_size(0.01) == 2  # 1% → [0-2%)

    def test_medium_gap(self) -> None:
        assert _score_gap_size(0.05) == 3  # 5% → [2-10%]

    def test_large_gap_risk(self) -> None:
        assert _score_gap_size(0.15) == 2  # 15% → >10% risk penalty

    def test_boundary_2pct(self) -> None:
        assert _score_gap_size(0.02) == 3  # exactly 2%

    def test_boundary_10pct(self) -> None:
        assert _score_gap_size(0.10) == 3  # exactly 10%


class TestScorePreTrend:
    def test_deep_negative(self) -> None:
        assert _score_pre_trend(-15.0) == 1

    def test_slight_negative(self) -> None:
        assert _score_pre_trend(-5.0) == 2

    def test_flat(self) -> None:
        assert _score_pre_trend(5.0) == 3

    def test_moderate_positive(self) -> None:
        assert _score_pre_trend(15.0) == 4

    def test_strong_positive(self) -> None:
        assert _score_pre_trend(25.0) == 5


class TestScoreVolumeTrend:
    def test_below_90(self) -> None:
        assert _score_volume_trend(80.0) == 2

    def test_90_to_100(self) -> None:
        assert _score_volume_trend(95.0) == 3

    def test_100_to_110(self) -> None:
        assert _score_volume_trend(105.0) == 4

    def test_110_to_120(self) -> None:
        assert _score_volume_trend(115.0) == 5

    def test_above_120_unusual(self) -> None:
        assert _score_volume_trend(130.0) == 1


class TestScoreMAPosition:
    def test_well_below(self) -> None:
        assert _score_ma_position(85.0) == 1

    def test_below(self) -> None:
        assert _score_ma_position(95.0) == 2

    def test_near(self) -> None:
        assert _score_ma_position(102.0) == 3

    def test_above(self) -> None:
        assert _score_ma_position(107.0) == 4

    def test_well_above(self) -> None:
        assert _score_ma_position(115.0) == 5


# --- Weighted score ---


class TestComputeWeightedScore:
    def test_max_score(self) -> None:
        """All elements at 5 → score = 100."""
        scores = {
            k: 5
            for k in [
                "gap_size",
                "pre_earnings_trend",
                "volume_trend",
                "ma200_position",
                "ma50_position",
            ]
        }
        assert compute_weighted_score(scores) == pytest.approx(100.0)

    def test_min_score(self) -> None:
        """All elements at 1 → score = 20."""
        scores = {
            k: 1
            for k in [
                "gap_size",
                "pre_earnings_trend",
                "volume_trend",
                "ma200_position",
                "ma50_position",
            ]
        }
        assert compute_weighted_score(scores) == pytest.approx(20.0)

    def test_custom_weights(self) -> None:
        """Custom weights should be applied correctly."""
        scores = {
            "gap_size": 5,
            "pre_earnings_trend": 1,
            "volume_trend": 1,
            "ma200_position": 1,
            "ma50_position": 1,
        }
        weights = {
            "gap_size": 100.0,
            "pre_earnings_trend": 0,
            "volume_trend": 0,
            "ma200_position": 0,
            "ma50_position": 0,
        }
        assert compute_weighted_score(scores, weights) == pytest.approx(100.0)


# --- Grade boundaries ---


class TestScoreToGrade:
    def test_grade_a(self) -> None:
        assert score_to_grade(85.0) == "A"
        assert score_to_grade(100.0) == "A"

    def test_grade_b(self) -> None:
        assert score_to_grade(84.9) == "B"
        assert score_to_grade(55.0) == "B"

    def test_grade_c(self) -> None:
        assert score_to_grade(54.9) == "C"
        assert score_to_grade(30.0) == "C"

    def test_grade_d(self) -> None:
        assert score_to_grade(29.9) == "D"
        assert score_to_grade(0.0) == "D"


# --- Entry quality filter ---


class TestShouldSkipEntry:
    def test_price_band_skip(self) -> None:
        """$25 stock → in [$10, $30) → skip."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 10.0,
            "exclude_price_band_high": 30.0,
            "risk_gap_threshold": 10.0,
            "risk_score_threshold": 85.0,
        }
        assert should_skip_entry(25.0, 0.05, 60.0, cfg) is True

    def test_price_band_pass(self) -> None:
        """$50 stock → outside [$10, $30) → pass."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 10.0,
            "exclude_price_band_high": 30.0,
            "risk_gap_threshold": 10.0,
            "risk_score_threshold": 85.0,
        }
        assert should_skip_entry(50.0, 0.05, 60.0, cfg) is False

    def test_risk_combo_skip(self) -> None:
        """gap=12% + score=90 → skip."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 10.0,
            "exclude_price_band_high": 30.0,
            "risk_gap_threshold": 10.0,
            "risk_score_threshold": 85.0,
        }
        assert should_skip_entry(50.0, 0.12, 90.0, cfg) is True

    def test_risk_combo_pass(self) -> None:
        """gap=12% + score=80 → pass (score below threshold)."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 10.0,
            "exclude_price_band_high": 30.0,
            "risk_gap_threshold": 10.0,
            "risk_score_threshold": 85.0,
        }
        assert should_skip_entry(50.0, 0.12, 80.0, cfg) is False

    def test_filter_disabled(self) -> None:
        """With filter disabled, nothing is skipped."""
        cfg = {"enabled": False}
        assert should_skip_entry(25.0, 0.12, 90.0, cfg) is False

    def test_month_perf_skip_when_negative(self) -> None:
        """month_perf < threshold should be skipped."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 0.0,
            "exclude_price_band_high": 0.01,
            "risk_gap_threshold": 100.0,
            "risk_score_threshold": 100.0,
            "month_perf_min": 0.0,
            "month_perf_lookback_bars": 21,
        }
        assert should_skip_entry(50.0, 0.05, 60.0, cfg, month_perf=-0.05) is True

    def test_month_perf_pass_when_positive(self) -> None:
        """month_perf >= threshold should pass."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 0.0,
            "exclude_price_band_high": 0.01,
            "risk_gap_threshold": 100.0,
            "risk_score_threshold": 100.0,
            "month_perf_min": 0.0,
            "month_perf_lookback_bars": 21,
        }
        assert should_skip_entry(50.0, 0.05, 60.0, cfg, month_perf=0.03) is False

    def test_month_perf_skip_when_missing(self) -> None:
        """month_perf filter enabled but unavailable should be skipped."""
        cfg = {
            "enabled": True,
            "exclude_price_band_low": 0.0,
            "exclude_price_band_high": 0.01,
            "risk_gap_threshold": 100.0,
            "risk_score_threshold": 100.0,
            "month_perf_min": 0.0,
            "month_perf_lookback_bars": 21,
        }
        assert should_skip_entry(50.0, 0.05, 60.0, cfg, month_perf=None) is True


# --- find_gap_open_candidates ---


def _make_ohlcv(n: int = 250, gap_idx: int = 220, gap_pct: float = 0.07) -> dict[str, pd.Series]:
    """Create synthetic OHLCV with a gap-up at gap_idx."""
    rng = np.random.default_rng(42)
    base_price = 100.0
    closes = np.full(n, base_price)
    # Gentle uptrend
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + rng.normal(0.001, 0.005))
    opens = closes * (1 + rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.01, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(500_000, 2_000_000, n).astype(float)

    # Insert gap-up at gap_idx
    if gap_idx < n:
        opens[gap_idx] = closes[gap_idx - 1] * (1 + gap_pct)
        highs[gap_idx] = opens[gap_idx] * 1.02
        closes[gap_idx] = opens[gap_idx] * 1.01
        lows[gap_idx] = opens[gap_idx] * 0.99

    return {
        "open": pd.Series(opens),
        "high": pd.Series(highs),
        "low": pd.Series(lows),
        "close": pd.Series(closes),
        "volume": pd.Series(volume),
    }


class TestFindGapOpenCandidates:
    def test_detects_gap(self) -> None:
        """Should detect a 7% gap-up with 4% threshold."""
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)
        candidates = find_gap_open_candidates(data["open"], data["close"], min_gap_pct=0.04)
        assert 220 in candidates

    def test_no_volume_used(self) -> None:
        """find_gap_open_candidates should not use volume at all.

        Even with zero volume on gap day, the gap should still be detected.
        """
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)
        data["volume"].iloc[220] = 0  # zero volume — should not matter
        candidates = find_gap_open_candidates(data["open"], data["close"], min_gap_pct=0.04)
        assert 220 in candidates

    def test_gap_below_threshold_not_detected(self) -> None:
        """A 2% gap should not pass 4% threshold."""
        data = _make_ohlcv(gap_idx=220, gap_pct=0.02)
        candidates = find_gap_open_candidates(data["open"], data["close"], min_gap_pct=0.04)
        assert 220 not in candidates


# --- No look-ahead bias ---


class TestNoLookAheadBias:
    def test_scores_use_only_pre_gap_data(self) -> None:
        """Element scores should only depend on data before gap bar.

        Changing data at and after gap_idx should not change scores.
        """
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)
        scores1 = compute_element_scores(data["open"], data["close"], data["volume"], 220)

        # Modify data after gap (bars 221+)
        data2 = {k: v.copy() for k, v in data.items()}
        for i in range(221, 250):
            data2["close"].iloc[i] = 999.0
            data2["volume"].iloc[i] = 99_999_999

        scores2 = compute_element_scores(data2["open"], data2["close"], data2["volume"], 220)
        assert scores1 == scores2


# --- Full detection ---


class TestDetectGapOpenScored:
    def test_detects_and_returns_entry(self) -> None:
        """Should detect gap and return entry with all required fields."""
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)
        config = GapOpenScoredConfig(
            min_gap_pct=0.04,
            stop_loss_pct=0.10,
            min_grade="D",
            entry_filter=EntryQualityFilter(enabled=False),
        )
        entries = detect_gap_open_scored(
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            config=config,
        )
        gap_entries = [e for e in entries if e["idx"] == 220]
        assert len(gap_entries) == 1
        entry = gap_entries[0]
        assert "stop_price" in entry
        assert "score" in entry
        assert "grade" in entry
        assert entry["stop_price"] < float(data["open"].iloc[220])

    def test_cooldown_prevents_rapid_signals(self) -> None:
        """Signals within cooldown should be suppressed."""
        # Create data with two gaps close together
        data = _make_ohlcv(gap_idx=100, gap_pct=0.07)
        # Insert a second gap at 105
        data["open"].iloc[105] = float(data["close"].iloc[104]) * 1.08
        data["high"].iloc[105] = float(data["open"].iloc[105]) * 1.02
        data["close"].iloc[105] = float(data["open"].iloc[105]) * 1.01

        config = GapOpenScoredConfig(
            min_gap_pct=0.04,
            min_grade="D",
            cooldown=20,
            entry_filter=EntryQualityFilter(enabled=False),
        )
        entries = detect_gap_open_scored(
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            config=config,
        )
        entry_indices = [e["idx"] for e in entries]
        # Both 100 and 105 are gaps, but 105 should be suppressed by cooldown
        if 100 in entry_indices:
            assert 105 not in entry_indices

    def test_grade_filter(self) -> None:
        """Entries below min_grade should be filtered out."""
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)
        config_strict = GapOpenScoredConfig(
            min_gap_pct=0.04,
            min_grade="A",
            entry_filter=EntryQualityFilter(enabled=False),
        )
        entries = detect_gap_open_scored(
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            config=config_strict,
        )
        # With min_grade="A", most entries will be filtered
        for e in entries:
            assert e["grade"] == "A"

    def test_stop_price_calculation(self) -> None:
        """stop_price should be entry * (1 - stop_loss_pct)."""
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)
        config = GapOpenScoredConfig(
            min_gap_pct=0.04,
            stop_loss_pct=0.10,
            min_grade="D",
            entry_filter=EntryQualityFilter(enabled=False),
        )
        entries = detect_gap_open_scored(
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            config=config,
        )
        gap_entries = [e for e in entries if e["idx"] == 220]
        if gap_entries:
            entry = gap_entries[0]
            expected_stop = float(data["open"].iloc[220]) * (1 - 0.10)
            assert entry["stop_price"] == pytest.approx(expected_stop, rel=1e-6)

    def test_month_filter_is_applied_at_entry_time(self) -> None:
        """Entry-time month momentum filter should control acceptance."""
        data = _make_ohlcv(gap_idx=220, gap_pct=0.07)

        # Force negative month momentum at entry point:
        # month_perf = close[219] / close[198] - 1
        data["close"].iloc[198] = 130.0
        data["close"].iloc[219] = 100.0
        data["open"].iloc[220] = float(data["close"].iloc[219]) * 1.07
        data["high"].iloc[220] = float(data["open"].iloc[220]) * 1.02
        data["close"].iloc[220] = float(data["open"].iloc[220]) * 1.01
        data["low"].iloc[220] = float(data["open"].iloc[220]) * 0.99

        cfg_off = GapOpenScoredConfig(
            min_gap_pct=0.04,
            min_grade="D",
            entry_filter=EntryQualityFilter(
                enabled=True,
                exclude_price_band_low=0.0,
                exclude_price_band_high=0.01,
                risk_gap_threshold=100.0,
                risk_score_threshold=100.0,
                month_perf_min=None,
            ),
        )
        entries_off = detect_gap_open_scored(
            data["open"], data["high"], data["low"], data["close"], data["volume"], config=cfg_off
        )

        cfg_on = GapOpenScoredConfig(
            min_gap_pct=0.04,
            min_grade="D",
            entry_filter=EntryQualityFilter(
                enabled=True,
                exclude_price_band_low=0.0,
                exclude_price_band_high=0.01,
                risk_gap_threshold=100.0,
                risk_score_threshold=100.0,
                month_perf_min=0.0,
                month_perf_lookback_bars=21,
            ),
        )
        entries_on = detect_gap_open_scored(
            data["open"], data["high"], data["low"], data["close"], data["volume"], config=cfg_on
        )

        assert any(e["idx"] == 220 for e in entries_off)
        assert not any(e["idx"] == 220 for e in entries_on)
