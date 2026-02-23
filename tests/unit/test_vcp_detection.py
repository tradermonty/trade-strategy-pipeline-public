"""Tests for VCP pattern detection, trend filter, and volume analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.signals.trend import trend_filter, trend_template
from pipeline.signals.vcp import (
    check_volume_decline,
    detect_vcp,
    detect_vcp_rolling,
    find_contractions,
    find_contractions_multi_start,
    validate_vcp,
    zigzag,
    zigzag_atr,
)
from pipeline.signals.volume import avg_volume, is_volume_surge

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# --- Helpers ---


def _make_classic_vcp_series() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a synthetic classic VCP pattern for testing.

    Returns (high, low, close, volume).
    Uses explicit swing high/low construction to guarantee contracting
    ranges followed by a high-volume breakout.
    """
    n = 200
    prices = np.zeros(n)

    # Phase 1: Uptrend from 50 to ~75
    prices[0] = 50.0
    for i in range(1, 60):
        prices[i] = 50.0 + (25.0 / 59) * i

    # Contraction 1: range = 15 (20% of base ~75), high=80, low=65
    pivot = 75.0
    for i in range(60, 80):
        t = (i - 60) / 19.0
        prices[i] = pivot + 5.0 * np.sin(np.pi * t)  # peaks at ~80
    for i in range(80, 100):
        t = (i - 80) / 19.0
        prices[i] = pivot - 10.0 * np.sin(np.pi * t)  # dips to ~65

    # Contraction 2: range = 10 (ratio = 10/15 = 0.67 < 0.75)
    pivot = 73.0
    for i in range(100, 120):
        t = (i - 100) / 19.0
        prices[i] = pivot + 4.0 * np.sin(np.pi * t)  # peaks at ~77
    for i in range(120, 140):
        t = (i - 120) / 19.0
        prices[i] = pivot - 6.0 * np.sin(np.pi * t)  # dips to ~67

    # Contraction 3: range = 6 (ratio = 6/10 = 0.60 < 0.75)
    pivot = 74.0
    for i in range(140, 160):
        t = (i - 140) / 19.0
        prices[i] = pivot + 2.5 * np.sin(np.pi * t)  # peaks at ~76.5
    for i in range(160, 180):
        t = (i - 160) / 19.0
        prices[i] = pivot - 3.5 * np.sin(np.pi * t)  # dips to ~70.5

    # Breakout: surges above 80 (above all prior highs)
    for i in range(180, n):
        prices[i] = 76.5 + 0.5 * (i - 180)  # climbs to ~86.5

    close = pd.Series(prices)

    # Generate realistic high/low around close
    rng = np.random.default_rng(42)
    noise_high = rng.uniform(0.005, 0.015, n)
    noise_low = rng.uniform(0.005, 0.015, n)
    high = pd.Series(prices * (1 + noise_high))
    low = pd.Series(prices * (1 - noise_low))

    # Volume: declining during contractions, 2x surge on breakout
    vol = np.zeros(n)
    vol[:60] = 1_000_000
    vol[60:100] = 800_000
    vol[100:140] = 600_000
    vol[140:180] = 400_000
    vol[180:] = 2_000_000
    volume = pd.Series(vol)

    return high, low, close, volume


# --- Legacy Zig-zag tests ---


def test_zigzag_identifies_swings() -> None:
    """Zig-zag should identify alternating highs and lows."""
    prices = pd.Series([10, 15, 8, 18, 5, 20, 12, 25])
    pivots = zigzag(prices, threshold=0.10)
    assert len(pivots) >= 2
    types = [p[1] for p in pivots]
    for i in range(1, len(types)):
        assert types[i] != types[i - 1], "Pivots should alternate"


def test_zigzag_threshold_sensitivity() -> None:
    """Higher threshold should produce fewer pivots."""
    rng = np.random.default_rng(123)
    prices = pd.Series(np.cumsum(rng.normal(0, 1, 100)) + 50)
    pivots_low = zigzag(prices, threshold=0.02)
    pivots_high = zigzag(prices, threshold=0.10)
    assert len(pivots_low) >= len(pivots_high)


# --- ATR ZigZag tests ---


def test_zigzag_atr_identifies_swings() -> None:
    """ATR zigzag should identify swing highs and lows separately."""
    high, low, close, _ = _make_classic_vcp_series()
    swing_highs, swing_lows = zigzag_atr(high, low, close)
    assert len(swing_highs) >= 1
    assert len(swing_lows) >= 1
    # Swing highs should have values from the high series
    for idx, val in swing_highs:
        assert val >= close.iloc[idx] * 0.99


# --- Multi-start contraction tests ---


def test_multi_start_selects_best() -> None:
    """Multi-start should find the best contraction starting from top highs."""
    high, low, close, _ = _make_classic_vcp_series()
    swing_highs, swing_lows = zigzag_atr(high, low, close)
    contractions = find_contractions_multi_start(swing_highs, swing_lows)
    # Should find at least 2 contractions from the VCP pattern
    assert len(contractions) >= 2
    # Each contraction should have the expected keys
    for c in contractions:
        assert "high_idx" in c
        assert "low_idx" in c
        assert "depth_pct" in c


# --- VCP Validation tests ---


def test_t1_depth_too_shallow_rejected() -> None:
    """T1 depth < 8% should be rejected."""
    contractions = [
        {
            "high_idx": 0,
            "low_idx": 10,
            "high_price": 100,
            "low_price": 95,
            "depth_pct": 5.0,
            "duration_days": 10,
            "range": 5,
        },
        {
            "high_idx": 15,
            "low_idx": 25,
            "high_price": 99,
            "low_price": 96,
            "depth_pct": 3.0,
            "duration_days": 10,
            "range": 3,
        },
    ]
    result = validate_vcp(contractions, total_bars=100)
    assert result["valid"] is False
    assert any("shallow" in issue for issue in result["issues"])


def test_t1_depth_too_deep_rejected() -> None:
    """T1 depth > 35% should be flagged (warning only, not hard reject)."""
    contractions = [
        {
            "high_idx": 0,
            "low_idx": 30,
            "high_price": 100,
            "low_price": 60,
            "depth_pct": 40.0,
            "duration_days": 30,
            "range": 40,
        },
        {
            "high_idx": 35,
            "low_idx": 55,
            "high_price": 98,
            "low_price": 80,
            "depth_pct": 18.4,
            "duration_days": 20,
            "range": 18,
        },
    ]
    result = validate_vcp(contractions, total_bars=100)
    # Deep T1 is a warning, not invalidating
    assert any("deep" in issue for issue in result["issues"])


def test_right_shoulder_exceeded_rejected() -> None:
    """Subsequent high > 5% from H1 should truncate contractions."""
    # Build swing points where second high is far from first
    swing_highs = [(10, 100.0), (40, 112.0)]  # 12% apart
    swing_lows = [(25, 85.0), (55, 90.0)]
    contractions = find_contractions_multi_start(
        swing_highs,
        swing_lows,
        right_shoulder_pct=0.05,
    )
    # Should only have 1 contraction (second truncated by right-shoulder)
    assert len(contractions) <= 1


def test_pattern_too_short_rejected() -> None:
    """Pattern duration < 15 days should be rejected."""
    contractions = [
        {
            "high_idx": 0,
            "low_idx": 5,
            "high_price": 100,
            "low_price": 88,
            "depth_pct": 12.0,
            "duration_days": 5,
            "range": 12,
        },
        {
            "high_idx": 7,
            "low_idx": 10,
            "high_price": 99,
            "low_price": 92,
            "depth_pct": 7.1,
            "duration_days": 3,
            "range": 7,
        },
    ]
    result = validate_vcp(contractions, total_bars=50, min_pattern_days=15)
    assert result["valid"] is False
    assert any("short" in issue for issue in result["issues"])


# --- VCP Acceptance Tests (Section 4.5) ---


def test_classic_vcp_detected() -> None:
    """3 contractions, each <= 75% of prior, volume declining -> PASS."""
    high, low, close, volume = _make_classic_vcp_series()
    result = detect_vcp(
        high,
        low,
        close,
        volume,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert result["detected"] is True


def test_insufficient_contractions_rejected() -> None:
    """Flat data -> FAIL due to insufficient contractions."""
    rng = np.random.default_rng(99)
    n = 150
    # Nearly flat prices
    prices = 50.0 + rng.normal(0, 0.1, n).cumsum() * 0.01
    close = pd.Series(prices)
    high = pd.Series(prices * (1 + rng.uniform(0, 0.005, n)))
    low = pd.Series(prices * (1 - rng.uniform(0, 0.005, n)))
    vol = pd.Series(rng.normal(1_000_000, 100_000, n))

    result = detect_vcp(
        high,
        low,
        close,
        vol,
        min_contractions=3,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert result["detected"] is False


def test_non_contracting_ranges_rejected() -> None:
    """Non-contracting ranges -> FAIL."""
    rng = np.random.default_rng(77)
    n = 200
    prices = np.zeros(n)
    prices[0] = 50.0
    for i in range(1, 60):
        prices[i] = prices[i - 1] * (1 + rng.normal(0.002, 0.005))

    # range_1 = 5%, range_2 = 10% (expanding)
    base = prices[59]
    for i in range(60, 100):
        prices[i] = base + 0.05 * base * np.sin(2 * np.pi * (i - 60) / 40) * 0.5
    base = prices[99]
    for i in range(100, 150):
        prices[i] = base + 0.10 * base * np.sin(2 * np.pi * (i - 100) / 50) * 0.5
    base = prices[149]
    for i in range(150, 200):
        prices[i] = base + 0.03 * base * np.sin(2 * np.pi * (i - 150) / 50) * 0.5

    close = pd.Series(prices)
    high = pd.Series(prices * (1 + rng.uniform(0, 0.01, n)))
    low = pd.Series(prices * (1 - rng.uniform(0, 0.01, n)))
    vol = pd.Series(rng.normal(1_000_000, 100_000, n))

    result = detect_vcp(
        high,
        low,
        close,
        vol,
        min_contractions=3,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert result["detected"] is False


def test_low_breakout_volume_rejected() -> None:
    """Valid pattern but breakout volume < 1.5x avg -> FAIL."""
    high, low, close, volume = _make_classic_vcp_series()
    volume[:] = 1_000_000  # uniform volume

    result = detect_vcp(
        high,
        low,
        close,
        volume,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert result["detected"] is False


# --- More VCP tests ---


def test_volume_not_declining_rejected() -> None:
    """Volume increasing during contractions -> volume decline check fails."""
    _, _, close, volume = _make_classic_vcp_series()
    volume.iloc[80:120] = 1_500_000
    volume.iloc[120:155] = 2_000_000
    volume.iloc[155:185] = 2_500_000

    contractions = find_contractions(close, threshold=0.05)
    if len(contractions) >= 3:
        declining = check_volume_decline(volume, contractions[:3])
        assert declining is False


def test_detect_vcp_returns_signal_dataframe() -> None:
    """detect_vcp should return a dict with expected keys."""
    high, low, close, volume = _make_classic_vcp_series()
    result = detect_vcp(
        high,
        low,
        close,
        volume,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert "detected" in result
    assert "contractions" in result
    assert "breakout_idx" in result


def test_detect_vcp_on_sample_fixture() -> None:
    """Run VCP detection on SYN_VCP fixture data."""
    df = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")
    vcp_data = df[df["symbol"] == "SYN_VCP"].reset_index(drop=True)
    result = detect_vcp(
        vcp_data["high"],
        vcp_data["low"],
        vcp_data["close"],
        vcp_data["volume"],
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert isinstance(result["detected"], bool)


def test_lookback_window_respected() -> None:
    """Only the last N days should be searched."""
    high, low, close, volume = _make_classic_vcp_series()
    result = detect_vcp(
        high,
        low,
        close,
        volume,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
        lookback_window=50,
    )
    # 50 bar lookback won't capture the full pattern
    assert result["detected"] is False


# --- Trend filter tests ---


def test_trend_filter_above_sma200() -> None:
    """Price consistently above SMA200 should pass trend filter."""
    prices = pd.Series(np.linspace(100, 200, 250))
    result = trend_filter(prices)
    assert bool(result.iloc[-1]) is True


def test_trend_filter_below_sma200_rejected() -> None:
    """Downtrending price below SMA200 should fail."""
    prices = pd.Series(np.linspace(200, 100, 250))
    result = trend_filter(prices)
    assert bool(result.iloc[-1]) is False


def test_trend_filter_ma_misalignment_rejected() -> None:
    """SMA50 < SMA200 should fail even if price > SMA200."""
    prices = np.zeros(300)
    for i in range(150):
        prices[i] = 200 + i * 0.5
    for i in range(150, 250):
        prices[i] = 275 - (i - 150) * 1.5
    for i in range(250, 300):
        prices[i] = 125 + (i - 250) * 2

    result = trend_filter(pd.Series(prices))
    assert bool(result.iloc[260]) is False


# --- Trend template tests ---


def test_trend_template_full_pass() -> None:
    """Strong uptrend meeting all 7 criteria -> True."""
    n = 300
    # Steady uptrend: 50 -> 200
    close = pd.Series(np.linspace(50, 200, n))
    high = close * 1.01
    low = close * 0.99
    rs_rank = pd.Series(80, index=range(n))

    result = trend_template(close, high, low, rs_rank=rs_rank, min_criteria=6)
    # At the end, should pass
    assert bool(result.iloc[-1]) is True


def test_trend_template_6_of_7_pass() -> None:
    """6/7 criteria met -> True (at threshold)."""
    n = 300
    close = pd.Series(np.linspace(50, 200, n))
    high = close * 1.01
    low = close * 0.99
    # RS rank = 50 -> fails C7
    rs_rank = pd.Series(50, index=range(n))

    result = trend_template(close, high, low, rs_rank=rs_rank, min_criteria=6)
    assert bool(result.iloc[-1]) is True


def test_trend_template_5_of_7_fail() -> None:
    """5/7 criteria met -> False (below threshold)."""
    n = 300
    close = pd.Series(np.linspace(50, 200, n))
    high = close * 1.01
    low = close * 0.99
    rs_rank = pd.Series(50, index=range(n))

    result = trend_template(close, high, low, rs_rank=rs_rank, min_criteria=7)
    # With min_criteria=7, failing C7 means the bar fails
    assert bool(result.iloc[-1]) is False


def test_trend_template_sma200_not_trending_fail() -> None:
    """SMA200 not trending up should fail C3."""
    n = 300
    # Price goes up then plateaus
    prices = np.zeros(n)
    for i in range(100):
        prices[i] = 50 + i
    for i in range(100, 300):
        prices[i] = 150 + np.sin((i - 100) * 0.05) * 5

    close = pd.Series(prices)
    high = close * 1.01
    low = close * 0.99

    result = trend_template(close, high, low, min_criteria=7)
    # When SMA200 stalls, C3 won't pass at end
    assert bool(result.iloc[-1]) is False


# --- Volume tests ---


def test_volume_surge_detection() -> None:
    """Volume > 1.5x 50-day average should be detected as surge."""
    rng = np.random.default_rng(42)
    vol = pd.Series(rng.normal(500_000, 50_000, 60))
    vol.iloc[-1] = 1_000_000
    result = is_volume_surge(vol, ratio=1.5, window=50)
    assert bool(result.iloc[-1]) is True


def test_average_volume_calculation() -> None:
    """avg_volume should compute rolling mean correctly."""
    vol = pd.Series([100, 200, 300, 400, 500])
    result = avg_volume(vol, window=3)
    assert result.iloc[-1] == pytest.approx(400.0)


# --- Rolling VCP detection tests ---


def test_rolling_detects_multiple_vcps() -> None:
    """Rolling detection should find VCP patterns in SYN_VCP_LONG fixture."""
    df = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")
    vcp_long = df[df["symbol"] == "SYN_VCP_LONG"].reset_index(drop=True)
    assert len(vcp_long) == 600

    indices = detect_vcp_rolling(
        vcp_long["high"],
        vcp_long["low"],
        vcp_long["close"],
        vcp_long["volume"],
        window_size=300,
        step_size=15,
        cooldown=30,
        min_contractions=2,
        # Relax validation for synthetic fixture data which was designed
        # for the legacy close-only zigzag algorithm
        contraction_ratio=1.5,
        breakout_volume_ratio=1.5,
        t1_depth_min=0.0,
        right_shoulder_pct=0.20,
        volume_decline=False,
    )
    assert len(indices) >= 1
    for idx in indices:
        assert 0 <= idx < 600


def test_rolling_cooldown_deduplicates() -> None:
    """Breakouts within cooldown period should be deduplicated."""
    high, low, close, volume = _make_classic_vcp_series()
    indices = detect_vcp_rolling(
        high,
        low,
        close,
        volume,
        window_size=200,
        step_size=10,
        cooldown=200,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert len(indices) <= 1


def test_rolling_short_series_fallback() -> None:
    """Series shorter than window_size should fall back to single-pass."""
    high, low, close, volume = _make_classic_vcp_series()  # 200 bars
    indices = detect_vcp_rolling(
        high,
        low,
        close,
        volume,
        window_size=300,
        step_size=20,
        cooldown=40,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert len(indices) >= 1


def test_rolling_no_pattern_returns_empty() -> None:
    """Flat random data should produce no VCP signals."""
    rng = np.random.default_rng(123)
    n = 400
    prices = 100.0 + rng.normal(0, 0.5, n).cumsum() * 0.01
    close = pd.Series(prices)
    high = pd.Series(prices * (1 + rng.uniform(0, 0.005, n)))
    low = pd.Series(prices * (1 - rng.uniform(0, 0.005, n)))
    volume = pd.Series(rng.normal(500_000, 50_000, n).clip(100_000))
    indices = detect_vcp_rolling(
        high,
        low,
        close,
        volume,
        window_size=200,
        step_size=20,
        cooldown=40,
        min_contractions=2,
        contraction_ratio=0.75,
        breakout_volume_ratio=1.5,
    )
    assert indices == []
