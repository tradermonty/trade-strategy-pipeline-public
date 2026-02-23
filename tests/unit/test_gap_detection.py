"""Tests for gap-up continuation pattern detection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from pipeline.signals.gap import detect_gap_up, detect_gap_up_rolling, find_gap_ups


def _as_float(value: object) -> float:
    """Convert pandas scalar-like value to float for arithmetic in tests."""
    return float(cast(Any, value))


def _make_uptrend_series(n: int = 300, start: float = 50.0) -> pd.DataFrame:
    """Create synthetic data with a steady Stage 2 uptrend.

    Returns DataFrame with open/high/low/close/volume columns.
    The uptrend grows ~0.15% per day so SMA50 > SMA200 is established.
    """
    rng = np.random.default_rng(42)
    close_values: list[float] = [start]
    for _ in range(1, n):
        close_values.append(close_values[-1] * (1 + 0.0015 + rng.normal(0, 0.005)))
    closes = np.array(close_values, dtype=float)

    # Normal volume around 500k
    volume = rng.integers(400_000, 600_000, size=n).astype(float)

    df = pd.DataFrame(
        {
            "open": closes * (1 + rng.normal(0, 0.002, n)),
            "high": closes * (1 + abs(rng.normal(0.005, 0.003, n))),
            "low": closes * (1 - abs(rng.normal(0.005, 0.003, n))),
            "close": closes,
            "volume": volume,
        }
    )
    return df


def _inject_gap_up(
    df: pd.DataFrame,
    bar: int,
    gap_pct: float = 0.06,
    vol_mult: float = 3.0,
) -> pd.DataFrame:
    """Inject a gap-up at the given bar index.

    Sets open[bar] = close[bar-1] * (1 + gap_pct) and scales volume.
    The gap day closes at its open (flat candle) so high == open.
    """
    df = df.copy()
    prev_close = _as_float(df.loc[bar - 1, "close"])
    gap_open = prev_close * (1 + gap_pct)
    df.loc[bar, "open"] = gap_open
    df.loc[bar, "high"] = gap_open * 1.005
    df.loc[bar, "low"] = gap_open * 0.995
    df.loc[bar, "close"] = gap_open  # flat close at open
    df.loc[bar, "volume"] = _as_float(df["volume"].iloc[max(0, bar - 50) : bar].mean()) * vol_mult
    return df


def _inject_breakout(
    df: pd.DataFrame,
    gap_bar: int,
    breakout_bar: int,
) -> pd.DataFrame:
    """Make the breakout bar close above the gap day high."""
    df = df.copy()
    gap_high = _as_float(df.loc[gap_bar, "high"])
    df.loc[breakout_bar, "close"] = gap_high * 1.01
    df.loc[breakout_bar, "high"] = gap_high * 1.015
    return df


class TestFindGapUps:
    def test_detects_qualifying_gap(self) -> None:
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        gaps = find_gap_ups(
            df["open"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
        )
        assert 220 in gaps

    def test_rejects_small_gap(self) -> None:
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.02, vol_mult=3.0)
        gaps = find_gap_ups(
            df["open"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
        )
        assert 220 not in gaps

    def test_rejects_low_volume(self) -> None:
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=1.0)
        gaps = find_gap_ups(
            df["open"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
        )
        assert 220 not in gaps


class TestDetectGapUp:
    def test_full_detection_with_breakout(self) -> None:
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)

        result = detect_gap_up(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
        )
        assert result["detected"] is True
        assert result["gap_idx"] == 220
        assert result["entry_idx"] == 225
        assert result["gap_day_high"] is not None
        assert result["gap_day_low"] is not None

    def test_no_breakout_within_window(self) -> None:
        """Gap found but no close above gap high within max_entry_days."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        # Suppress prices after gap to stay below gap day high
        gap_high = _as_float(df.loc[220, "high"])
        for i in range(221, 235):
            df.loc[i, "close"] = gap_high * 0.95
            df.loc[i, "high"] = gap_high * 0.97

        result = detect_gap_up(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
        )
        assert result["detected"] is False
        assert "breakout" in result["reason"].lower() or "no" in result["reason"].lower()

    def test_no_trend_rejects(self) -> None:
        """Gap in early bars where SMA200 is not established."""
        df = _make_uptrend_series(300, start=50.0)
        # Inject gap at bar 50 where SMA200 is NaN
        df = _inject_gap_up(df, bar=50, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=50, breakout_bar=55)

        result = detect_gap_up(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
        )
        # SMA200 not established at bar 50, so trend filter fails
        assert result["detected"] is False

    def test_no_gap_returns_not_detected(self) -> None:
        """Clean uptrend with no gaps."""
        df = _make_uptrend_series(300)
        result = detect_gap_up(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
        )
        assert result["detected"] is False
        assert "no qualifying" in result["reason"].lower()


class TestDetectGapUpRolling:
    def test_finds_multiple_entries(self) -> None:
        df = _make_uptrend_series(400)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        df = _inject_gap_up(df, bar=280, gap_pct=0.05, vol_mult=2.5)
        df = _inject_breakout(df, gap_bar=280, breakout_bar=285)

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            cooldown=20,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 in entry_indices
        assert 285 in entry_indices
        # Each entry carries gap_day_low
        for _idx, gap_low in entries:
            assert gap_low > 0

    def test_cooldown_deduplication(self) -> None:
        df = _make_uptrend_series(400)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Second gap too close (within cooldown)
        df = _inject_gap_up(df, bar=230, gap_pct=0.05, vol_mult=2.5)
        df = _inject_breakout(df, gap_bar=230, breakout_bar=235)

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            cooldown=20,
        )
        entry_indices = [idx for idx, _ in entries]
        # First entry accepted, second suppressed by cooldown
        assert 225 in entry_indices
        assert 235 not in entry_indices

    def test_rejects_low_volume_breakout(self) -> None:
        """Breakout bar with insufficient volume should be rejected."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Set breakout bar volume to very low (below 1.5x average)
        avg_vol = df["volume"].iloc[170:220].mean()
        df.loc[225, "volume"] = avg_vol * 0.8

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_volume_ratio=1.5,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 not in entry_indices

    def test_accepts_high_volume_breakout(self) -> None:
        """Breakout bar with sufficient volume should be accepted."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Set breakout bar volume to high (above 1.5x average)
        avg_vol = df["volume"].iloc[170:220].mean()
        df.loc[225, "volume"] = avg_vol * 2.0

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_volume_ratio=1.5,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 in entry_indices

    def test_rejects_far_stop_distance(self) -> None:
        """Entry with stop distance > max_stop_pct should be rejected."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Set gap day low very far from breakout close (>15% away)
        breakout_close = _as_float(df.loc[225, "close"])
        df.loc[220, "low"] = breakout_close * 0.80  # 20% stop distance

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            max_stop_pct=0.15,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 not in entry_indices

    def test_accepts_near_stop_distance(self) -> None:
        """Entry with stop distance <= max_stop_pct should be accepted."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Gap day low is close to breakout price (~1% stop)
        breakout_close = _as_float(df.loc[225, "close"])
        df.loc[220, "low"] = breakout_close * 0.95  # 5% stop distance

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            max_stop_pct=0.15,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 in entry_indices

    # --- Entry quality filter tests ---

    def test_rejects_weak_close_position(self) -> None:
        """Breakout bar with close near low should be rejected by close_pos filter."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Make breakout bar close near the low (weak candle)
        gap_high = _as_float(df.loc[220, "high"])
        df.loc[225, "close"] = gap_high * 1.001  # just above gap high
        df.loc[225, "high"] = gap_high * 1.03  # high much above close
        df.loc[225, "low"] = gap_high * 0.98  # low below close
        # close_pos = (1.001 - 0.98) / (1.03 - 0.98) = 0.021/0.05 = 0.42

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_close_pos_min=0.6,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 not in entry_indices

    def test_accepts_strong_close_position(self) -> None:
        """Breakout bar with close near high should pass close_pos filter."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Make breakout bar close near the high (strong candle)
        gap_high = _as_float(df.loc[220, "high"])
        df.loc[225, "close"] = gap_high * 1.025
        df.loc[225, "high"] = gap_high * 1.03
        df.loc[225, "low"] = gap_high * 0.98
        # close_pos = (1.025 - 0.98) / (1.03 - 0.98) = 0.045/0.05 = 0.90

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_close_pos_min=0.6,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 in entry_indices

    def test_rejects_overextended_entry(self) -> None:
        """Entry too far above gap_day_high should be rejected."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        gap_high = _as_float(df.loc[220, "high"])
        # Breakout bar closes 8% above gap_day_high
        df.loc[225, "close"] = gap_high * 1.08
        df.loc[225, "high"] = gap_high * 1.09

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_max_extension_pct=0.05,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 not in entry_indices

    def test_accepts_moderate_extension(self) -> None:
        """Entry within extension limit should be accepted."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        gap_high = _as_float(df.loc[220, "high"])
        # Breakout bar closes 2% above gap_day_high
        df.loc[225, "close"] = gap_high * 1.02
        df.loc[225, "high"] = gap_high * 1.03

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_max_extension_pct=0.05,
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 in entry_indices

    def test_rejects_high_atr_pct(self) -> None:
        """High-volatility entry (ATR/close too high) should be rejected."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)
        # Inflate high-low range around breakout to make ATR very high
        for i in range(210, 226):
            mid = _as_float(df.loc[i, "close"])
            df.loc[i, "high"] = mid * 1.10
            df.loc[i, "low"] = mid * 0.90

        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_atr_pct_max=0.03,  # 3% max ATR/close
        )
        entry_indices = [idx for idx, _ in entries]
        assert 225 not in entry_indices

    def test_filters_disabled_when_none(self) -> None:
        """All quality filters at None should not affect existing behavior."""
        df = _make_uptrend_series(300)
        df = _inject_gap_up(df, bar=220, gap_pct=0.06, vol_mult=3.0)
        df = _inject_breakout(df, gap_bar=220, breakout_bar=225)

        entries_without = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
        )
        entries_with_none = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            min_gap_pct=0.04,
            volume_ratio=2.0,
            avg_volume_window=50,
            max_entry_days=10,
            breakout_close_pos_min=None,
            breakout_max_extension_pct=None,
            breakout_min_body_atr=None,
            breakout_atr_pct_max=None,
        )
        assert entries_without == entries_with_none

    def test_empty_series(self) -> None:
        df = pd.DataFrame(
            {
                "open": pd.Series(dtype=float),
                "high": pd.Series(dtype=float),
                "low": pd.Series(dtype=float),
                "close": pd.Series(dtype=float),
                "volume": pd.Series(dtype=float),
            }
        )
        entries = detect_gap_up_rolling(
            df["open"],
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
        )
        assert entries == []
