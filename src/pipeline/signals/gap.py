"""Earnings gap-up continuation pattern detection.

Detects large gap-ups with volume surge during Stage 2 uptrend,
then waits for a breakout above the gap day's high as entry signal.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.signals.trend import trend_filter
from pipeline.signals.volume import is_volume_surge


def find_gap_ups(
    open_: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    min_gap_pct: float = 0.04,
    volume_ratio: float = 2.0,
    avg_volume_window: int = 50,
) -> list[int]:
    """Scan for gap-up bars meeting minimum gap size and volume surge.

    A gap-up on bar i requires:
      open[i] > close[i-1] * (1 + min_gap_pct)
      volume[i] >= volume_ratio * avg_volume(window)

    Returns list of bar indices where gap-ups occur.
    """
    vol_surge = is_volume_surge(volume, ratio=volume_ratio, window=avg_volume_window)
    gap_indices: list[int] = []

    for i in range(1, len(open_)):
        prev_close = close.iloc[i - 1]
        if prev_close <= 0:
            continue
        gap_pct = (open_.iloc[i] - prev_close) / prev_close
        if gap_pct >= min_gap_pct and vol_surge.iloc[i]:
            gap_indices.append(i)

    return gap_indices


def detect_gap_up(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    min_gap_pct: float = 0.04,
    volume_ratio: float = 2.0,
    avg_volume_window: int = 50,
    max_entry_days: int = 10,
) -> dict[str, Any]:
    """Detect a single gap-up continuation pattern.

    Steps:
      1. Find the first qualifying gap-up bar
      2. Verify trend filter at gap bar (Stage 2 uptrend)
      3. Wait up to max_entry_days for close above gap day's high

    Returns dict with:
        detected: bool
        gap_idx: int or None
        entry_idx: int or None
        gap_day_high: float or None
        gap_day_low: float or None
        reason: str (if not detected)
    """
    gap_indices = find_gap_ups(
        open_,
        close,
        volume,
        min_gap_pct=min_gap_pct,
        volume_ratio=volume_ratio,
        avg_volume_window=avg_volume_window,
    )

    if not gap_indices:
        return {
            "detected": False,
            "gap_idx": None,
            "entry_idx": None,
            "gap_day_high": None,
            "gap_day_low": None,
            "reason": "No qualifying gap-up found",
        }

    # Compute trend filter once for the full series
    tf = trend_filter(close)

    for gap_idx in gap_indices:
        # Check trend at gap bar
        if gap_idx >= len(tf) or not tf.iloc[gap_idx]:
            continue

        gap_day_high = float(high.iloc[gap_idx])
        gap_day_low = float(low.iloc[gap_idx])

        # Wait for breakout above gap day high
        search_end = min(gap_idx + max_entry_days + 1, len(close))
        for j in range(gap_idx + 1, search_end):
            if close.iloc[j] > gap_day_high:
                return {
                    "detected": True,
                    "gap_idx": gap_idx,
                    "entry_idx": j,
                    "gap_day_high": gap_day_high,
                    "gap_day_low": gap_day_low,
                    "reason": "",
                }

    # If we found gaps but none passed all filters
    return {
        "detected": False,
        "gap_idx": None,
        "entry_idx": None,
        "gap_day_high": None,
        "gap_day_low": None,
        "reason": "Gap-ups found but no trend confirmation or breakout within window",
    }


def detect_gap_up_rolling(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    min_gap_pct: float = 0.04,
    volume_ratio: float = 2.0,
    avg_volume_window: int = 50,
    max_entry_days: int = 10,
    cooldown: int = 20,
    breakout_volume_ratio: float = 0.0,
    max_stop_pct: float = 0.0,
    breakout_close_pos_min: float | None = None,
    breakout_max_extension_pct: float | None = None,
    breakout_min_body_atr: float | None = None,
    breakout_atr_pct_max: float | None = None,
) -> list[tuple[int, float]]:
    """Detect all gap-up continuation entries across the full series.

    Args:
        breakout_volume_ratio: Minimum volume ratio on breakout bar (0 = disabled).
        max_stop_pct: Maximum allowed stop distance as fraction of entry price (0 = disabled).
        breakout_close_pos_min: Min (close-low)/(high-low) on breakout bar (None = disabled).
        breakout_max_extension_pct: Max (close - gap_day_high)/gap_day_high (None = disabled).
        breakout_min_body_atr: Min abs(close-open)/ATR14 on breakout bar (None = disabled).
        breakout_atr_pct_max: Max ATR14/close on breakout bar (None = disabled).

    Returns list of (entry_bar_index, gap_day_low) tuples,
    deduplicated with cooldown gap.
    """
    gap_indices = find_gap_ups(
        open_,
        close,
        volume,
        min_gap_pct=min_gap_pct,
        volume_ratio=volume_ratio,
        avg_volume_window=avg_volume_window,
    )

    if not gap_indices:
        return []

    tf = trend_filter(close)
    vol_surge = (
        is_volume_surge(volume, ratio=breakout_volume_ratio, window=avg_volume_window)
        if breakout_volume_ratio > 0
        else None
    )

    # Pre-compute ATR14 if any ATR-based filter is active
    need_atr = breakout_min_body_atr is not None or breakout_atr_pct_max is not None
    atr: pd.Series | None = None
    if need_atr and len(close) > 14:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

    entries: list[tuple[int, float]] = []
    last_entry = -cooldown  # allow first signal

    for gap_idx in gap_indices:
        # Check trend at gap bar
        if gap_idx >= len(tf) or not tf.iloc[gap_idx]:
            continue

        gap_day_high = float(high.iloc[gap_idx])
        gap_day_low = float(low.iloc[gap_idx])

        # Wait for breakout above gap day high
        search_end = min(gap_idx + max_entry_days + 1, len(close))
        for j in range(gap_idx + 1, search_end):
            if close.iloc[j] > gap_day_high:
                # Breakout volume check — try later bars if volume insufficient
                if vol_surge is not None and not vol_surge.iloc[j]:
                    continue

                entry_proxy = float(close.iloc[j])

                # Max stop distance check (use close[j] as entry proxy)
                # Later bars will have equal or wider stop distance, so reject gap
                if max_stop_pct > 0 and entry_proxy > 0:
                    stop_dist = (entry_proxy - gap_day_low) / entry_proxy
                    if stop_dist > max_stop_pct:
                        break

                # --- Entry quality filters (continue = try next bar) ---

                # Close position: (close-low)/(high-low) — higher = bullish close
                if breakout_close_pos_min is not None:
                    h_j = float(high.iloc[j])
                    l_j = float(low.iloc[j])
                    rng = h_j - l_j
                    close_pos = (entry_proxy - l_j) / rng if rng > 0 else 0.0
                    if close_pos < breakout_close_pos_min:
                        continue

                # Extension: (close - gap_day_high) / gap_day_high
                if breakout_max_extension_pct is not None and gap_day_high > 0:
                    extension = (entry_proxy - gap_day_high) / gap_day_high
                    if extension > breakout_max_extension_pct:
                        continue

                # Body/ATR ratio: abs(close-open)/ATR14
                if breakout_min_body_atr is not None and atr is not None:
                    atr_val = float(atr.iloc[j]) if j < len(atr) else 0.0
                    if atr_val > 0:
                        body = abs(entry_proxy - float(open_.iloc[j]))
                        if body / atr_val < breakout_min_body_atr:
                            continue

                # ATR% filter: ATR14/close — reject high-volatility entries
                if breakout_atr_pct_max is not None and atr is not None:
                    atr_val = float(atr.iloc[j]) if j < len(atr) else 0.0
                    if entry_proxy > 0 and atr_val / entry_proxy > breakout_atr_pct_max:
                        continue

                if j - last_entry >= cooldown:
                    entries.append((j, gap_day_low))
                    last_entry = j
                break  # move to next gap

    return entries
