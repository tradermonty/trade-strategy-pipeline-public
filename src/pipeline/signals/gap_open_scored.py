"""Scored gap-up open strategy — 5-element scoring, same-bar-open entry.

Detects gap-ups using Open price only (no look-ahead bias from intraday volume),
scores candidates on 5 elements computed from bar T-1 data, and filters by grade
and entry quality rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.spec.schema import GapOpenScoredConfig

# --- Grade thresholds ---
_GRADE_THRESHOLDS: list[tuple[str, float]] = [
    ("A", 85.0),
    ("B", 55.0),
    ("C", 30.0),
]
_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def score_to_grade(score: float) -> str:
    """Convert weighted score (0-100) to letter grade."""
    for grade, threshold in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "D"


def _score_gap_size(gap_pct: float) -> int:
    """Score gap size element (1-5 scale, but capped at 3 per spec).

    neg → 1, 0-2% → 2, 2-10% → 3, >10% → 2 (risk penalty).
    gap_pct is a fraction (e.g., 0.05 = 5%).
    """
    pct = gap_pct * 100.0
    if pct < 0:
        return 1
    if pct < 2:
        return 2
    if pct <= 10:
        return 3
    return 2  # >10% risk penalty


def _score_pre_trend(trend_pct: float) -> int:
    """Score pre-gap 20-day price trend.

    <-10% → 1, -10~0% → 2, 0~10% → 3, 10~20% → 4, >20% → 5.
    """
    if trend_pct < -10:
        return 1
    if trend_pct < 0:
        return 2
    if trend_pct < 10:
        return 3
    if trend_pct < 20:
        return 4
    return 5


def _score_volume_trend(ratio: float) -> int:
    """Score volume trend: avg_vol_20 / avg_vol_60 ratio (percentage).

    <90% → 2, 90-100% → 3, 100-110% → 4, 110-120% → 5, >120% → 1 (unusual).
    """
    if ratio < 90:
        return 2
    if ratio < 100:
        return 3
    if ratio < 110:
        return 4
    if ratio < 120:
        return 5
    return 1  # >120% unusual activity


def _score_ma_position(pct_above: float) -> int:
    """Score price position relative to MA (200 or 50).

    <90% → 1, 90-100% → 2, 100-105% → 3, 105-110% → 4, >110% → 5.
    pct_above = (close / MA) * 100.
    """
    if pct_above < 90:
        return 1
    if pct_above < 100:
        return 2
    if pct_above < 105:
        return 3
    if pct_above < 110:
        return 4
    return 5


def compute_element_scores(
    open_: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    gap_idx: int,
    avg_volume_window: int = 50,
) -> dict[str, int]:
    """Compute 5-element raw scores (1-5) for a gap candidate.

    All computations use data up to bar T-1 (no look-ahead bias).
    The gap_pct is computed from open[T] vs close[T-1] — this is known at open.

    Args:
        avg_volume_window: Long-period window for volume trend ratio denominator.
    """
    prev_close = float(close.iloc[gap_idx - 1])
    gap_pct = (float(open_.iloc[gap_idx]) - prev_close) / prev_close if prev_close > 0 else 0.0

    # Legacy scoring key name is "pre_earnings_trend", but the signal itself
    # is based on generic gap-ups (not earnings-calendar constrained).
    # This trend score is close[T-1] vs close[T-21].
    t1 = gap_idx - 1
    t21 = max(0, gap_idx - 21)
    close_t1 = float(close.iloc[t1])
    close_t21 = float(close.iloc[t21])
    pre_trend_pct = ((close_t1 - close_t21) / close_t21 * 100) if close_t21 > 0 else 0.0

    # Volume trend: avg_vol_20[T-1] / avg_vol_long[T-1]
    vol_start_20 = max(0, gap_idx - 20)
    vol_start_long = max(0, gap_idx - avg_volume_window)
    avg_vol_20 = float(volume.iloc[vol_start_20:gap_idx].mean()) if gap_idx > 0 else 0.0
    avg_vol_long = float(volume.iloc[vol_start_long:gap_idx].mean()) if gap_idx > 0 else 1.0
    vol_ratio = (avg_vol_20 / avg_vol_long * 100) if avg_vol_long > 0 else 100.0

    # MA200 position: close[T-1] / SMA200[T-1]
    ma200_start = max(0, gap_idx - 200)
    ma200 = float(close.iloc[ma200_start:gap_idx].mean()) if gap_idx > 0 else close_t1
    ma200_pct = (close_t1 / ma200 * 100) if ma200 > 0 else 100.0

    # MA50 position: close[T-1] / SMA50[T-1]
    ma50_start = max(0, gap_idx - 50)
    ma50 = float(close.iloc[ma50_start:gap_idx].mean()) if gap_idx > 0 else close_t1
    ma50_pct = (close_t1 / ma50 * 100) if ma50 > 0 else 100.0

    return {
        "gap_size": _score_gap_size(gap_pct),
        "pre_earnings_trend": _score_pre_trend(pre_trend_pct),
        "volume_trend": _score_volume_trend(vol_ratio),
        "ma200_position": _score_ma_position(ma200_pct),
        "ma50_position": _score_ma_position(ma50_pct),
    }


def compute_weighted_score(
    element_scores: dict[str, int],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted score (0-100) from element scores (1-5).

    Each element contributes: (raw_score / 5) * weight.
    """
    if weights is None:
        weights = {
            "gap_size": 25.0,
            "pre_earnings_trend": 30.0,
            "volume_trend": 20.0,
            "ma200_position": 15.0,
            "ma50_position": 10.0,
        }
    total = 0.0
    for key, raw in element_scores.items():
        w = weights.get(key, 0.0)
        total += (raw / 5.0) * w
    return total


def should_skip_entry(
    price: float,
    gap_pct: float,
    score: float,
    filter_config: dict[str, Any],
    month_perf: float | None = None,
) -> bool:
    """Apply entry quality filter. Returns True if entry should be skipped.

    Filters:
    1. Price band exclusion: stocks in [$low, $high) → skip.
    2. Risk combo: gap% >= threshold AND score >= threshold → skip.
    """
    if not filter_config.get("enabled", True):
        return False

    low = filter_config.get("exclude_price_band_low", 10.0)
    high = filter_config.get("exclude_price_band_high", 30.0)
    if low <= price < high:
        return True

    # gap_pct is fraction, threshold is in percent
    gap_pct_val = gap_pct * 100.0
    risk_gap = filter_config.get("risk_gap_threshold", 10.0)
    risk_score = filter_config.get("risk_score_threshold", 85.0)
    if gap_pct_val >= risk_gap and score >= risk_score:
        return True

    # Optional trade-time month momentum filter.
    month_perf_min = filter_config.get("month_perf_min")
    if month_perf_min is not None:
        if month_perf is None or pd.isna(month_perf):
            return True
        if float(month_perf) < float(month_perf_min):
            return True
    return False


def find_gap_open_candidates(
    open_: pd.Series,
    close: pd.Series,
    min_gap_pct: float = 0.04,
) -> list[int]:
    """Find gap-up candidates using Open price only (no intraday volume).

    A gap at bar i requires: open[i] > close[i-1] * (1 + min_gap_pct).
    Does NOT use volume[i] — volume is unknown at market open.
    """
    gap_indices: list[int] = []
    close_arr = close.to_numpy(dtype=np.float64)
    open_arr = open_.to_numpy(dtype=np.float64)

    for i in range(1, len(open_arr)):
        prev_close = close_arr[i - 1]
        if prev_close <= 0:
            continue
        if open_arr[i] > prev_close * (1 + min_gap_pct):
            gap_indices.append(i)

    return gap_indices


def detect_gap_open_scored(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    config: GapOpenScoredConfig,
) -> list[dict[str, Any]]:
    """Main detection function: find gap-up opens, score, filter, return signals.

    Returns list of dicts with keys: idx, stop_price, score, grade, gap_pct.
    """
    gap_indices = find_gap_open_candidates(open_, close, min_gap_pct=config.min_gap_pct)

    if not gap_indices:
        return []

    weights = config.scoring.model_dump()
    filter_cfg = config.entry_filter.model_dump()
    min_grade_rank = _GRADE_ORDER.get(config.min_grade, 3)

    entries: list[dict[str, Any]] = []
    last_entry = -config.cooldown  # allow first signal

    for gap_idx in gap_indices:
        if gap_idx - last_entry < config.cooldown:
            continue

        # Compute scores
        elem_scores = compute_element_scores(
            open_,
            close,
            volume,
            gap_idx,
            avg_volume_window=config.avg_volume_window,
        )
        score = compute_weighted_score(elem_scores, weights)
        grade = score_to_grade(score)

        # Grade filter
        grade_rank = _GRADE_ORDER.get(grade, 3)
        if grade_rank > min_grade_rank:
            continue

        # Entry quality filter
        entry_price = float(open_.iloc[gap_idx])
        prev_close = float(close.iloc[gap_idx - 1])
        gap_pct = (entry_price - prev_close) / prev_close if prev_close > 0 else 0.0

        month_perf: float | None = None
        month_perf_min = filter_cfg.get("month_perf_min")
        if month_perf_min is not None:
            lookback = int(filter_cfg.get("month_perf_lookback_bars", 21))
            end_idx = gap_idx - 1
            start_idx = end_idx - lookback
            if start_idx >= 0:
                start_close = float(close.iloc[start_idx])
                end_close = float(close.iloc[end_idx])
                if start_close > 0:
                    month_perf = (end_close / start_close) - 1.0

        if should_skip_entry(entry_price, gap_pct, score, filter_cfg, month_perf=month_perf):
            continue

        stop_price = entry_price * (1 - config.stop_loss_pct)

        entries.append(
            {
                "idx": gap_idx,
                "stop_price": stop_price,
                "score": score,
                "grade": grade,
                "gap_pct": gap_pct,
            }
        )
        last_entry = gap_idx

    return entries


def detect_earnings_gap_open(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    config: GapOpenScoredConfig,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for the legacy function name."""
    return detect_gap_open_scored(open_, high, low, close, volume, config)
