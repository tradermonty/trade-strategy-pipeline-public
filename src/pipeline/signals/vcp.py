"""VCP (Volatility Contraction Pattern) detection.

Implements ATR-based zig-zag swing identification, multi-start contraction
detection, VCP validation (T1 depth, right-shoulder, pattern duration),
volume decline checking, and breakout confirmation.

Migrated from vcp-screener's Minervini-grade logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ATR calculation
# ---------------------------------------------------------------------------


def _calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float:
    """Calculate Average True Range over the last *period* bars.

    Returns 0.0 when there is insufficient data.
    """
    n = len(high)
    if n < period + 1:
        return 0.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_arr: np.ndarray[Any, np.dtype[np.floating[Any]]] = tr.dropna().to_numpy()
    if len(tr_arr) < period:
        return 0.0

    return float(np.mean(tr_arr[-period:]))


# ---------------------------------------------------------------------------
# ATR-based ZigZag (primary) — replaces the legacy close-only zigzag
# ---------------------------------------------------------------------------


def zigzag_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_multiplier: float = 1.5,
    atr_period: int = 14,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """ATR-based ZigZag swing detection.

    A reversal is recognised only when price moves ATR * multiplier from
    the current extreme.  This filters noise while adapting to volatility.

    Returns ``(swing_highs, swing_lows)`` as separate lists of
    ``(index, price)`` tuples.
    """
    highs = high.values.astype(float)
    lows = low.values.astype(float)
    n = len(highs)

    atr_val = _calculate_atr(high, low, close, atr_period)
    if atr_val <= 0 or n < atr_period + 1:
        return [], []

    threshold = atr_val * atr_multiplier

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    # Seed with the highest bar in the first *atr_period* bars
    extreme_idx = int(
        np.argmax(highs[: min(atr_period, n)])  # type: ignore[arg-type]
    )
    extreme_val = float(highs[extreme_idx])
    direction = 1  # 1 → looking for swing high, -1 → looking for swing low

    for i in range(extreme_idx + 1, n):
        if direction == 1:
            if highs[i] > extreme_val:
                extreme_val = float(highs[i])
                extreme_idx = i
            elif extreme_val - lows[i] >= threshold:
                swing_highs.append((extreme_idx, extreme_val))
                direction = -1
                extreme_val = float(lows[i])
                extreme_idx = i
        else:
            if lows[i] < extreme_val:
                extreme_val = float(lows[i])
                extreme_idx = i
            elif highs[i] - extreme_val >= threshold:
                swing_lows.append((extreme_idx, extreme_val))
                direction = 1
                extreme_val = float(highs[i])
                extreme_idx = i

    return swing_highs, swing_lows


# ---------------------------------------------------------------------------
# Legacy close-only zigzag (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _zigzag_legacy(prices: pd.Series, threshold: float = 0.05) -> list[tuple[int, int, float]]:
    """Identify swing highs (+1) and lows (-1) using close-only zig-zag.

    Returns list of ``(index, type, price)`` where *type* is +1 (high) or
    -1 (low).  Kept for backward compatibility.
    """
    if len(prices) < 3:
        return []

    pivots: list[tuple[int, int, float]] = []
    values = prices.values.astype(float)

    last_high_idx = 0
    last_low_idx = 0
    last_high = values[0]
    last_low = values[0]
    direction = 0

    for i in range(1, len(values)):
        val = values[i]

        if direction == 0:
            if val >= last_low * (1 + threshold):
                direction = 1
                last_high = val
                last_high_idx = i
                pivots.append((last_low_idx, -1, last_low))
            elif val <= last_high * (1 - threshold):
                direction = -1
                last_low = val
                last_low_idx = i
                pivots.append((last_high_idx, 1, last_high))
        elif direction == 1:
            if val > last_high:
                last_high = val
                last_high_idx = i
            elif val <= last_high * (1 - threshold):
                pivots.append((last_high_idx, 1, last_high))
                direction = -1
                last_low = val
                last_low_idx = i
        elif direction == -1:
            if val < last_low:
                last_low = val
                last_low_idx = i
            elif val >= last_low * (1 + threshold):
                pivots.append((last_low_idx, -1, last_low))
                direction = 1
                last_high = val
                last_high_idx = i

    return pivots


# Public alias — some callers may still import ``zigzag``
zigzag = _zigzag_legacy


# ---------------------------------------------------------------------------
# Multi-start contraction detection (screener migration)
# ---------------------------------------------------------------------------


def find_contractions_multi_start(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    *,
    min_contraction_days: int = 5,
    contraction_ratio: float = 0.75,
    right_shoulder_pct: float = 0.05,
) -> list[dict[str, Any]]:
    """Try the top-3 swing highs as starting points and return the best
    contraction sequence.

    Each contraction dict contains:
    ``high_idx, low_idx, high_price, low_price, depth_pct, duration_days, range``.
    """
    if not swing_highs or not swing_lows:
        return []

    sorted_highs = sorted(swing_highs, key=lambda x: x[1], reverse=True)

    best: list[dict[str, Any]] = []
    best_score = -1

    for start_high in sorted_highs[:3]:
        candidate = _build_contractions_from(
            start_high,
            swing_highs,
            swing_lows,
            min_contraction_days=min_contraction_days,
            right_shoulder_pct=right_shoulder_pct,
        )
        if len(candidate) < 2:
            continue
        v = validate_vcp(candidate, total_bars=0)
        score = len(candidate) * 10 + (10 if v.get("valid", False) else 0)
        if score > best_score:
            best = candidate
            best_score = score

    return best


def _build_contractions_from(
    start_high: tuple[int, float],
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    *,
    min_contraction_days: int = 5,
    right_shoulder_pct: float = 0.05,
) -> list[dict[str, Any]]:
    """Build a contraction sequence starting from *start_high*."""
    h1_idx, h1_val = start_high
    contractions: list[dict[str, Any]] = []
    current_high_idx = h1_idx
    current_high_val = h1_val

    for _ in range(4):
        # Find next swing low after current high
        next_low = None
        for idx, val in swing_lows:
            if idx > current_high_idx and (idx - current_high_idx) >= min_contraction_days:
                next_low = (idx, val)
                break

        if next_low is None:
            break

        low_idx, low_val = next_low
        depth_pct = (
            (current_high_val - low_val) / current_high_val * 100 if current_high_val > 0 else 0.0
        )
        duration = low_idx - current_high_idx
        range_val = current_high_val - low_val

        # Right-shoulder check: subsequent highs must be within pct of H1
        if contractions:
            pct_from_h1 = abs(current_high_val - h1_val) / h1_val if h1_val > 0 else 0.0
            if pct_from_h1 > right_shoulder_pct:
                break

        contractions.append(
            {
                "high_idx": current_high_idx,
                "low_idx": low_idx,
                "high_price": current_high_val,
                "low_price": low_val,
                "depth_pct": depth_pct,
                "duration_days": duration,
                "range": range_val,
                # legacy compat aliases
                "high": current_high_val,
                "low": low_val,
            }
        )

        # Find next swing high after this low
        next_high = None
        for idx, val in swing_highs:
            if idx > low_idx:
                next_high = (idx, val)
                break
        if next_high is None:
            break

        current_high_idx, current_high_val = next_high

    return contractions


# ---------------------------------------------------------------------------
# Legacy contraction finder (kept for backward compatibility)
# ---------------------------------------------------------------------------


def find_contractions(prices: pd.Series, threshold: float = 0.05) -> list[dict[str, Any]]:
    """Find contraction phases from legacy zig-zag pivots.

    Kept for backward compatibility.
    """
    pivots = _zigzag_legacy(prices, threshold)
    contractions: list[dict[str, Any]] = []

    for i in range(len(pivots) - 1):
        p1 = pivots[i]
        p2 = pivots[i + 1]

        if p1[1] == 1 and p2[1] == -1:
            contractions.append(
                {
                    "high_idx": p1[0],
                    "low_idx": p2[0],
                    "high": p1[2],
                    "low": p2[2],
                    "range": p1[2] - p2[2],
                }
            )

    return contractions


# ---------------------------------------------------------------------------
# VCP validation (screener migration)
# ---------------------------------------------------------------------------


def validate_vcp(
    contractions: list[dict[str, Any]],
    total_bars: int,
    *,
    t1_depth_min: float = 0.08,
    t1_depth_max: float = 0.35,
    contraction_ratio: float = 0.75,
    min_pattern_days: int = 15,
    max_pattern_days: int = 325,
) -> dict[str, Any]:
    """Validate whether contractions form a proper VCP.

    Checks:
    - T1 depth in [t1_depth_min, t1_depth_max] (as fractions, e.g. 0.08 = 8%)
    - Each successive contraction tightens by *contraction_ratio*
    - Pattern duration within [min_pattern_days, max_pattern_days]

    Returns dict with ``valid`` bool and ``issues`` list.
    """
    issues: list[str] = []
    valid = True

    if len(contractions) < 2:
        return {"valid": False, "issues": ["Need at least 2 contractions"]}

    # T1 depth
    t1_depth = contractions[0].get("depth_pct", 0.0)
    t1_frac = t1_depth / 100.0
    if t1_frac < t1_depth_min:
        issues.append(f"T1 depth too shallow ({t1_depth:.1f}%, need >= {t1_depth_min * 100:.0f}%)")
        valid = False
    elif t1_frac > t1_depth_max:
        issues.append(f"T1 depth too deep ({t1_depth:.1f}%, prefer <= {t1_depth_max * 100:.0f}%)")

    # Contraction tightening
    for i in range(1, len(contractions)):
        prev_depth = contractions[i - 1].get("depth_pct", 0.0)
        curr_depth = contractions[i].get("depth_pct", 0.0)
        if prev_depth > 0:
            ratio = curr_depth / prev_depth
            if ratio > contraction_ratio:
                issues.append(
                    f"T{i + 1} ({curr_depth:.1f}%) does not contract vs "
                    f"T{i} ({prev_depth:.1f}%), ratio={ratio:.2f}"
                )
                valid = False

    # Pattern duration
    if len(contractions) >= 2:
        duration = contractions[-1]["low_idx"] - contractions[0]["high_idx"]
        if duration < min_pattern_days:
            issues.append(f"Pattern too short ({duration} days, need >= {min_pattern_days})")
            valid = False
        elif duration > max_pattern_days:
            issues.append(f"Pattern too long ({duration} days, prefer <= {max_pattern_days})")

    return {"valid": valid, "issues": issues}


# ---------------------------------------------------------------------------
# Contraction ratio validation (legacy helper, still used)
# ---------------------------------------------------------------------------


def validate_contraction_ratios(
    contractions: list[dict[str, Any]], max_ratio: float = 0.75
) -> list[dict[str, Any]]:
    """Find the longest sequence of successive contracting ranges.

    Kept for backward compatibility with legacy code paths.
    """
    if len(contractions) < 2:
        return contractions

    best_sequence: list[dict[str, Any]] = [contractions[0]]
    current_sequence: list[dict[str, Any]] = [contractions[0]]

    for i in range(1, len(contractions)):
        prev_range = current_sequence[-1]["range"]
        curr_range = contractions[i]["range"]

        if prev_range > 0 and curr_range / prev_range <= max_ratio:
            current_sequence.append(contractions[i])
        else:
            if len(current_sequence) > len(best_sequence):
                best_sequence = current_sequence[:]
            current_sequence = [contractions[i]]

    if len(current_sequence) > len(best_sequence):
        best_sequence = current_sequence

    return best_sequence


# ---------------------------------------------------------------------------
# Volume checks
# ---------------------------------------------------------------------------


def check_volume_decline(volume: pd.Series, contractions: list[dict[str, Any]]) -> bool:
    """Verify volume is declining across contraction phases."""
    if len(contractions) < 2:
        return True

    avg_vols: list[float] = []
    for c in contractions:
        start = min(c["high_idx"], c["low_idx"])
        end = max(c["high_idx"], c["low_idx"]) + 1
        avg_vols.append(float(volume.iloc[start:end].mean()))

    return all(avg_vols[i] < avg_vols[i - 1] for i in range(1, len(avg_vols)))


def check_breakout(
    prices: pd.Series,
    volume: pd.Series,
    contractions: list[dict[str, Any]],
    breakout_volume_ratio: float = 1.5,
    avg_volume_window: int = 50,
) -> dict[str, Any]:
    """Check for breakout above the pivot point of the final contraction."""
    if not contractions:
        return {"confirmed": False, "breakout_idx": None}

    last_contraction = contractions[-1]
    pivot_price = last_contraction.get("high_price", last_contraction.get("high", 0.0))
    search_start = max(last_contraction["high_idx"], last_contraction["low_idx"]) + 1

    for i in range(search_start, len(prices)):
        if prices.iloc[i] > pivot_price:
            avg_vol_start = max(0, i - avg_volume_window)
            avg_vol = volume.iloc[avg_vol_start:i].mean()
            if avg_vol > 0 and volume.iloc[i] >= breakout_volume_ratio * avg_vol:
                return {"confirmed": True, "breakout_idx": i}

    return {"confirmed": False, "breakout_idx": None}


# ---------------------------------------------------------------------------
# Main detection entry points
# ---------------------------------------------------------------------------


def detect_vcp(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    min_contractions: int = 2,
    contraction_ratio: float = 0.75,
    breakout_volume_ratio: float = 1.5,
    lookback_window: int | None = None,
    atr_multiplier: float = 1.5,
    atr_period: int = 14,
    min_contraction_days: int = 5,
    right_shoulder_pct: float = 0.05,
    t1_depth_min: float = 0.08,
    t1_depth_max: float = 0.35,
    min_pattern_days: int = 15,
    max_pattern_days: int = 325,
    volume_decline: bool = True,
    # legacy — ignored when ATR zigzag succeeds
    swing_threshold: float = 0.05,
) -> dict[str, Any]:
    """Full VCP pattern detection pipeline.

    Uses ATR-based zigzag with multi-start contraction detection,
    falling back to legacy close-only zigzag when ATR is unavailable.

    Returns dict with ``detected``, ``contractions``, ``breakout_idx``,
    ``reason``.
    """
    # Apply lookback window
    offset = 0
    if lookback_window is not None and len(close) > lookback_window:
        offset = len(close) - lookback_window
        high = high.iloc[offset:].reset_index(drop=True)
        low = low.iloc[offset:].reset_index(drop=True)
        close = close.iloc[offset:].reset_index(drop=True)
        volume = volume.iloc[offset:].reset_index(drop=True)

    # Step 1: ATR zigzag → multi-start contractions
    swing_highs, swing_lows = zigzag_atr(
        high,
        low,
        close,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
    )

    use_atr = len(swing_highs) >= 1 and len(swing_lows) >= 1

    if use_atr:
        contractions = find_contractions_multi_start(
            swing_highs,
            swing_lows,
            min_contraction_days=min_contraction_days,
            contraction_ratio=contraction_ratio,
            right_shoulder_pct=right_shoulder_pct,
        )
    else:
        # Fallback to legacy zigzag
        all_contractions = find_contractions(close, threshold=swing_threshold)
        contractions = validate_contraction_ratios(all_contractions, max_ratio=contraction_ratio)

    if len(contractions) < min_contractions:
        return {
            "detected": False,
            "contractions": contractions,
            "breakout_idx": None,
            "reason": (f"Only {len(contractions)} contractions found, need {min_contractions}"),
        }

    # Step 2: VCP validation (ATR path only)
    if use_atr:
        validation = validate_vcp(
            contractions,
            total_bars=len(close),
            t1_depth_min=t1_depth_min,
            t1_depth_max=t1_depth_max,
            contraction_ratio=contraction_ratio,
            min_pattern_days=min_pattern_days,
            max_pattern_days=max_pattern_days,
        )
        if not validation["valid"]:
            return {
                "detected": False,
                "contractions": contractions,
                "breakout_idx": None,
                "reason": "; ".join(validation["issues"]),
            }

    # Step 3: Volume decline
    check_contractions = contractions[: max(min_contractions, 2)]
    if volume_decline and not check_volume_decline(volume, check_contractions):
        return {
            "detected": False,
            "contractions": contractions,
            "breakout_idx": None,
            "reason": "Volume is not declining across contraction phases",
        }

    # Step 4: Breakout confirmation
    breakout = check_breakout(
        close,
        volume,
        contractions,
        breakout_volume_ratio=breakout_volume_ratio,
    )

    if not breakout["confirmed"]:
        return {
            "detected": False,
            "contractions": contractions,
            "breakout_idx": None,
            "reason": "No confirmed breakout with sufficient volume",
        }

    return {
        "detected": True,
        "contractions": contractions,
        "breakout_idx": breakout["breakout_idx"],
        "reason": "",
    }


def detect_vcp_rolling(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window_size: int = 250,
    step_size: int = 20,
    cooldown: int = 40,
    min_contractions: int = 2,
    contraction_ratio: float = 0.75,
    breakout_volume_ratio: float = 1.5,
    atr_multiplier: float = 1.5,
    atr_period: int = 14,
    min_contraction_days: int = 5,
    right_shoulder_pct: float = 0.05,
    t1_depth_min: float = 0.08,
    t1_depth_max: float = 0.35,
    min_pattern_days: int = 15,
    max_pattern_days: int = 325,
    volume_decline: bool = True,
    swing_threshold: float = 0.05,
) -> list[int]:
    """Detect VCP patterns using a sliding window across the full series.

    Returns list of breakout bar indices (in original series coordinates),
    deduplicated with a cooldown gap.
    """
    n = len(close)
    if n < window_size:
        result = detect_vcp(
            high,
            low,
            close,
            volume,
            min_contractions=min_contractions,
            contraction_ratio=contraction_ratio,
            breakout_volume_ratio=breakout_volume_ratio,
            lookback_window=None,
            atr_multiplier=atr_multiplier,
            atr_period=atr_period,
            min_contraction_days=min_contraction_days,
            right_shoulder_pct=right_shoulder_pct,
            t1_depth_min=t1_depth_min,
            t1_depth_max=t1_depth_max,
            min_pattern_days=min_pattern_days,
            max_pattern_days=max_pattern_days,
            volume_decline=volume_decline,
            swing_threshold=swing_threshold,
        )
        if result["detected"] and result["breakout_idx"] is not None:
            return [result["breakout_idx"]]
        return []

    breakout_indices: list[int] = []
    last_breakout = -cooldown

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        w_high = high.iloc[start:end].reset_index(drop=True)
        w_low = low.iloc[start:end].reset_index(drop=True)
        w_close = close.iloc[start:end].reset_index(drop=True)
        w_volume = volume.iloc[start:end].reset_index(drop=True)

        result = detect_vcp(
            w_high,
            w_low,
            w_close,
            w_volume,
            min_contractions=min_contractions,
            contraction_ratio=contraction_ratio,
            breakout_volume_ratio=breakout_volume_ratio,
            lookback_window=None,
            atr_multiplier=atr_multiplier,
            atr_period=atr_period,
            min_contraction_days=min_contraction_days,
            right_shoulder_pct=right_shoulder_pct,
            t1_depth_min=t1_depth_min,
            t1_depth_max=t1_depth_max,
            min_pattern_days=min_pattern_days,
            max_pattern_days=max_pattern_days,
            volume_decline=volume_decline,
            swing_threshold=swing_threshold,
        )

        if result["detected"] and result["breakout_idx"] is not None:
            original_idx = start + result["breakout_idx"]
            in_latter_half = result["breakout_idx"] >= window_size // 2
            if in_latter_half and original_idx - last_breakout >= cooldown:
                breakout_indices.append(original_idx)
                last_breakout = original_idx

    return breakout_indices
