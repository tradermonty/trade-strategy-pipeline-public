"""Deprecated compatibility shim for scored gap-up open signals.

New code should import from ``pipeline.signals.gap_open_scored``.
"""

from __future__ import annotations

import warnings

from pipeline.signals.gap_open_scored import (
    _score_gap_size,
    _score_ma_position,
    _score_pre_trend,
    _score_volume_trend,
    compute_element_scores,
    compute_weighted_score,
    detect_earnings_gap_open,
    detect_gap_open_scored,
    find_gap_open_candidates,
    score_to_grade,
    should_skip_entry,
)

warnings.warn(
    "pipeline.signals.earnings_gap_open is deprecated; use pipeline.signals.gap_open_scored",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "_score_gap_size",
    "_score_ma_position",
    "_score_pre_trend",
    "_score_volume_trend",
    "compute_element_scores",
    "compute_weighted_score",
    "detect_earnings_gap_open",
    "detect_gap_open_scored",
    "find_gap_open_candidates",
    "score_to_grade",
    "should_skip_entry",
]
