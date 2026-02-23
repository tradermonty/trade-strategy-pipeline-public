"""Volume analysis utilities."""

from __future__ import annotations

import pandas as pd


def avg_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Compute rolling average volume."""
    return volume.rolling(window=window, min_periods=1).mean()


def is_volume_surge(volume: pd.Series, ratio: float = 1.5, window: int = 50) -> pd.Series:
    """Detect volume surges (volume >= ratio * rolling average)."""
    avg = avg_volume(volume, window)
    return volume >= ratio * avg
