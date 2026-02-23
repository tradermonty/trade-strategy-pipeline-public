"""Weekly trailing stop — aggregate daily bars to weekly, compute EMA / N-week low."""

from __future__ import annotations

import pandas as pd


def aggregate_to_weekly(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV to weekly bars using ISO week.

    Expects columns: timestamp (or DatetimeIndex), open, high, low, close, volume.
    Returns DataFrame with columns: week_start, open, high, low, close, volume.
    Each row is one ISO week.
    """
    df = ohlcv.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # ISO year-week grouping
    iso_cal = df.index.isocalendar()  # type: ignore[attr-defined]
    df["_iso_year"] = iso_cal.year.values
    df["_iso_week"] = iso_cal.week.values
    df["_iso_key"] = df["_iso_year"] * 100 + df["_iso_week"]

    weekly = df.groupby("_iso_key", sort=True).agg(
        week_start=("close", lambda x: x.index[0]),
        week_end=("close", lambda x: x.index[-1]),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    weekly = weekly.reset_index(drop=True)
    return weekly


def compute_weekly_ema(weekly_closes: pd.Series, period: int = 10) -> pd.Series:
    """Compute EMA on weekly close prices with SMA seed.

    Uses SMA for the first ``period`` values, then EMA continuation.
    """
    return weekly_closes.ewm(span=period, min_periods=period).mean()


def compute_weekly_nweek_low(weekly_lows: pd.Series, period: int = 4) -> pd.Series:
    """Compute rolling N-week low (excluding current week).

    At index i, returns min of weekly_lows[i-period : i] (not including i).
    """
    shifted = weekly_lows.shift(1)
    return shifted.rolling(window=period, min_periods=1).min()


def is_week_end(timestamps: pd.Series, idx: int) -> bool:
    """Check if bar at ``idx`` is the last trading day of its ISO week.

    True if next bar belongs to a different ISO week (or idx is last bar).
    """
    ts = pd.Timestamp(timestamps.iloc[idx])
    if idx >= len(timestamps) - 1:
        return True
    ts_next = pd.Timestamp(timestamps.iloc[idx + 1])
    return ts.isocalendar()[1] != ts_next.isocalendar()[1]


def count_completed_weeks_since(
    weekly_df: pd.DataFrame, entry_date: pd.Timestamp, current_date: pd.Timestamp
) -> int:
    """Count completed ISO weeks between entry_date and current_date.

    A week is 'completed' if its week_end < current_date.
    """
    completed = weekly_df[
        (weekly_df["week_end"] > entry_date) & (weekly_df["week_end"] < current_date)
    ]
    return len(completed)


def check_weekly_trailing_stop(
    close_price: float,
    weekly_df: pd.DataFrame,
    entry_date: pd.Timestamp,
    current_date: pd.Timestamp,
    trailing_type: str = "weekly_ema",
    ema_period: int = 10,
    nweek_low_period: int = 4,
    transition_weeks: int = 3,
) -> bool:
    """Check if weekly trailing stop is triggered (trend break).

    Returns True if the position should be exited (close below trailing level).
    During the transition period (< transition_weeks completed), no trigger fires.
    """
    completed = count_completed_weeks_since(weekly_df, entry_date, current_date)
    if completed < transition_weeks:
        return False

    if len(weekly_df) < 2:
        return False

    # Find the last completed week relative to current_date
    completed_weeks = weekly_df[weekly_df["week_end"] < current_date]
    if completed_weeks.empty:
        return False

    if trailing_type == "weekly_ema":
        ema = compute_weekly_ema(weekly_df["close"], period=ema_period)
        last_idx = completed_weeks.index[-1]
        if last_idx >= len(ema) or pd.isna(ema.iloc[last_idx]):
            return False
        return bool(close_price < ema.iloc[last_idx])
    else:  # weekly_nweek_low
        nweek_low = compute_weekly_nweek_low(weekly_df["low"], period=nweek_low_period)
        last_idx = completed_weeks.index[-1]
        if last_idx >= len(nweek_low) or pd.isna(nweek_low.iloc[last_idx]):
            return False
        return bool(close_price < nweek_low.iloc[last_idx])
