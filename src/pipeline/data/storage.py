"""Parquet I/O utilities and data normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_OHLCV_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def validate_ohlcv_schema(df: pd.DataFrame) -> list[str]:
    """Validate OHLCV DataFrame has required columns, types, and invariants.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Column presence check
    for col in REQUIRED_OHLCV_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Early return if columns are missing — further checks would fail
    if errors:
        return errors

    # Row count check
    if len(df) == 0:
        errors.append("DataFrame is empty (0 rows)")
        return errors

    # NaN/null check on OHLCV columns
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null values")

    # OHLC invariants: high >= low, high >= open, high >= close
    if (df["high"] < df["low"]).any():
        errors.append("OHLC invariant violated: high < low")
    if (df["high"] < df["open"]).any():
        errors.append("OHLC invariant violated: high < open")
    if (df["high"] < df["close"]).any():
        errors.append("OHLC invariant violated: high < close")

    # Volume non-negative
    if (df["volume"] < 0).any():
        errors.append("Negative volume detected")

    # Timestamp UTC check — naive (tz=None) is also invalid
    tz = df["timestamp"].dt.tz
    if tz is None:
        errors.append("Timestamps are timezone-naive (expected UTC)")
    elif str(tz) != "UTC":
        errors.append(f"Timestamps are not UTC (tz={tz})")

    return errors


def _check_required_columns(df: pd.DataFrame) -> list[str]:
    """Check that required OHLCV columns are present (schema-only)."""
    return [
        f"Missing required column: {col}" for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns
    ]


def write_ohlcv(df: pd.DataFrame, path: Path) -> None:
    """Write OHLCV DataFrame to Parquet with UTC timestamps."""
    errors = _check_required_columns(df)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")

    df = df.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    elif str(df["timestamp"].dt.tz) != "UTC":
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_ohlcv(path: Path) -> pd.DataFrame:
    """Read OHLCV Parquet file, ensuring UTC timestamps."""
    df = pd.read_parquet(path)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def forward_fill_ohlcv(df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    """Forward-fill missing trading days and add is_filled flag.

    Symbols with >5% filled bars are marked via attrs['excluded'].
    """
    df = df.copy()
    tz = df["timestamp"].dt.tz

    full_range = pd.bdate_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=freq,
        tz=tz,
    )

    df = df.set_index("timestamp")
    df = df.reindex(full_range)
    df.index.name = "timestamp"

    is_filled = df["close"].isna()
    df = df.ffill()
    df["is_filled"] = is_filled
    df = df.reset_index()

    # Mark if excessive fills
    fill_pct = df["is_filled"].mean()
    if fill_pct > 0.05:
        df.attrs["excluded"] = True
    else:
        df.attrs["excluded"] = False

    return df


def save_snapshot(df: pd.DataFrame, snapshot_dir: Path, date: str) -> None:
    """Save an immutable date-stamped snapshot."""
    dest = snapshot_dir / date / "ohlcv.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)


def read_snapshot(snapshot_dir: Path, date: str) -> pd.DataFrame:
    """Read a snapshot by date."""
    path = snapshot_dir / date / "ohlcv.parquet"
    return pd.read_parquet(path)


def query_parquet(path: Path, sql: str) -> pd.DataFrame:
    """Query a Parquet file using DuckDB."""
    import duckdb

    con = duckdb.connect()
    result: pd.DataFrame = con.execute(sql.replace("{path}", str(path))).fetchdf()
    return result
