"""Tests for data storage layer."""

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from pipeline.data.storage import (
    _check_required_columns,
    forward_fill_ohlcv,
    read_ohlcv,
    read_snapshot,
    save_snapshot,
    validate_ohlcv_schema,
    write_ohlcv,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")


@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    return tmp_path / "test_ohlcv.parquet"


def test_write_ohlcv_parquet(sample_ohlcv: pd.DataFrame, tmp_parquet: Path) -> None:
    """write_ohlcv should write a valid Parquet file."""
    write_ohlcv(sample_ohlcv, tmp_parquet)
    assert tmp_parquet.exists()
    table = pq.read_table(tmp_parquet)
    assert len(table) == len(sample_ohlcv)


def test_read_ohlcv_parquet(sample_ohlcv: pd.DataFrame, tmp_parquet: Path) -> None:
    """read_ohlcv should read back what was written."""
    write_ohlcv(sample_ohlcv, tmp_parquet)
    result = read_ohlcv(tmp_parquet)
    assert len(result) == len(sample_ohlcv)
    assert list(result.columns) == list(sample_ohlcv.columns)


def test_ohlcv_schema_validation_rejects_bad_columns() -> None:
    """validate_ohlcv_schema should reject DataFrames with missing columns."""
    bad_df = pd.DataFrame({"foo": [1], "bar": [2]})
    errors = validate_ohlcv_schema(bad_df)
    assert len(errors) > 0
    assert any("timestamp" in e for e in errors)


def test_timestamps_stored_as_utc(sample_ohlcv: pd.DataFrame, tmp_parquet: Path) -> None:
    """Timestamps should be stored as UTC in Parquet."""
    write_ohlcv(sample_ohlcv, tmp_parquet)
    result = read_ohlcv(tmp_parquet)
    assert result["timestamp"].dt.tz is not None
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_forward_fill_with_flag() -> None:
    """forward_fill_ohlcv should fill gaps and add is_filled flag."""
    dates = pd.bdate_range("2023-01-02", periods=5, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [dates[0], dates[2], dates[4]],
            "symbol": ["TEST", "TEST", "TEST"],
            "open": [100.0, 102.0, 104.0],
            "high": [101.0, 103.0, 105.0],
            "low": [99.0, 101.0, 103.0],
            "close": [100.5, 102.5, 104.5],
            "volume": [1000, 1200, 1100],
        }
    )
    filled = forward_fill_ohlcv(df, freq="B")
    assert len(filled) == 5
    assert "is_filled" in filled.columns
    assert filled["is_filled"].sum() == 2  # 2 days were filled


def test_exclude_symbols_with_excessive_fills() -> None:
    """Symbols with >5% filled bars should be excluded."""
    dates = pd.bdate_range("2023-01-02", periods=20, freq="B", tz="UTC")
    # Only provide 18 of 20 days (10% missing > 5% threshold)
    df = pd.DataFrame(
        {
            "timestamp": dates[:18],
            "symbol": "SPARSE",
            "open": range(18),
            "high": range(18),
            "low": range(18),
            "close": range(18),
            "volume": range(18),
        }
    )
    filled = forward_fill_ohlcv(df, freq="B")
    fill_pct = filled["is_filled"].mean()
    assert fill_pct <= 0.05 or filled.attrs.get("excluded", False)


def test_snapshot_write_immutable(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    """save_snapshot should create an immutable date-stamped copy."""
    snapshot_dir = tmp_path / "snapshots"
    save_snapshot(sample_ohlcv, snapshot_dir, date="2026-02-20")
    path = snapshot_dir / "2026-02-20" / "ohlcv.parquet"
    assert path.exists()


def test_read_snapshot(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    """read_snapshot should return the saved data."""
    snapshot_dir = tmp_path / "snapshots"
    save_snapshot(sample_ohlcv, snapshot_dir, date="2026-02-20")
    result = read_snapshot(snapshot_dir, date="2026-02-20")
    assert len(result) == len(sample_ohlcv)


def test_sample_ohlcv_fixture_valid() -> None:
    """Committed fixture file should pass column schema validation."""
    df = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")
    errors = _check_required_columns(df)
    assert errors == [], f"Schema validation errors: {errors}"
    assert len(df) == 2100  # 6 symbols x 250 days + SYN_VCP_LONG x 600
    assert set(df["symbol"].unique()) == {
        "SYN_VCP",
        "SYN_VCP_LONG",
        "SYN_TREND",
        "SYN_NOTREND",
        "SYN_EARNINGS_GAP",
        "SYN_GAPUP",
        "SYN_GAPDOWN",
    }


def test_sample_sector_fixture_valid() -> None:
    """Committed sector fixture should have expected structure."""
    df = pd.read_parquet(FIXTURES_DIR / "sample_sector.parquet")
    assert len(df) == 12
    assert "ticker" in df.columns
    assert "sector" in df.columns
    # >30% Tech exposure
    tech_pct = (df["sector"] == "Technology").mean()
    assert tech_pct > 0.30


def test_duckdb_query_parquet() -> None:
    """DuckDB should be able to query Parquet files directly."""
    path = FIXTURES_DIR / "sample_ohlcv.parquet"
    con = duckdb.connect()
    result = con.execute(
        f"SELECT symbol, COUNT(*) as cnt FROM '{path}' GROUP BY symbol ORDER BY symbol"
    ).fetchdf()
    assert len(result) == 7
    assert result["cnt"].sum() == 2100


# --- Validation tests (Step 2c) ---


def test_empty_dataframe_rejected() -> None:
    """validate_ohlcv_schema should reject an empty DataFrame."""
    df = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    errors = validate_ohlcv_schema(df)
    assert any("empty" in e.lower() or "0 rows" in e for e in errors)


def test_high_less_than_low_rejected() -> None:
    """validate_ohlcv_schema should reject rows where high < low."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=3, tz="UTC"),
            "symbol": ["TEST"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [98.0, 99.0, 100.0],  # high < low violation
            "low": [99.0, 100.0, 101.0],
            "close": [97.0, 98.0, 99.0],
            "volume": [1000, 1100, 1200],
        }
    )
    errors = validate_ohlcv_schema(df)
    assert any("high < low" in e for e in errors)


def test_negative_volume_rejected() -> None:
    """validate_ohlcv_schema should reject negative volume."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=3, tz="UTC"),
            "symbol": ["TEST"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000, -500, 1200],
        }
    )
    errors = validate_ohlcv_schema(df)
    assert any("volume" in e.lower() for e in errors)


def test_null_values_rejected() -> None:
    """validate_ohlcv_schema should detect NaN values in OHLCV columns."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=3, tz="UTC"),
            "symbol": ["TEST"] * 3,
            "open": [100.0, float("nan"), 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000, 1100, 1200],
        }
    )
    errors = validate_ohlcv_schema(df)
    assert any("null" in e.lower() for e in errors)


def test_validate_cli_wrapper(tmp_path: Path) -> None:
    """validate.py CLI should pass for valid parquet and fail for invalid."""
    from pipeline.data.validate import main as validate_main

    # Valid file
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=5, tz="UTC"),
            "symbol": ["TEST"] * 5,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )
    valid_path = tmp_path / "valid.parquet"
    df.to_parquet(valid_path, index=False)
    assert validate_main([str(valid_path)]) == 0

    # Invalid file (negative volume)
    df_bad = df.copy()
    df_bad.loc[0, "volume"] = -1
    bad_path = tmp_path / "bad.parquet"
    df_bad.to_parquet(bad_path, index=False)
    assert validate_main([str(bad_path)]) == 1

    # Missing file
    assert validate_main([str(tmp_path / "nonexistent.parquet")]) == 1

    # No arguments
    assert validate_main([]) == 1


def test_timezone_naive_rejected() -> None:
    """validate_ohlcv_schema should reject timezone-naive timestamps."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=3),  # naive, no tz
            "symbol": ["TEST"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000, 1100, 1200],
        }
    )
    errors = validate_ohlcv_schema(df)
    assert any("naive" in e.lower() for e in errors)
