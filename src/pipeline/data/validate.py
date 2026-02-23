"""CLI wrapper for OHLCV data validation.

Usage:
    python -m pipeline.data.validate path/to/ohlcv.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pipeline.data.storage import validate_ohlcv_schema


def main(argv: list[str] | None = None) -> int:
    """Validate an OHLCV parquet file and print any errors."""
    args = argv if argv is not None else sys.argv[1:]

    if len(args) != 1:
        print("Usage: python -m pipeline.data.validate <path>", file=sys.stderr)
        return 1

    path = Path(args[0])
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(path)
    errors = validate_ohlcv_schema(df)

    if errors:
        print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Validation passed: {len(df)} rows, {df['symbol'].nunique()} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
