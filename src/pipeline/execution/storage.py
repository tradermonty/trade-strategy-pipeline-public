"""Persistence helpers for execution state and execution journal."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from pipeline.execution.models import ExecutionState

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an exclusive advisory lock for file mutation."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_execution_state(state: ExecutionState, path: Path) -> None:
    """Atomically persist execution state JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")

    with _exclusive_file_lock(lock_path):
        payload = state.model_dump(mode="json")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
            f.write("\n")

        tmp_path.replace(path)


def load_execution_state(path: Path, strategy_id: str | None = None) -> ExecutionState:
    """Load execution state from JSON.

    - If `path` does not exist and `strategy_id` is provided, return empty state.
    - If JSON is malformed or schema invalid, raise RuntimeError with clear message.
    """
    if not path.exists():
        if strategy_id is not None:
            return ExecutionState.empty(strategy_id)
        raise FileNotFoundError(path)

    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to load execution state from {path}: invalid JSON") from exc

    try:
        state = ExecutionState.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"Failed to load execution state from {path}: schema validation failed"
        ) from exc

    if state.version != 1:
        raise RuntimeError(
            f"Failed to load execution state from {path}: "
            f"unsupported state version {state.version} (expected 1)"
        )

    return state


def append_execution_journal(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append journal rows into a Parquet file."""
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")

    with _exclusive_file_lock(lock_path):
        if path.exists():
            existing = pd.read_parquet(path)
            out = pd.concat([existing, new_df], ignore_index=True)
        else:
            out = new_df

        out.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)


def read_execution_journal(path: Path) -> pd.DataFrame:
    """Read execution journal parquet. Returns empty frame if missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
