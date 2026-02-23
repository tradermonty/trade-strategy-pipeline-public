"""Tests for execution state/journal persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from pipeline.execution.models import ExecutionState, OrderIntent
from pipeline.execution.storage import (
    append_execution_journal,
    load_execution_state,
    read_execution_journal,
    save_execution_state,
)


def _state() -> ExecutionState:
    return ExecutionState(
        strategy_id="test_strategy",
        pending_intents=[
            OrderIntent(
                intent_id="intent-1",
                strategy_id="test_strategy",
                session_date="2026-02-24",
                created_at=datetime(2026, 2, 22, 12, 0, tzinfo=UTC),
                symbol="AAPL",
                side="buy",
                qty=10,
            )
        ],
    )


def test_save_and_load_execution_state_roundtrip(tmp_path: Path) -> None:
    """save_execution_state/load_execution_state should roundtrip."""
    path = tmp_path / "state.json"
    save_execution_state(_state(), path)
    loaded = load_execution_state(path)
    assert loaded.strategy_id == "test_strategy"
    assert len(loaded.pending_intents) == 1
    assert loaded.pending_intents[0].symbol == "AAPL"


def test_save_execution_state_atomic_tmp_cleanup(tmp_path: Path) -> None:
    """Atomic write should not leave *.tmp after successful save."""
    path = tmp_path / "state.json"
    save_execution_state(_state(), path)
    assert path.exists()
    assert not (tmp_path / "state.json.tmp").exists()
    assert (tmp_path / "state.json.lock").exists()


def test_load_execution_state_missing_returns_empty_with_strategy_id(tmp_path: Path) -> None:
    """Missing state file should return ExecutionState.empty when strategy_id passed."""
    path = tmp_path / "missing.json"
    state = load_execution_state(path, strategy_id="new_strategy")
    assert state.strategy_id == "new_strategy"
    assert state.pending_intents == []


def test_load_execution_state_corrupted_json_raises_runtime_error(tmp_path: Path) -> None:
    """Corrupted JSON should raise RuntimeError with clear message."""
    path = tmp_path / "state.json"
    path.write_text("{ this is invalid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        load_execution_state(path)


def test_load_execution_state_schema_error_raises_runtime_error(tmp_path: Path) -> None:
    """Schema-invalid JSON should raise RuntimeError (graceful failure)."""
    path = tmp_path / "state.json"
    path.write_text('{"version": 1, "pending_intents": []}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="schema validation failed"):
        load_execution_state(path)


def test_load_execution_state_version_mismatch_raises_runtime_error(tmp_path: Path) -> None:
    """Unsupported state version should raise RuntimeError."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 2, "strategy_id": "test_strategy", "pending_intents": []}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported state version"):
        load_execution_state(path)


def test_append_execution_journal_create_and_append(tmp_path: Path) -> None:
    """append_execution_journal should create file then append rows."""
    path = tmp_path / "journal.parquet"
    append_execution_journal(
        path,
        [
            {
                "timestamp": datetime(2026, 2, 22, 13, 0, tzinfo=UTC),
                "event_type": "intent_created",
                "symbol": "AAPL",
            }
        ],
    )
    append_execution_journal(
        path,
        [
            {
                "timestamp": datetime(2026, 2, 22, 14, 0, tzinfo=UTC),
                "event_type": "order_submitted",
                "symbol": "AAPL",
            }
        ],
    )

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert list(df["event_type"]) == ["intent_created", "order_submitted"]
    assert not (tmp_path / "journal.parquet.tmp").exists()
    assert (tmp_path / "journal.parquet.lock").exists()


def test_read_execution_journal_missing_returns_empty_df(tmp_path: Path) -> None:
    """read_execution_journal should return empty DataFrame when missing."""
    df = read_execution_journal(tmp_path / "missing.parquet")
    assert df.empty
