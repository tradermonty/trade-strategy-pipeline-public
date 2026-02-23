"""Tests for execution close-cycle orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.execution.service import run_close_cycle
from pipeline.execution.storage import load_execution_state


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "entry": True, "score": 0.9, "qty": 1},
            {"symbol": "MSFT", "entry": True, "score": 0.8, "qty": 1},
            {"symbol": "GOOG", "entry": False, "score": 1.0, "qty": 1},
        ]
    )


def test_run_close_cycle_persists_state_and_journal(tmp_path: Path) -> None:
    """Close cycle should create pending intents and write journal/state."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"

    result = run_close_cycle(
        strategy_id="strat-x",
        session_date="2026-02-24",
        signals=_signals(),
        state_path=state_path,
        journal_path=journal_path,
        max_positions=5,
        max_sector_exposure=1.0,
    )

    assert result.accepted_count == 2
    assert state_path.exists()
    assert journal_path.exists()

    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 2
    assert {i.symbol for i in state.pending_intents} == {"AAPL", "MSFT"}

    journal = pd.read_parquet(journal_path)
    assert len(journal) == 2
    assert set(journal["event_type"]) == {"intent_created"}


def test_run_close_cycle_dry_run_does_not_persist_state(tmp_path: Path) -> None:
    """dry_run should avoid state persistence while still journaling decisions."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"

    result = run_close_cycle(
        strategy_id="strat-x",
        session_date="2026-02-24",
        signals=_signals(),
        state_path=state_path,
        journal_path=journal_path,
        max_positions=5,
        max_sector_exposure=1.0,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.accepted_count == 2
    assert not state_path.exists()
    assert journal_path.exists()


def test_run_close_cycle_idempotent_replay_no_duplicate_pending(tmp_path: Path) -> None:
    """Re-running close cycle with same session/signals should not duplicate intents."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"

    first = run_close_cycle(
        strategy_id="strat-x",
        session_date="2026-02-24",
        signals=_signals(),
        state_path=state_path,
        journal_path=journal_path,
        max_positions=5,
        max_sector_exposure=1.0,
    )
    second = run_close_cycle(
        strategy_id="strat-x",
        session_date="2026-02-24",
        signals=_signals(),
        state_path=state_path,
        journal_path=journal_path,
        max_positions=5,
        max_sector_exposure=1.0,
    )

    assert first.accepted_count == 2
    assert second.accepted_count == 0

    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 2
