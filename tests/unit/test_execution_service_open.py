"""Tests for execution open-cycle orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

import pipeline.execution.service as execution_service
from pipeline.execution.broker.base import BrokerRateLimitError
from pipeline.execution.models import (
    AccountSnapshot,
    BrokerOrderRequest,
    ExecutionState,
    OrderIntent,
    OrderRecord,
    PositionSnapshot,
)
from pipeline.execution.service import run_open_cycle
from pipeline.execution.storage import load_execution_state, save_execution_state

OrderStatus = Literal[
    "new", "accepted", "partially_filled", "filled", "canceled", "rejected", "expired"
]


def _intent(symbol: str, seq: int) -> OrderIntent:
    return OrderIntent(
        intent_id=f"intent-{seq}",
        strategy_id="strat-x",
        session_date="2026-02-24",
        symbol=symbol,
        side="buy",
        qty=1,
        client_order_id=f"strat-x:2026-02-24:{symbol}:buy:{seq}",
    )


def _order(
    symbol: str, seq: int, status: OrderStatus = "accepted", filled_qty: int = 0
) -> OrderRecord:
    return OrderRecord(
        broker_order_id=f"ord-{seq}",
        client_order_id=f"strat-x:2026-02-24:{symbol}:buy:{seq}",
        strategy_id="strat-x",
        symbol=symbol,
        side="buy",
        qty=10,
        filled_qty=filled_qty,
        status=status,
        avg_fill_price=100.0 if filled_qty > 0 else None,
        submitted_at=datetime(2026, 2, 24, 9, 30, tzinfo=UTC),
        updated_at=datetime(2026, 2, 24, 9, 31, tzinfo=UTC),
    )


class _FakeBroker:
    def __init__(self) -> None:
        self.open_orders: dict[str, OrderRecord] = {}
        self.orders_by_id: dict[str, OrderRecord] = {}
        self.submit_calls = 0

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(account_id="acct-1", status="ACTIVE")

    def list_positions(self) -> list[PositionSnapshot]:
        return []

    def list_open_orders(self, strategy_id: str) -> list[OrderRecord]:
        _ = strategy_id
        return list(self.open_orders.values())

    def submit_order(self, order: BrokerOrderRequest) -> OrderRecord:
        self.submit_calls += 1
        symbol = order.symbol
        seq = self.submit_calls
        rec = _order(symbol, seq, status="accepted", filled_qty=0)
        rec.client_order_id = order.client_order_id
        self.open_orders[rec.broker_order_id] = rec
        self.orders_by_id[rec.broker_order_id] = rec
        return rec

    def cancel_order(self, broker_order_id: str) -> None:
        self.open_orders.pop(broker_order_id, None)

    def get_order(self, broker_order_id: str) -> OrderRecord:
        return self.orders_by_id[broker_order_id]


class _PartiallyFailingBroker(_FakeBroker):
    def submit_order(self, order: BrokerOrderRequest) -> OrderRecord:
        if order.symbol == "MSFT":
            raise RuntimeError("simulated submit failure")
        return super().submit_order(order)


class _FilteringBroker(_FakeBroker):
    """Mimic strategy-prefixed filtering behavior of the real adapter."""

    def list_open_orders(self, strategy_id: str) -> list[OrderRecord]:
        if strategy_id == "":
            return list(self.open_orders.values())
        prefix = f"{strategy_id}:"
        return [
            o
            for o in self.open_orders.values()
            if o.client_order_id is not None and o.client_order_id.startswith(prefix)
        ]


class _ReconcileFailingBroker(_FakeBroker):
    def list_positions(self) -> list[PositionSnapshot]:
        raise RuntimeError("simulated reconcile failure")


class _RetryingSubmitBroker(_FakeBroker):
    def __init__(self) -> None:
        super().__init__()
        self.submit_attempts = 0

    def submit_order(self, order: BrokerOrderRequest) -> OrderRecord:
        self.submit_attempts += 1
        if self.submit_attempts == 1:
            raise BrokerRateLimitError("429 rate limit")
        return super().submit_order(order)


class _LossGuardBroker(_FakeBroker):
    def __init__(self, equity: float) -> None:
        super().__init__()
        self._equity = equity

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(account_id="acct-1", status="ACTIVE", equity=self._equity)


def test_run_open_cycle_submits_pending_and_clears_state(tmp_path: Path) -> None:
    """Pending intents should be submitted and cleared from state."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    save_execution_state(
        ExecutionState(
            strategy_id="strat-x",
            pending_intents=[_intent("AAPL", 1), _intent("MSFT", 2)],
        ),
        state_path,
    )
    broker = _FakeBroker()

    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )

    assert result.submitted_count == 2
    assert result.skipped_existing_count == 0
    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 0
    assert len(state.active_orders) == 2
    journal = pd.read_parquet(journal_path)
    assert (journal["event_type"] == "order_submitted").sum() == 2


def test_run_open_cycle_idempotent_when_order_already_exists(tmp_path: Path) -> None:
    """If broker already has matching client_order_id, open cycle should skip re-submit."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    pending = _intent("AAPL", 1)
    save_execution_state(
        ExecutionState(strategy_id="strat-x", pending_intents=[pending]),
        state_path,
    )

    broker = _FakeBroker()
    existing = _order("AAPL", 99, status="accepted", filled_qty=0)
    existing.client_order_id = pending.client_order_id
    broker.open_orders[existing.broker_order_id] = existing
    broker.orders_by_id[existing.broker_order_id] = existing

    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )

    assert result.submitted_count == 0
    assert result.skipped_existing_count == 1
    assert broker.submit_calls == 0
    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 0
    assert len(state.active_orders) == 1


def test_run_open_cycle_reconcile_fill_event(tmp_path: Path) -> None:
    """Open cycle should emit fill event when order filled_qty increases."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    active = _order("AAPL", 1, status="partially_filled", filled_qty=2)
    save_execution_state(ExecutionState(strategy_id="strat-x", active_orders=[active]), state_path)

    broker = _FakeBroker()
    # Not open anymore -> list_open_orders empty, get_order returns filled.
    broker.orders_by_id[active.broker_order_id] = _order(
        "AAPL",
        1,
        status="filled",
        filled_qty=10,
    )

    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )

    assert result.fill_events_count == 1
    assert result.active_orders_count == 0
    journal = pd.read_parquet(journal_path)
    assert (journal["event_type"] == "fill_detected").sum() == 1


def test_run_open_cycle_submit_partial_failure_keeps_failed_pending(tmp_path: Path) -> None:
    """Failed submit should keep only failed intent in pending state."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    save_execution_state(
        ExecutionState(
            strategy_id="strat-x",
            pending_intents=[_intent("AAPL", 1), _intent("MSFT", 2)],
        ),
        state_path,
    )

    broker = _PartiallyFailingBroker()
    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )
    assert result.submitted_count == 1
    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 1
    assert state.pending_intents[0].symbol == "MSFT"
    journal = pd.read_parquet(journal_path)
    assert (journal["event_type"] == "order_submit_failed").sum() == 1


def test_run_open_cycle_reconcile_failure_persists_submission_progress(tmp_path: Path) -> None:
    """Submitted orders should persist even if reconciliation fails afterwards."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    save_execution_state(
        ExecutionState(
            strategy_id="strat-x",
            pending_intents=[_intent("AAPL", 1)],
        ),
        state_path,
    )

    broker = _ReconcileFailingBroker()
    try:
        run_open_cycle(
            strategy_id="strat-x",
            state_path=state_path,
            journal_path=journal_path,
            broker=broker,
        )
    except RuntimeError as exc:
        assert "reconciliation failed" in str(exc).lower()
    else:
        raise AssertionError("Expected reconciliation failure")

    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 0
    assert len(state.active_orders) == 1
    assert state.active_orders[0].symbol == "AAPL"
    assert state.last_cycle_at is not None

    journal = pd.read_parquet(journal_path)
    assert (journal["event_type"] == "order_submitted").sum() == 1
    assert (journal["event_type"] == "reconcile_failed").sum() == 1


def test_run_open_cycle_reconcile_tracks_unmanaged_open_orders(tmp_path: Path) -> None:
    """Reconcile should include unmanaged open orders for downstream risk visibility."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    save_execution_state(ExecutionState(strategy_id="strat-x"), state_path)

    broker = _FilteringBroker()
    unmanaged = _order("TSLA", 77, status="accepted", filled_qty=0)
    unmanaged.client_order_id = None
    broker.open_orders[unmanaged.broker_order_id] = unmanaged
    broker.orders_by_id[unmanaged.broker_order_id] = unmanaged

    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )

    assert result.active_orders_count == 1
    state = load_execution_state(state_path)
    assert len(state.active_orders) == 1
    assert state.active_orders[0].symbol == "TSLA"
    assert state.active_orders[0].client_order_id is None


def test_run_open_cycle_retries_transient_submit_error(tmp_path: Path) -> None:
    """Transient broker rate-limit errors should be retried before failing."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    save_execution_state(
        ExecutionState(strategy_id="strat-x", pending_intents=[_intent("AAPL", 1)]),
        state_path,
    )

    broker = _RetryingSubmitBroker()
    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )

    assert result.submitted_count == 1
    assert broker.submit_attempts == 2


def test_run_open_cycle_daily_loss_guard_skips_new_submissions(tmp_path: Path) -> None:
    """When daily loss guard breaches threshold, intents remain pending."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    save_execution_state(
        ExecutionState(
            strategy_id="strat-x",
            pending_intents=[_intent("AAPL", 1)],
            metadata={"risk_session_date": today, "peak_equity": 100.0},
        ),
        state_path,
    )

    broker = _LossGuardBroker(equity=90.0)
    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
        max_daily_loss_pct=0.05,
    )

    assert result.submitted_count == 0
    state = load_execution_state(state_path)
    assert len(state.pending_intents) == 1
    journal = pd.read_parquet(journal_path)
    assert (journal["event_type"] == "risk_daily_loss_triggered").sum() == 1
    assert (journal["event_type"] == "order_submit_skipped_risk").sum() == 1


def test_run_open_cycle_persists_incremental_submission_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful submissions should persist progress before final reconcile."""
    state_path = tmp_path / "state.json"
    journal_path = tmp_path / "journal.parquet"
    save_execution_state(
        ExecutionState(
            strategy_id="strat-x",
            pending_intents=[_intent("AAPL", 1), _intent("MSFT", 2)],
        ),
        state_path,
    )

    save_snapshots: list[tuple[int, int]] = []
    real_save = save_execution_state

    def _spy_save(state: ExecutionState, path: Path) -> None:
        save_snapshots.append((len(state.pending_intents), len(state.active_orders)))
        real_save(state, path)

    monkeypatch.setattr(execution_service, "save_execution_state", _spy_save)

    broker = _FakeBroker()
    result = run_open_cycle(
        strategy_id="strat-x",
        state_path=state_path,
        journal_path=journal_path,
        broker=broker,
    )

    assert result.submitted_count == 2
    # two per-intent checkpoints + final persisted reconcile state
    assert len(save_snapshots) >= 3
    assert (1, 1) in save_snapshots
    assert any(pending == 0 for pending, _ in save_snapshots)
