"""Tests for execution state reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pipeline.execution.models import ExecutionState, OrderRecord, PositionSnapshot
from pipeline.execution.reconciler import reconcile_execution_state

OrderStatus = Literal[
    "new", "accepted", "partially_filled", "filled", "canceled", "rejected", "expired"
]


def _order(
    *,
    order_id: str,
    filled_qty: int,
    status: OrderStatus,
    avg_fill_price: float | None = None,
) -> OrderRecord:
    return OrderRecord(
        broker_order_id=order_id,
        client_order_id=f"strat-x:2026-02-24:AAPL:buy:{order_id}",
        strategy_id="strat-x",
        symbol="AAPL",
        side="buy",
        qty=10,
        filled_qty=filled_qty,
        status=status,
        avg_fill_price=avg_fill_price,
        submitted_at=datetime(2026, 2, 24, 9, 30, tzinfo=UTC),
        updated_at=datetime(2026, 2, 24, 9, 31, tzinfo=UTC),
    )


def test_reconcile_detects_incremental_fill_delta() -> None:
    """FillEvent should be emitted only for new filled_qty delta."""
    state = ExecutionState(
        strategy_id="strat-x",
        active_orders=[_order(order_id="ord-1", filled_qty=3, status="partially_filled")],
    )
    broker_orders = [_order(order_id="ord-1", filled_qty=10, status="filled", avg_fill_price=111.0)]
    broker_positions = [
        PositionSnapshot(
            strategy_id="strat-x",
            symbol="AAPL",
            qty=10,
            avg_entry_price=111.0,
        )
    ]

    recon = reconcile_execution_state(
        state,
        broker_orders,
        broker_positions,
        now=datetime(2026, 2, 24, 9, 35, tzinfo=UTC),
    )
    assert len(recon.fill_events) == 1
    assert recon.fill_events[0].fill_qty == 7
    assert recon.fill_events[0].fill_price == 111.0
    assert len(recon.state.active_orders) == 0
    assert len(recon.state.positions_cache) == 1


def test_reconcile_keeps_open_orders_without_fill_delta() -> None:
    """Open orders should remain active when no new fills are observed."""
    state = ExecutionState(
        strategy_id="strat-x",
        active_orders=[_order(order_id="ord-2", filled_qty=0, status="accepted")],
    )
    broker_orders = [_order(order_id="ord-2", filled_qty=0, status="accepted")]

    recon = reconcile_execution_state(
        state,
        broker_orders,
        broker_positions=[],
        now=datetime(2026, 2, 24, 9, 35, tzinfo=UTC),
    )
    assert len(recon.fill_events) == 0
    assert len(recon.state.active_orders) == 1
    assert recon.state.active_orders[0].broker_order_id == "ord-2"


def test_reconcile_inferrs_incremental_fill_price_from_cumulative_average() -> None:
    """FillEvent price should reflect new-fill price, not cumulative avg."""
    state = ExecutionState(
        strategy_id="strat-x",
        active_orders=[
            _order(
                order_id="ord-4",
                filled_qty=2,
                status="partially_filled",
                avg_fill_price=100.0,
            )
        ],
    )
    broker_orders = [
        _order(
            order_id="ord-4",
            filled_qty=7,
            status="partially_filled",
            avg_fill_price=110.0,
        )
    ]

    recon = reconcile_execution_state(
        state,
        broker_orders,
        broker_positions=[],
        now=datetime(2026, 2, 24, 9, 36, tzinfo=UTC),
    )
    assert len(recon.fill_events) == 1
    assert recon.fill_events[0].fill_qty == 5
    # incremental price = (7*110 - 2*100) / 5 = 114
    assert recon.fill_events[0].fill_price == 114.0


def test_reconcile_does_not_mutate_input_state() -> None:
    """reconcile_execution_state should return a new state (non-mutating)."""
    state = ExecutionState(
        strategy_id="strat-x",
        active_orders=[_order(order_id="ord-3", filled_qty=0, status="accepted")],
    )
    original_active_len = len(state.active_orders)
    recon = reconcile_execution_state(
        state,
        broker_orders=[],
        broker_positions=[],
        now=datetime(2026, 2, 24, 9, 35, tzinfo=UTC),
    )
    assert len(state.active_orders) == original_active_len
    assert len(recon.state.active_orders) == 0
