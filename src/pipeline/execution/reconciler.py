"""Broker reconciliation utilities for execution state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from pipeline.execution.models import ExecutionState, FillEvent, OrderRecord, PositionSnapshot

OPEN_ORDER_STATUSES = {"new", "accepted", "partially_filled"}


@dataclass
class ReconcileResult:
    state: ExecutionState
    fill_events: list[FillEvent]


def reconcile_execution_state(
    state: ExecutionState,
    broker_orders: list[OrderRecord],
    broker_positions: list[PositionSnapshot],
    *,
    now: datetime | None = None,
) -> ReconcileResult:
    """Reconcile local state to broker truth and emit fill deltas.

    This function is pure from caller perspective: input ``state`` is not mutated.
    A new ``ExecutionState`` instance is returned in ``ReconcileResult.state``.
    """
    ts = now or datetime.now(tz=UTC)

    prev_by_id = {o.broker_order_id: o for o in state.active_orders}
    fill_events: list[FillEvent] = []

    for order in broker_orders:
        prev = prev_by_id.get(order.broker_order_id)
        prev_filled = prev.filled_qty if prev else 0
        delta = order.filled_qty - prev_filled
        if delta > 0 and order.avg_fill_price is not None:
            fill_price = _infer_incremental_fill_price(
                prev_filled=prev_filled,
                prev_avg_fill_price=prev.avg_fill_price if prev else None,
                new_filled=order.filled_qty,
                new_avg_fill_price=order.avg_fill_price,
                delta_filled=delta,
            )
            fill_events.append(
                FillEvent(
                    strategy_id=order.strategy_id,
                    broker_order_id=order.broker_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    fill_qty=delta,
                    fill_price=fill_price,
                    filled_at=ts,
                )
            )

    next_state = state.model_copy(
        update={
            "active_orders": [o for o in broker_orders if o.status in OPEN_ORDER_STATUSES],
            "positions_cache": broker_positions,
            "last_cycle_at": ts,
        }
    )

    return ReconcileResult(state=next_state, fill_events=fill_events)


def _infer_incremental_fill_price(
    *,
    prev_filled: int,
    prev_avg_fill_price: float | None,
    new_filled: int,
    new_avg_fill_price: float,
    delta_filled: int,
) -> float:
    """Infer price of newly filled shares from cumulative fill snapshots.

    If previous cumulative average is unavailable, fall back to broker's current
    average fill price.
    """
    if prev_filled > 0 and prev_avg_fill_price is not None:
        prev_notional = prev_filled * prev_avg_fill_price
        new_notional = new_filled * new_avg_fill_price
        delta_notional = new_notional - prev_notional
        inferred = delta_notional / delta_filled
        if inferred > 0:
            return float(inferred)
    return float(new_avg_fill_price)
