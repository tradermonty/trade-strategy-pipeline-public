"""Tests for execution domain models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pipeline.execution.models import (
    ExecutionState,
    FillEvent,
    OrderIntent,
    OrderRecord,
    PositionSnapshot,
)


def _ts() -> datetime:
    return datetime(2026, 2, 22, 12, 0, tzinfo=UTC)


def test_execution_state_roundtrip() -> None:
    """ExecutionState should survive model_dump/model_validate roundtrip."""
    state = ExecutionState(
        strategy_id="test_strategy",
        pending_intents=[
            OrderIntent(
                intent_id="intent-1",
                strategy_id="test_strategy",
                session_date="2026-02-24",
                created_at=_ts(),
                symbol="aapl",
                side="buy",
                qty=10,
                client_order_id="coid-1",
            )
        ],
        active_orders=[
            OrderRecord(
                broker_order_id="ord-1",
                strategy_id="test_strategy",
                symbol="msft",
                side="buy",
                qty=10,
                filled_qty=3,
                status="partially_filled",
                submitted_at=_ts(),
                updated_at=_ts(),
            )
        ],
        positions_cache=[
            PositionSnapshot(
                strategy_id="test_strategy",
                symbol="nvda",
                qty=5,
                avg_entry_price=500.0,
                updated_at=_ts(),
            )
        ],
        last_cycle_at=_ts(),
        metadata={"cycle": "open"},
    )

    dumped = state.model_dump(mode="json")
    restored = ExecutionState.model_validate(dumped)

    assert restored == state
    assert restored.pending_intents[0].symbol == "AAPL"
    assert restored.active_orders[0].symbol == "MSFT"
    assert restored.positions_cache[0].symbol == "NVDA"


def test_order_record_filled_qty_must_not_exceed_qty() -> None:
    """OrderRecord should reject filled_qty > qty."""
    with pytest.raises(ValidationError, match="filled_qty"):
        OrderRecord(
            broker_order_id="ord-1",
            strategy_id="test_strategy",
            symbol="AAPL",
            side="buy",
            qty=10,
            filled_qty=11,
        )


def test_execution_state_empty_factory() -> None:
    """ExecutionState.empty should create a zero-state object."""
    state = ExecutionState.empty("my_strategy")
    assert state.strategy_id == "my_strategy"
    assert state.pending_intents == []
    assert state.active_orders == []
    assert state.positions_cache == []


def test_fill_event_validation() -> None:
    """FillEvent should validate positive quantity/price."""
    event = FillEvent(
        strategy_id="test_strategy",
        broker_order_id="ord-1",
        symbol="tsla",
        side="buy",
        fill_qty=2,
        fill_price=200.0,
    )
    assert event.symbol == "TSLA"

    with pytest.raises(ValidationError):
        FillEvent(
            strategy_id="test_strategy",
            broker_order_id="ord-1",
            symbol="TSLA",
            side="buy",
            fill_qty=0,
            fill_price=200.0,
        )
