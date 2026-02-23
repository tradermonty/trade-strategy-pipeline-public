"""Domain models for Paper/Live execution state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class OrderIntent(BaseModel):
    """Planned order to be submitted in an execution cycle."""

    intent_id: str
    strategy_id: str
    session_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    symbol: str
    side: Literal["buy", "sell"]
    qty: int = Field(gt=0)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    time_in_force: Literal["day", "gtc"] = "day"
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    target_price: float | None = Field(default=None, gt=0)
    client_order_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol must not be empty")
        return s


class BrokerOrderRequest(BaseModel):
    """Broker-agnostic order request."""

    strategy_id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: int = Field(gt=0)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    time_in_force: Literal["day", "gtc"] = "day"
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    client_order_id: str | None = None

    @field_validator("symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol must not be empty")
        return s


class OrderRecord(BaseModel):
    """Broker order snapshot."""

    broker_order_id: str
    client_order_id: str | None = None
    strategy_id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: int = Field(gt=0)
    filled_qty: int = Field(default=0, ge=0)
    status: Literal[
        "new",
        "accepted",
        "partially_filled",
        "filled",
        "canceled",
        "rejected",
        "expired",
    ] = "new"
    avg_fill_price: float | None = Field(default=None, gt=0)
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol must not be empty")
        return s

    @model_validator(mode="after")
    def filled_qty_le_qty(self) -> OrderRecord:
        if self.filled_qty > self.qty:
            raise ValueError("filled_qty must be <= qty")
        return self


class FillEvent(BaseModel):
    """Fill event for an order execution."""

    strategy_id: str
    broker_order_id: str
    client_order_id: str | None = None
    symbol: str
    side: Literal["buy", "sell"]
    fill_qty: int = Field(gt=0)
    fill_price: float = Field(gt=0)
    commission: float = Field(default=0.0, ge=0)
    slippage_bps: float | None = None
    filled_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol must not be empty")
        return s


class PositionSnapshot(BaseModel):
    """Cached position snapshot synced from broker."""

    strategy_id: str
    symbol: str
    qty: int
    avg_entry_price: float = Field(gt=0)
    market_price: float | None = Field(default=None, gt=0)
    market_value: float | None = None
    unrealized_pnl: float | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @field_validator("symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol must not be empty")
        return s


class AccountSnapshot(BaseModel):
    """Broker account snapshot."""

    account_id: str
    status: str
    currency: str = "USD"
    cash: float = 0.0
    equity: float = 0.0
    buying_power: float = 0.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ExecutionState(BaseModel):
    """Durable execution state for restart-safe operation."""

    version: int = 1
    strategy_id: str
    pending_intents: list[OrderIntent] = Field(default_factory=list)
    active_orders: list[OrderRecord] = Field(default_factory=list)
    positions_cache: list[PositionSnapshot] = Field(default_factory=list)
    last_cycle_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def empty(cls, strategy_id: str) -> ExecutionState:
        """Factory for an empty state object."""
        return cls(strategy_id=strategy_id)
