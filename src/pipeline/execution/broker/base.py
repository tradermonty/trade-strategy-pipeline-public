"""Broker abstraction for execution layer."""

from __future__ import annotations

from typing import Protocol

from pipeline.execution.models import (
    AccountSnapshot,
    BrokerOrderRequest,
    OrderRecord,
    PositionSnapshot,
)


class BrokerError(RuntimeError):
    """Base broker error."""


class BrokerAuthError(BrokerError):
    """Authentication/authorization error."""


class BrokerRateLimitError(BrokerError):
    """Rate limit error."""


class BrokerNetworkError(BrokerError):
    """Connectivity/timeout error."""


class BrokerOrderRejectError(BrokerError):
    """Order rejected by broker."""


class BrokerPayloadError(BrokerError):
    """Broker returned payload that cannot be parsed into domain models."""


class BrokerGateway(Protocol):
    """Broker adapter protocol."""

    def get_account(self) -> AccountSnapshot: ...

    def list_positions(self) -> list[PositionSnapshot]: ...

    def list_open_orders(self, strategy_id: str) -> list[OrderRecord]: ...

    def submit_order(self, order: BrokerOrderRequest) -> OrderRecord: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def get_order(self, broker_order_id: str) -> OrderRecord: ...
