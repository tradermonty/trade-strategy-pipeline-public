"""Alpaca broker adapter for Paper/Live execution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import ValidationError

from pipeline.execution.broker.base import (
    BrokerAuthError,
    BrokerError,
    BrokerNetworkError,
    BrokerOrderRejectError,
    BrokerPayloadError,
    BrokerRateLimitError,
)
from pipeline.execution.models import (
    AccountSnapshot,
    BrokerOrderRequest,
    OrderRecord,
    PositionSnapshot,
)

OrderSideLiteral = Literal["buy", "sell"]
OrderStatusLiteral = Literal[
    "new",
    "accepted",
    "partially_filled",
    "filled",
    "canceled",
    "rejected",
    "expired",
]


def _read(obj: Any, key: str, default: Any = None) -> Any:
    """Read attribute/key value from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_scalar(value: Any) -> Any:
    """Normalize enum-like values to scalar."""
    if hasattr(value, "value"):
        return value.value
    return value


def _to_datetime(value: Any) -> datetime:
    """Best-effort datetime conversion with UTC fallback."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(tz=UTC)
    return datetime.now(tz=UTC)


def _to_order_side(value: Any) -> OrderSideLiteral:
    side = str(_to_scalar(value)).lower()
    return "sell" if side == "sell" else "buy"


def _to_order_status(value: Any) -> OrderStatusLiteral:
    status = str(_to_scalar(value)).lower()
    valid: tuple[OrderStatusLiteral, ...] = (
        "new",
        "accepted",
        "partially_filled",
        "filled",
        "canceled",
        "rejected",
        "expired",
    )
    if status in valid:
        return cast(OrderStatusLiteral, status)
    if status in {"cancelled", "done_for_day"}:
        return "canceled"
    if status in {"partially-filled", "partial_fill"}:
        return "partially_filled"
    return "new"


def _to_required_positive_int(value: Any, field_name: str) -> int:
    """Parse a strictly positive integer from broker payload."""
    if value in (None, ""):
        raise ValueError(f"{field_name} is missing")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def _to_non_negative_int(value: Any, field_name: str) -> int:
    """Parse a non-negative integer from broker payload."""
    if value in (None, ""):
        return 0
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got {parsed}")
    return parsed


class AlpacaBroker:
    """Broker adapter for Alpaca paper/live trading API."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        mode: Literal["paper", "live"] = "paper",
        strategy_id: str = "",
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.mode = mode
        self.strategy_id = strategy_id
        self._client: Any | None = None

    def _create_trading_client(self) -> Any:
        from alpaca.trading.client import TradingClient

        return TradingClient(self.api_key, self.secret_key, paper=self.mode == "paper")

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._create_trading_client()
        return self._client

    @staticmethod
    def _map_error(exc: Exception) -> BrokerError:
        msg = str(exc).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            return BrokerAuthError(str(exc))
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return BrokerRateLimitError(str(exc))
        if "timeout" in msg or "timed out" in msg or "connection" in msg or "network" in msg:
            return BrokerNetworkError(str(exc))
        if "reject" in msg:
            return BrokerOrderRejectError(str(exc))
        return BrokerError(str(exc))

    def _to_order_record(self, order: Any) -> OrderRecord:
        qty = _to_required_positive_int(_read(order, "qty"), "qty")
        filled_qty = _to_non_negative_int(_read(order, "filled_qty"), "filled_qty")
        return OrderRecord(
            broker_order_id=str(_read(order, "id", "")),
            client_order_id=_read(order, "client_order_id"),
            strategy_id=self.strategy_id,
            symbol=str(_read(order, "symbol", "")),
            side=_to_order_side(_read(order, "side", "buy")),
            qty=qty,
            filled_qty=filled_qty,
            status=_to_order_status(_read(order, "status", "new")),
            avg_fill_price=(
                float(_read(order, "filled_avg_price"))
                if _read(order, "filled_avg_price") not in (None, "")
                else None
            ),
            submitted_at=_to_datetime(_read(order, "submitted_at")),
            updated_at=_to_datetime(_read(order, "updated_at")),
            raw=_read(order, "raw", {}),
        )

    def _to_position_snapshot(self, position: Any) -> PositionSnapshot:
        market_value = _read(position, "market_value")
        unrealized_pnl = _read(position, "unrealized_pl")
        qty = _to_non_negative_int(_read(position, "qty"), "position.qty")
        avg_entry_price_raw = _read(position, "avg_entry_price")
        if avg_entry_price_raw in (None, ""):
            raise ValueError("position.avg_entry_price is missing")
        try:
            avg_entry_price = float(avg_entry_price_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"position.avg_entry_price is invalid: {avg_entry_price_raw!r}"
            ) from exc
        if avg_entry_price <= 0:
            raise ValueError(f"position.avg_entry_price must be > 0, got {avg_entry_price}")
        return PositionSnapshot(
            strategy_id=self.strategy_id,
            symbol=str(_read(position, "symbol", "")),
            qty=qty,
            avg_entry_price=avg_entry_price,
            market_price=(
                float(_read(position, "current_price"))
                if _read(position, "current_price") not in (None, "")
                else None
            ),
            market_value=float(market_value) if market_value not in (None, "") else None,
            unrealized_pnl=float(unrealized_pnl) if unrealized_pnl not in (None, "") else None,
            updated_at=datetime.now(tz=UTC),
        )

    def _build_order_request(self, order: BrokerOrderRequest) -> Any:
        """Build alpaca order request object from broker-agnostic request."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if order.time_in_force == "day" else TimeInForce.GTC
        common = {
            "symbol": order.symbol,
            "qty": order.qty,
            "side": side,
            "time_in_force": tif,
            "client_order_id": order.client_order_id,
        }

        if order.order_type == "market":
            return MarketOrderRequest(**common)
        if order.order_type == "limit":
            return LimitOrderRequest(**common, limit_price=order.limit_price)
        if order.order_type == "stop":
            return StopOrderRequest(**common, stop_price=order.stop_price)
        return StopLimitOrderRequest(
            **common,
            stop_price=order.stop_price,
            limit_price=order.limit_price,
        )

    def _build_open_orders_request(self) -> Any:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False, limit=500)

    def get_account(self) -> AccountSnapshot:
        try:
            account = self._get_client().get_account()
            return AccountSnapshot(
                account_id=str(_read(account, "id", "")),
                status=str(_to_scalar(_read(account, "status", "UNKNOWN"))),
                currency=str(_read(account, "currency", "USD")),
                cash=float(_read(account, "cash", 0.0)),
                equity=float(_read(account, "equity", 0.0)),
                buying_power=float(_read(account, "buying_power", 0.0)),
                updated_at=datetime.now(tz=UTC),
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

    def list_positions(self) -> list[PositionSnapshot]:
        try:
            positions = self._get_client().get_all_positions()
            return [self._to_position_snapshot(p) for p in positions]
        except (ValidationError, ValueError) as exc:
            raise BrokerPayloadError(f"Invalid position payload from broker: {exc}") from exc
        except Exception as exc:
            raise self._map_error(exc) from exc

    def list_open_orders(self, strategy_id: str) -> list[OrderRecord]:
        try:
            req = self._build_open_orders_request()
            orders = self._get_client().get_orders(filter=req)
            records = [self._to_order_record(o) for o in orders]
            if strategy_id == "":
                return records
            prefix = f"{strategy_id}:"
            filtered = [
                r
                for r in records
                if r.client_order_id is not None and r.client_order_id.startswith(prefix)
            ]
            return filtered
        except (ValidationError, ValueError) as exc:
            raise BrokerPayloadError(f"Invalid order payload from broker: {exc}") from exc
        except Exception as exc:
            raise self._map_error(exc) from exc

    def submit_order(self, order: BrokerOrderRequest) -> OrderRecord:
        try:
            order_data = self._build_order_request(order)
            response = self._get_client().submit_order(order_data=order_data)
            return self._to_order_record(response)
        except (ValidationError, ValueError) as exc:
            raise BrokerPayloadError(f"Invalid submit response payload from broker: {exc}") from exc
        except Exception as exc:
            raise self._map_error(exc) from exc

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            self._get_client().cancel_order_by_id(broker_order_id)
        except Exception as exc:
            raise self._map_error(exc) from exc

    def get_order(self, broker_order_id: str) -> OrderRecord:
        try:
            order = self._get_client().get_order_by_id(broker_order_id)
            return self._to_order_record(order)
        except (ValidationError, ValueError) as exc:
            raise BrokerPayloadError(f"Invalid order payload from broker: {exc}") from exc
        except Exception as exc:
            raise self._map_error(exc) from exc
