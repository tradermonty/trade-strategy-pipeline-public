"""Tests for Alpaca execution broker adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from pipeline.execution.broker.alpaca import AlpacaBroker
from pipeline.execution.broker.base import (
    BrokerAuthError,
    BrokerNetworkError,
    BrokerOrderRejectError,
    BrokerPayloadError,
    BrokerRateLimitError,
)
from pipeline.execution.models import BrokerOrderRequest


class _FakeClient:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.submitted_order_data: Any | None = None

    def get_account(self) -> dict[str, Any]:
        return {
            "id": "acct-1",
            "status": "ACTIVE",
            "currency": "USD",
            "cash": "10000",
            "equity": "10200",
            "buying_power": "15000",
        }

    def get_all_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "AAPL",
                "qty": "5",
                "avg_entry_price": "100",
                "current_price": "110",
                "market_value": "550",
                "unrealized_pl": "50",
            }
        ]

    def get_orders(self, filter: Any = None) -> list[dict[str, Any]]:
        _ = filter
        return [
            {
                "id": "ord-1",
                "client_order_id": "strat-a:2026-02-22:AAPL:buy:1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "10",
                "filled_qty": "0",
                "status": "accepted",
                "submitted_at": "2026-02-22T10:00:00+00:00",
                "updated_at": "2026-02-22T10:00:00+00:00",
            },
            {
                "id": "ord-2",
                "client_order_id": "other:2026-02-22:MSFT:buy:1",
                "symbol": "MSFT",
                "side": "buy",
                "qty": "10",
                "filled_qty": "0",
                "status": "accepted",
                "submitted_at": "2026-02-22T10:00:00+00:00",
                "updated_at": "2026-02-22T10:00:00+00:00",
            },
        ]

    def submit_order(self, order_data: Any) -> dict[str, Any]:
        self.submitted_order_data = order_data
        return {
            "id": "ord-3",
            "client_order_id": "strat-a:2026-02-22:AAPL:buy:2",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "filled_qty": "0",
            "status": "accepted",
            "submitted_at": "2026-02-22T10:00:01+00:00",
            "updated_at": "2026-02-22T10:00:01+00:00",
        }

    def cancel_order_by_id(self, broker_order_id: str) -> None:
        self.cancelled.append(broker_order_id)

    def get_order_by_id(self, broker_order_id: str) -> dict[str, Any]:
        return {
            "id": broker_order_id,
            "client_order_id": "strat-a:2026-02-22:AAPL:buy:2",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "filled_qty": "10",
            "status": "filled",
            "filled_avg_price": "111.0",
            "submitted_at": "2026-02-22T10:00:01+00:00",
            "updated_at": "2026-02-22T10:01:05+00:00",
        }


def _request() -> BrokerOrderRequest:
    return BrokerOrderRequest(
        strategy_id="strat-a",
        symbol="aapl",
        side="buy",
        qty=10,
        order_type="market",
        time_in_force="day",
        client_order_id="strat-a:2026-02-22:AAPL:buy:2",
    )


def test_alpaca_broker_mode_switches_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broker should preserve configured mode for client creation."""
    calls: list[str] = []

    def fake_create(self: AlpacaBroker) -> object:
        calls.append(self.mode)
        return object()

    monkeypatch.setattr(AlpacaBroker, "_create_trading_client", fake_create)

    AlpacaBroker("key", "secret", mode="paper")._get_client()
    AlpacaBroker("key", "secret", mode="live")._get_client()
    assert calls == ["paper", "live"]


def test_submit_cancel_get_order_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """submit/cancel/get-order should map through adapter correctly."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")
    fake_client = _FakeClient()
    broker._client = fake_client

    monkeypatch.setattr(broker, "_build_order_request", lambda order: {"mock": order.symbol})

    submitted = broker.submit_order(_request())
    assert submitted.broker_order_id == "ord-3"
    assert submitted.symbol == "AAPL"
    assert fake_client.submitted_order_data == {"mock": "AAPL"}

    broker.cancel_order("ord-3")
    assert fake_client.cancelled == ["ord-3"]

    fetched = broker.get_order("ord-3")
    assert fetched.status == "filled"
    assert fetched.filled_qty == 10
    assert fetched.avg_fill_price == 111.0


def test_get_account_and_positions_success() -> None:
    """get_account/list_positions should map broker payloads to snapshots."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")
    broker._client = _FakeClient()

    account = broker.get_account()
    assert account.account_id == "acct-1"
    assert account.cash == 10000.0
    assert account.equity == 10200.0

    positions = broker.list_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 5


def test_list_open_orders_filters_by_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """list_open_orders should only return orders with matching strategy prefix."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")
    broker._client = _FakeClient()
    monkeypatch.setattr(broker, "_build_open_orders_request", lambda: object())

    orders = broker.list_open_orders("strat-a")
    assert len(orders) == 1
    assert orders[0].client_order_id is not None
    assert orders[0].client_order_id.startswith("strat-a:")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("401 unauthorized", BrokerAuthError),
        ("429 rate limit exceeded", BrokerRateLimitError),
        ("network timeout while connecting", BrokerNetworkError),
        ("order rejected by venue", BrokerOrderRejectError),
    ],
)
def test_submit_order_error_mapping(
    monkeypatch: pytest.MonkeyPatch, message: str, expected: type[Exception]
) -> None:
    """submit_order should map broker exceptions to typed broker errors."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")

    class _ErrClient:
        def submit_order(self, order_data: Any) -> dict[str, Any]:
            _ = order_data
            raise RuntimeError(message)

    broker._client = _ErrClient()
    monkeypatch.setattr(broker, "_build_order_request", lambda order: {"mock": order.symbol})

    with pytest.raises(expected):
        broker.submit_order(_request())


def test_to_datetime_handles_iso_and_naive() -> None:
    """Sanity: get_order path should normalize datetime strings safely."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")
    broker._client = _FakeClient()

    out = broker.get_order("ord-xyz")
    assert isinstance(out.submitted_at, datetime)
    assert out.submitted_at.tzinfo is not None
    assert out.updated_at.tzinfo is not None
    assert out.updated_at.year == 2026


def test_list_open_orders_payload_error_mapping() -> None:
    """Malformed broker payload should raise BrokerPayloadError."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")

    class _BadClient:
        def get_orders(self, **kwargs: Any) -> list[dict[str, Any]]:
            _ = kwargs
            return [
                {
                    "id": "ord-bad",
                    "client_order_id": "strat-a:2026-02-22:AAPL:buy:1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": "",
                    "filled_qty": "0",
                    "status": "accepted",
                }
            ]

    broker._client = _BadClient()
    with pytest.raises(BrokerPayloadError, match="Invalid order payload"):
        broker.list_open_orders("strat-a")


def test_list_open_orders_empty_strategy_returns_unfiltered() -> None:
    """Empty strategy_id should return all open orders (including non-prefixed ones)."""
    broker = AlpacaBroker("key", "secret", mode="paper", strategy_id="strat-a")
    broker._client = _FakeClient()

    orders = broker.list_open_orders("")
    assert len(orders) == 2
