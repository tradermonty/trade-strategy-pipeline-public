# Extensibility Guide

This framework supports extension at several integration points. This document defines the expected interface and wiring steps.

## Broker Adapter API

To add a new broker, implement the `BrokerGateway` protocol in `src/pipeline/execution/broker/base.py`.

Required methods:

- `get_account() -> AccountSnapshot`
- `list_positions() -> list[PositionSnapshot]`
- `list_open_orders(strategy_id: str) -> list[OrderRecord]`
- `submit_order(order: BrokerOrderRequest) -> OrderRecord`
- `cancel_order(broker_order_id: str) -> None`
- `get_order(broker_order_id: str) -> OrderRecord`

Reference implementation:

- `src/pipeline/execution/broker/alpaca.py`

Integration steps:

1. Add a new adapter module under `src/pipeline/execution/broker/`.
2. Instantiate the adapter in `src/pipeline/runner/cli.py` for `execute` and `reconcile` commands.
3. Add unit tests for adapter error mapping (`BrokerAuthError`, `BrokerRateLimitError`, etc.).

## Signal Extension Contract

Custom signals are generated inside `Pipeline._generate_signals()` in `src/pipeline/runner/pipeline.py`.

Current required signal columns:

- `timestamp`
- `symbol`
- `entry` (`bool`)
- `stop_price` (`float`, optional via `NaN`)
- `entry_at_open` (`bool`)
- `score` (`float`)

Integration steps:

1. Add a new signal module under `src/pipeline/signals/`.
2. Add strategy config fields to `src/pipeline/spec/schema.py` if the signal requires new parameters.
3. Wire dispatch logic in `Pipeline._generate_signals()` to populate the required columns.
4. Add unit tests under `tests/unit/` and one integration test under `tests/integration/`.

## Strategy Spec Extensibility

Strategy YAML is validated by `StrategySpec` in `src/pipeline/spec/schema.py`. New top-level config blocks should:

1. Use explicit typed models with bounds/defaults.
2. Include backward-compatible alias behavior where needed.
3. Add validation tests in `tests/unit/test_spec_schema.py`.

## Legacy Alias Notice

`src/pipeline/signals/earnings_gap_open.py` is a backward-compatible import shim.
Use `src/pipeline/signals/gap_open_scored.py` for new code.
