# Architecture

## Overview

Trade Strategy Pipeline is a **pharmaceutical-style stage-gate pipeline** for systematic trade strategy development. The core idea is borrowed from drug development: strategies must pass through progressively stricter validation stages before being deployed with real capital.

## Stage-Gate Model

```
Discovery → Phase I (Full Sample) → Phase I Statistical (Bootstrap CI)
         → Phase II (Walk-Forward OOS) → Paper Trading → Live Trading
```

Each gate has quantitative pass/fail criteria (min trades, Sharpe, profit factor, max drawdown).

## Data Flow

```
strategy.yaml
      │
      ▼
┌─────────────────┐
│  Pipeline.run()  │
└────────┬────────┘
         │
    ┌────▼────┐     ┌──────────────┐
    │  Load   │────►│  OHLCV       │
    │  Data   │     │  (Parquet)   │
    └────┬────┘     └──────────────┘
         │
    ┌────▼────────────┐
    │  Generate        │
    │  Signals         │
    │  (per symbol)    │
    └────┬────────────┘
         │
    ┌────▼────────────┐
    │  Run Backtest    │
    │  (event-driven)  │
    └────┬────────────┘
         │
    ┌────▼────────────┐
    │  Compute         │
    │  Metrics         │
    └────┬────────────┘
         │
    ┌────▼────────────┐
    │  Evaluate        │
    │  Gates           │
    └────┬────────────┘
         │
    ┌────▼────────────┐
    │  Generate        │
    │  Report          │
    └─────────────────┘
```

## Key Modules

### `src/pipeline/spec/`

Pydantic v2 models for `strategy.yaml`. `StrategySpec` is the root model containing universe, signals, risk, cost model, promotion gates, and detection config.

### `src/pipeline/signals/`

Signal generators:
- **`vcp.py`** — VCP pattern detection: zig-zag swing identification → contraction validation → volume decline check → breakout confirmation
- **`gap_open_scored.py`** — Scored gap-up open detection: 5-element scoring (gap size, trend, volume, MA position) with grade filtering
- **`trend.py`**, **`volume.py`**, **`gap.py`** — Supporting signal modules

### `src/pipeline/backtest/`

Custom event-driven backtest engine (not vectorbt). Implements next-day-open execution with fill rules:
- Exit priority: stop-loss > take-profit > trailing stop
- Gap penetration: if Open crosses stop/target, fill at Open price
- Portfolio mode with score-based entry priority under max_positions constraint

### `src/pipeline/evaluation/`

- **`metrics.py`** — Sharpe, profit factor, max drawdown, win rate, etc.
- **`gates.py`** — `evaluate_gates()` checks metrics against promotion criteria
- **`bootstrap.py`** — Bootstrap confidence intervals on trade-level metrics
- **`walk_forward.py`** — Walk-forward validation with configurable splits

### `src/pipeline/execution/`

Paper/live execution service:
- **`service.py`** — Close cycle (signal generation) and open cycle (order submission)
- **`broker/alpaca.py`** — Alpaca broker adapter
- **`risk.py`** — Runtime risk checks and guardrails
- **`reconciler.py`** — State reconciliation with broker

### `src/pipeline/data/`

- **`storage.py`** — Parquet I/O, schema validation, forward-fill, DuckDB query helper
- **`alpaca_client.py`**, **`fmp_client.py`** — API clients with `CI_OFFLINE` guard

## Execution Model (Day T / Day T+1)

1. Day T close: signals generated
2. Day T+1 open: orders executed
3. Fill simulation: stop-loss → take-profit → trailing stop priority
4. Trailing stop: tighter of 21-day EMA and 10-day low (daily mode) or weekly EMA (weekly mode)

## CI_OFFLINE Guard

When `CI_OFFLINE=1`:
- All API client classes raise `RuntimeError` on any HTTP request
- `tests/conftest.py` monkeypatches httpx, urllib3, and `socket.create_connection`
- Tests marked `@pytest.mark.online` are auto-skipped

## Extensibility

See [`docs/extensibility.md`](extensibility.md) for extension contracts and wiring steps for:
- Broker adapters
- Custom signal modules
- Strategy spec schema evolution
