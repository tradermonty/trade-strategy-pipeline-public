# Execution: Paper & Live Trading

## Overview

The execution layer handles the transition from backtest signals to actual order submission. It follows a two-phase daily cycle:

1. **Close cycle** (after market close): Generate signals, plan orders
2. **Open cycle** (before/at market open): Submit orders, reconcile fills

## Safety by Default

- `execution.enabled` defaults to `false`
- Live mode requires **both** CLI `--confirm-live` flag and YAML `execution.confirm_live: true`
- Canary mode limits position size and count during initial live deployment

## CLI Usage

### Paper Trading — Close Cycle

```bash
uv run python -m pipeline.runner.cli execute \
  --strategy demo_gap_scored \
  --phase close \
  --mode paper \
  --session-date 2026-01-15
```

### Paper Trading — Open Cycle

```bash
uv run python -m pipeline.runner.cli execute \
  --strategy demo_gap_scored \
  --phase open \
  --mode paper \
  --session-date 2026-01-15
```

### Reconciliation

```bash
uv run python -m pipeline.runner.cli reconcile \
  --strategy demo_gap_scored \
  --mode paper
```

## Guardrails

Configured in `strategy.yaml`:

```yaml
execution:
  enabled: true
  mode: "paper"
  broker: "alpaca"
  guardrails:
    allow_new_entries: true
    max_daily_loss_pct: 0.03
    max_order_slippage_bps: 50
    emergency_kill_switch: true
  paper_gates:
    min_paper_trades: 50
    max_reject_cancel_ratio: 0.20
    max_partial_fill_ratio: 0.60
```

## Paper Gates

Before promoting to live, paper trading must pass:

| Gate | Threshold |
|------|-----------|
| Minimum paper trades | >= 50 |
| Reject/cancel ratio | <= 20% |
| Partial fill ratio | <= 60% |
| Realized slippage | <= 50 bps |

## State Management

Execution state is stored as JSON in `data/execution/<strategy_id>/state.json`.
Daily journals are stored as Parquet in `data/execution/<strategy_id>/journal/<date>.parquet`.

## Broker Adapter

The broker adapter interface (`execution/broker/base.py`) can be extended to support additional brokers beyond Alpaca.
