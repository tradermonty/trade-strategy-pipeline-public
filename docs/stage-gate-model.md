# Stage-Gate Model

## Overview

Strategies progress through validation stages with automated promotion gates. This model is inspired by pharmaceutical drug development, where compounds must pass through increasingly rigorous trials.

## Stages

### Discovery

- Define hypothesis and strategy parameters in YAML
- Initial signal exploration with synthetic or historical data
- No quantitative gates — exploratory phase

### Phase I — Full Sample Backtest

- Full-sample backtest on historical data
- Promotion gates:
  - `min_trades >= 200`
  - `max_drawdown <= 15%`
  - `profit_factor >= 1.2`
  - `sharpe >= 1.0` (optional)

### Phase I Statistical — Bootstrap CI

- Bootstrap confidence intervals (10,000 iterations by default)
- Promotion gates:
  - `PF CI lower bound > 1.0`
  - `Sharpe CI lower bound > 0.0`

### Phase II — Walk-Forward OOS

- Walk-forward validation (e.g., 5 splits, 30% OOS)
- Aggregated OOS metrics must pass promotion gates
- Split-level analysis for regime sensitivity

### Paper Trading

- Paper execution via broker API (Alpaca)
- Guardrails: max daily loss, emergency kill switch, position limits
- Paper gates:
  - `min_paper_trades >= 50`
  - `max_reject_cancel_ratio <= 20%`
  - `max_partial_fill_ratio <= 60%`
  - `max_realized_slippage_bps <= 50`

### Live Trading

- Requires explicit double confirmation (CLI flag + YAML flag)
- Canary mode: reduced position size and count
- Full reconciliation and reporting

## Gate Evaluation

Gates are evaluated by `evaluation/gates.py`:

```python
from pipeline.evaluation.gates import evaluate_gates

result = evaluate_gates(
    metrics={"total_trades": 250, "max_drawdown": 0.12, ...},
    gates=promotion_gates_config,
)
print(result.passed)  # True/False
print(result.details)  # Per-metric pass/fail
```

## Configuration

Gates are configured in `strategy.yaml`:

```yaml
promotion_gates:
  min_trades: 200
  max_drawdown: 0.15
  sharpe: 1.0
  profit_factor: 1.2

validation:
  method: "walk_forward"
  n_splits: 5
  oos_ratio: 0.30
  bootstrap:
    n_iterations: 10000
    confidence_level: 0.95
```
