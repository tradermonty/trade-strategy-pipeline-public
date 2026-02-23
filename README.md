# Trade Strategy Pipeline

> **NOT FINANCIAL ADVICE** — This repository is a software engineering framework for systematic trade strategy development.
> It does **not** contain live strategies, real trading data, or investment recommendations.
> Any use for actual trading is entirely at your own risk.

A **pharmaceutical-style stage-gate pipeline** for systematic trade strategy development. Strategies are defined as YAML specs and progress through validation stages (Discovery → Phase I → Phase II → Live) with automated promotion gates.

## Features

- **YAML-driven strategy specs** — Define universe, signals, risk, cost model, and promotion gates declaratively
- **Stage-gate validation** — Automated promotion gates with configurable thresholds (Sharpe, profit factor, max drawdown, etc.)
- **Custom backtest engine** — Event-driven, next-day-open execution with stop-loss > take-profit > trailing stop priority
- **Walk-forward validation** — Out-of-sample evaluation with configurable splits and OOS ratio
- **Bootstrap CI** — Statistical confidence intervals on backtest metrics
- **Paper/Live execution** — Broker integration (Alpaca) with guardrails, kill switch, and reconciliation
- **Automated reporting** — Markdown reports with equity curves, drawdown charts, and metrics tables
- **Extensible architecture** — Clear extension points for broker adapters and custom signals ([docs/extensibility.md](docs/extensibility.md))

## Architecture

```
strategy.yaml → Pipeline.run()
 1. Load OHLCV data (Parquet)
 2. Generate signals (VCP / Gap-Up / custom per symbol)
 3. Run backtest (event-driven engine, next-day-open execution)
 4. Compute metrics (Sharpe, profit factor, max drawdown, etc.)
 5. Evaluate promotion gates (pass/fail)
 6. Generate Markdown report + PNG charts
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/YOUR_USERNAME/trade-strategy-pipeline.git
cd trade-strategy-pipeline
make setup   # uv sync --extra dev
```

### Run tests (offline, no API keys required)

```bash
make test    # CI_OFFLINE=1 — all HTTP blocked
```

> Tests run with `CI_OFFLINE=1` by default. All API clients raise `RuntimeError` when this flag is set, ensuring no external dependencies.

### Run demo pipeline (synthetic data)

```bash
uv run python scripts/run_demo_pipeline.py --strategy demo_vcp
```

### Full CI (lint + typecheck + test)

```bash
make ci
```

## Project Structure

```
src/pipeline/
  spec/         — Pydantic v2 models for strategy YAML
  signals/      — Signal generators (VCP, gap-up, trend, volume)
  backtest/     — Event-driven backtest engine + cost model
  evaluation/   — Metrics, gates, bootstrap, walk-forward
  execution/    — Paper/live execution service + broker adapters
  runner/       — Pipeline orchestrator + CLI
  reporting/    — Markdown + chart generation
  data/         — Parquet I/O, API clients (Alpaca, FMP, Finviz)

strategies/
  examples/     — Demo strategies (synthetic data only)

tests/
  unit/         — Unit tests for each module
  integration/  — End-to-end pipeline tests
  fixtures/     — Synthetic Parquet data + generation script
```

## Strategy Spec

Each strategy is a YAML file validated by the `StrategySpec` Pydantic model:

```yaml
id: "demo_vcp"
name: "Demo VCP Trend Following"
universe:
  type: "us_equities"
  index: "sp500+nasdaq100"
signals:
  entry:
    type: "pivot_breakout"
    conditions: ["vcp_pattern_detected", "breakout_above_pivot_point"]
  exit:
    stop_loss: "7% below entry"
    trailing_stop: "below 21-day EMA or 10-day low"
    take_profit: "risk_reward_3x"
risk:
  risk_per_trade: 0.01
  max_positions: 5
promotion_gates:
  min_trades: 100
  max_drawdown: 0.15
  profit_factor: 1.2
```

## Execution Model

- Signal at Day T close → execute at Day T+1 Open
- Exit priority: stop-loss → take-profit → trailing stop
- Gap penetration: if Open crosses stop/target, fill at Open price
- Paper trading with full guardrails before live deployment

## Online Tests (optional)

To run tests that require API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
make test-online
```

## Disclaimer

This software is provided "as is", without warranty of any kind. This is **not** financial advice.
The authors are not responsible for any financial losses incurred through the use of this software.
Past performance of any strategy does not guarantee future results.

Market data usage is subject to the terms of service of respective data providers.
Broker API usage is subject to the terms of service of the respective broker.

## License

[MIT](LICENSE)
