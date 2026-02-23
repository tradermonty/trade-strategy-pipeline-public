# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (requires uv)
make setup # uv sync --extra dev

# Run offline tests (default — blocks all HTTP)
make test # CI_OFFLINE=1 uv run pytest --cov=src/pipeline -m "not online"
make test ARGS="-k test_name" # Run a single test

# Run online tests (requires API keys in .env)
make test-online

# Lint & format
make lint # ruff check + ruff format --check
make format # auto-fix formatting and lint issues

# Type checking
make typecheck # uv run mypy src/pipeline

# Full CI (lint → typecheck → test)
make ci
```

## Architecture

This is a **pharmaceutical-style stage-gate pipeline** for systematic trade strategy development. Strategies are defined as YAML specs and progress through validation stages (Discovery → Phase I → Phase II → ...) with automated promotion gates.

### Data Flow

```
strategy.yaml → Pipeline.run()
 1. Load OHLCV Parquet from data dir
 2. Generate signals (VCP detection + trend filter per symbol)
 3. Run backtest (custom event-driven engine, next-day-open execution)
 4. Compute metrics (Sharpe, profit factor, max drawdown, etc.)
 5. Evaluate promotion gates (pass/fail)
 6. Generate Markdown report + PNG charts
```

### Key Modules (`src/pipeline/`)

- **`spec/schema.py`** — Pydantic v2 models for `strategy.yaml`. `StrategySpec` is the root model.
- **`signals/vcp.py`** — VCP pattern detection: zig-zag swing identification → contraction validation → volume decline check → breakout confirmation.
- **`backtest/engine.py`** — Custom event-driven backtest engine. Implements next-day-open execution with fill rules: stop-loss > take-profit > trailing stop priority.
- **`backtest/cost_model.py`** — `CostModel` (commission + slippage), `FillSimulator`, `PositionSizer`.
- **`evaluation/gates.py`** — `evaluate_gates()` checks metrics against promotion criteria.
- **`runner/pipeline.py`** — `Pipeline` class orchestrates the full flow.
- **`data/storage.py`** — Parquet I/O, schema validation, forward-fill, DuckDB query helper.

### CI_OFFLINE Guard

When `CI_OFFLINE=1` (set by default in `make test`):
- All API client classes raise `RuntimeError` on any HTTP request
- `tests/conftest.py` monkeypatches httpx, urllib3, and `socket.create_connection`
- Tests marked `@pytest.mark.online` are auto-skipped

### Test Structure

- **`tests/fixtures/`** — Committed Parquet files with synthetic data
- **`tests/unit/`** — Unit tests for each module
- **`tests/integration/`** — End-to-end pipeline tests using fixture data

### Strategy Spec

Demo strategies live in `strategies/examples/`. The YAML is validated by `StrategySpec` Pydantic model.

## Development Policy

- **TDD**: All features and bug fixes follow Red-Green-Refactor cycle.
- Run `make ci` after each cycle.

## Code Quality

- **Python**: 3.12+, strict mypy, ruff (line-length=100)
- **Coverage**: ≥80% for `src/pipeline/`
- Imports: use `from __future__ import annotations` in all modules
