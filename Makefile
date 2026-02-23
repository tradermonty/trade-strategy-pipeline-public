.PHONY: setup test lint typecheck ci clean

setup:
	uv sync --extra dev

test:
	CI_OFFLINE=1 uv run pytest --cov=src/pipeline -m "not online" $(ARGS)

test-online:
	uv run pytest -m "online" $(ARGS)

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/pipeline

ci: lint typecheck test

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
