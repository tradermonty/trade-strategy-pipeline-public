# Contributing

Thank you for considering contributing to Trade Strategy Pipeline!

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch
3. Install development dependencies: `make setup`
4. Make your changes
5. Run the full CI suite: `make ci`
6. Submit a pull request

## Development Workflow

This project follows **TDD (Test-Driven Development)**:

1. **Red** — Write a failing test that defines the expected behavior
2. **Green** — Write the minimal implementation to make the test pass
3. **Refactor** — Clean up while keeping tests green

Run `make ci` after each cycle to ensure lint, typecheck, and full test suite pass.

## Code Standards

- **Python 3.12+** with strict mypy
- **Ruff** for linting and formatting (line-length=100)
- **Coverage**: >= 80% for `src/pipeline/`
- Use `from __future__ import annotations` in all modules
- Tests run offline by default (`CI_OFFLINE=1`)

## What to Contribute

- Bug fixes with regression tests
- New signal modules (with documentation and tests)
- Improved documentation
- Performance improvements with benchmarks

## What NOT to Submit

- Real trading strategies or proprietary parameters
- Real market data or API keys
- Changes that break offline test mode

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.
