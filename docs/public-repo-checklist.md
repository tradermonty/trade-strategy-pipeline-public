# Public Repository Checklist

This document tracks the preparation steps for the public release of this framework.

See the [architecture](architecture.md) and [stage-gate model](stage-gate-model.md) for technical details.

## Principles

- **Open Core**: Framework is public; strategies, data, and operational parameters are private
- **Reproducible**: Synthetic data allows README Quick Start to complete without external dependencies
- **Safe by default**: Live execution requires explicit opt-in confirmation
- **CI_OFFLINE=1**: Default test mode blocks all HTTP requests

## Operational Checklist

- [x] Quick Start command uses current demo strategy IDs
- [x] Demo runner resolves `strategies/examples/<strategy_id>/strategy.yaml`
- [x] `uv.lock` tracked for reproducible dependency resolution
- [x] GitHub Actions CI workflow runs `make ci` on push/PR
- [x] GitHub Issue templates configured for bug/feature reports
- [x] Tag-based GitHub Release workflow publishes build artifacts
