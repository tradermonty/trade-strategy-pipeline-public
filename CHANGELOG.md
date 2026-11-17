# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Unused `pipeline.data.finviz_client.FinvizClient`. It called the unauthenticated
  `finviz.com/export.ashx` endpoint and parsed the raw response as CSV, so an HTML
  sign-in page could be read as data. No pipeline code used it, and the default
  pipeline never needed a Finviz key (#1).

## [0.1.0] - 2026-02-23

### Added

- Public repository baseline with strict lint/typecheck/test gates.
- GitHub Actions workflows for CI and tag-based release publication.
- Issue templates for bug reports and feature requests.
- Extensibility documentation for broker adapters, signals, and spec schema evolution.

### Changed

- Demo pipeline script now resolves example strategy specs from
  `strategies/examples/<strategy_id>/strategy.yaml`.
- README and execution docs aligned to current demo strategy IDs.
- CLI strategy resolution supports both shorthand IDs (for example `demo_vcp`)
  and explicit paths (for example `examples/demo_vcp`).

### Fixed

- Quick Start demo command path mismatch that prevented end-to-end completion.
