#!/usr/bin/env python3
"""Run a demo strategy pipeline using synthetic fixture data.

Usage:
    uv run python scripts/run_demo_pipeline.py --strategy demo_vcp
    uv run python scripts/run_demo_pipeline.py --strategy demo_gap_scored
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEMO_STRATEGIES_DIR = Path("strategies/examples")


def _list_available_demo_strategies() -> list[str]:
    """Return sorted demo strategy IDs under strategies/examples."""
    strategy_ids: set[str] = set()

    # Current format: strategies/examples/<strategy_id>/strategy.yaml
    for strategy_yaml in DEMO_STRATEGIES_DIR.glob("*/strategy.yaml"):
        strategy_ids.add(strategy_yaml.parent.name)

    # Legacy format kept for backward compatibility: <strategy_id>.yaml
    for legacy_yaml in DEMO_STRATEGIES_DIR.glob("*.yaml"):
        strategy_ids.add(legacy_yaml.stem)

    return sorted(strategy_ids)


def _resolve_demo_spec_path(strategy_id: str) -> Path:
    """Resolve demo spec path with current format first, then legacy fallback."""
    current_path = DEMO_STRATEGIES_DIR / strategy_id / "strategy.yaml"
    if current_path.exists():
        return current_path

    legacy_path = DEMO_STRATEGIES_DIR / f"{strategy_id}.yaml"
    if legacy_path.exists():
        return legacy_path

    return current_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run demo pipeline with synthetic data")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Demo strategy ID (e.g., demo_vcp, demo_gap_scored)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="Data directory (defaults to tests/fixtures for synthetic data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports",
    )
    args = parser.parse_args()

    spec_path = _resolve_demo_spec_path(args.strategy)
    if not spec_path.exists():
        print(f"Error: Strategy spec not found: {spec_path}", file=sys.stderr)
        print("Available demo strategies:", file=sys.stderr)
        for strategy_id in _list_available_demo_strategies():
            print(f"  - {strategy_id}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or Path("reports") / args.strategy

    from pipeline.runner.pipeline import Pipeline

    pipeline = Pipeline(
        spec_path=spec_path,
        data_dir=args.data_dir,
        output_dir=output_dir,
        data_profile="sample",
    )

    result = pipeline.run()
    if result.success:
        print(f"Demo pipeline completed successfully for {args.strategy}")
        if result.gate_result:
            status = "PASSED" if result.gate_result.passed else "FAILED"
            print(f"Promotion gate: {status}")
        if result.report_path:
            print(f"Report: {result.report_path}")
        return 0

    print(f"Pipeline failed: {result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
