#!/usr/bin/env python3
"""Example: Load pipeline result and export metrics summary.

Usage:
    uv run python scripts/export_report_example.py --report-dir reports/demo_vcp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export report metrics summary")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    report_md = args.report_dir / "report.md"
    if not report_md.exists():
        print(f"Report not found: {report_md}", file=sys.stderr)
        return 1

    text = report_md.read_text()
    lines = text.splitlines()
    metrics: dict[str, str] = {}
    in_table = False

    for line in lines:
        if "| Metric | Value |" in line:
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) == 2:
                metrics[parts[0]] = parts[1]

    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("No metrics table found in report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
