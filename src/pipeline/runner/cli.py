"""CLI entry point for the trade strategy pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pipeline.spec.schema import ExecutionConfig, StrategySpec


def _build_default_paper_gates(execution: ExecutionConfig | None) -> dict[str, float]:
    """Resolve paper-gate thresholds from spec with safe defaults."""
    if execution is None:
        return {
            "min_paper_trades": 50.0,
            "max_reject_cancel_ratio": 0.20,
            "max_partial_fill_ratio": 0.60,
            "max_realized_slippage_bps": 50.0,
        }

    pg = execution.paper_gates
    gates: dict[str, float] = {
        "min_paper_trades": float(pg.min_paper_trades),
        "max_reject_cancel_ratio": float(pg.max_reject_cancel_ratio),
        "max_partial_fill_ratio": float(pg.max_partial_fill_ratio),
    }
    slippage_bps = pg.max_realized_slippage_bps
    if slippage_bps is None:
        slippage_bps = float(execution.guardrails.max_order_slippage_bps)
    gates["max_realized_slippage_bps"] = float(slippage_bps)

    if pg.max_backtest_deviation is not None:
        gates["max_backtest_deviation"] = float(pg.max_backtest_deviation)
    if pg.min_profit_factor is not None:
        gates["min_profit_factor"] = float(pg.min_profit_factor)
    if pg.max_drawdown is not None:
        gates["max_drawdown"] = float(pg.max_drawdown)
    return gates


def _load_metrics_json(path: Path | None) -> dict[str, float] | None:
    """Load numeric metrics from a JSON file.

    Supports either:
    - {"metrics": {...}}
    - direct flat dict {...}
    """
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    raw_metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if raw_metrics is None and isinstance(payload, dict):
        raw_metrics = payload
    if not isinstance(raw_metrics, dict):
        return None
    out: dict[str, float] = {}
    for key, value in raw_metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or None


def _load_backtest_metrics_from_report(report_path: Path) -> dict[str, float] | None:
    """Extract backtest metrics from markdown report table as fallback."""
    if not report_path.exists():
        return None

    text = report_path.read_text()
    lines = text.splitlines()
    in_section = False
    in_table = False
    metrics: dict[str, float] = {}

    for line in lines:
        if line.startswith("## ") and "Performance Metrics" in line:
            in_section = True
            in_table = False
            continue
        if in_section and line.startswith("## ") and "Performance Metrics" not in line:
            break
        if not in_section:
            continue
        if line.startswith("| Metric | Value |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            # skip separator
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) != 2:
                continue
            key, raw_val = parts
            val = raw_val.strip()
            try:
                if val.endswith("%"):
                    metrics[key] = float(val[:-1]) / 100.0
                else:
                    metrics[key] = float(val)
            except ValueError:
                continue

    return metrics or None


def _resolve_spec_path(strategy_id: str) -> Path:
    """Resolve strategy spec from either full or shorthand strategy ID.

    Accepted examples:
    - examples/demo_vcp
    - demo_vcp (resolved to strategies/examples/demo_vcp/strategy.yaml)
    """
    clean_id = strategy_id.strip("/")
    strategies_root = Path("strategies")
    primary_path = strategies_root / clean_id / "strategy.yaml"
    if primary_path.exists():
        return primary_path

    if "/" not in clean_id:
        example_path = strategies_root / "examples" / clean_id / "strategy.yaml"
        if example_path.exists():
            return example_path

    return primary_path


def _validate_live_execute_config(*, spec: StrategySpec, confirm_live_flag: bool) -> str | None:
    """Validate that live execution has both CLI and spec-level confirmations."""
    canary_enabled = True
    canary_max_positions = 1
    canary_max_risk_per_trade = 0.005

    if not confirm_live_flag:
        return "missing --confirm-live flag"

    exe = spec.execution
    if exe is None or not exe.enabled:
        return "strategy.execution.enabled=true is required"
    if exe.mode != "live":
        return "strategy.execution.mode must be 'live'"
    if not exe.confirm_live:
        return "strategy.execution.confirm_live must be true"

    live_canary_cfg = getattr(exe, "live_canary", None)
    if live_canary_cfg is not None:
        canary_enabled = bool(getattr(live_canary_cfg, "enabled", True))
        canary_max_positions = int(getattr(live_canary_cfg, "max_positions", 1))
        canary_max_risk_per_trade = float(getattr(live_canary_cfg, "max_risk_per_trade", 0.005))

    if canary_enabled:
        if spec.risk.max_positions > canary_max_positions:
            return f"live canary violation: risk.max_positions must be <= {canary_max_positions}"
        if spec.risk.risk_per_trade > canary_max_risk_per_trade:
            return (
                "live canary violation: risk.risk_per_trade must be <= "
                f"{canary_max_risk_per_trade:.4f}"
            )
    return None


def _main_legacy(argv: list[str]) -> int:
    """Backward-compatible CLI for backtest/fetch workflows."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Trade Strategy Pipeline — run backtests and evaluations",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy ID (e.g., demo_vcp or examples/demo_vcp)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running",
    )
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch data from APIs and save to data directory, then exit",
    )
    parser.add_argument(
        "--data-profile",
        choices=["auto", "sample", "production"],
        default="auto",
        help="OHLCV source selection: auto-detect, sample_ohlcv.parquet, or ohlcv.parquet",
    )
    parser.add_argument(
        "--spec-stage",
        choices=["phase1", "phase1_statistical", "phase2"],
        default=None,
        help="Optional stage override for runtime spec validation",
    )
    args = parser.parse_args(argv)

    spec_path = _resolve_spec_path(args.strategy)
    if not spec_path.exists():
        print(f"Error: Strategy spec not found: {spec_path}", file=sys.stderr)
        return 1

    data_dir = args.data_dir or Path("data")
    output_dir = args.output_dir or Path("reports") / args.strategy

    if args.dry_run:
        print(f"Strategy: {args.strategy}")
        print(f"Spec: {spec_path}")
        print(f"Data dir: {data_dir}")
        print(f"Data profile: {args.data_profile}")
        print(f"Spec stage: {args.spec_stage or 'auto'}")
        print(f"Output dir: {output_dir}")
        return 0

    from pipeline.runner.pipeline import Pipeline

    pipeline = Pipeline(
        spec_path=spec_path,
        data_dir=data_dir,
        output_dir=output_dir,
        data_profile=args.data_profile,
        spec_stage=args.spec_stage,
    )

    if args.fetch_data:
        try:
            ohlcv_path = pipeline.fetch_data()
            print(f"Data fetched and saved to {ohlcv_path}")
            return 0
        except Exception as e:
            print(f"Data fetch failed: {e}", file=sys.stderr)
            return 1

    result = pipeline.run()
    if result.success:
        print(f"Pipeline completed successfully for {args.strategy}")
        if result.gate_result:
            status = "PASSED" if result.gate_result.passed else "FAILED"
            print(f"Promotion gate: {status}")
        if result.report_path:
            print(f"Report: {result.report_path}")
        return 0

    print(f"Pipeline failed: {result.error}", file=sys.stderr)
    if result.failed_stage:
        print(f"Failed stage: {result.failed_stage}", file=sys.stderr)
    if result.failure_report_path:
        print(f"Failure report: {result.failure_report_path}", file=sys.stderr)
    return 1


def _main_execute(argv: list[str]) -> int:
    """Execution command for close/open cycle."""
    parser = argparse.ArgumentParser(
        prog="pipeline execute",
        description="Execution service (paper/live) close/open cycles",
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--phase", required=True, choices=["close", "open"])
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--session-date", default=datetime.now(tz=UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--state-dir", type=Path, default=Path("data") / "execution")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paper-metrics-json", type=Path, default=None)
    parser.add_argument("--backtest-metrics-json", type=Path, default=None)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required together with execution.confirm_live=true when --mode live",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh universe/ohlcv via fetch_data before close-cycle signal generation",
    )
    parser.add_argument("--data-profile", choices=["auto", "sample", "production"], default="auto")
    parser.add_argument(
        "--spec-stage",
        choices=["phase1", "phase1_statistical", "phase2"],
        default=None,
    )
    args = parser.parse_args(argv)

    spec_path = _resolve_spec_path(args.strategy)
    if not spec_path.exists():
        print(f"Error: Strategy spec not found: {spec_path}", file=sys.stderr)
        return 1

    strategy_state_dir = args.state_dir / args.strategy
    state_path = strategy_state_dir / "state.json"
    journal_path = strategy_state_dir / "journal" / f"{args.session_date}.parquet"
    base_output_dir = args.output_dir or Path("reports") / args.strategy
    exec_report_output_dir = base_output_dir / "execution" / args.session_date

    try:
        with open(spec_path) as f:
            spec_data: dict[str, Any] = yaml.safe_load(f) or {}
        spec = StrategySpec(**spec_data)

        if args.mode == "live":
            live_error = _validate_live_execute_config(
                spec=spec,
                confirm_live_flag=args.confirm_live,
            )
            if live_error is not None:
                print(f"Execution live-mode blocked: {live_error}", file=sys.stderr)
                return 1

        if args.phase == "close":
            from pipeline.execution.reporting import generate_execution_report
            from pipeline.execution.service import run_close_cycle
            from pipeline.runner.pipeline import Pipeline

            data_dir = args.data_dir or Path("data")
            pipeline = Pipeline(
                spec_path=spec_path,
                data_dir=data_dir,
                output_dir=base_output_dir,
                data_profile=args.data_profile,
                spec_stage=args.spec_stage,
            )
            if args.refresh_data:
                refreshed_path = pipeline.fetch_data()
                print(f"Data refreshed: {refreshed_path}")

            ohlcv = pipeline._load_data()
            if ohlcv is None or ohlcv.empty:
                print("Execution close cycle failed: No OHLCV data found", file=sys.stderr)
                return 1

            signals = pipeline._generate_signals(ohlcv)
            exe = pipeline.spec.execution
            allow_new_entries = exe.guardrails.allow_new_entries if exe else True
            kill_switch = exe.guardrails.emergency_kill_switch if exe else False

            close_result = run_close_cycle(
                strategy_id=args.strategy,
                session_date=args.session_date,
                signals=signals,
                state_path=state_path,
                journal_path=journal_path,
                max_positions=pipeline.spec.risk.max_positions,
                max_sector_exposure=pipeline.spec.risk.max_sector_exposure,
                allow_new_entries=allow_new_entries,
                kill_switch=kill_switch,
                dry_run=args.dry_run,
            )
            print(
                "Close cycle completed: "
                f"accepted={close_result.accepted_count}, "
                f"rejected={close_result.rejected_count}, "
                f"dry_run={close_result.dry_run}"
            )
            try:
                report_result = generate_execution_report(
                    strategy_id=args.strategy,
                    session_date=args.session_date,
                    journal_path=journal_path,
                    output_dir=exec_report_output_dir,
                )
                print(f"Execution report: {report_result.report_path}")
            except Exception as report_err:
                print(
                    f"Warning: execution report generation failed: {report_err}",
                    file=sys.stderr,
                )
            return 0

        # phase=open
        from pipeline.execution.broker.alpaca import AlpacaBroker
        from pipeline.execution.reporting import generate_execution_report
        from pipeline.execution.service import run_open_cycle

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            print(
                "Execution open cycle requires ALPACA_API_KEY and ALPACA_SECRET_KEY",
                file=sys.stderr,
            )
            return 1

        broker = AlpacaBroker(
            api_key=api_key,
            secret_key=secret_key,
            mode=args.mode,
            strategy_id=args.strategy,
        )
        open_result = run_open_cycle(
            strategy_id=args.strategy,
            state_path=state_path,
            journal_path=journal_path,
            broker=broker,
            max_daily_loss_pct=(
                spec.execution.guardrails.max_daily_loss_pct if spec.execution is not None else None
            ),
            dry_run=args.dry_run,
        )
        paper_gates = _build_default_paper_gates(spec.execution) if args.mode == "paper" else None
        paper_metrics = _load_metrics_json(args.paper_metrics_json)
        backtest_metrics = _load_metrics_json(args.backtest_metrics_json)
        if backtest_metrics is None:
            backtest_metrics = _load_backtest_metrics_from_report(base_output_dir / "report.md")

        print(
            "Open cycle completed: "
            f"submitted={open_result.submitted_count}, "
            f"skipped_existing={open_result.skipped_existing_count}, "
            f"fills={open_result.fill_events_count}, "
            f"dry_run={open_result.dry_run}"
        )
        try:
            report_result = generate_execution_report(
                strategy_id=args.strategy,
                session_date=args.session_date,
                journal_path=journal_path,
                output_dir=exec_report_output_dir,
                paper_metrics=paper_metrics,
                backtest_metrics=backtest_metrics,
                paper_gates=paper_gates,
            )
            print(f"Execution report: {report_result.report_path}")
            if report_result.gate_result is not None:
                gate_status = "PASSED" if report_result.gate_result.passed else "FAILED"
                print(f"Paper gate: {gate_status}")
        except Exception as report_err:
            print(
                f"Warning: execution report generation failed: {report_err}",
                file=sys.stderr,
            )
        return 0
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        return 1


def _main_reconcile(argv: list[str]) -> int:
    """Reconcile helper (open-cycle dry-run, no new submissions)."""
    parser = argparse.ArgumentParser(
        prog="pipeline reconcile",
        description="Reconcile state with broker without submitting new orders",
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--session-date", default=datetime.now(tz=UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--state-dir", type=Path, default=Path("data") / "execution")
    args = parser.parse_args(argv)

    spec_path = _resolve_spec_path(args.strategy)
    if not spec_path.exists():
        print(f"Error: Strategy spec not found: {spec_path}", file=sys.stderr)
        return 1

    from pipeline.execution.broker.alpaca import AlpacaBroker
    from pipeline.execution.service import run_open_cycle

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("Reconcile requires ALPACA_API_KEY and ALPACA_SECRET_KEY", file=sys.stderr)
        return 1

    strategy_state_dir = args.state_dir / args.strategy
    state_path = strategy_state_dir / "state.json"
    journal_path = strategy_state_dir / "journal" / f"{args.session_date}.parquet"

    try:
        broker = AlpacaBroker(
            api_key=api_key,
            secret_key=secret_key,
            mode=args.mode,
            strategy_id=args.strategy,
        )
        result = run_open_cycle(
            strategy_id=args.strategy,
            state_path=state_path,
            journal_path=journal_path,
            broker=broker,
            dry_run=True,
        )
        print(
            "Reconcile completed: "
            f"active_orders={result.active_orders_count}, fills_detected={result.fill_events_count}"
        )
        return 0
    except Exception as e:
        print(f"Reconcile failed: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Run CLI in legacy/backtest mode or execution mode."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "execute":
        return _main_execute(args[1:])
    if args and args[0] == "reconcile":
        return _main_reconcile(args[1:])
    return _main_legacy(args)


if __name__ == "__main__":
    sys.exit(main())
