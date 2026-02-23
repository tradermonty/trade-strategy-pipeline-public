"""Execution reporting utilities (Paper/Live cycles)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.evaluation.gates import GateResult, evaluate_execution_paper_gate
from pipeline.execution.storage import read_execution_journal


@dataclass
class ExecutionReportResult:
    """Execution report generation result and artifact paths."""

    strategy_id: str
    session_date: str
    metrics: dict[str, float | int | None]
    gate_result: GateResult | None
    report_path: Path
    metrics_json_path: Path
    events_csv_path: Path


def compute_execution_metrics(
    journal: pd.DataFrame,
    *,
    paper_metrics: dict[str, float] | None = None,
    backtest_metrics: dict[str, float] | None = None,
) -> dict[str, float | int | None]:
    """Compute core execution metrics from journal events.

    Metrics include:
    - realized_slippage_bps
    - reject/cancel ratio
    - partial-fill ratio (event-based proxy: orders with 2+ fill events)
      Note: This is derived from polling-time fill deltas, not broker-native fill lots.
      It is useful as a stability proxy, not a precise market microstructure metric.
    - paper vs backtest deviation (when both inputs are provided)
    """
    if journal.empty:
        return {
            "intent_created_count": 0,
            "intent_rejected_count": 0,
            "order_submitted_count": 0,
            "order_submit_failed_count": 0,
            "order_canceled_count": 0,
            "fill_event_count": 0,
            "filled_order_count": 0,
            "reject_ratio": 0.0,
            "cancel_ratio": 0.0,
            "reject_cancel_ratio": 0.0,
            "partial_fill_ratio": 0.0,
            "realized_slippage_bps": None,
            "paper_backtest_deviation": None,
        }

    event_col = (
        journal["event_type"] if "event_type" in journal.columns else pd.Series([], dtype=str)
    )

    intent_created_count = int((event_col == "intent_created").sum())
    intent_rejected_count = int((event_col == "intent_rejected").sum())
    order_submitted_count = int((event_col == "order_submitted").sum())
    order_submit_failed_count = int((event_col == "order_submit_failed").sum())
    order_rejected_count = int((event_col == "order_rejected").sum())
    order_canceled_count = int(
        event_col.isin(["order_canceled", "order_cancelled", "order_cancel"]).sum()
    )

    fill_mask = event_col == "fill_detected"
    fill_events = journal[fill_mask].copy()
    fill_event_count = len(fill_events)

    filled_order_count = 0
    partial_fill_ratio = 0.0
    if fill_event_count > 0 and "broker_order_id" in fill_events.columns:
        by_order = fill_events.dropna(subset=["broker_order_id"]).groupby("broker_order_id").size()
        filled_order_count = len(by_order)
        partial_count = int((by_order > 1).sum())
        partial_fill_ratio = (partial_count / filled_order_count) if filled_order_count > 0 else 0.0

    total_order_outcomes = (
        order_submitted_count
        + order_submit_failed_count
        + order_rejected_count
        + order_canceled_count
    )
    reject_count = order_submit_failed_count + order_rejected_count
    reject_ratio = (reject_count / total_order_outcomes) if total_order_outcomes > 0 else 0.0
    cancel_ratio = (
        (order_canceled_count / total_order_outcomes) if total_order_outcomes > 0 else 0.0
    )
    reject_cancel_ratio = (
        (reject_count + order_canceled_count) / total_order_outcomes
        if total_order_outcomes > 0
        else 0.0
    )

    realized_slippage_bps = _compute_realized_slippage_bps(fill_events)
    paper_backtest_deviation = _compute_paper_backtest_deviation(
        paper_metrics=paper_metrics,
        backtest_metrics=backtest_metrics,
        filled_order_count=filled_order_count,
    )

    return {
        "intent_created_count": intent_created_count,
        "intent_rejected_count": intent_rejected_count,
        "order_submitted_count": order_submitted_count,
        "order_submit_failed_count": order_submit_failed_count,
        "order_canceled_count": order_canceled_count,
        "fill_event_count": fill_event_count,
        "filled_order_count": filled_order_count,
        "reject_ratio": reject_ratio,
        "cancel_ratio": cancel_ratio,
        "reject_cancel_ratio": reject_cancel_ratio,
        "partial_fill_ratio": partial_fill_ratio,
        "realized_slippage_bps": realized_slippage_bps,
        "paper_backtest_deviation": paper_backtest_deviation,
    }


def generate_execution_report(
    *,
    strategy_id: str,
    session_date: str,
    journal_path: Path,
    output_dir: Path,
    paper_metrics: dict[str, float] | None = None,
    backtest_metrics: dict[str, float] | None = None,
    paper_gates: dict[str, float] | None = None,
) -> ExecutionReportResult:
    """Generate execution artifacts (markdown + csv + json)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = read_execution_journal(journal_path)
    metrics = compute_execution_metrics(
        journal,
        paper_metrics=paper_metrics,
        backtest_metrics=backtest_metrics,
    )
    gate_result = (
        evaluate_execution_paper_gate(metrics, paper_gates) if paper_gates is not None else None
    )

    report_path = output_dir / "report.md"
    metrics_json_path = output_dir / "metrics.json"
    events_csv_path = output_dir / "events.csv"

    # Save raw events (audit artifact)
    events_out = journal
    if events_out.empty and len(events_out.columns) == 0:
        events_out = pd.DataFrame(
            columns=[
                "timestamp",
                "event_type",
                "strategy_id",
                "session_date",
                "intent_id",
                "symbol",
                "side",
                "client_order_id",
                "broker_order_id",
                "fill_qty",
                "fill_price",
                "error",
            ]
        )
    events_out.to_csv(events_csv_path, index=False)

    # Save metrics JSON
    metrics_payload: dict[str, Any] = {
        "strategy_id": strategy_id,
        "session_date": session_date,
        "metrics": metrics,
        "paper_metrics": paper_metrics,
        "backtest_metrics": backtest_metrics,
    }
    if gate_result is not None:
        metrics_payload["paper_gate"] = {
            "passed": gate_result.passed,
            "checks": gate_result.checks,
            "failures": gate_result.failures,
        }
    metrics_json_path.write_text(json.dumps(metrics_payload, ensure_ascii=True, indent=2) + "\n")

    # Save markdown summary
    lines: list[str] = []
    lines.append(f"# Execution Report: {strategy_id}")
    lines.append("")
    lines.append(f"**Session Date**: `{session_date}`")
    lines.append(f"**Journal Path**: `{journal_path}`")
    lines.append("")
    lines.append("## Execution Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in sorted(metrics.keys()):
        lines.append(f"| {key} | {_fmt_metric(metrics[key])} |")
    lines.append("")

    if gate_result is not None:
        status = "PASS" if gate_result.passed else "FAIL"
        lines.append(f"## Paper Gate: **{status}**")
        lines.append("")
        lines.append("| Gate | Actual | Threshold | Result |")
        lines.append("|---|---:|---:|---|")
        for check in gate_result.checks:
            c_status = "PASS" if check["passed"] else "FAIL"
            lines.append(
                "| {name} | {actual} | {threshold} | {status} |".format(
                    name=check["name"],
                    actual=_fmt_metric(_to_number_or_none(check["actual"])),
                    threshold=_fmt_metric(_to_number_or_none(check["threshold"])),
                    status=c_status,
                )
            )
        lines.append("")
        if gate_result.failures:
            lines.append("### Failures")
            for msg in gate_result.failures:
                lines.append(f"- {msg}")
            lines.append("")

    report_path.write_text("\n".join(lines))
    return ExecutionReportResult(
        strategy_id=strategy_id,
        session_date=session_date,
        metrics=metrics,
        gate_result=gate_result,
        report_path=report_path,
        metrics_json_path=metrics_json_path,
        events_csv_path=events_csv_path,
    )


def _compute_realized_slippage_bps(fill_events: pd.DataFrame) -> float | None:
    """Compute realized slippage in bps from fill events.

    Priority:
    1) explicit `slippage_bps` column (mean absolute value)
    2) inferred from expected_price vs fill_price when present
    """
    if fill_events.empty:
        return None

    if "slippage_bps" in fill_events.columns:
        vals = pd.to_numeric(fill_events["slippage_bps"], errors="coerce").dropna()
        if len(vals) > 0:
            return float(vals.abs().mean())

    required_cols = {"side", "fill_price", "expected_price"}
    if not required_cols.issubset(fill_events.columns):
        return None

    tmp = fill_events.copy()
    tmp["fill_price"] = pd.to_numeric(tmp["fill_price"], errors="coerce")
    tmp["expected_price"] = pd.to_numeric(tmp["expected_price"], errors="coerce")
    tmp = tmp.dropna(subset=["fill_price", "expected_price"])
    tmp = tmp[tmp["expected_price"] > 0]
    if tmp.empty:
        return None

    buy_mask = tmp["side"].astype(str).str.lower() == "buy"
    bps = pd.Series(index=tmp.index, dtype=float)
    bps.loc[buy_mask] = (
        (tmp.loc[buy_mask, "fill_price"] - tmp.loc[buy_mask, "expected_price"])
        / tmp.loc[buy_mask, "expected_price"]
    ) * 10_000.0
    bps.loc[~buy_mask] = (
        (tmp.loc[~buy_mask, "expected_price"] - tmp.loc[~buy_mask, "fill_price"])
        / tmp.loc[~buy_mask, "expected_price"]
    ) * 10_000.0
    bps = bps.dropna()
    return float(bps.abs().mean()) if len(bps) > 0 else None


def _compute_paper_backtest_deviation(
    *,
    paper_metrics: dict[str, float] | None,
    backtest_metrics: dict[str, float] | None,
    filled_order_count: int,
) -> float | None:
    """Compute aggregate deviation between paper metrics and backtest metrics.

    Returns mean absolute relative deviation across available comparable metrics.
    """
    if paper_metrics is None and backtest_metrics is None:
        return None

    paper = dict(paper_metrics or {})
    backtest = dict(backtest_metrics or {})

    # Fallback paper proxy when caller does not pass a paper trade count.
    if "trade_count" not in paper:
        paper["trade_count"] = float(filled_order_count)

    keys = ("profit_factor", "sharpe", "win_rate", "trade_count")
    deviations: list[float] = []
    for key in keys:
        p = _to_number_or_none(paper.get(key))
        b = _to_number_or_none(backtest.get(key))
        if p is None or b is None:
            continue
        if b == 0:
            # Keep deviation as relative metric only; skip zero baseline.
            continue
        deviations.append(abs(p - b) / abs(b))

    if not deviations:
        return None
    return float(sum(deviations) / len(deviations))


def _fmt_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.4f}"


def _to_number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v
