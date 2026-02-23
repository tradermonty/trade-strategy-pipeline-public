"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.evaluation.gates import GateResult


def generate_report(data: dict[str, Any], output_path: Path) -> None:
    """Generate a Phase I backtest report in Markdown format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategy_id = data.get("strategy_id", "unknown")
    strategy_name = data.get("strategy_name", "Unknown Strategy")
    metrics = data.get("metrics", {})
    gate_result: GateResult = data.get("gate_result", GateResult(passed=False))
    diagnostic_only = bool(data.get("diagnostic_only", False))
    equity_chart = data.get("equity_chart", "")
    drawdown_chart = data.get("drawdown_chart", "")
    trade_dist_chart = data.get("trade_distribution_chart", "")

    lines: list[str] = []

    # Header
    lines.append(f"# Phase I Backtest Report: {strategy_name}")
    lines.append("")
    lines.append(f"**Strategy ID**: `{strategy_id}`")
    lines.append("")
    if diagnostic_only:
        lines.append(
            "> Note: `backtest.portfolio_mode=false`. Metrics/gates are diagnostic only "
            "and not eligible for promotion."
        )
        lines.append("")

    # Gate Result Summary
    gate_status = "PASS" if gate_result.passed else "FAIL"
    lines.append(f"## Promotion Gate: **{gate_status}**")
    lines.append("")

    if gate_result.checks:
        lines.append("| Gate | Actual | Threshold | Result |")
        lines.append("|------|--------|-----------|--------|")
        for check in gate_result.checks:
            status = "PASS" if check["passed"] else "FAIL"
            actual = check["actual"]
            threshold = check["threshold"]
            name = check["name"]
            # max_drawdown and win_rate are ratios displayed as percentages
            if name in ("max_drawdown", "win_rate"):
                actual_str = f"{actual:.1%}"
                threshold_str = f"{threshold:.1%}"
            elif isinstance(actual, float):
                actual_str = f"{actual:.2f}"
                threshold_str = f"{threshold:.2f}"
            else:
                actual_str = f"{actual}"
                threshold_str = f"{threshold}"
            lines.append(f"| {name} | {actual_str} | {threshold_str} | {status} |")
        lines.append("")

    if gate_result.failures:
        lines.append("### Failures")
        for f in gate_result.failures:
            lines.append(f"- {f}")
        lines.append("")

    # Metrics Table
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ("max_drawdown", "win_rate"):
                lines.append(f"| {key} | {value:.1%} |")
            else:
                lines.append(f"| {key} | {value:.2f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")

    # Charts
    lines.append("## Charts")
    lines.append("")
    if equity_chart:
        lines.append("### Equity Curve")
        lines.append(f"![Equity Curve]({equity_chart})")
        lines.append("")
    if drawdown_chart:
        lines.append("### Drawdown")
        lines.append(f"![Drawdown]({drawdown_chart})")
        lines.append("")
    if trade_dist_chart:
        lines.append("### Trade Distribution")
        lines.append(f"![Trade Distribution]({trade_dist_chart})")
        lines.append("")

    # Bootstrap CI section (optional)
    bootstrap_result = data.get("bootstrap_result")
    if bootstrap_result is not None:
        cl_pct = int(bootstrap_result.confidence_level * 100)
        lines.append(
            f"## Bootstrap Confidence Intervals ({cl_pct}%, N={bootstrap_result.n_iterations:,})"
        )
        lines.append("")
        lines.append("| Metric | Mean | CI Lower | CI Upper |")
        lines.append("|--------|------|----------|----------|")
        lines.append(
            f"| Profit Factor | {bootstrap_result.pf_mean:.2f} "
            f"| {bootstrap_result.pf_ci_lower:.2f} "
            f"| {bootstrap_result.pf_ci_upper:.2f} |"
        )
        lines.append(
            f"| Sharpe (trade-level) | {bootstrap_result.sharpe_mean:.2f} "
            f"| {bootstrap_result.sharpe_ci_lower:.2f} "
            f"| {bootstrap_result.sharpe_ci_upper:.2f} |"
        )
        lines.append("")
        go_str = "PASS" if bootstrap_result.go_decision else "FAIL"
        lines.append(f"**Go/No-Go**: {go_str}")
        lines.append("")
        lines.append(
            "> Note: Trade-level Sharpe = mean(PnL)/std(PnL)*sqrt(N). "
            "Gate Sharpe uses daily equity returns — different definition."
        )
        lines.append("")

    # OOS Bootstrap CI section (optional — from walk-forward pooled trades)
    oos_bootstrap = data.get("oos_bootstrap_result")
    if oos_bootstrap is not None:
        cl_pct = int(oos_bootstrap.confidence_level * 100)
        lines.append(
            f"## OOS Bootstrap Confidence Intervals ({cl_pct}%, N={oos_bootstrap.n_iterations:,})"
        )
        lines.append("")
        lines.append("| Metric | Mean | CI Lower | CI Upper |")
        lines.append("|--------|------|----------|----------|")
        lines.append(
            f"| Profit Factor | {oos_bootstrap.pf_mean:.2f} "
            f"| {oos_bootstrap.pf_ci_lower:.2f} "
            f"| {oos_bootstrap.pf_ci_upper:.2f} |"
        )
        lines.append(
            f"| Sharpe (trade-level) | {oos_bootstrap.sharpe_mean:.2f} "
            f"| {oos_bootstrap.sharpe_ci_lower:.2f} "
            f"| {oos_bootstrap.sharpe_ci_upper:.2f} |"
        )
        lines.append("")

    # OOS Gate Result section (optional)
    oos_gate: GateResult | None = data.get("oos_gate_result")
    if oos_gate is not None:
        oos_gate_status = "PASS" if oos_gate.passed else "FAIL"
        lines.append(f"## OOS Promotion Gate: **{oos_gate_status}**")
        lines.append("")
        if oos_gate.checks:
            lines.append("| Gate | Actual | Threshold | Result |")
            lines.append("|------|--------|-----------|--------|")
            for check in oos_gate.checks:
                status = "PASS" if check["passed"] else "FAIL"
                actual_val = check["actual"]
                threshold_val = check["threshold"]
                if isinstance(actual_val, float):
                    actual_str = f"{actual_val:.2f}"
                else:
                    actual_str = f"{actual_val}"
                if isinstance(threshold_val, float):
                    threshold_str = f"{threshold_val:.2f}"
                else:
                    threshold_str = f"{threshold_val}"
                lines.append(f"| {check['name']} | {actual_str} | {threshold_str} | {status} |")
            lines.append("")
        if oos_gate.failures:
            for f in oos_gate.failures:
                lines.append(f"- {f}")
            lines.append("")

    # Walk-Forward section (optional)
    wf_result = data.get("walk_forward_result")
    if wf_result is not None:
        n_splits = len(wf_result.splits)
        oos_trades_total = sum(s.oos_trade_count for s in wf_result.splits)
        lines.append(f"## Walk-Forward Analysis ({n_splits} splits, {oos_trades_total} OOS trades)")
        lines.append("")
        lines.append(
            "| Split | IS Period | OOS Period | IS PF | OOS PF "
            "| IS Sharpe | OOS Sharpe | OOS Trades |"
        )
        lines.append(
            "|-------|-----------|------------|-------|--------"
            "|-----------|------------|------------|"
        )
        for i, split in enumerate(wf_result.splits):
            is_pf = split.is_metrics.get("profit_factor", 0)
            oos_pf = split.oos_metrics.get("profit_factor", 0)
            is_sh = split.is_metrics.get("sharpe", 0)
            oos_sh = split.oos_metrics.get("sharpe", 0)
            lines.append(
                f"| {i + 1} | {split.train_start}-{split.train_end} "
                f"| {split.test_start}-{split.test_end} "
                f"| {is_pf:.2f} | {oos_pf:.2f} "
                f"| {is_sh:.2f} | {oos_sh:.2f} "
                f"| {split.oos_trade_count} |"
            )
        lines.append("")
        lines.append(
            f"**Degradation**: PF {wf_result.degradation_ratio_pf:.2f}x, "
            f"Sharpe {wf_result.degradation_ratio_sharpe:.2f}x"
        )
        lines.append("")

    # Disclaimer
    lines.append("---")
    lines.append("")
    lines.append("## Disclaimers")
    lines.append("")
    lines.append(
        "**Survivorship Bias**: This Phase I backtest uses current S&P 500 "
        "constituents only. Results exclude delisted or removed companies and "
        "may overstate historical performance. Point-in-time constituent data "
        "will be introduced in Phase II to address this limitation."
    )
    lines.append("")

    output_path.write_text("\n".join(lines))
