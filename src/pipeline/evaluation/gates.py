"""Promotion gate evaluation for Phase I and OOS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.evaluation.bootstrap import BootstrapResult
    from pipeline.evaluation.walk_forward import WalkForwardResult
    from pipeline.spec.schema import OOSPromotionGates


@dataclass
class GateResult:
    """Result of gate evaluation."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def evaluate_gates(metrics: dict[str, float], gates: dict[str, float]) -> GateResult:
    """Evaluate metrics against promotion gate criteria.

    Returns GateResult with pass/fail status and detailed check results.
    """
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    # Min trades
    if "min_trades" in gates:
        actual = metrics.get("trade_count", 0)
        threshold = gates["min_trades"]
        passed = actual >= threshold
        checks.append(
            {
                "name": "min_trades",
                "passed": passed,
                "actual": actual,
                "threshold": threshold,
            }
        )
        if not passed:
            failures.append(f"Trade count {actual} < {threshold} minimum trades required")

    # Max drawdown
    if "max_drawdown" in gates:
        actual = metrics.get("max_drawdown", 1.0)
        threshold = gates["max_drawdown"]
        passed = actual <= threshold
        checks.append(
            {
                "name": "max_drawdown",
                "passed": passed,
                "actual": actual,
                "threshold": threshold,
            }
        )
        if not passed:
            failures.append(f"Max drawdown {actual:.1%} exceeds {threshold:.1%} limit")

    # Sharpe ratio (skip when None — some strategies use PF-only gates)
    if "sharpe" in gates and gates["sharpe"] is not None:
        actual = metrics.get("sharpe", 0.0)
        threshold = gates["sharpe"]
        passed = actual >= threshold
        checks.append(
            {
                "name": "sharpe",
                "passed": passed,
                "actual": actual,
                "threshold": threshold,
            }
        )
        if not passed:
            failures.append(f"Sharpe ratio {actual:.2f} < {threshold:.2f} minimum")

    # Profit factor
    if "profit_factor" in gates:
        actual = metrics.get("profit_factor", 0.0)
        threshold = gates["profit_factor"]
        passed = actual >= threshold
        checks.append(
            {
                "name": "profit_factor",
                "passed": passed,
                "actual": actual,
                "threshold": threshold,
            }
        )
        if not passed:
            failures.append(f"Profit factor {actual:.2f} < {threshold:.2f} minimum")

    return GateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
    )


def evaluate_oos_gates(
    wf_result: WalkForwardResult,
    bs_result: BootstrapResult,
    gates: OOSPromotionGates,
) -> GateResult:
    """Evaluate OOS metrics + Bootstrap CI against promotion criteria.

    Checks:
    1. OOS aggregate Sharpe >= min_sharpe
    2. OOS aggregate PF >= min_profit_factor
    3. Bootstrap PF CI lower >= min_pf_ci_lower
    4. Bootstrap Sharpe CI lower >= min_sharpe_ci_lower
    """
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    # 1. OOS Sharpe
    actual = wf_result.oos_aggregate_sharpe
    threshold = gates.min_sharpe
    passed = actual >= threshold
    checks.append(
        {"name": "oos_sharpe", "passed": passed, "actual": actual, "threshold": threshold}
    )
    if not passed:
        failures.append(f"OOS Sharpe {actual:.2f} < {threshold:.2f} minimum")

    # 2. OOS PF
    actual = wf_result.oos_aggregate_pf
    threshold = gates.min_profit_factor
    passed = actual >= threshold
    checks.append(
        {"name": "oos_profit_factor", "passed": passed, "actual": actual, "threshold": threshold}
    )
    if not passed:
        failures.append(f"OOS Profit Factor {actual:.2f} < {threshold:.2f} minimum")

    # 3. PF CI lower
    actual = bs_result.pf_ci_lower
    threshold = gates.min_pf_ci_lower
    passed = actual >= threshold
    checks.append(
        {"name": "pf_ci_lower", "passed": passed, "actual": actual, "threshold": threshold}
    )
    if not passed:
        failures.append(f"PF CI lower {actual:.2f} < {threshold:.2f} minimum")

    # 4. Sharpe CI lower
    actual = bs_result.sharpe_ci_lower
    threshold = gates.min_sharpe_ci_lower
    passed = actual >= threshold
    checks.append(
        {"name": "sharpe_ci_lower", "passed": passed, "actual": actual, "threshold": threshold}
    )
    if not passed:
        failures.append(f"Sharpe CI lower {actual:.2f} < {threshold:.2f} minimum")

    return GateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
    )


def evaluate_execution_paper_gate(
    metrics: dict[str, float | int | None],
    gates: dict[str, float],
) -> GateResult:
    """Evaluate execution-specific paper promotion gates.

    Supported gate keys:
    - min_paper_trades
    - max_reject_cancel_ratio
    - max_partial_fill_ratio
    - max_realized_slippage_bps
    - max_backtest_deviation
    - min_profit_factor
    - max_drawdown
    """
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def _check_min(name: str, metric_key: str, threshold: float) -> None:
        actual = _to_float_or_none(metrics.get(metric_key))
        passed = actual is not None and actual >= threshold
        checks.append({"name": name, "passed": passed, "actual": actual, "threshold": threshold})
        if not passed:
            if actual is None:
                failures.append(f"{name}: metric '{metric_key}' is missing")
            else:
                failures.append(f"{name}: {actual:.4f} < {threshold:.4f}")

    def _check_max(
        name: str, metric_key: str, threshold: float, *, allow_missing: bool = False
    ) -> None:
        actual = _to_float_or_none(metrics.get(metric_key))
        passed = allow_missing if actual is None else actual <= threshold
        checks.append({"name": name, "passed": passed, "actual": actual, "threshold": threshold})
        if not passed:
            if actual is None:
                failures.append(f"{name}: metric '{metric_key}' is missing")
            else:
                failures.append(f"{name}: {actual:.4f} > {threshold:.4f}")

    if "min_paper_trades" in gates:
        _check_min("min_paper_trades", "filled_order_count", gates["min_paper_trades"])
    if "max_reject_cancel_ratio" in gates:
        _check_max(
            "max_reject_cancel_ratio",
            "reject_cancel_ratio",
            gates["max_reject_cancel_ratio"],
        )
    if "max_partial_fill_ratio" in gates:
        _check_max(
            "max_partial_fill_ratio",
            "partial_fill_ratio",
            gates["max_partial_fill_ratio"],
        )
    if "max_realized_slippage_bps" in gates:
        _check_max(
            "max_realized_slippage_bps",
            "realized_slippage_bps",
            gates["max_realized_slippage_bps"],
            allow_missing=True,
        )
    if "max_backtest_deviation" in gates:
        _check_max(
            "max_backtest_deviation",
            "paper_backtest_deviation",
            gates["max_backtest_deviation"],
            allow_missing=True,
        )
    # Optional parity with existing strategy promotion metrics if caller provides them.
    if "min_profit_factor" in gates:
        _check_min("min_profit_factor", "profit_factor", gates["min_profit_factor"])
    if "max_drawdown" in gates:
        _check_max("max_drawdown", "max_drawdown", gates["max_drawdown"])

    return GateResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
    )


def _to_float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if math.isnan(v):
        return None
    return v
