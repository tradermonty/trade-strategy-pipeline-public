"""Tests for promotion gate evaluation."""

import pytest

from pipeline.evaluation.bootstrap import BootstrapResult
from pipeline.evaluation.gates import GateResult, evaluate_gates, evaluate_oos_gates
from pipeline.evaluation.sensitivity import run_sensitivity
from pipeline.evaluation.walk_forward import WalkForwardResult
from pipeline.spec.schema import OOSPromotionGates


def test_gate_pass_all_criteria() -> None:
    """All gates should pass with good metrics."""
    metrics = {
        "sharpe": 1.5,
        "profit_factor": 1.8,
        "max_drawdown": 0.10,
        "trade_count": 250,
        "win_rate": 0.55,
    }
    gates = {
        "min_trades": 200,
        "max_drawdown": 0.15,
        "sharpe": 1.0,
        "profit_factor": 1.2,
    }
    result = evaluate_gates(metrics, gates)
    assert result.passed is True
    assert len(result.failures) == 0


def test_gate_fail_insufficient_trades() -> None:
    """150 trades < 200 minimum → fail."""
    metrics = {
        "sharpe": 1.5,
        "profit_factor": 1.8,
        "max_drawdown": 0.10,
        "trade_count": 150,
        "win_rate": 0.55,
    }
    gates = {"min_trades": 200, "max_drawdown": 0.15, "sharpe": 1.0, "profit_factor": 1.2}
    result = evaluate_gates(metrics, gates)
    assert result.passed is False
    assert any("trade" in f.lower() for f in result.failures)


def test_gate_fail_excessive_drawdown() -> None:
    """18% drawdown > 15% limit → fail."""
    metrics = {
        "sharpe": 1.5,
        "profit_factor": 1.8,
        "max_drawdown": 0.18,
        "trade_count": 250,
        "win_rate": 0.55,
    }
    gates = {"min_trades": 200, "max_drawdown": 0.15, "sharpe": 1.0, "profit_factor": 1.2}
    result = evaluate_gates(metrics, gates)
    assert result.passed is False
    assert any("drawdown" in f.lower() for f in result.failures)


def test_gate_fail_low_sharpe() -> None:
    """Sharpe 0.8 < 1.0 minimum → fail."""
    metrics = {
        "sharpe": 0.8,
        "profit_factor": 1.8,
        "max_drawdown": 0.10,
        "trade_count": 250,
        "win_rate": 0.55,
    }
    gates = {"min_trades": 200, "max_drawdown": 0.15, "sharpe": 1.0, "profit_factor": 1.2}
    result = evaluate_gates(metrics, gates)
    assert result.passed is False
    assert any("sharpe" in f.lower() for f in result.failures)


def test_gate_fail_low_profit_factor() -> None:
    """Profit factor 1.05 < 1.2 minimum → fail."""
    metrics = {
        "sharpe": 1.5,
        "profit_factor": 1.05,
        "max_drawdown": 0.10,
        "trade_count": 250,
        "win_rate": 0.55,
    }
    gates = {"min_trades": 200, "max_drawdown": 0.15, "sharpe": 1.0, "profit_factor": 1.2}
    result = evaluate_gates(metrics, gates)
    assert result.passed is False
    assert any("profit factor" in f.lower() for f in result.failures)


def test_gate_returns_detailed_report() -> None:
    """GateResult should contain detailed check results."""
    metrics = {
        "sharpe": 1.5,
        "profit_factor": 1.8,
        "max_drawdown": 0.10,
        "trade_count": 250,
        "win_rate": 0.55,
    }
    gates = {"min_trades": 200, "max_drawdown": 0.15, "sharpe": 1.0, "profit_factor": 1.2}
    result = evaluate_gates(metrics, gates)
    assert isinstance(result, GateResult)
    assert isinstance(result.checks, list)
    assert len(result.checks) > 0
    for check in result.checks:
        assert "name" in check
        assert "passed" in check
        assert "actual" in check
        assert "threshold" in check


def test_gate_multiple_failures() -> None:
    """Multiple failing criteria should all be reported."""
    metrics = {
        "sharpe": 0.5,
        "profit_factor": 0.9,
        "max_drawdown": 0.20,
        "trade_count": 100,
        "win_rate": 0.40,
    }
    gates = {"min_trades": 200, "max_drawdown": 0.15, "sharpe": 1.0, "profit_factor": 1.2}
    result = evaluate_gates(metrics, gates)
    assert result.passed is False
    assert len(result.failures) == 4  # All 4 gates fail


def test_sensitivity_placeholder() -> None:
    """Sensitivity analysis should raise NotImplementedError (Phase II stub)."""
    with pytest.raises(NotImplementedError, match="Phase II"):
        run_sensitivity({}, {})


# --- OOS Gate evaluation tests ---


def _make_wf_result(oos_sharpe: float = 1.25, oos_pf: float = 1.64) -> WalkForwardResult:
    """Create a minimal WalkForwardResult for testing."""
    return WalkForwardResult(
        splits=[],
        oos_aggregate_pf=oos_pf,
        oos_aggregate_sharpe=oos_sharpe,
        degradation_ratio_pf=0.8,
        degradation_ratio_sharpe=0.7,
        passed=True,
        all_oos_trades=[],
    )


def _make_bs_result(pf_ci_lower: float = 1.12, sharpe_ci_lower: float = 0.31) -> BootstrapResult:
    """Create a minimal BootstrapResult for testing."""
    return BootstrapResult(
        pf_mean=1.50,
        pf_ci_lower=pf_ci_lower,
        pf_ci_upper=2.00,
        sharpe_mean=0.80,
        sharpe_ci_lower=sharpe_ci_lower,
        sharpe_ci_upper=1.30,
        n_iterations=10_000,
        confidence_level=0.95,
        go_decision=True,
    )


def test_evaluate_oos_gates_pass() -> None:
    """All OOS gates should pass with good metrics."""
    wf = _make_wf_result(oos_sharpe=1.25, oos_pf=1.64)
    bs = _make_bs_result(pf_ci_lower=1.12, sharpe_ci_lower=0.31)
    gates = OOSPromotionGates(
        min_sharpe=1.0, min_profit_factor=1.2, min_pf_ci_lower=1.0, min_sharpe_ci_lower=0.0
    )
    result = evaluate_oos_gates(wf, bs, gates)
    assert result.passed is True
    assert len(result.failures) == 0
    assert len(result.checks) == 4


def test_evaluate_oos_gates_fail_sharpe() -> None:
    """OOS Sharpe below threshold should fail."""
    wf = _make_wf_result(oos_sharpe=0.5, oos_pf=1.64)
    bs = _make_bs_result()
    gates = OOSPromotionGates(min_sharpe=1.0)
    result = evaluate_oos_gates(wf, bs, gates)
    assert result.passed is False
    assert any("OOS Sharpe" in f for f in result.failures)


def test_evaluate_oos_gates_fail_pf() -> None:
    """OOS PF below threshold should fail."""
    wf = _make_wf_result(oos_sharpe=1.5, oos_pf=0.8)
    bs = _make_bs_result()
    gates = OOSPromotionGates(min_profit_factor=1.2)
    result = evaluate_oos_gates(wf, bs, gates)
    assert result.passed is False
    assert any("OOS Profit Factor" in f for f in result.failures)


def test_evaluate_oos_gates_fail_ci_lower() -> None:
    """Bootstrap PF CI lower below threshold should fail."""
    wf = _make_wf_result()
    bs = _make_bs_result(pf_ci_lower=0.80, sharpe_ci_lower=-0.1)
    gates = OOSPromotionGates(min_pf_ci_lower=1.0, min_sharpe_ci_lower=0.0)
    result = evaluate_oos_gates(wf, bs, gates)
    assert result.passed is False
    assert any("PF CI lower" in f for f in result.failures)
    assert any("Sharpe CI lower" in f for f in result.failures)
