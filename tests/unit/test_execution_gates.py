"""Tests for execution-specific paper gate evaluation."""

from pipeline.evaluation.gates import evaluate_execution_paper_gate


def test_execution_paper_gate_pass() -> None:
    """All configured execution paper gates should pass with strong metrics."""
    metrics = {
        "filled_order_count": 60,
        "reject_cancel_ratio": 0.04,
        "partial_fill_ratio": 0.20,
        "realized_slippage_bps": 12.0,
        "paper_backtest_deviation": 0.12,
        "profit_factor": 1.25,
        "max_drawdown": 0.11,
    }
    gates = {
        "min_paper_trades": 50,
        "max_reject_cancel_ratio": 0.20,
        "max_partial_fill_ratio": 0.60,
        "max_realized_slippage_bps": 20.0,
        "max_backtest_deviation": 0.20,
        "min_profit_factor": 1.0,
        "max_drawdown": 0.20,
    }

    result = evaluate_execution_paper_gate(metrics, gates)
    assert result.passed is True
    assert len(result.failures) == 0
    assert len(result.checks) == 7


def test_execution_paper_gate_multiple_failures() -> None:
    """Multiple failed execution gates should all be reported."""
    metrics = {
        "filled_order_count": 40,
        "reject_cancel_ratio": 0.35,
        "partial_fill_ratio": 0.75,
        "realized_slippage_bps": 28.0,
        "paper_backtest_deviation": 0.35,
        "profit_factor": 0.95,
        "max_drawdown": 0.24,
    }
    gates = {
        "min_paper_trades": 50,
        "max_reject_cancel_ratio": 0.20,
        "max_partial_fill_ratio": 0.60,
        "max_realized_slippage_bps": 20.0,
        "max_backtest_deviation": 0.20,
        "min_profit_factor": 1.0,
        "max_drawdown": 0.20,
    }

    result = evaluate_execution_paper_gate(metrics, gates)
    assert result.passed is False
    assert len(result.failures) == 7
    assert any("min_paper_trades" in msg for msg in result.failures)
    assert any("max_reject_cancel_ratio" in msg for msg in result.failures)


def test_execution_paper_gate_missing_metric_fails() -> None:
    """Required non-optional metric missing should fail clearly."""
    metrics = {
        "reject_cancel_ratio": 0.08,
    }
    gates = {"min_paper_trades": 50.0}

    result = evaluate_execution_paper_gate(metrics, gates)
    assert result.passed is False
    assert len(result.failures) == 1
    assert "missing" in result.failures[0].lower()


def test_execution_paper_gate_missing_slippage_is_skipped() -> None:
    """realized_slippage_bps may be missing before first fills and should not fail."""
    metrics = {"filled_order_count": 55, "reject_cancel_ratio": 0.08}
    gates = {"max_realized_slippage_bps": 20.0}

    result = evaluate_execution_paper_gate(metrics, gates)
    assert result.passed is True
    assert len(result.failures) == 0


def test_execution_paper_gate_missing_backtest_deviation_is_skipped() -> None:
    """paper_backtest_deviation may be missing when metrics inputs are not provided."""
    metrics = {"filled_order_count": 55, "reject_cancel_ratio": 0.08}
    gates = {"max_backtest_deviation": 0.20}

    result = evaluate_execution_paper_gate(metrics, gates)
    assert result.passed is True
    assert len(result.failures) == 0
