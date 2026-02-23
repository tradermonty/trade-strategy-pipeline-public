"""Tests for Markdown report generation."""

from pathlib import Path

from pipeline.evaluation.bootstrap import BootstrapResult
from pipeline.evaluation.gates import GateResult
from pipeline.reporting.markdown import generate_report


def _make_report_data() -> dict[str, object]:
    """Create sample data for report generation."""
    return {
        "strategy_id": "demo_vcp",
        "strategy_name": "VCP Pivot Breakout",
        "metrics": {
            "sharpe": 1.5,
            "profit_factor": 1.8,
            "max_drawdown": 0.10,
            "trade_count": 250,
            "win_rate": 0.55,
        },
        "gate_result": GateResult(
            passed=True,
            checks=[
                {"name": "min_trades", "passed": True, "actual": 250, "threshold": 200},
                {"name": "max_drawdown", "passed": True, "actual": 0.10, "threshold": 0.15},
                {"name": "sharpe", "passed": True, "actual": 1.5, "threshold": 1.0},
                {"name": "profit_factor", "passed": True, "actual": 1.8, "threshold": 1.2},
            ],
            failures=[],
        ),
        "equity_chart": "charts/equity.png",
        "drawdown_chart": "charts/drawdown.png",
        "trade_distribution_chart": "charts/trade_dist.png",
    }


def test_markdown_report_contains_metrics_table(tmp_path: Path) -> None:
    """Report should contain a metrics table."""
    data = _make_report_data()
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "sharpe" in content.lower() or "Sharpe" in content
    assert "1.5" in content or "1.50" in content


def test_markdown_report_contains_gate_results(tmp_path: Path) -> None:
    """Report should contain gate pass/fail results."""
    data = _make_report_data()
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "PASS" in content or "pass" in content.lower()


def test_markdown_report_shows_diagnostic_note_when_non_portfolio(tmp_path: Path) -> None:
    """diagnostic_only=True should add a non-portfolio note near the header."""
    data = _make_report_data()
    data["diagnostic_only"] = True
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "diagnostic only" in content.lower()
    assert "portfolio_mode=false" in content


def test_markdown_report_contains_chart_references(tmp_path: Path) -> None:
    """Report should reference chart image files."""
    data = _make_report_data()
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "equity" in content.lower()
    assert ".png" in content


def test_markdown_report_contains_survivorship_disclaimer(tmp_path: Path) -> None:
    """Report must contain survivorship bias disclaimer (Section 6.1.1)."""
    data = _make_report_data()
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "survivorship" in content.lower()


def test_report_output_directory_structure(tmp_path: Path) -> None:
    """Report should be written to the specified output path."""
    data = _make_report_data()
    output_dir = tmp_path / "reports" / "vcp" / "2026-02-20"
    output = output_dir / "report.md"
    generate_report(data, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_markdown_report_valid_syntax(tmp_path: Path) -> None:
    """Report should have valid Markdown structure (headers, tables)."""
    data = _make_report_data()
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    # Should have at least one H1 or H2 header
    assert content.startswith("#") or "\n#" in content
    # Should have table separator
    assert "|" in content


def test_report_with_bootstrap_section(tmp_path: Path) -> None:
    """Report with bootstrap_result should include CI table."""
    data = _make_report_data()
    data["bootstrap_result"] = BootstrapResult(
        pf_mean=1.29,
        pf_ci_lower=1.08,
        pf_ci_upper=1.52,
        sharpe_mean=0.15,
        sharpe_ci_lower=0.02,
        sharpe_ci_upper=0.28,
        n_iterations=10_000,
        confidence_level=0.95,
        go_decision=True,
    )
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "Bootstrap Confidence Intervals" in content
    assert "1.08" in content
    assert "Go/No-Go" in content
    assert "PASS" in content


def test_report_without_bootstrap(tmp_path: Path) -> None:
    """Report without bootstrap_result should not have Bootstrap section."""
    data = _make_report_data()
    data["bootstrap_result"] = None
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "Bootstrap" not in content


def test_report_with_oos_bootstrap(tmp_path: Path) -> None:
    """Report with oos_bootstrap_result + oos_gate_result should include OOS sections."""
    data = _make_report_data()
    data["oos_bootstrap_result"] = BootstrapResult(
        pf_mean=1.64,
        pf_ci_lower=1.12,
        pf_ci_upper=2.10,
        sharpe_mean=0.80,
        sharpe_ci_lower=0.31,
        sharpe_ci_upper=1.40,
        n_iterations=10_000,
        confidence_level=0.95,
        go_decision=True,
    )
    data["oos_gate_result"] = GateResult(
        passed=True,
        checks=[
            {"name": "oos_sharpe", "passed": True, "actual": 1.25, "threshold": 1.0},
            {"name": "oos_profit_factor", "passed": True, "actual": 1.64, "threshold": 1.2},
            {"name": "pf_ci_lower", "passed": True, "actual": 1.12, "threshold": 1.0},
            {"name": "sharpe_ci_lower", "passed": True, "actual": 0.31, "threshold": 0.0},
        ],
        failures=[],
    )
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "OOS Bootstrap Confidence Intervals" in content
    assert "1.12" in content
    assert "OOS Promotion Gate" in content
    assert "oos_sharpe" in content


def test_report_without_oos_sections(tmp_path: Path) -> None:
    """Report without OOS data should not have OOS sections."""
    data = _make_report_data()
    output = tmp_path / "report.md"
    generate_report(data, output)
    content = output.read_text()
    assert "OOS Bootstrap" not in content
    assert "OOS Promotion Gate" not in content
