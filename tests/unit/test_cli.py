"""Tests for CLI entry point."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.runner.cli import main

STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies"


def test_cli_strategy_argument() -> None:
    """CLI should accept --strategy argument."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.runner.cli",
            "--strategy",
            "examples/demo_vcp",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (
        result.returncode == 0
        or "strategy" in result.stdout.lower()
        or "vcp" in result.stdout.lower()
    )


def test_cli_data_source_override() -> None:
    """CLI should accept --data-dir override."""
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.runner.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "data-dir" in result.stdout or "data_dir" in result.stdout
    assert "data-profile" in result.stdout
    assert "spec-stage" in result.stdout


def test_cli_help_message() -> None:
    """CLI --help should show usage information."""
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.runner.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "strategy" in result.stdout.lower()


# --- Direct-call tests (Step 3a) ---


def test_cli_dry_run_direct() -> None:
    """main() with --dry-run should return 0 and print config."""
    rc = main(["--strategy", "examples/demo_vcp", "--dry-run"])
    assert rc == 0


def test_cli_dry_run_direct_with_short_strategy_id() -> None:
    """main() should resolve shorthand example strategy IDs."""
    rc = main(["--strategy", "demo_vcp", "--dry-run"])
    assert rc == 0


def test_cli_missing_strategy_returns_error() -> None:
    """main() with a nonexistent strategy should return 1."""
    rc = main(["--strategy", "nonexistent_strategy_xyz", "--dry-run"])
    assert rc == 1


def test_cli_fetch_data_runtime_error() -> None:
    """main() --fetch-data should return 1 when fetch_data raises RuntimeError."""
    mock_pipeline = MagicMock()
    mock_pipeline.fetch_data.side_effect = RuntimeError("ALPACA_API_KEY not set")

    with patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline):
        rc = main(["--strategy", "examples/demo_vcp", "--fetch-data"])
    assert rc == 1


def test_cli_fetch_data_success() -> None:
    """main() --fetch-data should return 0 on successful fetch."""
    mock_pipeline = MagicMock()
    mock_pipeline.fetch_data.return_value = Path("data/ohlcv.parquet")

    with patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline):
        rc = main(["--strategy", "examples/demo_vcp", "--fetch-data"])
    assert rc == 0


def test_cli_full_run_mocked() -> None:
    """main() full run with mocked Pipeline.run() should return 0 on success."""
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.gate_result = None
    mock_result.report_path = Path("reports/test/report.md")

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_result

    with patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline):
        rc = main(["--strategy", "examples/demo_vcp"])
    assert rc == 0


def test_cli_full_run_failure() -> None:
    """main() full run should return 1 when pipeline fails."""
    mock_result = MagicMock()
    mock_result.success = False
    mock_result.error = "Backtest failed"

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_result

    with patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline):
        rc = main(["--strategy", "examples/demo_vcp"])
    assert rc == 1


def test_cli_full_run_failure_prints_failure_report(capsys: pytest.CaptureFixture[str]) -> None:
    """main() should print failed stage and failure report path when available."""
    mock_result = MagicMock()
    mock_result.success = False
    mock_result.error = "Backtest failed"
    mock_result.failed_stage = "backtest"
    mock_result.failure_report_path = Path("reports/test/failure.md")

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_result

    with patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline):
        rc = main(["--strategy", "examples/demo_vcp"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Failed stage: backtest" in captured.err
    assert "Failure report: reports/test/failure.md" in captured.err
