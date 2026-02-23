"""Tests for execution/reconcile CLI commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from pipeline.runner.cli import main


def test_cli_execute_close_calls_run_close_cycle() -> None:
    """`execute --phase close` should invoke run_close_cycle."""
    mock_pipeline = MagicMock()
    mock_pipeline._load_data.return_value = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-21"], utc=True),
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        }
    )
    mock_pipeline._generate_signals.return_value = pd.DataFrame(
        [{"symbol": "AAPL", "entry": True, "score": 0.8}]
    )
    mock_pipeline.spec = SimpleNamespace(
        risk=SimpleNamespace(max_positions=5, max_sector_exposure=0.3),
        execution=None,
    )

    mock_result = SimpleNamespace(accepted_count=1, rejected_count=0, dry_run=True)

    with (
        patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline),
        patch("pipeline.execution.service.run_close_cycle", return_value=mock_result) as run_close,
        patch(
            "pipeline.execution.reporting.generate_execution_report",
            return_value=SimpleNamespace(report_path=Path("reports/r.md"), gate_result=None),
        ) as report_gen,
    ):
        rc = main(
            [
                "execute",
                "--strategy",
                "examples/demo_vcp",
                "--phase",
                "close",
                "--dry-run",
            ]
        )

    assert rc == 0
    assert run_close.called
    assert report_gen.called


def test_cli_execute_close_refresh_data_calls_fetch() -> None:
    """`execute --phase close --refresh-data` should call pipeline.fetch_data first."""
    mock_pipeline = MagicMock()
    mock_pipeline.fetch_data.return_value = Path("data/ohlcv.parquet")
    mock_pipeline._load_data.return_value = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-21"], utc=True),
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        }
    )
    mock_pipeline._generate_signals.return_value = pd.DataFrame(
        [{"symbol": "AAPL", "entry": True, "score": 0.8}]
    )
    mock_pipeline.spec = SimpleNamespace(
        risk=SimpleNamespace(max_positions=5, max_sector_exposure=0.3),
        execution=None,
    )

    with (
        patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline),
        patch(
            "pipeline.execution.service.run_close_cycle",
            return_value=SimpleNamespace(accepted_count=1, rejected_count=0, dry_run=True),
        ),
        patch(
            "pipeline.execution.reporting.generate_execution_report",
            return_value=SimpleNamespace(report_path=Path("reports/r.md"), gate_result=None),
        ),
    ):
        rc = main(
            [
                "execute",
                "--strategy",
                "examples/demo_vcp",
                "--phase",
                "close",
                "--refresh-data",
                "--dry-run",
            ]
        )

    assert rc == 0
    mock_pipeline.fetch_data.assert_called_once()


def test_cli_execute_open_requires_alpaca_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """`execute --phase open` should fail when Alpaca keys are missing."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    rc = main(["execute", "--strategy", "examples/demo_vcp", "--phase", "open"])
    assert rc == 1


def test_cli_execute_live_requires_confirm_live_flag() -> None:
    """Live execute should require explicit --confirm-live CLI flag."""
    rc = main(
        [
            "execute",
            "--strategy",
            "examples/demo_vcp",
            "--phase",
            "open",
            "--mode",
            "live",
        ]
    )
    assert rc == 1


def test_cli_execute_live_requires_strategy_live_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live execute should fail if strategy execution.live confirmation is absent."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    rc = main(
        [
            "execute",
            "--strategy",
            "examples/demo_vcp",
            "--phase",
            "open",
            "--mode",
            "live",
            "--confirm-live",
        ]
    )
    assert rc == 1


def test_cli_execute_live_with_double_confirmation_calls_open_cycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live execute should run when both spec and CLI confirmations are present."""
    strategy_path = tmp_path / "strategy.yaml"
    strategy_path.write_text(
        yaml.safe_dump(
            {
                "id": "tmp_live_canary",
                "name": "Tmp Live Canary",
                "description": "tmp",
                "universe": {"type": "us_equities", "index": "sp500", "filters": []},
                "data": {"timeframe": "daily", "lookback_years": 8},
                "signals": {
                    "entry": {"type": "pivot_breakout", "conditions": ["x"], "trend_filter": []},
                    "exit": {
                        "stop_loss": "7% below entry",
                        "trailing_stop": "below 21-day EMA",
                        "take_profit": "risk_reward_3x",
                    },
                },
                "risk": {
                    "position_sizing": "fixed_risk",
                    "risk_per_trade": 0.005,
                    "max_positions": 1,
                    "max_sector_exposure": 0.30,
                },
                "cost_model": {"commission_per_share": 0.0, "slippage_bps": 5},
                "validation": {"method": "full_sample"},
                "promotion_gates": {
                    "min_trades": 200,
                    "max_drawdown": 0.15,
                    "profit_factor": 1.2,
                },
                "execution": {
                    "enabled": True,
                    "mode": "live",
                    "broker": "alpaca",
                    "confirm_live": True,
                },
            }
        )
    )

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    mock_result = SimpleNamespace(
        submitted_count=1,
        skipped_existing_count=0,
        fill_events_count=0,
        dry_run=False,
    )
    with (
        patch("pipeline.runner.cli._resolve_spec_path", return_value=strategy_path),
        patch("pipeline.execution.broker.alpaca.AlpacaBroker"),
        patch("pipeline.execution.service.run_open_cycle", return_value=mock_result) as run_open,
        patch(
            "pipeline.execution.reporting.generate_execution_report",
            return_value=SimpleNamespace(
                report_path=Path("reports/r.md"),
                gate_result=SimpleNamespace(passed=True),
            ),
        ),
    ):
        rc = main(
            [
                "execute",
                "--strategy",
                "tmp_live_canary",
                "--phase",
                "open",
                "--mode",
                "live",
                "--confirm-live",
            ]
        )

    assert rc == 0
    assert run_open.called


def test_cli_execute_live_canary_violation_is_blocked(tmp_path: Path) -> None:
    """Live execute should be blocked when canary limits are exceeded."""
    strategy_path = tmp_path / "strategy.yaml"
    strategy_path.write_text(
        yaml.safe_dump(
            {
                "id": "tmp_live_violation",
                "name": "Tmp Live Violation",
                "description": "tmp",
                "universe": {"type": "us_equities", "index": "sp500", "filters": []},
                "data": {"timeframe": "daily", "lookback_years": 8},
                "signals": {
                    "entry": {"type": "pivot_breakout", "conditions": ["x"], "trend_filter": []},
                    "exit": {
                        "stop_loss": "7% below entry",
                        "trailing_stop": "below 21-day EMA",
                        "take_profit": "risk_reward_3x",
                    },
                },
                "risk": {
                    "position_sizing": "fixed_risk",
                    "risk_per_trade": 0.01,
                    "max_positions": 5,
                    "max_sector_exposure": 0.30,
                },
                "cost_model": {"commission_per_share": 0.0, "slippage_bps": 5},
                "validation": {"method": "full_sample"},
                "promotion_gates": {
                    "min_trades": 200,
                    "max_drawdown": 0.15,
                    "profit_factor": 1.2,
                },
                "execution": {
                    "enabled": True,
                    "mode": "live",
                    "broker": "alpaca",
                    "confirm_live": True,
                },
            }
        )
    )
    with patch("pipeline.runner.cli._resolve_spec_path", return_value=strategy_path):
        rc = main(
            [
                "execute",
                "--strategy",
                "tmp_live_violation",
                "--phase",
                "open",
                "--mode",
                "live",
                "--confirm-live",
            ]
        )
    assert rc == 1


def test_cli_execute_open_calls_run_open_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """`execute --phase open` should call run_open_cycle when keys exist."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

    mock_result = SimpleNamespace(
        submitted_count=1,
        skipped_existing_count=0,
        fill_events_count=0,
        dry_run=False,
    )

    with (
        patch("pipeline.execution.broker.alpaca.AlpacaBroker") as broker_cls,
        patch("pipeline.execution.service.run_open_cycle", return_value=mock_result) as run_open,
        patch(
            "pipeline.execution.reporting.generate_execution_report",
            return_value=SimpleNamespace(
                report_path=Path("reports/r.md"),
                gate_result=SimpleNamespace(passed=True),
            ),
        ) as report_gen,
    ):
        rc = main(["execute", "--strategy", "examples/demo_vcp", "--phase", "open"])

    assert rc == 0
    assert broker_cls.called
    assert run_open.called
    assert "max_daily_loss_pct" in run_open.call_args.kwargs
    assert report_gen.called


def test_cli_reconcile_calls_run_open_cycle_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`reconcile` should run open cycle in dry_run mode."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

    mock_result = SimpleNamespace(
        submitted_count=0,
        skipped_existing_count=0,
        active_orders_count=1,
        fill_events_count=0,
        dry_run=True,
    )

    with (
        patch("pipeline.execution.broker.alpaca.AlpacaBroker"),
        patch("pipeline.execution.service.run_open_cycle", return_value=mock_result) as run_open,
    ):
        rc = main(["reconcile", "--strategy", "examples/demo_vcp"])

    assert rc == 0
    assert run_open.called
    assert run_open.call_args.kwargs["dry_run"] is True


def test_cli_execute_close_report_failure_does_not_fail_cycle() -> None:
    """close-cycle success should return 0 even if report generation fails."""
    mock_pipeline = MagicMock()
    mock_pipeline._load_data.return_value = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-02-21"], utc=True),
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        }
    )
    mock_pipeline._generate_signals.return_value = pd.DataFrame(
        [{"symbol": "AAPL", "entry": True, "score": 0.8}]
    )
    mock_pipeline.spec = SimpleNamespace(
        risk=SimpleNamespace(max_positions=5, max_sector_exposure=0.3),
        execution=None,
    )
    mock_result = SimpleNamespace(accepted_count=1, rejected_count=0, dry_run=False)

    with (
        patch("pipeline.runner.pipeline.Pipeline", return_value=mock_pipeline),
        patch("pipeline.execution.service.run_close_cycle", return_value=mock_result),
        patch(
            "pipeline.execution.reporting.generate_execution_report",
            side_effect=RuntimeError("report failed"),
        ),
    ):
        rc = main(["execute", "--strategy", "examples/demo_vcp", "--phase", "close"])

    assert rc == 0


def test_cli_execute_open_report_failure_does_not_fail_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open-cycle success should return 0 even if report generation fails."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

    mock_result = SimpleNamespace(
        submitted_count=1,
        skipped_existing_count=0,
        fill_events_count=0,
        dry_run=False,
    )
    with (
        patch("pipeline.execution.broker.alpaca.AlpacaBroker"),
        patch("pipeline.execution.service.run_open_cycle", return_value=mock_result),
        patch(
            "pipeline.execution.reporting.generate_execution_report",
            side_effect=RuntimeError("report failed"),
        ),
    ):
        rc = main(["execute", "--strategy", "examples/demo_vcp", "--phase", "open"])

    assert rc == 0
