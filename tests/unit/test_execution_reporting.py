"""Tests for execution reporting and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.execution.reporting import compute_execution_metrics, generate_execution_report


def _journal_df() -> pd.DataFrame:
    ts = pd.Timestamp("2026-02-24T16:15:00Z")
    return pd.DataFrame(
        [
            {"timestamp": ts, "event_type": "intent_created", "symbol": "AAPL"},
            {"timestamp": ts, "event_type": "intent_created", "symbol": "MSFT"},
            {"timestamp": ts, "event_type": "intent_rejected", "symbol": "GOOG"},
            {
                "timestamp": ts,
                "event_type": "order_submitted",
                "symbol": "AAPL",
                "broker_order_id": "ord-1",
            },
            {
                "timestamp": ts,
                "event_type": "order_submitted",
                "symbol": "MSFT",
                "broker_order_id": "ord-2",
            },
            {"timestamp": ts, "event_type": "order_submit_failed", "symbol": "NFLX"},
            {"timestamp": ts, "event_type": "order_canceled", "symbol": "MSFT"},
            {
                "timestamp": ts,
                "event_type": "fill_detected",
                "broker_order_id": "ord-1",
                "symbol": "AAPL",
                "side": "buy",
                "fill_price": 101.0,
                "slippage_bps": 5.0,
            },
            {
                "timestamp": ts,
                "event_type": "fill_detected",
                "broker_order_id": "ord-1",
                "symbol": "AAPL",
                "side": "buy",
                "fill_price": 102.0,
                "slippage_bps": 10.0,
            },
            {
                "timestamp": ts,
                "event_type": "fill_detected",
                "broker_order_id": "ord-2",
                "symbol": "MSFT",
                "side": "sell",
                "fill_price": 50.0,
                "slippage_bps": -2.0,
            },
        ]
    )


def test_compute_execution_metrics_core_values() -> None:
    """Core execution metrics should be computed from journal events."""
    journal = _journal_df()
    metrics = compute_execution_metrics(
        journal,
        paper_metrics={"profit_factor": 1.20, "sharpe": 1.00, "win_rate": 0.55, "trade_count": 45},
        backtest_metrics={
            "profit_factor": 1.00,
            "sharpe": 1.25,
            "win_rate": 0.50,
            "trade_count": 50,
        },
    )

    assert metrics["intent_created_count"] == 2
    assert metrics["intent_rejected_count"] == 1
    assert metrics["order_submitted_count"] == 2
    assert metrics["order_submit_failed_count"] == 1
    assert metrics["order_canceled_count"] == 1
    assert metrics["fill_event_count"] == 3
    assert metrics["filled_order_count"] == 2
    assert metrics["partial_fill_ratio"] == pytest.approx(0.5)
    assert metrics["reject_ratio"] == pytest.approx(0.25)
    assert metrics["cancel_ratio"] == pytest.approx(0.25)
    assert metrics["reject_cancel_ratio"] == pytest.approx(0.5)
    assert metrics["realized_slippage_bps"] == pytest.approx((5.0 + 10.0 + 2.0) / 3.0)
    # deviation = mean([
    #   |1.20-1.00|/1.00=0.20, |1.00-1.25|/1.25=0.20,
    #   |0.55-0.50|/0.50=0.10, |45-50|/50=0.10
    # ]) = 0.15
    assert metrics["paper_backtest_deviation"] == pytest.approx(0.15)


def test_generate_execution_report_writes_md_csv_json(tmp_path: Path) -> None:
    """Report generator should emit markdown, raw events CSV, and metrics JSON."""
    journal_path = tmp_path / "journal.parquet"
    _journal_df().to_parquet(journal_path, index=False)
    output_dir = tmp_path / "reports" / "demo_vcp" / "execution" / "2026-02-24"

    result = generate_execution_report(
        strategy_id="demo_vcp",
        session_date="2026-02-24",
        journal_path=journal_path,
        output_dir=output_dir,
        paper_metrics={"profit_factor": 1.20, "sharpe": 1.00, "win_rate": 0.55, "trade_count": 45},
        backtest_metrics={
            "profit_factor": 1.00,
            "sharpe": 1.25,
            "win_rate": 0.50,
            "trade_count": 50,
        },
        paper_gates={
            "min_paper_trades": 2,
            "max_reject_cancel_ratio": 0.60,
            "max_partial_fill_ratio": 0.80,
            "max_realized_slippage_bps": 6.0,
            "max_backtest_deviation": 0.20,
        },
    )

    assert result.report_path.exists()
    assert result.metrics_json_path.exists()
    assert result.events_csv_path.exists()
    assert result.gate_result is not None
    assert result.gate_result.passed is True

    report_text = result.report_path.read_text()
    assert "Execution Report" in report_text
    assert "Execution Metrics" in report_text
    assert "Paper Gate: **PASS**" in report_text
    assert "reject_cancel_ratio" in report_text

    payload = json.loads(result.metrics_json_path.read_text())
    assert payload["strategy_id"] == "demo_vcp"
    assert payload["session_date"] == "2026-02-24"
    assert payload["paper_gate"]["passed"] is True

    events_df = pd.read_csv(result.events_csv_path)
    assert len(events_df) == 10


def test_compute_execution_metrics_deviation_skips_zero_backtest_denominator() -> None:
    """Zero backtest baselines should be skipped to keep deviation ratio-based."""
    journal = _journal_df()
    metrics = compute_execution_metrics(
        journal,
        paper_metrics={"profit_factor": 1.20, "sharpe": 1.00, "win_rate": 0.55, "trade_count": 45},
        backtest_metrics={
            "profit_factor": 1.00,
            "sharpe": 0.00,
            "win_rate": 0.50,
            "trade_count": 0,
        },
    )
    # sharpe/trade_count are skipped because backtest baseline is zero.
    # deviation = mean([|1.20-1.00|/1.00=0.20, |0.55-0.50|/0.50=0.10]) = 0.15
    assert metrics["paper_backtest_deviation"] == pytest.approx(0.15)


def test_compute_execution_metrics_deviation_all_zero_backtest_returns_none() -> None:
    """When every comparable backtest baseline is zero, deviation should be unavailable."""
    journal = _journal_df()
    metrics = compute_execution_metrics(
        journal,
        paper_metrics={"sharpe": 1.00, "trade_count": 45},
        backtest_metrics={"sharpe": 0.00, "trade_count": 0},
    )
    assert metrics["paper_backtest_deviation"] is None
