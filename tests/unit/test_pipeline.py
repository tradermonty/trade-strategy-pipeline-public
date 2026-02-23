"""Tests for the pipeline runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.runner.pipeline import Pipeline, PipelineResult
from pipeline.spec.schema import ExecutionConfig

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies" / "examples"


def test_pipeline_loads_spec() -> None:
    """Pipeline should load and validate the strategy spec."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    assert pipeline.spec.id == "demo_vcp"


def test_pipeline_stages_sequential() -> None:
    """Pipeline stages should be defined in order."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    stages = pipeline.stages
    assert len(stages) > 0
    expected_order = ["load_data", "generate_signals", "backtest", "evaluate", "report"]
    for expected, actual in zip(expected_order, stages, strict=True):
        assert actual == expected


def test_pipeline_returns_result() -> None:
    """Pipeline.run() should return a PipelineResult."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    assert result.strategy_id == "demo_vcp"


def test_pipeline_promotion_gate_requires_portfolio_mode() -> None:
    """Promotion gate should fail when portfolio_mode is disabled."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    result = pipeline.run()
    assert result.gate_result is not None
    assert result.gate_result.passed is False
    checks = [c for c in result.gate_result.checks if c["name"] == "portfolio_mode_required"]
    assert len(checks) == 1
    assert checks[0]["actual"] == 0.0
    assert checks[0]["threshold"] == 1.0


def test_pipeline_result_has_bootstrap() -> None:
    """Pipeline.run() should populate bootstrap_result."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    result = pipeline.run()
    assert result.bootstrap_result is not None
    assert result.bootstrap_result.n_iterations == 10_000
    assert isinstance(result.bootstrap_result.go_decision, bool)


def test_pipeline_fails_gracefully_on_data_error(tmp_path: Path) -> None:
    """Pipeline should handle missing data gracefully."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    # Point to an empty directory
    empty_data = tmp_path / "empty"
    empty_data.mkdir()
    pipeline = Pipeline(spec_path=spec_path, data_dir=empty_data, output_dir=tmp_path / "out")
    result = pipeline.run()
    assert result.success is False
    assert result.error is not None
    assert result.failed_stage == "load_data"
    assert result.failure_report_path is not None
    assert result.failure_report_path.exists()
    content = result.failure_report_path.read_text()
    assert "No OHLCV data found" in content
    assert "Suggested Action" in content
    assert "## Reproduce" in content
    assert "--strategy demo_vcp" in content
    assert "## Environment" in content


def test_pipeline_failure_report_records_exception_stage(tmp_path: Path) -> None:
    """Runtime exception should preserve failed stage and write failure report."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR, output_dir=tmp_path / "out")

    with patch.object(pipeline, "_generate_signals", side_effect=RuntimeError("signal boom")):
        result = pipeline.run()

    assert result.success is False
    assert result.error is not None
    assert "signal boom" in result.error
    assert result.failed_stage == "generate_signals"
    assert result.failure_report_path is not None
    assert result.failure_report_path.exists()
    content = result.failure_report_path.read_text()
    assert "generate_signals" in content
    assert "signal boom" in content
    assert "Spec Path" in content


def test_pipeline_load_data_auto_raises_when_ambiguous(tmp_path: Path) -> None:
    """auto profile should fail when sample and production parquet both exist."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    sample = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")
    sample.to_parquet(tmp_path / "sample_ohlcv.parquet", index=False)
    sample.to_parquet(tmp_path / "ohlcv.parquet", index=False)

    pipeline = Pipeline(spec_path=spec_path, data_dir=tmp_path, data_profile="auto")
    result = pipeline.run()
    assert result.success is False
    assert result.error is not None
    assert "Ambiguous data source" in result.error


def test_pipeline_load_data_profile_production_selects_ohlcv(tmp_path: Path) -> None:
    """production profile should load ohlcv.parquet."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    sample = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")
    sample.to_parquet(tmp_path / "sample_ohlcv.parquet", index=False)
    sample.iloc[:20].to_parquet(tmp_path / "ohlcv.parquet", index=False)

    pipeline = Pipeline(spec_path=spec_path, data_dir=tmp_path, data_profile="production")
    loaded = pipeline._load_data()
    assert loaded is not None
    assert len(loaded) == 20


def test_pipeline_load_data_profile_production_rejects_sample_only(tmp_path: Path) -> None:
    """production profile should fail-fast when only sample parquet exists."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    sample = pd.read_parquet(FIXTURES_DIR / "sample_ohlcv.parquet")
    sample.to_parquet(tmp_path / "sample_ohlcv.parquet", index=False)

    pipeline = Pipeline(spec_path=spec_path, data_dir=tmp_path, data_profile="production")
    result = pipeline.run()
    assert result.success is False
    assert result.error is not None
    assert "data_profile='production'" in result.error
    assert "sample_ohlcv.parquet" in result.error


def test_pipeline_spec_stage_validation_rejects_walk_forward_in_phase1() -> None:
    """spec_stage override should enforce stage validation at runtime."""
    spec_path = STRATEGIES_DIR / "demo_gap_basic" / "strategy.yaml"
    with pytest.raises(ValueError, match="runtime validation"):
        Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR, spec_stage="phase1")


def test_pipeline_skip_validation_allows_stage_override() -> None:
    """skip_validation=True should allow explicit stage override for diagnostics."""
    spec_path = STRATEGIES_DIR / "demo_gap_basic" / "strategy.yaml"
    pipeline = Pipeline(
        spec_path=spec_path,
        data_dir=FIXTURES_DIR,
        spec_stage="phase1",
        skip_validation=True,
    )
    assert pipeline.spec_stage == "phase1"


def test_pipeline_infer_stage_phase2_when_execution_enabled() -> None:
    """Execution-enabled specs should infer phase2 stage by default."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    spec = pipeline.spec.model_copy(update={"execution": ExecutionConfig(enabled=True)})
    assert Pipeline._infer_spec_stage(spec) == "phase2"


# --- fetch_data tests (Step 3b) ---


def _make_pipeline(tmp_path: Path) -> Pipeline:
    """Create a Pipeline pointing at tmp_path for data."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    return Pipeline(spec_path=spec_path, data_dir=tmp_path)


def _mock_bars(symbol: str = "AAPL") -> pd.DataFrame:
    """Create a mock OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=5, tz="UTC"),
            "open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "high": [155.0, 156.0, 157.0, 158.0, 159.0],
            "low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "close": [153.0, 154.0, 155.0, 156.0, 157.0],
            "volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
            "symbol": [symbol] * 5,
        }
    )


def test_fetch_data_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_data should save OHLCV parquet and snapshot on success."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("CI_OFFLINE", raising=False)

    pipeline = _make_pipeline(tmp_path)

    mock_alpaca = MagicMock()
    mock_alpaca.fetch_sp500_constituents.return_value = ["AAPL", "MSFT"]
    mock_alpaca.fetch_nasdaq100_constituents.return_value = ["AAPL"]  # overlap
    mock_alpaca.fetch_daily_bars.side_effect = [_mock_bars("AAPL"), _mock_bars("MSFT")]

    with patch("pipeline.data.alpaca_client.AlpacaClient", return_value=mock_alpaca):
        result_path = pipeline.fetch_data()

    assert result_path.exists()
    df = pd.read_parquet(result_path)
    assert len(df) == 10  # 5 bars x 2 symbols
    # Snapshot should exist
    snapshots = list((tmp_path / "snapshots").glob("*/ohlcv.parquet"))
    assert len(snapshots) == 1


def test_fetch_data_missing_api_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_data should raise RuntimeError when API keys are missing."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    pipeline = _make_pipeline(tmp_path)

    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        pipeline.fetch_data()


def test_fetch_data_unsupported_universe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_data should raise RuntimeError for unsupported universe index."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("CI_OFFLINE", raising=False)

    pipeline = _make_pipeline(tmp_path)
    pipeline.spec.universe.index = "nasdaq"

    with pytest.raises(RuntimeError, match="Unsupported universe index"):
        pipeline.fetch_data()


def test_fetch_data_no_symbols_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_data should raise RuntimeError when all symbols fail."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("CI_OFFLINE", raising=False)

    pipeline = _make_pipeline(tmp_path)

    mock_alpaca = MagicMock()
    mock_alpaca.fetch_sp500_constituents.return_value = ["FAIL1", "FAIL2"]
    mock_alpaca.fetch_daily_bars.side_effect = Exception("API error")

    with (
        patch("pipeline.data.alpaca_client.AlpacaClient", return_value=mock_alpaca),
        pytest.raises(RuntimeError, match="No data fetched"),
    ):
        pipeline.fetch_data()


def test_pipeline_oos_bootstrap_with_walk_forward() -> None:
    """Pipeline with walk_forward spec should run OOS bootstrap + OOS gate."""
    spec_path = STRATEGIES_DIR / "demo_gap_basic" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    assert pipeline.spec.validation.method == "walk_forward"
    # Fixture history is short; keep first IS split >= warmup(200) bars.
    pipeline.spec.validation.oos_ratio = 0.20
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    # WF should have produced a walk_forward_result
    if result.walk_forward_result is not None:
        wf = result.walk_forward_result
        # If OOS trades exist, OOS bootstrap and gate should be populated
        if wf.all_oos_trades:
            assert result.oos_bootstrap_result is not None
            assert result.oos_gate_result is not None
            assert isinstance(result.oos_gate_result.passed, bool)
            assert any(
                c["name"] == "portfolio_mode_required" for c in result.oos_gate_result.checks
            )


def test_pipeline_trades_have_symbol() -> None:
    """Trades produced by Pipeline._run_backtest() should contain 'symbol' key."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    result = pipeline.run()
    assert result.backtest_result is not None
    for trade in result.backtest_result.trades:
        assert "symbol" in trade, "Trade dict missing 'symbol' key"
        assert isinstance(trade["symbol"], str)


def test_pipeline_trades_have_bars_held() -> None:
    """Trades produced by Pipeline._run_backtest() should contain 'bars_held' key."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    result = pipeline.run()
    assert result.backtest_result is not None
    for trade in result.backtest_result.trades:
        assert "bars_held" in trade, "Trade dict missing 'bars_held' key"
        assert isinstance(trade["bars_held"], int)


def test_pipeline_gap_up_open_scored_runs() -> None:
    """Pipeline with demo_gap_scored should complete successfully."""
    spec_path = STRATEGIES_DIR / "demo_gap_scored" / "strategy.yaml"
    assert spec_path.exists(), f"Strategy YAML not found: {spec_path}"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    assert pipeline.spec.id == "demo_gap_scored"
    assert pipeline.spec.gap_open_scored is not None
    assert pipeline.spec.earnings_gap_open is not None
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    assert result.success is True, f"Pipeline failed: {result.error}"
    assert result.metrics is not None
    assert result.gate_result is not None


def test_pipeline_gap_up_open_scored_entry_at_open() -> None:
    """Gap-up open scored strategy should produce entry_at_open signals."""
    spec_path = STRATEGIES_DIR / "demo_gap_scored" / "strategy.yaml"
    pipeline = Pipeline(spec_path=spec_path, data_dir=FIXTURES_DIR)
    ohlcv = pipeline._load_data()
    assert ohlcv is not None
    signals = pipeline._generate_signals(ohlcv)
    # Check entry_at_open column exists
    assert "entry_at_open" in signals.columns
    # Any entry should have entry_at_open=True (for this strategy)
    entries = signals[signals["entry"]]
    if len(entries) > 0:
        assert entries["entry_at_open"].all(), (
            "All entries should be entry_at_open for this strategy"
        )


def test_fetch_data_russell1000_universe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_data resolves russell1000 universe via FMP screener."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("FMP_API_KEY", "test-fmp-key")
    monkeypatch.delenv("CI_OFFLINE", raising=False)

    pipeline = _make_pipeline(tmp_path)
    pipeline.spec.universe.index = "russell1000"
    pipeline.spec.universe.selection.fetch_limit = 2000
    pipeline.spec.universe.selection.top_n_by_market_cap = 900
    pipeline.spec.universe.selection.include_etf = False
    pipeline.spec.universe.selection.include_fund = False

    mock_alpaca = MagicMock()
    mock_alpaca.fetch_daily_bars.side_effect = [_mock_bars("AAPL"), _mock_bars("MSFT")]

    mock_fmp = MagicMock()
    mock_fmp.fetch_russell1000_approx.return_value = ["AAPL", "MSFT"]
    mock_fmp.get_bulk_sectors.return_value = pd.DataFrame(
        {"ticker": ["AAPL"], "sector": ["Tech"], "industry": ["Consumer Electronics"]}
    )

    with (
        patch("pipeline.data.alpaca_client.AlpacaClient", return_value=mock_alpaca),
        patch("pipeline.data.fmp_client.FmpClient", return_value=mock_fmp),
    ):
        result_path = pipeline.fetch_data()

    assert result_path.exists()
    df = pd.read_parquet(result_path)
    assert len(df) == 10  # 5 bars x 2 symbols
    mock_fmp.fetch_russell1000_approx.assert_called_once()
    kwargs = mock_fmp.fetch_russell1000_approx.call_args.kwargs
    assert kwargs["fetch_limit"] == 2000
    assert kwargs["top_n_by_market_cap"] == 900
    assert kwargs["include_etf"] is False
    assert kwargs["include_fund"] is False
    metadata_path = tmp_path / "universe_selection.json"
    assert metadata_path.exists()


def test_fetch_data_russell1000_requires_fmp_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch_data should raise RuntimeError for russell1000 without FMP key."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.delenv("CI_OFFLINE", raising=False)

    pipeline = _make_pipeline(tmp_path)
    pipeline.spec.universe.index = "russell1000"

    mock_alpaca = MagicMock()
    with (
        patch("pipeline.data.alpaca_client.AlpacaClient", return_value=mock_alpaca),
        pytest.raises(RuntimeError, match="FMP_API_KEY is required"),
    ):
        pipeline.fetch_data()


def test_fetch_data_fmp_failure_graceful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_data should save OHLCV even when FMP sector fetch fails."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("FMP_API_KEY", "test-fmp-key")
    monkeypatch.delenv("CI_OFFLINE", raising=False)

    pipeline = _make_pipeline(tmp_path)

    mock_alpaca = MagicMock()
    mock_alpaca.fetch_sp500_constituents.return_value = ["AAPL"]
    mock_alpaca.fetch_nasdaq100_constituents.return_value = ["AAPL"]
    mock_alpaca.fetch_daily_bars.return_value = _mock_bars("AAPL")

    mock_fmp = MagicMock()
    mock_fmp.get_bulk_sectors.side_effect = Exception("FMP rate limit")

    with (
        patch("pipeline.data.alpaca_client.AlpacaClient", return_value=mock_alpaca),
        patch("pipeline.data.fmp_client.FmpClient", return_value=mock_fmp),
    ):
        result_path = pipeline.fetch_data()

    # OHLCV should still be saved despite FMP failure
    assert result_path.exists()
    df = pd.read_parquet(result_path)
    assert len(df) == 5
