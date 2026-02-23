"""Integration test: end-to-end pipeline with fixture data."""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.runner.pipeline import Pipeline, PipelineResult

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies" / "examples"


def test_pipeline_end_to_end_offline(tmp_path: Path) -> None:
    """Full pipeline should complete with fixture data in offline mode."""
    spec_path = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"
    pipeline = Pipeline(
        spec_path=spec_path,
        data_dir=FIXTURES_DIR,
        output_dir=tmp_path / "reports",
    )
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    assert result.strategy_id == "demo_vcp"
    # Pipeline should complete (even if gates don't pass with limited fixture data)
    assert result.metrics is not None
    assert result.gate_result is not None


def _make_gap_up_fixture(data_dir: Path) -> None:
    """Create a synthetic parquet with a gap-up pattern for E2E testing."""
    rng = np.random.default_rng(123)
    n = 300
    # Steady uptrend so SMA50 > SMA200
    close_values: list[float] = [50.0]
    for _ in range(1, n):
        close_values.append(close_values[-1] * (1 + 0.0015 + rng.normal(0, 0.003)))
    closes = np.array(close_values, dtype=float)
    volume = rng.integers(400_000, 600_000, size=n).astype(float)

    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = closes * (1 + abs(rng.normal(0.003, 0.002, n)))
    lows = closes * (1 - abs(rng.normal(0.003, 0.002, n)))

    # Inject gap-up at bar 250
    prev_close = closes[249]
    gap_open = prev_close * 1.06
    opens[250] = gap_open
    highs[250] = gap_open * 1.005
    lows[250] = gap_open * 0.995
    closes[250] = gap_open
    volume[250] = volume[200:250].mean() * 3.0

    # Inject breakout at bar 255
    gap_high = highs[250]
    closes[255] = gap_high * 1.01
    highs[255] = gap_high * 1.015

    timestamps = pd.date_range("2018-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SYN_GAP_TEST",
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
    )
    df.to_parquet(data_dir / "sample_ohlcv.parquet", index=False)


def test_gap_up_pipeline_end_to_end(tmp_path: Path) -> None:
    """Gap-up strategy pipeline should complete end-to-end."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_gap_up_fixture(data_dir)

    spec_path = STRATEGIES_DIR / "demo_gap_basic" / "strategy.yaml"
    pipeline = Pipeline(
        spec_path=spec_path,
        data_dir=data_dir,
        output_dir=tmp_path / "reports",
    )
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    assert result.strategy_id == "demo_gap_basic"
    assert result.success is True
    assert result.metrics is not None
    assert result.gate_result is not None
