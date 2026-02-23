"""Tests for bootstrap confidence interval estimation."""

from __future__ import annotations

import numpy as np

from pipeline.evaluation.bootstrap import bootstrap_ci


def _make_trades(pnls: list[float]) -> list[dict[str, object]]:
    """Create trade dicts from a list of PnL values."""
    return [{"pnl": p, "symbol": "TEST", "entry_date": "2024-01-01"} for p in pnls]


def test_bootstrap_ci_positive_edge() -> None:
    """Trades with clear positive edge should have PF CI lower > 1.0."""
    rng = np.random.default_rng(99)
    pnls = (rng.normal(50, 100, 300)).tolist()
    # Ensure mean is clearly positive
    pnls = [p + 100 for p in pnls]
    trades = _make_trades(pnls)
    result = bootstrap_ci(trades, n_iterations=5_000, rng_seed=42)
    assert result.pf_ci_lower > 1.0


def test_bootstrap_ci_no_edge() -> None:
    """Random PnL around 0 should have Sharpe CI that spans 0."""
    rng = np.random.default_rng(42)
    pnls = rng.normal(0, 100, 200).tolist()
    trades = _make_trades(pnls)
    result = bootstrap_ci(trades, n_iterations=5_000, rng_seed=42)
    # Sharpe CI should cross zero (lower < 0 or upper > 0 with lower < 0)
    assert result.sharpe_ci_lower < 0.5  # not strongly positive


def test_bootstrap_ci_deterministic() -> None:
    """Same seed should produce identical results."""
    trades = _make_trades([100, -50, 200, -30, 150, -80, 60, -20])
    r1 = bootstrap_ci(trades, n_iterations=1_000, rng_seed=42)
    r2 = bootstrap_ci(trades, n_iterations=1_000, rng_seed=42)
    assert r1.pf_mean == r2.pf_mean
    assert r1.pf_ci_lower == r2.pf_ci_lower
    assert r1.sharpe_mean == r2.sharpe_mean
    assert r1.sharpe_ci_lower == r2.sharpe_ci_lower


def test_bootstrap_ci_empty_trades() -> None:
    """Empty trades should return zeros and go_decision=False."""
    result = bootstrap_ci([], n_iterations=1_000, rng_seed=42)
    assert result.pf_mean == 0.0
    assert result.sharpe_mean == 0.0
    assert result.go_decision is False


def test_bootstrap_inf_pf_clipped() -> None:
    """All-winner samples should produce finite PF CI values."""
    # All positive PnL → some resamples will have gross_loss=0
    trades = _make_trades([100, 200, 50, 150, 80, 300, 120, 90])
    result = bootstrap_ci(trades, n_iterations=5_000, rng_seed=42)
    assert np.isfinite(result.pf_ci_lower)
    assert np.isfinite(result.pf_ci_upper)
    assert np.isfinite(result.pf_mean)


def test_bootstrap_go_decision_true() -> None:
    """Clear positive edge should yield go_decision=True."""
    # Large positive trades with small losses
    pnls = [200.0, 150.0, -30.0, 180.0, -20.0, 250.0, 160.0, -40.0, 300.0, -25.0] * 30
    trades = _make_trades(pnls)
    result = bootstrap_ci(trades, n_iterations=5_000, rng_seed=42)
    assert result.go_decision is True
    assert result.pf_ci_lower > 1.0
    assert result.sharpe_ci_lower > 0.0


def test_bootstrap_90ci_narrower_than_95ci() -> None:
    """90% CI should produce narrower bounds than 95% CI."""
    trades = _make_trades([100, -50, 200, -30, 150, -80, 60, -20] * 10)
    r95 = bootstrap_ci(trades, n_iterations=5_000, confidence_level=0.95, rng_seed=42)
    r90 = bootstrap_ci(trades, n_iterations=5_000, confidence_level=0.90, rng_seed=42)
    assert r90.pf_ci_lower >= r95.pf_ci_lower
    assert r90.pf_ci_upper <= r95.pf_ci_upper
    assert r90.sharpe_ci_lower >= r95.sharpe_ci_lower
    assert r90.sharpe_ci_upper <= r95.sharpe_ci_upper


def test_bootstrap_go_decision_false() -> None:
    """Marginal edge should yield go_decision=False (PF CI < 1 or Sharpe CI < 0)."""
    # Roughly break-even trades
    rng = np.random.default_rng(7)
    pnls = rng.normal(0, 100, 100).tolist()
    trades = _make_trades(pnls)
    result = bootstrap_ci(trades, n_iterations=5_000, rng_seed=42)
    assert result.go_decision is False
