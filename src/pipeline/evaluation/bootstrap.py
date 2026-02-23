"""Bootstrap confidence interval estimation for trade-level metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.evaluation.metrics import compute_profit_factor


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""

    pf_mean: float
    pf_ci_lower: float
    pf_ci_upper: float
    sharpe_mean: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    n_iterations: int
    confidence_level: float
    go_decision: bool


def bootstrap_ci(
    trades: list[dict[str, Any]],
    n_iterations: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int | None = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence intervals for PF and trade-level Sharpe.

    Args:
        trades: List of trade dicts, each must have a ``"pnl"`` key.
        n_iterations: Number of bootstrap resamples.
        confidence_level: CI level (e.g. 0.95 for 95%).
        rng_seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with CI bounds and go/no-go decision.
    """
    if not trades:
        return BootstrapResult(
            pf_mean=0.0,
            pf_ci_lower=0.0,
            pf_ci_upper=0.0,
            sharpe_mean=0.0,
            sharpe_ci_lower=0.0,
            sharpe_ci_upper=0.0,
            n_iterations=n_iterations,
            confidence_level=confidence_level,
            go_decision=False,
        )

    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    n_trades = len(pnls)
    rng = np.random.default_rng(rng_seed)

    pf_samples = np.empty(n_iterations)
    sharpe_samples = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n_trades, size=n_trades)
        sample = pnls[idx]

        # Profit Factor
        pf_val = compute_profit_factor(sample.tolist())
        pf_samples[i] = pf_val

        # Trade-level Sharpe: mean(pnl) / std(pnl) * sqrt(N)
        std = sample.std(ddof=1) if n_trades > 1 else 0.0
        if std == 0.0:
            sharpe_samples[i] = 0.0
        else:
            sharpe_samples[i] = sample.mean() / std * np.sqrt(n_trades)

    # Clip infinite PF values to max finite value
    finite_mask = np.isfinite(pf_samples)
    if finite_mask.any():
        max_finite = np.nanmax(pf_samples[finite_mask])
        pf_samples[~finite_mask] = max_finite
    else:
        pf_samples[:] = 0.0

    alpha = 1.0 - confidence_level
    lo_pct = alpha / 2.0 * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0

    pf_ci_lower, pf_ci_upper = np.percentile(pf_samples, [lo_pct, hi_pct])
    sharpe_ci_lower, sharpe_ci_upper = np.percentile(sharpe_samples, [lo_pct, hi_pct])

    go_decision = bool(pf_ci_lower > 1.0 and sharpe_ci_lower > 0.0)

    return BootstrapResult(
        pf_mean=float(np.mean(pf_samples)),
        pf_ci_lower=float(pf_ci_lower),
        pf_ci_upper=float(pf_ci_upper),
        sharpe_mean=float(np.mean(sharpe_samples)),
        sharpe_ci_lower=float(sharpe_ci_lower),
        sharpe_ci_upper=float(sharpe_ci_upper),
        n_iterations=n_iterations,
        confidence_level=confidence_level,
        go_decision=go_decision,
    )
