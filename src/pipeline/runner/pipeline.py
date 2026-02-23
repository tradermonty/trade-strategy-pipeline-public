"""Pipeline orchestrator — runs all stages in sequence."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from pipeline.backtest.engine import BacktestEngine, BacktestResult
from pipeline.data.storage import save_snapshot, write_ohlcv
from pipeline.evaluation.bootstrap import BootstrapResult, bootstrap_ci
from pipeline.evaluation.gates import GateResult, evaluate_gates, evaluate_oos_gates
from pipeline.evaluation.metrics import compute_metrics
from pipeline.reporting.charts import (
    generate_drawdown_chart,
    generate_equity_chart,
    generate_trade_distribution,
)
from pipeline.reporting.markdown import generate_report
from pipeline.signals.trend import compute_rs_ranks, trend_template
from pipeline.signals.vcp import detect_vcp_rolling
from pipeline.spec.schema import StrategySpec
from pipeline.spec.validator import validate_spec

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    strategy_id: str
    success: bool = True
    error: str | None = None
    metrics: dict[str, float] | None = None
    gate_result: GateResult | None = None
    backtest_result: BacktestResult | None = None
    bootstrap_result: BootstrapResult | None = None
    walk_forward_result: Any | None = None
    oos_bootstrap_result: BootstrapResult | None = None
    oos_gate_result: GateResult | None = None
    report_path: Path | None = None
    failed_stage: str | None = None
    failure_report_path: Path | None = None


class Pipeline:
    """Main pipeline orchestrator."""

    stages: list[str] = ["load_data", "generate_signals", "backtest", "evaluate", "report"]  # noqa: RUF012

    def __init__(
        self,
        spec_path: Path,
        data_dir: Path,
        output_dir: Path | None = None,
        data_profile: Literal["auto", "sample", "production"] = "auto",
        spec_stage: Literal["phase1", "phase1_statistical", "phase2"] | None = None,
        skip_validation: bool = False,
    ) -> None:
        with open(spec_path) as f:
            data = yaml.safe_load(f)
        self.spec_path = spec_path
        self.spec = StrategySpec(**data)
        self.data_dir = data_dir
        self.output_dir = output_dir or Path("reports") / self.spec.id
        self.data_profile = data_profile
        self.spec_stage = spec_stage or self._infer_spec_stage(self.spec)
        if not skip_validation:
            self._validate_runtime_spec()

    @staticmethod
    def _infer_spec_stage(
        spec: StrategySpec,
    ) -> Literal["phase1", "phase1_statistical", "phase2"]:
        """Infer validation stage from spec content."""
        if spec.execution is not None and spec.execution.enabled:
            return "phase2"
        if spec.validation.method == "walk_forward":
            return "phase1_statistical"
        return "phase1"

    def _validate_runtime_spec(self) -> None:
        """Validate strategy spec against the chosen stage."""
        errors = validate_spec(self.spec, stage=self.spec_stage)
        if errors:
            joined = "; ".join(errors)
            raise ValueError(
                f"Strategy spec failed runtime validation (stage='{self.spec_stage}'): {joined}"
            )

    def _resolve_data_path(self) -> Path | None:
        """Resolve the single OHLCV parquet path for this run."""
        sample_path = self.data_dir / "sample_ohlcv.parquet"
        production_path = self.data_dir / "ohlcv.parquet"
        parquet_files = sorted(p for p in self.data_dir.glob("*.parquet") if "sector" not in p.name)

        if self.data_profile == "sample":
            return sample_path if sample_path.exists() else None

        if self.data_profile == "production":
            if production_path.exists():
                return production_path
            non_sample = [p for p in parquet_files if p.name != sample_path.name]
            if len(non_sample) == 1:
                return non_sample[0]
            if len(non_sample) > 1:
                raise RuntimeError(
                    "Ambiguous production data source: multiple parquet files found. "
                    "Specify one file by keeping only a single production OHLCV parquet."
                )
            if sample_path.exists():
                raise RuntimeError(
                    "data_profile='production' was selected, but only sample_ohlcv.parquet exists. "
                    "Provide ohlcv.parquet or switch to data_profile='sample'."
                )
            return None

        # auto mode
        if sample_path.exists() and production_path.exists():
            raise RuntimeError(
                "Ambiguous data source: both sample_ohlcv.parquet and ohlcv.parquet exist. "
                "Set data_profile='sample' or 'production'."
            )
        if production_path.exists():
            return production_path
        if sample_path.exists():
            return sample_path
        if len(parquet_files) == 1:
            return parquet_files[0]
        if len(parquet_files) > 1:
            raise RuntimeError(
                "Ambiguous data source: multiple OHLCV parquet files found. "
                "Set data_profile or keep only one parquet file."
            )
        return None

    @staticmethod
    def _portfolio_mode_required_gate(base: GateResult, scope: Literal["is", "oos"]) -> GateResult:
        """Return a new gate result with portfolio-mode requirement appended."""
        failure_msg = (
            "OOS promotion gate requires backtest.portfolio_mode=true "
            "(non-portfolio results are diagnostic only)"
            if scope == "oos"
            else "Promotion gate requires backtest.portfolio_mode=true "
            "(non-portfolio results are diagnostic only)"
        )
        checks = [
            *list(base.checks),
            {
                "name": "portfolio_mode_required",
                "passed": False,
                # Keep numeric shape to avoid mixed bool/float formatting issues downstream.
                "actual": 0.0,
                "threshold": 1.0,
            },
        ]
        failures = [*list(base.failures), failure_msg]
        return GateResult(passed=False, checks=checks, failures=failures)

    def run(self) -> PipelineResult:
        """Run the full pipeline."""
        result = PipelineResult(strategy_id=self.spec.id)
        current_stage: str | None = None

        try:
            # Stage 1: Load data
            current_stage = "load_data"
            ohlcv = self._load_data()
            if ohlcv is None or ohlcv.empty:
                result.success = False
                result.error = "No OHLCV data found"
                result.failed_stage = current_stage
                result.failure_report_path = self._write_failure_report(current_stage, result.error)
                return result

            # Stage 2: Generate signals
            current_stage = "generate_signals"
            signals = self._generate_signals(ohlcv)

            # Stage 3: Backtest
            current_stage = "backtest"
            backtest_result = self._run_backtest(ohlcv, signals)
            result.backtest_result = backtest_result

            # Stage 4: Evaluate
            current_stage = "evaluate"
            metrics = compute_metrics(backtest_result.equity_curve, backtest_result.trades)
            result.metrics = metrics

            gate_config = self.spec.promotion_gates.model_dump()
            gate_result = evaluate_gates(metrics, gate_config)
            if not self.spec.backtest.portfolio_mode:
                gate_result = self._portfolio_mode_required_gate(gate_result, scope="is")
            result.gate_result = gate_result

            # Stage 4b: Bootstrap CI
            bs_config = self.spec.validation.bootstrap
            result.bootstrap_result = bootstrap_ci(
                trades=backtest_result.trades,
                n_iterations=bs_config.n_iterations,
                confidence_level=bs_config.confidence_level,
                rng_seed=bs_config.seed,
            )

            # Stage 4c: Walk-Forward (optional, spec-driven)
            val = self.spec.validation
            if val.method == "walk_forward":
                from pipeline.evaluation.walk_forward import run_walk_forward

                # model_validator guarantees both are set when method="walk_forward"
                assert val.n_splits is not None
                assert val.oos_ratio is not None
                wf = run_walk_forward(
                    pipeline=self,
                    ohlcv=ohlcv,
                    n_splits=val.n_splits,
                    oos_ratio=val.oos_ratio,
                    fail_on_short_warmup=True,
                )
                result.walk_forward_result = wf

                # OOS Bootstrap CI on pooled OOS trades
                if wf.all_oos_trades:
                    result.oos_bootstrap_result = bootstrap_ci(
                        trades=wf.all_oos_trades,
                        n_iterations=bs_config.n_iterations,
                        confidence_level=bs_config.confidence_level,
                        rng_seed=bs_config.seed,
                    )

                # OOS Gate evaluation
                if result.oos_bootstrap_result is not None:
                    if self.spec.backtest.portfolio_mode:
                        result.oos_gate_result = evaluate_oos_gates(
                            wf, result.oos_bootstrap_result, val.oos_gates
                        )
                    else:
                        result.oos_gate_result = self._portfolio_mode_required_gate(
                            GateResult(passed=True), scope="oos"
                        )

            # Stage 5: Report
            current_stage = "report"
            report_path = self._generate_report(
                metrics,
                gate_result,
                backtest_result,
                bootstrap_result=result.bootstrap_result,
                walk_forward_result=result.walk_forward_result,
                oos_bootstrap_result=result.oos_bootstrap_result,
                oos_gate_result=result.oos_gate_result,
            )
            result.report_path = report_path

        except Exception as e:
            logger.exception("Pipeline failed")
            result.success = False
            result.error = str(e)
            result.failed_stage = current_stage
            result.failure_report_path = self._write_failure_report(current_stage, result.error)

        return result

    def _failure_hint(self, stage: str | None, error: str) -> str | None:
        """Return short actionable guidance for common failure modes."""
        message = error.lower()
        if "indicator warmup" in message:
            return (
                "Warmup bars are insufficient. Reduce validation.oos_ratio or extend "
                "data.lookback_years/history."
            )
        if "ambiguous data source" in message:
            return (
                "Choose one OHLCV source with --data-profile sample|production "
                "or remove extra parquet files."
            )
        if "runtime validation" in message:
            return (
                "Spec does not match the required stage. Set --spec-stage explicitly "
                "or update strategy fields to match the stage rules."
            )
        if stage == "load_data":
            return "Place a valid OHLCV parquet in the data directory or run --fetch-data first."
        return None

    def _write_failure_report(self, stage: str | None, error: str) -> Path | None:
        """Write failure diagnostics to output_dir/failure.md."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.output_dir / "failure.md"
            generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            reproduce_cmd = (
                "uv run python -m pipeline.runner.cli "
                f"--strategy {self.spec.id} "
                f"--data-dir {self.data_dir} "
                f"--output-dir {self.output_dir} "
                f"--data-profile {self.data_profile} "
                f"--spec-stage {self.spec_stage}"
            )
            lines = [
                f"# Pipeline Failure Report: {self.spec.name}",
                "",
                f"**Strategy ID**: `{self.spec.id}`",
                f"**Spec Path**: `{self.spec_path}`",
                f"**Generated At**: {generated_at}",
                f"**Stage**: `{stage or 'unknown'}`",
                f"**Data Profile**: `{self.data_profile}`",
                f"**Spec Stage**: `{self.spec_stage}`",
                "",
                "## Reproduce",
                "",
                "```bash",
                reproduce_cmd,
                "```",
                "",
                "## Environment",
                "",
                f"- Python: `{sys.version.split()[0]}`",
                f"- Platform: `{platform.platform()}`",
                f"- CWD: `{Path.cwd()}`",
                "",
                "## Error",
                "",
                f"`{error}`",
                "",
            ]
            hint = self._failure_hint(stage, error)
            if hint:
                lines.extend(
                    [
                        "## Suggested Action",
                        "",
                        f"- {hint}",
                        "",
                    ]
                )
            report_path.write_text("\n".join(lines))
            return report_path
        except Exception:
            logger.warning("Failed to write failure report", exc_info=True)
            return None

    def fetch_data(self) -> Path:
        """Fetch OHLCV data from APIs and save to data directory.

        Uses AlpacaClient for price data and FmpClient for sector classification.
        Saves both a current file and a dated snapshot.
        Returns the path to the saved OHLCV parquet file.
        """
        from pipeline.data.alpaca_client import AlpacaClient
        from pipeline.data.fmp_client import FmpClient

        alpaca_key = os.environ.get("ALPACA_API_KEY", "")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        fmp_key = os.environ.get("FMP_API_KEY", "")

        if not alpaca_key or not alpaca_secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        alpaca = AlpacaClient(api_key=alpaca_key, secret_key=alpaca_secret)

        # Determine universe from spec
        universe = self.spec.universe
        selection_metadata: dict[str, Any] | None = None
        if universe.index == "sp500":
            tickers = alpaca.fetch_sp500_constituents()
        elif universe.index == "sp500+nasdaq100":
            sp500 = alpaca.fetch_sp500_constituents()
            ndx100 = alpaca.fetch_nasdaq100_constituents()
            tickers = sorted(set(sp500 + ndx100))
        elif universe.index == "russell1000":
            if not fmp_key:
                raise RuntimeError("FMP_API_KEY is required for russell1000 universe")
            fmp = FmpClient(api_key=fmp_key)
            selection = universe.selection
            tickers = fmp.fetch_russell1000_approx(
                market_cap_more_than=selection.market_cap_more_than,
                exchanges=selection.exchanges,
                active_only=selection.active_only,
                fetch_limit=selection.fetch_limit,
                top_n_by_market_cap=selection.top_n_by_market_cap,
                include_etf=selection.include_etf,
                include_fund=selection.include_fund,
            )
            selection_metadata = {
                "index": universe.index,
                "selection": selection.model_dump(),
                "requested_symbol_count": len(tickers),
                "requested_symbols": tickers,
                "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            logger.info(
                "russell1000_approx selection requested=%d (top_n=%d, fetch_limit=%d, "
                "include_etf=%s, include_fund=%s)",
                len(tickers),
                selection.top_n_by_market_cap,
                selection.fetch_limit,
                selection.include_etf,
                selection.include_fund,
            )
        else:
            raise RuntimeError(f"Unsupported universe index: {universe.index}")

        # Date range from spec
        end_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        lookback_years = self.spec.data.lookback_years
        start_date = (datetime.now(tz=UTC) - pd.Timedelta(days=lookback_years * 365)).strftime(
            "%Y-%m-%d"
        )

        logger.info("Fetching %d symbols from %s to %s", len(tickers), start_date, end_date)

        # Fetch bars for each symbol
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            try:
                df = alpaca.fetch_daily_bars(ticker, start=start_date, end=end_date)
                df["symbol"] = ticker
                frames.append(df)
                logger.info("Fetched %d bars for %s", len(df), ticker)
            except Exception:
                logger.warning("Failed to fetch %s, skipping", ticker, exc_info=True)

        if not frames:
            raise RuntimeError("No data fetched for any symbol")

        ohlcv = pd.concat(frames, ignore_index=True)
        fetched_symbols = sorted(set(ohlcv["symbol"].astype(str)))
        logger.info(
            "Fetched OHLCV symbols=%d (requested=%d, dropped=%d)",
            len(fetched_symbols),
            len(tickers),
            len(tickers) - len(fetched_symbols),
        )

        # Save to data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ohlcv_path = self.data_dir / "ohlcv.parquet"
        write_ohlcv(ohlcv, ohlcv_path)
        logger.info("Saved %d rows to %s", len(ohlcv), ohlcv_path)

        # Save dated snapshot
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        snapshot_dir = self.data_dir / "snapshots"
        save_snapshot(ohlcv, snapshot_dir, date=today)
        logger.info("Snapshot saved to %s/%s", snapshot_dir, today)

        # Save universe selection metadata for reproducibility.
        if selection_metadata is not None:
            selection_metadata["fetched_symbol_count"] = len(fetched_symbols)
            selection_metadata["fetched_symbols"] = fetched_symbols
            selection_metadata["dropped_symbol_count"] = len(tickers) - len(fetched_symbols)
            selection_metadata["dropped_symbols"] = sorted(set(tickers) - set(fetched_symbols))
            selection_path = self.data_dir / "universe_selection.json"
            selection_path.write_text(
                json.dumps(selection_metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("Saved universe selection metadata to %s", selection_path)

        # Fetch sector data if FMP key available
        if fmp_key:
            try:
                fmp = FmpClient(api_key=fmp_key)
                sectors = fmp.get_bulk_sectors(tickers[:50])  # Limit to avoid rate limits
                sector_path = self.data_dir / "sector.parquet"
                sectors.to_parquet(sector_path, index=False)
                logger.info("Saved sector data for %d tickers", len(sectors))
            except Exception:
                logger.warning("Failed to fetch sector data", exc_info=True)

        return ohlcv_path

    def _load_data(self) -> pd.DataFrame | None:
        """Load OHLCV data from data directory."""
        path = self._resolve_data_path()
        if path is None:
            return None
        logger.info("Loading OHLCV from %s (data_profile=%s)", path, self.data_profile)
        return pd.read_parquet(path)

    @staticmethod
    def compute_market_breadth(ohlcv: pd.DataFrame, sma_window: int = 200) -> pd.Series:
        """Compute daily % of symbols above their SMA(sma_window).

        Returns a Series indexed by timestamp with values in [0, 1].
        """
        records: list[tuple[pd.Timestamp, bool]] = []
        for symbol in ohlcv["symbol"].unique():
            sym = ohlcv[ohlcv["symbol"] == symbol].sort_values("timestamp")
            sma = sym["close"].rolling(sma_window, min_periods=sma_window).mean()
            above = sym["close"] > sma
            for ts, val in zip(sym["timestamp"].values, above.values, strict=True):
                records.append((ts, bool(val)))

        df = pd.DataFrame(records, columns=["timestamp", "above"])
        breadth = df.groupby("timestamp")["above"].mean()
        return breadth.sort_index()

    def _generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Generate entry signals for all symbols."""
        all_signals: list[pd.DataFrame] = []
        vcp_config = self.spec.vcp_detection

        # Pre-compute market breadth filter if enabled
        regime = self.spec.market_regime
        breadth_series: pd.Series | None = None
        if regime and regime.enabled:
            breadth_series = self.compute_market_breadth(ohlcv, regime.breadth_sma_window)

        # Pre-compute cross-sectional RS ranks once
        rs_ranks = compute_rs_ranks(ohlcv)

        for symbol in ohlcv["symbol"].unique():
            sym_data = ohlcv[ohlcv["symbol"] == symbol].reset_index(drop=True)
            n = len(sym_data)

            sig = pd.DataFrame(index=range(n))
            sig["timestamp"] = sym_data["timestamp"]
            sig["symbol"] = symbol
            sig["entry"] = False
            sig["stop_price"] = float("nan")
            sig["entry_at_open"] = False
            sig["score"] = 0.0

            # Apply 7-point trend template (with rolling RS rank)
            rs_rank_series = rs_ranks.get(symbol)
            if rs_rank_series is not None:
                # Align rolling rank to sym_data index via timestamp
                rs_rank_series = rs_rank_series.reindex(sym_data["timestamp"].values)
                rs_rank_series.index = sym_data.index
            tf = trend_template(
                sym_data["close"],
                sym_data["high"],
                sym_data["low"],
                rs_rank=rs_rank_series,
                min_criteria=vcp_config.trend_min_criteria if vcp_config else 6,
            )

            # Detect VCP (rolling) with high/low
            if vcp_config:
                breakout_indices = detect_vcp_rolling(
                    sym_data["high"],
                    sym_data["low"],
                    sym_data["close"],
                    sym_data["volume"],
                    window_size=vcp_config.rolling_window_size,
                    step_size=vcp_config.rolling_step_size,
                    cooldown=vcp_config.rolling_cooldown,
                    min_contractions=vcp_config.min_contractions,
                    contraction_ratio=vcp_config.contraction_ratio,
                    breakout_volume_ratio=vcp_config.breakout_volume_ratio,
                    atr_multiplier=vcp_config.atr_multiplier,
                    atr_period=vcp_config.atr_period,
                    min_contraction_days=vcp_config.min_contraction_days,
                    right_shoulder_pct=vcp_config.right_shoulder_pct,
                    t1_depth_min=vcp_config.t1_depth_min,
                    t1_depth_max=vcp_config.t1_depth_max,
                    min_pattern_days=vcp_config.min_pattern_days,
                    max_pattern_days=vcp_config.max_pattern_days,
                    volume_decline=vcp_config.volume_decline,
                    swing_threshold=vcp_config.swing_threshold,
                )

                for idx in breakout_indices:
                    if idx < n and tf.iloc[min(idx, len(tf) - 1)]:
                        sig.loc[idx, "entry"] = True

            # Detect gap-up continuation
            gap_config = self.spec.gap_up_detection
            if gap_config:
                from pipeline.signals.gap import detect_gap_up_rolling

                entries = detect_gap_up_rolling(
                    sym_data["open"],
                    sym_data["high"],
                    sym_data["low"],
                    sym_data["close"],
                    sym_data["volume"],
                    min_gap_pct=gap_config.min_gap_pct,
                    volume_ratio=gap_config.volume_ratio,
                    avg_volume_window=gap_config.avg_volume_window,
                    max_entry_days=gap_config.max_entry_days,
                    breakout_volume_ratio=gap_config.breakout_volume_ratio,
                    max_stop_pct=gap_config.max_stop_pct,
                    breakout_close_pos_min=gap_config.breakout_close_pos_min,
                    breakout_max_extension_pct=gap_config.breakout_max_extension_pct,
                    breakout_min_body_atr=gap_config.breakout_min_body_atr,
                    breakout_atr_pct_max=gap_config.breakout_atr_pct_max,
                )

                for idx, gap_day_low in entries:
                    if idx < n:
                        sig.loc[idx, "entry"] = True
                        sig.loc[idx, "stop_price"] = gap_day_low

            # Detect scored gap-up open (not earnings-calendar constrained).
            gap_open_config = self.spec.gap_open_scored
            if gap_open_config:
                from pipeline.signals.gap_open_scored import detect_gap_open_scored

                gap_open_entries = detect_gap_open_scored(
                    sym_data["open"],
                    sym_data["high"],
                    sym_data["low"],
                    sym_data["close"],
                    sym_data["volume"],
                    config=gap_open_config,
                )
                for entry in gap_open_entries:
                    idx = entry["idx"]
                    if idx < n:
                        sig.loc[idx, "entry"] = True
                        sig.loc[idx, "stop_price"] = entry["stop_price"]
                        sig.loc[idx, "entry_at_open"] = True
                        sig.loc[idx, "score"] = entry["score"]

            # Apply market breadth filter: suppress entries when breadth < threshold
            if breadth_series is not None and regime is not None:
                for idx in sig.index[sig["entry"]]:
                    ts = sig.loc[idx, "timestamp"]
                    if ts in breadth_series.index and breadth_series[ts] < regime.min_breadth_pct:
                        sig.loc[idx, "entry"] = False

            all_signals.append(sig)

        return pd.concat(all_signals, ignore_index=True)

    def _run_backtest(self, ohlcv: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """Run backtest for each symbol and aggregate results."""
        exit_rules = self.spec.signals.exit
        bt = self.spec.backtest

        # Weekly trailing parameters
        weekly_kwargs: dict[str, Any] = {}
        gap_open_config = self.spec.gap_open_scored
        if bt.trailing_mode == "weekly":
            if gap_open_config:
                wt = gap_open_config.weekly_trailing
            elif bt.weekly_trailing:
                wt = bt.weekly_trailing
            else:
                wt = None
            if wt:
                weekly_kwargs = {
                    "weekly_ema_period": wt.ema_period,
                    "weekly_nweek_low_period": wt.nweek_low_period,
                    "weekly_transition_weeks": wt.transition_weeks,
                    "weekly_trailing_type": wt.mode,
                }

        # GapOpenScoredConfig.stop_loss_pct is the truth source for this strategy;
        # fall back to signals.exit.stop_loss_pct for other strategies.
        engine_stop_loss_pct = (
            gap_open_config.stop_loss_pct if gap_open_config else exit_rules.stop_loss_pct
        )

        engine = BacktestEngine(
            initial_capital=25_000.0,
            commission_per_share=self.spec.cost_model.commission_per_share,
            slippage_bps=self.spec.cost_model.slippage_bps,
            max_positions=self.spec.risk.max_positions,
            risk_per_trade=self.spec.risk.risk_per_trade,
            stop_loss_pct=engine_stop_loss_pct,
            take_profit_rr=exit_rules.take_profit_rr,
            breakeven_at_rr=exit_rules.breakeven_at_rr,
            trailing_start_bars=exit_rules.trailing_start_bars,
            trailing_ema_window=bt.trailing_ema_window,
            trailing_low_window=bt.trailing_low_window,
            entry_mode=bt.entry_mode,
            trailing_mode=bt.trailing_mode,
            **weekly_kwargs,
        )

        # Portfolio mode: shared capital across all symbols
        if bt.portfolio_mode:
            return engine.run_portfolio(ohlcv, signals)

        # Per-symbol mode (VCP etc.)
        all_trades: list[dict[str, Any]] = []
        equity_curves: list[pd.Series] = []

        for symbol in ohlcv["symbol"].unique():
            sym_data = ohlcv[ohlcv["symbol"] == symbol].reset_index(drop=True)
            sym_signals = signals[signals["symbol"] == symbol].reset_index(drop=True)

            if len(sym_data) == 0:
                continue

            result = engine.run(sym_data, sym_signals)
            for trade in result.trades:
                trade["symbol"] = symbol
            all_trades.extend(result.trades)
            equity_curves.append(result.equity_curve)

        # Combine equity curves (simple average for now)
        if equity_curves:
            combined_equity = equity_curves[0].copy()
            for ec in equity_curves[1:]:
                aligned = ec.reindex(combined_equity.index, method="ffill")
                combined_equity = combined_equity + (aligned - 25_000.0)
        else:
            combined_equity = pd.Series([25_000.0])

        return BacktestResult(
            equity_curve=combined_equity,
            trades=all_trades,
            initial_capital=25_000.0,
            final_capital=float(combined_equity.iloc[-1]),
        )

    def _generate_report(
        self,
        metrics: dict[str, float],
        gate_result: GateResult,
        backtest_result: BacktestResult,
        bootstrap_result: BootstrapResult | None = None,
        walk_forward_result: Any | None = None,
        oos_bootstrap_result: BootstrapResult | None = None,
        oos_gate_result: GateResult | None = None,
    ) -> Path:
        """Generate charts and markdown report."""
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Generate charts
        generate_equity_chart(backtest_result.equity_curve, charts_dir / "equity.png")
        generate_drawdown_chart(backtest_result.equity_curve, charts_dir / "drawdown.png")
        if backtest_result.trades:
            generate_trade_distribution(backtest_result.trades, charts_dir / "trade_dist.png")

        # Generate markdown report
        report_path = self.output_dir / "report.md"
        report_data: dict[str, Any] = {
            "strategy_id": self.spec.id,
            "strategy_name": self.spec.name,
            "metrics": metrics,
            "gate_result": gate_result,
            "equity_chart": "charts/equity.png",
            "drawdown_chart": "charts/drawdown.png",
            "trade_distribution_chart": "charts/trade_dist.png",
            "bootstrap_result": bootstrap_result,
            "walk_forward_result": walk_forward_result,
            "oos_bootstrap_result": oos_bootstrap_result,
            "oos_gate_result": oos_gate_result,
            "diagnostic_only": not self.spec.backtest.portfolio_mode,
        }
        generate_report(report_data, report_path)

        return report_path
