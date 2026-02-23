"""Pydantic v2 models for strategy.yaml schema."""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _default_russell_exchanges() -> list[Literal["NYSE", "NASDAQ"]]:
    return ["NYSE", "NASDAQ"]


class UniverseConfig(BaseModel):
    type: str
    index: str
    filters: list[str] = Field(default_factory=list)
    selection: Russell1000ApproxSelectionConfig = Field(
        default_factory=lambda: Russell1000ApproxSelectionConfig()
    )


class Russell1000ApproxSelectionConfig(BaseModel):
    """Explicit selection policy for the russell1000 approximate universe."""

    market_cap_more_than: int = Field(default=2_000_000_000, ge=0)
    exchanges: list[Literal["NYSE", "NASDAQ"]] = Field(
        default_factory=_default_russell_exchanges,
        min_length=1,
    )
    active_only: bool = True
    fetch_limit: int = Field(default=3_000, ge=100, le=10_000)
    top_n_by_market_cap: int = Field(default=1_000, ge=1, le=5_000)
    include_etf: bool = False
    include_fund: bool = False

    @model_validator(mode="after")
    def top_n_not_exceed_fetch_limit(self) -> Russell1000ApproxSelectionConfig:
        if self.top_n_by_market_cap > self.fetch_limit:
            raise ValueError("top_n_by_market_cap must be <= fetch_limit")
        return self


class DataConfig(BaseModel):
    timeframe: str = "daily"
    lookback_years: int = 8


class EntrySignal(BaseModel):
    type: str
    conditions: list[str] = Field(min_length=1)
    trend_filter: list[str] = Field(default_factory=list)


class ExitRules(BaseModel):
    stop_loss: str
    trailing_stop: str | None = None
    take_profit: str | None = None
    # Numeric parameters (parsed by engine)
    stop_loss_pct: float = Field(default=0.07, gt=0, le=0.30)
    take_profit_rr: float = Field(default=3.0, gt=0, le=20.0)
    breakeven_at_rr: float | None = Field(default=None, ge=0)
    trailing_start_bars: int = Field(default=0, ge=0)

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("stop_loss must not be empty")
        return v

    @model_validator(mode="after")
    def require_trailing_or_take_profit(self) -> ExitRules:
        if not self.trailing_stop and not self.take_profit:
            raise ValueError("exit rules must include trailing_stop or take_profit (or both)")
        return self


class SignalsConfig(BaseModel):
    entry: EntrySignal
    exit: ExitRules


class RiskConfig(BaseModel):
    position_sizing: str = "fixed_risk"
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.10)
    max_positions: int = Field(default=5, ge=1)
    max_sector_exposure: float = Field(default=0.30, gt=0, le=1.0)

    @field_validator("risk_per_trade")
    @classmethod
    def validate_risk_per_trade(cls, v: float) -> float:
        if v > 0.10:
            raise ValueError("risk_per_trade must not exceed 0.10 (10%)")
        return v


class CostModelConfig(BaseModel):
    commission_per_share: float = 0.0
    slippage_bps: int = 5


class BootstrapConfig(BaseModel):
    n_iterations: int = Field(default=10_000, ge=1_000, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = Field(default=42)


class OOSPromotionGates(BaseModel):
    min_sharpe: float = Field(default=0.5, ge=0)
    min_profit_factor: float = Field(default=1.0, gt=0)
    min_pf_ci_lower: float = Field(default=1.0, gt=0)
    min_sharpe_ci_lower: float = Field(default=0.0, ge=0)


class ValidationConfig(BaseModel):
    method: str = "full_sample"
    oos_ratio: float | None = Field(default=None, gt=0, lt=1.0)
    n_splits: int | None = Field(default=None, ge=2, le=20)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    oos_gates: OOSPromotionGates = Field(default_factory=OOSPromotionGates)

    @model_validator(mode="after")
    def walk_forward_fields_consistent(self) -> ValidationConfig:
        if self.method == "walk_forward" and (self.n_splits is None or self.oos_ratio is None):
            raise ValueError("walk_forward method requires both n_splits and oos_ratio")
        if (
            self.n_splits is not None
            and self.oos_ratio is not None
            and self.method != "walk_forward"
        ):
            raise ValueError("n_splits and oos_ratio require method='walk_forward'")
        return self


class PromotionGates(BaseModel):
    min_trades: int = 200
    max_drawdown: float = 0.15
    sharpe: float | None = None
    profit_factor: float = 1.2


class VCPDetectionConfig(BaseModel):
    min_contractions: int = 2
    contraction_ratio: float = 0.75
    lookback_window: int = 120
    swing_threshold: float = 0.05  # legacy; ATR zigzag is default
    volume_decline: bool = True
    breakout_volume_ratio: float = 1.5
    # Rolling detection params
    rolling_window_size: int = Field(default=250, ge=100)
    rolling_step_size: int = Field(default=20, ge=5)
    rolling_cooldown: int = Field(default=40, ge=10)
    # ATR zigzag params (screener migration)
    atr_multiplier: float = Field(default=1.5, gt=0)
    atr_period: int = Field(default=14, ge=5)
    min_contraction_days: int = Field(default=5, ge=1)
    t1_depth_min: float = Field(default=0.08, ge=0)
    t1_depth_max: float = Field(default=0.35, le=1.0)
    right_shoulder_pct: float = Field(default=0.05, ge=0)
    min_pattern_days: int = Field(default=15, ge=1)
    max_pattern_days: int = Field(default=325, ge=1)
    # Trend template param (pipeline wiring)
    trend_min_criteria: int = Field(default=6, ge=1, le=7)


class GapUpDetectionConfig(BaseModel):
    min_gap_pct: float = Field(default=0.04, gt=0, le=0.50)
    volume_ratio: float = Field(default=2.0, gt=0)
    avg_volume_window: int = Field(default=50, ge=10)
    max_entry_days: int = Field(default=10, ge=1, le=60)
    breakout_volume_ratio: float = Field(default=1.5, gt=0)
    max_stop_pct: float = Field(default=0.15, gt=0, le=0.30)
    # Entry quality filters (None = disabled)
    breakout_close_pos_min: float | None = Field(default=None, ge=0, le=1.0)
    breakout_max_extension_pct: float | None = Field(default=None, ge=0, le=0.50)
    breakout_min_body_atr: float | None = Field(default=None, ge=0)
    breakout_atr_pct_max: float | None = Field(default=None, gt=0, le=1.0)


class MarketRegimeConfig(BaseModel):
    enabled: bool = False
    breadth_sma_window: int = Field(default=200, ge=50, le=500)
    min_breadth_pct: float = Field(default=0.40, gt=0, lt=1.0)


class ScoringWeights(BaseModel):
    gap_size: float = Field(default=25.0, ge=0, le=100)
    # Legacy key name retained for backward compatibility; used as pre-gap trend weight.
    pre_earnings_trend: float = Field(default=30.0, ge=0, le=100)
    volume_trend: float = Field(default=20.0, ge=0, le=100)
    ma200_position: float = Field(default=15.0, ge=0, le=100)
    ma50_position: float = Field(default=10.0, ge=0, le=100)

    @model_validator(mode="after")
    def weights_sum_approximately_100(self) -> ScoringWeights:
        total = (
            self.gap_size
            + self.pre_earnings_trend
            + self.volume_trend
            + self.ma200_position
            + self.ma50_position
        )
        if abs(total - 100.0) > 0.1:
            raise ValueError(f"Scoring weights must sum to ~100, got {total}")
        return self


class EntryQualityFilter(BaseModel):
    """Entry quality filter — excludes historically weak profiles.

    exclude_price_band: stocks priced in [$low, $high) are EXCLUDED.
    risk combo: gap >= threshold% AND score >= threshold → EXCLUDED.
    """

    enabled: bool = True
    exclude_price_band_low: float = Field(default=10.0, ge=0)
    exclude_price_band_high: float = Field(default=30.0, gt=0)
    risk_gap_threshold: float = Field(default=10.0, ge=0)
    risk_score_threshold: float = Field(default=85.0, ge=0, le=100)
    # Trade-time momentum filter. If set, require:
    # close[T-1] / close[T-1-lookback_bars] - 1 >= month_perf_min
    month_perf_min: float | None = None
    month_perf_lookback_bars: int = Field(default=21, ge=1)

    @model_validator(mode="after")
    def band_high_gt_low(self) -> EntryQualityFilter:
        if self.exclude_price_band_high <= self.exclude_price_band_low:
            raise ValueError("exclude_price_band_high must be > exclude_price_band_low")
        return self


class WeeklyTrailingStopConfig(BaseModel):
    mode: Literal["weekly_ema", "weekly_nweek_low"] = "weekly_ema"
    ema_period: int = Field(default=10, ge=2, le=52)
    nweek_low_period: int = Field(default=4, ge=2, le=26)
    transition_weeks: int = Field(default=3, ge=0, le=12)


class ExecutionScheduleConfig(BaseModel):
    close_cycle_time_et: str = Field(default="16:15", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    open_cycle_time_et: str = Field(default="09:30", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class ExecutionGuardrailsConfig(BaseModel):
    allow_new_entries: bool = True
    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=0.20)
    max_order_slippage_bps: int = Field(default=50, ge=0, le=500)
    emergency_kill_switch: bool = True


class ExecutionPaperGatesConfig(BaseModel):
    min_paper_trades: int = Field(default=50, ge=0)
    max_reject_cancel_ratio: float = Field(default=0.20, ge=0, le=1.0)
    max_partial_fill_ratio: float = Field(default=0.60, ge=0, le=1.0)
    # None = skip this check (e.g., before first fills).
    max_realized_slippage_bps: float | None = Field(default=None, ge=0)
    max_backtest_deviation: float | None = Field(default=None, ge=0)
    min_profit_factor: float | None = Field(default=None, gt=0)
    max_drawdown: float | None = Field(default=None, ge=0, le=1.0)


class ExecutionConfig(BaseModel):
    enabled: bool = False
    mode: Literal["paper", "live"] = "paper"
    broker: Literal["alpaca"] = "alpaca"
    confirm_live: bool = False
    schedule: ExecutionScheduleConfig = Field(default_factory=ExecutionScheduleConfig)
    guardrails: ExecutionGuardrailsConfig = Field(default_factory=ExecutionGuardrailsConfig)
    paper_gates: ExecutionPaperGatesConfig = Field(default_factory=ExecutionPaperGatesConfig)

    @model_validator(mode="after")
    def validate_live_confirmation(self) -> ExecutionConfig:
        if self.enabled and self.mode == "live" and not self.confirm_live:
            raise ValueError("execution.mode='live' requires execution.confirm_live=true")
        return self


class GapOpenScoredConfig(BaseModel):
    """Scored gap-up open strategy config.

    Despite the legacy key name ``earnings_gap_open``, this config is NOT
    constrained by earnings-calendar events. It is a generic open-gap signal.

    stop_loss_pct here is the TRUTH SOURCE for this strategy.
    Pipeline wiring:
      - Signal stop_price = open[T] * (1 - stop_loss_pct)  (in gap_open_scored.py)
      - Engine stop_loss_pct = this value                    (in pipeline.py _run_backtest)
    signals.exit.stop_loss_pct in YAML should match but is only a descriptive fallback.
    """

    min_gap_pct: float = Field(default=0.04, gt=0, le=0.50)
    avg_volume_window: int = Field(default=50, ge=10)
    stop_loss_pct: float = Field(default=0.10, gt=0, le=0.30)
    min_grade: Literal["A", "B", "C", "D"] = "C"
    cooldown: int = Field(default=20, ge=1)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    entry_filter: EntryQualityFilter = Field(default_factory=EntryQualityFilter)
    weekly_trailing: WeeklyTrailingStopConfig = Field(default_factory=WeeklyTrailingStopConfig)


# Backward-compatible type alias; prefer GapOpenScoredConfig in new code.
EarningsGapOpenConfig = GapOpenScoredConfig


class BacktestConfig(BaseModel):
    stop_loss_pct: float = Field(default=0.07, gt=0, le=0.20)
    # Deprecated compatibility field. Truth source is signals.exit.take_profit_rr.
    take_profit_rr: float | None = Field(default=None, gt=0)
    breakeven_at_rr: float | None = Field(default=None, gt=0)
    trailing_ema_window: int = Field(default=21, ge=5, le=50)
    trailing_low_window: int = Field(default=10, ge=3, le=30)
    entry_mode: Literal["next_day_open", "same_bar_open"] = "next_day_open"
    trailing_mode: Literal["daily", "weekly"] = "daily"
    portfolio_mode: bool = False
    weekly_trailing: WeeklyTrailingStopConfig | None = None


class StrategySpec(BaseModel):
    id: str
    name: str
    description: str = ""

    universe: UniverseConfig
    data: DataConfig = Field(default_factory=DataConfig)
    signals: SignalsConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    promotion_gates: PromotionGates = Field(default_factory=PromotionGates)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    execution: ExecutionConfig | None = None
    vcp_detection: VCPDetectionConfig | None = None
    gap_up_detection: GapUpDetectionConfig | None = None
    # Preferred key.
    gap_open_scored: GapOpenScoredConfig | None = None
    # Legacy key kept for backward compatibility.
    earnings_gap_open: GapOpenScoredConfig | None = None
    market_regime: MarketRegimeConfig | None = None

    @model_validator(mode="after")
    def normalize_gap_open_aliases(self) -> StrategySpec:
        new_cfg = self.gap_open_scored
        legacy_cfg = self.earnings_gap_open
        if (
            new_cfg is not None
            and legacy_cfg is not None
            and new_cfg.model_dump() != legacy_cfg.model_dump()
        ):
            raise ValueError("gap_open_scored and earnings_gap_open are aliases; values must match")
        active = new_cfg or legacy_cfg
        self.gap_open_scored = active
        self.earnings_gap_open = active
        return self

    @model_validator(mode="after")
    def gap_configs_mutually_exclusive(self) -> StrategySpec:
        if self.gap_up_detection is not None and self.gap_open_scored is not None:
            raise ValueError(
                "gap_up_detection and gap_open_scored/earnings_gap_open are mutually "
                "exclusive — use one signal type per strategy"
            )
        return self

    @model_validator(mode="after")
    def take_profit_rr_consistent(self) -> StrategySpec:
        """Ensure duplicated take-profit RR fields do not silently diverge."""
        backtest_rr = self.backtest.take_profit_rr
        exit_rr = self.signals.exit.take_profit_rr
        if backtest_rr is not None:
            if abs(backtest_rr - exit_rr) > 1e-12:
                raise ValueError(
                    "backtest.take_profit_rr and signals.exit.take_profit_rr "
                    "must match when both are set"
                )
            warnings.warn(
                "backtest.take_profit_rr is deprecated; use signals.exit.take_profit_rr only.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self
