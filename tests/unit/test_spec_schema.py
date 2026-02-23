"""Tests for strategy spec schema and validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from pipeline.spec.schema import (
    BacktestConfig,
    BootstrapConfig,
    CostModelConfig,
    EarningsGapOpenConfig,
    EntryQualityFilter,
    EntrySignal,
    ExecutionConfig,
    ExecutionGuardrailsConfig,
    ExecutionPaperGatesConfig,
    ExecutionScheduleConfig,
    ExitRules,
    GapOpenScoredConfig,
    GapUpDetectionConfig,
    MarketRegimeConfig,
    OOSPromotionGates,
    PromotionGates,
    Russell1000ApproxSelectionConfig,
    ScoringWeights,
    StrategySpec,
    ValidationConfig,
    VCPDetectionConfig,
    WeeklyTrailingStopConfig,
)
from pipeline.spec.validator import validate_spec

STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies" / "examples"
VCP_YAML = STRATEGIES_DIR / "demo_vcp" / "strategy.yaml"


def _make_valid_spec_dict() -> dict[str, Any]:
    """Return a minimal valid spec dictionary for testing."""
    return {
        "id": "test_strategy_v1",
        "name": "Test Strategy",
        "description": "A test strategy",
        "universe": {"type": "us_equities", "index": "sp500", "filters": ["price > 10"]},
        "data": {"timeframe": "daily", "lookback_years": 8},
        "signals": {
            "entry": {
                "type": "pivot_breakout",
                "conditions": ["vcp_pattern_detected"],
                "trend_filter": ["price > sma_200"],
            },
            "exit": {
                "stop_loss": "7% below entry",
                "trailing_stop": "below 21-day EMA or 10-day low",
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
            "sharpe": 1.0,
            "profit_factor": 1.2,
        },
    }


def test_valid_vcp_spec_parses() -> None:
    """A complete VCP YAML should parse successfully."""
    spec_dict = _make_valid_spec_dict()
    spec = StrategySpec(**spec_dict)
    assert spec.id == "test_strategy_v1"
    assert spec.universe.index == "sp500"
    assert spec.risk.max_positions == 5


def test_russell1000_selection_defaults() -> None:
    """Russell1000 approx selection should have explicit defaults."""
    c = Russell1000ApproxSelectionConfig()
    assert c.market_cap_more_than == 2_000_000_000
    assert c.exchanges == ["NYSE", "NASDAQ"]
    assert c.fetch_limit == 3000
    assert c.top_n_by_market_cap == 1000
    assert c.include_etf is False
    assert c.include_fund is False


def test_russell1000_selection_top_n_validation() -> None:
    """top_n_by_market_cap must not exceed fetch_limit."""
    with pytest.raises(ValidationError, match="top_n_by_market_cap"):
        Russell1000ApproxSelectionConfig(fetch_limit=500, top_n_by_market_cap=1000)


def test_missing_required_field_raises() -> None:
    """Omitting id or universe should raise ValidationError."""
    d = _make_valid_spec_dict()
    del d["id"]
    with pytest.raises(ValidationError):
        StrategySpec(**d)

    d = _make_valid_spec_dict()
    del d["universe"]
    with pytest.raises(ValidationError):
        StrategySpec(**d)


def test_invalid_risk_per_trade_range() -> None:
    """risk_per_trade > 0.10 should be rejected."""
    d = _make_valid_spec_dict()
    d["risk"]["risk_per_trade"] = 0.15
    with pytest.raises(ValidationError, match="risk_per_trade"):
        StrategySpec(**d)


def test_promotion_gates_model() -> None:
    """Promotion gates should validate IS metrics (sharpe, profit_factor)."""
    gates = PromotionGates(min_trades=200, max_drawdown=0.15, sharpe=1.0, profit_factor=1.2)
    assert gates.sharpe == 1.0
    assert gates.profit_factor == 1.2
    assert gates.min_trades == 200


def test_promotion_gates_sharpe_optional() -> None:
    """PromotionGates.sharpe=None should be valid (skip Sharpe gate)."""
    gates = PromotionGates(min_trades=200, max_drawdown=0.15, profit_factor=1.0)
    assert gates.sharpe is None


def test_cost_model_defaults() -> None:
    """Default cost model: commission=0.0, slippage=5bps."""
    cost = CostModelConfig()
    assert cost.commission_per_share == 0.0
    assert cost.slippage_bps == 5


def test_entry_signal_conditions_list() -> None:
    """Entry signal conditions must be a non-empty list."""
    with pytest.raises(ValidationError):
        EntrySignal(type="pivot_breakout", conditions=[], trend_filter=["price > sma_200"])


def test_exit_rules_present() -> None:
    """Exit rules must have stop_loss + (trailing_stop or take_profit)."""
    # Valid: all three
    rules = ExitRules(
        stop_loss="7% below entry",
        trailing_stop="below 21-day EMA",
        take_profit="risk_reward_3x",
    )
    assert rules.stop_loss == "7% below entry"

    # Invalid: no stop_loss
    with pytest.raises(ValidationError):
        ExitRules(stop_loss="", trailing_stop="below 21-day EMA")


def test_validator_rejects_phase1_oos() -> None:
    """Phase I should reject walk_forward validation method."""
    d = _make_valid_spec_dict()
    d["validation"]["method"] = "walk_forward"
    d["validation"]["n_splits"] = 5
    d["validation"]["oos_ratio"] = 0.30
    spec = StrategySpec(**d)
    errors = validate_spec(spec, stage="phase1")
    assert any("walk_forward" in e for e in errors)


def test_spec_to_dict_roundtrip() -> None:
    """model_dump() → reconstruct should match original."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    dumped = spec.model_dump()
    spec2 = StrategySpec(**dumped)
    assert spec == spec2


def test_load_spec_from_yaml_file() -> None:
    """Loading strategy.yaml from file should produce valid StrategySpec."""
    assert VCP_YAML.exists(), f"Strategy YAML not found: {VCP_YAML}"
    with open(VCP_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.id == "demo_vcp"


# --- GapUpDetectionConfig tests ---

GAP_YAML = STRATEGIES_DIR / "demo_gap_basic" / "strategy.yaml"


def test_gap_up_detection_defaults() -> None:
    """GapUpDetectionConfig should have sensible defaults."""
    config = GapUpDetectionConfig()
    assert config.min_gap_pct == 0.04
    assert config.volume_ratio == 2.0
    assert config.avg_volume_window == 50
    assert config.max_entry_days == 10


def test_gap_up_detection_validation() -> None:
    """GapUpDetectionConfig should reject out-of-range values."""
    with pytest.raises(ValidationError):
        GapUpDetectionConfig(min_gap_pct=0)  # gt=0 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(min_gap_pct=0.60)  # le=0.50 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(avg_volume_window=5)  # ge=10 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(max_entry_days=0)  # ge=1 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(max_entry_days=61)  # le=60 fails


def test_strategy_spec_with_gap_up_detection() -> None:
    """StrategySpec should accept gap_up_detection field."""
    d = _make_valid_spec_dict()
    d["gap_up_detection"] = {"min_gap_pct": 0.05, "volume_ratio": 2.5}
    spec = StrategySpec(**d)
    assert spec.gap_up_detection is not None
    assert spec.gap_up_detection.min_gap_pct == 0.05
    assert spec.gap_up_detection.volume_ratio == 2.5
    # Defaults still apply
    assert spec.gap_up_detection.avg_volume_window == 50


def test_strategy_spec_gap_up_none_by_default() -> None:
    """StrategySpec without gap_up_detection should have None."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    assert spec.gap_up_detection is None


def test_load_gap_up_spec_from_yaml() -> None:
    """Loading gap-up strategy YAML should produce valid StrategySpec."""
    assert GAP_YAML.exists(), f"Strategy YAML not found: {GAP_YAML}"
    with open(GAP_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.id == "demo_gap_basic"
    assert spec.gap_up_detection is not None
    assert spec.gap_up_detection.min_gap_pct == 0.06
    assert spec.vcp_detection is None


# --- BacktestConfig tests ---


def test_backtest_config_defaults() -> None:
    """BacktestConfig should have sensible defaults matching old hardcoded values."""
    config = BacktestConfig()
    assert config.stop_loss_pct == 0.07
    assert config.take_profit_rr is None
    assert config.breakeven_at_rr is None
    assert config.trailing_ema_window == 21
    assert config.trailing_low_window == 10


def test_backtest_config_validation() -> None:
    """BacktestConfig should reject out-of-range values."""
    with pytest.raises(ValidationError):
        BacktestConfig(stop_loss_pct=0)  # gt=0 fails

    with pytest.raises(ValidationError):
        BacktestConfig(stop_loss_pct=0.25)  # le=0.20 fails

    with pytest.raises(ValidationError):
        BacktestConfig(trailing_ema_window=4)  # ge=5 fails

    with pytest.raises(ValidationError):
        BacktestConfig(trailing_ema_window=51)  # le=50 fails

    with pytest.raises(ValidationError):
        BacktestConfig(trailing_low_window=2)  # ge=3 fails

    with pytest.raises(ValidationError):
        BacktestConfig(trailing_low_window=31)  # le=30 fails


def test_strategy_spec_backtest_defaults() -> None:
    """StrategySpec without backtest section should use BacktestConfig defaults."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    assert spec.backtest.stop_loss_pct == 0.07
    assert spec.backtest.take_profit_rr is None
    assert spec.backtest.breakeven_at_rr is None


def test_strategy_spec_backtest_custom() -> None:
    """StrategySpec should accept custom backtest config."""
    d = _make_valid_spec_dict()
    d["signals"]["exit"]["take_profit_rr"] = 2.0
    d["backtest"] = {
        "stop_loss_pct": 0.05,
        "take_profit_rr": 2.0,
        "breakeven_at_rr": 1.0,
        "trailing_ema_window": 15,
        "trailing_low_window": 7,
    }
    with pytest.warns(DeprecationWarning, match="deprecated"):
        spec = StrategySpec(**d)
    assert spec.backtest.stop_loss_pct == 0.05
    assert spec.backtest.take_profit_rr == 2.0
    assert spec.backtest.breakeven_at_rr == 1.0
    assert spec.backtest.trailing_ema_window == 15


def test_strategy_spec_take_profit_rr_mismatch_rejected() -> None:
    """Duplicated take_profit_rr fields must match when both are set."""
    d = _make_valid_spec_dict()
    d["signals"]["exit"]["take_profit_rr"] = 3.0
    d["backtest"] = {"take_profit_rr": 2.0}
    with pytest.raises(ValidationError, match="take_profit_rr"):
        StrategySpec(**d)


def test_vcp_yaml_backward_compatible() -> None:
    """VCP YAML with backtest.trailing_mode=weekly should parse correctly."""
    assert VCP_YAML.exists()
    with open(VCP_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.backtest.trailing_mode == "weekly"


def test_gap_up_yaml_has_backtest_config() -> None:
    """Gap-up YAML with backtest section should parse correctly."""
    assert GAP_YAML.exists()
    with open(GAP_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.backtest.breakeven_at_rr is None
    assert spec.backtest.trailing_ema_window == 21
    assert spec.backtest.trailing_low_window == 10


# --- GapUpDetectionConfig extended fields tests ---


def test_gap_up_detection_extended_defaults() -> None:
    """GapUpDetectionConfig should have breakout_volume_ratio and max_stop_pct."""
    config = GapUpDetectionConfig()
    assert config.breakout_volume_ratio == 1.5
    assert config.max_stop_pct == 0.15


def test_gap_up_detection_max_stop_pct_validation() -> None:
    """max_stop_pct should reject values > 0.30."""
    with pytest.raises(ValidationError):
        GapUpDetectionConfig(max_stop_pct=0.35)  # le=0.30 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(max_stop_pct=0)  # gt=0 fails


# --- Entry quality filter fields ---


def test_gap_up_entry_quality_filters_none_by_default() -> None:
    """Entry quality filters should be None (disabled) by default."""
    config = GapUpDetectionConfig()
    assert config.breakout_close_pos_min is None
    assert config.breakout_max_extension_pct is None
    assert config.breakout_min_body_atr is None
    assert config.breakout_atr_pct_max is None


def test_gap_up_entry_quality_filters_valid() -> None:
    """Entry quality filters should accept valid values."""
    config = GapUpDetectionConfig(
        breakout_close_pos_min=0.6,
        breakout_max_extension_pct=0.05,
        breakout_min_body_atr=0.5,
        breakout_atr_pct_max=0.08,
    )
    assert config.breakout_close_pos_min == 0.6
    assert config.breakout_max_extension_pct == 0.05
    assert config.breakout_min_body_atr == 0.5
    assert config.breakout_atr_pct_max == 0.08


def test_gap_up_entry_quality_filters_validation() -> None:
    """Entry quality filters should reject out-of-range values."""
    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_close_pos_min=-0.1)  # ge=0 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_close_pos_min=1.1)  # le=1.0 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_max_extension_pct=-0.01)  # ge=0 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_max_extension_pct=0.6)  # le=0.50 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_min_body_atr=-0.1)  # ge=0 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_atr_pct_max=0)  # gt=0 fails

    with pytest.raises(ValidationError):
        GapUpDetectionConfig(breakout_atr_pct_max=1.1)  # le=1.0 fails


def test_gap_up_v1_yaml_backward_compatible_with_quality_filters() -> None:
    """v1 YAML without quality filter fields should still parse."""
    with open(GAP_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.gap_up_detection is not None
    assert spec.gap_up_detection.breakout_close_pos_min is None
    assert spec.gap_up_detection.breakout_max_extension_pct is None


# --- VCPDetectionConfig trend_min_criteria tests ---


def test_vcp_trend_min_criteria_default() -> None:
    """VCPDetectionConfig.trend_min_criteria defaults to 6."""
    config = VCPDetectionConfig()
    assert config.trend_min_criteria == 6


def test_vcp_trend_min_criteria_custom() -> None:
    """trend_min_criteria can be set to valid values 1-7."""
    config = VCPDetectionConfig(trend_min_criteria=7)
    assert config.trend_min_criteria == 7

    config = VCPDetectionConfig(trend_min_criteria=1)
    assert config.trend_min_criteria == 1


def test_vcp_trend_min_criteria_validation() -> None:
    """trend_min_criteria must be between 1 and 7."""
    with pytest.raises(ValidationError):
        VCPDetectionConfig(trend_min_criteria=0)  # ge=1 fails

    with pytest.raises(ValidationError):
        VCPDetectionConfig(trend_min_criteria=8)  # le=7 fails


def test_strategy_spec_with_trend_min_criteria() -> None:
    """StrategySpec should accept vcp_detection.trend_min_criteria."""
    d = _make_valid_spec_dict()
    d["vcp_detection"] = {"trend_min_criteria": 7}
    spec = StrategySpec(**d)
    assert spec.vcp_detection is not None
    assert spec.vcp_detection.trend_min_criteria == 7


# --- BootstrapConfig tests ---


def test_bootstrap_config_defaults() -> None:
    """BootstrapConfig should have sensible defaults."""
    config = BootstrapConfig()
    assert config.n_iterations == 10_000
    assert config.confidence_level == 0.95
    assert config.seed == 42


def test_bootstrap_config_validation() -> None:
    """BootstrapConfig should reject out-of-range values."""
    with pytest.raises(ValidationError):
        BootstrapConfig(n_iterations=500)  # ge=1_000 fails

    with pytest.raises(ValidationError):
        BootstrapConfig(n_iterations=200_000)  # le=100_000 fails

    with pytest.raises(ValidationError):
        BootstrapConfig(confidence_level=0.5)  # gt=0.5 fails

    with pytest.raises(ValidationError):
        BootstrapConfig(confidence_level=1.0)  # lt=1.0 fails


def test_validation_config_has_bootstrap_default() -> None:
    """ValidationConfig should have BootstrapConfig by default."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    assert spec.validation.bootstrap is not None
    assert spec.validation.bootstrap.n_iterations == 10_000


def test_validator_allows_walk_forward_in_statistical_stage() -> None:
    """phase1_statistical stage should allow walk_forward and oos_ratio."""
    d = _make_valid_spec_dict()
    d["validation"]["method"] = "walk_forward"
    d["validation"]["n_splits"] = 5
    d["validation"]["oos_ratio"] = 0.30
    spec = StrategySpec(**d)
    errors = validate_spec(spec, stage="phase1_statistical")
    assert errors == []


# --- ValidationConfig cross-field validation tests ---


def test_validation_walk_forward_requires_splits() -> None:
    """method='walk_forward' without n_splits/oos_ratio should raise."""
    with pytest.raises(ValidationError, match="walk_forward method requires"):
        ValidationConfig(method="walk_forward")

    with pytest.raises(ValidationError, match="walk_forward method requires"):
        ValidationConfig(method="walk_forward", n_splits=5)

    with pytest.raises(ValidationError, match="walk_forward method requires"):
        ValidationConfig(method="walk_forward", oos_ratio=0.30)


def test_validation_splits_require_walk_forward() -> None:
    """n_splits + oos_ratio without method='walk_forward' should raise."""
    with pytest.raises(ValidationError, match="n_splits and oos_ratio require"):
        ValidationConfig(n_splits=5, oos_ratio=0.30)


# --- MarketRegimeConfig tests ---


def test_market_regime_config_defaults() -> None:
    """MarketRegimeConfig should have sensible defaults."""
    config = MarketRegimeConfig()
    assert config.enabled is False
    assert config.breadth_sma_window == 200
    assert config.min_breadth_pct == 0.40


def test_market_regime_config_validation() -> None:
    """MarketRegimeConfig should reject out-of-range values."""
    with pytest.raises(ValidationError):
        MarketRegimeConfig(breadth_sma_window=30)  # ge=50 fails

    with pytest.raises(ValidationError):
        MarketRegimeConfig(breadth_sma_window=600)  # le=500 fails

    with pytest.raises(ValidationError):
        MarketRegimeConfig(min_breadth_pct=0.0)  # gt=0 fails

    with pytest.raises(ValidationError):
        MarketRegimeConfig(min_breadth_pct=1.0)  # lt=1.0 fails


def test_strategy_spec_market_regime_none_by_default() -> None:
    """StrategySpec without market_regime should have None."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    assert spec.market_regime is None


def test_strategy_spec_with_market_regime() -> None:
    """StrategySpec should accept market_regime field."""
    d = _make_valid_spec_dict()
    d["market_regime"] = {"enabled": True, "breadth_sma_window": 200, "min_breadth_pct": 0.35}
    spec = StrategySpec(**d)
    assert spec.market_regime is not None
    assert spec.market_regime.enabled is True
    assert spec.market_regime.min_breadth_pct == 0.35


# --- ExecutionConfig tests ---


def test_execution_config_defaults() -> None:
    """ExecutionConfig should have safe defaults for backward compatibility."""
    c = ExecutionConfig()
    assert c.enabled is False
    assert c.mode == "paper"
    assert c.broker == "alpaca"
    assert c.confirm_live is False
    assert isinstance(c.schedule, ExecutionScheduleConfig)
    assert isinstance(c.guardrails, ExecutionGuardrailsConfig)
    assert isinstance(c.paper_gates, ExecutionPaperGatesConfig)
    assert c.paper_gates.min_paper_trades == 50


def test_execution_paper_gates_defaults_and_validation() -> None:
    """ExecutionPaperGatesConfig should validate ratios and optional thresholds."""
    g = ExecutionPaperGatesConfig()
    assert g.max_reject_cancel_ratio == 0.20
    assert g.max_realized_slippage_bps is None

    with pytest.raises(ValidationError):
        ExecutionPaperGatesConfig(max_partial_fill_ratio=1.5)

    with pytest.raises(ValidationError):
        ExecutionPaperGatesConfig(min_profit_factor=0.0)


def test_execution_schedule_time_format_validation() -> None:
    """Execution schedule times must be HH:MM (24h)."""
    with pytest.raises(ValidationError):
        ExecutionScheduleConfig(close_cycle_time_et="4:15")  # missing leading zero

    with pytest.raises(ValidationError):
        ExecutionScheduleConfig(open_cycle_time_et="24:00")  # invalid hour


def test_execution_live_requires_confirm_live() -> None:
    """Live mode must require explicit confirmation when execution is enabled."""
    with pytest.raises(ValidationError, match="confirm_live"):
        ExecutionConfig(enabled=True, mode="live", confirm_live=False)

    c = ExecutionConfig(enabled=True, mode="live", confirm_live=True)
    assert c.mode == "live"
    assert c.confirm_live is True


def test_strategy_spec_execution_none_by_default() -> None:
    """StrategySpec without execution should keep execution=None."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    assert spec.execution is None


def test_strategy_spec_with_execution_paper() -> None:
    """StrategySpec should accept execution block in paper mode."""
    d = _make_valid_spec_dict()
    d["execution"] = {
        "enabled": True,
        "mode": "paper",
        "broker": "alpaca",
        "schedule": {"close_cycle_time_et": "16:15", "open_cycle_time_et": "09:30"},
        "guardrails": {"max_daily_loss_pct": 0.02, "max_order_slippage_bps": 40},
    }
    spec = StrategySpec(**d)
    assert spec.execution is not None
    assert spec.execution.enabled is True
    assert spec.execution.mode == "paper"
    assert spec.execution.guardrails.max_daily_loss_pct == 0.02


def test_strategy_spec_execution_roundtrip() -> None:
    """ExecutionConfig should survive StrategySpec dump/reload roundtrip."""
    d = _make_valid_spec_dict()
    d["execution"] = {
        "enabled": True,
        "mode": "live",
        "broker": "alpaca",
        "confirm_live": True,
    }
    spec = StrategySpec(**d)
    dumped = spec.model_dump()
    spec2 = StrategySpec(**dumped)
    assert spec == spec2


# --- OOSPromotionGates tests ---


def test_oos_promotion_gates_defaults() -> None:
    """OOSPromotionGates should have sensible defaults."""
    gates = OOSPromotionGates()
    assert gates.min_sharpe == 0.5
    assert gates.min_profit_factor == 1.0
    assert gates.min_pf_ci_lower == 1.0
    assert gates.min_sharpe_ci_lower == 0.0


def test_oos_promotion_gates_validation() -> None:
    """OOSPromotionGates should reject out-of-range values."""
    with pytest.raises(ValidationError):
        OOSPromotionGates(min_sharpe=-1.0)  # ge=0 fails

    with pytest.raises(ValidationError):
        OOSPromotionGates(min_profit_factor=0)  # gt=0 fails

    with pytest.raises(ValidationError):
        OOSPromotionGates(min_pf_ci_lower=0)  # gt=0 fails

    with pytest.raises(ValidationError):
        OOSPromotionGates(min_sharpe_ci_lower=-0.1)  # ge=0 fails


def test_validation_config_has_oos_gates() -> None:
    """ValidationConfig should have oos_gates with defaults."""
    config = ValidationConfig()
    assert config.oos_gates is not None
    assert config.oos_gates.min_sharpe == 0.5
    assert config.oos_gates.min_profit_factor == 1.0


def test_gap_yaml_with_walk_forward() -> None:
    """Updated gap-up YAML with walk_forward + oos_gates should parse correctly."""
    assert GAP_YAML.exists()
    with open(GAP_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.validation.method == "walk_forward"
    assert spec.validation.n_splits == 5
    assert spec.validation.oos_ratio == 0.30
    assert spec.validation.oos_gates.min_sharpe == 1.0
    assert spec.validation.oos_gates.min_profit_factor == 1.2
    assert spec.validation.oos_gates.min_pf_ci_lower == 1.0
    assert spec.validation.oos_gates.min_sharpe_ci_lower == 0.0


# --- EarningsGapOpenConfig tests ---


def test_scoring_weights_defaults() -> None:
    """ScoringWeights should sum to 100 by default."""
    w = ScoringWeights()
    assert w.gap_size == 25.0
    assert w.pre_earnings_trend == 30.0
    total = w.gap_size + w.pre_earnings_trend + w.volume_trend + w.ma200_position + w.ma50_position
    assert total == pytest.approx(100.0)


def test_scoring_weights_must_sum_100() -> None:
    """ScoringWeights that don't sum to ~100 should raise."""
    with pytest.raises(ValidationError, match="sum to ~100"):
        ScoringWeights(gap_size=50.0, pre_earnings_trend=50.0, volume_trend=20.0)


def test_entry_quality_filter_defaults() -> None:
    """EntryQualityFilter should have sensible defaults."""
    f = EntryQualityFilter()
    assert f.enabled is True
    assert f.exclude_price_band_low == 10.0
    assert f.exclude_price_band_high == 30.0
    assert f.month_perf_min is None
    assert f.month_perf_lookback_bars == 21


def test_entry_quality_filter_band_validation() -> None:
    """exclude_price_band_high must be > exclude_price_band_low."""
    with pytest.raises(ValidationError, match="exclude_price_band_high"):
        EntryQualityFilter(exclude_price_band_low=30.0, exclude_price_band_high=10.0)

    with pytest.raises(ValidationError, match="exclude_price_band_high"):
        EntryQualityFilter(exclude_price_band_low=20.0, exclude_price_band_high=20.0)


def test_entry_quality_filter_month_lookback_validation() -> None:
    """month_perf_lookback_bars must be >= 1 when provided."""
    with pytest.raises(ValidationError, match="month_perf_lookback_bars"):
        EntryQualityFilter(month_perf_min=0.0, month_perf_lookback_bars=0)


def test_weekly_trailing_stop_config_defaults() -> None:
    """WeeklyTrailingStopConfig defaults."""
    c = WeeklyTrailingStopConfig()
    assert c.mode == "weekly_ema"
    assert c.ema_period == 10
    assert c.nweek_low_period == 4
    assert c.transition_weeks == 3


def test_weekly_trailing_stop_config_literal() -> None:
    """WeeklyTrailingStopConfig.mode must be a valid literal."""
    with pytest.raises(ValidationError):
        WeeklyTrailingStopConfig(mode="invalid_mode")  # type: ignore[arg-type]


def test_earnings_gap_open_config_defaults() -> None:
    """EarningsGapOpenConfig should have sensible defaults."""
    c = EarningsGapOpenConfig()
    assert c.min_gap_pct == 0.04
    assert c.stop_loss_pct == 0.10
    assert c.min_grade == "C"
    assert c.cooldown == 20
    assert isinstance(c.scoring, ScoringWeights)
    assert isinstance(c.entry_filter, EntryQualityFilter)
    assert isinstance(c.weekly_trailing, WeeklyTrailingStopConfig)


def test_earnings_gap_open_min_grade_literal() -> None:
    """min_grade must be A/B/C/D."""
    with pytest.raises(ValidationError):
        EarningsGapOpenConfig(min_grade="E")  # type: ignore[arg-type]


def test_earnings_gap_open_validation() -> None:
    """EarningsGapOpenConfig should reject out-of-range values."""
    with pytest.raises(ValidationError):
        EarningsGapOpenConfig(min_gap_pct=0)  # gt=0 fails

    with pytest.raises(ValidationError):
        EarningsGapOpenConfig(stop_loss_pct=0.35)  # le=0.30 fails


def test_gap_open_scored_config_defaults() -> None:
    """GapOpenScoredConfig should share defaults with legacy alias."""
    c = GapOpenScoredConfig()
    assert c.min_gap_pct == 0.04
    assert c.stop_loss_pct == 0.10
    assert c.min_grade == "C"


def test_strategy_spec_earnings_gap_open_none_by_default() -> None:
    """StrategySpec without earnings_gap_open should have None."""
    d = _make_valid_spec_dict()
    spec = StrategySpec(**d)
    assert spec.earnings_gap_open is None


def test_strategy_spec_with_earnings_gap_open() -> None:
    """StrategySpec should accept earnings_gap_open field."""
    d = _make_valid_spec_dict()
    d["earnings_gap_open"] = {"min_gap_pct": 0.05, "stop_loss_pct": 0.10}
    spec = StrategySpec(**d)
    assert spec.earnings_gap_open is not None
    assert spec.gap_open_scored is not None
    assert spec.earnings_gap_open.min_gap_pct == 0.05


def test_strategy_spec_with_gap_open_scored() -> None:
    """StrategySpec should accept gap_open_scored field."""
    d = _make_valid_spec_dict()
    d["gap_open_scored"] = {"min_gap_pct": 0.05, "stop_loss_pct": 0.10}
    spec = StrategySpec(**d)
    assert spec.gap_open_scored is not None
    assert spec.earnings_gap_open is not None
    assert spec.gap_open_scored.min_gap_pct == 0.05


def test_strategy_spec_alias_mismatch_rejected() -> None:
    """If both alias keys are present, values must match."""
    d = _make_valid_spec_dict()
    d["gap_open_scored"] = {"min_gap_pct": 0.05}
    d["earnings_gap_open"] = {"min_gap_pct": 0.04}
    with pytest.raises(ValidationError, match="aliases; values must match"):
        StrategySpec(**d)


def test_gap_configs_mutually_exclusive() -> None:
    """gap_up_detection and earnings_gap_open cannot both be defined."""
    d = _make_valid_spec_dict()
    d["gap_up_detection"] = {"min_gap_pct": 0.05}
    d["earnings_gap_open"] = {"min_gap_pct": 0.04}
    with pytest.raises(ValidationError, match="mutually exclusive"):
        StrategySpec(**d)


def test_earnings_gap_open_roundtrip() -> None:
    """model_dump → reconstruct should match original."""
    d = _make_valid_spec_dict()
    d["earnings_gap_open"] = {
        "min_gap_pct": 0.05,
        "stop_loss_pct": 0.10,
        "min_grade": "B",
        "scoring": {
            "gap_size": 20.0,
            "pre_earnings_trend": 30.0,
            "volume_trend": 25.0,
            "ma200_position": 15.0,
            "ma50_position": 10.0,
        },
    }
    spec = StrategySpec(**d)
    dumped = spec.model_dump()
    spec2 = StrategySpec(**dumped)
    assert spec == spec2


def test_exit_rules_take_profit_rr_accepts_15() -> None:
    """ExitRules.take_profit_rr=15.0 should be accepted (le=20.0)."""
    rules = ExitRules(
        stop_loss="10% below entry",
        take_profit="risk_reward_15x",
        take_profit_rr=15.0,
    )
    assert rules.take_profit_rr == 15.0


def test_exit_rules_take_profit_rr_boundary_20() -> None:
    """ExitRules.take_profit_rr=20.0 should be accepted (le=20.0 boundary)."""
    rules = ExitRules(
        stop_loss="10% below entry",
        take_profit="risk_reward_20x",
        take_profit_rr=20.0,
    )
    assert rules.take_profit_rr == 20.0


def test_exit_rules_take_profit_rr_rejects_above_20() -> None:
    """ExitRules.take_profit_rr > 20.0 should be rejected."""
    with pytest.raises(ValidationError):
        ExitRules(
            stop_loss="10% below entry",
            take_profit="risk_reward_25x",
            take_profit_rr=25.0,
        )


def test_backtest_config_entry_mode_defaults() -> None:
    """BacktestConfig entry_mode/trailing_mode defaults for backward compat."""
    config = BacktestConfig()
    assert config.entry_mode == "next_day_open"
    assert config.trailing_mode == "daily"


def test_backtest_config_entry_mode_literal() -> None:
    """BacktestConfig entry_mode/trailing_mode must be valid literals."""
    with pytest.raises(ValidationError):
        BacktestConfig(entry_mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        BacktestConfig(trailing_mode="invalid")  # type: ignore[arg-type]


# --- Earnings Gap Open YAML loading ---

EGO_YAML = STRATEGIES_DIR / "demo_gap_scored" / "strategy.yaml"


def test_load_gap_up_open_scored_yaml() -> None:
    """Loading gap_up_open_scored YAML should produce valid StrategySpec."""
    assert EGO_YAML.exists(), f"Strategy YAML not found: {EGO_YAML}"
    with open(EGO_YAML) as f:
        data = yaml.safe_load(f)
    spec = StrategySpec(**data)
    assert spec.id == "demo_gap_scored"
    assert spec.gap_open_scored is not None
    assert spec.earnings_gap_open is not None
    assert spec.gap_up_detection is None
    assert spec.backtest.entry_mode == "same_bar_open"
    assert spec.backtest.trailing_mode == "weekly"
    assert spec.backtest.portfolio_mode is True


# --- BacktestConfig.weekly_trailing tests ---


def test_backtest_config_weekly_trailing_none_by_default() -> None:
    """BacktestConfig.weekly_trailing defaults to None (backward compat)."""
    config = BacktestConfig()
    assert config.weekly_trailing is None


def test_backtest_config_weekly_trailing_with_defaults() -> None:
    """BacktestConfig accepts WeeklyTrailingStopConfig with defaults."""
    config = BacktestConfig(weekly_trailing=WeeklyTrailingStopConfig())
    assert config.weekly_trailing is not None
    assert config.weekly_trailing.ema_period == 10
    assert config.weekly_trailing.nweek_low_period == 4
    assert config.weekly_trailing.transition_weeks == 3


def test_backtest_config_weekly_trailing_custom() -> None:
    """BacktestConfig accepts custom WeeklyTrailingStopConfig values."""
    config = BacktestConfig(
        weekly_trailing=WeeklyTrailingStopConfig(
            ema_period=8,
            transition_weeks=5,
        )
    )
    assert config.weekly_trailing is not None
    assert config.weekly_trailing.ema_period == 8
    assert config.weekly_trailing.transition_weeks == 5


def test_strategy_spec_backtest_weekly_trailing_roundtrip() -> None:
    """BacktestConfig with weekly_trailing should survive model_dump roundtrip."""
    d = _make_valid_spec_dict()
    d["backtest"] = {
        "trailing_mode": "weekly",
        "weekly_trailing": {"ema_period": 12, "transition_weeks": 1},
    }
    spec = StrategySpec(**d)
    assert spec.backtest.weekly_trailing is not None
    assert spec.backtest.weekly_trailing.ema_period == 12
    dumped = spec.model_dump()
    spec2 = StrategySpec(**dumped)
    assert spec == spec2
