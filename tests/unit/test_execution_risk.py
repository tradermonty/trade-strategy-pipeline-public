"""Tests for execution pre-trade risk guardrails."""

from __future__ import annotations

from pipeline.execution.models import OrderIntent
from pipeline.execution.risk import evaluate_pre_trade


def _intent(symbol: str, seq: int) -> OrderIntent:
    return OrderIntent(
        intent_id=f"intent-{seq}",
        strategy_id="strat-x",
        session_date="2026-02-24",
        symbol=symbol,
        side="buy",
        qty=1,
        client_order_id=f"strat-x:2026-02-24:{symbol}:buy:{seq}",
    )


def test_kill_switch_rejects_all() -> None:
    """Kill switch should reject all intents."""
    intents = [_intent("AAPL", 1), _intent("MSFT", 2)]
    decision = evaluate_pre_trade(
        intents,
        allow_new_entries=True,
        kill_switch=True,
        max_positions=5,
        max_sector_exposure=1.0,
    )
    assert decision.passed is False
    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 2
    assert all(r.reason_code == "kill_switch" for r in decision.rejected)


def test_max_positions_rejects_excess() -> None:
    """Intents beyond available slots should be rejected."""
    intents = [_intent("AAPL", 1), _intent("MSFT", 2)]
    decision = evaluate_pre_trade(
        intents,
        allow_new_entries=True,
        kill_switch=False,
        max_positions=5,
        max_sector_exposure=1.0,
        current_position_symbols={"NVDA", "AMZN", "GOOG", "META"},
    )
    assert len(decision.accepted) == 1
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason_code == "max_positions"


def test_duplicate_symbol_rejected() -> None:
    """Existing symbol in current/pending sets must be rejected."""
    intents = [_intent("AAPL", 1), _intent("MSFT", 2)]
    decision = evaluate_pre_trade(
        intents,
        allow_new_entries=True,
        kill_switch=False,
        max_positions=10,
        max_sector_exposure=1.0,
        current_position_symbols={"AAPL"},
    )
    assert len(decision.accepted) == 1
    assert decision.accepted[0].symbol == "MSFT"
    assert decision.rejected[0].reason_code == "duplicate_symbol"


def test_sector_exposure_rejected_when_exceeds_limit() -> None:
    """Sector exposure should block intents that breach configured cap."""
    intents = [_intent("MSFT", 1), _intent("GOOG", 2)]
    decision = evaluate_pre_trade(
        intents,
        allow_new_entries=True,
        kill_switch=False,
        max_positions=5,
        max_sector_exposure=0.40,  # max 2/5 in same sector
        current_position_symbols={"AAPL", "NVDA"},  # both Technology
        sector_map={
            "AAPL": "Technology",
            "NVDA": "Technology",
            "MSFT": "Technology",
            "GOOG": "Technology",
        },
    )
    # Already 2/5 tech; both new tech entries should be rejected.
    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 2
    assert all(r.reason_code == "sector_exposure" for r in decision.rejected)


def test_sector_exposure_counts_pending_symbols() -> None:
    """Pending orders should count toward sector exposure capacity."""
    intents = [_intent("MSFT", 1)]
    decision = evaluate_pre_trade(
        intents,
        allow_new_entries=True,
        kill_switch=False,
        max_positions=5,
        max_sector_exposure=0.40,  # max 2 tech slots out of 5
        current_position_symbols={"AAPL"},  # 1 tech
        pending_order_symbols={"NVDA"},  # 1 tech pending
        sector_map={
            "AAPL": "Technology",
            "NVDA": "Technology",
            "MSFT": "Technology",
        },
    )
    # projected tech exposure becomes 3/5 = 0.60 > 0.40
    assert len(decision.accepted) == 0
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason_code == "sector_exposure"
