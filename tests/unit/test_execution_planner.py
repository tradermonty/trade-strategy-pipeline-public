"""Tests for execution planner."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.execution.planner import build_order_intents


def test_build_order_intents_deterministic_ordering() -> None:
    """Intents should be sorted by score desc, then symbol asc."""
    signals = pd.DataFrame(
        [
            {"symbol": "AAPL", "entry": True, "score": 0.5, "qty": 2},
            {"symbol": "MSFT", "entry": True, "score": 0.9, "qty": 1},
            {"symbol": "AMZN", "entry": True, "score": 0.9, "qty": 3},
        ]
    )
    intents = build_order_intents(signals, "strat-x", "2026-02-24")
    assert [i.symbol for i in intents] == ["AMZN", "MSFT", "AAPL"]
    assert [i.qty for i in intents] == [3, 1, 2]


def test_build_order_intents_client_order_id_and_collision_skip() -> None:
    """Planner should generate canonical client_order_id and avoid collisions."""
    signals = pd.DataFrame(
        [
            {"symbol": "AAPL", "entry": True, "score": 1.0},
            {"symbol": "MSFT", "entry": True, "score": 0.9},
        ]
    )
    existing = {"strat-x:2026-02-24:AAPL:buy:1"}
    intents = build_order_intents(
        signals,
        strategy_id="strat-x",
        session_date="2026-02-24",
        existing_client_order_ids=existing,
    )
    # First sequence ID is taken, planner should advance sequence.
    assert intents[0].client_order_id == "strat-x:2026-02-24:AAPL:buy:2"
    assert intents[1].client_order_id == "strat-x:2026-02-24:MSFT:buy:3"


def test_build_order_intents_defaults_and_optional_fields() -> None:
    """Planner should default qty=1 and carry stop/target if present."""
    signals = pd.DataFrame(
        [
            {
                "symbol": "tsla",
                "entry": True,
                "stop_price": 180.0,
                "target_price": 240.0,
            }
        ]
    )
    intents = build_order_intents(signals, "strat-x", "2026-02-24")
    assert len(intents) == 1
    assert intents[0].symbol == "TSLA"
    assert intents[0].qty == 1
    assert intents[0].stop_price == 180.0
    assert intents[0].target_price == 240.0


def test_build_order_intents_requires_columns() -> None:
    """Planner should reject malformed signal frame."""
    bad = pd.DataFrame([{"symbol": "AAPL"}])
    with pytest.raises(ValueError) as ei:
        build_order_intents(bad, "strat-x", "2026-02-24")
    assert "symbol" in str(ei.value) and "entry" in str(ei.value)
