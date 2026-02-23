"""Pre-trade risk guardrails for execution cycle."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pipeline.execution.models import OrderIntent


@dataclass
class RiskReject:
    intent_id: str
    symbol: str
    reason_code: str
    message: str


@dataclass
class RiskDecision:
    accepted: list[OrderIntent]
    rejected: list[RiskReject]

    @property
    def passed(self) -> bool:
        return len(self.rejected) == 0


def evaluate_pre_trade(
    intents: list[OrderIntent],
    *,
    allow_new_entries: bool,
    kill_switch: bool,
    max_positions: int,
    max_sector_exposure: float,
    current_position_symbols: set[str] | None = None,
    pending_order_symbols: set[str] | None = None,
    sector_map: dict[str, str] | None = None,
) -> RiskDecision:
    """Evaluate pre-trade guardrails and return accepted/rejected intents."""
    current = {s.upper() for s in (current_position_symbols or set())}
    pending = {s.upper() for s in (pending_order_symbols or set())}
    sectors = {k.upper(): v for k, v in (sector_map or {}).items()}

    rejected: list[RiskReject] = []
    accepted: list[OrderIntent] = []

    if kill_switch:
        return RiskDecision(
            accepted=[],
            rejected=[
                RiskReject(
                    intent_id=i.intent_id,
                    symbol=i.symbol,
                    reason_code="kill_switch",
                    message="Emergency kill switch is active",
                )
                for i in intents
            ],
        )

    if not allow_new_entries:
        return RiskDecision(
            accepted=[],
            rejected=[
                RiskReject(
                    intent_id=i.intent_id,
                    symbol=i.symbol,
                    reason_code="new_entries_disabled",
                    message="New entries are disabled by guardrail",
                )
                for i in intents
            ],
        )

    if max_positions <= 0:
        return RiskDecision(
            accepted=[],
            rejected=[
                RiskReject(
                    intent_id=i.intent_id,
                    symbol=i.symbol,
                    reason_code="max_positions_invalid",
                    message="max_positions must be >= 1",
                )
                for i in intents
            ],
        )

    used_slots = len(current | pending)
    sector_counts = Counter(sectors.get(sym, "UNKNOWN") for sym in (current | pending))

    for intent in intents:
        symbol = intent.symbol.upper()

        if symbol in current or symbol in pending:
            rejected.append(
                RiskReject(
                    intent_id=intent.intent_id,
                    symbol=symbol,
                    reason_code="duplicate_symbol",
                    message="Symbol already exists in position/pending set",
                )
            )
            continue

        projected_slots = used_slots + len(accepted) + 1
        if projected_slots > max_positions:
            rejected.append(
                RiskReject(
                    intent_id=intent.intent_id,
                    symbol=symbol,
                    reason_code="max_positions",
                    message="max_positions constraint exceeded",
                )
            )
            continue

        sector = sectors.get(symbol, "UNKNOWN")
        projected_sector_count = sector_counts[sector] + 1
        projected_exposure = projected_sector_count / max_positions
        if projected_exposure > max_sector_exposure:
            rejected.append(
                RiskReject(
                    intent_id=intent.intent_id,
                    symbol=symbol,
                    reason_code="sector_exposure",
                    message="max_sector_exposure constraint exceeded",
                )
            )
            continue

        accepted.append(intent)
        sector_counts[sector] += 1

    return RiskDecision(accepted=accepted, rejected=rejected)
