"""Execution service orchestration (close/open cycles)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.execution.broker.base import BrokerGateway, BrokerNetworkError, BrokerRateLimitError
from pipeline.execution.models import BrokerOrderRequest, ExecutionState, OrderRecord
from pipeline.execution.planner import build_order_intents
from pipeline.execution.reconciler import reconcile_execution_state
from pipeline.execution.risk import evaluate_pre_trade
from pipeline.execution.storage import (
    append_execution_journal,
    load_execution_state,
    save_execution_state,
)


@dataclass
class CloseCycleResult:
    strategy_id: str
    session_date: str
    signals_count: int
    intents_count: int
    accepted_count: int
    rejected_count: int
    dry_run: bool


@dataclass
class OpenCycleResult:
    strategy_id: str
    submitted_count: int
    skipped_existing_count: int
    active_orders_count: int
    fill_events_count: int
    dry_run: bool


def run_close_cycle(
    *,
    strategy_id: str,
    session_date: str,
    signals: pd.DataFrame,
    state_path: Path,
    journal_path: Path,
    max_positions: int,
    max_sector_exposure: float,
    allow_new_entries: bool = True,
    kill_switch: bool = False,
    sector_map: dict[str, str] | None = None,
    dry_run: bool = False,
    persist_journal_on_dry_run: bool = True,
) -> CloseCycleResult:
    """Run close-cycle planning: signals -> intents -> risk -> persist.

    Dry-run policy:
    - state.json is never written when `dry_run=True`
    - journal write follows `persist_journal_on_dry_run`
    """
    state = load_execution_state(state_path, strategy_id=strategy_id)

    existing_client_order_ids = {
        i.client_order_id for i in state.pending_intents if i.client_order_id is not None
    } | {o.client_order_id for o in state.active_orders if o.client_order_id is not None}

    intents = build_order_intents(
        signals=signals,
        strategy_id=strategy_id,
        session_date=session_date,
        existing_client_order_ids=existing_client_order_ids,
    )

    # Idempotency guard: do not create duplicate pending intents for same session/symbol/side
    existing_keys = {(i.session_date, i.symbol, i.side) for i in state.pending_intents}
    deduped_intents = [
        i for i in intents if (i.session_date, i.symbol, i.side) not in existing_keys
    ]

    decision = evaluate_pre_trade(
        deduped_intents,
        allow_new_entries=allow_new_entries,
        kill_switch=kill_switch,
        max_positions=max_positions,
        max_sector_exposure=max_sector_exposure,
        current_position_symbols={p.symbol for p in state.positions_cache},
        pending_order_symbols={o.symbol for o in state.active_orders},
        sector_map=sector_map,
    )

    now = datetime.now(tz=UTC)
    journal_rows: list[dict[str, Any]] = []
    for intent in decision.accepted:
        journal_rows.append(
            {
                "timestamp": now,
                "event_type": "intent_created",
                "strategy_id": strategy_id,
                "session_date": session_date,
                "intent_id": intent.intent_id,
                "symbol": intent.symbol,
                "side": intent.side,
                "client_order_id": intent.client_order_id,
            }
        )
    for rej in decision.rejected:
        journal_rows.append(
            {
                "timestamp": now,
                "event_type": "intent_rejected",
                "strategy_id": strategy_id,
                "session_date": session_date,
                "intent_id": rej.intent_id,
                "symbol": rej.symbol,
                "side": "buy",
                "reason_code": rej.reason_code,
                "message": rej.message,
            }
        )

    if journal_rows and (not dry_run or persist_journal_on_dry_run):
        append_execution_journal(journal_path, journal_rows)

    if not dry_run:
        state.pending_intents.extend(decision.accepted)
        state.last_cycle_at = now
        save_execution_state(state, state_path)

    return CloseCycleResult(
        strategy_id=strategy_id,
        session_date=session_date,
        signals_count=int(signals["entry"].sum()) if "entry" in signals.columns else 0,
        intents_count=len(deduped_intents),
        accepted_count=len(decision.accepted),
        rejected_count=len(decision.rejected),
        dry_run=dry_run,
    )


def run_open_cycle(
    *,
    strategy_id: str,
    state_path: Path,
    journal_path: Path,
    broker: BrokerGateway,
    max_daily_loss_pct: float | None = None,
    dry_run: bool = False,
    persist_journal_on_dry_run: bool = True,
) -> OpenCycleResult:
    """Run open-cycle execution: pending intents -> broker orders -> reconcile.

    Robustness policy:
    - submit failures are handled per-intent (other intents continue)
    - successful submissions are persisted even if later reconciliation fails
    - pending intents are consumed and persisted incrementally per successful submit/match
    """
    state = load_execution_state(state_path, strategy_id=strategy_id)
    now = datetime.now(tz=UTC)
    metadata = dict(state.metadata)
    session_date = now.strftime("%Y-%m-%d")

    loss_guard_triggered = False
    if max_daily_loss_pct is not None and max_daily_loss_pct > 0 and not dry_run:
        account = _call_with_retry(broker.get_account, op_name="get_account")
        prev_session = str(metadata.get("risk_session_date", ""))
        prev_peak = _to_float_or_none(metadata.get("peak_equity"))
        equity = float(account.equity)
        if prev_session != session_date or prev_peak is None or prev_peak <= 0:
            peak_equity = equity
        else:
            peak_equity = max(prev_peak, equity)
        daily_loss_pct = max(0.0, (peak_equity - equity) / peak_equity) if peak_equity > 0 else 0.0
        metadata.update(
            {
                "risk_session_date": session_date,
                "peak_equity": peak_equity,
                "current_equity": equity,
                "daily_loss_pct": daily_loss_pct,
            }
        )
        loss_guard_triggered = daily_loss_pct > max_daily_loss_pct

    state = state.model_copy(update={"metadata": metadata})

    existing_open = _call_with_retry(
        broker.list_open_orders,
        strategy_id,
        op_name="list_open_orders",
    )
    open_by_client_id = {
        o.client_order_id: o for o in existing_open if o.client_order_id is not None
    }

    submitted_count = 0
    skipped_existing_count = 0
    submitted_or_existing: list[OrderRecord] = []
    journal_rows: list[dict[str, Any]] = []

    for intent in list(state.pending_intents):
        if intent.client_order_id is not None and intent.client_order_id in open_by_client_id:
            existing = open_by_client_id[intent.client_order_id]
            skipped_existing_count += 1
            submitted_or_existing.append(existing)
            if not dry_run:
                state = _persist_submission_progress(
                    state=state,
                    state_path=state_path,
                    consumed_intent_id=intent.intent_id,
                    active_order=existing,
                    now=now,
                )
            continue

        if loss_guard_triggered:
            journal_rows.append(
                {
                    "timestamp": now,
                    "event_type": "order_submit_skipped_risk",
                    "strategy_id": strategy_id,
                    "intent_id": intent.intent_id,
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "client_order_id": intent.client_order_id,
                    "reason_code": "max_daily_loss_pct",
                }
            )
            continue

        if dry_run:
            continue

        req = BrokerOrderRequest(
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            order_type=intent.order_type,
            time_in_force=intent.time_in_force,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            client_order_id=intent.client_order_id,
        )
        try:
            order = _call_with_retry(broker.submit_order, req, op_name="submit_order")
            submitted_count += 1
            submitted_or_existing.append(order)
            state = _persist_submission_progress(
                state=state,
                state_path=state_path,
                consumed_intent_id=intent.intent_id,
                active_order=order,
                now=now,
            )
            journal_rows.append(
                {
                    "timestamp": now,
                    "event_type": "order_submitted",
                    "strategy_id": strategy_id,
                    "intent_id": intent.intent_id,
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "client_order_id": intent.client_order_id,
                    "broker_order_id": order.broker_order_id,
                }
            )
        except Exception as exc:
            journal_rows.append(
                {
                    "timestamp": now,
                    "event_type": "order_submit_failed",
                    "strategy_id": strategy_id,
                    "intent_id": intent.intent_id,
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "client_order_id": intent.client_order_id,
                    "error": str(exc),
                }
            )

    reconciliation_error: Exception | None = None
    recon_state = state
    fill_events = []
    if loss_guard_triggered:
        journal_rows.append(
            {
                "timestamp": now,
                "event_type": "risk_daily_loss_triggered",
                "strategy_id": strategy_id,
                "session_date": session_date,
                "threshold_pct": max_daily_loss_pct,
                "daily_loss_pct": metadata.get("daily_loss_pct"),
            }
        )

    try:
        # Pull latest broker truth for active orders.
        # Use unfiltered query (strategy_id="") so unmanaged/manual open orders are
        # visible in state.active_orders and downstream risk checks.
        latest_open = _call_with_retry(
            broker.list_open_orders,
            "",
            op_name="list_open_orders_unfiltered",
        )
        latest_map = {o.broker_order_id: o for o in latest_open}
        for local in state.active_orders:
            if local.broker_order_id not in latest_map:
                latest_map[local.broker_order_id] = _call_with_retry(
                    broker.get_order,
                    local.broker_order_id,
                    op_name="get_order",
                )
        for order in submitted_or_existing:
            latest_map[order.broker_order_id] = order

        latest_orders = list(latest_map.values())
        positions = _call_with_retry(broker.list_positions, op_name="list_positions")
        recon = reconcile_execution_state(state, latest_orders, positions, now=now)
        recon_state = recon.state
        fill_events = recon.fill_events
    except Exception as exc:
        reconciliation_error = exc
        # Preserve submission progress to avoid duplicate re-submit on next cycle.
        merged_active: dict[str, OrderRecord] = {o.broker_order_id: o for o in state.active_orders}
        for order in submitted_or_existing:
            merged_active[order.broker_order_id] = order
        recon_state = state.model_copy(
            update={
                "active_orders": list(merged_active.values()),
                "last_cycle_at": now,
                "metadata": metadata,
            }
        )
        journal_rows.append(
            {
                "timestamp": now,
                "event_type": "reconcile_failed",
                "strategy_id": strategy_id,
                "error": str(exc),
            }
        )

    for fill in fill_events:
        journal_rows.append(
            {
                "timestamp": now,
                "event_type": "fill_detected",
                "strategy_id": strategy_id,
                "symbol": fill.symbol,
                "side": fill.side,
                "broker_order_id": fill.broker_order_id,
                "client_order_id": fill.client_order_id,
                "fill_qty": fill.fill_qty,
                "fill_price": fill.fill_price,
            }
        )

    if journal_rows and (not dry_run or persist_journal_on_dry_run):
        append_execution_journal(journal_path, journal_rows)

    if not dry_run:
        save_execution_state(recon_state, state_path)

    if reconciliation_error is not None:
        raise RuntimeError(
            "Open cycle reconciliation failed after partial progress was persisted"
        ) from reconciliation_error

    return OpenCycleResult(
        strategy_id=strategy_id,
        submitted_count=submitted_count,
        skipped_existing_count=skipped_existing_count,
        active_orders_count=len(recon_state.active_orders),
        fill_events_count=len(fill_events),
        dry_run=dry_run,
    )


def _call_with_retry[T](
    op: Callable[..., T],
    *args: Any,
    op_name: str,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.2,
    max_delay_seconds: float = 2.0,
) -> T:
    """Retry transient broker operations with exponential backoff."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return op(*args)
        except (BrokerRateLimitError, BrokerNetworkError):
            if attempt >= max_attempts:
                raise
            delay = min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
            _ = op_name
            time.sleep(delay)


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _persist_submission_progress(
    *,
    state: ExecutionState,
    state_path: Path,
    consumed_intent_id: str,
    active_order: OrderRecord | None,
    now: datetime,
) -> ExecutionState:
    """Persist incremental open-cycle progress to reduce duplicate-submission risk."""
    remaining_pending = [i for i in state.pending_intents if i.intent_id != consumed_intent_id]
    merged_active: dict[str, OrderRecord] = {o.broker_order_id: o for o in state.active_orders}
    if active_order is not None:
        merged_active[active_order.broker_order_id] = active_order
    next_state = state.model_copy(
        update={
            "pending_intents": remaining_pending,
            "active_orders": list(merged_active.values()),
            "last_cycle_at": now,
        }
    )
    save_execution_state(next_state, state_path)
    return next_state
