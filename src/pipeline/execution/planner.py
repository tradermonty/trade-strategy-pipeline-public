"""Signal-to-intent planner for execution close cycle."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.execution.models import OrderIntent


def build_order_intents(
    signals: pd.DataFrame,
    strategy_id: str,
    session_date: str,
    existing_client_order_ids: set[str] | None = None,
) -> list[OrderIntent]:
    """Build deterministic buy-side order intents from signal rows.

    Required columns:
    - symbol
    - entry (bool)

    Optional columns:
    - score (float, default 0.0)
    - qty (int, default 1)
    - stop_price (float)
    - target_price (float)
    """
    if "symbol" not in signals.columns or "entry" not in signals.columns:
        raise ValueError("signals must contain 'symbol' and 'entry' columns")

    existing = set(existing_client_order_ids or set())

    df = signals[signals["entry"]].copy()
    if df.empty:
        return []

    if "score" not in df.columns:
        df["score"] = 0.0
    if "qty" not in df.columns:
        df["qty"] = 1

    # Stable deterministic ordering: score desc, symbol asc
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.sort_values(by=["score", "symbol"], ascending=[False, True], kind="mergesort")

    intents: list[OrderIntent] = []
    # Global per-session sequence by design (not per-symbol).
    # This keeps client_order_id unique and deterministic across the whole run.
    seq = 1
    has_stop = "stop_price" in df.columns
    has_target = "target_price" in df.columns

    for row in df.to_dict(orient="records"):
        symbol = str(row.get("symbol", ""))
        score = float(row.get("score", 0.0))
        qty_raw = row.get("qty", 1)
        qty = int(qty_raw) if qty_raw not in (None, "") and not pd.isna(qty_raw) else 1
        if qty <= 0:
            continue

        client_order_id = f"{strategy_id}:{session_date}:{symbol}:buy:{seq}"
        while client_order_id in existing:
            seq += 1
            client_order_id = f"{strategy_id}:{session_date}:{symbol}:buy:{seq}"

        stop_price_raw = row.get("stop_price") if has_stop else None
        target_price_raw = row.get("target_price") if has_target else None
        stop_price = _as_positive_float_or_none(stop_price_raw)
        target_price = _as_positive_float_or_none(target_price_raw)

        intent = OrderIntent(
            intent_id=client_order_id,
            strategy_id=strategy_id,
            session_date=session_date,
            symbol=symbol,
            side="buy",
            qty=qty,
            order_type="market",
            time_in_force="day",
            stop_price=stop_price,
            target_price=target_price,
            client_order_id=client_order_id,
            metadata={"score": score},
        )
        intents.append(intent)
        existing.add(client_order_id)
        seq += 1

    return intents


def _as_positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    num = float(value)
    if num <= 0:
        return None
    return num
