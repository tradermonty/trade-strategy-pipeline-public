"""FMP (Financial Modeling Prep) API client."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pandas as pd


class FmpClient:
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _check_offline(self) -> None:
        if os.environ.get("CI_OFFLINE") == "1":
            raise RuntimeError("API calls are disabled in offline CI mode")

    def get_sector(self, ticker: str) -> dict[str, str]:
        """Get sector/industry classification for a single ticker."""
        self._check_offline()
        return self._fetch_profile(ticker)

    def _fetch_profile(self, ticker: str) -> dict[str, str]:
        """Actual API call. Separated for easy mocking."""
        import httpx

        resp = httpx.get(
            f"{self.BASE_URL}/profile/{ticker}",
            params={"apikey": self.api_key},
        )
        data = resp.json()
        if data:
            return {"sector": data[0]["sector"], "industry": data[0]["industry"]}
        return {"sector": "Unknown", "industry": "Unknown"}

    def fetch_russell1000_approx(
        self,
        *,
        market_cap_more_than: int = 2_000_000_000,
        exchanges: Sequence[str] = ("NYSE", "NASDAQ"),
        active_only: bool = True,
        fetch_limit: int = 3_000,
        top_n_by_market_cap: int = 1_000,
        include_etf: bool = False,
        include_fund: bool = False,
    ) -> list[str]:
        """Approximate Russell 1000 via market-cap-ranked US equities.

        Selection flow:
        1) query FMP stock screener with market cap / exchange / active constraints
        2) optionally exclude ETFs/Funds
        3) sort by market cap desc and keep top_n_by_market_cap
        """
        self._check_offline()
        rows = self._fetch_large_cap_us_rows(
            market_cap_more_than=market_cap_more_than,
            exchanges=exchanges,
            active_only=active_only,
            fetch_limit=fetch_limit,
        )
        filtered: list[dict[str, Any]] = []
        for row in rows:
            symbol = row.get("symbol")
            if not symbol:
                continue
            if not include_etf and bool(row.get("isEtf", False)):
                continue
            if not include_fund and bool(row.get("isFund", False)):
                continue
            filtered.append(row)

        filtered.sort(key=lambda x: x.get("marketCap", 0), reverse=True)
        top = filtered[:top_n_by_market_cap]
        return [str(row["symbol"]) for row in top]

    def _fetch_large_cap_us_stocks(self) -> list[str]:
        """Backward-compatible helper for existing callers."""
        return self.fetch_russell1000_approx()

    def _fetch_large_cap_us_rows(
        self,
        *,
        market_cap_more_than: int,
        exchanges: Sequence[str],
        active_only: bool,
        fetch_limit: int,
    ) -> list[dict[str, Any]]:
        """Use FMP stock screener to get large-cap US instruments."""
        import httpx

        resp = httpx.get(
            f"{self.BASE_URL}/stock-screener",
            params={
                "apikey": self.api_key,
                "marketCapMoreThan": market_cap_more_than,
                "exchange": ",".join(exchanges),
                "isActivelyTrading": active_only,
                "limit": fetch_limit,
            },
        )
        resp.raise_for_status()
        rows: list[dict[str, Any]] = resp.json()
        return rows

    def get_bulk_sectors(self, tickers: list[str]) -> pd.DataFrame:
        """Get sector classification for multiple tickers."""
        self._check_offline()
        return self._fetch_bulk_profiles(tickers)

    def _fetch_bulk_profiles(self, tickers: list[str]) -> pd.DataFrame:
        """Actual bulk fetch. Separated for easy mocking."""
        records = []
        for ticker in tickers:
            profile = self._fetch_profile(ticker)
            records.append(
                {
                    "ticker": ticker,
                    "sector": profile["sector"],
                    "industry": profile["industry"],
                }
            )
        return pd.DataFrame(records)
