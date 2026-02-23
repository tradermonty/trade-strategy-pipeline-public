"""Alpaca Markets API client for historical data."""

from __future__ import annotations

import os

import pandas as pd


class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key

    def _check_offline(self) -> None:
        if os.environ.get("CI_OFFLINE") == "1":
            raise RuntimeError("API calls are disabled in offline CI mode")

    def fetch_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a symbol."""
        self._check_offline()
        return self._fetch_bars_from_api(symbol, start, end)

    def _fetch_bars_from_api(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Actual API call to Alpaca. Separated for easy mocking."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,  # type: ignore[arg-type]
            end=end,  # type: ignore[arg-type]
            adjustment="all",  # type: ignore[arg-type]
        )
        bars = client.get_stock_bars(request)
        df: pd.DataFrame = bars.df.reset_index()  # type: ignore[union-attr]
        df = df.rename(columns={"timestamp": "timestamp"})
        return df

    def fetch_sp500_constituents(self) -> list[str]:
        """Fetch current S&P 500 constituent tickers."""
        self._check_offline()
        return self._fetch_constituents_from_api()

    def _fetch_constituents_from_api(self) -> list[str]:
        """Actual API call. Separated for easy mocking."""
        import io

        import httpx

        headers = {
            "User-Agent": "trade-strategy-pipeline/0.1 (https://github.com; research)",
        }
        resp = httpx.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            follow_redirects=True,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        return tables[0]["Symbol"].tolist()

    def fetch_nasdaq100_constituents(self) -> list[str]:
        """Fetch current NASDAQ 100 constituent tickers."""
        self._check_offline()
        return self._fetch_nasdaq100_from_api()

    def _fetch_nasdaq100_from_api(self) -> list[str]:
        """Actual API call for NASDAQ 100. Separated for easy mocking."""
        import io

        import httpx

        headers = {
            "User-Agent": "trade-strategy-pipeline/0.1 (https://github.com; research)",
        }
        resp = httpx.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=headers,
            follow_redirects=True,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        # The constituents table has a "Ticker" column
        for tbl in tables:
            if "Ticker" in tbl.columns:
                return tbl["Ticker"].tolist()
            if "Symbol" in tbl.columns:
                return tbl["Symbol"].tolist()
        raise RuntimeError("Could not find NASDAQ 100 constituents table")
