"""Tests for data acquisition clients."""

import os
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline.data.alpaca_client import AlpacaClient
from pipeline.data.fmp_client import FmpClient


class TestAlpacaClient:
    def test_raises_in_offline_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI_OFFLINE", "1")
        client = AlpacaClient(api_key="fake", secret_key="fake")
        with pytest.raises(RuntimeError, match="offline CI mode"):
            client.fetch_daily_bars("AAPL", "2023-01-01", "2023-12-31")

    def test_fetch_daily_bars_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = AlpacaClient(api_key="fake", secret_key="fake")

        mock_bars = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-02", periods=3, tz="UTC"),
                "open": [150.0, 151.0, 152.0],
                "high": [155.0, 156.0, 157.0],
                "low": [149.0, 150.0, 151.0],
                "close": [153.0, 154.0, 155.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        with patch.object(client, "_fetch_bars_from_api", return_value=mock_bars):
            result = client.fetch_daily_bars("AAPL", "2023-01-02", "2023-01-04")
            assert len(result) == 3
            assert "close" in result.columns

    def test_fetch_sp500_constituents_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = AlpacaClient(api_key="fake", secret_key="fake")

        mock_constituents = ["AAPL", "MSFT", "GOOG"]
        with patch.object(client, "_fetch_constituents_from_api", return_value=mock_constituents):
            result = client.fetch_sp500_constituents()
            assert len(result) == 3
            assert "AAPL" in result

    @pytest.mark.online
    def test_alpaca_fetch_online(self) -> None:
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            pytest.skip("ALPACA_API_KEY not set")
        client = AlpacaClient(api_key=api_key, secret_key=secret_key)
        result = client.fetch_daily_bars("AAPL", "2023-01-02", "2023-01-06")
        assert len(result) > 0


class TestFmpClient:
    def test_raises_in_offline_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI_OFFLINE", "1")
        client = FmpClient(api_key="fake")
        with pytest.raises(RuntimeError, match="offline CI mode"):
            client.get_sector("AAPL")

    def test_get_sector_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = FmpClient(api_key="fake")

        mock_data = {"sector": "Technology", "industry": "Consumer Electronics"}
        with patch.object(client, "_fetch_profile", return_value=mock_data):
            result = client.get_sector("AAPL")
            assert result["sector"] == "Technology"

    def test_get_bulk_sectors_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = FmpClient(api_key="fake")

        mock_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Technology", "Technology"],
                "industry": ["Consumer Electronics", "Software"],
            }
        )
        with patch.object(client, "_fetch_bulk_profiles", return_value=mock_data):
            result = client.get_bulk_sectors(["AAPL", "MSFT"])
            assert len(result) == 2

    def test_fmp_empty_profile_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_fetch_profile returning empty data should yield 'Unknown' sector."""
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = FmpClient(api_key="fake")

        with patch.object(
            client, "_fetch_profile", return_value={"sector": "Unknown", "industry": "Unknown"}
        ):
            result = client.get_sector("INVALID_TICKER")
            assert result["sector"] == "Unknown"
            assert result["industry"] == "Unknown"

    def test_fetch_russell1000_approx_excludes_fund_etf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fund/ETF rows should be filtered out when include flags are False."""
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = FmpClient(api_key="fake")

        rows = [
            {"symbol": "AAA", "marketCap": 1000, "isEtf": False, "isFund": False},
            {"symbol": "ETF1", "marketCap": 2000, "isEtf": True, "isFund": False},
            {"symbol": "FUNDX", "marketCap": 3000, "isEtf": False, "isFund": True},
            {"symbol": "BBB", "marketCap": 1500, "isEtf": False, "isFund": False},
        ]
        with patch.object(client, "_fetch_large_cap_us_rows", return_value=rows):
            result = client.fetch_russell1000_approx(
                include_etf=False,
                include_fund=False,
                top_n_by_market_cap=10,
            )
        assert result == ["BBB", "AAA"]

    def test_fetch_russell1000_approx_top_n(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Top-N should be applied after market-cap sort."""
        monkeypatch.delenv("CI_OFFLINE", raising=False)
        client = FmpClient(api_key="fake")

        rows = [
            {"symbol": "A", "marketCap": 30, "isEtf": False, "isFund": False},
            {"symbol": "B", "marketCap": 10, "isEtf": False, "isFund": False},
            {"symbol": "C", "marketCap": 20, "isEtf": False, "isFund": False},
        ]
        with patch.object(client, "_fetch_large_cap_us_rows", return_value=rows):
            result = client.fetch_russell1000_approx(top_n_by_market_cap=2)
        assert result == ["A", "C"]

    @pytest.mark.online
    def test_fmp_sector_online(self) -> None:
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            pytest.skip("FMP_API_KEY not set")
        client = FmpClient(api_key=api_key)
        result = client.get_sector("AAPL")
        assert "sector" in result
