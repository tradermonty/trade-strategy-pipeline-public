"""Finviz screening client."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


class FinvizClient:
    def _check_offline(self) -> None:
        if os.environ.get("CI_OFFLINE") == "1":
            raise RuntimeError("API calls are disabled in offline CI mode")

    def screen(self, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a Finviz screen with given filters."""
        self._check_offline()
        return self._screen_from_api(filters)

    def _screen_from_api(self, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        """Actual Finviz API call. Separated for easy mocking."""
        from io import StringIO

        import httpx

        # Simplified implementation
        resp = httpx.get("https://finviz.com/export.ashx", params=filters or {})
        df = pd.read_csv(StringIO(resp.text))
        return df
