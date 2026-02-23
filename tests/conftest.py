"""Shared test configuration and fixtures.

CI_OFFLINE guard: When CI_OFFLINE=1, all HTTP requests are blocked to ensure
Tier 1 CI never depends on external services.
"""

import os
from typing import Any, cast

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Block HTTP requests when CI_OFFLINE=1."""
    if os.environ.get("CI_OFFLINE") == "1":
        _install_offline_guard()


def _install_offline_guard() -> None:
    """Monkeypatch HTTP libraries to raise RuntimeError on any request."""
    import importlib

    # Block httpx
    try:
        httpx = importlib.import_module("httpx")

        def _blocked_client_request(
            self: Any, method: str, url: Any, *args: Any, **kwargs: Any
        ) -> Any:
            _ = (self, args, kwargs)
            raise RuntimeError(
                f"API calls are disabled in offline CI mode. Attempted: {method} {url}"
            )

        async def _blocked_async_client_request(
            self: Any, method: str, url: Any, *args: Any, **kwargs: Any
        ) -> Any:
            _ = (self, args, kwargs)
            raise RuntimeError(
                f"API calls are disabled in offline CI mode. Attempted: {method} {url}"
            )

        cast(Any, httpx.Client).request = _blocked_client_request
        cast(Any, httpx.AsyncClient).request = _blocked_async_client_request
    except ImportError:
        pass

    # Block urllib3
    try:
        urllib3 = importlib.import_module("urllib3")

        def _blocked_urlopen(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
            _ = (self, args, kwargs)
            raise RuntimeError(
                f"API calls are disabled in offline CI mode. Attempted: {method} {url}"
            )

        cast(Any, urllib3.HTTPConnectionPool).urlopen = _blocked_urlopen
    except ImportError:
        pass

    # Block socket.create_connection as a last resort
    import socket

    def _blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise RuntimeError(
            "API calls are disabled in offline CI mode. Network connection attempted."
        )

    cast(Any, socket).create_connection = _blocked_create_connection


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.online tests when CI_OFFLINE=1."""
    if os.environ.get("CI_OFFLINE") != "1":
        return

    skip_online = pytest.mark.skip(reason="CI_OFFLINE=1: online tests are skipped")
    for item in items:
        if "online" in item.keywords:
            item.add_marker(skip_online)
