"""Guards against reintroducing retired adapters."""

from __future__ import annotations

import importlib.util


def test_legacy_finviz_client_is_retired() -> None:
    """The unauthenticated Finviz export adapter was retired (issue #1).

    It called ``finviz.com/export.ashx`` without authentication and passed the raw
    response to the CSV parser, so a sign-in page could be parsed as data. A future
    Finviz integration must use the Elite export endpoint with validated parsing.
    """
    assert importlib.util.find_spec("pipeline.data.finviz_client") is None
