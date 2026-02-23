"""Tests for scripts/run_demo_pipeline.py helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_run_demo_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_demo_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_demo_pipeline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_demo_spec_path_uses_directory_layout() -> None:
    module = _load_run_demo_module()
    path = module._resolve_demo_spec_path("demo_vcp")
    assert path == Path("strategies/examples/demo_vcp/strategy.yaml")
    assert path.exists()


def test_list_available_demo_strategies_includes_examples() -> None:
    module = _load_run_demo_module()
    strategy_ids = module._list_available_demo_strategies()
    assert "demo_vcp" in strategy_ids
    assert "demo_gap_scored" in strategy_ids
    assert "demo_gap_basic" in strategy_ids
