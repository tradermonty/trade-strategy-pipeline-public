"""Tests for CI_OFFLINE guard functionality."""

import subprocess
import sys

import pytest


def test_ci_offline_blocks_requests() -> None:
    """CI_OFFLINE=1 should block HTTP calls with RuntimeError."""
    code = """
import os
os.environ["CI_OFFLINE"] = "1"

# Trigger the guard by importing conftest
import importlib.util
spec = importlib.util.spec_from_file_location("conftest", "tests/conftest.py")
conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conftest)
conftest._install_offline_guard()

import socket
try:
    socket.create_connection(("example.com", 80), timeout=1)
    print("FAIL: no error raised")
    exit(1)
except RuntimeError as e:
    if "offline CI mode" in str(e):
        print("PASS")
        exit(0)
    else:
        print(f"FAIL: wrong error: {e}")
        exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "PASS" in result.stdout


def test_online_marker_skips_in_offline() -> None:
    """@pytest.mark.online tests should be skipped when CI_OFFLINE=1."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_offline_guard.py::test_online_placeholder",
            "-v",
            "--no-header",
            "-rN",
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "CI_OFFLINE": "1"},
        timeout=30,
    )
    assert "SKIPPED" in result.stdout or "skipped" in result.stdout.lower(), (
        f"Expected online test to be skipped.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.online
def test_online_placeholder() -> None:
    """Placeholder online test that should be skipped in offline mode."""
