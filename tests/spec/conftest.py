"""Make the cpu_cpp extension available to tests/spec in a fresh worktree.

If backends/cpu_cpp/_gsplat_cpu<EXT_SUFFIX> is missing for this Python, build
it once with opt/build_cpu_ext.sh (10 min cap). If that fails, or
GSPLAT_NO_AUTOBUILD=1 is set, skip the tests that need it with the reason.
Tests marked slow are skipped unless GSPLAT_SLOW_TESTS=1 or -m slow.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "backends" / "cpu_cpp" / ("_gsplat_cpu" + sysconfig.get_config_var("EXT_SUFFIX"))
NEEDS_EXT = {"test_thin_splat_stipple.py", "test_simd_portable.py"}


def _ensure_ext() -> str | None:
    """Return None if the extension imports, else the skip reason."""
    if not EXT.exists():
        if os.environ.get("GSPLAT_NO_AUTOBUILD"):
            return f"{EXT.name} not built (GSPLAT_NO_AUTOBUILD set); run opt/build_cpu_ext.sh {sys.executable}"
        try:
            r = subprocess.run(
                ["bash", str(REPO / "opt" / "build_cpu_ext.sh"), sys.executable],
                cwd=REPO, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            return "opt/build_cpu_ext.sh timed out after 600 s"
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            return f"opt/build_cpu_ext.sh failed (rc={r.returncode}): {' | '.join(tail)}"
    try:
        from backends.cpu_cpp import _gsplat_cpu  # noqa: F401
    except ImportError as e:
        return f"cannot import {EXT.name}: {e}"
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: minutes-long CPU tests; run with GSPLAT_SLOW_TESTS=1 or -m slow"
    )


def _skip_slow(config, items):
    if os.environ.get("GSPLAT_SLOW_TESTS") or "slow" in (config.getoption("-m") or ""):
        return
    mark = pytest.mark.skip(reason="slow; set GSPLAT_SLOW_TESTS=1 or pass -m slow")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(mark)


def pytest_collection_modifyitems(config, items):
    _skip_slow(config, items)
    needing = [it for it in items if Path(str(it.fspath)).name in NEEDS_EXT]
    if not needing:
        return
    reason = _ensure_ext()
    if reason:
        mark = pytest.mark.skip(reason=reason)
        for it in needing:
            it.add_marker(mark)
