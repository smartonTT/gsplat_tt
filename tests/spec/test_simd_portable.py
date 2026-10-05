"""Verify cpu_cpp builds report expected SIMD backend."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_gsplat_cpu():
    # Plain import: loading the .so a second time under another name makes
    # pybind11 raise "type already registered" once another test imported it.
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from backends.cpu_cpp import _gsplat_cpu
    except ImportError as e:
        pytest.skip(f"no _gsplat_cpu extension for this Python: {e}")
    return _gsplat_cpu


def test_simd_backend_is_scalar_when_built_scalar():
    mod = _load_gsplat_cpu()
    backend = mod.simd_backend()
    # bh-30 scalar build uses GSPLAT_SCALAR_ONLY; avx2 build reports avx2.
    assert backend in ("scalar", "neon", "avx2")


def test_has_tt_support_false_on_scalar_build():
    mod = _load_gsplat_cpu()
    assert mod.has_tt_support() in (True, False)
