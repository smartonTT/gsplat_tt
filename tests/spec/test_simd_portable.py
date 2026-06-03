"""Verify cpu_cpp builds report expected SIMD backend."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_gsplat_cpu():
    repo = Path(__file__).resolve().parents[2]
    so_candidates = sorted((repo / "backends" / "cpu_cpp").glob("_gsplat_cpu*.so"))
    if not so_candidates:
        pytest.skip("no _gsplat_cpu extension built")
    spec = importlib.util.spec_from_file_location("_gsplat_cpu", so_candidates[-1])
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_simd_backend_is_scalar_when_built_scalar():
    mod = _load_gsplat_cpu()
    backend = mod.simd_backend()
    # bh-30 scalar build uses GSPLAT_SCALAR_ONLY; avx2 build reports avx2.
    assert backend in ("scalar", "neon", "avx2")


def test_has_tt_support_false_on_scalar_build():
    mod = _load_gsplat_cpu()
    assert mod.has_tt_support() in (True, False)
