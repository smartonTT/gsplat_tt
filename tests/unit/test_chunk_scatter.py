"""Task #493: run.chunk_scatter keeps each 1024-gaussian chunk's set (so the chunk cull
bounds are unchanged) and spreads neighbouring positions across the chunk.

    python3 -m pytest -q tests/unit/test_chunk_scatter.py
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _chunk_scatter():
    # Load only the function: render/run.py imports torch and the device stack at the top.
    src = (ROOT / "render/run.py").read_text()
    start = src.index("def chunk_scatter(")
    end = src.index("\ndef ", start + 1)
    mod = types.ModuleType("cs493")
    mod.np = np
    exec(src[start:end], mod.__dict__)
    return mod.chunk_scatter


def test_same_set_per_chunk_and_partial_tail_kept():
    cs = _chunk_scatter()
    order = np.random.default_rng(3).permutation(5 * 1024 + 300)
    out = cs(order)
    assert sorted(out) == sorted(order)
    for c in range(5):
        a, b = order[c * 1024:(c + 1) * 1024], out[c * 1024:(c + 1) * 1024]
        assert set(a) == set(b) and not np.array_equal(a, b)
    assert np.array_equal(out[5 * 1024:], order[5 * 1024:])


def test_neighbours_spread():
    cs = _chunk_scatter()
    out = cs(np.arange(2048))
    # Consecutive gids come from opposite halves of the Morton chunk.
    assert np.all(np.abs(np.diff(out[:1024])) >= 256)
    assert list(out[:4]) == [0, 512, 256, 768]


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
