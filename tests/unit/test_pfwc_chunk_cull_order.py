"""Task #466 model (docs/pfwc-chunk-cull-t466/cullshare.py), synthetic scene, no device:
pfwc tiles in a spatially random (training-like) order cannot be culled by the
conservative chunk box test, Morton-ordered tiles can, and no visible gaussian is
ever inside a culled tile. The per-core Morton order keeps each core's gaussian set
(vis_tile::SeqMap deal), so only the order inside a core changes.

    python3 tests/unit/test_pfwc_chunk_cull_order.py      (or pytest)
"""
import importlib.util
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cullshare", ROOT / "docs/pfwc-chunk-cull-t466/cullshare.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

W = H = 256
F = 0.5 * W / math.tan(math.radians(25))


def _scene(n=64 * 1024, seed=1):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-20, 20, (n, 3)) * [4.0, 4.0, 1.0]
    scales = np.exp(rng.uniform(-5, -2, (n, 3)))
    q = rng.normal(size=(n, 4))
    op = rng.uniform(0, 1, n)
    return means, scales, q, op


def _w2c():
    w2c = np.eye(4)
    w2c[:3, 3] = [0.0, 0.0, 22.0]  # camera at z = -22 looking down +z at the box
    return w2c


def _stats(order, real=None):
    means, scales, q, op = _scene()
    rho = 3.0 * np.sqrt((scales ** 2).sum(1))
    vis = cs.visible(means, scales, q, op, _w2c(), W, H, F, 1.0 / 255)
    tab = cs.tables(means, rho, order)
    return cs.view_stats(vis, order, tab, _w2c(), F, W, H, real), vis


def test_random_order_culls_nothing():
    n = len(_scene()[0])
    (tcull, gcull, empty, lost, _), vis = _stats(np.arange(n))
    assert 0.01 < vis.mean() < 0.9, vis.mean()
    assert tcull == 0.0 and empty == 0.0 and lost == 0


def test_morton_order_culls_and_loses_nothing():
    means = _scene()[0]
    (tcull, gcull, empty, lost, _), _ = _stats(cs.morton_order(means))
    assert tcull > 0.3 and lost == 0
    assert tcull <= empty + 1e-12  # a culled tile is always an empty tile


def test_core_local_order_keeps_each_core_set():
    means = _scene()[0]
    n, C = len(means), 7
    order, real = cs.core_local_order(means, C)
    assert len(order) % cs.TILE == 0 and real.sum() == n
    assert np.array_equal(np.sort(order[real]), np.arange(n))
    nt = (n + cs.TILE - 1) // cs.TILE
    pos = 0
    for c in range(C):
        own = np.concatenate([np.arange(t * cs.TILE, min(n, (t + 1) * cs.TILE)) for t in range(c, nt, C)])
        k = len(own) + (-len(own)) % cs.TILE
        seg = order[pos:pos + k][real[pos:pos + k]]
        assert np.array_equal(np.sort(seg), own)
        pos += k
    (tcull, _, _, lost, _), _ = _stats(order, real)
    assert lost == 0


if __name__ == "__main__":
    test_random_order_culls_nothing()
    test_morton_order_culls_and_loses_nothing()
    test_core_local_order_keeps_each_core_set()
    print("ok")
