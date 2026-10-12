"""Task #536: the opacity prune keeps pfwc's sequence order of the survivors, so
storage ids (tie key, record address) map monotonically. No device.

    python3 tests/unit/test_opacity_prune.py      (or pytest)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "render"))
from opacity_prune import TILE, prune_layout, seq_positions  # noqa: E402


def _seq_ref(n, c):
    """Slow reference of vis_tile::SeqMap: sequence position q -> tile."""
    t = (n + TILE - 1) // TILE
    base, rem = t // c, t % c
    out = []
    for q in range(t):
        big = rem * (base + 1)
        if q < big:
            cc, k = q // (base + 1), q % (base + 1)
        else:
            cc, k = rem + (q - big) // base, (q - big) % base
        tile = cc + k * c
        out.extend(range(tile * TILE, min((tile + 1) * TILE, n)))
    return np.array(out, dtype=np.int64)


def test_seq_positions_match_seqmap():
    for n in (1, 1023, 1024, 5 * TILE + 7, 37 * TILE, 130 * TILE + 434):
        for c in (1, 3, 110, 120):
            np.testing.assert_array_equal(seq_positions(n, c), _seq_ref(n, c))


def test_prune_keeps_sequence_order():
    rng = np.random.default_rng(5)
    thr = np.float32(1.0 / 255.0)
    for n, c in ((300 * TILE + 434, 110), (300 * TILE + 434, 120), (77 * TILE, 7), (5000, 120)):
        op = rng.uniform(0, 0.05, n).astype(np.float32)
        op[rng.integers(0, n, n // 50)] = thr  # boundary values are kept
        src = prune_layout(op, 1.0 / 255.0, c)
        keep = op >= thr
        assert src.shape[0] == int(keep.sum())
        assert np.all(op[src] >= thr)
        assert np.unique(src).shape[0] == src.shape[0]
        # The pruned scene, read in its own sequence order, is the old scene's
        # sequence order without the pruned gaussians.
        old_seq = seq_positions(n, c)
        np.testing.assert_array_equal(src[seq_positions(src.shape[0], c)], old_seq[keep[old_seq]])


def test_no_prune_is_identity():
    op = np.ones(9 * TILE + 3, dtype=np.float32)
    np.testing.assert_array_equal(prune_layout(op, 1.0 / 255.0, 4), np.arange(op.shape[0]))


if __name__ == "__main__":
    test_seq_positions_match_seqmap()
    test_prune_keeps_sequence_order()
    test_no_prune_is_identity()
    print("ok")


def test_nan_opacity_kept_like_device():
    # pfwc drops only op < thr, so NaN survives; the prune must keep it.
    op = np.array([0.5, np.nan, 0.0, 1.0 / 255.0, np.nan], dtype=np.float32)
    src = prune_layout(op, np.float32(1.0 / 255.0), 1)
    np.testing.assert_array_equal(src, [0, 1, 3, 4])


def test_prune_blocker():
    from opacity_prune import prune_blocker
    for env in ({}, {"GSPLAT_TT_PFWC_DEAL": ""}, {"GSPLAT_TT_PFWC_DEAL": "off"},
                {"GSPLAT_TT_CHUNK_CULL": "0"}, {"GSPLAT_TT_CHUNK_CULL": ""},
                {"GSPLAT_TT_PFWC_FUSE": "1"}, {"GSPLAT_TT_PFWC_FUSE": ""}):
        assert prune_blocker(env) is None, env
    for env in ({"GSPLAT_TT_PFWC_DEAL": "lpt"}, {"GSPLAT_TT_CHUNK_CULL": "1"},
                {"GSPLAT_TT_CHUNK_CULL": "2"}, {"GSPLAT_TT_PFWC_FUSE": "0"}):
        assert prune_blocker(env) is not None, env
