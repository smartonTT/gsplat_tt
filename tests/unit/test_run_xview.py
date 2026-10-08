"""render/run.py cross-view overlap plumbing (task #379/#386) with fake
pipelines, no device: with GSPLAT_TT_XVIEW_OVERLAP on, every render is told the
next view's w2c (wrapping to the first view between back-to-back passes, none
after the very last), CleanBackend.render_fused passes it to render_view as
next_extrinsics and clears it, and with the flag off nothing is passed.

    python3 tests/unit/test_run_xview.py      (or pytest)
"""
import argparse
import pathlib
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_run_back_to_back import _load_run  # noqa: E402


class _HintBackend:
    """Stands in for CleanBackend: takes the hint the way render_fused does."""

    def __init__(self):
        self.next_extrinsics = None


class _XviewPipeline:
    """Records (extr, hint) per render; counts hint hits like xview::Prefetch."""

    def __init__(self):
        self.backend = _HintBackend()
        self.seen = []
        self.pending = None
        self.hits = 0
        self.misses = 0

    def render(self, gauss, extr, K, H, W):
        if self.pending is not None:
            if np.array_equal(np.asarray(self.pending), np.asarray(extr)):
                self.hits += 1
            else:
                self.misses += 1
        nxt, self.backend.next_extrinsics = self.backend.next_extrinsics, None
        self.pending = nxt
        self.seen.append((extr, nxt))
        img = np.full((H, W, 3), int(float(extr.sum()) * 7) % 251, dtype=np.uint8)
        return types.SimpleNamespace(image=img)


def _cam(order):
    cam = {"views": {n: {"c2w": np.eye(4, dtype=np.float32) * (i + 1)}
                     for i, n in enumerate(order)}}
    for n in order:
        cam["views"][n]["c2w"][3, 3] = 1.0
    return cam


def _b2b(run, pipe, passes=2):
    order = ["a", "b", "c"]
    args = argparse.Namespace(scene="t", b2b_passes=passes, b2b_drop=False)
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        rc = run._back_to_back(args, pipe, None, _cam(order), order, None, 8, 8, "b",
                               None, None, d)
    return rc


def test_b2b_hints_next_view_and_wraps_between_passes():
    run = _load_run()
    run._XVIEW = True
    pipe = _XviewPipeline()
    assert _b2b(run, pipe) == 0
    assert len(pipe.seen) == 9
    # Every hint is the next render's extrinsics; the last render has none.
    for (_, nxt), (extr_next, _) in zip(pipe.seen, pipe.seen[1:]):
        assert nxt is not None and np.array_equal(np.asarray(nxt), np.asarray(extr_next))
    assert pipe.seen[-1][1] is None
    assert pipe.hits == 8 and pipe.misses == 0


def test_b2b_no_hints_when_flag_off():
    run = _load_run()
    run._XVIEW = False
    pipe = _XviewPipeline()
    assert _b2b(run, pipe) == 0
    assert all(nxt is None for _, nxt in pipe.seen)


def test_latency_view_hint():
    run = _load_run()
    run._XVIEW = True
    pipe = _XviewPipeline()
    cam = _cam(["a", "b"])
    run.render_clean_view_timed(pipe, None, cam["views"]["a"]["c2w"], None, 8, 8,
                                next_c2w=cam["views"]["b"]["c2w"])
    run.render_clean_view_timed(pipe, None, cam["views"]["b"]["c2w"], None, 8, 8)
    assert pipe.hits == 1 and pipe.misses == 0 and pipe.seen[-1][1] is None


class _FakeClean:
    def __init__(self):
        self.calls = []

    def render_view(self, *a, **kw):
        self.calls.append(kw)
        return np.zeros((2, 2, 3), np.uint8), {}


def _backend(run):
    torch = run.torch
    b = object.__new__(run.CleanBackend)
    b._clean = _FakeClean()
    b._cached_gauss_np = lambda g: {k: np.zeros(1, np.float32)
                                    for k in ("means", "opacities", "colors", "scales", "rotations")}
    b._cached_cov3d = lambda s, r: np.zeros(1, np.float32)
    for k, v in dict(_mb_contrib_floor=0.0, contrib_floor_override=None, min_opacity=0.0,
                     cull_disabled=False, transmittance_threshold=0.0, max_radius=0,
                     k_cap=3.0, use_isoellipse=False, next_extrinsics=None).items():
        setattr(b, k, v)
    return b, torch.eye(4) * 2.0


def test_render_fused_passes_and_clears_hint():
    run = _load_run()
    run._XVIEW = True
    b, nxt = _backend(run)
    eye = run.torch.eye(4)
    b.next_extrinsics = nxt
    b.render_fused(None, eye, run.torch.eye(3), 2, 2)
    b.render_fused(None, nxt, run.torch.eye(3), 2, 2)
    got = b._clean.calls[0]["next_extrinsics"]
    assert got.dtype == np.float32 and got.flags.c_contiguous and got.shape == (4, 4)
    assert np.array_equal(got, nxt.numpy())
    assert "next_extrinsics" not in b._clean.calls[1] and b.next_extrinsics is None


def test_render_fused_ignores_hint_when_flag_off():
    run = _load_run()
    run._XVIEW = False
    b, nxt = _backend(run)
    b.next_extrinsics = nxt
    b.render_fused(None, run.torch.eye(4), run.torch.eye(3), 2, 2)
    assert "next_extrinsics" not in b._clean.calls[0] and b.next_extrinsics is None


def test_xview_flag_default_on_and_opt_out():
    """Task #393: the overlap is the default; only a value atoi() reads as 0
    turns it off, as env_config::env_uint does in C++."""
    run = _load_run()
    assert run._env_flag_on({}, "GSPLAT_TT_XVIEW_OVERLAP")
    assert run._env_flag_on({"GSPLAT_TT_XVIEW_OVERLAP": ""}, "GSPLAT_TT_XVIEW_OVERLAP")
    assert run._env_flag_on({"GSPLAT_TT_XVIEW_OVERLAP": "1"}, "GSPLAT_TT_XVIEW_OVERLAP")
    assert not run._env_flag_on({"GSPLAT_TT_XVIEW_OVERLAP": "0"}, "GSPLAT_TT_XVIEW_OVERLAP")
    assert not run._env_flag_on({"GSPLAT_TT_XVIEW_OVERLAP": " 0 "}, "GSPLAT_TT_XVIEW_OVERLAP")


if __name__ == "__main__":
    test_b2b_hints_next_view_and_wraps_between_passes()
    test_b2b_no_hints_when_flag_off()
    test_latency_view_hint()
    test_render_fused_passes_and_clears_hint()
    test_render_fused_ignores_hint_when_flag_off()
    test_xview_flag_default_on_and_opt_out()
    print("ok")
