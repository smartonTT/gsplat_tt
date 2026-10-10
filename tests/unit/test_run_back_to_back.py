"""render/run.py --back-to-back (task #275) with a fake pipeline, no device:
the measured passes are timed and compared against the check pass, the check
pass dumps the latency mode's file names, and a differing frame fails (rc 5).

    python3 tests/unit/test_run_back_to_back.py      (or pytest)
"""
import argparse
import importlib.util
import pathlib
import sys
import tempfile
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_run():
    # run.py imports the full renderer stack; stub optional deps this host lacks
    # (e.g. plyfile on the Mac). The code under test needs only numpy/torch/PIL.
    for _ in range(10):
        spec = importlib.util.spec_from_file_location("render_run", ROOT / "render/run.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            return mod
        except ModuleNotFoundError as e:
            if e.name in ("numpy", "torch", "PIL"):
                raise
            sys.modules[e.name] = types.ModuleType(e.name)
        except ImportError as e:
            name = getattr(e, "name", None)
            if not name or name in ("numpy", "torch", "PIL"):
                raise
            sys.modules[name] = types.SimpleNamespace(**{
                a: object for a in ("PlyData", "PlyElement")})
    raise RuntimeError("could not import render/run.py")


class _FakePipeline:
    """Returns a frame derived from the extrinsics; `flip_on` corrupts one call."""

    def __init__(self, flip_on=None):
        self.calls = 0
        self.flip_on = flip_on

    def render(self, gauss, extr, K, H, W):
        img = np.full((H, W, 3), int(float(extr.sum()) * 7) % 251, dtype=np.uint8)
        if self.calls == self.flip_on:
            img[0, 0, 0] ^= 1
        self.calls += 1
        return types.SimpleNamespace(image=img)


def _run(run, pipe, drop=False, gap=None):
    order = ["a", "b", "c"]
    cam = {"views": {n: {"c2w": np.eye(4, dtype=np.float32) * (i + 1)}
                     for i, n in enumerate(order)}}
    for n in order:
        cam["views"][n]["c2w"][3, 3] = 1.0
    args = argparse.Namespace(scene="t", b2b_passes=2, b2b_drop=drop)
    if gap is not None:
        args.view_gap_ms = gap
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "dump").mkdir()
        rc = run._back_to_back(args, pipe, None, cam, order, None, 8, 8, "b",
                               None, d / "dump", d)
        dumped = sorted(p.name for p in (d / "dump").iterdir())
        assert (d / "hero_clean.png").exists()
    return rc, dumped


def test_back_to_back_passes_and_dump():
    run = _load_run()
    pipe = _FakePipeline()
    rc, dumped = _run(run, pipe)
    assert rc == 0
    assert pipe.calls == 9  # check pass + 2 measured passes, 3 views each
    assert dumped == ["view00_a.png", "view01_b.png", "view02_c.png"]


def test_back_to_back_detects_differing_frame():
    run = _load_run()
    rc, _ = _run(run, _FakePipeline(flip_on=4))  # pass 1, view b
    assert rc == 5


def test_back_to_back_drop_skips_compare():
    run = _load_run()
    pipe = _FakePipeline(flip_on=4)
    rc, _ = _run(run, pipe, drop=True)
    assert rc == 0 and pipe.calls == 9


class _FakeClean:
    """stage_timings()/reset_stage_timings() like the C++ module (task #464)."""

    def __init__(self):
        self.resets = 0
        self.views = 0

    def reset_stage_timings(self):
        self.resets += 1
        self.views = 0

    def stage_timings(self):
        return {"views": self.views, "view_total": 0.5 * self.views,
                "blend": 0.25 * self.views}


class _FakeBackendPipeline(_FakePipeline):
    def __init__(self):
        super().__init__()
        self.backend = types.SimpleNamespace(_clean=_FakeClean())

    def render(self, gauss, extr, K, H, W):
        self.backend._clean.views += 1
        return super().render(gauss, extr, K, H, W)


def test_back_to_back_gap_and_stages(capsys):
    """--view-gap-ms: the spin gap counts in the period, not in render_ms_frame;
    each pass prints its per-view stage means (task #464)."""
    run = _load_run()
    pipe = _FakeBackendPipeline()
    rc, dumped = _run(run, pipe, gap=2.0)
    out = capsys.readouterr().out
    assert rc == 0 and pipe.calls == 9 and len(dumped) == 3
    assert pipe.backend._clean.resets == 3
    lines = [l for l in out.splitlines() if l.startswith("B2B_STAGES")]
    assert len(lines) == 3
    kv = dict(t.split("=") for t in lines[-1].split()[1:])
    assert kv["views"] == "3" and kv["gap_ms"] == "2.00"
    assert float(kv["view_total"]) == 0.5 and float(kv["blend"]) == 0.25
    assert float(kv["period"]) >= float(kv["render"]) + 2.0
    summary = [l for l in out.splitlines() if l.startswith("B2B scene=")][0]
    sk = dict(t.split("=", 1) for t in summary.split() if "=" in t)
    assert float(sk["ms_frame"]) >= float(sk["render_ms_frame"]) + 2.0
    passes = [float(x) for x in sk["pass_ms_frame"].split(",")]
    assert len(passes) == 2
    assert abs(float(sk["ms_frame_median"]) - sorted(passes)[0] / 2 - sorted(passes)[1] / 2) < 2e-3
    assert any(l.startswith("TTW_TIMING b2b_ms_frame_median=") for l in out.splitlines())


def test_spin_ms_waits():
    import time
    run = _load_run()
    t = time.perf_counter()
    run._spin_ms(3.0)
    assert (time.perf_counter() - t) * 1000.0 >= 3.0
    run._spin_ms(0.0)


def test_b2b_keep_mode_sizes_out_ring():
    """Task #485: keep mode holds every frame of a pass (+ hero_clean), so the
    zero-copy ring gets n_views + 2 slots and never re-pins a slot mid-pass."""
    run = _load_run()
    ns = argparse.Namespace
    env = {}
    assert run._b2b_keep_out_slots(ns(back_to_back=True, b2b_drop=False), 30, env) == "32"
    assert env == {"GSPLAT_TT_OUT_ZEROCOPY_SLOTS": "32"}
    env = {"GSPLAT_TT_OUT_ZEROCOPY_SLOTS": "4"}  # an explicit setting wins
    assert run._b2b_keep_out_slots(ns(back_to_back=True, b2b_drop=False), 30, env) is None
    assert env == {"GSPLAT_TT_OUT_ZEROCOPY_SLOTS": "4"}
    for a in (ns(back_to_back=True, b2b_drop=True), ns(back_to_back=False, b2b_drop=False)):
        env = {}
        assert run._b2b_keep_out_slots(a, 30, env) is None and env == {}


if __name__ == "__main__":
    test_back_to_back_passes_and_dump()
    test_back_to_back_detects_differing_frame()
    test_back_to_back_drop_skips_compare()
    test_b2b_keep_mode_sizes_out_ring()
    print("ok")
