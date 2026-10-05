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


def _run(run, pipe, drop=False):
    order = ["a", "b", "c"]
    cam = {"views": {n: {"c2w": np.eye(4, dtype=np.float32) * (i + 1)}
                     for i, n in enumerate(order)}}
    for n in order:
        cam["views"][n]["c2w"][3, 3] = 1.0
    args = argparse.Namespace(scene="t", b2b_passes=2, b2b_drop=drop)
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


if __name__ == "__main__":
    test_back_to_back_passes_and_dump()
    test_back_to_back_detects_differing_frame()
    test_back_to_back_drop_skips_compare()
    print("ok")
