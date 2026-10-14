"""T1 device decomposition (#685): the numpy model of the device kernel (same prep files,
vector masks and lane order) must match the reference slots on the T0 oracle dumps."""
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_backward_dev as dev  # noqa: E402

DUMPS = sorted(str(p) for p in (ROOT / "benchmarks" / "train_oracle").glob("t0_sh*_64x64_n50.npz"))


@pytest.mark.parametrize("path", DUMPS, ids=[os.path.basename(p) for p in DUMPS])
def test_emulator_matches_ref(path, tmp_path):
    d, cfg = dev.load_dump(path)
    p = dev.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 d["image"], d["dL_dimage"], cfg)
    dev.write(p, str(tmp_path))
    assert np.fromfile(tmp_path / "recs.u32", np.uint32).size == 16 * p["n_slots"]
    rep = dev.compare_slots(dev.emulate(p), dev.ref_slots(d, cfg))
    assert rep["rel_err"] < 1e-4, rep


def test_vector_lanes_cover_tile():
    assert sorted(dev.vector_lanes().ravel().tolist()) == list(range(1024))


def test_bf16_grade_exp_fails_oracle():
    """#693: the forward's exp_21f (~1.7e-3 rel) failed the device check on t0_sh0 (1.25e-1);
    the backward needs the fp32-accurate exp."""
    path = next(p for p in DUMPS if "sh0" in p)
    d, cfg = dev.load_dump(path)
    x = -np.linspace(0, 20, 4001, dtype=np.float32)
    assert np.abs(dev.exp21f(x) / np.exp(x) - 1).max() > 1e-3
    p = dev.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 d["image"], d["dL_dimage"], cfg)
    assert dev.compare_slots(dev.emulate(p, exp=dev.exp21f), dev.ref_slots(d, cfg))["rel_err"] > 1e-2


def test_synth_dump_emulates(tmp_path):
    """docs/t685/synth.py (scaling dumps for the device runs) gives a consistent dump."""
    import subprocess
    out = tmp_path / "s.npz"
    subprocess.run([sys.executable, str(ROOT / "docs/t685/synth.py"), str(out), "64", "96", "120", "3"],
                   check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    d, cfg = dev.load_dump(str(out))
    p = dev.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 d["image"], d["dL_dimage"], cfg)
    assert p["n_slots"] > 100
    assert dev.compare_slots(dev.emulate(p), dev.ref_slots(d, cfg))["rel_err"] < 1e-2


def test_pack_v2_matches_host_finish():
    """#704 v2: on-device finish + lane reduce + writer gather (modelled) equals the v1 host
    reduce and finish for a full pack of 16 pairs."""
    rng = np.random.default_rng(7)
    n = dev.BATCH
    acc = rng.standard_normal((n, 9, 4, 8)).astype(np.float32)
    A, B, C = (rng.uniform(0.1, 2.0, n).astype(np.float32) for _ in range(3))
    _, got = dev.pack_v2(acc, -0.5 * A, -B, -0.5 * C)
    S = acc.astype(np.float64).sum((2, 3))
    want = np.stack([A * S[:, 7] + B * S[:, 8], B * S[:, 7] + C * S[:, 8], -0.5 * S[:, 4], -S[:, 5],
                     -0.5 * S[:, 6], S[:, 0], S[:, 1], S[:, 2], S[:, 3]], 1)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def _synth(tmp_path, args=("64", "96", "120", "3")):
    import subprocess
    out = tmp_path / "s.npz"
    subprocess.run([sys.executable, str(ROOT / "docs/t685/synth.py"), str(out), *args],
                   check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    return dev.load_dump(str(out))


@pytest.mark.parametrize("which", ["sh0", "sh3", "synth"])
def test_vectorised_prep_and_fast_ref_match_loops(which, tmp_path):
    """#707: vectorised prep == per-pair prep_loop, ref_slots_fast == blend_backward_ref (bit equal)."""
    if which == "synth":
        d, cfg = _synth(tmp_path)
    else:
        d, cfg = dev.load_dump(str(ROOT / f"benchmarks/train_oracle/t0_{which}_64x64_n50.npz"))
    args = [d[k] for k in ("means2d", "conic", "opacity", "colors", "order", "valid", "image", "dL_dimage")]
    a, b = dev.prep(*args, cfg), dev.prep_loop(*args, cfg)
    assert a["n_slots"] == b["n_slots"]
    for k in ("tiles", "recs", "planes", "ramps", "P", "counts"):
        assert np.array_equal(a[k], b[k]), k
    assert np.array_equal(dev.ref_slots_fast(d, cfg), dev.ref_slots(d, cfg))


def test_forward_fast_matches_synth_forward(tmp_path):
    """forward_fast (used for the hero image) reproduces synth.py's per-Gaussian fp32 forward."""
    d, cfg = _synth(tmp_path)
    H, W = d["image"].shape[:2]
    img, st = dev.forward_fast(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                               H, W, cfg)
    assert np.abs(img - d["image"]).max() < 1e-5
    assert st["n_pairs"] > 100


def test_fast_ref_flags_threshold_ties():
    """#728: a T_min stop within a few ulp of t_min has two valid outcomes (the hero's one failing
    slot); ref_slots_fast(ties=True) flags exactly the pairs such a decision can change."""
    from dataclasses import replace
    rng = np.random.default_rng(0)
    n = 4   # broad near-flat Gaussians, alpha 0.9 everywhere: T after pair k is ~0.1^(k+1)
    d = {"means2d": np.full((n, 2), 8.0, np.float32), "conic": np.tile(np.float32([1e-8, 0, 1e-8]), (n, 1)),
         "opacity": np.full(n, 0.9, np.float32), "colors": rng.random((n, 3)).astype(np.float32),
         "order": np.arange(n), "valid": np.ones(n, bool),
         "image": rng.random((16, 16, 3)).astype(np.float32),
         "dL_dimage": rng.standard_normal((16, 16, 3)).astype(np.float32)}
    cfg = replace(dev.BlendCfg(), t_min=0.01)       # T after pair 1 lands on t_min
    lo, tie = dev.ref_slots_fast(d, replace(cfg, t_min=0.01 * (1 - 1e-5)), np.float64, ties=True)
    hi = dev.ref_slots_fast(d, replace(cfg, t_min=0.01 * (1 + 1e-5)), np.float64)
    _, tie_at = dev.ref_slots_fast(d, cfg, np.float64, ties=True)
    changed = (np.abs(lo - hi) > 1e-3 * np.abs(lo).max(0)).any(1)
    assert changed[1] and not changed[0], changed
    assert tie_at.tolist() == [False, True, True, True]   # pair 1's stop, and the pixels after it
    assert not tie.any()                                   # 1e-5 off t_min is a clear decision
    rep = dev.compare_slots(hi, lo, skip=tie_at)
    assert rep["tie_slots"] == 3 and rep["rel_err"] == 0.0 and rep["rel_err_all"] > 1e-2
