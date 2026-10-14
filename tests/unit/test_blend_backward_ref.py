"""T1 gate (task #677): the device-decomposition blend backward vs the T0 oracle.

Covers PLAN.md T1's acceptance (per-Gaussian blend grads within rel 1e-2 of T0) on the
committed SH0/SH3 dumps, a dense case (alpha clamp and early stop active) and a
multi-tile case with partial tiles, plus the Gaussian-major slot layout and determinism.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
from gsplat import blend_backward_ref as b  # noqa: E402
from gsplat import train_oracle as o  # noqa: E402

DUMPS = sorted((ROOT / "benchmarks" / "train_oracle").glob("t0_sh*_64x64_n50.npz"))
BAR = 1e-2


def _oracle_case(h, w, n, dense):
    cfg = o.RenderConfig(sh_degree=0)
    cam = o.make_camera(h, w)
    kw = {"opacity_range": (0.9, 0.999), "scale_mult": 2.5} if dense else {}
    scene = o.make_scene(n, cam, 0, **kw)
    with torch.no_grad():
        target = o.render(scene, cam, cfg).image
    _, _, g2, out = o.loss_and_grads(o.perturb(scene, seed=1), target, cam, cfg)
    fr = out.frozen
    d = {"means2d": out.means2d, "conic": out.conic, "opacity": out.opacity,
         "colors": out.colors, "order": fr.order, "valid": fr.valid, "image": out.image,
         "visible": out.visible, "live": fr.live, "dL_dimage": g2["image"]}
    d = {k: v.detach().numpy() for k, v in d.items()}
    for k in ("means2d", "conic", "opacity", "colors"):
        d["dL_d" + k] = g2[k].detach().numpy()
    d["cfg_json"] = json.dumps(cfg.__dict__)
    stats = {"clamped": int((fr.alpha_clamp & fr.live).sum()),
             "stopped": int(out.extras["stopped"].sum())}
    return d, stats


@pytest.mark.parametrize("path", DUMPS, ids=lambda p: p.name)
@pytest.mark.parametrize("frozen", [False, True], ids=["recompute", "frozen"])
def test_dump_gate(path, frozen):
    d = dict(np.load(path))
    r = b.compare(d, frozen=frozen)
    assert r["n_live_px"] == r["oracle_live_px"]
    assert r["worst_rel_err"] <= BAR, r


def test_dense_case_frozen():
    d, st = _oracle_case(64, 64, 50, dense=True)
    assert st["clamped"] > 0 and st["stopped"] > 0  # both discrete paths exercised
    r = b.compare(d, frozen=True)
    assert r["n_live_px"] == r["oracle_live_px"]
    assert r["worst_rel_err"] <= BAR, r


def test_multitile_partial_tiles_frozen():
    d, _ = _oracle_case(80, 104, 120, dense=False)   # 3x4 tiles, last row/col partial
    r = b.compare(d, frozen=True)
    assert r["n_pairs"] > 120  # Gaussians span several tiles
    assert r["worst_rel_err"] <= BAR, r


def test_slots_gaussian_major_and_deterministic():
    d = dict(np.load(DUMPS[0]))
    args = [d[k] for k in ("means2d", "conic", "opacity", "colors", "order", "valid",
                           "image", "dL_dimage")]
    o1, o2 = b.blend_backward(*args), b.blend_backward(*args)
    assert o1["slots"].shape[0] == int(o1["counts"].sum())
    assert o1["n_pairs"] == o1["slots"].shape[0]  # every slot written by exactly one pair
    np.testing.assert_array_equal(o1["P"][1:], np.cumsum(o1["counts"])[:-1])
    for k in b.GRAD_KEYS:
        assert o1[k].tobytes() == o2[k].tobytes()


def test_bf16_round():
    x = np.array([1.0, 1.00390625, 1.005859375, -3.14159], np.float32)
    np.testing.assert_array_equal(b.bf16_round(x),
                                  torch.tensor(x).to(torch.bfloat16).float().numpy())
