"""#803: exact per-pair alpha cull of the training pair lists (GSPLAT_TT_TRAIN_PAIR_CULL=1).

A (Gaussian, tile) pair whose largest alpha over the tile is below alpha_min is skipped by every
pixel of the blend, so dropping it must leave the rendered image, every kept T1 slot and the
per-Gaussian (T0) gradients bit-identical; the dropped slots are exactly zero without the cull."""
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsplat import blend_backward_dev as dev  # noqa: E402
from gsplat import t2_param_backward as t2  # noqa: E402

f32 = np.float32
DUMPS = {w: ROOT / f"benchmarks/train_oracle/t0_{w}_64x64_n50.npz" for w in ("sh0", "sh3")}


def _scene(n=400, H=70, W=100, seed=3):
    """Random rotated, anisotropic Gaussians with opacities from 1e-3 to 0.99, many of them
    far off-screen or faint, so the 3-sigma rectangles hold plenty of dead pairs."""
    rng = np.random.default_rng(seed)
    m2 = np.stack([rng.uniform(-40, W + 40, n), rng.uniform(-40, H + 40, n)], 1)
    sx, sy = np.exp(rng.uniform(np.log(0.7), np.log(30.0), (2, n)))
    th = rng.uniform(0, np.pi, n)
    c, s = np.cos(th), np.sin(th)
    a = c * c * sx * sx + s * s * sy * sy + 0.3
    b = c * s * (sx * sx - sy * sy)
    cc = s * s * sx * sx + c * c * sy * sy + 0.3
    det = a * cc - b * b
    d = {"means2d": m2.astype(f32), "conic": np.stack([cc / det, -b / det, a / det], 1).astype(f32),
         "opacity": np.exp(rng.uniform(np.log(1e-3), np.log(0.99), n)).astype(f32),
         "colors": rng.random((n, 3)).astype(f32), "order": rng.permutation(n)[: n - 7],
         "valid": rng.random(n) > 0.02}
    cfg = dev.BlendCfg()
    d["image"], _ = dev.forward_fast(*(d[k] for k in ("means2d", "conic", "opacity", "colors", "order", "valid")),
                                     H, W, cfg)
    d["dL_dimage"] = rng.standard_normal((H, W, 3)).astype(f32)
    return d, cfg


def _case(which):
    return _scene() if which == "scene" else dev.load_dump(str(DUMPS[which]))


def _kept(d, cfg):
    H, W = d["image"].shape[:2]
    rects = dev.tile_rects_vec(d["means2d"], d["valid"], d["conic"], H, W)
    live = dev.pair_alive(d["means2d"], d["conic"], d["opacity"], cfg, H, W)
    return dev.pair_lists(rects, d["order"], (W + 31) // 32, live)


@pytest.mark.parametrize("which", ["scene", "sh0", "sh3"])
def test_cull_is_lossless(which):
    d, cfg = _case(which)
    H, W = d["image"].shape[:2]
    args = [d[k] for k in ("means2d", "conic", "opacity", "colors", "order", "valid")]
    img0, st0 = dev.forward_fast(*args, H, W, cfg, cull=False)
    img1, st1 = dev.forward_fast(*args, H, W, cfg, cull=True)
    assert np.array_equal(img0, img1)
    assert st0["n_live_px"] == st1["n_live_px"]
    kept = _kept(d, cfg)["kept"]
    s0, tie0 = dev.ref_slots_fast(d, cfg, ties=True, cull=False)
    s1, tie1 = dev.ref_slots_fast(d, cfg, ties=True, cull=True)
    assert s1.shape[0] == kept.sum()
    assert np.array_equal(s1, s0[kept]) and np.array_equal(tie1, tie0[kept])
    assert not s0[~kept].any() and not tie0[~kept].any()   # dropped pairs: exactly zero slots
    assert np.array_equal(dev.ref_slots_fast(d, cfg, np.float64, cull=True),
                          dev.ref_slots_fast(d, cfg, np.float64, cull=False)[kept])
    # per-Gaussian sums (T2's input) and the T0 gradients
    p0 = dev.prep(*args, d["image"], d["dL_dimage"], cfg, cull=False)
    p1 = dev.prep(*args, d["image"], d["dL_dimage"], cfg, cull=True)
    assert p1["n_slots"] == s1.shape[0] and len(p1["recs"]) == st1["n_pairs"]
    g0 = t2.segment_reduce(s0, p0["P"], p0["counts"])
    g1 = t2.segment_reduce(s1, p1["P"], p1["counts"])
    assert np.array_equal(g0, g1)
    if which == "scene":
        assert (~kept).sum() > 0.2 * kept.size, (int((~kept).sum()), kept.size)
    if which == "sh0":
        z = np.load(DUMPS["sh0"])
        params = {k: np.asarray(z[f"param_{k}"], f32) for k in t2.PARAM_NAMES}
        m = {k: np.asarray(z[f"adam_bf16_m_in_{k}"], f32) for k in t2.PARAM_NAMES}
        v = {k: np.asarray(z[f"adam_bf16_v_in_{k}"], f32) for k in t2.PARAM_NAMES}
        import json
        run = lambda s, p: t2.t2_step(params, s, p["P"], p["counts"], z["w2c"], z["K"], H, W, m, v,  # noqa: E731
                                      int(z["adam_step_in"]), json.loads(str(z["adam_lr_json"])),
                                      tuple(float(b) for b in z["adam_betas"]), float(z["adam_eps"]))
        r0, r1 = run(s0, p0), run(s1, p1)
        for k in r0["grads"]:
            assert np.array_equal(r0["grads"][k], r1["grads"][k]), k
        for k in r0["params"]:
            assert np.array_equal(r0["params"][k], r1["params"][k]), k
    # the device kernel model on the culled lists matches the culled reference
    assert dev.compare_slots(dev.emulate(p1), s1)["rel_err"] < 1e-4


def test_kept_records_are_the_uncut_ones():
    """The culled T1 records are the uncut ones minus the dead pairs, same order and bits,
    with only the slot word renumbered."""
    d, cfg = _scene()
    args = [d[k] for k in ("means2d", "conic", "opacity", "colors", "order", "valid", "image", "dL_dimage")]
    p0, p1 = dev.prep(*args, cfg, cull=False), dev.prep(*args, cfg, cull=True)
    kept = _kept(d, cfg)["kept"]
    newslot = np.cumsum(kept) - 1
    r0 = p0["recs"][kept[p0["recs"][:, 13]]]
    assert np.array_equal(np.delete(r0, 13, 1), np.delete(p1["recs"], 13, 1))
    assert np.array_equal(newslot[r0[:, 13]], p1["recs"][:, 13])
    assert (r0[:, 12] != 0).all()   # pairs no vector of T1 visits (mask 0) are all dropped


def test_alive_is_conservative_and_tight():
    """Brute force over every pixel centre of every pair's tile (fp32 tile-centred power, as the
    blend): no dropped pair reaches alpha_min anywhere; every pair 1% below it is dropped."""
    d, cfg = _scene(n=300, seed=11)
    H, W = d["image"].shape[:2]
    rects = dev.tile_rects_vec(d["means2d"], d["valid"], d["conic"], H, W)
    pl = dev.pair_lists(rects, np.arange(len(d["opacity"])), (W + 31) // 32)
    live = dev.pair_alive(d["means2d"], d["conic"], d["opacity"], cfg, H, W)(pl["g"], pl["tid"])
    g, tid, tx = pl["g"], pl["tid"], (W + 31) // 32
    ox, oy = (tid % tx * 32).astype(f32), (tid // tx * 32).astype(f32)
    con, m2, op = d["conic"], d["means2d"], d["opacity"]
    cf = dev.centred_power_coeffs(*(con[g, i][:, None, None] for i in range(3)),
                                  m2[g, 0][:, None, None] - ox[:, None, None],
                                  m2[g, 1][:, None, None] - oy[:, None, None])
    yy, xx = np.meshgrid(np.arange(32, dtype=f32) + f32(0.5), np.arange(32, dtype=f32) + f32(0.5), indexing="ij")
    pw = dev.centred_power(*cf, xx - f32(16), yy - f32(16))
    al = np.minimum(op[g][:, None, None] * np.exp(np.minimum(pw, f32(0))), f32(cfg.alpha_max))
    inside = ((ox[:, None, None] + xx < W) & (oy[:, None, None] + yy < H))
    amax = np.where(inside, al, 0).max((1, 2))
    assert not (amax[~live] >= f32(cfg.alpha_min)).any()
    assert live[amax >= f32(cfg.alpha_min)].all()
    assert not live[amax < 0.99 * cfg.alpha_min].any()
    assert (~live).sum() > 0.2 * live.size


def test_env_flag_default_off(monkeypatch):
    d, cfg = _scene(n=120)
    args = [d[k] for k in ("means2d", "conic", "opacity", "colors", "order", "valid", "image", "dL_dimage")]
    monkeypatch.delenv(dev.PAIR_CULL_ENV, raising=False)
    off, cut = dev.prep(*args, cfg, cull=False), dev.prep(*args, cfg, cull=True)
    assert cut["n_slots"] < off["n_slots"]
    assert dev.prep(*args, cfg)["n_slots"] == off["n_slots"]
    monkeypatch.setenv(dev.PAIR_CULL_ENV, "1")
    assert dev.prep(*args, cfg)["n_slots"] == cut["n_slots"]
    assert dev.ref_slots_fast(d, cfg).shape[0] == cut["n_slots"]
    monkeypatch.setenv(dev.PAIR_CULL_ENV, "0")
    assert dev.ref_slots_fast(d, cfg).shape[0] == off["n_slots"]
