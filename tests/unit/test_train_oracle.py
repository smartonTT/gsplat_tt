"""Task #674 (PLAN.md T0): the torch gradient oracle in gsplat/train_oracle.py.

CPU only. Checks the forward against gsplat/rasterization.py, the banded-matmul SSIM
against 3DGS's conv2d SSIM, autograd against central finite differences (a fixed
random subset of every param group, SH0 and SH3, sparse and dense scenes), the
sparse Adam step (fp32 vs torch.optim.Adam, bf16 moments) and that the committed
dumps in benchmarks/train_oracle/ are current.

    python3 -m pytest -q tests/unit/test_train_oracle.py
The full every-entry FD check: python3 -m gsplat.train_oracle --fd-report <json>
"""
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gsplat import train_oracle as o  # noqa: E402
from gsplat.rasterization import alpha_blend, project_gaussians  # noqa: E402

DUMP_DIR = ROOT / "benchmarks" / "train_oracle"
REL_TOL = 1e-3


def _case(deg, dense=False, n=50, size=64):
    cfg = o.RenderConfig(sh_degree=deg)
    cam = o.make_camera(size, size)
    kw = {"opacity_range": (0.9, 0.999), "scale_mult": 2.5} if dense else {}
    scene = o.make_scene(n, cam, deg, **kw)
    with torch.no_grad():
        target = o.render(scene, cam, cfg).image
    return cfg, cam, scene, target, o.perturb(scene)


def test_projection_matches_rasterization():
    cfg, cam, scene, _, _ = _case(3)
    with torch.no_grad():
        out = o.render(scene, cam, cfg)
    q = scene["quats"] / scene["quats"].norm(dim=-1, keepdim=True)
    m2d, cov2d, depths, _, valid = project_gaussians(
        scene["means"], torch.exp(scene["log_scales"]), q, cam.w2c, cam.K,
        cam.height, cam.width, max_radius=-1)
    assert torch.equal(valid, out.visible)
    torch.testing.assert_close(out.means2d[valid], m2d, rtol=1e-12, atol=1e-9)
    A, B, C = out.conic[valid].unbind(-1)
    conic_ref = torch.linalg.inv(cov2d)
    torch.testing.assert_close(A, conic_ref[:, 0, 0], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(B, conic_ref[:, 0, 1], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(C, conic_ref[:, 1, 1], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(out.depth[valid], depths)


def test_blend_matches_alpha_blend():
    """No alpha floor and no early stop: equal to rasterization.alpha_blend with one
    tile covering the image and the valid Gaussians in depth order."""
    cfg = o.RenderConfig(sh_degree=0, alpha_min=0.0, t_min=0.0)
    cam = o.make_camera()
    scene = o.make_scene(50, cam, 0)
    with torch.no_grad():
        out = o.render(scene, cam, cfg)
    ids = [g for g in out.frozen.order.tolist() if out.frozen.valid[g]]
    ref = alpha_blend(
        out.means2d.float(), _cov_from_conic(out.conic).float(), out.colors.float(), out.opacity.float(), torch.tensor(ids),
        torch.tensor([[0, len(ids)]]), cam.height, cam.width, tile_size=64,
        transmittance_threshold=0.0)
    np.testing.assert_allclose(out.image.numpy(), ref.numpy(), atol=2e-5)
    assert out.image.max() > 0.3  # the scene is not empty


def _cov_from_conic(conic):
    A, B, C = conic.unbind(-1)
    det = A * C - B * B
    return torch.stack([torch.stack([C / det, -B / det], -1),
                        torch.stack([-B / det, A / det], -1)], 1)


def test_ssim_matches_3dgs_conv2d():
    gen = torch.Generator().manual_seed(3)
    x = torch.rand(40, 56, 3, generator=gen, dtype=torch.float64)
    y = (x + 0.1 * torch.randn(40, 56, 3, generator=gen, dtype=torch.float64)).clamp(0, 1)
    g = o._gauss_1d(11, 1.5)
    w = torch.outer(g, g).expand(3, 1, 11, 11)
    a, b = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]

    def filt(z):
        return F.conv2d(z, w, padding=5, groups=3)

    mu1, mu2 = filt(a), filt(b)
    s1 = filt(a * a) - mu1 ** 2
    s2 = filt(b * b) - mu2 ** 2
    s12 = filt(a * b) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ref = (((2 * mu1 * mu2 + C1) * (2 * s12 + C2))
           / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()
    torch.testing.assert_close(o.ssim(x, y), ref, rtol=1e-12, atol=1e-12)


def test_sh_degree0_matches_loader():
    pytest.importorskip("plyfile")
    from gsplat.loading_gaussians import sh_to_rgb
    sh = torch.randn(20, 1, 3, dtype=torch.float64)
    dirs = torch.nn.functional.normalize(torch.randn(20, 3, dtype=torch.float64), dim=-1)
    rgb = (o.sh_eval(sh, dirs, 0) + 0.5).clamp(0, 1)
    torch.testing.assert_close(rgb, sh_to_rgb(sh[:, 0]))


@pytest.mark.parametrize("deg,dense", [(0, False), (3, False), (0, True), (3, True)])
def test_grads_match_finite_differences(deg, dense):
    cfg, cam, _, target, params = _case(deg, dense)
    loss, grads, _, out = o.loss_and_grads(params, target, cam, cfg)
    if dense:  # the frozen decisions that make FD need them are exercised
        assert int((out.frozen.alpha_clamp & out.frozen.live).sum()) > 0
        assert int(out.extras["stopped"].sum()) > 0
    gen = torch.Generator().manual_seed(7)
    index = {}
    for k, v in params.items():
        # the largest-gradient entry plus a random sample of the group
        top = int(grads[k].abs().reshape(-1).argmax())
        rnd = torch.randperm(v.numel(), generator=gen)[:6].tolist()
        index[k] = sorted({top, *rnd})
    if deg == 3:  # every SH band of one visible Gaussian, all channels
        g = int(out.visible.nonzero()[0])
        index["sh"] = sorted(set(index["sh"]) | set(range(g * 48, g * 48 + 48)))
    fd = o.finite_difference(params, target, cam, cfg, out.frozen, index=index)
    for k in o.PARAM_NAMES:
        err = o.rel_error(grads[k], fd[k])
        assert err <= REL_TOL, (k, err)
        assert grads[k].abs().max() > 0, k
    # invisible Gaussians get no gradient
    for k in o.PARAM_NAMES:
        assert grads[k][~out.visible].abs().max() == 0, k


def test_frozen_record_reproduces_forward():
    cfg, cam, _, _, params = _case(3, dense=True)
    with torch.no_grad():
        a = o.render(params, cam, cfg)
        b = o.render(params, cam, cfg, frozen=a.frozen)
    torch.testing.assert_close(a.image, b.image, rtol=0, atol=0)


def _adam_inputs(deg=0):
    cfg, cam, _, target, params = _case(deg)
    _, grads, _, out = o.loss_and_grads(params, target, cam, cfg)
    gen = torch.Generator().manual_seed(11)
    p = {k: v.float() for k, v in params.items()}
    g = {k: v.float() for k, v in grads.items()}
    m = {k: 0.01 * torch.randn(v.shape, generator=gen) for k, v in p.items()}
    v = {k: (0.01 * torch.rand(v.shape, generator=gen)) ** 2 for k, v in p.items()}
    return p, g, m, v, out.visible


def test_adam_fp32_matches_torch_adam():
    p, g, m, v, _ = _adam_inputs()
    allvis = torch.ones(p["means"].shape[0], dtype=torch.bool)
    step = 4
    lr = dict(o.DEFAULT_LR)
    lr["sh_rest"] = lr["sh_dc"]  # one lr per tensor for torch.optim.Adam
    newp, newm, newv, t = o.sparse_adam_step(p, g, m, v, step, allvis, lr=lr)
    assert t == step + 1
    for k in o.PARAM_NAMES:
        x = torch.nn.Parameter(p[k].clone())
        opt = torch.optim.Adam([x], lr=lr["sh_dc" if k == "sh" else k],
                               betas=(0.9, 0.999), eps=1e-15, foreach=False)
        x.grad = g[k].clone()
        opt.state[x] = {"step": torch.tensor(float(step)), "exp_avg": m[k].clone(),
                        "exp_avg_sq": v[k].clone()}
        opt.step()
        torch.testing.assert_close(newp[k], x.detach(), rtol=1e-6, atol=1e-7)
        # torch uses lerp_/addcmul_: same formula, different fp32 rounding
        torch.testing.assert_close(newm[k], opt.state[x]["exp_avg"], rtol=1e-5, atol=1e-12)
        torch.testing.assert_close(newv[k], opt.state[x]["exp_avg_sq"], rtol=1e-5, atol=1e-12)


def test_adam_sparse_leaves_invisible_rows():
    p, g, m, v, vis = _adam_inputs(3)
    assert 0 < int(vis.sum()) < vis.numel()
    newp, newm, newv, _ = o.sparse_adam_step(p, g, m, v, 4, vis)
    for k in o.PARAM_NAMES:
        assert torch.equal(newp[k][~vis], p[k][~vis])
        assert torch.equal(newm[k][~vis], m[k][~vis])
        assert torch.equal(newv[k][~vis], v[k][~vis])
        assert not torch.equal(newp[k][vis], p[k][vis])
    # SH band 0 uses lr sh_dc, bands 1.. use lr sh_rest
    lr = dict(o.DEFAULT_LR, sh_rest=0.0)
    newp0, _, _, _ = o.sparse_adam_step(p, g, m, v, 4, vis, lr=lr)
    assert torch.equal(newp0["sh"][:, 1:], p["sh"][:, 1:])
    torch.testing.assert_close(newp0["sh"][:, 0], newp["sh"][:, 0], rtol=0, atol=0)


def test_adam_bf16_moments():
    p, g, m, v, vis = _adam_inputs(3)
    mb = {k: t.to(torch.bfloat16) for k, t in m.items()}
    vb = {k: t.to(torch.bfloat16) for k, t in v.items()}
    pb, mo, vo, _ = o.sparse_adam_step(p, g, mb, vb, 4, vis, moment_dtype=torch.bfloat16)
    # reference: fp32 step from the upcast bf16 moments; storage rounds to bf16
    mf = {k: t.float() for k, t in mb.items()}
    vf = {k: t.float() for k, t in vb.items()}
    pf, mof, vof, _ = o.sparse_adam_step(p, g, mf, vf, 4, vis)
    pd, _, _, _ = o.sparse_adam_step(p, g, m, v, 4, vis)
    for k in o.PARAM_NAMES:
        assert mo[k].dtype == torch.bfloat16 and vo[k].dtype == torch.bfloat16
        assert torch.equal(pb[k], pf[k])
        assert torch.equal(mo[k], mof[k].to(torch.bfloat16))
        assert torch.equal(vo[k], vof[k].to(torch.bfloat16))
        # close to the fp32-moment step, but not identical (moments were rounded)
        torch.testing.assert_close(pb[k], pd[k], rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("deg", [0, 3])
def test_dumps_are_current(deg):
    path = DUMP_DIR / f"t0_sh{deg}_64x64_n50.npz"
    assert path.exists(), f"run: python3 -m gsplat.train_oracle --out {DUMP_DIR}"
    have = np.load(path)
    want = o.build_case(deg)
    assert set(have.files) == set(want)
    for k, w in want.items():
        h = have[k]
        if np.asarray(w).dtype.kind in "fc":
            np.testing.assert_allclose(h, w, rtol=1e-9, atol=1e-12, err_msg=k)
        else:
            np.testing.assert_array_equal(h, w, err_msg=k)


def test_training_colour_clamps_at_zero_only():
    # Decision 2026-10-14 (#763): the trainer follows 3DGS/gsplat, colour = max(SH + 0.5, 0).
    # A colour above 1 is not clamped and its SH0 gradient flows; one below 0 is clamped.
    cfg, cam, scene, target, params = _case(0)
    p = {k: v.clone() for k, v in params.items()}
    p["sh"][0, 0] = torch.tensor([3.0, 0.5, -3.0], dtype=p["sh"].dtype)
    out = o.render(p, cam, cfg)
    raw = o.SH_C0 * p["sh"][0, 0] + 0.5
    assert raw[0] > 1 and raw[2] < 0
    torch.testing.assert_close(out.colors[0], torch.stack([raw[0], raw[1], raw.new_zeros(())]))
    assert bool(out.frozen.color_hi[0, 0]) and bool(out.frozen.color_lo[0, 2])
    _, grads, g2d, _ = o.loss_and_grads(p, target, cam, cfg)
    if g2d["colors"][0, 0] != 0:
        assert grads["sh"][0, 0, 0] != 0
    assert grads["sh"][0, 0, 2] == 0
