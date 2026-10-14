"""Gradient oracle for on-device training (task #674, docs/training-plan/PLAN.md T0).

A differentiable pure-torch port of the forward pass in gsplat/rasterization.py
(project_gaussians + alpha_blend), with SH degree 0 or 3 colour, the 3DGS loss
0.8 * L1 + 0.2 * (1 - SSIM), and a sparse Adam step (fp32 moments, or bf16-stored
moments). Device tasks (T1 blend backward, T2 projection backward + Adam, T4 SH3,
T7 loss) compare against the dumps this module writes:

    python3 -m gsplat.train_oracle --out benchmarks/train_oracle

Conventions (same as rasterization.py unless noted):
  * params are in optimizer space: means (N,3), log_scales (N,3), quats (N,4,
    w-first, unnormalised), opacity_logits (N,), sh (N,K,3) with K = (deg+1)^2.
  * extrinsics are world-to-camera (4,4); pixel centres at +0.5; near plane 0.2;
    0.3 px low-pass added to the 2D covariance diagonal.
  * conic = inverse 2D covariance stored as (A, B, C);
    power = -0.5 * (A dx^2 + 2 B dx dy + C dy^2), alpha = min(alpha_max, o * exp(power)).
  * colour = max(SH(dir) + 0.5, 0), dir = normalize(mean - camera centre)
    (3DGS SH basis; training follows 3DGS/gsplat and clamps at 0 only, decision
    2026-10-14 #763; the display renderer's clamp to [0, 1] is unchanged).
  * per pixel, front to back in depth order (3DGS rules): a pair with
    alpha < alpha_min is skipped; when T * (1 - alpha) < t_min the pixel stops
    and that pair is not blended. out = C + T * background.

The discrete decisions (frustum valid, depth order, live pair masks, alpha and
colour clamps) are recorded in a `Frozen` record. Autograd's gradient is the
gradient of the loss with those decisions held fixed, which is also what a device
backward that reuses the forward's live masks computes; the finite-difference
check evaluates the loss with the same Frozen record.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import torch

from gsplat.utils import quat_to_rotation_matrix

PARAM_NAMES = ("means", "log_scales", "quats", "opacity_logits", "sh")

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435)


@dataclass
class Camera:
    w2c: torch.Tensor  # (4, 4) world-to-camera
    K: torch.Tensor    # (3, 3) intrinsics
    height: int
    width: int


@dataclass
class RenderConfig:
    sh_degree: int = 0
    near: float = 0.2
    lowpass: float = 0.3
    alpha_max: float = 0.99
    alpha_min: float = 1.0 / 255.0
    t_min: float = 1e-4
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Frozen:
    """Discrete decisions of one forward pass (piecewise-constant in the params)."""
    valid: torch.Tensor        # (N,) bool, z > near
    order: torch.Tensor        # (N,) long, front-to-back by camera z
    alpha_clamp: torch.Tensor  # (N, P) bool, o * exp(power) > alpha_max (row = Gaussian id)
    live: torch.Tensor         # (N, P) bool, pair blended into the pixel
    color_lo: torch.Tensor     # (N, 3) bool, colour clamped at 0
    color_hi: torch.Tensor     # (N, 3) bool, colour clamped at 1


@dataclass
class RenderOut:
    image: torch.Tensor        # (H, W, 3)
    final_T: torch.Tensor      # (H, W)
    means2d: torch.Tensor      # (N, 2)
    conic: torch.Tensor        # (N, 3) A, B, C
    opacity: torch.Tensor      # (N,)
    colors: torch.Tensor       # (N, 3) after the clamp
    depth: torch.Tensor        # (N,)
    frozen: Frozen
    visible: torch.Tensor      # (N,) bool, projection visibility used by sparse Adam
    extras: dict = field(default_factory=dict)


def sh_eval(sh: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """3DGS real SH basis up to `degree` (0..3). sh (N,K,3), dirs (N,3) unit."""
    out = SH_C0 * sh[:, 0]
    if degree < 1:
        return out
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    out = out - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    if degree < 2:
        return out
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out = (out + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
           + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
           + SH_C2[3] * xz * sh[:, 7] + SH_C2[4] * (xx - yy) * sh[:, 8])
    if degree < 3:
        return out
    out = (out + SH_C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
           + SH_C3[1] * xy * z * sh[:, 10]
           + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
           + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
           + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
           + SH_C3[5] * z * (xx - yy) * sh[:, 14]
           + SH_C3[6] * x * (xx - 3.0 * yy) * sh[:, 15])
    return out


def render(params: dict[str, torch.Tensor], cam: Camera, cfg: RenderConfig,
           frozen: Frozen | None = None, retain: bool = False) -> RenderOut:
    """Differentiable forward. With `frozen`, reuse its discrete decisions."""
    means, quats = params["means"], params["quats"]
    scales = torch.exp(params["log_scales"])
    opacity = torch.sigmoid(params["opacity_logits"])
    N = means.shape[0]
    H, W = cam.height, cam.width
    P = H * W
    dt = means.dtype
    R = cam.w2c[:3, :3].to(dt)
    t = cam.w2c[:3, 3].to(dt)
    fx, fy = cam.K[0, 0].item(), cam.K[1, 1].item()
    cx, cy = cam.K[0, 2].item(), cam.K[1, 2].item()

    pc = means @ R.T + t
    tx, ty, tz = pc[:, 0], pc[:, 1], pc[:, 2]
    valid = (tz > cfg.near) if frozen is None else frozen.valid
    # Behind-camera Gaussians never reach the blend; keep their math finite.
    tz_safe = torch.where(valid, tz, torch.ones_like(tz))
    means2d = torch.stack([fx * tx / tz_safe + cx, fy * ty / tz_safe + cy], dim=-1)

    q = quats / quats.norm(dim=-1, keepdim=True)
    Rq = quat_to_rotation_matrix(q)
    M = Rq * scales.unsqueeze(1)
    cov3d = M @ M.transpose(1, 2)
    zero = torch.zeros_like(tz)
    J = torch.stack([
        torch.stack([fx / tz_safe, zero, -fx * tx / (tz_safe * tz_safe)], -1),
        torch.stack([zero, fy / tz_safe, -fy * ty / (tz_safe * tz_safe)], -1),
    ], dim=1)  # (N, 2, 3)
    cov2d = J @ (R @ cov3d @ R.T) @ J.transpose(1, 2)
    a = cov2d[:, 0, 0] + cfg.lowpass
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + cfg.lowpass
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)

    cam_center = -(R.T @ t)
    dirs = means - cam_center
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    raw_rgb = sh_eval(params["sh"], dirs, cfg.sh_degree) + 0.5
    if frozen is None:
        color_lo, color_hi = raw_rgb < 0.0, raw_rgb > 1.0
    else:
        color_lo, color_hi = frozen.color_lo, frozen.color_hi
    # 3DGS/gsplat training colour: max(raw, 0), no upper clamp (decision 2026-10-14, #763);
    # color_hi is recorded for reporting only. The display renderer still clamps to [0, 1].
    colors = torch.where(color_lo, torch.zeros_like(raw_rgb), raw_rgb)

    # Projection visibility for sparse Adam: the 3-sigma box overlaps the image
    # (project_gaussians' valid mask with opacities=None, max_radius=-1).
    rx = torch.ceil(3.0 * torch.sqrt(a.detach().clamp(min=0.0)))
    ry = torch.ceil(3.0 * torch.sqrt(c.detach().clamp(min=0.0)))
    m2 = means2d.detach()
    visible = (valid & (m2[:, 0] + rx > 0) & (m2[:, 0] - rx < W)
               & (m2[:, 1] + ry > 0) & (m2[:, 1] - ry < H))

    if retain:
        for tns in (means2d, conic, opacity, colors):
            if tns.requires_grad:
                tns.retain_grad()

    order = torch.argsort(tz.detach(), stable=True) if frozen is None else frozen.order
    py, px = torch.meshgrid(torch.arange(H, dtype=dt) + 0.5,
                            torch.arange(W, dtype=dt) + 0.5, indexing="ij")
    px, py = px.reshape(-1), py.reshape(-1)

    T = torch.ones(P, dtype=dt)
    C = torch.zeros(P, 3, dtype=dt)
    alive = torch.ones(P, dtype=torch.bool)
    rec_clamp = torch.zeros(N, P, dtype=torch.bool)
    rec_live = torch.zeros(N, P, dtype=torch.bool)
    for g in order.tolist():
        if not bool(valid[g]):
            continue
        dx = px - means2d[g, 0]
        dy = py - means2d[g, 1]
        power = -0.5 * (conic[g, 0] * dx * dx + 2.0 * conic[g, 1] * dx * dy
                        + conic[g, 2] * dy * dy)
        raw = opacity[g] * torch.exp(torch.clamp(power, max=0.0))
        if frozen is None:
            clamp = raw.detach() > cfg.alpha_max
        else:
            clamp = frozen.alpha_clamp[g]
        alpha = torch.where(clamp, torch.full_like(raw, cfg.alpha_max), raw)
        if frozen is None:
            ad = alpha.detach()
            live = alive & (ad >= cfg.alpha_min)
            stop = live & (T.detach() * (1.0 - ad) < cfg.t_min)
            live = live & ~stop
            alive = alive & ~stop
        else:
            live = frozen.live[g]
        if frozen is None:
            rec_clamp[g] = clamp
            rec_live[g] = live
        a_eff = torch.where(live, alpha, torch.zeros_like(alpha))
        C = C + (a_eff * T).unsqueeze(-1) * colors[g]
        T = T * (1.0 - a_eff)

    bg = torch.tensor(cfg.background, dtype=dt)
    image = (C + T.unsqueeze(-1) * bg).reshape(H, W, 3)
    fr = frozen if frozen is not None else Frozen(
        valid=valid, order=order, alpha_clamp=rec_clamp, live=rec_live,
        color_lo=color_lo, color_hi=color_hi)
    return RenderOut(image=image, final_T=T.reshape(H, W), means2d=means2d, conic=conic,
                     opacity=opacity, colors=colors, depth=tz, frozen=fr, visible=visible,
                     extras={"stopped": (~alive).reshape(H, W)} if frozen is None else {})


# ---------------------------------------------------------------- loss

def _gauss_1d(size: int = 11, sigma: float = 1.5, dtype=torch.float64) -> torch.Tensor:
    x = torch.arange(size, dtype=dtype) - size // 2
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def filter_matrix(n: int, size: int = 11, sigma: float = 1.5, dtype=torch.float64) -> torch.Tensor:
    """(n, n) banded matrix of the 1D window: G @ z filters z's rows with zero padding."""
    g = _gauss_1d(size, sigma, dtype)
    i = torch.arange(n)
    off = i.unsqueeze(0) - i.unsqueeze(1) + size // 2  # [out, in] -> tap index
    ok = (off >= 0) & (off < size)
    return torch.where(ok, g[off.clamp(0, size - 1)], torch.zeros((), dtype=dtype))


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """3DGS ssim() on (H,W,3) images: 11x11 Gaussian window (sigma 1.5), zero padding.
    The separable window is applied as two banded matmuls, Gh @ z @ Gw^T, which equals
    3DGS's conv2d(padding=5) (the form T7 plans for the FPU)."""
    H, W = img1.shape[0], img1.shape[1]
    Gh = filter_matrix(H, window_size, 1.5, img1.dtype)
    Gw = filter_matrix(W, window_size, 1.5, img1.dtype)
    x = img1.permute(2, 0, 1)
    y = img2.permute(2, 0, 1)

    def filt(z):
        return Gh @ z @ Gw.T

    mu1, mu2 = filt(x), filt(y)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = filt(x * x) - mu1_sq
    s2 = filt(y * y) - mu2_sq
    s12 = filt(x * y) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


def loss_fn(image: torch.Tensor, target: torch.Tensor, lambda_dssim: float = 0.2) -> torch.Tensor:
    """3DGS training loss: (1 - lambda) * L1 + lambda * (1 - SSIM)."""
    l1 = (image - target).abs().mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim(image, target))


# ---------------------------------------------------------------- synthetic scene

def make_camera(height: int = 64, width: int = 64, dtype=torch.float64) -> Camera:
    """Camera at a non-trivial pose (small rotation + translation) looking down +z."""
    ang = 0.1
    Rx = torch.tensor([[1, 0, 0], [0, math.cos(ang), -math.sin(ang)],
                       [0, math.sin(ang), math.cos(ang)]], dtype=dtype)
    Ry = torch.tensor([[math.cos(-ang / 2), 0, math.sin(-ang / 2)], [0, 1, 0],
                       [-math.sin(-ang / 2), 0, math.cos(-ang / 2)]], dtype=dtype)
    w2c = torch.eye(4, dtype=dtype)
    w2c[:3, :3] = Rx @ Ry
    w2c[:3, 3] = torch.tensor([0.1, -0.05, 0.3], dtype=dtype)
    f = 1.1 * width
    K = torch.tensor([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=dtype)
    return Camera(w2c=w2c, K=K, height=height, width=width)


def make_scene(n: int, cam: Camera, sh_degree: int, seed: int = 0, n_hidden: int = 4,
               opacity_range: tuple[float, float] = (0.2, 0.9), scale_mult: float = 1.0,
               dtype=torch.float64) -> dict[str, torch.Tensor]:
    """n random Gaussians in the camera frustum; the last n_hidden are behind the
    camera or off screen (invisible, so sparse Adam leaves them alone). A high
    opacity_range and scale_mult give a dense scene that hits the alpha clamp and
    the per-pixel early stop."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return torch.rand(*s, generator=gen, dtype=dtype)

    def rn(*s):
        return torch.randn(*s, generator=gen, dtype=dtype)

    z = 2.0 + 3.0 * rnd(n)
    half_w = 0.5 * cam.width / cam.K[0, 0].item()
    half_h = 0.5 * cam.height / cam.K[1, 1].item()
    x = (2.0 * rnd(n) - 1.0) * half_w * z * 0.9
    y = (2.0 * rnd(n) - 1.0) * half_h * z * 0.9
    pc = torch.stack([x, y, z], -1)
    nh = min(n_hidden, n)
    if nh:
        pc[n - nh:(n - nh) + (nh + 1) // 2, 2] = -1.0 - rnd((nh + 1) // 2)  # behind
        k = nh // 2
        if k:
            pc[n - k:, 0] = 3.0 * half_w * pc[n - k:, 2]  # far right, off screen
    R = cam.w2c[:3, :3].to(dtype)
    t = cam.w2c[:3, 3].to(dtype)
    means = (pc - t) @ R  # invert x_c = R x_w + t
    log_scales = (torch.log(scale_mult * (0.04 + 0.12 * rnd(n, 3)))
                  + torch.log(z).unsqueeze(-1) - math.log(3.0))
    quats = rn(n, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    lo, hi = opacity_range
    opacity = lo + (hi - lo) * rnd(n)
    opacity_logits = torch.log(opacity / (1.0 - opacity))
    K = (sh_degree + 1) ** 2
    sh = torch.zeros(n, K, 3, dtype=dtype)
    sh[:, 0] = ((0.15 + 0.7 * rnd(n, 3)) - 0.5) / SH_C0
    if K > 1:
        sh[:, 1:] = 0.08 * rn(n, K - 1, 3)
    return {"means": means, "log_scales": log_scales, "quats": quats,
            "opacity_logits": opacity_logits, "sh": sh}


def perturb(params: dict[str, torch.Tensor], seed: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    sig = {"means": 0.03, "log_scales": 0.08, "quats": 0.08, "opacity_logits": 0.3, "sh": 0.08}
    return {k: v + sig[k] * torch.randn(v.shape, generator=gen, dtype=v.dtype)
            for k, v in params.items()}


# ---------------------------------------------------------------- gradients

def loss_and_grads(params: dict[str, torch.Tensor], target: torch.Tensor, cam: Camera,
                   cfg: RenderConfig):
    """Loss, per-param grads, 2D (per-Gaussian) grads, dL/dimage and the forward."""
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    out = render(p, cam, cfg, retain=True)
    out.image.retain_grad()
    loss = loss_fn(out.image, target)
    loss.backward()

    def g(t):
        return torch.zeros_like(t) if t.grad is None else t.grad.detach()

    grads = {k: g(v) for k, v in p.items()}
    grads2d = {"means2d": g(out.means2d), "conic": g(out.conic),
               "opacity": g(out.opacity), "colors": g(out.colors),
               "image": g(out.image)}
    return loss.detach(), grads, grads2d, out


def finite_difference(params: dict[str, torch.Tensor], target: torch.Tensor, cam: Camera,
                      cfg: RenderConfig, frozen: Frozen, eps: float = 1e-6,
                      names=PARAM_NAMES, index: dict | None = None,
                      chunk: int = 128) -> dict[str, torch.Tensor]:
    """Central differences of the loss with the forward's decisions held fixed.
    `index` optionally maps a name to the flat entries to probe (others stay nan).
    The +-eps copies are rendered in batches with torch.func.vmap."""
    base = {k: v.detach() for k, v in params.items()}

    def f(p):
        return loss_fn(render(p, cam, cfg, frozen=frozen).image, target)

    vf = torch.func.vmap(f)
    jobs = []
    for name in names:
        idx = index[name] if index is not None and name in index else range(base[name].numel())
        jobs += [(name, int(i)) for i in idx]
    fd = {name: torch.full_like(base[name], float("nan")) for name in names}
    with torch.no_grad():
        for s0 in range(0, len(jobs), chunk):
            js = jobs[s0:s0 + chunk]
            B = 2 * len(js)
            batch = {k: v.unsqueeze(0).repeat(B, *([1] * v.dim())) for k, v in base.items()}
            for j, (name, i) in enumerate(js):
                flat = batch[name].view(B, -1)
                flat[2 * j, i] += eps
                flat[2 * j + 1, i] -= eps
            L = vf(batch)
            for j, (name, i) in enumerate(js):
                fd[name].view(-1)[i] = (L[2 * j] - L[2 * j + 1]) / (2.0 * eps)
    return fd


def rel_error(ad: torch.Tensor, fd: torch.Tensor, floor_frac: float = 1e-4) -> float:
    """max_i |ad - fd| / max(|fd_i|, floor_frac * max|fd|) over probed (non-nan) entries."""
    m = ~torch.isnan(fd)
    a, f = ad[m], fd[m]
    if f.numel() == 0:
        return 0.0
    scale = f.abs().max().item()
    if scale == 0.0:
        return a.abs().max().item()
    den = torch.clamp(f.abs(), min=floor_frac * scale)
    return ((a - f).abs() / den).max().item()


# ---------------------------------------------------------------- sparse Adam

# 3DGS defaults (position lr is scaled by the scene extent there; 1.0 here).
DEFAULT_LR = {"means": 1.6e-4, "log_scales": 5e-3, "quats": 1e-3,
              "opacity_logits": 0.05, "sh_dc": 2.5e-3, "sh_rest": 2.5e-3 / 20.0}


def _lr_rows(name: str, p: torch.Tensor, lr: dict) -> torch.Tensor | float:
    if name != "sh":
        return lr[name]
    rows = torch.full((1, p.shape[1], 1), lr["sh_rest"], dtype=torch.float32)
    rows[:, 0] = lr["sh_dc"]
    return rows


def sparse_adam_step(params: dict[str, torch.Tensor], grads: dict[str, torch.Tensor],
                     m: dict[str, torch.Tensor], v: dict[str, torch.Tensor], step: int,
                     visible: torch.Tensor, lr: dict | None = None,
                     betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-15,
                     moment_dtype: torch.dtype = torch.float32):
    """One Adam step over the rows where `visible` is set; other rows (params and
    moments) are unchanged. `step` is the global step count before this step
    (bias correction uses step + 1 for every row, as gsplat's SelectiveAdam).

    Params and arithmetic are fp32. Moments are stored as `moment_dtype`: with
    bfloat16 the stored m, v are upcast to fp32, updated in fp32, the update uses
    the fp32 m_new/v_new, and m_new/v_new are rounded to bf16 (round to nearest
    even) for storage. Returns (params, m, v, step + 1), new tensors."""
    lr = DEFAULT_LR if lr is None else lr
    b1, b2 = betas
    t = step + 1
    bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
    vis = visible.bool()
    new_p, new_m, new_v = {}, {}, {}
    for name in PARAM_NAMES:
        p = params[name].to(torch.float32).clone()
        mm = m[name].to(moment_dtype).clone()
        vv = v[name].to(moment_dtype).clone()
        g = grads[name].to(torch.float32)[vis]
        m1 = b1 * mm[vis].float() + (1.0 - b1) * g
        v1 = b2 * vv[vis].float() + (1.0 - b2) * g * g
        denom = v1.sqrt() / math.sqrt(bc2) + eps
        step_size = _lr_rows(name, p, lr)
        if isinstance(step_size, torch.Tensor):
            step_size = step_size / bc1
        else:
            step_size = torch.tensor(step_size / bc1, dtype=torch.float32)
        p[vis] = p[vis] - step_size * (m1 / denom)
        mm[vis] = m1.to(moment_dtype)
        vv[vis] = v1.to(moment_dtype)
        new_p[name], new_m[name], new_v[name] = p, mm, vv
    return new_p, new_m, new_v, t


# ---------------------------------------------------------------- dumps

def build_case(sh_degree: int, n: int = 50, height: int = 64, width: int = 64,
               seed: int = 0, cfg: RenderConfig | None = None) -> dict:
    """Everything a device task needs for one oracle case (fp64 forward/backward,
    fp32 Adam inputs and outputs)."""
    cfg = cfg or RenderConfig(sh_degree=sh_degree)
    cam = make_camera(height, width)
    scene = make_scene(n, cam, sh_degree, seed=seed)
    with torch.no_grad():
        target = render(scene, cam, cfg).image
    params = perturb(scene, seed=seed + 1)
    loss, grads, grads2d, out = loss_and_grads(params, target, cam, cfg)

    # Adam input state: step 4 with moments from a fixed synthetic history.
    gen = torch.Generator().manual_seed(seed + 2)
    step_in = 4
    m_in, v_in = {}, {}
    for k, gk in grads.items():
        s = gk.abs().max().item() or 1.0
        m_in[k] = (0.3 * s * torch.randn(gk.shape, generator=gen, dtype=torch.float64)).float()
        v_in[k] = ((0.5 * s * torch.rand(gk.shape, generator=gen, dtype=torch.float64)) ** 2).float()
    p32 = {k: t.float() for k, t in params.items()}
    g32 = {k: t.float() for k, t in grads.items()}
    pa, ma, va, _ = sparse_adam_step(p32, g32, m_in, v_in, step_in, out.visible)
    m_bf = {k: t.to(torch.bfloat16) for k, t in m_in.items()}
    v_bf = {k: t.to(torch.bfloat16) for k, t in v_in.items()}
    pb, mb, vb, _ = sparse_adam_step(p32, g32, m_bf, v_bf, step_in, out.visible,
                                     moment_dtype=torch.bfloat16)

    def npy(t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            t = t.float()  # bf16 values are exact in fp32
        return t.numpy()

    d = {
        "H": np.int32(height), "W": np.int32(width), "N": np.int32(n),
        "sh_degree": np.int32(sh_degree), "seed": np.int32(seed),
        "cfg_json": np.array(json.dumps(cfg.__dict__)),
        "w2c": npy(cam.w2c), "K": npy(cam.K),
        "target": npy(target), "image": npy(out.image), "final_T": npy(out.final_T),
        "loss": np.float64(loss.item()),
        "dL_dimage": npy(grads2d["image"]),
        "means2d": npy(out.means2d), "conic": npy(out.conic), "opacity": npy(out.opacity),
        "colors": npy(out.colors), "depth": npy(out.depth),
        "order": npy(out.frozen.order), "valid": npy(out.frozen.valid),
        "visible": npy(out.visible), "live": npy(out.frozen.live.reshape(n, height, width)),
        "dL_dmeans2d": npy(grads2d["means2d"]), "dL_dconic": npy(grads2d["conic"]),
        "dL_dopacity": npy(grads2d["opacity"]), "dL_dcolors": npy(grads2d["colors"]),
        "adam_step_in": np.int32(step_in),
        "adam_lr_json": np.array(json.dumps(DEFAULT_LR)),
        "adam_betas": np.array([0.9, 0.999]), "adam_eps": np.float64(1e-15),
    }
    for k in PARAM_NAMES:
        d[f"scene_{k}"] = npy(scene[k])
        d[f"param_{k}"] = npy(params[k])
        d[f"grad_{k}"] = npy(grads[k])
        d[f"adam_m_in_{k}"] = npy(m_in[k])
        d[f"adam_v_in_{k}"] = npy(v_in[k])
        d[f"adam_fp32_param_{k}"] = npy(pa[k])
        d[f"adam_fp32_m_{k}"] = npy(ma[k])
        d[f"adam_fp32_v_{k}"] = npy(va[k])
        d[f"adam_bf16_m_in_{k}"] = npy(m_bf[k])
        d[f"adam_bf16_v_in_{k}"] = npy(v_bf[k])
        d[f"adam_bf16_param_{k}"] = npy(pb[k])
        d[f"adam_bf16_m_{k}"] = npy(mb[k])
        d[f"adam_bf16_v_{k}"] = npy(vb[k])
    return d


def write_dumps(out_dir: str | pathlib.Path, n: int = 50, size: int = 64) -> list[pathlib.Path]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for deg in (0, 3):
        d = build_case(deg, n=n, height=size, width=size)
        path = out_dir / f"t0_sh{deg}_{size}x{size}_n{n}.npz"
        np.savez_compressed(path, **d)
        paths.append(path)
    return paths


def fd_report(sh_degree: int, n: int = 50, size: int = 64, eps: float = 1e-6,
              dense: bool = False) -> dict:
    """Check every param entry of one case against central differences."""
    cfg = RenderConfig(sh_degree=sh_degree)
    cam = make_camera(size, size)
    kw = {"opacity_range": (0.9, 0.999), "scale_mult": 2.5} if dense else {}
    scene = make_scene(n, cam, sh_degree, **kw)
    with torch.no_grad():
        target = render(scene, cam, cfg).image
    params = perturb(scene)
    loss, grads, _, out = loss_and_grads(params, target, cam, cfg)
    fd = finite_difference(params, target, cam, cfg, out.frozen, eps=eps)
    fr = out.frozen
    return {"sh_degree": sh_degree, "n": n, "size": size, "dense": dense, "eps": eps,
            "loss": loss.item(), "live_pairs": int(fr.live.sum()),
            "alpha_clamped": int((fr.alpha_clamp & fr.live).sum()),
            "stopped_pixels": int(out.extras["stopped"].sum()),
            "entries": {k: grads[k].numel() for k in PARAM_NAMES},
            "max_rel_err": {k: rel_error(grads[k], fd[k]) for k in PARAM_NAMES}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="benchmarks/train_oracle")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--fd-report", metavar="JSON",
                    help="check every param entry against finite differences, write JSON")
    a = ap.parse_args(argv)
    if a.fd_report:
        rep = [fd_report(deg, a.n, a.size, dense=dense)
               for deg in (0, 3) for dense in (False, True)]
        pathlib.Path(a.fd_report).write_text(json.dumps(rep, indent=1) + "\n")
        for r in rep:
            print(json.dumps({k: r[k] for k in ("sh_degree", "dense", "max_rel_err")}))
        return
    for p in write_dumps(a.out, a.n, a.size):
        print(p)


if __name__ == "__main__":
    main()
