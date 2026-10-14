"""Training loss on device (task #690, docs/training-plan/PLAN-single-chip.md T7): model.

The 3DGS loss 0.8 * L1 + 0.2 * (1 - SSIM) and dL/dimage, in the form the device kernel
computes it, checked against gsplat/train_oracle.py (torch autograd, fp64):

  * L1: per pixel, d/dx = 0.8 * sign(x - y) / (3 H W) (torch's abs grad: sign(0) = 0).
  * SSIM fused forward + backward (the fused-ssim / Taming form). The target y is fixed
    for a view, so mu_y = G*y and E_yy = G*(y*y) are per-view constants (computed once per
    GT image and cached). Per step, 3 filtered maps forward (mu_x, E_xx = G*x^2,
    E_xy = G*(x y)) and 3 backward:
        S   = A B / (C D),  A = 2 mu_x mu_y + C1,  B = 2 (E_xy - mu_x mu_y) + C2,
                            C = mu_x^2 + mu_y^2 + C1,  D = E_xx - mu_x^2 + E_yy - mu_y^2 + C2
        dS/dmu_x = S (2 mu_y / A - 2 mu_y / B - 2 mu_x / C + 2 mu_x / D)
        dS/dE_xx = -S / D,   dS/dE_xy = 2 S / B
        dL/dx = G*(w dS/dmu_x) + 2 x G*(w dS/dE_xx) + y G*(w dS/dE_xy),  w = -0.2 / (3 H W)
    (G is symmetric, so its adjoint is the same filter.) Pixels outside the H x W image
    (tile padding) are zero in x and y and masked out of S and of w.
  * The 11-tap separable Gaussian (sigma 1.5, zero padding) is G_h @ Z @ G_w^T. On 32x32
    tiles both banded matrices reduce to three constant tiles (previous, diagonal, next),
    so every output tile is three FPU tile matmuls per pass: band_tiles() below.

FPU operand precision: with fp32 circular buffers and fp32 dest accumulation the unpacker
feeds srcA/srcB as TF32 (8-bit exponent, 10-bit mantissa); products accumulate in fp32.
`scheme` picks how the model rounds matmul operands:
  "fp32"   reference (no operand rounding),
  "tf32"   one HiFi4 matmul per band tile (operands rounded to TF32),
  "split3" x = xh + xl, g = gh + gl (each TF32): xh gh + xl gh + xh gl, about fp32 accuracy.
"""
from __future__ import annotations

import numpy as np

TILE = 32
WIN, SIGMA = 11, 1.5
C1, C2 = 0.01 ** 2, 0.03 ** 2
LAMBDA = 0.2


def gauss_1d(size: int = WIN, sigma: float = SIGMA) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def band_tiles(size: int = WIN, sigma: float = SIGMA) -> dict[int, np.ndarray]:
    """Row pass (right multiply by G_w^T): out[:, J] = sum_k Z[:, J+k] @ R[k], k in -1,0,1.
    R[k][a, b] = g[(32 k + a) - b + r].  Column pass (left multiply by G_h):
    out[I, :] = sum_k R[k].T @ Z[I+k, :] (R[k].T is G_h's block (I, I+k)). fp64."""
    g, r = gauss_1d(size, sigma), size // 2
    assert r < TILE
    a = np.arange(TILE)[:, None]
    b = np.arange(TILE)[None, :]
    out = {}
    for k in (-1, 0, 1):
        off = TILE * k + a - b + r
        ok = (off >= 0) & (off < size)
        out[k] = np.where(ok, g[np.clip(off, 0, size - 1)], 0.0)
    return out


def to_tf32(x: np.ndarray, mode: str = "rne") -> np.ndarray:
    """Round fp32 values to TF32 (10 mantissa bits), round-to-nearest-even or truncate."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    if mode == "rne":
        u = u + np.uint32(0x0FFF) + ((u >> np.uint32(13)) & np.uint32(1))
    u &= np.uint32(0xFFFFE000)
    return u.view(np.float32)


@np.errstate(all="ignore")  # macOS Accelerate matmul raises spurious FP flags
def _mm(a: np.ndarray, b: np.ndarray, scheme: str, rmode: str) -> np.ndarray:
    """One tile matmul (batched over leading dims) with FPU operand rounding; fp32 result."""
    if scheme == "fp32":
        return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    ah = to_tf32(a, rmode)
    bh = to_tf32(b, rmode)
    acc = ah.astype(np.float64) @ bh.astype(np.float64)
    if scheme == "split3":
        al = to_tf32(a.astype(np.float32) - ah, rmode)
        bl = to_tf32(b.astype(np.float32) - bh, rmode)
        acc = acc + al.astype(np.float64) @ bh.astype(np.float64) \
            + ah.astype(np.float64) @ bl.astype(np.float64)
    elif scheme != "tf32":
        raise ValueError(scheme)
    return acc.astype(np.float32)


def filt_tiles(z: np.ndarray, scheme: str = "fp32", rmode: str = "rne") -> np.ndarray:
    """G_h @ z @ G_w^T on (..., Hp, Wp) fp32 planes (multiples of 32), as the device does it:
    row pass then column pass, three band-tile matmuls per output tile each, fp32 between
    passes. Zero padding beyond the plane."""
    R = {k: v.astype(np.float32) for k, v in band_tiles().items()}
    Hp, Wp = z.shape[-2], z.shape[-1]
    nh, nw = Hp // TILE, Wp // TILE
    t = z.reshape(z.shape[:-2] + (nh, TILE, nw, TILE))
    t = np.moveaxis(t, -3, -2)  # (..., nh, nw, 32, 32)
    row = np.zeros_like(t, dtype=np.float32)
    for k in (-1, 0, 1):
        lo, hi = max(0, -k), min(nw, nw - k)  # output tiles J with J+k in range
        if lo < hi:
            row[..., lo:hi, :, :] += _mm(t[..., lo + k:hi + k, :, :], R[k], scheme, rmode)
    out = np.zeros_like(row)
    for k in (-1, 0, 1):
        lo, hi = max(0, -k), min(nh, nh - k)
        if lo < hi:
            out[..., lo:hi, :, :, :] += _mm(np.ascontiguousarray(R[k].T),
                                            row[..., lo + k:hi + k, :, :, :], scheme, rmode)
    out = np.moveaxis(out, -2, -3)
    return out.reshape(z.shape).astype(np.float32)


def pad_planes(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> (3, Hp, Wp) fp32, zero padded to multiples of 32 (the device layout)."""
    H, W = img.shape[:2]
    Hp, Wp = -(-H // TILE) * TILE, -(-W // TILE) * TILE
    out = np.zeros((3, Hp, Wp), dtype=np.float32)
    out[:, :H, :W] = np.moveaxis(img, -1, 0)
    return out


def target_stats(y: np.ndarray, scheme: str = "fp32", rmode: str = "rne"):
    """Per-view constants from the padded target planes: mu_y, E_yy."""
    return filt_tiles(y, scheme, rmode), filt_tiles(y * y, scheme, rmode)


def loss_and_grad_device(image: np.ndarray, target: np.ndarray, scheme: str = "fp32",
                         rmode: str = "rne", lambda_dssim: float = LAMBDA):
    """The device algorithm in fp32: (loss, l1, ssim, dL/dimage (H, W, 3))."""
    H, W = image.shape[:2]
    x, y = pad_planes(image), pad_planes(target)
    mask = np.zeros(x.shape[1:], dtype=np.float32)
    mask[:H, :W] = 1.0
    n = np.float32(3 * H * W)
    f32 = np.float32
    mu_y, e_yy = target_stats(y, scheme, rmode)
    mu_x = filt_tiles(x, scheme, rmode)
    e_xx = filt_tiles(x * x, scheme, rmode)
    e_xy = filt_tiles(x * y, scheme, rmode)
    mxy = mu_x * mu_y
    A = f32(2) * mxy + f32(C1)
    B = f32(2) * (e_xy - mxy) + f32(C2)
    C = mu_x * mu_x + mu_y * mu_y + f32(C1)
    D = (e_xx - mu_x * mu_x) + (e_yy - mu_y * mu_y) + f32(C2)
    rB, rD = f32(1) / B, f32(1) / D
    S = (A * B) / (C * D)
    w = f32(-lambda_dssim) / n * mask
    Sw = S * w
    d_mu = Sw * (f32(2) * mu_y / A - f32(2) * mu_y * rB - f32(2) * mu_x / C + f32(2) * mu_x * rD)
    d_xx = -Sw * rD
    d_xy = f32(2) * Sw * rB
    P = filt_tiles(d_mu, scheme, rmode)
    Q = filt_tiles(d_xx, scheme, rmode)
    Rr = filt_tiles(d_xy, scheme, rmode)
    diff = x - y
    g_l1 = f32((1.0 - lambda_dssim)) / n * np.sign(diff)
    grad = (P + f32(2) * x * Q + y * Rr + g_l1) * mask
    l1 = float(np.abs(diff * mask).astype(np.float64).sum() / float(n))
    ssim = float((S * mask).astype(np.float64).sum() / float(n))
    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim)
    return loss, l1, ssim, np.moveaxis(grad[:, :H, :W], 0, -1)


def loss_and_grad_l1(image: np.ndarray, target: np.ndarray, lambda_dssim: float = LAMBDA):
    """The L1-only device step (first kernel): (0.8 * L1, its dL/dimage)."""
    n = float(image.size)
    d = image.astype(np.float32) - target.astype(np.float32)
    w = np.float32((1.0 - lambda_dssim) / n)
    return (1.0 - lambda_dssim) * float(np.abs(d).astype(np.float64).sum() / n), w * np.sign(d)
