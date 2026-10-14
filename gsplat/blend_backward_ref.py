"""Blend backward in the device decomposition (task #677, docs/training-plan/PLAN.md T1).

A numpy model of the on-device blend backward, written to the shapes and orders the
device kernel uses, so the kernel can be checked lane for lane against it and the
decomposition itself is checked against the T0 gradient oracle (gsplat/train_oracle.py):

    python3 -m gsplat.blend_backward_ref benchmarks/train_oracle/t0_sh0_64x64_n50.npz

Decomposition (PLAN.md §3 steps 1-4):

1. The forward keeps, per pixel, the final colour (image, background included) and
   nothing else per pair: no last-contributor index, no per-pair T.
2. Per 32x32 tile, on the core that blended it, pairs run front to back in the tile's
   sorted list. Per pixel the backward carries T and the colour accumulated so far
   (as C_acc . g). It recomputes the forward's alpha and live decision with the forward's own
   math (fp32 power polynomial in tile-local coordinates, tile-centred as centred_power
   (#728), alpha clamp 0.99, skip below 1/255, stop when T * (1 - alpha) < t_min), so no live mask is stored. The colour
   behind pair i is image - C_acc(<= i), so
       dL/dalpha_i = T_i (c_i . g) - (image . g - C_acc . g) / (1 - alpha_i)
   needs one reciprocal of (1 - alpha_i) per live pixel. Only the dot products are
   needed: per pixel the state is T, s = C_acc . g, Ig = image . g and g (3), which with
   the x/y ramps is 8 fp32 tiles, the Dst capacity in full-sync mode.
3. Per pair and per live pixel it forms 9 terms, with G = dL/dpower and u, v the pixel
   centre minus the Gaussian mean:
       dopacity: dalpha * w            (0 where the alpha clamp is active)
       dcolour:  alpha * T * g         (3)
       moments:  G u^2, G u v, G v^2, G u, G v
   and sums them over the 32 lanes of each 4x8 microblock, then over the tile's
   microblocks in microblock order. The 2D-mean and conic gradients follow per pair:
       dA = -0.5 S_uu, dB = -S_uv, dC = -0.5 S_vv,
       dmx = A S_u + B S_v, dmy = B S_u + C S_v.
4. Each pair writes its 9 floats to slot P[g] + j of a Gaussian-major array: P[g] is
   the exclusive prefix sum of the Gaussians' tile counts (tile rectangles as
   gsplat_cpu::tile_assign), j the tile's row-major index inside g's rectangle. No two
   pairs share a slot. The per-Gaussian sum then reads slots j = 0..count-1 in order,
   so the result is deterministic (T2 fuses that sum into its per-Gaussian pass).

Numbers are float32 throughout (the device's fp32 SFPU path and the forward's fp32 pair
records). `bf16_inputs` rounds the per-pair records (means2d, conic, opacity, colour) to
bf16 first; that is informational only: it moves the forward away from the image the
gradient was taken at and misses the 1e-2 gate by orders of magnitude (means2d in bf16
has 1/8 px resolution at x = 32), so the device backward keeps fp32 records.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass

import numpy as np

TILE = 32
MB_H, MB_W = 4, 8          # one SFPU vector: 4 rows x 8 columns
f32 = np.float32
N_TERMS = 9                # dmx dmy dA dB dC dopacity dr dg db (after the pair epilogue)


@dataclass
class BlendCfg:
    alpha_max: float = 0.99
    alpha_min: float = 1.0 / 255.0
    t_min: float = 1e-4
    background: tuple = (0.0, 0.0, 0.0)


def centred_power_coeffs(A, B, C, mx, my):
    """Per-pair coefficients of the tile-centred power polynomial (#728).

    power(u, v) = A' u^2 + B' u v + C' v^2 with A' = -A/2, B' = -B, C' = -C/2 (fp32, as the
    pair record) and u, v the pixel centre minus the tile-local mean. Evaluated as is in fp32
    it cancels catastrophically for Gaussians whose mean lies 1e4+ px off the tile (tiny conic,
    huge u, v: the hero view has them at the front of tiles, #707). Instead, with du = X - 16,
    dv = Y - 16 (exact in fp32) and U0 = 16 - mx, V0 = 16 - my:
        power = P0 + du (Lx + A' du + B' dv) + dv (Ly + C' dv)
    P0 = power(U0, V0), Lx = 2 A' U0 + B' V0, Ly = B' U0 + 2 C' V0 are formed in float64 here
    (host side) and rounded to fp32 once; the per-pixel part only adds small terms.
    Returns fp32 (Af, Bf, Cf, P0, Lx, Ly); inputs are fp32 arrays or scalars."""
    Af, Bf, Cf = f32(-0.5) * f32(A), -f32(B), f32(-0.5) * f32(C)
    a, b, c = (np.asarray(x, np.float64) for x in (Af, Bf, Cf))
    U0 = 16.0 - np.asarray(mx, np.float64)
    V0 = 16.0 - np.asarray(my, np.float64)
    P0 = a * U0 * U0 + b * U0 * V0 + c * V0 * V0
    Lx = 2.0 * a * U0 + b * V0
    Ly = b * U0 + 2.0 * c * V0
    return Af, Bf, Cf, f32(P0), f32(Lx), f32(Ly)


def centred_power(Af, Bf, Cf, P0, Lx, Ly, du, dv):
    """fp32 power in the device kernel's order (blend_bwd_compute.cpp bwd_body)."""
    return (P0 + du * (Lx + Af * du + Bf * dv) + dv * (Ly + Cf * dv)).astype(f32)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 (nearest even), returned as float32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def tile_rects(means2d: np.ndarray, conic: np.ndarray, valid: np.ndarray, lowpass_cov: np.ndarray,
               H: int, W: int, tile: int = TILE) -> np.ndarray:
    """(N, 4) int tile rectangle [x0, x1, y0, y1] inclusive; x1 < x0 for none.

    Radii are ceil(3 sqrt(cov2d diagonal)) as rasterization.project_gaussians; the
    rectangle follows gsplat_cpu::tile_assign (truncate, clamp to the grid).
    """
    tx, ty = (W + tile - 1) // tile, (H + tile - 1) // tile
    N = means2d.shape[0]
    out = np.zeros((N, 4), dtype=np.int64)
    out[:, 1] = -1
    for g in range(N):
        if not valid[g]:
            continue
        a, c = lowpass_cov[g]
        rx, ry = math.ceil(3.0 * math.sqrt(max(a, 0.0))), math.ceil(3.0 * math.sqrt(max(c, 0.0)))
        px, py = float(means2d[g, 0]), float(means2d[g, 1])
        if px + rx <= 0 or px - rx >= W or py + ry <= 0 or py - ry >= H:
            continue
        cl = lambda v, hi: min(max(int(v), 0), hi)  # noqa: E731  (C++ static_cast<int> truncates)
        out[g] = (cl((px - rx) / tile, tx - 1), cl((px + rx) / tile, tx - 1),
                  cl((py - ry) / tile, ty - 1), cl((py + ry) / tile, ty - 1))
    return out


def cov_diag_from_conic(conic: np.ndarray) -> np.ndarray:
    """(a, c) of the low-passed 2D covariance from its inverse (A, B, C)."""
    A, B, C = conic[:, 0], conic[:, 1], conic[:, 2]
    det = A * C - B * B
    return np.stack([C / det, A / det], axis=-1)


def blend_backward(means2d, conic, opacity, colors, order, valid, image, dL_dimage,
                   cfg: BlendCfg = BlendCfg(), bf16_inputs: bool = False, tile: int = TILE,
                   rects: np.ndarray | None = None, frozen_live: np.ndarray | None = None) -> dict:
    """Device-decomposition blend backward. Returns per-Gaussian grads (float32) and stats.

    frozen_live (N, H, W) bool: use these live decisions (the oracle's) instead of
    recomputing them, and tile rectangles that cover them. This isolates the backward
    from forward decisions that differ between the fp64 oracle (no tile rectangle, fp64
    alpha) and the device forward (3-sigma tile rectangle, fp32 alpha).
    """
    f32 = np.float32
    H, W, _ = image.shape
    N = means2d.shape[0]
    m2 = np.asarray(means2d, f32)
    con = np.asarray(conic, f32)
    op = np.asarray(opacity, f32)
    col = np.asarray(colors, f32)
    if bf16_inputs:
        m2, con, op, col = (bf16_round(t) for t in (m2, con, op, col))
    if rects is None and frozen_live is not None:
        rects = np.zeros((N, 4), np.int64)
        rects[:, 1] = -1
        for g in range(N):
            ys, xs = np.nonzero(frozen_live[g])
            if xs.size:
                rects[g] = (xs.min() // tile, xs.max() // tile, ys.min() // tile, ys.max() // tile)
    if rects is None:
        rects = tile_rects(np.asarray(means2d, np.float64), conic, valid,
                           cov_diag_from_conic(np.asarray(conic, np.float64)), H, W, tile)
    counts = np.where(rects[:, 1] >= rects[:, 0],
                      (rects[:, 1] - rects[:, 0] + 1) * (rects[:, 3] - rects[:, 2] + 1), 0)
    P = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    slots = np.zeros((int(counts.sum()), N_TERMS), f32)

    img = np.asarray(image, f32)
    gimg = np.asarray(dL_dimage, f32)
    tiles_x, tiles_y = (W + tile - 1) // tile, (H + tile - 1) // tile
    yy, xx = np.meshgrid(np.arange(tile, dtype=f32) + f32(0.5),
                         np.arange(tile, dtype=f32) + f32(0.5), indexing="ij")
    du, dv = xx - f32(16.0), yy - f32(16.0)
    one = f32(1.0)
    n_pairs = 0
    n_live_px = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            ox, oy = tx * tile, ty * tile
            h, w = min(tile, H - oy), min(tile, W - ox)
            pix = np.zeros((tile, tile), bool)
            pix[:h, :w] = True
            I = np.zeros((tile, tile, 3), f32)
            Gp = np.zeros((tile, tile, 3), f32)
            I[:h, :w] = img[oy:oy + h, ox:ox + w]
            Gp[:h, :w] = gimg[oy:oy + h, ox:ox + w]
            T = np.ones((tile, tile), f32)
            Ig = (I * Gp).sum(-1, dtype=f32)        # image . g, once per pixel
            s_acc = np.zeros((tile, tile), f32)     # C_acc . g, carried as one scalar
            alive = pix.copy()
            for g in order:
                g = int(g)
                x0, x1, y0, y1 = rects[g]
                if not (x0 <= tx <= x1 and y0 <= ty <= y1):
                    continue
                n_pairs += 1
                A, B, C = con[g]
                mx, my = m2[g, 0] - f32(ox), m2[g, 1] - f32(oy)   # tile-local mean
                u = xx - mx
                v = yy - my
                power = centred_power(*centred_power_coeffs(A, B, C, mx, my), du, dv)
                wgt = np.exp(np.minimum(power, f32(0.0))).astype(f32)
                raw = op[g] * wgt
                clamp = raw > f32(cfg.alpha_max)
                alpha = np.where(clamp, f32(cfg.alpha_max), raw).astype(f32)
                if frozen_live is None:
                    live = alive & (alpha >= f32(cfg.alpha_min))
                    stop = live & (T * (one - alpha) < f32(cfg.t_min))
                    live &= ~stop
                    alive &= ~stop
                else:
                    live = np.zeros((tile, tile), bool)
                    live[:h, :w] = frozen_live[g, oy:oy + h, ox:ox + w]
                n_live_px += int(live.sum())
                at = np.where(live, alpha * T, f32(0.0)).astype(f32)
                cg = (col[g] * Gp).sum(-1, dtype=f32)
                s_acc = s_acc + at * cg
                behind = Ig - s_acc
                dalpha = T * cg - behind / (one - alpha)
                T = np.where(live, T * (one - alpha), T).astype(f32)
                act = live & ~clamp & (power <= f32(0.0))
                G = np.where(act, dalpha * alpha, f32(0.0)).astype(f32)
                terms = np.stack([
                    np.where(live & ~clamp, dalpha * wgt, f32(0.0)),
                    at * Gp[..., 0], at * Gp[..., 1], at * Gp[..., 2],
                    G * u * u, G * u * v, G * v * v, G * u, G * v,
                ], axis=-1).astype(f32)
                # 32-lane reduce per 4x8 microblock, then microblocks in row-major order.
                mb = terms.reshape(tile // MB_H, MB_H, tile // MB_W, MB_W, 9)
                mb = mb.transpose(0, 2, 1, 3, 4).reshape(-1, MB_H * MB_W, 9).sum(1, dtype=f32)
                s = np.zeros(9, f32)
                for k in range(mb.shape[0]):
                    s += mb[k]
                do, dr, dgc, db, Suu, Suv, Svv, Su, Sv = s
                pair = np.array([A * Su + B * Sv, B * Su + C * Sv,
                                 f32(-0.5) * Suu, -Suv, f32(-0.5) * Svv,
                                 do, dr, dgc, db], f32)
                j = (ty - y0) * (x1 - x0 + 1) + (tx - x0)
                slots[P[g] + j] = pair
    grads = np.zeros((N, N_TERMS), f32)
    for g in range(N):
        acc = np.zeros(N_TERMS, f32)
        for j in range(int(counts[g])):
            acc += slots[P[g] + j]
        grads[g] = acc
    return {"dL_dmeans2d": grads[:, 0:2], "dL_dconic": grads[:, 2:5],
            "dL_dopacity": grads[:, 5], "dL_dcolors": grads[:, 6:9],
            "slots": slots, "P": P, "counts": counts, "rects": rects,
            "n_pairs": n_pairs, "n_live_px": n_live_px}


GRAD_KEYS = ("dL_dmeans2d", "dL_dconic", "dL_dopacity", "dL_dcolors")


def rel_error(ad: np.ndarray, ref: np.ndarray, floor_frac: float = 1e-4) -> float:
    """train_oracle.rel_error: max |ad - ref| / max(|ref|, floor_frac * max|ref|)."""
    a, f = np.asarray(ad, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    scale = np.abs(f).max() if f.size else 0.0
    if scale == 0.0:
        return float(np.abs(a).max()) if a.size else 0.0
    return float((np.abs(a - f) / np.maximum(np.abs(f), floor_frac * scale)).max())


def compare(dump: dict, **kw) -> dict:
    """Run the model on an oracle dump and compare with its blend-backward outputs."""
    cfgd = json.loads(str(dump["cfg_json"]))
    cfg = BlendCfg(cfgd["alpha_max"], cfgd["alpha_min"], cfgd["t_min"], tuple(cfgd["background"]))
    frozen = kw.pop("frozen", False)
    if frozen:
        kw["frozen_live"] = np.asarray(dump["live"], bool).reshape(-1, *np.shape(dump["image"])[:2])
    out = blend_backward(dump["means2d"], dump["conic"], dump["opacity"], dump["colors"],
                         dump["order"], dump["valid"], dump["image"], dump["dL_dimage"], cfg, **kw)
    vis = np.asarray(dump["visible"], bool)
    rep = {"n_pairs": out["n_pairs"], "n_live_px": out["n_live_px"],
           "oracle_live_px": int(np.asarray(dump["live"]).sum())}
    for k in GRAD_KEYS:
        a, f = out[k][vis].astype(np.float64), np.asarray(dump[k])[vis]
        scale = float(np.abs(f).max())
        err = np.abs(a - f)
        rep[k] = {"rel_err": rel_error(a, f),
                  "max_abs_over_max": float(err.max() / scale) if scale else float(err.max()),
                  "mean_abs_over_max": float(err.mean() / scale) if scale else float(err.mean())}
    rep["worst_rel_err"] = max(rep[k]["rel_err"] for k in GRAD_KEYS)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--bar", type=float, default=1e-2, help="PLAN.md T1 gate (rel err)")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    report, ok = {}, True
    for path in a.dumps:
        d = dict(np.load(path))
        # recompute: device-faithful (fp32 live decisions, 3-sigma tile rectangles);
        # frozen: the oracle's live masks; bf16: informational, never gated.
        for mode, kw in (("fp32-recompute", {}), ("fp32-frozen", {"frozen": True}),
                         ("bf16-recompute", {"bf16_inputs": True})):
            r = compare(d, **kw)
            key = f"{path.rsplit('/', 1)[-1]}:{mode}"
            report[key] = r
            if not mode.startswith("bf16"):
                ok &= r["worst_rel_err"] <= a.bar
            print(f"{key}: worst rel err {r['worst_rel_err']:.3e} "
                  f"(pairs {r['n_pairs']}, live px {r['n_live_px']} vs oracle {r['oracle_live_px']})")
            for k in GRAD_KEYS:
                e = r[k]
                print(f"  {k:12s} rel {e['rel_err']:.3e}  max/max {e['max_abs_over_max']:.3e}"
                      f"  mean/max {e['mean_abs_over_max']:.3e}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1)
    print("PASS" if ok else "FAIL", f"(bar {a.bar:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
