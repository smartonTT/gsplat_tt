"""Synthetic blend-backward dump for T1 scaling runs (#693): `python3 docs/t685/synth.py out.npz H W N seed`.

Writes only the keys `gsplat.blend_backward_dev prep/check` read (means2d, conic, opacity,
colors, order, valid, image, dL_dimage, cfg_json). `image` is an fp32 front-to-back forward
with the device rules (tile rectangles, alpha clamp, alpha_min skip, T_min stop), so the
backward's cancellations look like a real view's. Timing workload only, not an oracle case:
the device result is still checked against blend_backward_ref on the same inputs.
"""
import json
import sys

import numpy as np

from gsplat.blend_backward_ref import TILE, BlendCfg, cov_diag_from_conic, tile_rects


def forward(m2, con, op, col, order, valid, H, W, cfg):
    f32 = np.float32
    rects = tile_rects(m2.astype(np.float64), con, valid, cov_diag_from_conic(con.astype(np.float64)), H, W, TILE)
    T = np.ones((H, W), f32)
    C = np.zeros((H, W, 3), f32)
    done = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W].astype(f32) + f32(0.5)
    for g in order:
        x0, x1, y0, y1 = rects[g]
        if x1 < x0:
            continue
        sl = (slice(y0 * TILE, min(H, (y1 + 1) * TILE)), slice(x0 * TILE, min(W, (x1 + 1) * TILE)))
        u, v = xx[sl] - m2[g, 0], yy[sl] - m2[g, 1]
        A, B, Cc = con[g]
        pw = f32(-0.5) * A * (u * u) + (-B) * (u * v) + f32(-0.5) * Cc * (v * v)
        al = np.minimum(op[g] * np.exp(np.minimum(pw, f32(0))), f32(cfg.alpha_max)).astype(f32)
        Tg = T[sl]
        Tn = Tg * (f32(1) - al)
        act = (al >= cfg.alpha_min) & ~done[sl]
        live = act & (Tn >= cfg.t_min)
        C[sl] += np.where(live[..., None], (al * Tg)[..., None] * col[g], f32(0))
        T[sl] = np.where(live, Tn, Tg)
        done[sl] |= act & ~live
    return C + T[..., None] * np.asarray(cfg.background, f32)


def main():
    out, H, W, N, seed = sys.argv[1], *map(int, sys.argv[2:6])
    rng = np.random.default_rng(seed)
    f32 = np.float32
    cfg = BlendCfg()
    m2 = np.stack([rng.uniform(-8, W + 8, N), rng.uniform(-8, H + 8, N)], 1).astype(f32)
    s = np.exp(rng.uniform(np.log(0.8), np.log(10.0), (N, 2)))
    th = rng.uniform(0, np.pi, N)
    c, sn = np.cos(th), np.sin(th)
    cov = np.empty((N, 2, 2))
    cov[:, 0, 0] = c * c * s[:, 0] ** 2 + sn * sn * s[:, 1] ** 2
    cov[:, 1, 1] = sn * sn * s[:, 0] ** 2 + c * c * s[:, 1] ** 2
    cov[:, 0, 1] = cov[:, 1, 0] = c * sn * (s[:, 0] ** 2 - s[:, 1] ** 2)
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    inv = np.linalg.inv(cov)
    con = np.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], 1).astype(f32)
    op = rng.uniform(0.05, 0.99, N).astype(f32)
    col = rng.uniform(0, 1, (N, 3)).astype(f32)
    order = rng.permutation(N).astype(np.int64)
    valid = np.ones(N, bool)
    img = forward(m2, con, op, col, order, valid, H, W, cfg)
    g = (rng.standard_normal((H, W, 3)) * 1e-3).astype(f32)
    np.savez(out, H=H, W=W, N=N, seed=seed, means2d=m2, conic=con, opacity=op, colors=col, order=order,
             valid=valid, image=img, dL_dimage=g,
             cfg_json=json.dumps({"alpha_max": cfg.alpha_max, "alpha_min": cfg.alpha_min, "t_min": cfg.t_min,
                                  "background": list(cfg.background)}))


if __name__ == "__main__":
    main()
