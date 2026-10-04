"""Lever C (task #140): host estimate of the dead records the opacity-aware rect removes.

Projects bicycle for one view (docs/r16-record/aniso_hero.py math), expands every visible
gaussian's 3-sigma tile rectangle into (gaussian, tile) records, and marks a record live
when the alpha >= floor ellipse (m2 <= 2 ln(op / floor) + 0.05, the band cull's test)
meets the tile's pixel-centre box. Then counts the records left by the pre-cull rect
r' = min(r, k_e sqrt(cov) + slack), k_e = sqrt(max(2 ln(op / floor) + margin, 0)), and
checks that no live record falls outside it. Host float64 model, not the device:
an estimate of the share, not a measurement.
Usage (repo root): python3 docs/precull-t140/estimate.py [view] [ply]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r16-record"))
from aniso_hero import load_ply, quat_to_rot  # noqa: E402

CAM = "benchmarks/cameras_v2.json"
TILE = 32
MARGIN = 0.25   # == PRECULL_T_MARGIN in project_pfwc_compute.cpp
SLACK = 1.0     # == PRECULL_SLACK (pixels)
COND = 64.0     # == PRECULL_COND: shrink only when COND * det >= a * c


_PLY = {}


def project(view, ply):
    cam = json.load(open(CAM))["bicycle"]
    if ply not in _PLY:
        _PLY[ply] = load_ply(ply)
    W, H = cam["image_size"]
    floor = cam["contrib_floor"]
    c2w = np.array(cam["views"][view]["c2w"], np.float64)
    w2c = np.linalg.inv(c2w)
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op, _ = _PLY[ply]
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    fr = z > 0.01
    pc, z, scales, q, op = pc[fr], z[fr], scales[fr], q[fr], op[fr]
    lim = 1.3 * 0.5 * W / f
    tx = np.clip(pc[:, 0] / z, -lim, lim) * z
    ty = np.clip(pc[:, 1] / z, -lim, lim) * z
    j00 = f / z; j02 = -f * tx / (z * z); j11 = f / z; j12 = -f * ty / (z * z)
    M = quat_to_rot(q) * scales[:, None, :]
    Rc = w2c[:3, :3]
    covc = Rc[None] @ (M @ np.transpose(M, (0, 2, 1))) @ Rc.T[None]
    a = j00 * j00 * covc[:, 0, 0] + 2 * j00 * j02 * covc[:, 0, 2] + j02 * j02 * covc[:, 2, 2] + 0.3
    b = (j00 * j11 * covc[:, 0, 1] + j00 * j12 * covc[:, 0, 2] + j02 * j11 * covc[:, 2, 1]
         + j02 * j12 * covc[:, 2, 2])
    c = j11 * j11 * covc[:, 1, 1] + 2 * j11 * j12 * covc[:, 1, 2] + j12 * j12 * covc[:, 2, 2] + 0.3
    mx = f * pc[:, 0] / z + 0.5 * W
    my = f * pc[:, 1] / z + 0.5 * H
    rx = np.ceil(3 * np.sqrt(a)); ry = np.ceil(3 * np.sqrt(c))
    vis = (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H) & (op > floor)
    vis &= (a * c - b * b) > 0
    return [x[vis] for x in (a, b, c, mx, my, op, rx, ry)], W, H, floor


def cells(m, r, n):
    return (np.clip(np.floor((m - r) / TILE), 0, n - 1).astype(np.int64),
            np.clip(np.floor((m + r) / TILE), 0, n - 1).astype(np.int64))


def box_min_m2(ia, ib, ic, u_lo, u_hi, v_lo, v_hi):
    """min of ia u^2 + 2 ib u v + ic v^2 over the box (box relative to the mean)."""
    xin = (u_lo <= 0) & (u_hi >= 0)
    yin = (v_lo <= 0) & (v_hi >= 0)
    best = np.where(xin & yin, 0.0, np.inf)
    for u in (u_lo, u_hi):  # vertical edges
        v = np.clip(-ib * u / ic, v_lo, v_hi)
        best = np.minimum(best, ia * u * u + 2 * ib * u * v + ic * v * v)
    for v in (v_lo, v_hi):  # horizontal edges
        u = np.clip(-ib * v / ia, u_lo, u_hi)
        best = np.minimum(best, ia * u * u + 2 * ib * u * v + ic * v * v)
    return best


def one(view, ply):
    (a, b, c, mx, my, op, rx, ry), W, H, floor = project(view, ply)
    tx_n, ty_n = W // TILE, H // TILE
    x0, x1 = cells(mx, rx, tx_n)
    y0, y1 = cells(my, ry, ty_n)
    w = x1 - x0 + 1; h = y1 - y0 + 1
    n = w * h
    P = int(n.sum())
    g = np.repeat(np.arange(n.size), n)
    loc = np.arange(P) - np.repeat(np.cumsum(n) - n, n)
    tx = x0[g] + loc % w[g]
    ty = y0[g] + loc // w[g]
    det = a * c - b * b
    ia, ib, ic = c / det, -b / det, a / det
    t = 2 * np.log(op / floor)
    m2 = box_min_m2(ia[g], ib[g], ic[g],
                    tx * TILE + 0.5 - mx[g], tx * TILE + 31.5 - mx[g],
                    ty * TILE + 0.5 - my[g], ty * TILE + 31.5 - my[g])
    live = m2 <= t[g] + 0.05
    # pre-cull rect
    ke = np.sqrt(np.maximum(t + MARGIN, 0.0))
    well = COND * det >= a * c
    rpx = np.where(well, np.minimum(rx, ke * np.sqrt(a) + SLACK), rx)
    rpy = np.where(well, np.minimum(ry, ke * np.sqrt(c) + SLACK), ry)
    px0, px1 = cells(mx, rpx, tx_n)
    py0, py1 = cells(my, rpy, ty_n)
    inside = (tx >= px0[g]) & (tx <= px1[g]) & (ty >= py0[g]) & (ty <= py1[g])
    P2 = int(inside.sum())
    lost = int((live & ~inside).sum())
    print(f"view {view}: visible {n.size} records {P} live {int(live.sum())} "
          f"dead {P - int(live.sum())} ({(P - live.sum()) / P:.4f})")
    print(f"  shrinkable gaussians (k_e < 3): {(ke < 3).mean():.4f}; ill-conditioned (kept 3 sigma): "
          f"{(~well).mean():.4f}")
    print(f"  pre-cull rect: records {P2} removed {P - P2} ({(P - P2) / P:.4f} of all, "
          f"{(P - P2) / max(P - live.sum(), 1):.4f} of dead); live records lost {lost}")
    # follow-up model: sheared row band |u - (b/c) v| <= hw, hw = sqrt(t det / c) (+slack)
    s = b / c
    hw = np.sqrt(np.maximum(t + MARGIN, 0.0) * det / c) + SLACK
    v_lo = ty * TILE + 0.5 - my[g]; v_hi = ty * TILE + 31.5 - my[g]
    u_c_lo = np.minimum(s[g] * v_lo, s[g] * v_hi) - hw[g]
    u_c_hi = np.maximum(s[g] * v_lo, s[g] * v_hi) + hw[g]
    shear = ((tx * TILE + 31.5 - mx[g]) >= u_c_lo) & ((tx * TILE + 0.5 - mx[g]) <= u_c_hi) & inside
    print(f"  + sheared row band: removed {P - int(shear.sum())} ({(P - shear.sum()) / P:.4f}); "
          f"live lost {int((live & ~shear).sum())}")
    tid = ty * tx_n + tx
    cnt0 = np.bincount(tid, minlength=tx_n * ty_n)
    cnt1 = np.bincount(tid[inside], minlength=tx_n * ty_n)
    print(f"  max tile {cnt0.max()} -> {cnt1.max()}; tiles > 8192: {(cnt0 > 8192).sum()} -> "
          f"{(cnt1 > 8192).sum()}; > 16384: {(cnt0 > 16384).sum()} -> {(cnt1 > 16384).sum()}")
    return P, P - P2, P - int(shear.sum()), P - int(live.sum())


def main():
    views = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hero"]
    ply = sys.argv[2] if len(sys.argv) > 2 else "scenes/bicycle.ply"
    if views == ["all"]:
        views = json.load(open(CAM))["bicycle"]["order"]
    tot = np.zeros(4)
    for v in views:
        tot += one(v, ply)
    if len(views) > 1:
        print(f"ALL {len(views)} views: records {int(tot[0])} dead {tot[3] / tot[0]:.4f} "
              f"rect removes {tot[1] / tot[0]:.4f} rect+shear removes {tot[2] / tot[0]:.4f}")


if __name__ == "__main__":
    main()
