#!/usr/bin/env python3
"""Task #675: saturation-depth cull, step 0 (CPU sizing, no device).

Per view: project the bicycle model (same math as gsplat.rasterization.project_gaussians),
emit (tile, gaussian) pairs with the per-pair peak-contribution cull, sort by (tile, depth)
and scan each tile front to back until every pixel has T < eps. Then predict each tile's
cut depth from the PREVIOUS view: every saturated pixel of the previous view is lifted to a
3D point (pixel ray at the saturating gaussian's depth), forward-warped into the current view,
and carries that gaussian's current-view depth key. A tile gets a cut only if every 4x4 cell
in it received a warped sample and none came from an unsaturated pixel.

Usage: step0.py --cams FILE --out DIR [--first A --last B] [--max-views N]
Writes DIR/<set>_view<i>.json per view and caches DIR/<set>_pixg<i>.npy for the next view.
"""
import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TS = 32
EPS = 0.00390625  # device BLEND_T_EPS (1/256); the CPU reference uses 1/255
MARGINS = (0.0, 0.02, 0.05)
SLACKS = (0, 1)
PERIOD = 512  # device BLEND_T_PERIOD: live-mask refresh every 512 gaussians of a tile


def build_lib():
    so = Path(os.environ.get("TTP_RUN_DIR", "/tmp")) / "t675_satcull.so"
    if not so.exists():
        sdk = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk"
        cmd = ["cc", "-O3", "-shared", "-fPIC", "-o", str(so), str(HERE / "satcull.c")]
        if os.path.isdir(sdk):
            cmd[1:1] = ["-isysroot", sdk]
        subprocess.check_call(cmd)
    lib = ctypes.CDLL(str(so))
    P = ctypes.c_void_p
    lib.emit_pairs.restype = ctypes.c_int64
    lib.emit_pairs.argtypes = [ctypes.c_int] + [P] * 8 + [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                         P, P, ctypes.c_int64]
    lib.sat_scan.restype = None
    lib.sat_scan.argtypes = [ctypes.c_int] * 4 + [P] * 10 + [ctypes.c_float, P, P, P]
    return lib


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def load_ply(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                assert line.split()[1] == "float", line
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        off = f.tell()
    raw = np.fromfile(path, dtype=np.float32, offset=off, count=n * len(props)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    g = lambda *ks: np.ascontiguousarray(raw[:, [col[k] for k in ks]], dtype=np.float32)
    means = g("x", "y", "z")
    scales = np.exp(g("scale_0", "scale_1", "scale_2"))
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    op = 1.0 / (1.0 + np.exp(-np.ascontiguousarray(raw[:, col["opacity"]])))
    del raw
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    RS = R * scales[:, None, :]
    cov3 = np.matmul(RS, RS.transpose(0, 2, 1)).astype(np.float32)
    return means, cov3, op.astype(np.float32)


def cam_mats(cam, view):
    W, H = cam["image_size"]
    if "intrinsics" in cam:
        k = cam["intrinsics"]
        fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    else:
        f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
        fx = fy = f
        cx, cy = W * 0.5, H * 0.5
    c2w = np.asarray(cam["views"][view]["c2w"], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
    return W, H, (fx, fy, cx, cy), c2w, w2c


def project(means, cov3, op, W, H, K, w2c, floor):
    fx, fy, cx, cy = K
    R = w2c[:3, :3].astype(np.float32)  # noqa: N806
    t = w2c[:3, 3].astype(np.float32)
    mc = means @ R.T + t
    valid = (mc[:, 2] > 0.2) & (op >= 1.0 / 255.0)
    idx = np.nonzero(valid)[0]
    tx, ty, tz = mc[idx, 0], mc[idx, 1], mc[idx, 2]
    mx = fx * tx / tz + cx
    my = fy * ty / tz + cy
    # T = J @ R (rows t0, t1), cov2 = T Sigma T^T, all elementwise
    S = cov3[idx]
    j00, j02 = fx / tz, -fx * tx / (tz * tz)
    j11, j12 = fy / tz, -fy * ty / (tz * tz)
    t0 = [j00 * R[0, k] + j02 * R[2, k] for k in range(3)]
    t1 = [j11 * R[1, k] + j12 * R[2, k] for k in range(3)]
    quad = lambda p, q: sum(p[i] * q[j] * S[:, i, j] for i in range(3) for j in range(3))
    a = quad(t0, t0) + 0.3
    b = quad(t0, t1)
    c = quad(t1, t1) + 0.3
    o = op[idx]
    k = np.minimum(np.sqrt(2.0 * np.log(np.maximum(o / floor, 1.0))), 3.0)
    rx = np.ceil(k * np.sqrt(np.maximum(a, 0)))
    ry = np.ceil(k * np.sqrt(np.maximum(c, 0)))
    cap = min(W, H) // 2
    keep = ((mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H) & (rx > 0) & (ry > 0)
            & (rx <= cap) & (ry <= cap))
    det = np.maximum(a * c - b * b, 1e-6)
    f32 = lambda v: np.ascontiguousarray(v[keep], dtype=np.float32)
    return dict(gid=idx[keep].astype(np.int64), mx=f32(mx), my=f32(my), A=f32(c / det), B=f32(-b / det),
                C=f32(a / det), op=f32(o), rx=f32(rx), ry=f32(ry), sxx=f32(a), syy=f32(c), z=f32(tz))


def render_scan(lib, pr, W, H, floor, nthreads):
    n = len(pr["mx"])
    cap = n * 8 + 1_000_000
    while True:
        tile = np.empty(cap, np.int32)
        g = np.empty(cap, np.int32)
        P = lib.emit_pairs(n, ptr(pr["mx"]), ptr(pr["my"]), ptr(pr["A"]), ptr(pr["B"]), ptr(pr["C"]),
                           ptr(pr["op"]), ptr(pr["rx"]), ptr(pr["ry"]), W, H, floor, ptr(tile), ptr(g), cap)
        if P >= 0:
            break
        cap *= 2
    tile, g = tile[:P], g[:P]
    order = np.lexsort((pr["z"][g], tile))
    tile, g = tile[order], np.ascontiguousarray(g[order])
    tx_n, ty_n = (W + TS - 1) // TS, (H + TS - 1) // TS
    nt = tx_n * ty_n
    cnt = np.bincount(tile, minlength=nt).astype(np.int64)
    off = np.zeros(nt + 1, np.int64)
    off[1:] = np.cumsum(cnt)
    tile_sat = np.empty(nt, np.int32)
    pix_g = np.empty(W * H, np.int32)
    blended = np.empty(nt, np.int64)
    lib.sat_scan(nt, W, H, nthreads, ptr(off), ptr(g), ptr(pr["mx"]), ptr(pr["my"]), ptr(pr["A"]),
                 ptr(pr["B"]), ptr(pr["C"]), ptr(pr["op"]), ptr(pr["sxx"]), ptr(pr["syy"]), EPS,
                 ptr(tile_sat), ptr(pix_g), ptr(blended))
    depth = pr["z"][g]
    # global gaussian id per pixel (-1: pixel never reached T < eps)
    pix_gg = np.where(pix_g >= 0, pr["gid"][np.maximum(pix_g, 0)], -1)
    return dict(tile=tile, depth=depth, cnt=cnt, off=off, tile_sat=tile_sat, pix_gg=pix_gg, nt=nt,
                tx_n=tx_n, ty_n=ty_n)


def predict_cuts(means, prev, cur, sc):
    """Per-tile predicted cut depth for the current view from the previous view's pixels."""
    Wp, Hp, Kp, c2w_p, w2c_p, pix_gg = prev
    W, H, K, c2w, w2c = cur
    fx, fy, cx, cy = Kp
    v, u = np.divmod(np.arange(Wp * Hp), Wp)
    sat = pix_gg >= 0
    gg = np.maximum(pix_gg, 0)
    mw = means[gg].astype(np.float64)
    zp = mw @ w2c_p[2, :3] + w2c_p[2, 3]
    zray = np.where(sat, zp, 1e4)  # unsaturated pixel: far point (rotation-like warp)
    pc = np.stack([(u + 0.5 - cx) / fx * zray, (v + 0.5 - cy) / fy * zray, zray], -1)
    Xw = pc @ c2w_p[:3, :3].T + c2w_p[:3, 3]
    Xc = Xw @ w2c[:3, :3].T + w2c[:3, 3]
    key = np.where(sat, mw @ w2c[2, :3] + w2c[2, 3], np.inf)
    fx2, fy2, cx2, cy2 = K
    ok = Xc[:, 2] > 0.2
    u2 = np.where(ok, fx2 * Xc[:, 0] / np.where(ok, Xc[:, 2], 1) + cx2, -1)
    v2 = np.where(ok, fy2 * Xc[:, 1] / np.where(ok, Xc[:, 2], 1) + cy2, -1)
    ok &= (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    cw, chh = (W + 3) // 4, (H + 3) // 4
    cell = (v2[ok].astype(np.int64) // 4) * cw + (u2[ok].astype(np.int64) // 4)
    cellmax = np.full(cw * chh, -np.inf)
    np.maximum.at(cellmax, cell, key[ok])
    hole_cell = cellmax == -np.inf
    tx_n, ty_n = sc["tx_n"], sc["ty_n"]
    cy_i, cx_i = np.divmod(np.arange(cw * chh), cw)
    ct = (cy_i // 8) * tx_n + (cx_i // 8)
    nt = tx_n * ty_n
    tmax = np.full(nt, -np.inf)
    np.maximum.at(tmax, ct, np.where(hole_cell, -np.inf, cellmax))
    hole = np.zeros(nt, bool)
    np.logical_or.at(hole, ct, hole_cell)
    cuts = {}
    for s in SLACKS:
        t2 = tmax.reshape(ty_n, tx_n)
        if s:
            pad = np.pad(t2, s, constant_values=-np.inf)
            t2 = np.max(np.stack([pad[dy:dy + ty_n, dx:dx + tx_n] for dy in range(2 * s + 1)
                                  for dx in range(2 * s + 1)]), 0)
        base = np.where(hole, np.inf, t2.ravel())
        for m in MARGINS:
            cuts[(m, s)] = base * (1.0 + m)
    return cuts


def score(sc, cut):
    tile, depth, cnt, off, tile_sat = sc["tile"], sc["depth"], sc["cnt"], sc["off"], sc["tile_sat"]
    nt = sc["nt"]
    satk = np.full(nt, np.inf)
    s = tile_sat >= 0
    satk[s] = depth[off[:-1][s] + tile_sat[s]]
    le = np.bincount(tile, weights=(depth <= cut[tile]), minlength=nt)
    has = np.isfinite(cut) & (cnt > 0)
    succ = has & (satk <= cut)
    fail = has & ~succ
    return dict(cut_tiles=int(has.sum()), succ_tiles=int(succ.sum()), fail_tiles=int(fail.sum()),
                saved_pairs=int((cnt - le)[succ].sum()), wasted_pairs=int(le[fail].sum()),
                fail_full_pairs=int(cnt[fail].sum()))


def step_angle(c2w_a, c2w_b):
    Rr = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rr) - 1) / 2))))
    return ang, float(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", required=True)
    ap.add_argument("--ply", default="/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=10**9)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = json.load(open(a.cams))["bicycle"]
    order = cam["order"]
    floor = float(cam.get("contrib_floor", 1.0 / 255.0))
    lib = build_lib()
    t0 = time.time()
    means, cov3, op = load_ply(a.ply)
    print(f"loaded {len(means)} gaussians in {time.time() - t0:.1f}s", flush=True)
    for i in range(a.first, min(a.last, len(order))):
        t1 = time.time()
        W, H, K, c2w, w2c = cam_mats(cam, order[i])
        pr = project(means, cov3, op, W, H, K, w2c, floor)
        sc = render_scan(lib, pr, W, H, floor, a.threads)
        cnt, ts = sc["cnt"], sc["tile_sat"]
        P = int(cnt.sum())
        satt = ts >= 0
        behind = int((cnt[satt] - ts[satt] - 1).sum())
        nxt = np.minimum(cnt[satt], ((ts[satt] + 1 + PERIOD - 1) // PERIOD) * PERIOD)
        lag = int((nxt - ts[satt] - 1).sum())
        rec = dict(set=a.set, i=i, view=order[i], W=W, H=H, pairs=P, visible=int(len(pr["mx"])),
                   tiles=int(sc["nt"]), tiles_nonempty=int((cnt > 0).sum()), tiles_sat=int(satt.sum()),
                   pairs_behind=behind, pairs_behind_in_period_lag=lag,
                   pix_unsat=int((sc["pix_gg"] < 0).sum()))
        np.save(out / f"{a.set}_pixg{i}.npy", sc["pix_gg"])
        prev_f = out / f"{a.set}_pixg{i - 1}.npy"
        if i > 0 and prev_f.exists():
            Wp, Hp, Kp, c2w_p, w2c_p = cam_mats(cam, order[i - 1])
            ang, dist = step_angle(c2w_p, c2w)
            rec.update(step_deg=ang, step_dist=dist)
            cuts = predict_cuts(means, (Wp, Hp, Kp, c2w_p, w2c_p, np.load(prev_f)), (W, H, K, c2w, w2c), sc)
            rec["pred"] = {f"m{m}_s{s}": score(sc, c) for (m, s), c in cuts.items()}
            if i - 2 >= 0 and (out / f"{a.set}_pixg{i - 2}.npy").exists():
                os.remove(out / f"{a.set}_pixg{i - 2}.npy")
        json.dump(rec, open(out / f"{a.set}_view{i:03d}.json", "w"))
        best = rec.get("pred", {}).get("m0.02_s1", {})
        print(f"[{a.set} {i} {order[i]}] pairs={P} behind={behind / P:.3f} lag={lag / P:.3f} "
              f"step={rec.get('step_deg', 0):.2f}deg saved={best.get('saved_pairs', 0) / P:.3f} "
              f"wasted={best.get('wasted_pairs', 0) / P:.3f} t={time.time() - t1:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
