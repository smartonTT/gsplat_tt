#!/usr/bin/env python3
"""t528: how much per-view work repeats across the 30 bicycle views (host-side CPU model,
not the device's exact cull). For each view, in run order (cameras_v2.json 'order'):
  vis   = opacity >= contrib_floor, z > 0.2, and the 3-sigma circle (r = 3*max_scale*f/z)
          touches the 1024x1024 image
  pairs = (gaussian, 32x32 tile) for every tile the r-box covers
Reports per view: M_model, P_model, and against the previous view: visible-set Jaccard, pair-set
Jaccard, and depth-order coherence inside each tile (fraction of adjacent depth-sorted pairs of
the previous view, among pairs kept by both views, that stay in order in this view).
  venv/bin/python docs/profile-t224/xview.py [scene.ply]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gsplat.loading_gaussians import load_ply  # noqa: E402

cam = json.load(open(REPO / "benchmarks/cameras_v2.json"))["bicycle"]
ply = sys.argv[1] if len(sys.argv) > 1 else str(REPO / cam["ply"])
W, H = cam["image_size"]; fov = float(cam["fov_deg"]); floor = float(cam["contrib_floor"])
f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov)); cx, cy = W / 2, H / 2
t0 = time.time(); g = load_ply(ply)
mu = g.means.numpy().astype(np.float64); N = len(mu)
smax = g.scales.numpy().max(1).astype(np.float64); op = g.opacities.numpy()
keep_op = op >= floor
print(f"scene N={N} opacity>=floor {keep_op.sum()} load {time.time()-t0:.1f}s", flush=True)
TS = 32; TX, TY = W // TS, H // TS

def view(c2w):
    c2w = np.asarray(c2w, dtype=np.float64); R = c2w[:3, :3]; t = c2w[:3, 3]
    pc = (mu - t) @ R            # world -> camera (R^T (x - t))
    z = pc[:, 2]; ok = keep_op & (z > 0.2)
    zz = np.where(ok, z, 1.0)
    u = f * pc[:, 0] / zz + cx; v = f * pc[:, 1] / zz + cy; r = 3.0 * smax * f / zz
    ok &= (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H)
    idx = np.nonzero(ok)[0]
    x0 = np.clip(np.floor((u[idx] - r[idx]) / TS), 0, TX - 1).astype(np.int64)
    x1 = np.clip(np.floor((u[idx] + r[idx]) / TS), 0, TX - 1).astype(np.int64)
    y0 = np.clip(np.floor((v[idx] - r[idx]) / TS), 0, TY - 1).astype(np.int64)
    y1 = np.clip(np.floor((v[idx] + r[idx]) / TS), 0, TY - 1).astype(np.int64)
    w = x1 - x0 + 1; h = y1 - y0 + 1; cnt = w * h
    gid = np.repeat(idx, cnt); k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ww = np.repeat(w, cnt); tx = np.repeat(x0, cnt) + k % ww; ty = np.repeat(y0, cnt) + k // ww
    key = (ty * TX + tx) * np.int64(N) + gid          # tile-major pair key
    return idx, np.sort(key), z

order = cam["order"]; prev = None; rows = []
for i, name in enumerate(order):
    idx, key, z = view(cam["views"][name]["c2w"])
    row = dict(i=i, name=name, M=len(idx), P=len(key))
    if prev is not None:
        pidx, pkey, pz = prev
        inter = np.intersect1d(idx, pidx, assume_unique=True).size
        row["vis_jacc"] = inter / (len(idx) + len(pidx) - inter)
        shared = np.intersect1d(key, pkey, assume_unique=True)
        row["pair_jacc"] = shared.size / (len(key) + len(pkey) - shared.size)
        row["pair_shared_of_cur"] = shared.size / len(key)
        tile = shared // N; gid = shared % N
        o = np.lexsort((pz[gid], tile)); tile, gid = tile[o], gid[o]   # prev view's per-tile depth order
        same = tile[1:] == tile[:-1]
        inord = z[gid[1:]] >= z[gid[:-1]]
        row["adj_in_order"] = float(inord[same].mean())
    rows.append(row); prev = (idx, key, z)
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
allvis = np.zeros(N, bool); cntv = np.zeros(N, np.int32)
for name in order:
    idx, _, _ = view(cam["views"][name]["c2w"]); allvis[idx] = True; cntv[idx] += 1
print(f"SUMMARY N={N} union_visible={allvis.sum()} ({allvis.mean():.3f}) "
      f"visible_in_all_30={(cntv == len(order)).sum()} never_visible={(~allvis).sum()} "
      f"mean_M={np.mean([r['M'] for r in rows]):.0f} mean_P={np.mean([r['P'] for r in rows]):.0f} "
      f"mean_vis_jacc={np.mean([r['vis_jacc'] for r in rows[1:]]):.3f} "
      f"mean_pair_jacc={np.mean([r['pair_jacc'] for r in rows[1:]]):.3f} "
      f"mean_adj_in_order={np.mean([r['adj_in_order'] for r in rows[1:]]):.4f}")
