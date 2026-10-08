#!/usr/bin/env python3
"""t437: CPU model of pfwc per-tile work (visible gaussians, pairs) per view for bicycle,
then per-core sums under the fused strided deal (tile t -> core t mod C). Approximate
predicate (z > 0.2, opacity >= 1/255, 3-sigma AABB overlaps the 1024x1024 screen, 0.3 px
dilation); good enough for correlation, not bit-exact with the device classify.
Usage: model437.py PLY CAMERAS_JSON -> tilework.npz (vis[view, tile], pairs[view, tile])."""
import sys, json, math
import numpy as np
ply, camf = sys.argv[1], sys.argv[2]
with open(ply, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"): hdr += f.readline()
    off = len(hdr)
names = [l.split()[-1].decode() for l in hdr.split(b"\n") if l.startswith(b"property")]
N = int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
A = np.memmap(ply, dtype=np.float32, mode="r", offset=off, shape=(N, len(names)))
col = lambda n: np.asarray(A[:, names.index(n)], dtype=np.float32)
xyz = np.stack([col("x"), col("y"), col("z")], 1)
op = 1 / (1 + np.exp(-col("opacity")))
s = np.exp(np.stack([col(f"scale_{i}") for i in range(3)], 1))
q = np.stack([col(f"rot_{i}") for i in range(4)], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
w, x, y, z = q.T
R = np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
              2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
              2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
M = R * s[:, None, :]
cov = (M @ M.transpose(0, 2, 1)).astype(np.float32)  # N,3,3
del M, R
cam = json.load(open(camf))["bicycle"]
W, H = cam["image_size"]; fov = cam["fov_deg"]
f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov)); cx, cy = W / 2, H / 2
T = -(-N // 1024); TS = 32; TX, TY = W // TS, H // TS
keep = op >= 1 / 255
vis = np.zeros((len(cam["order"]), T), np.int32); prs = np.zeros_like(vis)
for vi, name in enumerate(cam["order"]):
    c2w = np.array(cam["views"][name]["c2w"], np.float64)
    Rw = c2w[:3, :3].T; tw = -Rw @ c2w[:3, 3]
    pc = xyz @ Rw.T.astype(np.float32) + tw.astype(np.float32)
    tz = pc[:, 2]
    ok = keep & (tz > 0.2)
    tzs = np.where(ok, tz, 1.0)
    u = f * pc[:, 0] / tzs + cx; v = f * pc[:, 1] / tzs + cy
    J00 = f / tzs; J02 = -f * pc[:, 0] / tzs**2; J11 = f / tzs; J12 = -f * pc[:, 1] / tzs**2
    Rf = Rw.astype(np.float32)
    # cov_cam = Rw cov Rw^T; only need entries for 2D cov
    Cc = np.einsum("ij,njk,lk->nil", Rf, cov, Rf, optimize=True)
    a = J00**2 * Cc[:, 0, 0] + 2 * J00 * J02 * Cc[:, 0, 2] + J02**2 * Cc[:, 2, 2] + 0.3
    c = J11**2 * Cc[:, 1, 1] + 2 * J11 * J12 * Cc[:, 1, 2] + J12**2 * Cc[:, 2, 2] + 0.3
    rx = 3 * np.sqrt(np.maximum(a, 0)); ry = 3 * np.sqrt(np.maximum(c, 0))
    x0 = np.clip(np.floor((u - rx) / TS), 0, TX); x1 = np.clip(np.floor((u + rx) / TS) + 1, 0, TX)
    y0 = np.clip(np.floor((v - ry) / TS), 0, TY); y1 = np.clip(np.floor((v + ry) / TS) + 1, 0, TY)
    npair = np.where(ok, np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0), 0).astype(np.int64)
    isv = npair > 0
    tid = np.arange(N) // 1024
    vis[vi] = np.bincount(tid, weights=isv, minlength=T)
    prs[vi] = np.bincount(tid, weights=npair, minlength=T)
    print(f"view {vi} {name}: visible {isv.sum()} pairs {npair.sum()}", flush=True)
np.savez_compressed("tilework.npz", vis=vis, pairs=prs, order=np.array(cam["order"]))
