#!/usr/bin/env python3
"""Task #408: project the bicycle scene for the 30 bench views (numpy only).

Mirrors gsplat/rasterization.py project_gaussians (EWA, +0.3 low-pass, near
0.2, 3-sigma AABB radii as pfwc, radius cap min(H,W)/2, opacity >= 1/255) and
the PRECULL rectangle shrink (opacity-aware floor-ellipse extent, margin 0.25).
Writes one binary per view for mbsim.cpp:
  header  int32 n, W, H; float32 floor
  n x 12 float32: mx, my, A, B, C, op, r, g, b, depth, tx0|tx1, ty0|ty1
  (A, B, C = pre-folded conic: power = A dx^2 + B dx dy + C dy^2; tile ranges
   stored as float, inclusive)
Usage: project.py OUT_DIR [--start I] [--views N] [--ply PATH]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ply", default=None)
ap.add_argument("--cameras", default=str(REPO / "benchmarks/cameras_v2.json"))
ap.add_argument("--views", type=int, default=0)
ap.add_argument("--start", type=int, default=0)
a = ap.parse_args()

cam = json.load(open(a.cameras))["bicycle"]
ply = a.ply or str(REPO / cam["ply"])
W, H = cam["image_size"]
floor = float(cam["contrib_floor"])
fov = float(cam["fov_deg"])
f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov))
fx = fy = np.float32(f)
cx, cy = np.float32(W * 0.5), np.float32(H * 0.5)

# ---- ply ----
with open(ply, "rb") as fh:
    props, n = [], 0
    while True:
        line = fh.readline().decode().strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property"):
            props.append(line.split()[-1])
        elif line == "end_header":
            break
    off = fh.tell()
v = np.memmap(ply, dtype=np.dtype([(p, "<f4") for p in props]), mode="r", offset=off, shape=(n,))
means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1).astype(np.float32))
q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1).astype(np.float32)
q /= np.linalg.norm(q, axis=-1, keepdims=True)
op = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))).astype(np.float32)
C0 = 0.28209479177387814
rgb = np.clip(0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1), 0, 1).astype(np.float32)
w, x, y, z = q.T
R = np.stack([
    np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
    np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
    np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
M = R * scales[:, None, :]
cov3 = M @ M.transpose(0, 2, 1)  # (n,3,3) float32
del M, R

Path(a.out).mkdir(parents=True, exist_ok=True)
order = cam["order"][a.start:]
order = order[: a.views] if a.views else order
for name in order:
    c2w = np.asarray(cam["views"][name]["c2w"], np.float32)
    Rw = c2w[:3, :3].T
    tw = -Rw @ c2w[:3, 3]
    mc = (means @ Rw.T.astype(np.float64) + tw.astype(np.float64)).astype(np.float32)
    tx, ty, tz = mc.T
    ok = tz > 0.2
    tzs = np.where(ok, tz, 1.0).astype(np.float32)
    mx = fx * tx / tzs + cx
    my = fy * ty / tzs + cy
    J = np.zeros((n, 2, 3), np.float32)
    J[:, 0, 0] = fx / tzs
    J[:, 0, 2] = -fx * tx / (tzs * tzs)
    J[:, 1, 1] = fy / tzs
    J[:, 1, 2] = -fy * ty / (tzs * tzs)
    T = J @ Rw
    cov2 = T @ cov3 @ T.transpose(0, 2, 1)
    ca = cov2[:, 0, 0] + np.float32(0.3)
    cb = cov2[:, 0, 1]
    cc = cov2[:, 1, 1] + np.float32(0.3)
    rx = np.ceil(3.0 * np.sqrt(np.maximum(ca, 0)))
    ry = np.ceil(3.0 * np.sqrt(np.maximum(cc, 0)))
    det = ca * cc - cb * cb
    cap = min(W, H) // 2
    ok &= (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H)
    ok &= (rx > 0) & (ry > 0) & (rx <= cap) & (ry <= cap) & (op >= 1.0 / 255.0) & (det > 0)
    i = np.nonzero(ok)[0]
    ca, cb, cc, det = ca[i].astype(np.float64), cb[i].astype(np.float64), cc[i].astype(np.float64), det[i].astype(np.float64)
    A = (-0.5 * cc / det).astype(np.float32)
    B = (cb / det).astype(np.float32)
    Cc = (-0.5 * ca / det).astype(np.float32)
    # PRECULL: shrink the 3-sigma rect to the floor-ellipse extent (margin 0.25).
    tpc = np.maximum(2.0 * np.log(np.maximum(op[i], 1e-30) / floor) + 0.25, 0.0)
    ex = np.minimum(rx[i], np.sqrt(tpc * ca))
    ey = np.minimum(ry[i], np.sqrt(tpc * cc))
    mxi, myi = mx[i].astype(np.float64), my[i].astype(np.float64)
    tx0 = np.clip(np.floor((mxi - ex) / 32), 0, W // 32 - 1)
    tx1 = np.clip(np.floor((mxi + ex) / 32), 0, W // 32 - 1)
    ty0 = np.clip(np.floor((myi - ey) / 32), 0, H // 32 - 1)
    ty1 = np.clip(np.floor((myi + ey) / 32), 0, H // 32 - 1)
    rec = np.stack([mx[i], my[i], A, B, Cc, op[i], rgb[i, 0], rgb[i, 1], rgb[i, 2], tz[i],
                    tx0 + 1024 * tx1, ty0 + 1024 * ty1], -1).astype(np.float32)
    with open(Path(a.out) / f"{name}.bin", "wb") as fh:
        np.array([len(i), W, H], np.int32).tofile(fh)
        np.array([floor], np.float32).tofile(fh)
        rec.tofile(fh)
    print(name, len(i), flush=True)
