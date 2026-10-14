#!/usr/bin/env python3
"""t667: summarize the training-view bench (docs/training-bench/out).

  python3 docs/training-bench/analyze.py [--out docs/training-bench/out]
      [--images <mipnerf360/bicycle/images_4>] [--ply <bicycle.ply>]

Prints per-arm b2b ms/view (median of the 4 rounds' medians, min/max over rounds),
per-view latency stage spread from the pv-* runs (GSPLAT_PER_VIEW_STAGES=1), the
slowest views with their in-frustum Gaussian count (--ply), and PSNR of the dumped
train views vs the dataset photo (--images)."""
import argparse
import glob
import json
import os
import re
import statistics as st

import numpy as np

ARMS = ["orbit", "orb1237", "shuf", "nn", "test"]
CAMS = {"orbit": "cameras_v2", "orb1237": "cameras_orbit_1237", "shuf": "cameras_train_shuffled",
        "nn": "cameras_train_nn", "test": "cameras_test"}


def b2b(out):
    res = {a: [] for a in ARMS}
    for f in sorted(glob.glob(f"{out}/run-r[1-4]-*.log")):
        arm = re.search(r"run-r\d-(\w+)\.log", f).group(1)
        for line in open(f):
            if line.startswith("B2B "):
                kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
                res[arm].append((float(kv["ms_frame_median"]), float(kv["ms_frame"]),
                                 kv["raw_md5"], kv["identical_across_passes"], kv["n_views"]))
    return res


def view_stages(out, arm):
    rows = []
    f = f"{out}/run-r5-pv-{arm}.log"
    if not os.path.exists(f):
        return rows
    for line in open(f):
        if line.startswith("VIEW_STAGES"):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            rows.append(kv)
    return rows[1:]  # first view carries one-off setup deltas


def load_ply_xyz(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            l = f.readline().decode().strip()
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            elif l.startswith("property"):
                props.append(l.split()[-1])
            elif l == "end_header":
                break
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    return data[:, :3].astype(np.float64)


def frustum_count(xyz, view, cam):
    W, H = cam["image_size"]
    c2w = np.asarray(view["c2w"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    p = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    intr = cam.get("intrinsics")
    if intr:
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    else:
        fx = fy = 0.5 * max(W, H) / np.tan(np.radians(cam["fov_deg"]) / 2)
        cx, cy = W / 2, H / 2
    z = p[:, 2]
    ok = z > 0.2
    u = fx * p[ok, 0] / z[ok] + cx
    v = fy * p[ok, 1] / z[ok] + cy
    return int(((u > -0.15 * W) & (u < 1.15 * W) & (v > -0.15 * H) & (v < 1.15 * H)).sum())


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/training-bench/out")
    ap.add_argument("--images", default=None)
    ap.add_argument("--ply", default=None)
    a = ap.parse_args()
    res = b2b(a.out)
    print("## b2b ms/view (per-round median of measured passes)")
    for arm in ARMS:
        r = res[arm]
        if not r:
            print(f"{arm}: no data")
            continue
        med = [x[0] for x in r]
        m = st.median(med)
        print(f"{arm:8s} n_views={r[0][4]:>3s} rounds={len(r)} median={m:.3f} ms/view "
              f"({1000 / m:.1f} FPS) min={min(med):.3f} max={max(med):.3f} "
              f"rounds=[{', '.join(f'{x:.3f}' for x in med)}] md5={sorted(set(x[2] for x in r))} "
              f"identical={sorted(set(x[3] for x in r))}")
    xyz = load_ply_xyz(a.ply) if a.ply else None
    print("\n## latency per-view stages (pv runs, GSPLAT_PER_VIEW_STAGES=1, first view dropped)")
    for arm in ["orbit", "shuf", "nn", "test"]:
        rows = view_stages(a.out, arm)
        if not rows:
            continue
        keys = [k for k in rows[0] if k not in ("i", "name")]
        allk = sorted({k for r in rows for k in r if k not in ("i", "name")})
        print(f"### {arm} ({len(rows)} views)")
        for k in ["wall"] + [k for k in allk if k != "wall"]:
            vals = [float(r.get(k, 0.0)) for r in rows]
            print(f"  {k:28s} min={min(vals):7.2f} med={st.median(vals):7.2f} max={max(vals):7.2f}")
        slow = sorted(rows, key=lambda r: -float(r["wall"]))
        cam = None
        if xyz is not None:
            cam = json.load(open(f"benchmarks/{CAMS[arm]}.json"))["bicycle"]
        for tagname, sel in (("slowest", slow[:5]), ("fastest", slow[-3:])):
            for r in sel:
                extra = ""
                if cam is not None:
                    extra = f" in_frustum={frustum_count(xyz, cam['views'][r['name']], cam)}"
                print(f"  {tagname} {r['name']} wall={r['wall']}{extra}")
        del keys
    if a.images:
        from PIL import Image
        print("\n## dumped train views vs dataset photo (images_4, 1237x822)")
        for png in sorted(glob.glob(f"{a.out}/dump/*.png")):
            name = re.search(r"view\d+_(.+)\.png", os.path.basename(png)).group(1)
            jpg = os.path.join(a.images, name + ".JPG")
            dev = np.asarray(Image.open(png).convert("RGB"))
            if not os.path.exists(jpg):
                print(f"{name}: no photo {jpg}")
                continue
            ph = np.asarray(Image.open(jpg).convert("RGB"))
            if ph.shape != dev.shape:
                print(f"{name}: shape {dev.shape} vs photo {ph.shape}")
                continue
            print(f"{name}: PSNR vs dataset photo = {psnr(dev, ph):.2f} dB")


if __name__ == "__main__":
    main()
