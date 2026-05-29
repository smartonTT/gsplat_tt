"""Diagnose device-blend striping: per-row MAE of blend_mode=2 vs oracle.

Renders view 0 of bicycle with device mode=2 and CPU-oracle mode=1 and prints
per-row mean abs error so we can read off the stripe period (face=16, etc.).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.a003_verify as v  # noqa: E402


def render_view0(mode, gauss, order, views, fov_deg, W, H, contrib_floor):
    os.environ["GSPLAT_TT_BLEND_MODE"] = str(mode)
    imgs, _ = v.render_all("tt", gauss, [order[0]], views, fov_deg, W, H,
                           contrib_floor)
    return imgs[order[0]]


def main():
    cam = json.loads(Path("benchmarks/cameras_v2.json").read_text())["bicycle"]
    fov_deg = float(cam["fov_deg"])
    W, H = cam["image_size"]
    contrib_floor = cam.get("contrib_floor", 1.0 / 255.0)
    order = cam["order"]
    views = cam["views"]
    gauss = v.load_ply(str(Path(cam["ply"])))

    print("rendering mode=2 (device)...", flush=True)
    dev = render_view0(2, gauss, order, views, fov_deg, W, H, contrib_floor)
    print("rendering mode=1 (oracle)...", flush=True)
    orc = render_view0(1, gauss, order, views, fov_deg, W, H, contrib_floor)
    print("dev", dev.shape, "orc", orc.shape, flush=True)

    # Dump actual values at a fixed column for rows 0..31 to tell apart a
    # layout shuffle (dev row == orc some-other-row) from compute saturation
    # (dev row is uniform/clipped).
    col = 500
    print(f"--- pixel values at col {col}, R channel, rows 0..31 ---")
    for r in range(32):
        print(f"row {r:2d}: dev={dev[r,col,0]:.4f} orc={orc[r,col,0]:.4f}")

    err = np.abs(dev - orc).mean(axis=(1, 2))  # per-row MAE
    print("per-row MAE rows 0..40:")
    for r in range(min(40, H)):
        flag = "  <-- BAD" if err[r] > 0.05 else ""
        print(f"row {r:3d}: {err[r]:.4f}{flag}")
    print("mean MAE even rows:", float(err[0::2].mean()))
    print("mean MAE odd rows :", float(err[1::2].mean()))
    for p in (2, 8, 16, 32):
        bands = [float(err[i::p].mean()) for i in range(p)]
        print(f"period {p}: bands min {min(bands):.4f} max {max(bands):.4f} "
              f"-> {[round(b,3) for b in bands]}")


if __name__ == "__main__":
    main()
