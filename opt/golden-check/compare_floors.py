#!/usr/bin/env python3
"""Compare two per-view render dumps (old | new contrib_floor) for tile artifacts.

For every view present in both dirs (render/run.py --dump-views output, 8-bit PNG):
  - sbs/<view>.png      full-res side-by-side  old | new           (--sbs views only)
  - heat/<view>.png     |new-old| max over RGB, x AMP, inferno colormap, full res
  - tilemax/<view>.png  per-32x32-tile max |diff| (8-bit levels), one cell per tile
  - crops/<view>_*.png  4x nearest zoom of the worst tile: old | new | heat
and stats.json with, per view:
  psnr8, max/mean |diff|, per-tile max stats, and seam ratios: the mean |step| of the
  diff across x/y positions on 32/16/8/4-px grid lines divided by the mean |step| at
  positions on no grid line (x % 4 != 0). A seam-free change gives ~1.0; a tile or
  microblock seam shows up as a ratio well above 1 on that grid period. The same
  ratios are reported for the old and the new image themselves.

Usage: compare_floors.py OLD_DIR NEW_DIR OUT_DIR [--sbs hero,orb_05] [--amp 16]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402

TILE = 32
PERIODS = (32, 16, 8, 4)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)


def psnr8(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def seam_ratios(img):
    """Grid-line step / off-grid step, per axis and period. img: HxWxC signed."""
    f = img.astype(np.float64)
    out = {}
    for axis, name in ((1, "x"), (0, "y")):
        step = np.abs(np.diff(f, axis=axis)).mean(axis=(1 - axis, 2))  # len N-1
        pos = np.arange(1, f.shape[axis])  # step[i] is across the line at pos[i]
        base = step[pos % 4 != 0].mean()
        for p in PERIODS:
            sel = (pos % p == 0)
            for q in PERIODS:
                if q > p:
                    sel &= (pos % q != 0)
            out[f"{name}{p}"] = float(step[sel].mean() / base) if base > 0 else float("nan")
    return out


def colorize(a, vmax):
    rgba = cm.inferno(np.clip(a / vmax, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("out")
    ap.add_argument("--sbs", default="hero")
    ap.add_argument("--amp", type=float, default=16.0)
    args = ap.parse_args()
    old_d, new_d, out = Path(args.old), Path(args.new), Path(args.out)
    for sub in ("sbs", "heat", "tilemax", "crops"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sbs_views = set(args.sbs.split(",")) if args.sbs else set()

    stats = {}
    for po in sorted(old_d.glob("view*.png")):
        pn = new_d / po.name
        if not pn.exists():
            continue
        view = po.stem.split("_", 1)[1]
        a, b = load(po), load(pn)
        d = b - a
        ad = np.abs(d).max(axis=2).astype(np.float64)  # HxW, 8-bit levels
        H, W = ad.shape
        tm = ad.reshape(H // TILE, TILE, W // TILE, TILE).max(axis=(1, 3))
        # localized seam/tile outliers: tile max vs median of its 8 neighbours
        pad = np.pad(tm, 1, mode="edge")
        neigh = np.stack([pad[1 + dy:1 + dy + tm.shape[0], 1 + dx:1 + dx + tm.shape[1]]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        nmed = np.median(neigh, axis=0)
        outlier = (tm >= 4) & (tm > 3 * np.maximum(nmed, 1))
        ty, tx = np.unravel_index(int(np.argmax(tm)), tm.shape)
        s = {
            "psnr8": psnr8(a, b),
            "max_abs": int(ad.max()),
            "mean_abs": float(np.abs(d).mean()),
            "px_ge1": float((ad >= 1).mean()),
            "px_ge4": float((ad >= 4).mean()),
            "px_ge8": float((ad >= 8).mean()),
            "tile_max_p50": float(np.median(tm)),
            "tile_max_p99": float(np.percentile(tm, 99)),
            "worst_tile": [int(ty), int(tx), int(tm[ty, tx])],
            "outlier_tiles": int(outlier.sum()),
            "seam_diff": seam_ratios(d),
            "seam_old": seam_ratios(a),
            "seam_new": seam_ratios(b),
        }
        stats[view] = s

        Image.fromarray(colorize(ad * args.amp, 255.0)).save(out / "heat" / f"{view}.png")
        cell = 16
        tmimg = colorize(np.kron(tm, np.ones((cell, cell))), max(8.0, float(tm.max())))
        Image.fromarray(tmimg).save(out / "tilemax" / f"{view}.png")
        if view in sbs_views:
            Image.fromarray(np.concatenate([a, b], axis=1).astype(np.uint8)).save(
                out / "sbs" / f"{view}.png")
        # 4x zoom crop of a 3x3-tile window around the worst tile: old | new | heat
        y0 = int(np.clip(ty * TILE - TILE, 0, H - 3 * TILE))
        x0 = int(np.clip(tx * TILE - TILE, 0, W - 3 * TILE))
        win = (slice(y0, y0 + 3 * TILE), slice(x0, x0 + 3 * TILE))
        heat = colorize(ad[win] * args.amp, 255.0)
        panel = np.concatenate([a[win].astype(np.uint8), b[win].astype(np.uint8), heat], axis=1)
        panel = np.kron(panel, np.ones((4, 4, 1), dtype=np.uint8))
        Image.fromarray(panel).save(out / "crops" / f"{view}_worst.png")

    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    hdr = f"{'view':8s} {'psnr8':>6s} {'max':>4s} {'mean':>6s} {'>=4':>7s} {'tmax99':>6s} {'outl':>4s}  " \
          + " ".join(f"d{k:>4s}" for k in ("x32", "y32", "x16", "y16", "x8", "y8", "x4", "y4")) \
          + "  new/old x32 y32"
    print(hdr)
    for v, s in stats.items():
        sd, sn, so = s["seam_diff"], s["seam_new"], s["seam_old"]
        print(f"{v:8s} {s['psnr8']:6.2f} {s['max_abs']:4d} {s['mean_abs']:6.3f} {s['px_ge4']:7.4%} "
              f"{s['tile_max_p99']:6.1f} {s['outlier_tiles']:4d}  "
              + " ".join(f"{sd[k]:5.2f}" for k in ("x32", "y32", "x16", "y16", "x8", "y8", "x4", "y4"))
              + f"  {sn['x32'] / so['x32']:5.3f} {sn['y32'] / so['y32']:5.3f}")


if __name__ == "__main__":
    main()
