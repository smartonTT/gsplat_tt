"""Analyze Gaussian → tile distribution for a fixed camera view; estimate the
theoretical peak kernel performance.

Usage:
  python scripts/binning_log.py <scene> <view> [--tile-sizes 32,16] [--ply PATH] [--camera-file PATH]

Pure CPU. Does not touch the device. Useful for:
  - knowing how much work the alpha-blend kernel actually has to do (per-tile
    Gaussian counts at 32×32 and 16×16, total pairs, overdraw)
  - establishing a theoretical performance floor so the optimization loop has
    a "done enough" stop condition

Outputs (stdout, plus a markdown table if --md is given):

  [tile=32]
    tiles:                  Tx × Ty = N
    non-empty tiles:        E (e×100% of N)
    Gaussian-tile pairs:    P
    per-tile count:         mean=μ  median=m  p95=q95  p99=q99  max=mx
    histogram of per-tile counts: ...
    average overdraw / pixel: P × 1024 / (W × H)

  [tile=16]
    ... same stats, 4× more tiles

  [theoretical peak]
    assuming X cycles per Gaussian-tile pair per core, 72 cores, 1.2 GHz:
       peak_ms_full       = P × X / (72 × 1.2e9) × 1000
       peak_ms_with_etx2  = (P/2) × X / (72 × 1.2e9) × 1000   (50% early-term reduction)
       peak_ms_with_etx5  = (P/5) × X / (72 × 1.2e9) × 1000   (5x reduction)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.rasterization import (  # noqa: E402
    project_gaussians, get_tile_assignments,
)
from gsplat.utils import c2w_to_w2c  # noqa: E402

# Match the rasterizer's fov→focal-length conversion.
from scripts.render_fixed import build_intrinsics  # noqa: E402


# Hardware peak constants (Blackhole P150). Approximate — the actual compute
# grid is queried from `compute_with_storage_grid_size()` at runtime; assume
# 130 worker cores after harvest, at ~1.4 GHz typical clock.
NUM_CORES = 130
SFPU_GHZ = 1.4

# Per Gaussian-tile pair, an extremely tight kernel takes roughly this many
# Tensix cycles. The number is approximate — it covers ~20-30 SFPU ops at
# ~2 cycles each, no spill, no NoC stalls. We log multiple variants so the
# user / supervisor can pick the assumption they trust.
CYCLES_PER_PAIR_AGGRESSIVE = 80
CYCLES_PER_PAIR_REALISTIC = 200
CYCLES_PER_PAIR_CURRENT_IMPL = 1500  # ~17 stages × 4 face passes × spill overhead


def project_for_view(ply_path: Path, c2w: np.ndarray, W: int, H: int,
                     fov_deg: float):
    """Run project + tile_assign once for the given view, return all data."""
    gauss = load_ply(str(ply_path))
    extr = c2w_to_w2c(c2w)
    intr = build_intrinsics(W, H, fov_deg)
    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        gauss.means, gauss.scales, gauss.rotations, extr, intr, H, W,
        opacities=gauss.opacities,
    )
    return gauss, means_2d, covs_2d, depths, radii, valid


def bin_at(means_2d: torch.Tensor, radii: torch.Tensor, H: int, W: int,
           tile_size: int) -> dict:
    """Return summary stats for tile-assignment at the given tile_size."""
    g_ids, t_ids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=tile_size)
    Tx = (W + tile_size - 1) // tile_size
    Ty = (H + tile_size - 1) // tile_size
    N = Tx * Ty
    counts = torch.bincount(t_ids, minlength=N).numpy()
    nonempty = (counts > 0).sum()
    pairs = int(counts.sum())
    nonzero = counts[counts > 0]
    stats = dict(
        tile_size=tile_size,
        Tx=Tx, Ty=Ty, total_tiles=N,
        non_empty=int(nonempty),
        gaussian_tile_pairs=pairs,
        mean=float(counts.mean()),
        median=float(np.median(counts)),
        p95=float(np.percentile(counts, 95)),
        p99=float(np.percentile(counts, 99)),
        max=int(counts.max()),
        mean_nonempty=float(nonzero.mean()) if len(nonzero) else 0.0,
        median_nonempty=float(np.median(nonzero)) if len(nonzero) else 0.0,
        avg_overdraw_per_pixel=pairs * (tile_size * tile_size) / float(W * H),
        # Histogram bins: 0, 1-10, 11-50, 51-100, 101-500, 501+
        hist_bins=[0, 1, 11, 51, 101, 501, 10_000_000],
        hist_counts=None,
    )
    h, _ = np.histogram(counts, bins=stats["hist_bins"])
    stats["hist_counts"] = h.tolist()
    return stats


def theoretical_peak_ms(pairs: int, cycles_per_pair: int) -> float:
    """ms assuming `cycles_per_pair` Tensix cycles per (G, tile) pair, perfectly
    parallelized across NUM_CORES cores at SFPU_GHZ."""
    cycles_total = pairs * cycles_per_pair
    cycles_per_core = cycles_total / NUM_CORES
    seconds = cycles_per_core / (SFPU_GHZ * 1e9)
    return seconds * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("view")
    ap.add_argument("--tile-sizes", default="32,16")
    ap.add_argument("--camera-file", type=Path, default=Path("benchmarks/cameras.json"))
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None,
                    help="Also write a markdown summary at this path")
    ap.add_argument("--resolution", default=None,
                    help="Override (W, H) from cameras.json. Format: WxH "
                         "(e.g. 1024x1024).")
    args = ap.parse_args()

    cams = json.loads(args.camera_file.read_text())
    entry = cams[args.scene]
    if args.resolution is not None:
        w_str, h_str = args.resolution.lower().split("x")
        W, H = int(w_str), int(h_str)
    else:
        W, H = entry["image_size"]
    fov_deg = entry["fov_deg"]
    c2w = np.asarray(entry["views"][args.view]["c2w"], dtype=np.float32)
    ply_path = args.ply if args.ply is not None else Path(entry["ply"])

    print(f"[scene={args.scene} view={args.view} W={W} H={H} fov={fov_deg}°]")
    gauss, means_2d, covs_2d, depths, radii, valid = project_for_view(
        ply_path, c2w, W, H, fov_deg,
    )
    n_total = gauss.num_gaussians
    n_visible = int(valid.sum().item())
    print(f"[gaussians] total={n_total:,}  visible_after_cull={n_visible:,} "
          f"({100.0 * n_visible / n_total:.1f}%)")
    print(f"[radii] min={radii.min().item():.2f}  med={radii.median().item():.2f}  "
          f"p95={np.percentile(radii.numpy(), 95):.2f}  max={radii.max().item():.2f}")

    tile_sizes = [int(x) for x in args.tile_sizes.split(",")]
    bins = {}
    md_rows = ["| tile | tiles | non-empty | pairs | mean | median | p95 | p99 | max | avg overdraw/pix |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for ts in tile_sizes:
        s = bin_at(means_2d, radii, H, W, ts)
        bins[ts] = s
        print(f"\n[tile_size={ts}]")
        print(f"  tiles:                  {s['Tx']} × {s['Ty']} = {s['total_tiles']}")
        print(f"  non-empty tiles:        {s['non_empty']} "
              f"({100.0 * s['non_empty'] / s['total_tiles']:.1f}%)")
        print(f"  G-tile pairs:           {s['gaussian_tile_pairs']:,}")
        print(f"  per-tile count (all):   mean={s['mean']:.1f}  median={s['median']:.0f}  "
              f"p95={s['p95']:.0f}  p99={s['p99']:.0f}  max={s['max']:,}")
        print(f"  per-tile count (≠0):    mean={s['mean_nonempty']:.1f}  "
              f"median={s['median_nonempty']:.0f}")
        print(f"  histogram (0 / 1-10 / 11-50 / 51-100 / 101-500 / 501+):")
        print(f"    {s['hist_counts']}")
        print(f"  avg overdraw/pixel:     {s['avg_overdraw_per_pixel']:.1f}")
        md_rows.append(
            f"| {ts}×{ts} | {s['total_tiles']} | {s['non_empty']} | "
            f"{s['gaussian_tile_pairs']:,} | {s['mean']:.1f} | {s['median']:.0f} | "
            f"{s['p95']:.0f} | {s['p99']:.0f} | {s['max']} | "
            f"{s['avg_overdraw_per_pixel']:.1f} |"
        )

    print(f"\n[theoretical peak] ({NUM_CORES} cores @ {SFPU_GHZ} GHz; per-tile pair cycles)")
    for ts in tile_sizes:
        pairs = bins[ts]["gaussian_tile_pairs"]
        agg = theoretical_peak_ms(pairs, CYCLES_PER_PAIR_AGGRESSIVE)
        rea = theoretical_peak_ms(pairs, CYCLES_PER_PAIR_REALISTIC)
        cur = theoretical_peak_ms(pairs, CYCLES_PER_PAIR_CURRENT_IMPL)
        # Early-termination factor: front-to-back, ~5x reduction is common
        # in real scenes once block-wide T<1e-4 exit lands.
        rea_etx5 = theoretical_peak_ms(pairs // 5, CYCLES_PER_PAIR_REALISTIC)
        print(f"  tile {ts}: pairs={pairs:,}")
        print(f"    aggressive  (~{CYCLES_PER_PAIR_AGGRESSIVE} cyc/pair):  {agg:.2f} ms")
        print(f"    realistic   (~{CYCLES_PER_PAIR_REALISTIC} cyc/pair):  {rea:.2f} ms")
        print(f"    + 5x early-term reduction:               {rea_etx5:.2f} ms")
        print(f"    current impl (~{CYCLES_PER_PAIR_CURRENT_IMPL} cyc/pair): {cur:.2f} ms")

    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        with args.md.open("w") as f:
            f.write(f"# Binning analysis — {args.scene} / {args.view}\n\n")
            f.write(f"- ply: `{ply_path}`\n")
            f.write(f"- image: {W} × {H}, fov {fov_deg}°\n")
            f.write(f"- gaussians total: {n_total:,}, visible after cull: {n_visible:,}\n\n")
            f.write("\n".join(md_rows))
            f.write("\n\n")
            f.write("## Theoretical peak\n\n")
            f.write(f"Assumes {NUM_CORES} cores at {SFPU_GHZ} GHz, perfectly parallel.\n\n")
            for ts in tile_sizes:
                pairs = bins[ts]["gaussian_tile_pairs"]
                f.write(f"### tile {ts}×{ts}\n\n")
                f.write(f"- aggressive (~{CYCLES_PER_PAIR_AGGRESSIVE} cyc/pair): "
                        f"{theoretical_peak_ms(pairs, CYCLES_PER_PAIR_AGGRESSIVE):.2f} ms\n")
                f.write(f"- realistic (~{CYCLES_PER_PAIR_REALISTIC} cyc/pair): "
                        f"{theoretical_peak_ms(pairs, CYCLES_PER_PAIR_REALISTIC):.2f} ms\n")
                f.write(f"- realistic + 5x early-term: "
                        f"{theoretical_peak_ms(pairs // 5, CYCLES_PER_PAIR_REALISTIC):.2f} ms\n")
                f.write(f"- current impl (~{CYCLES_PER_PAIR_CURRENT_IMPL} cyc/pair): "
                        f"{theoretical_peak_ms(pairs, CYCLES_PER_PAIR_CURRENT_IMPL):.2f} ms\n\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
