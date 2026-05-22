"""Render one fixed-camera view of a scene; save PNG and print median kernel-only ms.

Usage:
  python scripts/render_fixed.py <scene> <view> [options]
  python scripts/render_fixed.py stitch hero --out /tmp/it.png
  python scripts/render_fixed.py stitch hero --backend cpu

Loads camera from `benchmarks/cameras.json` by (scene, view) key.

The TT backend requires the kernel binary at
`backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting`,
TT_METAL_HOME / TT_METAL_RUNTIME_ROOT exported, and a Tenstorrent device
available on the host. Run on `yyzc-wh-03`, not locally.

Options:
  --backend tt|cpu      (default tt)
  --warmup N            (default 3 frames; first frame is always discarded)
  --frames M            (default 10 timed frames)
  --out PATH            (default: benchmarks/cameras_preview/<scene>_<view>.png)
  --camera-file PATH    (default: benchmarks/cameras.json)
  --ply PATH            (default: from cameras.json scene entry)
  --json                (also emit one line of JSON with all timings)
  --profile             (run with TT_METAL_DEVICE_PROFILER=1; one frame only)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make `from gsplat...` and `from backends...` imports work from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


TILE_SIZE = 32


def fov_to_focal(fov_deg: float, length_px: int) -> float:
    """Pinhole focal length given a horizontal/vertical fov and that dim's pixel size."""
    return 0.5 * length_px / np.tan(0.5 * np.deg2rad(fov_deg))


def build_intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    """Square-pixel K. fov_deg applies to the longer image dim (so a fov of 50°
    on a 768x512 image is a 50° horizontal FoV; on a 512x768, vertical)."""
    longer = max(W, H)
    f = fov_to_focal(fov_deg, longer)
    K = np.array([
        [f, 0.0, W * 0.5],
        [0.0, f, H * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return torch.from_numpy(K)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Scene short name (key into cameras.json)")
    parser.add_argument("view", help="View name: hero | side | top")
    parser.add_argument("--backend", default="tt", choices=("cpu", "tt", "cuda"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--camera-file", type=Path, default=Path("benchmarks/cameras.json"))
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Also set TT_METAL_DEVICE_PROFILER=1; runs 1 frame")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help=(
            "Override the (W, H) from cameras.json. Useful for benchmarking "
            "the same view at a different resolution; the FOV is preserved."
        ),
    )
    args = parser.parse_args()

    if args.profile:
        # Profiler runs are one-shot.
        args.warmup = 1
        args.frames = 1
        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

    cameras = json.loads(args.camera_file.read_text())
    if args.scene not in cameras:
        print(f"ERROR: scene {args.scene!r} not in {args.camera_file}. "
              f"Known: {list(cameras)}", file=sys.stderr)
        sys.exit(2)
    entry = cameras[args.scene]
    if args.view not in entry["views"]:
        print(f"ERROR: view {args.view!r} not in scene {args.scene}. "
              f"Known: {list(entry['views'])}", file=sys.stderr)
        sys.exit(2)

    ply_path = args.ply if args.ply is not None else Path(entry["ply"])
    if not ply_path.exists():
        print(f"ERROR: ply not found: {ply_path}", file=sys.stderr)
        sys.exit(2)

    W, H = entry["image_size"]
    if args.image_size is not None:
        W, H = int(args.image_size[0]), int(args.image_size[1])
    fov_deg = entry["fov_deg"]
    # c2w_to_w2c expects numpy; build c2w as numpy and pass through.
    c2w = np.asarray(entry["views"][args.view]["c2w"], dtype=np.float32)

    out_path = args.out if args.out is not None else (
        Path("benchmarks/cameras_preview") / f"{args.scene}_{args.view}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[scene={args.scene} view={args.view} ply={ply_path} "
          f"backend={args.backend} W={W} H={H} fov={fov_deg}°]", flush=True)
    gauss = load_ply(str(ply_path))
    print(f"[gaussians] {gauss.num_gaussians:,} loaded", flush=True)

    pipeline = Pipeline(get_backend(args.backend, verbose=False), tile_size=TILE_SIZE)
    extrinsics = c2w_to_w2c(c2w)
    intrinsics = build_intrinsics(W, H, fov_deg)

    # Warmup + timed loop. Each call goes through the full pipeline (project,
    # tile_assign, sort, blend). For TT, kernel-only ms lives under
    # sub_timings["blend.daemon_rt.device_kernel"].
    n_total = args.warmup + args.frames
    samples: list[dict] = []
    last_image = None
    visible = 0
    entries = 0
    for i in range(n_total):
        t0 = time.perf_counter()
        result = pipeline.render(gauss, extrinsics, intrinsics, H, W)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if result.image is not None:
            last_image = result.image
        visible = result.num_visible
        entries = result.num_entries
        is_timed = i >= args.warmup
        tag = "timed " if is_timed else "warmup"
        sub_summary = " ".join(
            f"{k}={v:.1f}" for k, v in sorted(result.sub_timings.items())
        )
        print(f"[{tag} {i:2d}] wall={wall_ms:7.1f}ms total={result.timings.get('total', 0):.1f} "
              f"blend={result.timings.get('blend', 0):.1f} "
              f"visible={visible} entries={entries}  {sub_summary}", flush=True)
        if is_timed:
            samples.append({
                "wall_ms": wall_ms,
                **{f"timings.{k}": v for k, v in result.timings.items()},
                **{f"sub.{k}": v for k, v in result.sub_timings.items()},
            })

    pipeline.close()

    if last_image is None:
        print("ERROR: empty frame (no visible Gaussians); aborting", file=sys.stderr)
        sys.exit(2)

    img_uint8 = (np.clip(last_image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(out_path)
    print(f"[saved] {out_path}  ({img_uint8.shape[1]}x{img_uint8.shape[0]})", flush=True)

    if not samples:
        sys.exit(0)

    # Aggregate medians across timed samples.
    keys = sorted({k for s in samples for k in s})
    medians = {}
    for k in keys:
        vs = [s[k] for s in samples if k in s]
        if vs:
            medians[k] = statistics.median(vs)
    kernel_ms = medians.get("sub.blend.daemon_rt.device_kernel")  # TT
    blend_ms = medians.get("timings.blend", 0.0)
    total_ms = medians.get("timings.total", 0.0)

    print("=== summary (median over {} timed frames) ===".format(len(samples)))
    for k in keys:
        if k.startswith("timings.") or k.startswith("sub.") or k == "wall_ms":
            print(f"  {k:<45s} {medians[k]:8.2f} ms")
    if kernel_ms is not None:
        print(f"[fps] kernel-only={1000.0/kernel_ms:6.2f}  "
              f"end-to-end={1000.0/total_ms if total_ms>0 else 0:6.2f}")
    else:
        print(f"[fps] end-to-end={1000.0/total_ms if total_ms>0 else 0:6.2f}  "
              f"(no kernel-only sub-timing — backend doesn't report it)")

    if args.json:
        out_json = {
            "scene": args.scene,
            "view": args.view,
            "backend": args.backend,
            "image_size": [W, H],
            "num_gaussians": gauss.num_gaussians,
            "num_visible": visible,
            "num_entries": entries,
            "frames_timed": len(samples),
            "medians_ms": medians,
            "kernel_ms": kernel_ms,
            "blend_ms": blend_ms,
            "total_ms": total_ms,
            "output_png": str(out_path),
        }
        print("JSON:" + json.dumps(out_json))


if __name__ == "__main__":
    main()
