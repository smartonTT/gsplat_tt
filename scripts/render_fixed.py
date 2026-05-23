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

    backend = get_backend(args.backend, verbose=False)
    pipeline = Pipeline(backend, tile_size=TILE_SIZE)
    extrinsics = c2w_to_w2c(c2w)
    intrinsics = build_intrinsics(W, H, fov_deg)

    # Pipelined fast path (iter 029): when the backend exposes the split
    # submit/recv API, run frame N+1's CPU pre-blend (project + tile_assign
    # + sort + prep) while frame N's daemon kernel is in flight. The bench
    # measures per-frame STEADY-STATE INTERVAL between successive recv()s,
    # which is what the user actually perceives as latency once the
    # pipeline is full. Synchronous fallback below is kept for backends
    # that don't support pipelining (CPU/CUDA today).
    pipelined = args.backend == "tt" and hasattr(backend, "submit_frame")
    n_total = args.warmup + args.frames
    samples: list[dict] = []
    last_image = None
    visible = 0
    entries = 0

    if pipelined:
        # We hand-roll the pipeline directly here instead of going through
        # Pipeline.render() so the steady-state semantics (overlap of
        # pre-blend with daemon) are obvious and easy to reason about.
        # The Pipeline object is still used for project/tile_assign/sort
        # via its backend, so we don't duplicate the rasterization stage
        # logic.
        from gsplat.pipeline import Pipeline as _P  # noqa: F401  (timer)
        from contextlib import contextmanager

        @contextmanager
        def _t(d, k):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                d[k] = (time.perf_counter() - t0) * 1000.0

        # Bench-specific shortcut: the camera is fixed across all frames,
        # so the project culling mask is invariant. Slice colors/opacities
        # ONCE outside the loop and pass the SAME tensor to every
        # submit_frame call — set_scene's id-keyed cache then hits and we
        # skip a SCN1 round-trip per iteration, which would otherwise
        # serialize the pipeline (SCN1 blocks the daemon, breaking host
        # pre-blend / device kernel overlap).
        # The interactive viewer keeps going through Pipeline.render() ->
        # backend.blend() (synchronous), so it doesn't need this shortcut.
        with torch.no_grad():
            _, _, _, _, _initial_mask = backend.project(
                gauss.means, gauss.scales, gauss.rotations,
                extrinsics, intrinsics, H, W,
                opacities=gauss.opacities,
            )
        bench_colors = gauss.colors[_initial_mask]
        bench_opacities = gauss.opacities[_initial_mask]

        def _pre_blend(timings: dict[str, float]):
            """Run project + tile_assign + sort; return blend args + counts.
            Same logic as Pipeline.render() pre-blend stages, but without
            invoking blend() (we'll call submit_frame instead). Crucially,
            does NOT call set_scene — that happens lazily inside
            submit_frame, which only runs when no frame is in flight.
            Reuses bench_colors/bench_opacities (camera-fixed bench)."""
            with _t(timings, "project"):
                means_2d, covs_2d, depths, radii, valid_mask = backend.project(
                    gauss.means, gauss.scales, gauss.rotations,
                    extrinsics, intrinsics, H, W,
                    opacities=gauss.opacities,
                )
            v = int(valid_mask.sum().item())
            if v == 0:
                return None, 0, 0
            with _t(timings, "tile_assign"):
                gaussian_ids, tile_ids, _ = backend.tile_assign(
                    means_2d, radii, H, W, tile_size=TILE_SIZE,
                )
            tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
            tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
            # Iter 034: per-pair Mahalanobis cull (PSNR-safe).
            with _t(timings, "cull_pairs"):
                gaussian_ids, tile_ids = backend.cull_pairs(
                    gaussian_ids, tile_ids, means_2d, covs_2d, bench_opacities,
                    tiles_x, tile_size=TILE_SIZE,
                )
            with _t(timings, "sort"):
                sorted_gaussian_ids, tile_ranges = backend.sort(
                    gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
                )
            blend_args = (
                means_2d, covs_2d, bench_colors, bench_opacities,
                sorted_gaussian_ids, tile_ranges, H, W,
            )
            return blend_args, v, int(sorted_gaussian_ids.numel())

        # Prime the pipeline with frame 0.
        prime_t = {}
        prime_args, visible, entries = _pre_blend(prime_t)
        if prime_args is None:
            print("ERROR: empty frame on prime; aborting", file=sys.stderr)
            sys.exit(2)
        prime_partial = backend.submit_frame(*prime_args)
        # Stash this frame's pre-blend timings to merge with its recv.
        in_flight_timings = prime_t
        in_flight_partial = prime_partial

        # Steady-state loop: each iteration runs frame i+1's pre-blend
        # WHILE frame i is in the daemon, then recv()s frame i, then
        # submits frame i+1. The wall-time of each iteration is the
        # max(pre-blend, daemon) + recv overhead — i.e. exactly the
        # steady-state per-frame interval the user wants.
        for i in range(n_total):
            iter_t0 = time.perf_counter()
            timings_i = {}
            sub_i = {}

            is_last = (i == n_total - 1)
            if not is_last:
                next_args, next_v, next_e = _pre_blend(timings_i)
                if next_args is None:
                    print("ERROR: empty frame mid-bench; aborting",
                          file=sys.stderr)
                    sys.exit(2)
            else:
                next_args = None

            # Recv frame i (the prime / previous in_flight).
            image, sub_recv = backend.recv_frame(in_flight_partial)
            if image is not None:
                last_image = image  # ref to backend._image_buf, copied next iter

            # Stitch the frame i timings: pre-blend was measured BEFORE
            # the prime (or before the previous iter), prep + save_npy
            # come from the partial returned by submit_frame, and
            # daemon_rt + load_npy + device_kernel come from sub_recv.
            timings_for_frame_i = in_flight_timings
            sub_for_frame_i = {
                **{k: v for k, v in in_flight_partial.items()},
                **sub_recv,
            }

            # Submit frame i+1, if any.
            if not is_last:
                in_flight_partial = backend.submit_frame(*next_args)
                in_flight_timings = timings_i
                visible = next_v
                entries = next_e

            # blend == time spent waiting for recv + (time we did submit, ~5ms).
            # In steady state this is dominated by the daemon + IPC.
            iter_wall = (time.perf_counter() - iter_t0) * 1000.0
            timings_for_frame_i["blend"] = iter_wall - sum(
                timings_for_frame_i.get(k, 0.0)
                for k in ("project", "tile_assign", "sort")
            )
            timings_for_frame_i["total"] = iter_wall

            is_timed = i >= args.warmup
            tag = "timed " if is_timed else "warmup"
            sub_summary = " ".join(
                f"{k}={v:.1f}" for k, v in sorted(sub_for_frame_i.items())
            )
            print(
                f"[{tag} {i:2d}] wall={iter_wall:7.1f}ms "
                f"total={timings_for_frame_i.get('total', 0):.1f} "
                f"blend={timings_for_frame_i.get('blend', 0):.1f} "
                f"visible={visible} entries={entries}  {sub_summary}",
                flush=True,
            )
            if is_timed:
                samples.append({
                    "wall_ms": iter_wall,
                    **{f"timings.{k}": v for k, v in timings_for_frame_i.items()},
                    **{f"sub.{k}": v for k, v in sub_for_frame_i.items()},
                })
    else:
        # Sync path (CPU / CUDA / TT without submit_frame).
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
            print(
                f"[{tag} {i:2d}] wall={wall_ms:7.1f}ms "
                f"total={result.timings.get('total', 0):.1f} "
                f"blend={result.timings.get('blend', 0):.1f} "
                f"visible={visible} entries={entries}  {sub_summary}",
                flush=True,
            )
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

    img_uint8 = np.flipud((np.clip(last_image, 0.0, 1.0) * 255).astype(np.uint8))
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
