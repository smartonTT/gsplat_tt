"""Run scripts/render_fixed.py over all 4 scenes for a given view; emit a
comparison table with kernel ms / FPS / visible / entries / PSNR vs reference.

Usage:
  python scripts/bench_panel.py [--view hero] [--scenes stitch,luigi,strawberry,bicycle]
                                [--backend tt] [--warmup 3] [--frames 10]
                                [--ref-dir benchmarks/reference]
                                [--out-dir docs/optimization-log/screenshots]
                                [--label NNN-name]

This is the per-iteration top-level benchmark the supervisor calls (or, more
often, the per-iteration worker calls). It spawns one daemon per scene (each
`render_fixed.py` invocation owns its own daemon subprocess), which is the
correct lifecycle — daemons can't be reused across .ply files.

For each scene:
  1. python scripts/render_fixed.py <scene> <view> --json --out <out>.png
  2. If <ref-dir>/<scene>_<view>.png exists, run image_diff.py vs it
     (anchor=baseline; non-blocking, only logged).
  3. Collect numbers, print one row of the table.

Default --out-dir routes screenshots under docs/optimization-log/screenshots/
labeled by --label (default: timestamp).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Repo root for resolving paths.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SCENES = ["stitch", "luigi", "strawberry", "bicycle"]
DEFAULT_VIEW = "hero"


def parse_render_json(stdout: str) -> dict | None:
    """render_fixed.py prints one `JSON:{...}` line when called with --json."""
    for line in stdout.splitlines():
        if line.startswith("JSON:"):
            try:
                return json.loads(line[5:])
            except json.JSONDecodeError:
                return None
    return None


def parse_diff_json(stdout: str) -> dict | None:
    """image_diff.py with --json prints one JSON line."""
    last_line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", default=DEFAULT_VIEW)
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES),
                    help="Comma-separated scene names.")
    ap.add_argument("--backend", default="tt")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--ref-dir", type=Path, default=Path("benchmarks/reference"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--label", default=None,
                    help="Used as the subdir name under --out-dir. Default: timestamp.")
    ap.add_argument("--md", type=Path, default=None,
                    help="Also write a markdown table to this file.")
    ap.add_argument("--no-diff", action="store_true",
                    help="Skip image_diff vs reference PNGs.")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    label = args.label or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir if args.out_dir is not None
               else Path("docs/optimization-log/screenshots")) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Need TT_METAL_* env if backend=tt and not already set. We tolerate both
    # the "set externally before invocation" pattern and a sane fallback.
    env = os.environ.copy()
    if args.backend == "tt":
        env.setdefault("TT_METAL_HOME", os.path.abspath("backends/tt/tt-metal"))
        env.setdefault("TT_METAL_RUNTIME_ROOT", os.path.abspath("backends/tt/tt-metal"))

    rows: list[dict] = []
    print(f"=== bench_panel  label={label}  view={args.view}  backend={args.backend} ===")
    for scene in scenes:
        out_png = out_dir / f"{scene}_{args.view}.png"
        cmd = [
            sys.executable, "scripts/render_fixed.py", scene, args.view,
            "--backend", args.backend,
            "--warmup", str(args.warmup), "--frames", str(args.frames),
            "--out", str(out_png),
            "--json",
        ]
        print(f"\n--- {scene} ---")
        print(f"$ {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"!! render_fixed exited {proc.returncode} for {scene}; skipping")
            rows.append({"scene": scene, "error": f"exit {proc.returncode}"})
            continue
        render = parse_render_json(proc.stdout)
        if render is None:
            print(f"!! could not parse JSON output for {scene}")
            rows.append({"scene": scene, "error": "no JSON"})
            continue

        # Optional diff vs reference (baseline anchor — drift gate).
        ref_png = args.ref_dir / f"{scene}_{args.view}.png"
        diff = None
        if not args.no_diff and ref_png.exists():
            diff_png = out_dir / f"{scene}_{args.view}_diff.png"
            diff_cmd = [
                sys.executable, "scripts/image_diff.py",
                str(ref_png), str(out_png),
                "--anchor", "baseline",
                "--amplified-diff", str(diff_png),
                "--json",
            ]
            print(f"$ {' '.join(diff_cmd)}")
            proc2 = subprocess.run(diff_cmd, capture_output=True, text=True)
            sys.stdout.write(proc2.stdout)
            sys.stderr.write(proc2.stderr)
            diff = parse_diff_json(proc2.stdout)

        row = {
            "scene": scene,
            "view": args.view,
            "image_size": render["image_size"],
            "num_gaussians": render["num_gaussians"],
            "num_visible": render["num_visible"],
            "num_entries": render["num_entries"],
            "kernel_ms": render.get("kernel_ms"),
            "blend_ms": render.get("blend_ms"),
            "total_ms": render.get("total_ms"),
            "fps_kernel": (1000.0 / render["kernel_ms"]) if render.get("kernel_ms") else None,
            "fps_end_to_end": (1000.0 / render["total_ms"]) if render.get("total_ms") else None,
            "psnr_db": (diff or {}).get("psnr_db"),
            "ssim": (diff or {}).get("ssim"),
            "gate": (diff or {}).get("gate"),
            "out_png": str(out_png),
        }
        rows.append(row)

    # Final table.
    print("\n\n=== summary ===")
    header = (f"| {'scene':<11} | {'res':>9} | {'visible':>9} | {'entries':>9} | "
              f"{'kernel ms':>9} | {'fps':>6} | {'PSNR':>5} | {'SSIM':>5} | {'gate':<12} |")
    print(header)
    print("|" + "-|".join(["-" * (len(c) - 1) for c in header.strip("|").split("|")]) + "|")
    for r in rows:
        if "error" in r:
            print(f"| {r['scene']:<11} | {'ERR':>9} | {'-':>9} | {'-':>9} | "
                  f"{r['error']:>9} | {'-':>6} | {'-':>5} | {'-':>5} | {'-':<12} |")
            continue
        W, H = r["image_size"]
        k_ms = r["kernel_ms"]
        k_str = f"{k_ms:6.1f}" if k_ms else "  --"
        fps = r["fps_kernel"]
        fps_str = f"{fps:6.2f}" if fps else "  --"
        psnr = r["psnr_db"]
        psnr_str = f"{psnr:5.1f}" if psnr else "  --"
        ssim = r["ssim"]
        ssim_str = f"{ssim:5.3f}" if ssim is not None else "  --"
        gate = r["gate"] or "-"
        print(f"| {r['scene']:<11} | {W}x{H:<5} | {r['num_visible']:>9,} | "
              f"{r['num_entries']:>9,} | {k_str:>9} | {fps_str:>6} | "
              f"{psnr_str:>5} | {ssim_str:>5} | {gate:<12} |")

    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        with args.md.open("w") as f:
            f.write(f"# Bench panel — {label}  (view={args.view})\n\n")
            f.write("| scene | res | visible | entries | kernel ms | fps | PSNR | SSIM | gate | png |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                if "error" in r:
                    f.write(f"| {r['scene']} | ERR | - | - | {r['error']} | - | - | - | - | - |\n")
                    continue
                W, H = r["image_size"]
                k_ms = r["kernel_ms"]
                f.write(f"| {r['scene']} | {W}x{H} | {r['num_visible']:,} | "
                        f"{r['num_entries']:,} | "
                        f"{k_ms:.1f} | "
                        f"{r['fps_kernel']:.2f} | "
                        f"{r['psnr_db'] if r['psnr_db'] is not None else '-'} | "
                        f"{r['ssim'] if r['ssim'] is not None else '-'} | "
                        f"{r['gate'] or '-'} | "
                        f"`{r['out_png']}` |\n")
        print(f"\nwrote {args.md}")

    # Exit code: 0 unless any hard-reject.
    any_hard = any(r.get("gate") == "hard-reject" for r in rows)
    sys.exit(1 if any_hard else 0)


if __name__ == "__main__":
    main()
