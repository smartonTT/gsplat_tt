#!/usr/bin/env python3
"""Measure a CUDA 3DGS rasterizer on the *same* bench our TT pipeline uses.

Produces the GPU reference row for `opt/cpu-vs-tt-comparison.md`: avg/p50
ms/view over the 30 `benchmarks/cameras_v2.json` views at 1024x1024, plus hero
PSNR against `benchmarks/reference_v2/hero.png`.

Bench identity (must match `gsplat.viewer` / `render/run.py` exactly):
  * scene      scenes/bicycle.ply, all activations applied host-side by this
               script the same way `gsplat/loading_gaussians.py` does
               (exp(scale), normalize(quat), sigmoid(opacity),
               0.5 + C0*f_dc clamped to [0,1]).
  * colors     passed as *precomputed RGB* (sh_degree=None), so the GPU renders
               the same SH-degree-0 colors our pipeline uses. Do not let gsplat
               evaluate SH -- the ply's higher bands are not in our reference.
  * intrinsics `_intrinsics_from_fov(W, H, fov_deg)`:
                 f = 0.5 * max(W,H) / tan(0.5 * fov), cx = W/2, cy = H/2.
  * extrinsics viewmat = inverse(c2w) (OpenCV convention, +Z forward, +Y down),
               which is gsplat's `viewmats` convention unchanged.
  * order      the file's `order` list, hero first. The hero view is the warmup
               and is EXCLUDED from the timing stats -- same rule as the CPU
               rows in opt/cpu-vs-tt-comparison.md (stats over 29 views).
  * timing     CUDA-synchronised wall time around the rasterization call only
               (no ply load, no host->device upload, no PNG encode), matching
               "wall time per pipeline.render" on the CPU/TT rows.
  * background black, no alpha compositing, output clamped to [0,1].

Usage on a GPU host:

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install gsplat plyfile numpy pillow
    python bench/gpu_reference/run_gpu_bench.py \
        --ply scenes/bicycle.ply \
        --cameras benchmarks/cameras_v2.json \
        --reference benchmarks/reference_v2/hero.png \
        --out opt/cpu-vs-tt/gpu_result.json \
        --hero-png opt/cpu-vs-tt/gpu_hero.png

Backends: `--backend gsplat` (nerfstudio gsplat, default) or
`--backend inria` (graphdeco diff-gaussian-rasterization).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

C0 = 0.28209479177387814


# ---------------------------------------------------------------- scene load


def load_ply(path: Path, device: torch.device):
    """Same activations as gsplat/loading_gaussians.py::load_ply."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path)).elements[0]

    def col(*names):
        return np.stack([np.asarray(vertex[n], dtype=np.float32) for n in names], -1)

    means = torch.from_numpy(col("x", "y", "z"))
    scales = torch.exp(torch.from_numpy(col("scale_0", "scale_1", "scale_2")))
    quats = torch.from_numpy(col("rot_0", "rot_1", "rot_2", "rot_3"))
    quats = quats / quats.norm(dim=-1, keepdim=True)
    sh_dc = torch.from_numpy(col("f_dc_0", "f_dc_1", "f_dc_2"))
    colors = torch.clamp(0.5 + C0 * sh_dc, 0.0, 1.0)
    opacities = torch.sigmoid(
        torch.from_numpy(np.asarray(vertex["opacity"], dtype=np.float32))
    )

    return (
        means.to(device),
        quats.to(device),
        scales.to(device),
        opacities.to(device),
        colors.to(device),
    )


def intrinsics_from_fov(W: int, H: int, fov_deg: float) -> np.ndarray:
    """gsplat/viewer.py::_intrinsics_from_fov -- fov applies to the longer side."""
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov_deg))
    return np.array(
        [[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def c2w_to_viewmat(c2w: np.ndarray) -> np.ndarray:
    """gsplat/utils.py::c2w_to_w2c -- closed-form rigid inverse."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


# ------------------------------------------------------------------ backends


class GsplatBackend:
    """nerfstudio gsplat `rasterization` (CUDA)."""

    name = "gsplat"

    def __init__(self, scene, W, H, K, near, far):
        import gsplat as _g

        self.g = _g
        self.version = getattr(_g, "__version__", "unknown")
        self.means, self.quats, self.scales, self.opacities, self.colors = scene
        self.W, self.H = W, H
        self.K = torch.from_numpy(K)[None].cuda()
        self.near, self.far = near, far

    def render(self, viewmat: np.ndarray) -> torch.Tensor:
        out, _, _ = self.g.rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,           # precomputed RGB, SH already applied
            viewmats=torch.from_numpy(viewmat)[None].cuda(),
            Ks=self.K,
            width=self.W,
            height=self.H,
            sh_degree=None,
            near_plane=self.near,
            far_plane=self.far,
            render_mode="RGB",
            backgrounds=torch.zeros(1, 3, device="cuda"),
            packed=True,
        )
        return out[0]                     # (H, W, 3) in [0, 1]


class InriaBackend:
    """graphdeco-inria diff-gaussian-rasterization (the original CUDA rasterizer)."""

    name = "inria-diff-gaussian-rasterization"

    def __init__(self, scene, W, H, K, near, far):
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        self.Settings = GaussianRasterizationSettings
        self.Rasterizer = GaussianRasterizer
        self.version = "diff-gaussian-rasterization"
        self.means, self.quats, self.scales, self.opacities, self.colors = scene
        self.W, self.H = W, H
        self.K, self.near, self.far = K, near, far
        self.tanfovx = 0.5 * W / K[0, 0]
        self.tanfovy = 0.5 * H / K[1, 1]

    def render(self, viewmat: np.ndarray) -> torch.Tensor:
        # The INRIA kernel wants column-major (transposed) matrices and a full
        # projection matrix; build an OpenGL-style perspective from K.
        n, f = self.near, self.far
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / self.tanfovx
        proj[1, 1] = 1.0 / self.tanfovy
        proj[2, 2] = f / (f - n)
        proj[3, 2] = 1.0
        proj[2, 3] = -(f * n) / (f - n)
        full = proj @ viewmat

        view_t = torch.from_numpy(viewmat.T).cuda()
        full_t = torch.from_numpy(full.T).cuda()
        campos = torch.from_numpy(np.linalg.inv(viewmat)[:3, 3].copy()).cuda()

        settings = self.Settings(
            image_height=self.H,
            image_width=self.W,
            tanfovx=float(self.tanfovx),
            tanfovy=float(self.tanfovy),
            bg=torch.zeros(3, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=view_t,
            projmatrix=full_t,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self.Rasterizer(raster_settings=settings)
        img, _ = rasterizer(
            means3D=self.means,
            means2D=torch.zeros_like(self.means),
            shs=None,
            colors_precomp=self.colors,
            opacities=self.opacities[:, None],
            scales=self.scales,
            rotations=self.quats,
            cov3D_precomp=None,
        )
        return img.permute(1, 2, 0)       # (3, H, W) -> (H, W, 3)


BACKENDS = {"gsplat": GsplatBackend, "inria": InriaBackend}


# ------------------------------------------------------------------- helpers


def psnr_vs_reference(img: np.ndarray, ref_path: Path) -> float:
    """-10*log10(MSE) on [0,1], same formula as the CPU rows."""
    from PIL import Image

    ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0
    if ref.shape != img.shape:
        raise SystemExit(f"reference {ref.shape} != render {img.shape}")
    mse = float(np.mean((img - ref) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def gpu_identity() -> dict:
    out = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out.update(
            name=torch.cuda.get_device_name(0),
            sm=f"{p.major}.{p.minor}",
            vram_gb=round(p.total_memory / 2**30, 1),
            mp_count=p.multi_processor_count,
        )
    try:
        out["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        pass
    out["host"] = platform.node()
    return out


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, default=Path("scenes/bicycle.ply"))
    ap.add_argument("--cameras", type=Path,
                    default=Path("benchmarks/cameras_v2.json"))
    ap.add_argument("--scene-key", default="bicycle")
    ap.add_argument("--reference", type=Path,
                    default=Path("benchmarks/reference_v2/hero.png"))
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="gsplat")
    ap.add_argument("--out", type=Path,
                    default=Path("opt/cpu-vs-tt/gpu_result.json"))
    ap.add_argument("--hero-png", type=Path,
                    default=Path("opt/cpu-vs-tt/gpu_hero.png"))
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=1e10)
    ap.add_argument("--repeats", type=int, default=1,
                    help="render the 30-view sequence N times; stats use the "
                         "last pass so caches/JIT are warm")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device visible to torch. This bench is a GPU "
              "reference; do not report numbers from a CPU fallback.",
              file=sys.stderr)
        return 2

    ident = gpu_identity()
    print(f"GPU: {ident.get('name')}  ({ident.get('vram_gb')} GB, "
          f"sm{ident.get('sm')})  torch {ident['torch']} / cuda {ident['cuda']}")

    cams = json.loads(args.cameras.read_text())[args.scene_key]
    W, H = cams["image_size"]
    fov_deg = cams["fov_deg"]
    order = cams["order"]
    K = intrinsics_from_fov(W, H, fov_deg)
    viewmats = [c2w_to_viewmat(np.asarray(cams["views"][n]["c2w"], dtype=np.float32))
                for n in order]

    device = torch.device("cuda")
    t0 = time.perf_counter()
    scene = load_ply(args.ply, device)
    print(f"loaded {scene[0].shape[0]:,} gaussians in "
          f"{time.perf_counter() - t0:.1f}s")

    backend = BACKENDS[args.backend](scene, W, H, K, args.near, args.far)

    hero_img = None
    times_ms: list[float] = []
    for rep in range(args.repeats):
        times_ms = []
        for i, vm in enumerate(viewmats):
            torch.cuda.synchronize()
            t = time.perf_counter()
            img = backend.render(vm)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t) * 1e3)
            if i == 0 and rep == args.repeats - 1:
                hero_img = img.clamp(0, 1).detach().float().cpu().numpy()
        print(f"  pass {rep + 1}/{args.repeats}: "
              f"avg(excl hero) {np.mean(times_ms[1:]):.1f} ms")

    timed = np.asarray(times_ms[1:])       # hero = warmup, excluded
    stats = {
        "n_timed_views": int(timed.size),
        "avg_frame_ms": round(float(timed.mean()), 3),
        "p50_frame_ms": round(float(np.percentile(timed, 50)), 3),
        "min_frame_ms": round(float(timed.min()), 3),
        "max_frame_ms": round(float(timed.max()), 3),
        "fps_from_avg": round(1000.0 / float(timed.mean()), 2),
    }

    from PIL import Image

    args.hero_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((hero_img * 255.0 + 0.5).astype(np.uint8)).save(args.hero_png)
    hero_psnr = psnr_vs_reference(hero_img, args.reference)

    result = {
        "backend": backend.name,
        "backend_version": backend.version,
        "gpu": ident,
        "scene": str(args.ply),
        "n_gaussians": int(scene[0].shape[0]),
        "image_size": [W, H],
        "fov_deg": fov_deg,
        "views": len(order),
        "hero_psnr_db": round(hero_psnr, 4),
        "hero_png": str(args.hero_png),
        "per_view_ms": [round(v, 3) for v in times_ms],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n{backend.name} on {ident.get('name')}: "
          f"avg {stats['avg_frame_ms']} ms/view, p50 {stats['p50_frame_ms']} ms "
          f"({stats['fps_from_avg']} FPS over {stats['n_timed_views']} views)")
    print(f"hero PSNR vs {args.reference}: {hero_psnr:.2f} dB")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
