"""Regression test for the close-zoom "stippling" artifact.

The `render_full` C++ fused kernel used to diverge from the numpy reference
(`backends.cpu` → `gsplat.rasterization.alpha_blend`) at close zoom — it
produced a regular halftone-pattern smear on top of the actual scene. The
root cause was the Pipeline (and `CpuCppBackend`) default of `max_radius=-1`
(disabled). Near-camera, nearly-edge-on Gaussians project to enormous AABB
radii via the perspective Jacobian; with no cap, hundreds of these
ill-conditioned Gaussians stack and stipple. The numpy reference always uses
`max_radius=0` (cap at min(H,W)/2).

This test renders one giant, thin Gaussian close to the camera through both
backends and checks that the C++ fused render matches the numpy reference
within a tight PSNR threshold. Pre-fix this scene PSNR'd at ~17 dB; post-fix
it is >60 dB (essentially identical up to float32 reduction order).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backends import get_backend  # noqa: E402
from gsplat.data_structures import Gaussians  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def _identity_quat(n: int) -> torch.Tensor:
    q = torch.zeros((n, 4), dtype=torch.float32)
    q[:, 0] = 1.0
    return q


def _close_zoom_thin_scene() -> tuple[Gaussians, torch.Tensor, torch.Tensor, int, int]:
    """One nearly-edge-on, sub-pixel-thin Gaussian close to the camera.

    Geometry: a slab oriented so its small axis is perpendicular to view
    while a long axis lies close to the camera ray. This is the exact
    configuration the perspective Jacobian blows up the projected
    covariance for, exposing the missing AABB cap.
    """
    H = W = 256
    fov_deg = 50.0
    f = 0.5 * max(W, H) / np.tan(0.5 * np.deg2rad(fov_deg))
    K = torch.tensor(
        [[f, 0, W * 0.5], [0, f, H * 0.5], [0, 0, 1.0]], dtype=torch.float32
    )

    means = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32)
    scales = torch.tensor([[0.15, 0.002, 0.15]], dtype=torch.float32)
    rotations = _identity_quat(1)
    opacities = torch.tensor([0.9], dtype=torch.float32)
    colors = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

    g = Gaussians(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )

    c2w = torch.eye(4, dtype=torch.float32)
    extr = c2w_to_w2c(c2w)
    return g, extr, K, H, W


def _multi_splat_close_zoom() -> tuple[Gaussians, torch.Tensor, torch.Tensor, int, int]:
    """Many thin Gaussians at close zoom — closer to the chal_top regime.

    Each Gaussian has a different rotation so their projected AABBs span a
    wide range; under the old `max_radius=-1` default a sizeable subset
    projects to >512 px radii and overdraws the image into a halftone
    pattern.
    """
    rng = np.random.default_rng(0)
    N = 4096
    H = W = 256
    fov_deg = 50.0
    f = 0.5 * max(W, H) / np.tan(0.5 * np.deg2rad(fov_deg))
    K = torch.tensor(
        [[f, 0, W * 0.5], [0, f, H * 0.5], [0, 0, 1.0]], dtype=torch.float32
    )

    xy = rng.uniform(-0.15, 0.15, size=(N, 2)).astype(np.float32)
    z = rng.uniform(0.30, 0.60, size=(N, 1)).astype(np.float32)
    means = torch.from_numpy(np.concatenate([xy, z], axis=1))

    # Highly anisotropic scales so projected Σ_2D stretches under the
    # perspective Jacobian (long axis ≈ along camera ray for some splats).
    scales = torch.from_numpy(
        rng.uniform(0.0005, 0.05, size=(N, 3)).astype(np.float32)
    )
    # Random rotations as unit quaternions
    q = rng.standard_normal(size=(N, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rotations = torch.from_numpy(q)

    opacities = torch.from_numpy(rng.uniform(0.05, 0.9, size=(N,)).astype(np.float32))
    colors = torch.from_numpy(rng.uniform(0.0, 1.0, size=(N, 3)).astype(np.float32))

    g = Gaussians(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )
    c2w = torch.eye(4, dtype=torch.float32)
    extr = c2w_to_w2c(c2w)
    return g, extr, K, H, W


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _synth_blend_inputs(
    cov_a: float,
    cov_b: float,
    cov_c: float,
    *,
    W: int = 512,
    H: int = 512,
    opacity: float = 0.8,
    tile_size: int = 32,
):
    """Build tile_assign + sort outputs for a single 2D Gaussian at image
    center, ready to feed into both `alpha_blend` (numpy) and `cpp.blend`.
    """
    from gsplat.rasterization import get_tile_assignments, sort_and_bin

    means_2d = torch.tensor([[W * 0.5, H * 0.5]], dtype=torch.float32)
    covs_2d = torch.tensor(
        [[[cov_a, cov_b], [cov_b, cov_c]]], dtype=torch.float32
    )
    colors = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([opacity], dtype=torch.float32)
    rx = 3.0 * float(np.sqrt(max(cov_a, 0.0)))
    ry = 3.0 * float(np.sqrt(max(cov_c, 0.0)))
    radii = torch.tensor([[rx, ry]], dtype=torch.float32)
    depths = torch.tensor([1.0], dtype=torch.float32)
    gids, tids, _ = get_tile_assignments(
        means_2d, radii, H, W, tile_size=tile_size
    )
    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    sgids, tranges = sort_and_bin(gids, tids, depths, tiles_x, tiles_y)
    return {
        "means_2d": means_2d,
        "covs_2d": covs_2d,
        "colors": colors,
        "opacities": opacities,
        "sgids": sgids,
        "tranges": tranges,
        "H": H,
        "W": W,
        "tile_size": tile_size,
    }


def _render_alpha_blend(s) -> np.ndarray:
    from gsplat.rasterization import alpha_blend

    img = alpha_blend(
        s["means_2d"], s["covs_2d"], s["colors"], s["opacities"],
        s["sgids"], s["tranges"], s["H"], s["W"], tile_size=s["tile_size"],
    )
    return img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)


def _render_cpp_blend(s) -> np.ndarray:
    from backends.cpu_cpp import _gsplat_cpu as cpp

    return np.asarray(cpp.blend(
        np.ascontiguousarray(s["means_2d"].numpy(), dtype=np.float32),
        np.ascontiguousarray(s["covs_2d"].numpy().reshape(-1, 4), dtype=np.float32),
        np.ascontiguousarray(s["colors"].numpy(), dtype=np.float32),
        np.ascontiguousarray(s["opacities"].numpy(), dtype=np.float32),
        np.ascontiguousarray(s["sgids"].numpy(), dtype=np.int64),
        np.ascontiguousarray(s["tranges"].numpy(), dtype=np.int64),
        s["H"], s["W"], s["tile_size"],
    ))


# Single-Gaussian coverage of the well-behaved → degenerate → outright-broken
# regime. The blend kernel is supposed to give bit-identical pixels in all
# regimes (modulo float32 reduction order). If a future iter changes one
# kernel and not the other, the math contract breaks and this test
# discovers it long before it reaches a real scene.
_BLEND_EQUIVALENCE_CASES = [
    # (label, cov_a, cov_b, cov_c). Σ_2D = [[a, b], [b, c]].
    ("iso_smooth",                1.0e4, 0.0,       1.0e4),    # σ=100 both
    ("aniso_smooth",              1.0e4, 0.0,       1.0e2),    # σ=100×10
    ("aniso_pixel_thin",          1.0e4, 0.0,       4.0e-2),   # σ=100×0.2
    ("tilted_45",                 5050.0, 4950.0,   5050.0),   # σ_major≈100, 45°
    ("near_singular_huge",        1.0e6, -0.999e6,  1.0e6),    # det → 0
    ("giant_thin",                1.0e9, 0.0,       1.0),      # σ_x≈31623, σ_y=1
]


@pytest.mark.parametrize("label,a,b,c", _BLEND_EQUIVALENCE_CASES)
def test_cpp_blend_matches_alpha_blend(label, a, b, c):
    """`cpp.blend` and numpy `alpha_blend` must agree pixel-for-pixel on the
    SAME projected Gaussian — across the whole well-behaved → degenerate
    spectrum. This is the load-bearing math-equivalence guarantee: if both
    kernels stipple at a degenerate input it is the EWA math, not a code
    bug; if only one stipples that is a bug.
    """
    s = _synth_blend_inputs(a, b, c)
    img_np = _render_alpha_blend(s)
    img_cpp = _render_cpp_blend(s)
    psnr = _psnr(img_np, img_cpp)
    max_abs = float(np.max(np.abs(img_np - img_cpp)))
    assert psnr >= 60.0 and max_abs < 1e-3, (
        f"{label}: cpp.blend diverges from numpy alpha_blend "
        f"(PSNR={psnr:.1f} dB, max|err|={max_abs:.4f}) — the per-pixel "
        f"EWA evaluator is no longer numerically equivalent across "
        f"backends, which would silently corrupt any scene render."
    )


@pytest.mark.parametrize(
    "scene_fn,label,min_psnr",
    [
        (_close_zoom_thin_scene, "single_thin_close", 50.0),
        (_multi_splat_close_zoom, "multi_thin_close", 35.0),
    ],
)
def test_cpu_cpp_fused_matches_numpy_reference(scene_fn, label, min_psnr):
    g, extr, K, H, W = scene_fn()

    ref = Pipeline(get_backend("cpu"), contrib_floor=1.0 / 255.0).render(
        g, extr, K, H, W
    )
    cpp = Pipeline(get_backend("cpu_cpp"), contrib_floor=1.0 / 255.0).render(
        g, extr, K, H, W
    )

    ref_img = ref.image if ref.image is not None else np.zeros((H, W, 3), np.float32)
    cpp_img = cpp.image if cpp.image is not None else np.zeros((H, W, 3), np.float32)

    psnr = _psnr(cpp_img, ref_img)
    assert psnr >= min_psnr, (
        f"{label}: cpu_cpp render diverges from numpy reference ({psnr:.1f} dB "
        f"< {min_psnr} dB threshold). Likely regression of the close-zoom "
        f"`max_radius` cap or the project k-floor."
    )


def test_close_zoom_does_not_overdraw():
    """Pre-fix, the buggy `max_radius=-1` admitted Gaussians whose projected
    radii exceeded ~3000 px, blanketing the image with stripe artifacts.
    Verify the default-config render produces a finite, mostly-empty frame
    for the trivial single-splat scene (no halftone fills the background).
    """
    g, extr, K, H, W = _close_zoom_thin_scene()
    cpp = Pipeline(get_backend("cpu_cpp"), contrib_floor=1.0 / 255.0).render(
        g, extr, K, H, W
    )
    img = cpp.image if cpp.image is not None else np.zeros((H, W, 3), np.float32)
    assert np.isfinite(img).all(), "Render produced NaN/Inf pixels"
    # The numpy reference drops this near-camera splat entirely
    # (projected radius far exceeds the H/2 cap). Pre-fix the C++ fused
    # path kept it and overdrew ~6% of the image with a halftone smear;
    # post-fix the cap is respected and 0% of pixels carry contribution.
    nonzero_frac = float(np.mean(img.max(axis=-1) > 1e-4))
    assert nonzero_frac < 0.01, (
        f"Single thin near-camera Gaussian covers {nonzero_frac:.2%} of pixels "
        f"(expected <1%); the `max_radius` cap is not being applied to the "
        f"fused render path."
    )


if __name__ == "__main__":
    # Manual run: print metrics + dump PNGs for inspection.
    from PIL import Image

    OUT = Path("/tmp/stipple_repro")
    OUT.mkdir(exist_ok=True, parents=True)

    for fn, label in (
        (_close_zoom_thin_scene, "single_thin_close"),
        (_multi_splat_close_zoom, "multi_thin_close"),
    ):
        g, extr, K, H, W = fn()
        ref = Pipeline(get_backend("cpu"), contrib_floor=1.0 / 255.0).render(
            g, extr, K, H, W
        )
        cpp = Pipeline(get_backend("cpu_cpp"), contrib_floor=1.0 / 255.0).render(
            g, extr, K, H, W
        )
        ref_img = (
            ref.image if ref.image is not None else np.zeros((H, W, 3), np.float32)
        )
        cpp_img = (
            cpp.image if cpp.image is not None else np.zeros((H, W, 3), np.float32)
        )
        Image.fromarray((np.clip(ref_img, 0, 1) * 255).astype(np.uint8)).save(
            OUT / f"{label}_cpu_numpy.png"
        )
        Image.fromarray((np.clip(cpp_img, 0, 1) * 255).astype(np.uint8)).save(
            OUT / f"{label}_cpu_cpp.png"
        )
        print(
            f"{label:24s}  PSNR = {_psnr(cpp_img, ref_img):5.1f} dB   "
            f"entries: cpu={ref.num_entries:>6,} cpu_cpp={cpp.num_entries:>6,}"
        )
    print(f"\nPNGs in {OUT}/")
