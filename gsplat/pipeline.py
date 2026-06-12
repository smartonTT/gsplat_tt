"""Pipeline orchestrator with built-in per-stage timing.

The Pipeline calls each stage on the backend, wrapping every call in a
`time.perf_counter()` timer. Backend implementers do not write any
outer-timing code — adding a new backend automatically gets benchmarked
out of the box. Sub-timings (the breakdown inside a stage, e.g. blend's
prep/kernel/readback split) are reported by the backend via the second
return value of `blend(...)`.

Usage:

    pipeline = Pipeline(backend)
    result = pipeline.render(gaussians, extrinsics, intrinsics, H, W)
    result.image          # (H, W, 3) float32 numpy in [0, 1]
    result.timings        # {"project": 12.5, "tile_assign": 3.1,
                          #  "sort": 4.0, "blend": 80.2, "total": 99.8}
    result.sub_timings    # {"blend.prep": 3.5, "blend.kernel": 70.0, ...}
                          # (only populated if backend reports them)
    result.num_visible    # int — Gaussians that passed culling
    result.num_entries    # int — (gaussian, tile) pairs after sort
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat.data_structures import Gaussians


TILE_SIZE = 32
DEFAULT_CONTRIB_FLOOR = 1.0 / 16384.0


@dataclass
class RenderResult:
    """Result of a single Pipeline.render() call.

    `image` is None when the frame had zero visible Gaussians (early-exit).
    Caller should treat that as a black/empty frame; callers that need an
    actual array can do `result.image if result.image is not None else
    np.zeros((H, W, 3), dtype=np.float32)`.
    """
    image: np.ndarray | None
    timings: dict[str, float] = field(default_factory=dict)
    sub_timings: dict[str, float] = field(default_factory=dict)
    num_visible: int = 0
    num_entries: int = 0
    height: int = 0
    width: int = 0


class Pipeline:
    """Glue between a chosen backend and the per-frame stage sequence."""

    def __init__(
        self,
        backend: Backend,
        *,
        tile_size: int = TILE_SIZE,
        cull_disabled: bool = False,
        contrib_floor: float = DEFAULT_CONTRIB_FLOOR,
        transmittance_threshold: float = 1.0 / 255.0,
        min_opacity: float = 1.0 / 255.0,
        # max_radius semantics: 0 = default cap min(H,W)/2 (matches numpy
        # `project_gaussians`); >0 = explicit pixel cap; <0 = disabled.
        # Disabling the cap (-1) lets near-camera, nearly-edge-on Gaussians
        # project to enormous radii — at close zoom these ill-conditioned
        # Gaussians stack into a regular halftone "stipple" pattern (≈ -50 dB
        # vs numpy reference). See tests/spec/test_thin_splat_stipple.py.
        max_radius: int = 0,
        k_cap: float = 3.0,
        use_isoellipse: bool = False,
    ):
        self.backend = backend
        self.tile_size = tile_size
        # cull_disabled bypasses the per-pair Mahalanobis cull AND the
        # per-microblock cull (where applicable). Used for diagnostic
        # comparison against the unfused `alpha_blend` ground truth.
        self.cull_disabled = cull_disabled
        self.contrib_floor = contrib_floor
        self.transmittance_threshold = transmittance_threshold
        self.min_opacity = min_opacity
        self.max_radius = max_radius
        self.k_cap = k_cap
        self.use_isoellipse = use_isoellipse
        self._sync_render_settings_to_backend()

    def _sync_render_settings_to_backend(self) -> None:
        """Push Pipeline render knobs into the fused backend when supported."""
        b = self.backend
        if hasattr(b, "cull_disabled"):
            b.cull_disabled = self.cull_disabled
        if hasattr(b, "transmittance_threshold"):
            b.transmittance_threshold = self.transmittance_threshold
        if hasattr(b, "min_opacity"):
            b.min_opacity = self.min_opacity
        if hasattr(b, "max_radius"):
            b.max_radius = self.max_radius
        if hasattr(b, "contrib_floor_override"):
            b.contrib_floor_override = self.contrib_floor
        if hasattr(b, "_mb_contrib_floor"):
            b._mb_contrib_floor = self.contrib_floor
        if hasattr(b, "k_cap"):
            b.k_cap = self.k_cap
        if hasattr(b, "use_isoellipse"):
            b.use_isoellipse = self.use_isoellipse

    # ------------------------------------------------------------------
    # Per-stage timer
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def _timer(timings: dict[str, float], name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        gaussians: Gaussians,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> RenderResult:
        """Run the full forward pass: project → tile_assign → sort → blend.

        Each stage is timed separately; their wall-clock durations land in
        `result.timings`. The backend's `blend(...)` may also return a
        dict of internal sub-timings, prefixed and copied into
        `result.sub_timings` (e.g. `{"blend.kernel": 70.0}`).
        """
        # iter-029 fast path: backend implements all-stage fused render in
        # a single pybind crossing. Eliminates per-stage numpy<->torch
        # conversions, per-stage pybind call overhead, and the Python-side
        # colors[valid_mask] / opacities[valid_mask] torch indexing.
        if getattr(self.backend, "has_render_fused", lambda: False)():
            t_total = time.perf_counter()
            image, stats = self.backend.render_fused(
                gaussians, extrinsics, intrinsics, image_height, image_width,
            )
            wall_ms = (time.perf_counter() - t_total) * 1000.0
            timings = {
                "project": float(stats.get("project_ms", 0.0)),
                "tile_assign": float(stats.get("tile_assign_ms", 0.0)),
                "sort": float(stats.get("sort_ms", 0.0)),
                "blend": float(stats.get("blend_ms", 0.0)),
                "total": wall_ms,
            }
            pairs_in = int(stats.get("pairs_in", 0) or 0)
            pairs_kept = int(stats.get("pairs_kept_per_mb", 0) or 0)
            pairs_dropped = int(stats.get("pairs_dropped", 0) or 0)
            full_replay = float(pairs_in) * 32.0
            sub_timings: dict[str, float] = {
                "blend.microblock_drop_pct": (
                    0.0 if pairs_in == 0 else 100.0 * pairs_dropped / pairs_in
                ),
                "blend.microblock_work_reduction_pct": (
                    0.0 if full_replay == 0.0 else 100.0 * (1.0 - pairs_kept / full_replay)
                ),
            }
            return RenderResult(
                image=image,
                timings=timings,
                sub_timings=sub_timings,
                num_visible=int(stats.get("num_visible", 0) or 0),
                num_entries=int(stats.get("num_entries", 0) or 0),
                height=image_height,
                width=image_width,
            )

        timings: dict[str, float] = {}
        sub_timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # Stage 1: project (with per-step sub-timings).
        with self._timer(timings, "project"):
            means_2d, covs_2d, depths, radii, valid_mask = self.backend.project(
                gaussians.means, gaussians.scales, gaussians.rotations,
                extrinsics, intrinsics, image_height, image_width,
                opacities=gaussians.opacities,
                sub_timings=sub_timings,
            )
        num_visible = int(valid_mask.sum().item())

        # Empty frame: no visible Gaussians → skip downstream stages.
        if num_visible == 0:
            timings["tile_assign"] = 0.0
            timings["sort"] = 0.0
            timings["blend"] = 0.0
            timings["total"] = (time.perf_counter() - t_total) * 1000.0
            return RenderResult(
                image=None,
                timings=timings,
                sub_timings=sub_timings,
                num_visible=0,
                num_entries=0,
                height=image_height,
                width=image_width,
            )

        colors = gaussians.colors[valid_mask]
        opacities = gaussians.opacities[valid_mask]

        # Stage 2: tile assignment (+ per-pair Mahalanobis cull when cov+ω
        # are available — drops ~22% more pairs on stitch_doll, iter-024).
        # Cull bypassed when self.cull_disabled is set, for diagnostics.
        with self._timer(timings, "tile_assign"):
            gaussian_ids, tile_ids, _ = self.backend.tile_assign(
                means_2d, radii, image_height, image_width,
                tile_size=self.tile_size,
                covs_2d=None if self.cull_disabled else covs_2d,
                opacities=None if self.cull_disabled else opacities,
                contrib_floor=self.contrib_floor,
                sub_timings=sub_timings,
            )
        tiles_x = (image_width + self.tile_size - 1) // self.tile_size
        tiles_y = (image_height + self.tile_size - 1) // self.tile_size

        # Stage 3: sort + bin
        with self._timer(timings, "sort"):
            sorted_gaussian_ids, tile_ranges = self.backend.sort(
                gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
                sub_timings=sub_timings,
            )
        num_entries = int(sorted_gaussian_ids.numel())

        # Stage 4: blend (with optional sub-timings from backend)
        with self._timer(timings, "blend"):
            image, blend_sub = self.backend.blend(
                means_2d, covs_2d, colors, opacities,
                sorted_gaussian_ids, tile_ranges,
                image_height, image_width,
            )
        for k, v in (blend_sub or {}).items():
            sub_timings[f"blend.{k}"] = v

        timings["total"] = (time.perf_counter() - t_total) * 1000.0

        return RenderResult(
            image=image,
            timings=timings,
            sub_timings=sub_timings,
            num_visible=num_visible,
            num_entries=num_entries,
            height=image_height,
            width=image_width,
        )

    def close(self) -> None:
        """Release the backend's resources."""
        self.backend.close()


def format_timings(result: RenderResult) -> str:
    """Pretty-print the per-stage breakdown for verbose / debug output."""
    lines = [f"[stage] {name:<12} {result.timings.get(name, 0.0):6.1f} ms"
             for name in ("project", "tile_assign", "sort", "blend")]
    for k, v in result.sub_timings.items():
        # Indent sub-timings under their parent stage.
        lines.append(f"   └─ {k:<14} {v:6.1f} ms")
    total = result.timings.get("total", 0.0)
    fps = 1000.0 / total if total > 0 else 0.0
    lines.append(f"[total] {total:6.1f} ms  ({fps:.1f} fps)")
    return "\n".join(lines)
