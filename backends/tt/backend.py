"""Tenstorrent backend — amendment-002 in-process port.

tt-000: delegates entirely to cpu_cpp_mb (proves wiring + PSNR baseline).
tt-001a+: overrides blend() with gsplat_tt device kernels via pybind when
`has_tt_support()` is true.
tt-005:  overrides project() with device transform_means_cam (bounded
hotspot port) under `GSPLAT_TT_DEVICE_PROJECT=1`.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from backends.cpu_cpp.backend import CpuCppBackend


class TtBackend(CpuCppBackend):
    """cpu_cpp_mb pipeline with TT kernels swapped in stage-by-stage.

    Architecture (plan-amendment-002): inherit the entire cpu_cpp C++ pipeline
    and override ONE stage at a time with a TT-Metal kernel. The Pipeline
    fused fast-path (`render_fused`) is force-disabled here because it
    bypasses per-stage overrides — otherwise the TT kernel would never be
    invoked.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("microblock", True)
        kwargs.setdefault("fused", True)
        # Force render_fused OFF: the fused C++ render_full() entrypoint
        # bypasses .project()/.tile_assign()/.sort()/.blend() overrides, so
        # any TT kernel we wire in would never execute during full-pipeline
        # rendering. The 30-view bench would silently measure cpu_cpp_mb.
        kwargs["render_fused"] = False
        # bh-30 hosts have 96 hardware threads but the per-tile cpu_cpp blend
        # tops out around 48 workers (measured 2026-05-28: 96=578 ms,
        # 48=514 ms, 24=518 ms, 12=684 ms hero render). Cap the pool unless
        # the operator overrides via GSPLAT_TT_NUM_THREADS.
        import os
        os.environ.setdefault("GSPLAT_TT_NUM_THREADS", "48")
        super().__init__(**kwargs)
        self._tt_blend = None
        self._tt_transform_means_cam = None
        try:
            from backends.cpu_cpp import _gsplat_cpu as mod

            if mod.has_tt_support():
                self._tt_blend = mod.blend_microblock_tt
                if hasattr(mod, "transform_means_cam_tt"):
                    self._tt_transform_means_cam = mod.transform_means_cam_tt
        except (ImportError, AttributeError):
            pass

    def has_render_fused(self) -> bool:
        # Defensive: parent class also reads self._render_fused, but make
        # the contract explicit so any future code path that introspects
        # this can't accidentally re-enable the fused bypass on TtBackend.
        return False

    def project(
        self,
        means,
        scales,
        rotations,
        extrinsics,
        intrinsics,
        image_height: int,
        image_width: int,
        opacities=None,
        sub_timings: dict | None = None,
    ):
        """Project stage with optional device transform_means_cam.

        Gated by `GSPLAT_TT_DEVICE_PROJECT=1`. When enabled and device init
        succeeds, the world→camera matrix-vec (means_cam = R @ means) is
        computed on Tenstorrent device via `transform_means_cam_tt` and
        passed into `project_full_with_cov3d` as a precomputed array — this
        skips the inline host matmul in `project_full_fused` (project.cpp:
        518-535).

        When the env flag is off OR the device path fails, falls back to the
        unmodified CPU path via super().project(). Default-off semantics
        preserve the existing PSNR (no rendering math changes).
        """
        if (
            os.environ.get("GSPLAT_TT_DEVICE_PROJECT") != "1"
            or self._tt_transform_means_cam is None
        ):
            return super().project(
                means,
                scales,
                rotations,
                extrinsics,
                intrinsics,
                image_height,
                image_width,
                opacities=opacities,
                sub_timings=sub_timings,
            )

        import time as _time
        _t_detach0 = _time.perf_counter()
        means_np = means.detach().cpu().numpy().astype(np.float32, copy=False)
        scales_np = scales.detach().cpu().numpy().astype(np.float32, copy=False)
        rotations_np = rotations.detach().cpu().numpy().astype(np.float32, copy=False)
        extrinsics_np = extrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        intrinsics_np = intrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        opacities_np = None
        if opacities is not None:
            opacities_np = opacities.detach().cpu().numpy().astype(np.float32, copy=False)

        _t_marshal_in = _time.perf_counter()
        means_np_c = np.ascontiguousarray(means_np)
        extr_np_c = np.ascontiguousarray(extrinsics_np)
        _t_tt0 = _time.perf_counter()
        tt_ret = self._tt_transform_means_cam(means_np_c, extr_np_c)
        _t_tt1 = _time.perf_counter()
        # Backward-compat: legacy binding returned (means_cam, float kernel_ms).
        # tt-005c binding returns (means_cam, dict).
        if isinstance(tt_ret, tuple) and len(tt_ret) == 2:
            means_cam_np, _meta = tt_ret
        else:
            means_cam_np, _meta = tt_ret, {}
        if isinstance(_meta, dict):
            kernel_ms = float(_meta.get("total_ms", -1.0))
            tt_timings = _meta
        else:
            kernel_ms = float(_meta)
            tt_timings = {}

        if kernel_ms < 0.0:
            # Device init failed — fall back to CPU path for this frame.
            return super().project(
                means,
                scales,
                rotations,
                extrinsics,
                intrinsics,
                image_height,
                image_width,
                opacities=opacities,
                sub_timings=sub_timings,
            )

        _t_cov3d0 = _time.perf_counter()
        cov3d = self._cached_cov3d(scales_np, rotations_np)
        _t_cov3d1 = _time.perf_counter()
        means_2d, covs_2d_out, depths, radii, valid_mask = self._mod.project_full_with_cov3d(
            means_np,
            cov3d,
            extrinsics_np,
            intrinsics_np,
            image_height,
            image_width,
            opacities_np if opacities_np is not None else None,
            1.0 / 255.0,
            means_cam_np,
        )
        _t_pfwc = _time.perf_counter()

        if sub_timings is not None:
            sub_timings["tt_means_cam_kernel_ms"] = float(kernel_ms)
            for k, v in tt_timings.items():
                sub_timings[f"tt_mc_{k}"] = v
            sub_timings["tt_py_detach_ms"] = (_t_marshal_in - _t_detach0) * 1000.0
            sub_timings["tt_py_marshal_ms"] = (_t_tt0 - _t_marshal_in) * 1000.0
            sub_timings["tt_py_call_ms"] = (_t_tt1 - _t_tt0) * 1000.0
            sub_timings["tt_py_cov3d_ms"] = (_t_cov3d1 - _t_cov3d0) * 1000.0
            sub_timings["tt_py_pfwc_ms"] = (_t_pfwc - _t_cov3d1) * 1000.0

        return (
            torch.from_numpy(np.asarray(means_2d)),
            torch.from_numpy(np.asarray(covs_2d_out)),
            torch.from_numpy(np.asarray(depths)),
            torch.from_numpy(np.asarray(radii)),
            torch.from_numpy(np.asarray(valid_mask)),
        )

    def blend(
        self,
        means_2d,
        covs_2d,
        colors,
        opacities,
        sorted_gaussian_ids,
        tile_ranges,
        image_height,
        image_width,
    ):
        if self._tt_blend is None:
            return super().blend(
                means_2d,
                covs_2d,
                colors,
                opacities,
                sorted_gaussian_ids,
                tile_ranges,
                image_height,
                image_width,
            )
        import os

        from gsplat.rasterization import prepare_microblock_payload

        # Host correctness path (default): identical blend to cpu_cpp_mb.
        # GSPLAT_TT_DEVICE_BLEND=1 routes through blend_microblock_tt host
        # from_packs path (~53.5 dB vs fused cull_and_blend; passes 45 dB gate).
        # GSPLAT_TT_DEVICE_KERNEL=1 additionally selects the TT device kernel
        # (legacy full-tile loop; PSNR ~21 dB until Stage 3 lands).
        if os.environ.get("GSPLAT_TT_DEVICE_BLEND") != "1":
            return super().blend(
                means_2d,
                covs_2d,
                colors,
                opacities,
                sorted_gaussian_ids,
                tile_ranges,
                image_height,
                image_width,
            )

        payload = prepare_microblock_payload(
            means_2d,
            covs_2d,
            colors,
            opacities,
            sorted_gaussian_ids,
            tile_ranges,
            image_height,
            image_width,
            mb_contrib_floor=float(self._mb_contrib_floor),
        )
        import numpy as np

        image, kernel_ms = self._tt_blend(
            np.ascontiguousarray(payload["attribute_packs"], dtype=np.float32),
            np.ascontiguousarray(payload["tile_offsets"], dtype=np.float32),
            np.ascontiguousarray(payload["px_tiles"], dtype=np.float32),
            np.ascontiguousarray(payload["py_tiles"], dtype=np.float32),
            np.ascontiguousarray(payload["coeff_table"], dtype=np.float32),
            np.ascontiguousarray(payload["mb_header"], dtype=np.uint32),
            np.ascontiguousarray(payload["mb_stream_local"], dtype=np.uint32),
            int(image_height),
            int(image_width),
        )
        stats = payload["mb_stats"]
        pairs_in = stats["pairs_in"]
        pairs_out = stats["pairs_out"]
        full_replay = float(pairs_in) * 32.0
        work_reduction_pct = (
            0.0 if full_replay == 0.0 else 100.0 * (1.0 - float(pairs_out) / full_replay)
        )
        drop_pct = stats["drop_pct"]
        sub = {
            "microblock_drop_pct": float(drop_pct),
            "microblock_work_reduction_pct": work_reduction_pct,
            "device_kernel_ms": float(kernel_ms),
        }
        return image, sub

    def close(self) -> None:
        try:
            from backends.cpu_cpp import _gsplat_cpu as mod

            if mod.has_tt_support() and hasattr(mod, "tt_device_shutdown"):
                mod.tt_device_shutdown()
        except (ImportError, AttributeError):
            pass
