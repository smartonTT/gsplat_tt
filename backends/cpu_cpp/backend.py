"""C++ backend scaffolding (iter-001).

Every stage currently delegates to the numpy reference; the C++ work
is added one stage at a time in iters 002-004.
"""
from __future__ import annotations

import numpy as np
import torch

from gsplat.backend import Backend


class CpuCppBackend(Backend):
    """C++ backend scaffolding (iter-001).

    Every stage currently delegates to the numpy reference; the C++ work
    is added one stage at a time in iters 002-004. The point of this class
    existing now is to prove the build + binding pipeline works and to
    give a concrete registration target in backends/__init__.py.
    """

    def __init__(
        self,
        microblock: bool = True,
        mb_contrib_floor: float = 1.0 / 16384.0,
        fused: bool = True,
    ):
        # Lazily import the compiled module so import doesn't fail when the
        # extension hasn't been built yet.
        from backends.cpu_cpp import _gsplat_cpu  # the pybind11 extension

        self._mod = _gsplat_cpu
        self._microblock = microblock
        self._mb_contrib_floor = mb_contrib_floor
        self._fused = fused
        # cov3d cache (iter-012): per-Gaussian cov3d depends only on (scales,
        # rotations), not on camera. Cache it across views so the 30-frame
        # training-pattern bench (same scene, 30 cameras) builds cov3d once.
        # Cache key = (buffer ptr, shape) of scales + rotations. Buffer ptr
        # captures both "same numpy array" and "same memory region after
        # in-place updates"; shape catches resizes. If the user is training
        # and replaces scales/rotations with new tensors each step the cache
        # will simply miss and recompute — correct, not faster.
        self._cov3d_cache_key: tuple | None = None
        self._cov3d_cache_buf: np.ndarray | None = None
        assert self._mod.hello() == "hello from gsplat_cpu"

    def _cached_cov3d(
        self, scales_np: np.ndarray, rotations_np: np.ndarray
    ) -> np.ndarray:
        key = (
            scales_np.__array_interface__["data"][0],
            scales_np.shape,
            rotations_np.__array_interface__["data"][0],
            rotations_np.shape,
        )
        if self._cov3d_cache_key == key and self._cov3d_cache_buf is not None:
            return self._cov3d_cache_buf
        cov3d = np.asarray(self._mod.compute_cov3d(scales_np, rotations_np))
        self._cov3d_cache_key = key
        self._cov3d_cache_buf = cov3d
        return cov3d

    def project(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
        opacities: torch.Tensor | None = None,
        sub_timings: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del sub_timings  # C++ path does not emit sub-timings yet.
        means_np = means.detach().cpu().numpy().astype(np.float32, copy=False)
        scales_np = scales.detach().cpu().numpy().astype(np.float32, copy=False)
        rotations_np = rotations.detach().cpu().numpy().astype(np.float32, copy=False)
        extrinsics_np = extrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        intrinsics_np = intrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        opacities_np = None
        if opacities is not None:
            opacities_np = opacities.detach().cpu().numpy().astype(np.float32, copy=False)

        cov3d = self._cached_cov3d(scales_np, rotations_np)
        means_2d, covs_2d_out, depths, radii, valid_mask = self._mod.project_full_with_cov3d(
            means_np,
            cov3d,
            extrinsics_np,
            intrinsics_np,
            image_height,
            image_width,
            opacities_np if opacities_np is not None else None,
            1.0 / 255.0,
        )

        return (
            torch.from_numpy(np.asarray(means_2d)),
            torch.from_numpy(np.asarray(covs_2d_out)),
            torch.from_numpy(np.asarray(depths)),
            torch.from_numpy(np.asarray(radii)),
            torch.from_numpy(np.asarray(valid_mask)),
        )

    def tile_assign(
        self,
        means_2d: torch.Tensor,
        radii: torch.Tensor,
        image_height: int,
        image_width: int,
        tile_size: int = 32,
        covs_2d: torch.Tensor | None = None,
        opacities: torch.Tensor | None = None,
        sub_timings: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del sub_timings
        means_np = means_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        radii_np = radii.detach().cpu().numpy().astype(np.float32, copy=False)
        covs_np = None
        if covs_2d is not None:
            covs_np = covs_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        opacities_np = None
        if opacities is not None:
            opacities_np = opacities.detach().cpu().numpy().astype(np.float32, copy=False)

        gids, tids, tpg = self._mod.tile_assign(
            means_np,
            radii_np,
            int(image_height),
            int(image_width),
            int(tile_size),
            covs_np,
            opacities_np,
            15.0 / 255.0,
        )
        return (
            torch.from_numpy(np.asarray(gids)),
            torch.from_numpy(np.asarray(tids)),
            torch.from_numpy(np.asarray(tpg)),
        )

    def sort(
        self,
        gaussian_ids: torch.Tensor,
        tile_ids: torch.Tensor,
        depths: torch.Tensor,
        tiles_x: int,
        tiles_y: int,
        sub_timings: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del sub_timings
        gids_np = gaussian_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        tids_np = tile_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        depths_np = depths.detach().cpu().numpy().astype(np.float32, copy=False)

        sgids, tranges = self._mod.sort(
            gids_np,
            tids_np,
            depths_np,
            int(tiles_x),
            int(tiles_y),
        )
        return (
            torch.from_numpy(np.asarray(sgids)),
            torch.from_numpy(np.asarray(tranges)),
        )

    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if not self._microblock:
            img = self._mod.blend(
                np.ascontiguousarray(means_2d.detach().cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(
                    covs_2d.detach().cpu().numpy().reshape(-1, 4), dtype=np.float32
                ),
                np.ascontiguousarray(colors.detach().cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(opacities.detach().cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(
                    sorted_gaussian_ids.detach().cpu().numpy(), dtype=np.int64
                ),
                np.ascontiguousarray(tile_ranges.detach().cpu().numpy(), dtype=np.int64),
                int(image_height),
                int(image_width),
                32,
            )
            return img, {}

        tiles_x = (image_width + 31) // 32
        tiles_y = (image_height + 31) // 32
        # Single numpy conversion shared between microblock_cull and
        # blend_microblock — saves ~ms of redundant detach/cpu/cast on each
        # 30-frame iter.
        means_np = np.ascontiguousarray(means_2d.detach().cpu().numpy(), dtype=np.float32)
        covs_np = np.ascontiguousarray(
            covs_2d.detach().cpu().numpy().reshape(-1, 4), dtype=np.float32
        )
        opac_np = np.ascontiguousarray(opacities.detach().cpu().numpy(), dtype=np.float32)
        colors_np = np.ascontiguousarray(colors.detach().cpu().numpy(), dtype=np.float32)
        sgids_np = np.ascontiguousarray(
            sorted_gaussian_ids.detach().cpu().numpy(), dtype=np.int64
        )
        tranges_np = np.ascontiguousarray(tile_ranges.detach().cpu().numpy(), dtype=np.int64)

        if self._fused:
            image, stats = self._mod.cull_and_blend(
                means_np, covs_np, colors_np, opac_np,
                sgids_np, tranges_np,
                int(tiles_x), int(tiles_y), 32,
                int(image_height), int(image_width),
                float(self._mb_contrib_floor),
            )
            pairs_in = stats["pairs_in"]
            pairs_kept = stats["pairs_kept_per_mb"]
            pairs_dropped = stats["pairs_dropped"]
            full_replay = float(pairs_in) * 32.0
            work_reduction_pct = (
                0.0 if full_replay == 0.0 else 100.0 * (1.0 - float(pairs_kept) / full_replay)
            )
            drop_pct = (
                0.0 if pairs_in == 0 else 100.0 * float(pairs_dropped) / float(pairs_in)
            )
            return image, {
                "microblock_drop_pct": drop_pct,
                "microblock_work_reduction_pct": work_reduction_pct,
            }

        mb_header_np, mb_stream_np, stats = self._mod.microblock_cull(
            means_np, covs_np, opac_np, sgids_np, tranges_np,
            int(tiles_x), int(tiles_y), 32,
            float(self._mb_contrib_floor),
        )
        image = self._mod.blend_microblock(
            means_np, covs_np, colors_np, opac_np,
            mb_header_np, mb_stream_np,
            int(image_height), int(image_width), 32,
        )
        return image, {
            "microblock_drop_pct": stats["drop_pct"],
            "microblock_work_reduction_pct": stats["work_reduction_pct"],
        }
