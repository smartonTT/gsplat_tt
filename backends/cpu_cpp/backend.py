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

    def __init__(self):
        # Lazily import the compiled module so import doesn't fail when the
        # extension hasn't been built yet.
        from backends.cpu_cpp import _gsplat_cpu  # the pybind11 extension

        self._mod = _gsplat_cpu
        # Smoke check: call hello() to confirm the binding works.
        assert self._mod.hello() == "hello from gsplat_cpu"

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
        # Delegate to numpy reference for now.
        from backends.cpu.backend import CpuBackend

        return CpuBackend().blend(
            means_2d,
            covs_2d,
            colors,
            opacities,
            sorted_gaussian_ids,
            tile_ranges,
            image_height,
            image_width,
        )
