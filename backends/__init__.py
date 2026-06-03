"""Per-architecture backends for the alpha-blend forward pass.

Each subpackage exposes a `backend.py` with a class implementing the
`gsplat.backend.Backend` ABC: at minimum a `blend(...)` method, plus
`project / tile_assign / sort` if the backend accelerates those (defaults
fall back to CPU PyTorch). See `backends/README.md` for instructions on
adding a new backend.

Backends register here. The viewer / CLI consume `REGISTRY` so a new
backend is added by editing one line below — no edits in `gsplat/`
required.
"""
from __future__ import annotations

from gsplat.backend import Backend
from backends.cpu.backend import CpuBackend, CpuMicroblockBackend
from backends.tt.backend import TtBackend


REGISTRY: dict[str, type[Backend]] = {
    "cpu": CpuBackend,
    "cpu_mb": CpuMicroblockBackend,
    "tt": TtBackend,
}

# CudaBackend is optional — register only if its module imports cleanly.
# On a CPU-only box, the import itself is harmless (no top-level torch.cuda
# access), but we keep the try/except so a future kernel-import failure
# inside backends/cuda/* doesn't prevent the CPU/TT backends from loading.
try:
    from backends.cuda.backend import CudaBackend
    REGISTRY["cuda"] = CudaBackend
except ImportError:
    pass

try:
    from backends.cpu_cpp.backend import CpuCppBackend
    REGISTRY["cpu_cpp"] = CpuCppBackend

    class CpuCppMbBackend(CpuCppBackend):
        def __init__(self, **kwargs):
            kwargs["microblock"] = True
            super().__init__(**kwargs)

    REGISTRY["cpu_cpp_mb"] = CpuCppMbBackend
except (ImportError, ModuleNotFoundError, AssertionError):
    pass


def get_backend(name: str, **kwargs) -> Backend:
    """Construct the backend registered under `name`.

    Raises KeyError with a helpful message listing the available backends.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        avail = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown backend {name!r}; available: {avail}") from None
    return cls(**kwargs)


__all__ = ["Backend", "REGISTRY", "get_backend"]
