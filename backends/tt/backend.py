"""Long-lived wrapper around the v1c alpha-blend kernel daemon.

The daemon (`metal_example_gaussian_splatting --daemon`) opens the Wormhole
device once, JIT-compiles the three kernels, and then loops reading binary
FRAME requests on stdin. This module spawns it, sends one frame per `blend(...)`
call, and reads back the OK/ERR response — keeping the ~3 s init cost off
the per-frame path so interactive use is feasible.

Implements the `gsplat.backend.Backend` contract: only `blend(...)` is
overridden — projection / tile-assignment / sort run on CPU via the
default implementations inherited from the base class.
"""
from __future__ import annotations

import os
import struct
import subprocess
import time

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat.rasterization import prepare_kernel_inputs

# Binary IPC magic values (little-endian on the wire).
_MAGIC_FRM1 = 0x46524D31  # 'FRM1'
_MAGIC_OK11 = 0x4F4B3131  # 'OK11'
_MAGIC_ERR1 = 0x45525231  # 'ERR1'


def _read_exact(stream, nbytes: int) -> bytes:
    """Read exactly *nbytes* from a binary stream or raise."""
    buf = b""
    while len(buf) < nbytes:
        chunk = stream.read(nbytes - len(buf))
        if not chunk:
            raise RuntimeError("daemon closed stdout unexpectedly")
        buf += chunk
    return buf


def _read_exact_into(stream, buf: memoryview) -> None:
    """Read exactly len(buf) bytes into a writable buffer."""
    filled = 0
    while filled < len(buf):
        n = stream.readinto(buf[filled:])
        if not n:
            raise RuntimeError("daemon closed stdout unexpectedly")
        filled += n


def _wait_for_ready(stream, deadline: float) -> None:
    """Skip tt-metal init log lines until the READY sentinel."""
    buf = b""
    while time.perf_counter() < deadline:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        if b"READY\n" in buf:
            return
    tail = buf[-200:].decode("utf-8", errors="replace")
    raise RuntimeError(f"daemon failed to start: last bytes {tail!r}")


class KernelBackend(Backend):
    """Persistent IPC wrapper around the alpha-blend daemon subprocess.

    Spawn once at viewer/script start, call `blend(...)` per frame,
    `close()` on shutdown. After READY the protocol is binary on
    stdin/stdout (FRM1 frame request → OK11/ERR1 response).
    """

    BINARY_PATH = "backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        env = os.environ.copy()
        env.setdefault("TT_METAL_HOME", os.path.abspath("backends/tt/tt-metal"))
        env.setdefault("TT_METAL_RUNTIME_ROOT", os.path.abspath("backends/tt/tt-metal"))
        self._proc = subprocess.Popen(
            [self.BINARY_PATH, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=False,
            bufsize=0,
        )
        # The daemon may emit tt-metal init log lines on stdout before the
        # READY sentinel; skip past them with a wall-clock deadline. First-run
        # JIT compile on a cold cache (esp. on Blackhole) can exceed a minute,
        # so the deadline is generous.
        _wait_for_ready(self._proc.stdout, time.perf_counter() + 240.0)

    # ------------------------------------------------------------------
    # Backend API
    # ------------------------------------------------------------------

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
        H, W = image_height, image_width

        # Sub-stage A: SoA repack (kernel-friendly layout).
        t_prep = time.perf_counter()
        packs, offsets, px, py = prepare_kernel_inputs(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges, H, W,
        )
        prep_ms = (time.perf_counter() - t_prep) * 1000.0

        # Sub-stage B: write binary frame to daemon stdin (replaces .npy files).
        t_save = time.perf_counter()
        total_entries = int(packs.shape[0])
        offsets_count = int(offsets.shape[0])
        self._proc.stdin.write(struct.pack(
            "<6I",
            _MAGIC_FRM1,
            H,
            W,
            total_entries,
            offsets_count,
            0,
        ))
        for arr in (packs, offsets, px, py):
            self._proc.stdin.write(
                memoryview(np.ascontiguousarray(arr, dtype=np.float32)))
        self._proc.stdin.flush()
        save_ms = (time.perf_counter() - t_save) * 1000.0

        # Sub-stage C: daemon round-trip (kernel + response header).
        t_rt = time.perf_counter()
        resp_hdr = _read_exact(self._proc.stdout, 16)
        magic, image_bytes, kernel_us, err_len = struct.unpack("<4I", resp_hdr)
        if magic == _MAGIC_ERR1:
            err_msg = _read_exact(self._proc.stdout, err_len).decode("utf-8")
            raise RuntimeError(f"daemon error: {err_msg}")
        if magic != _MAGIC_OK11:
            raise RuntimeError(f"unexpected daemon response magic {magic:#010x}")
        rt_ms = (time.perf_counter() - t_rt) * 1000.0

        kernel_ms = kernel_us / 1000.0 if kernel_us else None

        # Sub-stage D: read rendered image bytes from stdout.
        t_load = time.perf_counter()
        image = np.empty((H, W, 3), dtype=np.float32)
        _read_exact_into(self._proc.stdout, memoryview(image).cast("B"))
        load_ms = (time.perf_counter() - t_load) * 1000.0

        sub_timings: dict[str, float] = {
            "prep": prep_ms,
            "save_npy": save_ms,
            "daemon_rt": rt_ms,
        }
        if kernel_ms is not None:
            sub_timings["daemon_rt.device_kernel"] = kernel_ms
        sub_timings["load_npy"] = load_ms

        return image, sub_timings

    def close(self) -> None:
        # Process cleanup: try graceful QUIT first; if the daemon doesn't
        # exit promptly, hard-kill so the Wormhole device is released.
        # Leaving a daemon orphaned holds the device and breaks the next
        # invocation (the tt-metal driver hangs trying to acquire it).
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(b"QUIT\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
