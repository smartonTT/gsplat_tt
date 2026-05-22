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
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat.rasterization import _get_px_py_grids

# Binary IPC magic values (little-endian on the wire).
_MAGIC_SCN1 = 0x53434E31  # 'SCN1'
_MAGIC_OK31 = 0x4F4B3331  # 'OK31'
_MAGIC_FRM2 = 0x46524D32  # 'FRM2'
_MAGIC_OK11 = 0x4F4B3131  # 'OK11'
_MAGIC_ERR1 = 0x45525231  # 'ERR1'
_MAGIC_SHM1 = 0x53484D31  # 'SHM1'

# ---------------------------------------------------------------------------
# Shared-memory IPC (iter 024) — enabled by GSPLAT_TT_USE_SHM=1
#
# Two POSIX SHM segments are negotiated with the daemon after READY via SHM1:
#
#   shm_in  (Python writes, daemon reads):  frame data for FRM2
#     [0..63]:  header  — total_entries, offsets_count, image_h, image_w
#                         (4 × uint32 at bytes 0..15, rest reserved)
#     [64..]:   attr data   — (total_entries, 5) float32
#               gids data   — (total_entries,)   uint32
#               offsets data— (offsets_count,)    float32 (converted to u32 in daemon)
#               px data     — (num_tiles*1024,)   float32  (tile-major bf16 input)
#               py data     — same
#
#   shm_out (daemon writes, Python reads):  rendered fp32 image
#     [0..]:    (image_h * image_w * 3) float32, row-major
#
# Environment variables:
#   GSPLAT_TT_USE_SHM=1          — enable (default: 0)
#   GSPLAT_TT_SHM_MAX_ENTRIES    — max (Gaussian,tile) pairs (default: 2097152)
#   GSPLAT_TT_SHM_MAX_TILES      — max tile count (default: 1024)
# ---------------------------------------------------------------------------

_SHM_HDR_BYTES = 64   # reserved bytes at start of shm_in for the per-frame header


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


# ---------------------------------------------------------------------------
# Per-resolution tile-origin cache
# ---------------------------------------------------------------------------

# Keyed by (image_height, image_width).
# tile_ox_all[t] = (t % tiles_x) * 32.0  (left edge of tile t)
# tile_oy_all[t] = (t // tiles_x) * 32.0  (top edge of tile t)
# These are constant for a given resolution and reused every frame.
_tile_origin_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_tile_origins(
    image_height: int, image_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached (tile_ox_all, tile_oy_all) of shape (num_tiles,) float32.

    Each element is the pixel-space origin of that tile along x or y.
    Cached per resolution so the division and float conversion only happen once.
    """
    key = (image_height, image_width)
    cached = _tile_origin_cache.get(key)
    if cached is not None:
        return cached
    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32
    num_tiles = tiles_x * tiles_y
    t_ids = np.arange(num_tiles, dtype=np.float32)
    tile_ox_all = (t_ids % tiles_x) * 32.0
    tile_oy_all = (t_ids // tiles_x) * 32.0
    _tile_origin_cache[key] = (tile_ox_all, tile_oy_all)
    return tile_ox_all, tile_oy_all


class KernelBackend(Backend):
    """Persistent IPC wrapper around the alpha-blend daemon subprocess.

    Spawn once at viewer/script start, call `blend(...)` per frame,
    `close()` on shutdown. After READY the protocol is binary on
    stdin/stdout (SCN1 scene init, FRM2 frame request → OK11/ERR1).

    Per-frame scratch buffers (self._attr_buf etc.) are preallocated on the
    first frame and grown lazily (with ~25% headroom) whenever the number of
    visible (Gaussian, tile) pairs exceeds the current capacity.  On
    steady-state interactive viewing at a fixed resolution the same buffers
    are reused every frame, eliminating the ~50 MB of per-frame malloc/free
    that previously dominated the blend.prep stage.
    """

    BINARY_PATH = "backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._scene_initialized = False
        self._scene_n_gaussians = 0
        self._scene_colors_id: int | None = None

        # ---- Preallocated per-frame scratch buffers ----
        # Grown lazily; capacity tracked separately from shape so we can
        # slice the active portion without a new allocation each frame.
        self._entry_cap: int = 0       # capacity in (Gaussian, tile) entries
        self._gauss_cap: int = 0       # capacity in visible Gaussian count
        self._tile_cap: int = 0        # capacity in number of tiles

        # Entry-indexed buffers (P = total (Gaussian, tile) pairs)
        self._attr_buf: np.ndarray | None = None        # (entry_cap, 5) float32
        self._tile_ox_buf: np.ndarray | None = None     # (entry_cap,)   float32
        self._tile_oy_buf: np.ndarray | None = None     # (entry_cap,)   float32
        self._gids_u32_buf: np.ndarray | None = None    # (entry_cap,)   uint32

        # Gaussian-indexed scratch (M = visible Gaussian count)
        self._cov_inv_a: np.ndarray | None = None       # (gauss_cap,)   float32
        self._cov_inv_b: np.ndarray | None = None       # (gauss_cap,)   float32
        self._cov_inv_c: np.ndarray | None = None       # (gauss_cap,)   float32
        self._det_buf: np.ndarray | None = None         # (gauss_cap,)   float32
        self._tmp_m_buf: np.ndarray | None = None       # (gauss_cap,)   float32

        # Tile-indexed (num_tiles + 1 slots for cumulative offsets)
        self._tile_offsets_buf: np.ndarray | None = None   # (tile_cap+1,) uint32
        self._offsets_f32_buf: np.ndarray | None = None    # (tile_cap+1,) float32

        # Image readback buffer — reused when resolution is unchanged.
        # Callers must NOT store a long-lived reference across frames;
        # the pipeline consumes (clips + converts to uint8) before the next
        # blend() call, so this is safe in the interactive/benchmark paths.
        self._image_buf: np.ndarray | None = None
        self._image_buf_shape: tuple[int, int] = (0, 0)

        # ---- Shared-memory IPC state (iter 024) ----
        self._use_shm: bool = os.environ.get("GSPLAT_TT_USE_SHM", "0") == "1"
        self._shm_in: SharedMemory | None = None        # frame data → daemon
        self._shm_out: SharedMemory | None = None       # rendered image ← daemon
        self._shm_max_entries: int = int(
            os.environ.get("GSPLAT_TT_SHM_MAX_ENTRIES", "2097152")
        )
        self._shm_max_tiles: int = int(
            os.environ.get("GSPLAT_TT_SHM_MAX_TILES", "1024")
        )
        # numpy views into shm_in data regions (set during SHM1 handshake)
        self._shm_hdr_view: np.ndarray | None = None    # (16,) uint32 header
        self._shm_attr_view: np.ndarray | None = None   # (max_entries, 5) float32
        self._shm_gids_view: np.ndarray | None = None   # (max_entries,) uint32
        self._shm_offs_view: np.ndarray | None = None   # (max_offsets,) float32
        self._shm_px_view: np.ndarray | None = None     # (max_px_floats,) float32
        self._shm_py_view: np.ndarray | None = None     # (max_px_floats,) float32
        self._shm_max_offsets: int = 0
        self._shm_max_px_floats: int = 0
        # numpy view into shm_out (max_pixels * 3 float32)
        self._shm_out_view: np.ndarray | None = None

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

        if self._use_shm:
            self._setup_shm()

    # ------------------------------------------------------------------
    # Shared-memory setup (iter 024)
    # ------------------------------------------------------------------

    def _setup_shm(self) -> None:
        """Create SHM segments and send SHM1 handshake to daemon.

        Input SHM layout (shm_in):
          [0..63]   : header  — 4×uint32: total_entries, offsets_count,
                                image_h, image_w  (bytes 16..63 reserved)
          [64..]    : attr    — (max_entries, 5) float32
                      gids    — (max_entries,)   uint32
                      offsets — (max_offsets,)   float32
                      px      — (max_px_floats,) float32
                      py      — (max_px_floats,) float32

        Output SHM layout (shm_out):
          [0..] : (max_pixels * 3) float32, row-major (set by daemon after kernel)
        """
        me = self._shm_max_entries
        mt = self._shm_max_tiles
        max_offsets = mt + 1
        max_px_floats = mt * 32 * 32

        # Byte offsets for each region within shm_in (after 64-byte header)
        attr_offset   = _SHM_HDR_BYTES
        gids_offset   = attr_offset   + me * 5 * 4
        offs_offset   = gids_offset   + me * 4
        px_offset     = offs_offset   + max_offsets * 4
        py_offset     = px_offset     + max_px_floats * 4
        shm_in_size   = py_offset     + max_px_floats * 4

        # Max pixels: assume square = mt tiles of 32×32 pixels
        max_pixels = mt * 32 * 32
        shm_out_size = max_pixels * 3 * 4  # fp32

        pid = os.getpid()
        shm_in_name  = f"gtt_in_{pid}"
        shm_out_name = f"gtt_out_{pid}"

        try:
            self._shm_in  = SharedMemory(name=shm_in_name,  create=True, size=shm_in_size)
            self._shm_out = SharedMemory(name=shm_out_name, create=True, size=shm_out_size)
        except Exception as exc:
            self._cleanup_shm()
            if self.verbose:
                print(f"[TTBackend] SHM creation failed ({exc}); falling back to pipe mode")
            self._use_shm = False
            return

        # Pre-build numpy views into the SHM regions (zero-copy per frame)
        buf_in = memoryview(self._shm_in.buf)
        self._shm_hdr_view  = np.frombuffer(buf_in[:_SHM_HDR_BYTES], dtype=np.uint32)
        self._shm_attr_view = np.frombuffer(
            buf_in[attr_offset : gids_offset], dtype=np.float32
        ).reshape(me, 5)
        self._shm_gids_view = np.frombuffer(
            buf_in[gids_offset : offs_offset], dtype=np.uint32
        )
        self._shm_offs_view = np.frombuffer(
            buf_in[offs_offset : px_offset], dtype=np.float32
        )
        self._shm_px_view   = np.frombuffer(
            buf_in[px_offset : py_offset], dtype=np.float32
        )
        self._shm_py_view   = np.frombuffer(
            buf_in[py_offset : shm_in_size], dtype=np.float32
        )
        self._shm_max_offsets   = max_offsets
        self._shm_max_px_floats = max_px_floats

        self._shm_out_view = np.frombuffer(
            memoryview(self._shm_out.buf)[:shm_out_size], dtype=np.float32
        )

        # SHM1 wire message: 256 bytes
        #   [0..3]   magic
        #   [4..7]   max_entries
        #   [8..11]  max_offsets
        #   [12..15] max_px_floats
        #   [16..79] shm_in_name (64 bytes, null-padded)
        #   [80..87] shm_in_size (uint64)
        #   [88..151] shm_out_name (64 bytes, null-padded)
        #   [152..159] shm_out_size (uint64)
        #   [160..255] reserved zeros
        name_in_bytes  = shm_in_name.encode()[:63].ljust(64, b"\x00")
        name_out_bytes = shm_out_name.encode()[:63].ljust(64, b"\x00")
        msg = struct.pack("<4I", _MAGIC_SHM1, me, max_offsets, max_px_floats)
        msg += name_in_bytes
        msg += struct.pack("<Q", shm_in_size)
        msg += name_out_bytes
        msg += struct.pack("<Q", shm_out_size)
        msg += b"\x00" * (256 - len(msg))

        try:
            self._proc.stdin.write(msg)
            self._proc.stdin.flush()
            resp = _read_exact(self._proc.stdout, 8)
            magic, _ = struct.unpack("<2I", resp)
            if magic != _MAGIC_OK31:
                raise RuntimeError(f"SHM1 daemon rejected: {magic:#010x}")
        except Exception as exc:
            self._cleanup_shm()
            if self.verbose:
                print(f"[TTBackend] SHM1 handshake failed ({exc}); falling back to pipe mode")
            self._use_shm = False

    def _cleanup_shm(self) -> None:
        for shm in (self._shm_in, self._shm_out):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        self._shm_in = self._shm_out = None
        self._shm_hdr_view = self._shm_attr_view = None
        self._shm_gids_view = self._shm_offs_view = None
        self._shm_px_view = self._shm_py_view = None
        self._shm_out_view = None

    # ------------------------------------------------------------------
    # Scratch-buffer lifecycle
    # ------------------------------------------------------------------

    def _ensure_bufs(
        self, total_entries: int, num_visible: int, num_tiles: int, H: int, W: int
    ) -> None:
        """Grow preallocated scratch buffers only when current capacity is exceeded.

        Each buffer is grown with ~25 % headroom to avoid repeated reallocation
        on gradually-increasing scenes.  On a stable interactive session (same
        scene, same resolution) this is called zero times after the first frame.
        """
        if total_entries > self._entry_cap:
            cap = total_entries + max(total_entries // 4, 65536)
            self._attr_buf = np.empty((cap, 5), dtype=np.float32)
            self._tile_ox_buf = np.empty(cap, dtype=np.float32)
            self._tile_oy_buf = np.empty(cap, dtype=np.float32)
            self._gids_u32_buf = np.empty(cap, dtype=np.uint32)
            self._entry_cap = cap

        if num_visible > self._gauss_cap:
            cap = num_visible + max(num_visible // 4, 16384)
            self._cov_inv_a = np.empty(cap, dtype=np.float32)
            self._cov_inv_b = np.empty(cap, dtype=np.float32)
            self._cov_inv_c = np.empty(cap, dtype=np.float32)
            self._det_buf = np.empty(cap, dtype=np.float32)
            self._tmp_m_buf = np.empty(cap, dtype=np.float32)
            self._gauss_cap = cap

        if num_tiles > self._tile_cap:
            cap = num_tiles + 64
            self._tile_offsets_buf = np.empty(cap + 1, dtype=np.uint32)
            self._offsets_f32_buf = np.empty(cap + 1, dtype=np.float32)
            self._tile_cap = cap

        if (H, W) != self._image_buf_shape:
            self._image_buf = np.empty((H, W, 3), dtype=np.float32)
            self._image_buf_shape = (H, W)

    # ------------------------------------------------------------------
    # Scene static attributes (SCN1)
    # ------------------------------------------------------------------

    def set_scene(self, colors: torch.Tensor, opacities: torch.Tensor) -> None:
        """Upload per-Gaussian (R, G, B, opacity) to daemon DRAM once."""
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise RuntimeError(
                f"set_scene expects colors shape (N_gaussians, 3), got {tuple(colors.shape)}"
            )
        if opacities.ndim != 1:
            raise RuntimeError(
                f"set_scene expects opacities shape (N_gaussians,), got {tuple(opacities.shape)}"
            )
        if colors.shape[0] != opacities.shape[0]:
            raise RuntimeError(
                f"colors/opacities length mismatch: {colors.shape[0]} vs {opacities.shape[0]}"
            )

        n = int(colors.shape[0])
        colors_id = id(colors)
        if (
            self._scene_initialized
            and self._scene_n_gaussians == n
            and self._scene_colors_id == colors_id
        ):
            return

        static = np.empty((n, 4), dtype=np.float32)
        static[:, :3] = np.ascontiguousarray(colors.detach().cpu().numpy(), dtype=np.float32)
        static[:, 3] = np.ascontiguousarray(opacities.detach().cpu().numpy(), dtype=np.float32)

        self._proc.stdin.write(struct.pack("<4I", _MAGIC_SCN1, n, 0, 0))
        self._proc.stdin.write(memoryview(static))
        self._proc.stdin.flush()

        resp = _read_exact(self._proc.stdout, 8)
        magic, _ = struct.unpack("<2I", resp)
        if magic != _MAGIC_OK31:
            raise RuntimeError(f"SCN1 failed: unexpected response magic {magic:#010x}")

        self._scene_initialized = True
        self._scene_n_gaussians = n
        self._scene_colors_id = colors_id

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

        if not self._scene_initialized:
            self.set_scene(colors, opacities)
        elif id(colors) != self._scene_colors_id:
            self.set_scene(colors, opacities)

        # ----------------------------------------------------------------
        # Sub-stage A: pack the 5-column dynamic-Gaussian attribute table.
        #
        # Uses preallocated scratch buffers (grown lazily) to avoid the
        # ~50 MB of per-frame malloc/free that previously dominated this
        # stage.  Key techniques:
        #   • np.take(..., out=col) — gathers into a preallocated column
        #     without creating a temporary array.
        #   • np.divide / np.multiply with out= — in-place covariance
        #     inversion, no intermediate PyTorch tensor allocations.
        #   • Python loop over num_tiles (≤1024) to fill tile origins —
        #     avoids np.repeat's 6 MB temporary allocation.
        # ----------------------------------------------------------------
        t_prep = time.perf_counter()

        M = int(means_2d.shape[0])   # visible Gaussian count
        gids_np = sorted_gaussian_ids.numpy()   # int64 view, no copy
        total_entries = int(gids_np.shape[0])   # total (Gaussian, tile) pairs
        tiles_x = (W + 31) // 32
        tiles_y = (H + 31) // 32
        num_tiles = tiles_x * tiles_y

        self._ensure_bufs(total_entries, M, num_tiles, H, W)

        # Active slices — views into preallocated buffers, no malloc.
        attr = self._attr_buf[:total_entries]           # (P, 5) float32
        tile_ox = self._tile_ox_buf[:total_entries]     # (P,) float32
        tile_oy = self._tile_oy_buf[:total_entries]     # (P,) float32
        gids_u32 = self._gids_u32_buf[:total_entries]  # (P,) uint32
        cov_inv_a = self._cov_inv_a[:M]                # (M,) float32
        cov_inv_b = self._cov_inv_b[:M]                # (M,) float32
        cov_inv_c = self._cov_inv_c[:M]                # (M,) float32
        det_buf = self._det_buf[:M]                    # (M,) float32
        tmp_m = self._tmp_m_buf[:M]                    # (M,) float32
        tile_offsets = self._tile_offsets_buf[:num_tiles + 1]   # (T+1,) uint32
        offsets_f32 = self._offsets_f32_buf[:num_tiles + 1]     # (T+1,) float32

        # Convert sorted gids to uint32 in-place (int64 values are ≤ M ≤ 2^24).
        np.copyto(gids_u32, gids_np, casting="unsafe")

        # Build cumulative tile offsets from tile_ranges (no new array).
        ranges_np = tile_ranges.numpy()
        counts_u32 = (ranges_np[:, 1] - ranges_np[:, 0]).astype(np.uint32)
        tile_offsets[0] = 0
        np.cumsum(counts_u32, out=tile_offsets[1:])

        # Covariance inversion — pure numpy into preallocated buffers,
        # avoiding PyTorch tensor creation for each intermediate.
        covs_np = covs_2d.numpy()  # (M, 2, 2), strides (16, 8, 4) bytes
        #   det = clamp(a*c - b*b, min=1e-6)
        np.multiply(covs_np[:, 0, 0], covs_np[:, 1, 1], out=det_buf)   # a*c
        np.multiply(covs_np[:, 0, 1], covs_np[:, 0, 1], out=tmp_m)     # b*b
        np.subtract(det_buf, tmp_m, out=det_buf)                        # a*c - b*b
        np.clip(det_buf, 1e-6, None, out=det_buf)
        #   cov_inv_a = c/det,  cov_inv_b = -b/det,  cov_inv_c = a/det
        np.divide(covs_np[:, 1, 1], det_buf, out=cov_inv_a)
        np.divide(covs_np[:, 0, 1], det_buf, out=cov_inv_b)
        # iter 026: fold the *2 from the Gaussian power expression
        #   power = a*dx² + 2b*dx*dy + c*dy²
        # into the M-sized precompute (~280k vs ~1.6M). Keeps the per-pair
        # gather to a pure copy (`attr[:, 3] = cov_inv_b[gids]`) without
        # the trailing scalar multiply.
        cov_inv_b *= -2.0
        np.divide(covs_np[:, 0, 0], det_buf, out=cov_inv_c)

        # Fill per-entry tile origins from the cached per-resolution arrays.
        # Python loop over ≤1024 tiles; each body is a vectorized slice fill,
        # costing ~0.3 ms total — cheaper than np.repeat's 6 MB alloc.
        tile_ox_all, tile_oy_all = _get_tile_origins(H, W)
        for t in range(num_tiles):
            s = int(tile_offsets[t])
            e = int(tile_offsets[t + 1])
            if e > s:
                tile_ox[s:e] = tile_ox_all[t]
                tile_oy[s:e] = tile_oy_all[t]

        # Gather Gaussian attributes into the packed table.
        # The attr buffer is preallocated so the column writes do not malloc.
        # Per-gather temporaries (6.4 MB each) are still created by arr[idx]
        # fancy-indexing — this is faster than np.take with non-contiguous
        # out= on every platform we've measured, because numpy's advanced-
        # indexing path is better optimised for contiguous output allocation.
        # On the Linux box the 5 × 6.4 MB gather temporaries use glibc's
        # mmap allocator (~0.5 ms each), which is unavoidable without a
        # numba/Cython gather primitive.  The large attr_buf malloc (32 MB)
        # and the cov_inv / tile-origin temporaries are fully eliminated.
        means_np = means_2d.numpy()  # (M, 2) float32 view
        attr[:, 0] = means_np[gids_np, 0] - tile_ox
        attr[:, 1] = means_np[gids_np, 1] - tile_oy
        attr[:, 2] = cov_inv_a[gids_np]
        attr[:, 3] = cov_inv_b[gids_np]   # iter 026: 2× already folded in
        attr[:, 4] = cov_inv_c[gids_np]

        # px/py tile-local pixel grids — cached per resolution.
        px_tiles, py_tiles = _get_px_py_grids(H, W)

        # Offsets as float32 for the FRM2 wire format (daemon reads as float).
        offsets_f32[:] = tile_offsets   # uint32 → float32, into preallocated buffer

        prep_ms = (time.perf_counter() - t_prep) * 1000.0

        # ----------------------------------------------------------------
        # Sub-stage B: write binary FRM2 frame to daemon stdin.
        #
        # Two paths:
        #   SHM path  (GSPLAT_TT_USE_SHM=1): write frame data to shm_in,
        #             then send a 24-byte FRM2 header only (shm_flag=1).
        #             Eliminates the ~46 MB stdin write (~17 ms → ~0.1 ms).
        #   Pipe path (default): write full payload to stdin as before.
        # ----------------------------------------------------------------
        t_save = time.perf_counter()
        offsets_count = int(offsets_f32.shape[0])

        shm_ok = (
            self._use_shm
            and self._shm_in is not None
            and total_entries <= self._shm_max_entries
            and offsets_count <= self._shm_max_offsets
            and num_tiles <= self._shm_max_tiles
        )

        if shm_ok:
            # Write header into shm_in (4 × uint32 at byte offset 0)
            self._shm_hdr_view[0] = total_entries
            self._shm_hdr_view[1] = offsets_count
            self._shm_hdr_view[2] = H
            self._shm_hdr_view[3] = W

            # Copy frame data into shm_in regions (numpy → SHM, ~memcpy speed)
            # attr is (total_entries, 5) float32
            np.copyto(self._shm_attr_view[:total_entries], attr[:total_entries])
            # gids is (total_entries,) uint32
            np.copyto(self._shm_gids_view[:total_entries], gids_u32[:total_entries])
            # offsets is (offsets_count,) float32
            np.copyto(self._shm_offs_view[:offsets_count], offsets_f32[:offsets_count])
            # px / py are (num_tiles, 32, 32) float32 — flatten to 1-D view
            px_flat = px_tiles.reshape(-1)
            py_flat = py_tiles.reshape(-1)
            px_n = int(px_flat.shape[0])
            np.copyto(self._shm_px_view[:px_n], px_flat)
            np.copyto(self._shm_py_view[:px_n], py_flat)

            # Send FRM2 header only (shm_flag=1, no payload)
            self._proc.stdin.write(struct.pack(
                "<6I",
                _MAGIC_FRM2,
                H,
                W,
                total_entries,
                offsets_count,
                1,   # shm_flag
            ))
            self._proc.stdin.flush()
        else:
            # Pipe fallback: full payload
            if self._use_shm and self.verbose:
                print(
                    f"[TTBackend] SHM fallback: entries={total_entries} "
                    f"max={self._shm_max_entries}"
                )
            self._proc.stdin.write(struct.pack(
                "<6I",
                _MAGIC_FRM2,
                H,
                W,
                total_entries,
                offsets_count,
                0,
            ))
            self._proc.stdin.write(memoryview(attr))          # (P, 5) float32
            self._proc.stdin.write(memoryview(gids_u32))      # (P,)   uint32
            self._proc.stdin.write(memoryview(offsets_f32))   # (T+1,) float32
            self._proc.stdin.write(memoryview(px_tiles))      # (T, 32, 32) float32
            self._proc.stdin.write(memoryview(py_tiles))      # (T, 32, 32) float32
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

        # Sub-stage D: read rendered image.
        # SHM path: image is already in shm_out (daemon wrote it before OK11).
        #           Copy into preallocated _image_buf; ~memcpy speed (~0.1 ms).
        # Pipe path: read from stdout as before (~4.7 ms for 12 MB at 1024×1024).
        t_load = time.perf_counter()
        if shm_ok:
            # shm_out holds (H*W*3) float32 written by the daemon
            n = H * W * 3
            np.copyto(
                self._image_buf.reshape(-1),
                self._shm_out_view[:n],
            )
        else:
            _read_exact_into(self._proc.stdout, memoryview(self._image_buf).cast("B"))
        load_ms = (time.perf_counter() - t_load) * 1000.0

        sub_timings: dict[str, float] = {
            "prep": prep_ms,
            "save_npy": save_ms,
            "daemon_rt": rt_ms,
        }
        if kernel_ms is not None:
            sub_timings["daemon_rt.device_kernel"] = kernel_ms
        sub_timings["load_npy"] = load_ms

        return self._image_buf, sub_timings

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
        self._cleanup_shm()
