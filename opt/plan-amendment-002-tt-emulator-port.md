# Plan Amendment 002 — TT-as-Emulator Step-Wise Port

**Date:** 2026-05-27
**Supersedes:** `opt/plan-amendment-001-metal-port.md` (daemon + `.npy` IPC + legacy full-tile kernel approach).
**Status:** APPROVED ARCHITECTURE — execution gated on user go-ahead.

---

## 0. Why this exists (the failure of amendment 001)

Amendment 001 extended the pre-existing `gsplat_tt` daemon (`alpha_blend.cpp`
host driver + `alpha_blend_compute.cpp` device kernel). That scaffolding has a
fundamentally different design from `cpu_cpp`:

| | `cpu_cpp` (emulator, correct) | Current TT path (wrong shape) |
|--|------------------------------|-------------------------------|
| Outer loop | Per-tile parallel via `ThreadPool` (one tile per "core") | Per-tile parallel on Tensix cores |
| Blend loop | Outer 32 microblocks → inner `mb_stream[off:off+cnt]` | Outer per-Gaussian → inner full 32×32 tile |
| Cull | Fused into blend via `cull_and_blend` (drops ~85% of pair-iters) | Host Python `microblock_cull`; **device ignores `mb_stream`** |
| Data buffers | `mb_header`, `mb_stream`, `coeff_table` (A..F basis) | Same payload uploaded; never read by kernel |
| IPC | In-process pybind, zero-copy numpy ↔ C++ | Python → `np.save` → subprocess stdin → daemon → `np.load` |
| Behavior on toggle | Bit-identical pipeline; only blend impl swapped | Different algorithm; different cull; different loop order |

The whole point of `cpu_cpp` is that it **is the renderer-emulator** — one
thread per Tensix core, microblock-major loop, the same `mb_header` /
`mb_stream` / `coeff_table` data layouts a TT kernel will read. The TT port
should be a kernel-by-kernel **drop-in replacement** inside that same C++
pipeline, with PSNR invariance as the contract at every step.

Amendment 001's daemon path is being retired (the existing kernel sources stay
in `gsplat_tt` as a reference; nothing in `gstt2` will continue to depend
on the daemon, the `.npy` IPC, or `prepare_microblock_payload` in Python).

---

## 1. Target architecture

```
┌──────────────────── Python (orchestration only) ────────────────────┐
│ gsplat.pipeline.Pipeline(backend)                                   │
│   backend.project / tile_assign / sort / blend  ← thin pybind calls │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │  _gsplat_cpu pybind extension (C++)         │
            │  ─────────────────────────────────          │
            │  src/gsplat_cpu/  (project, tile_assign,    │
            │     sort, microblock_cull, cull_and_blend,  │
            │     blend_microblock, thread_pool)          │
            │                                             │
            │  src/gsplat_tt/   (NEW — same ABI as        │
            │     gsplat_cpu equivalents; each function   │
            │     calls into tt-metal host API)           │
            └──────────────────────────────────────────────┘
                                   │
                                   │ tt-metal host API
                                   ▼
                       ┌───────────────────────┐
                       │ tt-metal device       │
                       │ (kernels in           │
                       │  src/gsplat_tt/kernels)│
                       └───────────────────────┘
```

Single pybind extension. The TT backend in Python is a 50-line file that
constructs `CpuCppMbBackend` and overrides one stage at a time as `gsplat_tt`
gains kernels. No daemon. No subprocess. No `.npy`. No Python prep
duplication.

**Why this works:**
- `tt-metal` exposes a C++ host API (`tt-metalium/host_api.hpp`,
  `distributed::MeshBuffer`, `EnqueueWriteMeshBuffer`, etc.). The pybind
  extension links it directly.
- Buffers (`means_2d`, `covs_2d`, `mb_header`, `mb_stream`, `coeff_table`,
  `cov_inv_*`) are already in the right SoA layout — `cpu_cpp` built them.
- Device init (open device, JIT-compile kernels) happens once at module
  load; subsequent calls reuse the live `IDevice*` / `Program`.

---

## 2. Iteration ladder (PSNR-gated)

Every iter must pass:
- **Hero PSNR ≥ reference cpu_cpp_mb hero PSNR − 0.5 dB**
- **30-view min PSNR ≥ 60 dB** (north star invariant)
- **`microblock_cull` drop counts identical** (when culling moves to device)

Reference for all comparisons: `cpu_cpp_mb` backend on the same fixtures.

| iter | what moves to TT | scope choice | first-pass blend granularity | gate |
|------|------------------|--------------|------------------------------|------|
| **tt-000** | none (infra) | Link tt-metal into `_gsplat_cpu` pybind. Add `--backend tt` Python skeleton that constructs `CpuCppMbBackend` (no kernels yet). Smoke: `cpu_cpp_mb` PSNR reproduced through the new module. | n/a | bit-identical to `cpu_cpp_mb` |
| **tt-001a** | `blend` only — **32×32 full-tile fallback** | Read `mb_header` + `mb_stream` from DRAM but blend each kept Gaussian against the full 32×32 tile (skip per-microblock pixel masking). Identical math to today's daemon kernel but driven from in-process pipeline + sharing `cpu_cpp`'s buffers. Goal: prove the wiring, not the perf. | 32×32 | hero ≥ cpu_cpp_mb − 0.5 dB; 30-view ≥ 60 dB; `mb_stream` traversal verified by counting Gaussians touched per tile |
| **tt-001b** | `blend` — **microblock-major 4×8** | Outer loop over 32 microblocks, inner over `mb_stream[off:off+cnt]`, 4×8 SFPU inner kernel using A..F basis rows from `coeff_table`. Mirror `blend_microblock_tile`'s structure verbatim. | 4×8 | same PSNR gate as 001a; perf budget: device kernel ≤ 2× `cpu_cpp` blend (then optimize) |
| **tt-002** | `microblock_cull` | Port `cull_tile` from `src/gsplat_cpu/microblock_cull.cpp`. Drop counts MUST match `cpu_cpp_mb` to the integer. | n/a | bit-identical drop stats; PSNR unchanged from tt-001b |
| **tt-003** | `sort` | Port `src/gsplat_cpu/sort.cpp`. Often not worth offloading — measure first; keep on CPU if device sort is slower. | n/a | bit-identical `tile_ranges`; PSNR unchanged |
| **tt-004** | `tile_assign` (+ per-pair Mahalanobis cull) | Port `src/gsplat_cpu/tile_assign.cpp` phase 4 cull. | n/a | bit-identical kept-pair counts; PSNR unchanged |
| **tt-005** | `project` | Port `src/gsplat_cpu/project.cpp`. Last because per-Gaussian projection is the smallest CPU cost today. | n/a | bit-identical `means_2d` / `covs_2d` / `radii` (float-tolerant); PSNR unchanged |
| **tt-006+** | perf | SFPU/bf16 microoptimization, replay buffers, LPT dispatch, frame coherency, persistent DRAM buffers. Allowed to drop PSNR ≤ 0.5 dB if perf gain justifies; never below 60 dB on 30-view. | varies | sum30 budget per iter |

**Key rule:** every iter is a behavior-preserving swap of ONE function in the
C++ pipeline. The previous iter's output stays the reference until the next
iter green-lights.

---

## 3. Concrete first-step plan (tt-001a, 32×32 fallback)

This is the smallest viable on-device iter; written out so it's executable.

### 3.1 Code layout

New files:
```
src/gsplat_tt/
  device.h                  — singleton IDevice* / Program* lazy init
  device.cpp
  blend.h                   — blend_microblock_tt() ABI (matches cpu blend_microblock)
  blend.cpp                 — host driver: alloc DRAM, upload, dispatch, readback
  kernels/
    reader_blend.cpp        — reads coeff_table[g], px/py for tile, scalars
    compute_blend.cpp       — 32×32 full-tile blend, walks mb_stream
    writer_blend.cpp        — writes RGB tiles back
src/CMakeLists.txt          — add gsplat_tt static lib; link tt-metal
backends/cpu_cpp/pybind_module.cpp  — expose blend_microblock_tt
backends/tt/backend.py      — TtBackend(CpuCppMbBackend) with .blend override
```

Delete (or move under `archive/`):
```
gstt2/backends/tt/tt-metal/...         (already a symlink we own on bh-30)
gsplat_tt/.../alpha_blend.cpp (daemon)        ← keep in git history; stop building
gsplat_tt/.../alpha_blend_compute.cpp         ← reference only
gstt2/gsplat/rasterization.py::prepare_microblock_payload  ← unused
gstt2/backends/tt/backend.py            ← replace entirely
gstt2/scripts/metal_supervisor_*.sh     ← decommission
opt/metal-supervisor-state.json               ← decommission
prompts/metal-iter-000-*, 001-*, 002-*, 003-*, 004-*, 005-*, metal-supervisor.md  ← deprecate
opt/plan-amendment-001-metal-port.md          ← already superseded
```

### 3.2 Build

The `_gsplat_cpu` pybind extension needs to link tt-metal **conditionally**
(Mac dev box has no tt-metal). Add `GSPLAT_WITH_TT` CMake option:
- Mac: `cmake -DGSPLAT_WITH_TT=OFF` (default off, current behavior preserved).
- bh-30: `cmake -DGSPLAT_WITH_TT=ON -DTT_METAL_HOME=/localdev/smarton/tt-metal`.

`backends/tt/__init__.py` does `import _gsplat_cpu; assert _gsplat_cpu.has_tt_support()` and registers `TtBackend` only when present, matching the existing CUDA-optional pattern in `backends/__init__.py`.

### 3.3 Device kernel (32×32 fallback)

```
runtime_args = [num_tiles_for_this_core, dram_addr_mb_header, dram_addr_mb_stream,
                dram_addr_coeff_table, dram_addr_px, dram_addr_py, dram_addr_image_out]

for tile in core's tile slice:
    init R = G = B = 0, T = 1 per pixel (whole 32×32)
    for m in 0..31:
        (off, cnt) = mb_header[tile, m]
        for k in off..off+cnt:
            g = mb_stream[k]               ← LOCAL g-id within tile
            load 9 fp32 scalars from coeff_table[g]
            for each of 32×32 pixels:      ← FULL TILE (the simplification)
                Mahalanobis -> alpha -> R/G/B/T update
        # NOTE: 32×32 means we blend g into pixels OUTSIDE microblock m too.
        # That's the SAME math as today's daemon kernel — produces SAME PSNR
        # as cpu_cpp_mb because the per-microblock cull only DROPS g's that
        # contribute < 1/16384 anywhere in the microblock; including those
        # g's in OTHER microblocks of the same tile is what cpu_cpp already
        # does for the surviving (g, tile) pair before per-mb decomposition.
    write tile to image_out
```

Correctness argument: per-microblock cull is a **proper subset** of per-tile
cull. Every `g` in `mb_stream` was already in the tile's sorted-Gaussian list
(it survived the per-pair Mahalanobis cull). Blending it against the full
32×32 tile reproduces the per-tile-only behavior of an earlier `cpu_cpp` iter
(pre iter-008). Expected hero PSNR ≈ 47.78 dB — same as today's daemon kernel,
because that's what today's daemon kernel already does, just without the
.npy/subprocess overhead.

### 3.4 Perf expectation (not the gate)

- tt-001a: device_kernel ≈ today's 216 ms (same kernel structure, fewer host hops).
- IPC overhead drops from ~530 ms → expect ~250 ms total (just kernel + DRAM).
- tt-001b (4×8 microblock): device_kernel target ≤ 60 ms (cpu_cpp blend × 2),
  then optimize.

### 3.5 Verification

```
# Mac: rebuild cpu_cpp (no TT), confirm cpu_cpp_mb hero PSNR unchanged.
# bh-30: rebuild with -DGSPLAT_WITH_TT=ON, run:
python3 scripts/verify_blend_metal.py --backend tt --psnr-floor 47.7
python3 scripts/render_fixed.py --backend tt --scene bicycle --frames 30
# Compare to:
python3 scripts/render_fixed.py --backend cpu_cpp_mb --scene bicycle --frames 30
```

Gates:
- Hero TT PSNR ≥ 47.28 dB (cpu_cpp_mb − 0.5 dB).
- 30-view TT min PSNR ≥ 60 dB.
- `mb_stream` traversal: device kernel emits a per-tile Gaussian-count
  side-channel; assert sum equals `mb_stream.size`.

Log iter to `opt/metal-iters.jsonl` (single source of truth — keep amendment
001's log format; supersede `opt/metal-supervisor-state.json`).

---

## 4. Workflow changes

- **No daemon.** `backends/tt/backend.py` constructs the device via pybind
  call on first `blend()`. Device close on Python interpreter shutdown.
- **No `.npy`.** Buffers cross the pybind boundary as zero-copy
  `py::array_t<float>` / `py::array_t<int64_t>`.
- **No supervisor loop.** This work is small enough to do interactively; the
  prior 10-minute autonomous tick loop is decommissioned. Iteration is:
  edit on Mac → `devsync` to bh-30 → `cmake --build` → run verify → log
  result → next iter.
- **`cpu_cpp_mb` is the reference at every step.** A failing PSNR gate means
  the TT port has a bug, not that we lower the gate.

---

## 5. What stays from amendment 001

- `opt/metal-iters.jsonl` (continue logging here).
- `scripts/verify_blend_metal.py` (still the layer-2 gate, just driven by a
  different backend).
- bh-30 SSH/build setup (`scripts/setup_bh30_metal.sh`).
- `opt/microblock-cpu-spec.md` §0–§3 (algorithm spec; bit-portable).
- Hero/30-view fixtures in `tests/fixtures/hero/` and `benchmarks/reference_v2/`.

---

## 6. Halt / done

- **tt-001a done** when `tt` backend reproduces today's 47.78 dB hero PSNR
  through the in-process pipeline (no daemon).
- **tt-001b done** when hero PSNR ≥ `cpu_cpp_mb` − 0.5 dB AND device kernel
  ≤ 2× `cpu_cpp` blend ms.
- **Full port done** when all 5 stages run on device and 30-view sum_total_ms
  < 30 ms on bh-30 (P150 Blackhole).
- **Halt early on:** PSNR gate fail that can't be diagnosed in one iter,
  unrecoverable device failure, user stop.
