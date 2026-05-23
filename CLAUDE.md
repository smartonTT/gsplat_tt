# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

MSc thesis project: forward-pass rasterization of 3D Gaussian Splatting (3DGS)
on Tenstorrent Wormhole/Blackhole hardware via custom tt-metal kernels.

**Scope:** inference / rendering only — load a pre-trained `.ply` and render
it. No training, backward pass, or differentiable rasterization.

## Repository layout

```
gsplat_tt/
├── pyproject.toml                       # `pip install -e .` makes `gsplat` importable
├── setup.sh                             # one-shot bootstrap (venv, vendor tt-metal, build)
├── README.md, CLAUDE.md, requirements.txt, conftest.py, .gitignore
├── docs/
│   ├── thesis_plan.md                   # private working doc (gitignored)
│   └── plan_progress.md                 # design decisions + KI-1/KI-2 history
├── scenes/                              # .ply scenes (only luigi.ply tracked)
├── tests/                               # pytest suite
├── scripts/                             # one-off dev helpers
├── gsplat/                              # importable Python package (CPU pipeline + viewer)
│   ├── __main__.py                      # CLI entry — `python -m gsplat ...` or `gsplat ...`
│   ├── rasterization.py                 # project, tile, sort, alpha_blend, prepare_kernel_inputs
│   ├── viewer.py                        # interactive viewer (viser + nerfview)
│   ├── data_structures.py               # Gaussians dataclass
│   ├── loading_gaussians.py             # .ply loader
│   └── utils.py                         # camera helpers (c2w↔w2c, build_covariance_3d)
└── backends/                            # one subdir per hardware target
    ├── README.md                        # how to add a backend
    ├── tt/                              # Tenstorrent (tt-metal) backend
    │   ├── backend.py                   # daemon-subprocess wrapper
    │   └── tt-metal/                    # vendored SDK + our C++ kernels under
    │       └── tt_metal/programming_examples/gaussian_splatting/
    └── cuda/                            # placeholder for future CUDA backend
```

`gsplat/` is the host-side Python; `backends/<arch>/` holds everything that
talks to a specific accelerator. Adding CUDA later means dropping a
`backends/cuda/backend.py` exposing the same `render(...) / close()`
interface as `backends/tt/backend.py`.

## Pipeline stages

```
load_ply → project_gaussians → get_tile_assignments → sort_and_bin
                                                         │
                                                         ↓
                              ┌──────────────────────────┴──────────────────────────┐
                       (CPU)  alpha_blend                  prepare_kernel_inputs (TT)
                                                                  │
                                                                  ↓
                                                      backends.tt.backend.KernelBackend
                                                                  ↓
                                                          tt-metal kernels:
                                                              reader (NCRISC) →
                                                              compute (TRISCs) →
                                                              writer (BRISC)
```

The kernel side splits into 3 RISC kernels per Tensix core:
- **Reader** (NCRISC, NoC1): DRAM → L1 via circular buffers.
- **Compute** (TRISC0/1/2): SFPU/FPU work, alpha-blend math.
- **Writer** (BRISC, NoC0): L1 → DRAM, packs RGB tiles to the output buffer.

Key tt-metal constants we use:
- `bfloat16` for on-device storage; `fp32_dest_acc_en=true` keeps Dst register accumulation in fp32.
- `HiFi3` (HiFi4 has WH B0 bug #38306).
- 32×32 native tile, 4 SFPU passes per tile (32 lanes wide).
- Wormhole N150 logical grid: 8×8 cores after 1 row harvested.

## Conventions

- 32×32 screen tiles (matches the hardware tile).
- Structure-of-Arrays for device DRAM (separate `packs`, `px`, `py`, `offsets`, `tile_ids` buffers).
- SH degree 0 (3 color floats per Gaussian); higher degrees not implemented.
- Validate the kernel against the CPU reference via PSNR/SSIM (≥35 dB target).
- Render at 480-960px range for interactive use; 4K is the design ceiling.

## Project venv

The host-side Python (viewer, CPU pipeline, tests, kernel-backend wrapper)
lives in `./venv`:
- Activate: `source venv/bin/activate`
- Used by: `gsplat ...`, `pytest`, anything in `gsplat/` or `tests/`.

Note: tt-metal's `ttnn` Python bindings (and the separate `python_env` they
need) are intentionally **not built** by setup. Our runtime only invokes
the C++ binary as a subprocess, so we pass `--without-python-bindings` to
`build_metal.sh` and skip `create_venv.sh`. If you ever need `ttnn`
directly, run tt-metal's `create_venv.sh` manually inside
`backends/tt/tt-metal/`.

## Setup

`./setup.sh` is the single canonical bootstrap. It is idempotent. Steps:

1. Creates `./venv`, installs `requirements.txt`, runs `pip install -e .`.
2. Vendors `tenstorrent/tt-metal` into `backends/tt/tt-metal/` (~5 GB; the
   target dir already contains our tracked kernel subdir, so the script
   clones to a temp location and merges with `cp -rn`).
3. Adds `add_subdirectory(gaussian_splatting)` to tt-metal's
   `programming_examples/CMakeLists.txt`.
4. `sudo ./build_metal.sh --build-programming-examples --without-python-bindings`
   to compile the C++ libs + our kernel host binary. `sudo` needed for
   tt-metal's root-owned SFPI / CPM caches.

Pin a tt-metal version with `TT_METAL_REF=v1.2.3 ./setup.sh`.

To rebuild only the kernel host binary after editing `alpha_blend.cpp` /
`alpha_blend_compute.cpp` / kernel sources:

```bash
sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting
```

(The `.cpp` kernel sources in `kernels/` are JIT-compiled at runtime by
tt-metal — only the host binary needs CMake rebuild.)

## Running

```bash
# CPU viewer
source venv/bin/activate
gsplat scenes/luigi.ply

# TT viewer (Tenstorrent)
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal
gsplat scenes/bonsai_room.ply --backend tt --max-resolution 960
```

`gsplat` is registered as a console_script in `pyproject.toml`. Equivalent:
`python -m gsplat scenes/luigi.ply ...`.

## Tests

```bash
source venv/bin/activate
pytest tests/
```

Three integration tests in `tests/test_kernel_integration.py`:
- `test_full_scene_psnr` (64×64, 50 random Gaussians) — PSNR target ≥35 dB
- `test_640_perf_baseline` (one-shot kernel binary, 640×640, 10K Gaussians)
- `test_640_perf_daemon` (5-frame daemon-mode benchmark)

Plus `tests/test_numeric_sanity.py` — pure-Python alpha-blend reference checks.

## Resolved known issues (kept here for context)

- **KI-1: T0.6 saturation** at 50 stacked α=0.99 Gaussians. Edges saturate
  to bf16 0x4790 (= 73728). Synthetic worst-case; real scenes hit
  41-52 dB PSNR. Deferred to v2 if ever needed. See `docs/plan_progress.md`.
- **KI-2: multi-tile dispatch deadlock** at ≥16 active cores with sparse
  tile occupancy. Was a CB deadlock from empty-tile churn. Fixed by
  filtering empty tiles in `compute_lpt_assignment` + pre-zeroing the
  output buffer (`process_frame`). See `docs/plan_progress.md::KI-2`.

## Reference implementations (for cross-checking)

- `hbb1/torch-splatting` — pure PyTorch reference for the CPU path.
- `gsplat` (nerfstudio) — production CUDA + PyTorch fallback.
- `antimatter15/splat` — ~300-line WebGL.
- `graphdeco-inria/diff-gaussian-rasterization` — original CUDA rasterizer.

## IRD box workflow

Edit on Mac, `devsync` mirrors to the box. See `~/dev/README.md` for the generic devsync/IRD workflow.

**Launch on the box (P300 Blackhole, e.g. `yyzo-bh-14`):**

```bash
cd /proj_sw/user_dev/smarton/gsplat_tt
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
export TT_METAL_LOGGER_LEVEL=warning
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
mkdir -p /localdev/smarton/.cache/tt-metal-cache
gsplat scenes/stitch_doll.ply --backend tt --force-square 1024
```

`TT_MESH_GRAPH_DESC_PATH` is required on P300; omitting it causes `TT_FATAL: Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`.

**Rebuild after editing kernel or host `.cpp`:**

```bash
sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting
```

Then wipe the JIT cache: `rm -rf /localdev/smarton/.cache/tt-metal-cache/`

**The viewer script** at `/tmp/start_viewer.sh` on the box sets all env vars, kills any prior daemon, and launches `gsplat scenes/stitch_doll.ply --backend tt --port 8080 --force-square 1024`. Recreate it if lost:

```bash
cat > /tmp/start_viewer.sh << 'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /proj_sw/user_dev/smarton/gsplat_tt
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
export TT_METAL_LOGGER_LEVEL=warning
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
unset GSPLAT_TT_MAX_G_PER_TILE GSPLAT_TT_CULL_EPS
pkill -TERM -f "scenes/stitch_doll" 2>/dev/null || true
pkill -TERM -f "metal_example_gaussian" 2>/dev/null || true
sleep 8
nohup gsplat scenes/stitch_doll.ply --backend tt --port 8080 --force-square 1024 --verbose \
  > /localdev/smarton/viewer_logs/viewer_$(date +%Y%m%d-%H%M%S).log 2>&1 &
echo "viewer PID: $!"
EOF
chmod +x /tmp/start_viewer.sh
```

**Mac-side tunnel:**

```bash
ssh -f -N -L 8080:127.0.0.1:8080 bh-30   # stable viewer on bh-30 (always live)
# ssh -f -N -L 8080:127.0.0.1:8080 yyzo-bh-14  # dev box (only during testing)
curl -sI http://127.0.0.1:8080/   # should return HTTP 200
```

## Two-box workflow (optimization)

- **bh-14 (yyzo-bh-14, Toronto P300)** — development and benchmarking. All rebuilds happen here.
- **bh-30 (Austin P150)** — stable viewer. Always running last KEEP binary; never rebuilt during development.

Both boxes share the same Weka filesystem at `/proj_sw/user_dev/smarton/`. The stable viewer binary lives outside the git working tree so devsync updates never clobber it:

```
/proj_sw/user_dev/smarton/stable_viewer/
  metal_example_gaussian_splatting_iter068   ← stable binary (updated on each KEEP)
  k/gaussian_splatting/kernels/              ← stable JIT kernel sources (frozen at last KEEP)
```

**Starting the stable viewer on bh-30:**
```bash
bash /tmp/start_stable_viewer.sh
# Or, if the script was lost, recreate from scripts/start_stable_viewer.sh in the repo.
```

**Env vars for stable viewer** (all handled by the script):
- `GSPLAT_TT_BINARY` — points to stable binary
- `GSPLAT_TT_KERNEL_PREFIX` — points to stable frozen kernel sources
- `TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache-stable` — separate JIT cache

**Promoting a new KEEP to stable viewer** (run on bh-14 after confirming PSNR ≥ 35 dB):
```bash
# 1. Save updated stable binary
cp backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting \
   /proj_sw/user_dev/smarton/stable_viewer/metal_example_gaussian_splatting_iter068
# 2. Update frozen kernel sources
cp -r backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/. \
   /proj_sw/user_dev/smarton/stable_viewer/k/gaussian_splatting/kernels/
# 3. Clear stable JIT cache (forces recompile with new kernels on bh-30)
ssh bh-30 "rm -rf /localdev/smarton/.cache/tt-metal-cache-stable/"
# 4. Restart stable viewer on bh-30
ssh bh-30 "bash /tmp/start_stable_viewer.sh"
```

A phase is only "done" when both the headless harness (`scripts/render_fixed.py`) AND a live viewer frame (`[wall] N ms` in the log after a browser connects) confirm the new binary is working.

## Benchmarking

```bash
cd /proj_sw/user_dev/smarton/gsplat_tt
source venv/bin/activate
# ... set env vars (see above) ...
python scripts/render_fixed.py stitch hero --backend tt --warmup 5 --frames 30 \
    --out /tmp/iter_NNN.png --json
```

The `--json` flag emits one line with all timings. Key fields: `daemon_rt.device_kernel` (kernel ms), `timings.blend` (total blend including IPC overhead).

Compare to reference: `benchmarks/reference/stitch_hero_480x640.png` (480×640) or `stitch_hero_1024x1024.png` (1024×1024).

Compute PSNR:

```python
from PIL import Image
import numpy as np, math
ref = np.array(Image.open('benchmarks/reference/stitch_hero_480x640.png').convert('RGB'), dtype=float)
cur = np.array(Image.open('/tmp/iter_NNN.png').convert('RGB'), dtype=float)
mse = np.mean((ref - cur)**2)
print(f'PSNR: {20 * math.log10(255.0 / math.sqrt(mse)):.2f} dB')
```

**Quality gate:** PSNR ≥ 35 dB = KEEP. 30–35 dB = NEEDS_REVIEW. < 30 dB = NO/revert.

## Optimization branch

Current optimization work is on `smarton/opt-stable`. Per-iteration results are in `docs/optimization-log/REPORT.md`. The target is ≤1 ms for the alpha-blend metal kernel (`daemon_rt.device_kernel`) at 1024×1024 with PSNR ≥ 35 dB.
