# render/ — clean extract of the production TT gsplat renderer

A standalone, readable copy of **only** the current production Tenstorrent
render pipeline. It faithfully reproduces the active path; every dead branch,
alternate path, debug toggle and CPU fallback has been removed, and the ~40
runtime `GSPLAT_TT_*` env flags are baked into compile-time constants
(`host/config.h`, `host/env_config.h`).

The renderer is **environment-independent**: it always runs the single
production configuration regardless of env. Correctness anchor (bicycle hero):

```
hero_vs_ref = 63.85 dB
```

(verified bit-identical with and without any `GSPLAT_TT_*` flags set).

> This directory is **additive**. It does not modify `src/`; the original
> pipeline stays as the reference.

---

## Pipeline (5 stages, host-free / resident)

The host (`host/render.cpp` → `render_view`) only orchestrates; all compute runs
on the device and buffers stay **resident** in DRAM/L1 between stages (no
inter-stage H2D/D2H). Each stage is one `*_device.cpp` driver + its kernels.

| # | Stage | Host driver | Kernels | Key outputs (resident) |
|---|-------|-------------|---------|------------------------|
| 1a | means → cam | `project_device.cpp` | `reader/writer_project_means_cam` + `project_means_cam_compute` | `means_cam_x/y/z` |
| 1b | pfwc (project + conic) | `pfwc_device.cpp` | `reader/writer_pfwc` + `pfwc_compute` | `pfwc_m2x/m2y/depth/a/b/c/rx/ry` |
| 1c | gather visible | `gather_visible_device.cpp` | `gather_visible_scatter` (+`GATHER_EMIT_BLENDREC`) + `gather_scan_bases` | `proj_m_*` + `proj_m_blendrec` + `proj_M` |
| 2 | tile assign | `tile_assign_device.cpp` | `tile_assign_bbox` + scan trio (`scan_reduce/scan_bases/scan_add`) + `tile_assign_scatter` | `ta_pairs_gid/tid/keep`, `ta_pairs_P` |
| 3 | sort + bin + publish + dir + mat | `sort_device.cpp` | `sort_bin` + `sort_radix_tile` + `sort_publish` + `sort_subchunk_dir` + `sort_subchunk_materialize` | `sort_sorted_ids`, `sort_tile_ranges`, `sort_subchunk_dir`, `sort_subchunk_payload`, `cull_mask_base` |
| 4 | tile L1 cull | `blend_device.cpp` (`tile_l1_cull::process_frame`) | `reader_tile_l1_cull` + `microblock_cull_compute` + `writer_tile_l1_mask` | per-tile L1 cull masks |
| 5 | blend + writeout | `blend_device.cpp` (`blend_mb_devcull_resident`) | `reader_alpha_blend_mb_devcull` + `alpha_blend_compute_mb` + `writer_alpha_blend` | `res_out` (bf16) → image |

Notes:
- **Sort layout is on the host** (`host_bin_layout_from_hist` + H2D); the device
  bin-layout kernel was dropped.
- **Stages 3–5 share one command queue** (`SortBlendContinuation`): after radix
  publish + subchunk directory, the host runs **tile L1 cull → blend** on the
  sort CQ; **subchunk materialize** is enqueued after blend returns (blend still
  reads `sort_sorted_ids` + `proj_m_blendrec` until step C wires payload-only).
- Final D2H reads `res_out` (bf16) and `tiles_to_image_mb` assembles the image.

### Baked configuration

`config.h` documents the production flag set. The functional constants live in
`env_config.h` (e.g. `tile_bucket_enabled()`/`l1_record_enabled()`/
`proj_device_scan_enabled()` all `return true`). The runtime env probes are
gone; each accessor returns a fixed constant.

The former on/off kernel feature macros (`MB_RESIDENT`, `MB_SFPU_CULL`,
`MB_BLEND_AOS`, `MB_TILE_BUCKET`, `MB_BUCKET_CB_FENCE`, `MB_L1_RECORD`,
`MB_DEVCONIC`, `CULL_LPT_CB`, `GATHER_EMIT_BLENDREC`, `BIN_EMIT_REC`,
`L1_BUCKET_REC`) were **always defined**, so their `#ifdef` bodies are now
**inlined directly in the kernel source** and the defines are no longer passed
from the host. Only the two **value** defines the kernels actually read remain
in the host `CreateKernel` maps:

```
MB_CULL_SPIN=512   MB_BUCKET_FIT=8192u     (blend reader: reader_alpha_blend_mb_devcull.cpp)
```

(`MB_BUCKET_FIT` is also a `constexpr` inside `alpha_blend_compute_mb.cpp`.)
The only conditional-compilation left in any kernel is `#ifdef TRISC_MATH` —
a tt-metal built-in per-RISC compute-kernel macro, not a config toggle.

---

## Layout

```
render/
  host/
    render.cpp              # orchestrator + pybind module `render_clean`
    config.h                # baked production constants (human-readable)
    env_config.h            # baked flag constants (constexpr, no getenv)
    project_device.cpp      # stage 1a
    pfwc_device.cpp         # stage 1b
    gather_visible_device.cpp  # stage 1c
    tile_assign_device.cpp  # stage 2
    sort_device.cpp         # stage 3 (+ host bin layout)
    blend_device.cpp        # stage 4 (cull) + stage 5 (blend/writeout)
    device_state.cpp/.h     # resident-buffer + device-context registry
    jit_warmup.cpp/.h       # one-shot JIT compile at scene open
    *.h                     # stage headers
  kernels/
    dataflow/               # reader/writer + scatter/scan kernels (above)
    compute/                # project/pfwc/cull/blend compute kernels
  CMakeLists.txt            # separate target -> render_clean pybind module
  run.py                    # render bicycle hero, print hero_vs_ref
  README.md
```

**Kernels are one file per kernel — by requirement, not by style.** tt-metal
JIT-compiles each kernel from its source path at run time (the host names the
exact `kernels/{dataflow,compute}/<name>.cpp` in `CreateKernel`), so kernels
cannot be merged into shared translation units.

**Host is modular: one `*_device.cpp` driver per stage** (plus `render.cpp`,
`device_state`, `jit_warmup`). Full single-file consolidation was deliberately
*not* done: the per-stage drivers each define file-local anonymous-namespace
helpers with the same names (`make_dram`, `soa_pool`, a `cb(...)` lambda,
`ensure_context`, …), so collapsing them would require renames — not the pure,
parity-safe mechanical movement this pass is limited to. The dead-code removal
below was the priority; the stage split is kept to protect the 63.85 dB anchor.

There are **no debug/alternate `#ifdef` branches** left in the host or kernels
(the only `#ifdef` is the built-in `TRISC_MATH`), and **no CPU fallback**:
unsupported inputs hard-fail (`set_fail()`/`throw`) instead of silently
computing on the host. Removed alternate *algorithms* (vs. pure debug cruft)
are archived with git refs in `opt/render-alternate-paths.md`.

---

## Build (on device)

The device builds under `/localdev`; sync first, then build the separate
`render_clean` target (it does not touch the existing `build-tt`).

```bash
cd /Users/smarton/dev/gstt2
scripts/sync_to_bh30.sh            # or: rsync render/ <host>:/localdev/.../render/

# on the device:
export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release
cmake --build render/build-tt -j 16
# -> render/render_clean.cpython-310-x86_64-linux-gnu.so
```

`CMakeLists.txt` builds two static libs (`render_gsplat_cpu` host helpers,
`render_tt` device drivers) and the `render_clean` pybind module, links
`tt-metal` via `cmake/TtMetalInTree.cmake`, and sets `OVERRIDE_KERNEL_PREFIX`
to `render/` so the JIT finds `render/kernels/...`.

## Run + validate

Run through `tt-workflows` `devrun.sh` so the per-host device lock is held:

```bash
cd /Users/smarton/dev/gstt2
/Users/smarton/dev/tt-workflows/scripts/devrun.sh --tag render-clean -- \
  "source .venv/bin/activate; python3 render/run.py --iter-dir render-clean"
# -> SUMMARY scene=bicycle hero='hero' hero_vs_ref=63.85dB out=tmp/render-clean
```

Outputs (`hero_clean.png`, `hero_ref.png`, `hero_diff10.png`) land in
`tmp/<iter-dir>/`.

> `run.py` sets the production `GSPLAT_TT_*` flags **only so the cpu_cpp_mb
> reference reproduces the production `verify_cmd` measurement** — the reference
> backend's CPU output depends on them. `render_clean` itself ignores all of
> them (its config is baked).

---

## Hacking a kernel

1. Edit the kernel in `render/kernels/{dataflow,compute}/` — there are no
   `MB_*`/`BIN_*` `#ifdef` branches to reason about; the live path is inlined
   (the only conditional is the built-in `#ifdef TRISC_MATH`).
2. `rsync` + `cmake --build render/build-tt` (kernels are JIT-compiled at run
   time, but rebuild the module if you touched host code).
3. Re-run `run.py` and check `hero_vs_ref` stays ~63.85 dB.

To change dispatch (buffer sizes, core grids, runtime args) edit the matching
`host/*_device.cpp` driver; `device_state` owns all resident buffers and
contexts so you can see exactly what persists across stages.
