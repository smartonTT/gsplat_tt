# Plan Amendment 003 — C++ Fused Render Loop + Full TT Port

**Date:** 2026-05-28
**Supersedes:** `opt/plan-amendment-002-tt-emulator-port.md` (in-process per-stage
Python orchestration via `gsplat/pipeline.py`).
**Owner:** Opus supervisor agent. Only the supervisor edits this file.

---

## 0. The one rule that drives this rewrite

**No Python in the render loop. C++ only.**

Amendment-002's `TtBackend` deliberately disables the fused C++ render path
(`has_render_fused() -> False`, `backends/tt/backend.py:89`) and runs the
per-stage loop in Python (`gsplat/pipeline.py:184-260`), marshaling
numpy<->torch between every stage. That marshaling *is* Python in the render
loop. Amendment-003 replaces it with a single fused C++ entrypoint
`gsplat_tt::render_full_tt(...)`, mirroring the existing CPU fused loop
`render_full_py` (`backends/cpu_cpp/pybind_module.cpp:883`). Python only loads
the scene + cameras and calls `render_fused(view)` once per frame.

---

## 1. Topology and reference

- **Mac** = dev/control box (this machine). Edit here.
- **bh-30** (SSH host) = Blackhole P150 + 24-core AMD EPYC. Build + render here.
  Repo mirror at `/localdev/smarton/gstt2`; push via `scripts/sync_to_bh30.sh`.
- **Reference render:** `cpu_cpp_mb` on `scenes/bicycle.ply` (6,131,954 Gaussians,
  1024x1024), 30 preset views in `benchmarks/cameras_v2.json`. The 30-view
  fixtures live in `benchmarks/reference_v2/`.
- **Perf target:** 1 ms/view render-only (30-view `sum_total_ms <= 30 ms`).
- **Baseline (amendment-002):** `cpu_cpp_mb` ~224 ms/view; stage split
  project 74% / blend 16% / tile_assign 7% / sort 2%.

---

## 2. Target architecture — `render_full_tt`

A single C++ function owns the whole frame, structured exactly like
`render_full_py`: project -> tile_assign -> sort -> cull+blend. Each stage
dispatches to either a TT kernel (once ported) or the existing `gsplat_cpu`
function (until ported). Intermediate buffers stay resident in
`gsplat_tt::device_state` so ported neighbors avoid D2H/H2D round trips; the
blend stage in particular must pay ~no DMA once its inputs are device-resident.

```
Python (setup only):  load_ply + cameras  ->  backend.render_fused(view)
                                                     |
                                          one pybind crossing
                                                     v
C++ render loop:  project -> tile_assign -> sort -> cull+blend -> image
                  (buffers held on device in device_state)
```

- `TtBackend.has_render_fused()` returns True again; `render_fused()` calls
  `_mod.render_full_tt(...)`. The per-stage `project()/blend()` overrides are
  retired from the render path (kept only as opt-in diagnostics behind env
  flags).
- Not-yet-ported stages call the same `gsplat_cpu::` functions the CPU loop
  uses, so at the start PSNR is bit-identical to `cpu_cpp_mb`.
- Reuses existing device scaffolding: `src/gsplat_tt/device_state.{h,cpp}`,
  `blend_device.cpp`, `project_device.cpp`, `pfwc_device.cpp`.

---

## 3. Step-by-step (execution order)

### Step 0 — Resilient autonomous loop (prereq)
- Fix the bh-30 build break: heartbeat last failed with `Could not find
  pybind11` during cmake configure. Pin `pybind11_DIR` /
  `CMAKE_PREFIX_PATH` from the venv (`python -m pybind11 --cmakedir`) in
  `scripts/build_cpu_cpp.sh` / `scripts/setup_bh30_metal.sh`.
- Harden + restart `scripts/amendment002_heartbeat.sh` (sync -> build ->
  diagnose -> `opt/build_report.py` -> log). The loop must never exit on a
  single failure and must self-recover (clear stale JIT cache, device reset,
  rebuild, revert last bad edit). Update
  `opt/amendment002-supervisor-state.json` each tick.
- Every iteration: append `opt/metal-iters.jsonl` and regenerate
  `opt/REPORT.html` via `opt/build_report.py`.

### Step 1 — cpu_cpp_mb viewer on bh-30 (bicycle)
- Build `cpu_cpp` (no TT) on bh-30; start viser (`gsplat/__main__.py`, host
  `0.0.0.0`) with `--backend cpu_cpp_mb scenes/bicycle.ply --port 8080`.
- Expose via SSH local-forward (`ssh -L 8080:localhost:8080 bh-30`) and return
  `http://localhost:8080` as the web link.

### Step 2 — cpu_cpp_mb efficiency parity with cpu_cpp
- Read the code; ensure `cpu_cpp_mb` is not much less optimal than `cpu_cpp`.
  Fix only very low-hanging fruit. Do not invest big effort that will be
  irrelevant after the TT port.

### Step 3 — Port the alpha-blend microblock kernel to TT
- Inputs already exist: `cull_and_blend` produces per-microblock splat lists
  (`mb_header[tile,m] = (off, cnt)`, `mb_stream`). Confirm these surface into
  `render_full_tt`; if not, expose them (built in `cull_and_blend.cpp`).
- Implement mb-major device kernel: outer loop over 32 microblocks/tile, inner
  over `mb_stream[off:off+cnt]`, 4x8 SFPU inner block using the A..F coeff
  basis. Detailed reader/compute spec in `opt/amendment002-phase2-handoff.md`
  §3.2 and the fuller proposal `opt/microblock-kernel-design.md` (DST slot map
  §6.1, microblock->DST addr §6.3, outer loop §6.4, replay §6.5, inline exp
  §6.6). The mb_mask approach is a correctness stepping-stone only if 4x8 is
  too big a lift; full 32x32 tile fallback is acceptable as a first wiring
  step. Prefer 4x8 from the start if feasible.
- **This first blend port (including the 4x8 microblock solution) is done by
  the supervisor Opus agent with no subagent help.**
- Gate (PSNR only): keep PSNR high, preferably > 60 dB, definitely not < 40.
  bf16 + fp32 dest accumulate; watch cov2d ulp drift (see `opt/backburner.md`)
  — 4x8 may pass PSNR more easily than the full-tile fallback.

### Step 4 — Port all pre-blend stages, data stays on device
- One by one into `render_full_tt`, PSNR-gated only (no perf gate):
  **tile_assign**, **sort**, **project** (project is already mostly on-device —
  tt-008c perspective+cov2d+radii in `pfwc_device.cpp`; finalize and fold into
  the fused loop). Keep buffers resident in `device_state` so blend pays
  ~no DMA.
- Do not get bogged down if any single port temporarily slows things down. The
  goal is to get all code on device; optimize after.

### Step 5 — Optimize to 1 ms/view
- Only after the full frame runs on-device. Attack highest-impact / low-hanging
  fruit first. Levers: persistent DRAM buffers across frames, multicast tile
  lists, replay buffers (skip Program rebuild), full bf16 with fp32 accum,
  frame coherency across adjacent views, LPT core dispatch, more LLK where it
  helps.
- Stop the loop when render-only frame time <= 1.0 ms/view.

---

## 4. Per-iteration workflow & gates

- Edit on Mac -> `scripts/sync_to_bh30.sh` -> `cmake --build build-tt -j` on
  bh-30 -> verify -> `scripts/render_30frame.py` -> append
  `opt/metal-iters.jsonl` -> `opt/build_report.py` regenerates REPORT.html.
- PSNR gates everywhere; `cpu_cpp_mb` is the reference at every step. A failed
  gate is a bug to fix, never a gate to lower.
- Known footguns: stale JIT cache (`rm -rf ~/.cache/tt-metal-cache/*`),
  `GSPLAT_TT_NUM_THREADS=48` cap, pybind11 cmake discovery, ShmResourceTracker
  atexit double-free (cosmetic, deferred — see `opt/backburner.md`).

---

## 5. Autonomy contract

- The loop is fully autonomous and resilient: never stop, never ask the user,
  self-unblock on every failure (rebuild, reset device, clear cache, revert
  last bad edit). Reserving a new device is an absolute last resort — it wipes
  data and resync takes a long time. Halt only at the 1 ms/view target.
- **Mandatory per-iteration deliverables — NO iteration may be logged PASS/FAIL
  without ALL THREE of these measured (never null/placeholder):**
  1. **PSNR** — hero dB vs `reference_v2` ground truth AND vs the `cpu_cpp_mb`
     reference (min over all 30 views for both). Infinite/bit-identical is a
     valid measured value; a missing number is not.
  2. **Timings** — `sum_total_ms`, `ms_per_view`, and `per_stage_median_ms`
     (project / tile_assign / sort / blend), measured on bh-30.
  3. **Screenshot** — the hero render PNG saved to
     `opt/metal-screenshots/<iter_dir>/hero.png` (build_report.py adds the 10x
     diff vs GT automatically). The verdict is invalid until the screenshot
     exists on the Mac.
  These are produced by `scripts/a003_verify.py --backend tt --iter-dir <id>`
  (run on bh-30; rsync the screenshot + `opt/a003-verify-last.json` back),
  then appended to `opt/metal-iters.jsonl`.
- **Deliverables refreshed every iteration (VERY IMPORTANT):**
  `opt/REPORT.html`, `opt/metal-iters.jsonl`,
  `opt/amendment002-supervisor-state.json`, plus the per-iter screenshot above.
  Only the supervisor modifies this plan file.
- **Keep the viewer running on port 8080** whenever the device is not needed
  for other testing, so the user can inspect the current state.
- **Roles:** Opus is the supervisor — makes all major decisions, writes all
  difficult code (algorithm design, hard debugging, what to optimize next, deep
  metal/LLK work). Composer 2.5 subagents may be delegated rote work (running
  on device, gathering results, simple fixes) — but only after the first 4x8
  alpha-blend TT port succeeds. The supervisor inspects worker progress
  periodically and redirects as needed.
- Kill any dangling agent loops at startup to avoid conflicting with this loop.

---

## 6. What carries over from amendment-002

- `opt/metal-iters.jsonl` (continue logging here).
- `opt/build_report.py` -> `opt/REPORT.html` generation.
- bh-30 build/setup (`scripts/setup_bh30_metal.sh`,
  `cmake/TtMetalInTree.cmake`), sync (`scripts/sync_to_bh30.sh`), and the
  diagnose script (`scripts/amendment002_diagnose_blend.sh`).
- `opt/microblock-cpu-spec.md` + `opt/microblock-kernel-design.md` (algorithm
  + TT kernel spec).
- Device scaffolding under `src/gsplat_tt/`.
- `opt/backburner.md` deferred issues.
