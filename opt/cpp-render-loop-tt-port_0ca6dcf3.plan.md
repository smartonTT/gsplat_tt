---
name: cpp-render-loop-tt-port
overview: Re-architect the Tenstorrent Gaussian-splat renderer around a single fused C++ render loop (no Python in the render loop), port blend then all pre-blend stages onto the Blackhole device keeping data resident, and run a resilient autonomous PSNR-gated optimization loop on bh-30 down to a 1 ms/view target — keeping REPORT.html and opt/metal-iters.jsonl updated every iteration.
todos:
  - id: plan-doc
    content: Write opt/plan-amendment-003-cpp-render-loop-tt-port.md superseding amendment-002 (C++ fused render loop, step order, autonomy contract)
    status: pending
  - id: step0-loop
    content: Fix pybind11 cmake discovery on bh-30 build; harden + restart the autonomous heartbeat/supervisor loop; verify REPORT.html regenerates each tick
    status: pending
  - id: step1-viewer
    content: Build cpu_cpp on bh-30, launch viser viewer with cpu_cpp_mb on scenes/bicycle.ply, return SSH-forwarded http://localhost:8080 link
    status: pending
  - id: step2-parity
    content: Inspect cpu_cpp_mb code; ensure it's similarly efficient as cpu_cpp; optimize very low hanging fruit only; tt will supersede
    status: pending
  - id: arch-fused
    content: Implement gsplat_tt::render_full_tt single-call C++ render loop mirroring render_full_py; re-enable TtBackend.has_render_fused; remove Python per-stage orchestration from render path
    status: pending
  - id: step3-blend
    content: Port mb-major 4x8 alpha-blend kernel to TT (per-microblock splat lists from cull_and_blend); PSNR gate hero>=ref-0.5dB, 30-view>=60dB
    status: pending
  - id: step4-preblend
    content: Port tile_assign, sort, project into render_full_tt keeping buffers device-resident; PSNR-gated only, no perf gate
    status: pending
  - id: step5-optimize
    content: Optimize on-device pipeline to 1 ms/view (project first); persistent buffers, replay, bf16, frame coherency; stop at <=1ms/view
    status: pending
isProject: false
---

## 1. Context and what changes vs amendment-002

- **Topology:** Mac = dev/control box. `bh-30` (SSH host) = Blackhole P150 + 24-core EPYC. Edit on Mac, `rsync` to `/localdev/smarton/gstt2` via [scripts/sync_to_bh30.sh](scripts/sync_to_bh30.sh), build `build-tt`, render there. Reference = `cpu_cpp_mb` on bicycle (6.13M Gaussians, 1024x1024, 30 preset views in [benchmarks/cameras_v2.json](benchmarks/cameras_v2.json)).
- **The core change (your KEY POINT):** amendment-002's `TtBackend` deliberately disables the fused path (`has_render_fused()->False` in [backends/tt/backend.py](backends/tt/backend.py):89) and runs Python per-stage orchestration in [gsplat/pipeline.py](gsplat/pipeline.py):184-260, marshaling numpy<->torch between every stage. This **is** Python in the render loop. We replace it with a single C++fused loop, mirroring `render_full_py` ([backends/cpu_cpp/pybind_module.cpp](backends/cpu_cpp/pybind_module.cpp):883), so orchestration + device dispatch + cross-stage buffers all live in C++.
- **Stage-port philosophy (yours):** no perf gate to accept a TT port — only the PSNR gate. Get everything on-device first (data stays resident, DMA shrinks), then optimize.

## 2. Target architecture — `render_full_tt`

New C++ entrypoint `gsplat_tt::render_full_tt(...)` in `src/gsplat_tt/` that owns the whole frame, structured exactly like `render_full_py` (project -> tile_assign -> sort -> cull+blend) but each stage dispatches to either a TT kernel (once ported) or the existing `gsplat_cpu` function (until ported), with intermediate buffers held in device-resident `device_state` so ported neighbors avoid D2H/H2D.

```mermaid
flowchart LR
  subgraph py [Python: setup only]
    load[load_ply + cameras] --> call["backend.render_fused(view)"]
  end
  call --> rf["render_full_tt (one pybind crossing)"]
  subgraph cpp [C++ render loop, data resident on device]
    rf --> proj[project] --> ta[tile_assign] --> srt[sort] --> bld[cull+blend] --> img[image out]
  end
  img --> save[PNG / viser frame]
```



- `TtBackend.has_render_fused()` returns True again; `render_fused()` calls `_mod.render_full_tt(...)`. The per-stage `project()/blend()` overrides are removed from the render path (kept only as opt-in diagnostics).
- Not-yet-ported stages call the same `gsplat_cpu::` functions the CPU loop uses, so PSNR is bit-identical to `cpu_cpp_mb` at the start.
- Reuses existing device scaffolding: [src/gsplat_tt/device_state.h](src/gsplat_tt/device_state.h), [src/gsplat_tt/blend_device.cpp](src/gsplat_tt/blend_device.cpp), [src/gsplat_tt/project_device.cpp](src/gsplat_tt/project_device.cpp), [src/gsplat_tt/pfwc_device.cpp](src/gsplat_tt/pfwc_device.cpp).

## 3. Step-by-step (your ordering)

### Step 0 — Resilient autonomous loop (prereq)

- Fix the dead supervisor: last heartbeat failed with `Could not find pybind11` during the bh-30 cmake configure. Pin `pybind11_DIR`/`CMAKE_PREFIX_PATH` from the venv (`python -m pybind11 --cmakedir`) in the build step (`scripts/build_cpu_cpp.sh` / `scripts/setup_bh30_metal.sh`).
- Harden + restart [scripts/amendment002_heartbeat.sh](scripts/amendment002_heartbeat.sh) (180 s tick: sync -> build -> diagnose -> `opt/build_report.py` -> log). Loop must never exit on a single failure (already `|| continue`-style) and must self-recover (stale JIT cache `rm -rf`, device reset, rebuild). Update [opt/amendment002-supervisor-state.json](opt/amendment002-supervisor-state.json) each tick.
- Every iteration: append to [opt/metal-iters.jsonl](opt/metal-iters.jsonl) and regenerate `file:///Users/smarton/dev/gstt2/opt/REPORT.html` via [opt/build_report.py](opt/build_report.py).

### Step 1 — cpu_cpp_mb viewer on bh-30 (bicycle)

- Build `cpu_cpp` (no TT) on bh-30; start viser ([gsplat/**main**.py](gsplat/__main__.py), host `0.0.0.0`) with `--backend cpu_cpp_mb scenes/bicycle.ply --port 8080`.
- Expose via SSH local-forward (`ssh -L 8080:localhost:8080 bh-30`) and return `http://localhost:8080` as the web link.

### Step 2 — cpu_cpp_mb efficiency parity with cpu_cpp

- Look at the code and make sure cpu_cpp_mb is not much less optimal than cpu_cpp. If it is, make it decent, but don't spend big effort that will be irrelevant when ported to tt.

### Step 3 — Port alpha-blend microblock 4x8 kernel to TT

- Inputs already exist: `cull_and_blend` produces per-microblock splat lists (`mb_header[tile,m]=(off,cnt)`, `mb_stream`). Confirm these are surfaced into `render_full_tt`; if not, expose them (they are built in `cull_and_blend.cpp`).
- Implement mb-major device kernel: outer loop over 32 microblocks/tile, inner over `mb_stream[off:off+cnt]`, 4x8 SFPU inner block using the A..F coeff basis. Detailed reader/compute spec already drafted in [opt/amendment002-phase2-handoff.md](opt/amendment002-phase2-handoff.md) §3.2  and a more extensive proposal you should read is here /Users/smarton/dev/gstt2/opt/[microblock-kernel-design.md](http://microblock-kernel-design.md) (mb_mask approach may be a correctness stepping-stone, only if 4x8 is too big of a lift, then DST-persistent 4x8 microblock state for perf). Start with whichever is faster to get correct — full 32x32 tile fallback is acceptable as tt-001a if 4x8 is not immediately easy.
- Gate (PSNR only): PSNR should stay high, preferably over 60 dB. Definitely not below 40. bf16 + fp32 dest accumulate; watch cov2d ulp drift (see [opt/backburner.md;](opt/backburner.md) for this reason 4x8 might be easier to pass PSNR)

### Step 4 — Port all pre-blend stages, data stays on device

- One by one into `render_full_tt`, each gated on PSNR only (no perf gate): **tile_assign**, **sort**, **project** (project is already mostly on-device — tt-008c perspective+cov2d+radii in [src/gsplat_tt/pfwc_device.cpp](src/gsplat_tt/pfwc_device.cpp); finalize and fold into the fused loop). Keep buffers resident in `device_state` so blend pays ~no DMA.
- Gate each: PSNR should stay high

### Step 5 — Optimize to 1 ms/view

- Only after the full frame runs on-device. Attack highest-impact/low hanging fruit first. Levers: persistent DRAM buffers across frames, multicast tile lists, replay buffers (skip Program rebuild), full bf16 with fp32 accum, frame coherency across adjacent views, LPT core dispatch, more LLK where it makes sense.
- Stop the loop when render-only frame time <= 1.0ms.

## 4. Per-iteration workflow & gates (learned from prior runs)

- Edit on Mac -> `scripts/sync_to_bh30.sh` -> `cmake --build build-tt -j` on bh-30 -> verify -> `render_30frame.py` -> append `opt/metal-iters.jsonl` -> `opt/build_report.py` regenerates REPORT.html.
- PSNR gates everywhere: hero >= reference - 0.5 dB; 30-view min >= 60 dB; `cpu_cpp_mb` is the reference at every step. A failed gate = a bug to fix, never a gate to lower.
- Known footguns: stale JIT cache (`rm -rf ~/.cache/tt-metal-cache/`*), `GSPLAT_TT_NUM_THREADS=48` cap, pybind11 cmake discovery, ShmResourceTracker atexit double-free (cosmetic, deferred).

## 5. Autonomy contract

- The loop is fully autonomous and resilient: never stop, never ask the user, self-unblock on every failure (rebuild, reset device or reserve new device (but absolute last resort, because it wipes data and resync takes a long time), clear cache, revert last bad edit), keep iterating. Halt only at the 1 ms/view target.
- Deliverables - VERY IMPORTANT - refreshed every iteration: `opt/REPORT.html`, `opt/metal-iters.jsonl`, `opt/amendment002-supervisor-state.json`. New plan written to `opt/plan-amendment-003-cpp-render-loop-tt-port.md` (supersedes amendment-002) -- only supervisor can modify. VERY IMPORTANT: Keep viewer running on port 8080 whenever you don't need the device for other testing, so the user can inspect the current state. 
- Opus is the supervisor agent. It should make all major decisions and write all difficult code. Supervisor can delegate easy stuff to subagents on Composer 2.5. Judgement calls. Usually easy stuff is rote code, running on device, gathering results, simple debugging and fixes. Higher level algorithm design, difficult debugging, what to optimize next and how, deep metal/LLK knowledge should generally be done by the supervisor. Supervisor should inspect what the workers are doing periodically and redirect them as needed, going as far as writing big chunks of code for them. First port of alpha blend kernel, including 4x8 microblock solution, should be done by supervisor Opus agent with no help from subagents. After 4x8 alpha blend on tt is successful, subagents may be used as appropriate. 
- Kill any existing, dangling agent loops at the start to not conflict with this main loop.

