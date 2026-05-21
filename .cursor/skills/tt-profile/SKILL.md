---
name: tt-profile
description: Profile the gsplat_tt alpha-blend kernel on the Tenstorrent Blackhole P150 device (host `bh-30`). Extract kernel-only device time (ms) and per-stage cycle counts so the optimization loop can identify the dominant bottleneck. Use when the supervisor or a worker needs to measure current kernel performance, compare iterations, or decide which kernel stage to optimize next.
---

# tt-profile

Measure where time goes inside the alpha-blend kernel on the Tenstorrent **Blackhole P150** device (SSH host `bh-30`), so the optimization loop has a current bottleneck to attack.

## Conventions

- All commands run from the project root (`/proj_sw/user_dev/smarton/gsplat_tt/` on `bh-30`, or `~/dev/gsplat_tt/` locally — the local checkout can't talk to the device).
- The kernel binary is `backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting`. After editing `kernels/compute/alpha_blend_compute.cpp` or its host driver, rebuild with `sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting`. The kernel `.cpp` sources under `kernels/` are JIT-compiled by tt-metal at runtime; only the host binary needs CMake.
- Device kernel time is the **only** trustworthy number in this skill. CPU prep, .npy save/load, daemon round-trip, etc. are all reported separately and not what we are optimizing.

## Three levels of profiling

### 1. Kernel-only wall time (default, every iteration)

The TT daemon already reports per-frame device-side kernel ms on stdout (`OK <ms>`) and the Python wrapper surfaces it as `sub_timings["blend.daemon_rt.device_kernel"]`. This is the ms number the loop optimizes.

```bash
ssh bh-30 "cd /proj_sw/user_dev/smarton/gsplat_tt && source venv/bin/activate && \
  TT_METAL_HOME=\$PWD/backends/tt/tt-metal \
  TT_METAL_RUNTIME_ROOT=\$PWD/backends/tt/tt-metal \
  python scripts/render_fixed.py stitch hero --warmup 3 --frames 10"
```

The script prints the median `device_kernel` ms (kernel-only) along with the daemon round-trip ms and other sub-timings.

### 2. Per-stage cycle counts (when zooming in on a hot stage)

When a single kernel ms number isn't enough — e.g. the loop has tried two changes and neither moved the needle and the supervisor wants to see whether stages B/C/D/E shifted — enable the tt-metal device profiler.

```bash
ssh bh-30 "cd /proj_sw/user_dev/smarton/gsplat_tt && source venv/bin/activate && \
  TT_METAL_HOME=\$PWD/backends/tt/tt-metal \
  TT_METAL_RUNTIME_ROOT=\$PWD/backends/tt/tt-metal \
  TT_METAL_DEVICE_PROFILER=1 \
  python scripts/render_fixed.py stitch hero --warmup 1 --frames 3 --profile"
```

CSV lands at `backends/tt/tt-metal/generated/profiler/.logs/profile_log_device.csv`. Each `DeviceZoneScopedN("name")` macro inside the kernel becomes a row with start/end cycle counts. **The current kernel has no zone markers** — to get per-stage numbers, add the markers, rebuild, re-run. Suggested zones (matching the comment labels in `alpha_blend_compute.cpp`):

- `StateInit` (lines ~131-194, per-tile state CB init)
- `StageA` (read scalars)
- `StageB1` (dx/dy)
- `StageB2` (dx², dy², dx·dy — keep one zone wrapping all three)
- `StageB3C` (Q + power + exp + alpha; the long acquire block at lines ~360-415)
- `StageD` (R/G/B accumulator update; lines ~430-553)
- `StageE` (transmittance update; lines ~554-601)
- `StageF` (sat_mask refresh; lines ~216-239)
- `Finalize` (lines ~604-622, the 3-tile output pack)

Add the zone markers only when the bottleneck question requires them; remove or guard with `#if 0` between profiling runs (zone markers themselves cost cycles).

### 3. Frame-level NoC / dispatch (when CB plumbing is suspected)

If the kernel itself stays fast but daemon round-trip is large, the bottleneck might be reader/writer NoC stalls or CB depth. Enable host-side dispatch traces:

```bash
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1
```

Watcher output goes to `backends/tt/tt-metal/generated/watcher/*.log`. Use sparingly — adds significant overhead.

## Interpreting output

`render_fixed.py` prints lines like:

```
[scene=stitch view=hero W=480 H=640 visible=256558 entries=832049]
[median over 10 frames]  kernel=69.13 ms  daemon_rt=122.43 ms  prep=58.62 ms  save_npy=10.90 ms  load_npy=0.81 ms
[fps] kernel-only=14.47  end-to-end=3.12
```

(numbers above are the iter-000 baseline on Blackhole; current best may be lower).

The supervisor compares `kernel=...` across iterations. **All four scenes' kernel ms must move in the same direction**; if stitch improves but bicycle regresses, the change is workload-specific and may not be a real win.

## Per-iteration recipe (called from optimization-loop)

```
1. Establish baseline kernel ms with `render_fixed.py stitch hero --warmup 3 --frames 10`
2. Worker edits kernel + Python plumbing.
3. `sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting`
4. Re-run the same `render_fixed.py` invocation, capture new median.
5. delta_ms = new - old. Report delta_ms + delta_% in iter log.
6. If a deeper question — e.g. "which stage did this affect?" — re-run with `--profile` after adding zone markers.
```

## Common pitfalls

- **Forgetting to rebuild.** The host binary `metal_example_gaussian_splatting` must be rebuilt after any C++ host-side change in `alpha_blend.cpp` or `alpha_blend_host.h`. Kernel `.cpp` files under `kernels/` are JIT-compiled — but you still need the host binary to pick up changes to its own compilation, e.g. flag changes.
- **Stale daemon.** `pkill -f metal_example_gaussian_splatting` between runs guarantees a fresh daemon. The Python wrapper spawns one each invocation but a stuck process from a previous crash holds the device.
- **First frame skew.** The first frame after daemon startup is 3-5x slower (JIT compile of kernel code). Always discard warmup frames; `render_fixed.py --warmup 3` is the default minimum.
- **Resolution drift.** The plan locks resolution per scene-view via `benchmarks/cameras.json`. Never let it drift across iterations — kernel ms scales superlinearly with H × W.
