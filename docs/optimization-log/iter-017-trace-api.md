# iter-017 — tt-metal Trace API

- Class: dispatch
- Track: host-dominated lever family, high-leverage move
- Date: 2026-05-26
- Status: PLANNED
- Predecessor: iter-015 baseline (96.80 ms KEEP) + iter-016 negative signal (per-buffer persistence regresses by +1.6 ms)

## Why this iter exists

iter-015 classified the kernel as host-dominated (52% overhead). iter-016
tried the smallest move in the lever family — persist 4 MB of px/py
buffers — and measured a +1.6 ms regression because the dispatch
hot-path branch + allocator pressure exceeded the EnqueueWrite savings.

The tt-metal Trace API collapses an entire dispatch sequence
(6 EnqueueWriteMeshBuffer + 1 EnqueueMeshWorkload + per-core SetRuntimeArgs
push) into a single `replay_mesh_trace()` call that replays cached
command-stream commands. This bypasses the entire per-frame host driver
work — exactly the bound we're trying to break.

Per iter-016 lesson: Trace replay also amortizes the persistent-buffer
overhead that regressed iter-016 (the per-frame branch is gone — it's
just replay).

## Hypothesis

The 50 ms host overhead is dominated by dispatch command construction +
per-core runtime-arg push. Trace replay should reduce kernel_ms toward
the device-FW floor (~46 ms). Expected: **≥10 ms improvement** (≤86 ms
median) for KEEP, with reach goal ≤55 ms (device FW + thin replay shim).

If trace replay shows <5 ms improvement, the bound is upstream of
dispatch (e.g. EnqueueRead blocking semantics or driver-level
serialization), which would be a major architectural finding.

## Implementation sketch

### Buffer lifecycle: max-allocate once

Move from per-frame `allocate_frame_buffers()` to `ensure_max_buffers()`
called once at first frame. Sizes are bounded by:
- packs: `MAX_TOTAL_ENTRIES × SCALAR_PACK_PAGE_BYTES`. Will measure max
  across the 3 bench views and round up to give headroom.
- offsets: `MAX_OFFSETS_COUNT × sizeof(uint32_t)`. Tied to num_tiles + 1
  in worst case.
- px / py: `MAX_NUM_TILES × TILE_BYTES_BF16`. Fixed for 1024² benchmark
  (1024 tiles).
- output: `MAX_NUM_TILES × 3 × TILE_BYTES_BF16`. Same.
- tile_ids: `MAX_TILE_IDS_BYTES`. Bounded by num_tiles when all are non-empty.

Conservative bound: allocate at 1024² × 2× safety. Output buffer dominates at ~6 MB
× 2 = 12 MB. Total persistent footprint ~80 MB DRAM, well within board limits.

### Trace cache keyed by per-frame dispatch parameters

```cpp
struct TraceEntry {
    distributed::MeshTraceId trace_id;
};
struct TraceKey {
    uint32_t image_h, image_w, total_entries;
    std::vector<uint32_t> per_core_count;  // hashes LPT distinctiveness
    bool operator==(const TraceKey&) const = default;
};
struct TraceKeyHash { size_t operator()(const TraceKey&) const; };
std::unordered_map<TraceKey, TraceEntry, TraceKeyHash> trace_cache;
```

For the benchmark this naturally yields 3 cache entries (one per fixed
view), each replayed 9× across the 10 cycles.

### Two command queues

```cpp
ctx.mesh_device = MeshDevice::create_unit_mesh(
    /*device_id=*/0,
    /*l1_small_size=*/0,
    /*trace_region_size=*/16 << 20,   // 16 MB
    /*num_command_queues=*/2);
ctx.data_cq = &ctx.mesh_device->mesh_command_queue(0);
ctx.workload_cq = &ctx.mesh_device->mesh_command_queue(1);
```

### Per-frame flow

```
1. Load .npy, compute LPT, build trace_key.
2. Look up trace_key in cache.
3. (cache miss path)
   a. ensure_max_buffers() — allocate persistent buffers if not yet allocated.
   b. set_per_core_runtime_args() with the persistent buffer addresses + new (start, count).
   c. Upload inputs to persistent buffers on data_cq, Finish().
   d. Dispatch once normally (JIT pre-warm + correctness sanity).
   e. BeginTraceCapture on workload_cq; EnqueueMeshWorkload (blocking=false); end_mesh_trace.
   f. Store trace_id in cache.
4. (cache hit path)
   a. Upload inputs to persistent buffers on data_cq.
   b. record_event on data_cq, wait_for_event on workload_cq.
   c. replay_mesh_trace(workload_cq, trace_id, blocking=false).
   d. EnqueueReadMeshBuffer(output) on workload_cq, blocking=true.
```

### Time window

`kernel_ms` continues to measure host-wall time of the
"upload → dispatch → readback" critical section, so we get apples-to-
apples comparison with iter-015 / iter-016.

## Validation gate

- PSNR floor: 40 dB any view. Kernel math + ordering identical to
  iter-015 baseline. Replay should be bit-identical to capture
  (validated via md5 of output PNGs as in iter-016 sanity check).
- kernel_ms target: ≤ iter-015 × 0.95 (= 91.96 ms) for KEEP; the explicit
  goal is ≥10 ms improvement (≤86 ms). Reach goal: approach device FW
  floor ~50 ms.
- No visible artifacts.

## Risks

- **Replay correctness depends on persistent buffer addresses being
  stable.** If MeshBuffer reallocates internally between frames (e.g.
  page table churn), the cached trace becomes invalid. Mitigation:
  one-shot allocate at first frame, retain shared_ptr in ctx for full
  daemon lifetime.
- **Trace region size**. If 16 MB is insufficient for the captured
  command stream we'll get a TT_FATAL at end_mesh_trace. Mitigation:
  start at 16 MB, increase if needed.
- **EnqueueReadMeshBuffer timing**. The buffer is shared across all
  views; reads happen on workload_cq after replay. If readback races
  with the next frame's upload, we corrupt output. The mandatory
  `blocking=true` read provides the sync point.

## Files to edit

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`
  - `DeviceContext`: add `data_cq`, `workload_cq`, `persistent_buffers` (max-alloc'd MeshBuffers), `trace_cache`.
  - `init_device_context`: switch to 2-CQ mesh, reserve trace_region.
  - `process_frame`: split into cache-miss and cache-hit branches.

## Outcome

(to be filled in after run)
