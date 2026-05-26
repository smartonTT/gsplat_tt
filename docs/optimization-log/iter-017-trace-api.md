# iter-017 — tt-metal Trace API

- Class: dispatch
- Track: host-dominated lever family, high-leverage move
- Date: 2026-05-26
- Status: REJECT (reverted, code rolled back)
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

**REJECT — reverted.**

Measured (30 frames, 10 cycles × 3 views):
- kernel_ms_median: **96.655 ms** vs iter-015 baseline 96.80 ms → **-0.145 ms** (essentially flat, within noise)
- PSNR identical to baseline: 40.40 / 43.70 / 40.15 dB
- Validation gate ≤91.96 ms (5% improvement) NOT met; explicit goal ≥10 ms missed by ~10 ms.

### Implementation reality vs plan

The plan stub was implemented and works end-to-end:
- 2-CQ mesh (data_cq + workload_cq) with MeshEvent sync
- Trace capture on cache miss, replay on cache hit
- Max-allocated persistent DRAM buffers (~256 MB packs + 24 MB output + ~16 MB px/py)
- Partial-region writes (`enqueue_write_shard_to_sub_grid` + BufferRegion) so per-frame uploads only touch actual payload bytes
- Partial-region readback (`enqueue_read_shards` + BufferRegion) so 6 MB of tiles transfers, not the full 24 MB persistent buffer

Render quality stayed identical to baseline — kernel math and ordering are bit-correct.

### Critical negative signal

Trace API replaces per-frame dispatch construction + per-core SetRuntimeArgs
push with a single `replay_mesh_trace()` call. Predicted savings: ≥10 ms.
Actual: ~0 ms.

**The host overhead measured by iter-015 (50.5 ms gap between kernel_ms_median
96.8 ms and device-FW max 46.3 ms) is NOT dispatch-command-construction
overhead.** Where it actually lives:

1. **Per-frame DRAM transfer**. We still upload 6 buffers per frame (output
   zero-fill + packs + offsets + px + py + tile_ids). The trace doesn't touch
   uploads — they're outside the capture window by design (writes are not
   trace-replayable; only kernel dispatch commands are).
2. **MeshCommandQueue / driver serialization** between writes and
   dispatch/read. The 2-CQ sync via record_event/wait_for_event adds latency
   that approximately matches whatever dispatch-construction was saved.
3. **Persistent-buffer-of-max-size overhead** (per iter-016 lesson): allocator
   pressure + DRAM-page coordination cost when long-lived large buffers sit
   alongside ephemeral per-frame state.

### Implications

The host-dominated lever family at "dispatch command construction"
granularity is **exhausted**. iter-016 (persist 4 MB), iter-017 (full Trace
API) both delivered ≤0.2 ms net change. The remaining levers in this family
are either much higher-risk or fundamentally different:

- **Skip per-frame uploads on cache-hit cycles** by keeping per-view
  persistent input buffers (one set per (resolution, view) key — 3× memory
  but avoids ~30-40 MB/frame of repeated DRAM transfer). High complexity,
  uncertain gain.
- **Batch the 6 small buffers into one large persistent buffer** so the
  per-frame upload is 1× EnqueueWrite of ~40 MB instead of 6× ~7 MB. May
  shave ~1-2 ms; lower-risk than the per-view approach.
- **Switch back to the compute-bound lever family**. Even though iter-015
  classified the op as 52% host-dominated, that classification assumed
  dispatch overhead was reducible. The non-reducible portion (DRAM transfer
  + driver serialization) appears to be a hard floor at ~50 ms for this
  workload size. The 46 ms device-FW path still has compute headroom
  (e.g. async LLK fusion, init-call reduction inside the dest-resident
  loop) that could trim 5-15 ms.

### Files changed (then reverted)

Reverted in commit (this commit). alpha_blend.cpp restored to the
pre-iter-017 state (1a8b773).

