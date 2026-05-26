# iter-018 — batch the 3 small-page DRAM uploads into a single allocation

- Class: dispatch
- Track: host-dominated lever family, post-iter-017 architectural finding
- Date: 2026-05-26
- Status: PLANNED
- Predecessor: iter-015 baseline (96.80 ms KEEP). iter-016 (persist px/py) and iter-017 (full Trace API) both REJECTED in the host-dominated lever family.

## Why this iter exists

iter-017 measured a critical negative signal: full Trace API + 2-CQ +
persistent buffers + capture/replay produced ~0 ms (-0.145 ms vs 96.80 ms
baseline). This proved the 50.5 ms host overhead is NOT in dispatch command
construction or per-core SetRuntimeArgs push — it is in **per-frame DRAM
transfer + driver serialization**.

iter-016 measured another negative signal: persisting individual <10 MB
buffers across frames is net-negative on this hardware (allocator pressure
+ branch overhead exceed the EnqueueWrite savings).

This iter targets a third hypothesis in the same lever family: **per-call
EnqueueWriteMeshBuffer overhead dominates the per-byte transfer cost**. If
true, then collapsing N small writes into 1 larger write should give
roughly (N-1) × per_call_overhead_ms saved, without persistence complexity.

## Hypothesis

Currently each frame issues 6 separate EnqueueWriteMeshBuffer calls:
1. output zero-fill (~6 MB, 2 KB pages)
2. packs (~few MB, 64 B pages)
3. offsets (~16 KB, 4 B pages)
4. px (~2 MB, 2 KB pages)
5. py (~2 MB, 2 KB pages)
6. tile_ids (~few KB, 64 B pages)

Three of these (packs, offsets, tile_ids) use small page sizes (≤64 B) and
together total ~few MB. They can be merged into a single 64-B-page DRAM
buffer with per-stage byte offsets passed as runtime args.

Expected savings: **2 EnqueueWrite calls removed × ~0.5 ms each = ~1 ms**.

Pessimistic case: per-call overhead is ≤0.1 ms (driver doesn't add fixed
cost per call, only per-byte). Then we'll measure ≤0.3 ms improvement and
that closes the door on this lever family entirely.

## Implementation sketch

### Per-frame combined small-page buffer

```cpp
struct SmallBufferLayout {
    size_t packs_offset_bytes    = 0;
    size_t offsets_offset_bytes  = 0;
    size_t tile_ids_offset_bytes = 0;
    size_t total_bytes           = 0;
};

static SmallBufferLayout compute_small_layout(
    uint32_t total_entries, size_t offsets_count, size_t tile_ids_bytes) {
    auto align64 = [](size_t n) { return (n + 63) & ~size_t{63}; };
    SmallBufferLayout L;
    L.packs_offset_bytes    = 0;
    const size_t packs_bytes = align64(static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES);
    L.offsets_offset_bytes  = packs_bytes;
    const size_t offsets_bytes = align64(offsets_count * sizeof(uint32_t));
    L.tile_ids_offset_bytes = packs_bytes + offsets_bytes;
    L.total_bytes           = packs_bytes + offsets_bytes + align64(tile_ids_bytes);
    return L;
}
```

Per frame:
1. Compute layout from `(total_entries, offsets_count, tile_ids_bytes)`.
2. Allocate ONE 64-B-page MeshBuffer of `L.total_bytes`.
3. Build ONE host vector of `L.total_bytes / 4` uint32s (or bytes), copy
   packs/offsets/tile_ids into the right offsets.
4. ONE `EnqueueWriteMeshBuffer` on the combined buffer.
5. Kernels receive `(combined_addr, packs_off, offsets_off, tile_ids_off,
   start, count)` runtime args. Reader/writer compute per-stage absolute
   addresses as `combined_addr + per_stage_off + per-core slice`.

### Kernel changes

Reader kernel currently reads `packs_addr, offsets_addr, tile_ids_addr` as
separate base addresses. Change to a single base + 3 offsets. Same for
writer (uses tile_ids_addr).

px/py/output remain separate buffers (different page sizes, would lose
alignment if folded in). 6 EnqueueWrites → 4 EnqueueWrites.

### Timing

`kernel_ms` continues to measure the upload→dispatch→readback critical
section, apples-to-apples vs iter-015 / iter-016 / iter-017.

## Validation gate

- PSNR floor: 40 dB any view. Kernel math unchanged (only buffer
  reorganization), so PSNR should be identical to iter-015 baseline.
- kernel_ms target: ≤ iter-015 × 0.95 (= 91.96 ms) for KEEP; explicit goal
  ≥1 ms improvement (≤95.8 ms).
- No visible artifacts.

## Decision branches based on outcome

- **KEEP (95.8 ms or better)**: per-call EnqueueWrite overhead IS the
  bottleneck; the natural follow-up is iter-019 to also fold px/py/output
  into a second large-page buffer (4 → 2 writes).
- **REJECT (~96.8 ms, flat)**: per-call overhead is negligible vs byte
  transfer; the host-dominated lever family is fully exhausted. Pivot back
  to compute-bound levers (e.g. kernel-side output zero-fill, or revisit
  iter-014's init-fuse with a different deadlock workaround).

## Files to edit

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`
  - `FrameDramBuffers`: replace `packs/offsets/tile_ids` with `small_combined` + offsets.
  - `allocate_frame_buffers`: allocate 1 combined buffer instead of 3.
  - `process_frame`: pack the 3 stages into one host vector, ONE EnqueueWrite.
  - `set_per_core_runtime_args`: pass (combined_addr, packs_off, offsets_off, tile_ids_off, ...).
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/reader.cpp`
  - Read combined_addr + per-stage offsets.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/writer.cpp`
  - Read combined_addr + tile_ids_off.

## Risks

- **Kernel page-size assumption**. The reader reads packs at 64-B pages,
  offsets as raw uint32 indices. If the kernel computes "DRAM page N"
  internally, having all 3 stages in one buffer shouldn't break addressing
  as long as each stage's start is 64-B-aligned (we enforce this).
- **Runtime-arg slot count**. Reader runtime args go from 7 → 7 (replace 3
  base addresses with 1 base + 3 offsets — same count). No layout change.
- **Tile_ids padding**. Currently tile_ids are at the end of their own
  buffer with padding for NoC alignment. The combined buffer needs the same
  padding semantics for the tile_ids region.

## Outcome

(to be filled in after run)
