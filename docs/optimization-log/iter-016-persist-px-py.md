# iter-016 — persist px/py DRAM buffers across frames

- Class: dispatch
- Track: post-iter-015 lever-family pivot
- Date: 2026-05-26
- Status: REJECT (reverted, code rolled back)
- Predecessor: iter-015 bound-class baseline (KEEP, 96.80 ms, classified as host-dominated 52% overhead)

## Why this iter exists

iter-015 confirmed the kernel is host-dominated: 96.80 ms host-side
kernel_ms vs 46.34 ms device FW max. The 50.5 ms gap is dispatch +
EnqueueWriteMeshBuffer calls for ~60 MB/frame of buffer uploads (packs,
offsets, px, py, tile_ids, output zero-fill).

The iter-007..014 kernel-algebra fusion lever family is out of room —
even at zero compute we can't go below ~50 ms.

This iter is the **first move of the host-dominated lever family**:
persist the tile-pixel-coord (px, py) buffers across frames since they
depend only on resolution, not on camera or scene.

## Hypothesis

px and py are bf16-encoded tile-local pixel coordinates of shape
`(num_tiles, 32, 32)`. At 1024² they're 2 MB each → 4 MB total uploaded
per frame. The contents are deterministically derived from
`(image_height, image_width)` — once computed and uploaded they never
need to change as long as the resolution stays the same.

Caching the device buffer in DeviceContext keyed by `(image_h, image_w)`
skips:
- 2× MeshBuffer allocation (~hundreds of µs of dispatch overhead each)
- 2× EnqueueWriteMeshBuffer of 2 MB each (~0.5 ms each at 8 GB/s effective)
- 2× host-side bf16 encoding (~0.2 ms at 1M elements)

Expected savings: **2-4 ms** per frame (~2-4% of 96.8 ms). Small but
non-zero, and it proves the lever family. If it works we move on to
the bigger output zero-fill upload (6 MB, ~iter-017) and ultimately
the tt-metal Trace API (~iter-018).

If this iter shows ≤ 0.5 ms improvement, it would suggest dispatch
overhead per call is the bottleneck more than transfer bytes — useful
signal for choosing between "batch uploads into one big buffer" vs
"avoid uploads entirely" levers in future iters.

## Implementation sketch

1. Add a `ResolutionBuffers` cache to `DeviceContext`:
   ```cpp
   struct ResolutionBuffers {
       uint32_t image_h = 0, image_w = 0;
       std::shared_ptr<distributed::MeshBuffer> px, py;
   };
   ResolutionBuffers res_cache;
   ```
2. Refactor `allocate_frame_buffers` to NOT allocate px/py.
3. In `process_frame`, before encoding/uploading px/py:
   - If `res_cache.image_h == image_h && res_cache.image_w == image_w`,
     reuse cached buffers.
   - Otherwise allocate new buffers, encode + upload, store in cache.
4. The cache holds at most one resolution (the current one) — keep it
   simple. If image resolution changes mid-session (rare in benchmark
   harness, possible during interactive zoom), we evict + reallocate.

## Validation gate

This is a dispatch-class iter:
- PSNR floor: 40 dB any view (NEEDS_REVIEW below) — kernel math
  unchanged, so PSNR should be identical to iter-015 baseline
  (40.40 / 43.70 / 40.15 dB).
- kernel_ms target: ≤ iter-015 × 1.02 (= 98.7 ms) for KEEP;
  improvement of ≥ 1 ms preferred (the explicit goal).
- No visible artifacts (validator gate).

## Risks

- bf16-encoded px/py might have alignment or DRAM-page issues if the
  buffer is allocated once at first frame and the page size changes
  between resolutions. Mitigation: cache eviction on resolution change
  is a hard re-allocate, so paging is fresh per resolution.
- The daemon may receive a frame with new H/W before the cache exists.
  First-frame path must allocate-and-cache.

## Files to edit

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`
  - `DeviceContext` struct: add `ResolutionBuffers res_cache`.
  - `allocate_frame_buffers`: drop px/py from the FrameDramBuffers it
    returns; rename or refactor as needed.
  - `process_frame`: route px/py allocation + upload through the cache.

## Outcome

**REJECT — reverted.**

Measured (30 frames, 10 cycles × 3 views):
- kernel_ms_median: **98.44 ms** vs iter-015 baseline 96.80 ms → **+1.64 ms / +1.7% regression**
- per-cycle: hero 98.30-98.50 every cycle (vs iter-015 96.72-96.89). Systematic, not noise.
- PSNR identical to 16 decimals: 40.40 / 43.70 / 40.15 dB.
- Hero/side/top PNGs **byte-identical** to iter-015 (md5 match) — kernel math truly unchanged.

The hypothesis expected -2 to -4 ms from skipping 2× EnqueueWriteMeshBuffer
+ encode of 4 MB total. The actual signature is the opposite direction.

### Negative-signal lesson

Persisting *small* (4 MB) DRAM buffers across frames is net-negative on
this hardware/driver combo. The ~0.5 ms saved per skipped EnqueueWrite is
dominated by:
1. Added per-frame branch in the dispatch hot path (cache hit/miss check).
2. Allocator pressure from long-lived buffers competing with the
   per-frame allocate-and-free pattern other buffers still use.
3. Possibly extra DRAM-page coordination cost when a persistent buffer
   sits alongside ephemeral ones (speculative — would need profiler to
   confirm).

### Implications for the host-dominated lever family

- **Don't persist small buffers individually.** The per-call dispatch
  overhead saved is too small to dominate added complexity.
- **Future iter-017+ should target either:**
  - A *bigger* persistent buffer where the per-frame upload bytes actually
    matter — e.g. the output zero-fill (6 MB of fixed-pattern data). If
    that also regresses, the lever family is exhausted at "persist
    buffers" granularity and we must jump to Trace.
  - **tt-metal Trace API**: collapses the entire 6-EnqueueWrite +
    dispatch sequence into one command-stream replay. This is the
    high-leverage move — eliminates per-frame host driver work entirely,
    not just transfer bytes.

### Files changed (then reverted)

Reverted in commit 0ee8ca5 (revert of e829b8d).
