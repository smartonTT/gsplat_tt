# Iter 015 — tile_ids cache: page-aligned geometric growth

- **Idea**: bugfix, not a perf iter. The persistent daemon's `tile_ids` DRAM
  buffer is grown geometrically as the per-frame non-empty tile count rises.
  The growth function `grow_cap_size(current, needed)` returns
  `max(needed, current + current/2)`. The `needed` arg is already
  page-aligned (rounded to `TILE_IDS_PAGE_BYTES = 64` in
  `build_tile_assignment`), but `current + current/2` is not. Result on the
  ~120-nonempty-tile transitions that occur as the viewer canvas is resized:

      grow_cap_size(320, 448) = max(448, 480) = 480
      480 % 64 = 32

  `MeshBuffer::create` then asserts:

      TT_FATAL: size % page_size == 0
      For valid non-interleaved buffers page size 64 must equal buffer size 480.

- **Branch**: `opt/015-tile-ids-page-align-fix`
- **Decision**: **KEPT** (correctness bugfix, not a perf change)

## What changed

`backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp`:

- New helper `grow_cap_bytes_page_aligned(current, needed, page_size)` —
  same geometric growth as `grow_cap_size`, then rounds up to a multiple of
  `page_size`.
- The tile_ids cache `ensure_buffer_cache` callsite uses the new helper with
  `TILE_IDS_PAGE_BYTES`.

Other geometric-growth callsites (`packs`, `offsets`) are already safe by
construction: `packs` grows by entry count where each entry is exactly
`SCALAR_PACK_PAGE_BYTES` bytes; `offsets` grows by uint32 count where each
element is exactly the page size. Only `tile_ids` grows in raw bytes.

## Verification

Multi-resolution stress test through one persistent daemon:

    $ python /tmp/multi_res_stress.py
    running 60 resolutions through one persistent daemon
      [  1/60] 96x96         OK   ...   [ 60/60] 96x96   OK
    ALL_OK n=60

Each resolution drives the LPT non-empty-tile count to different values,
forcing the cache through the previously crashing 320 -> 448 -> grow=480
transition repeatedly. Before the fix, this exact transition raised
`TT_FATAL: size % page_size == 0` at viewer-time as the user resized the
browser canvas. After the fix, the cache transitions cleanly to 512.

Headless harness on luigi hero 320x640 (baseline preserved):

      sub.blend.daemon_rt.device_kernel  11.75 ms
      timings.total                      35.90 ms
      PSNR vs base reference             37.16 dB
      VERDICT=PASS

## Lessons

1. **Geometric growth on byte counts must always round up to the buffer's
   page size**, not just match an already-aligned `needed` input. The
   `max()` step can pick the unaligned alternative.
2. Cache "grow on demand" + dynamic viewer resolution + non-page-aligned
   geometric growth is a class of bug that headless fixed-resolution tests
   can never catch. From now on, every iter that touches buffer sizing must
   run the multi-resolution stress test, not just `render_fixed.py`.
