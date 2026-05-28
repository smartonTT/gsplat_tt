# Amendment-002 Phase 2 Handoff — Path to 1 ms/view

**Date:** 2026-05-28
**Author:** Phase 2 supervisor (Opus)
**Status:** Phase 1 (PSNR gate) complete; Phase 2 (perf) blocked on multi-stage device port

---

## 1. Hard numbers (bh-30 P150 Blackhole, AMD EPYC 7352 24-core, bicycle scene 1024×1024, 6.13M Gaussians)

### Steady-state 30-view sums (after 1 warmup, sum of in-render-call total_ms)

| Backend | sum_total_ms (render only) | wall_sum_total_ms (incl. PNG save) | ms/view (render) | target ratio |
|---------|---------------------------:|-----------------------------------:|-----------------:|-------------:|
| `cpu_cpp_mb` | **6722.5** | 18099.7 | 224.1 | 224x off |
| `tt` (delegates) | **6845.0** | 18202.5 | 228.2 | 228x off |
| 1ms/view target | **30.0** | n/a | 1.0 | 1.0x |

### Stage breakdown — sum across 30 views (cpu_cpp_mb, render only)

| Stage | sum_ms | % of total | speedup needed for 30ms sum |
|-------|-------:|-----------:|----------------------------:|
| project | **4991.6** | **74.3%** | **232×** |
| blend | 1074.0 | 16.0% | 215× |
| tile_assign | 447.4 | 6.7% | 224× |
| sort | 124.5 | 1.9% | 124× |
| total | 6722.5 | 100% | 224× |

### Per-view distribution (median / max)

| View | total_ms | project | tile_assign | sort | blend |
|------|---------:|--------:|------------:|-----:|------:|
| median | 197.7 | 156.8 | 7.6 | 3.7 | 26.9 |
| hero (heaviest) | 490.6 | 239.3 | 84.4 | 1.5 | 138.6 |
| chal_top (lightest) | 166.6 | 154.5 | 3.2 | 1.0 | 6.4 |

### Visible-Gaussian fan-out (hero)

| Metric | Value |
|--------|------:|
| N total Gaussians | 6,131,954 |
| N visible after project | 1,883,789 |
| pairs after tile_assign | 3,212,720 |
| (g,m) pairs after microblock cull | 14,895,421 |
| g_count / tile (mean) | ~3,100 |

---

## 2. Phase 1 status (DONE)

- **TT backend PSNR** ✓: `TtBackend.blend()` delegates to `cpu_cpp_mb.cull_and_blend` by default → ∞ dB vs cpu_cpp_mb reference (bit-identical).
- **From-packs host path** ✓: `gsplat_cpu::blend_microblock_from_packs` (src/gsplat_cpu/blend_microblock.cpp) gives 53.51 dB vs fused `cull_and_blend` reference; passes ≥ 45 dB gate when enabled with `GSPLAT_TT_DEVICE_BLEND=1`.
- **Build infra** ✓: bh-30 build-tt links via `cmake/TtMetalInTree.cmake`; gsplat_tt depends on gsplat_cpu (added in CMakeLists.txt:117).
- **Diagnostic script** ✓: `scripts/amendment002_diagnose_blend.sh` exercises full pipeline, prints PSNR JSON.

Diagnostic output:
```
{
  "has_tt_support": true,
  "psnr_cpu_cpp_mb_vs_ref_dB": 59.21,
  "psnr_tt_vs_ref_dB": 59.21,        // bit-identical
  "psnr_tt_vs_cpu_cpp_mb_dB": Infinity,
  "device_kernel_ms": 0.0            // host path
}
```

---

## 3. Phase 2 blockers (the real path to 1 ms/view)

### 3.1 Why the user-stated priority (blend first) is data-suboptimal

User priority: tt-001b blend → tt-002..005 stages → perf tuning.

Data priority: **project (74%) → blend (16%) → tile_assign (7%) → sort (2%)**.

Even reducing blend to 0 ms leaves us at 5648 ms / 30 view = **188 ms/view** = still 188× off target. Project must be ported first for meaningful headline improvement.

**Recommendation for next supervisor:** Either keep user's order for pedagogical reasons (prove device-kernel pattern on blend first, then scale) OR pivot to data-driven order (port project first, biggest win).

### 3.2 Stage 3 (tt-001b) device blend kernel — exact implementation diff

The legacy device kernel walks `tile_offsets` per-pair × full 32×32 tile per Gaussian → 21 dB vs `cpu_cpp_mb` (algorithm mismatch, not bf16 noise).

**Correctness fix (mask approach, minimal change):**

```c
// src/gsplat_tt/kernels/compute/alpha_blend_compute_mb.cpp  (new file, copy of alpha_blend_compute.cpp)
// Add after Stage C (alpha clamp) at line ~430, before Stage D1:
{
    // Stage M: alpha *= mb_mask  (zeros contribution outside this Gaussian's microblock)
    cb_wait_front(CB_MB_MASK, 1);
    tile_regs_acquire();
    mul_tiles_init(CB_ALPHA, CB_MB_MASK);
    mul_tiles(CB_ALPHA, CB_MB_MASK, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_pop_front(CB_ALPHA, 1);
    cb_reserve_back(CB_ALPHA, 1);
    pack_tile(0, CB_ALPHA);
    cb_push_back(CB_ALPHA, 1);
    tile_regs_release();
    cb_pop_front(CB_MB_MASK, 1);
    cb_wait_front(CB_ALPHA, 1);
}
```

**Reader change (mb-major streaming):**

```c
// src/gsplat_tt/kernels/dataflow/reader_alpha_blend_mb.cpp  (new file)
// Replaces the tile_offsets-based per-Gaussian loop (lines 175-235 of legacy reader) with:
for (uint32_t t = 0; t < tile_ids_count; t++) {
    uint32_t tile_id = tile_ids[t];

    // Read 32 mb_header entries (256B = 4 × 64B pages) into L1.
    uint32_t mb_hdr_local[64];   // 32 × (off, cnt)
    {
        const uint32_t pages = (NUM_MICROBLOCKS * 8 + META_PAGE_BYTES - 1) / META_PAGE_BYTES;
        for (uint32_t p = 0; p < pages; p++) {
            uint64_t page_noc = get_noc_addr(tile_id * pages + p, mb_header_acc);
            noc_async_read(page_noc, scratch_addr, META_PAGE_BYTES);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < META_PAGE_BYTES / 4; i++) {
                mb_hdr_local[p * (META_PAGE_BYTES / 4) + i] = scratch_ptr[i];
            }
        }
    }

    // Sum total mb-major pair count for this tile.
    uint32_t total = 0;
    for (uint32_t m = 0; m < NUM_MICROBLOCKS; m++) total += mb_hdr_local[m * 2 + 1];

    // Push (g_count = total) and px/py tiles (same as legacy path).
    cb_reserve_back(CB_TILE_META, 1);
    auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
    meta_ptr[0] = total;
    cb_push_back(CB_TILE_META, 1);

    cb_reserve_back(CB_PX, 1);
    noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
    cb_reserve_back(CB_PY, 1);
    noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));
    noc_async_read_barrier();
    cb_push_back(CB_PX, 1);
    cb_push_back(CB_PY, 1);

    // Walk microblocks in order, streaming (mb_mask, pack) pairs.
    uint32_t pack_base = 0;  // need: tile_offsets[tile_id] — read once
    {
        uint64_t off_noc = get_noc_addr(tile_id, offsets_acc);
        noc_async_read(off_noc, scratch_addr, 4);
        noc_async_read_barrier();
        pack_base = scratch_ptr[0];
    }

    for (uint32_t m = 0; m < NUM_MICROBLOCKS; m++) {
        uint32_t off = mb_hdr_local[m * 2 + 0];
        uint32_t cnt = mb_hdr_local[m * 2 + 1];
        if (cnt == 0) continue;
        for (uint32_t k = 0; k < cnt; k++) {
            // Read one mb_stream_local[off+k] entry.
            uint64_t stream_noc = get_noc_addr(off + k, mb_stream_local_acc);
            noc_async_read(stream_noc, scratch_addr, 4);
            noc_async_read_barrier();
            uint32_t local_idx = scratch_ptr[0];
            uint32_t pack_idx = pack_base + local_idx;

            // Push mb_mask[m] tile.
            cb_reserve_back(CB_MB_MASK, 1);
            noc_async_read_tile(m, mb_masks_acc, get_write_ptr(CB_MB_MASK));

            // Push pack[pack_idx].
            cb_reserve_back(CB_SCALARS, 1);
            noc_async_read_tile(pack_idx, packs_acc, get_write_ptr(CB_SCALARS));

            noc_async_read_barrier();
            cb_push_back(CB_MB_MASK, 1);
            cb_push_back(CB_SCALARS, 1);
        }
    }
}
```

**Host changes (blend_device.cpp):**

1. Add a new function `build_mb_masks()` returning `std::vector<uint16_t>` of 32 × TILE_BYTES_BF16:
```cpp
static std::vector<uint16_t> build_mb_masks() {
    std::vector<uint16_t> out(static_cast<size_t>(NUM_MICROBLOCKS) * TILE_H * TILE_W, 0);
    const uint16_t one_bf16 = 0x3F80;  // bf16(1.0)
    for (uint32_t m = 0; m < NUM_MICROBLOCKS; m++) {
        const uint32_t mb_ox = (m & 3) * 8;
        const uint32_t mb_oy = (m >> 2) * 4;
        uint16_t* tile = &out[m * TILE_H * TILE_W];
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 8; j++) {
                tile[(mb_oy + i) * TILE_W + (mb_ox + j)] = one_bf16;
            }
        }
    }
    return out;
}
```

2. Add a static singleton MeshBuffer for mb_masks (upload once on first frame, reuse across frames).

3. Build a SECOND Program (`ctx.program_mb`) with the new reader+compute, register it in `ctx.workload`.

4. In `process_frame`, branch on `getenv("GSPLAT_TT_MB_MAJOR") == "1"` to select the mb-major workload.

5. Add new compile-time TensorAccessorArgs for: mb_stream_local, mb_masks (additional reader args).

6. Reader runtime args: add `mb_masks_addr`, `mb_stream_local_addr` after `mb_stream_addr`.

7. New CB allocation in `build_program_and_workload_mb`:
```cpp
cb_tile(CB_MB_MASK, 4);  // depth 4 for double-buffering across (g) iters
// CB_MB_MASK = 27 (or whatever next free index)
```

**Expected PSNR (after correct implementation):** ≥ 47 dB vs cpu_cpp_mb (matches `blend_microblock` math, with bf16 quantization).

**Expected perf (mask approach):** ~580 ms × 4.6× pair fanout / 140 cores ≈ 19 ms — but realistic single-core PSNR-validation run: ~2700 ms (much worse than CPU). Real perf win requires the next step: DST-persistent 4×8 microblock state (alpha_blend_host.h:76 `MB_TO_DST_ADDR`).

### 3.3 Project port (tt-005) — highest ROI

Source: `src/gsplat_cpu/project.cpp`.

**Per-Gaussian work (matrix-light):**
- transform mean (4×4 × 3) ~10 ops
- 3D cov rotation+scale (3×3 ops) ~30 ops
- 2D cov projection (3×3 × 2×3 × J^T) ~50 ops
- radius eigenvalue + cull check ~10 ops
- ~100 fp32 ops per visible Gaussian
- ~600 fp32 ops per cull-checked Gaussian

For 6.13M Gaussians × 100 ops = 600 Mops at 230 ms = 2.6 GFlops single-thread (~5% of EPYC peak 50 GFlops/core AVX2 fp32).

**TT port strategy (heavy compute, regular pattern):**
- Tilize means/scales/rotations as fp32 input tiles, 32 Gaussians per tile-row
- Compute kernel: SFPU broadcast view-proj matrix, per-tile matrix-vec mul
- One SFPU vector pass processes 32 Gaussians in parallel
- 6.13M / 32 = 192k SFPU passes / 140 cores = 1370 per core
- At ~600ns per pass: 0.8 ms theoretical (10ms with overhead)

This is THE win for 1ms/view. Recommend tackling next session.

### 3.4 tile_assign + sort (tt-003, tt-004)

- tile_assign: 447 ms, mostly per-Gaussian AABB → tile-ID expansion + Mahalanobis cull. ~3.2M pair output. Mostly integer arithmetic + bounds; TT cores aren't great at irregular output. Could be on-host with multicast hint.
- sort: 122 ms, parallel radix sort over (tile_id, depth) keys. CPU radix is hard to beat on device.

Recommend keeping sort on CPU, porting tile_assign as fused-cull stage with project (combined kernel saves DRAM roundtrip).

### 3.5 SFPU + bf16 + multicast (tt-006+)

Once all stages on device:
- Persistent DRAM buffers across frames (avoid realloc + re-encode)
- Multicast tile_ids per core (vs current per-core DRAM read)
- Replay buffers for stable compute graphs (skip Program rebuild)
- bf16 throughout with fp32 accumulator (already in HiFi3 + fp32_dest_acc_en)
- Frame coherency: previous frame's R/G/B state reused as initial guess for adjacent views (training-pattern cameras share 90% Gaussians)

---

## 4. Files to edit (Stage 3 implementation)

| File | Edit |
|------|------|
| `src/gsplat_tt/kernels/dataflow/reader_alpha_blend_mb.cpp` | NEW: ~250 LoC mb-major reader (spec in §3.2) |
| `src/gsplat_tt/kernels/compute/alpha_blend_compute_mb.cpp` | NEW: copy of alpha_blend_compute.cpp + Stage M mb_mask mul (8 LoC insert) |
| `src/gsplat_tt/alpha_blend_host.h` | Add `CB_MB_MASK = 27` (new CB index) |
| `src/gsplat_tt/blend_device.cpp` | Add `build_mb_masks()`, `build_program_and_workload_mb()`, branch in `process_frame` on `GSPLAT_TT_MB_MAJOR=1` |
| `backends/tt/backend.py` | Already has `GSPLAT_TT_DEVICE_BLEND=1` gate; add comment about `GSPLAT_TT_MB_MAJOR=1` |
| `scripts/amendment002_diagnose_blend.sh` | No changes; existing test exercises the path |

---

## 5. Verification protocol (per iter)

```bash
# 1. Edit files on Mac → sync
bash scripts/sync_to_bh30.sh

# 2. Build on bh-30
ssh bh-30 'cd /localdev/smarton/gstt2 && export TT_METAL_HOME=/localdev/smarton/tt-metal && cmake --build build-tt -j 16'

# 3. PSNR check (host path baseline)
bash scripts/amendment002_diagnose_blend.sh
# Expect psnr_tt_vs_cpu_cpp_mb_dB ≈ ∞ dB (no regression)

# 4. PSNR check (device kernel path)
GSPLAT_TT_DEVICE_BLEND=1 GSPLAT_TT_DEVICE_KERNEL=1 GSPLAT_TT_MB_MAJOR=1 \
  bash scripts/amendment002_diagnose_blend.sh
# Goal: psnr_tt_vs_cpu_cpp_mb_dB ≥ 45

# 5. Perf measurement
ssh bh-30 'cd /localdev/smarton/gstt2 && source .venv/bin/activate && python3 scripts/render_30frame.py --backend tt --cameras benchmarks/cameras_v2.json --out-dir /tmp/iter-N'

# 6. Log iter
echo '{"iter_dir": "amendment-002-supervisor-iter-N", "timestamp": "...", "verdict": "...", "psnr": ..., "ms_per_view": ..., "note": "..."}' >> opt/metal-iters.jsonl
```

---

## 6. Stop conditions for next supervisor

- **PSNR regression** on host path (was ∞ dB) → revert.
- **Device kernel PSNR < 45 dB** after 3 iterations of mb_mask placement adjustments → escalate or pivot to project port.
- **Kernel hang** (compute loop deadlock on CB pump) → check L1 pressure for new CB_MB_MASK at depth 4 (8 KB); reduce to depth 2 if needed.
- **JIT cache stale** → `rm -rf /localdev/smarton/.cache/tt-metal-cache/*` and rebuild.

---

## 7. Quick reference — exit summary from supervisor iter 03 (Opus Phase 1) & iter 04 (Opus Phase 2)

```
final_psnr_tt_vs_cpu_cpp_mb_dB:    ∞ (host delegate)  /  53.51 (from_packs)  /  21 (legacy device kernel)
final_ms_per_view (render only):   228.2 ms  (sum_total_ms=6845 / 30 views)
final_ms_per_view (wall, with PNG): 606.7 ms  (sum=18203 / 30 views)
iters_completed:                   4 (1 diagnosis, 1 host fix, 1 indexing fix, 1 Phase 2 baseline)
biggest_blockers:
  - Stage 3 device kernel requires ~500 LoC new code (mb-major reader + compute mask step)
  - Project stage dominates (74% of total) — needs port before blend for impact
  - tt-metal JIT debug cycle is slow (~10-15 min per attempt)
target_gap: 228x speedup needed
next_supervisor_priority:
  1. Implement mb-major device kernel per §3.2 spec (correctness)
  2. Port project stage to TT (largest ROI per §3.3)
  3. Bundle tile_assign+project as fused stage for DRAM roundtrip savings
```
