// SPDX-License-Identifier: Apache-2.0
//
// env_config.h — the production render configuration, baked to compile-time
// constants.
//
// In the original pipeline these were runtime env-flag probes that selected
// between many alternate code paths. The clean renderer hardcodes the single
// production configuration (the verify_cmd flag set that yields
// hero_vs_ref ~= 63.85 dB). Every function below returns a fixed constant; the
// dead alternate paths they used to gate have been removed. See
// config.h for the human-readable summary of this configuration.
//
// Kept as `namespace gsplat_tt::env_config` with the original function names so
// the stage drivers read as drop-in: `if (env_config::tile_bucket_enabled())`
// is now `if (true)`.

#pragma once

namespace gsplat_tt::env_config {

// L1-resident full-record bucket scatter.
inline constexpr bool tile_bucket_enabled() { return true; }       // TILE_BUCKET=1

inline constexpr bool proj_device_scan_enabled() { return true; }  // PROJ_DEVICE_SCAN=1

inline constexpr bool chunk_fusion_enabled() { return false; }     // CHUNK_FUSION unset

// Overlap blend host setup with the SFPU cull device window (same in-order CQ).
inline constexpr bool cull_pipeline_enabled() { return true; }

// Chain sort publish -> cull -> blend on one CQ drain.
inline constexpr bool sort_blend_pipe_enabled() { return true; }

// On-device bin histogram layout: OFF (iter-127) — reverts to the host
// host_bin_layout_from_hist + build_lpt path (the pre-iter-121 working path).
// The on-device layout (S5.1-S5.5, sort_bin_layout.cpp / sort_bin_emit.cpp) was
// justified SOLELY as a Metal Trace prerequisite; iter-126 MEASURED the trace
// endgame to be a no-go (replay removes <0.5ms; the ~92ms BRISC-FW is on-device
// firmware+NCRISC dataflow that trace replays unchanged), so the single-core
// device layout is pure regression (~+10-16ms/view vs the iter-116 baseline).
// Disabled to recover the frame. The on-device code is KEPT (gated off) — it may
// matter for a future fusion lever (see opt/sort-l1-resident-plan.md S5.x).
inline constexpr bool sort_device_layout_enabled() { return false; }

// S5.3 (host-free M/P): over-provision the M-domain (tile_assign K1/scan) to the
// static padded_n ceiling (= proj_m_* capacity, host-known, view-independent) and
// the P-domain (tile_assign pair buffers + sort bin work-split) to a static
// P_max ceiling. The kernels read the REAL M / P from the resident proj_M /
// ta_pairs_P control pages and guard every loop/work-split so the over-provisioned
// launches are exact no-ops beyond the real count. This DELETES the three
// mid-frame host blocking reads that previously sized the next dispatch (gather
// M-read, tile_assign P-read, sort P-read) — trace prerequisite (S5.6). Expected
// frame-neutral (iter-120: removing host drains is re-imposed by the in-order CQ;
// only Metal Trace removes the launch overhead). Bit-identical: the resident
// M/P the kernels read == the values the host args carried.
inline constexpr bool host_free_mp_enabled() { return false; }

// Static P-domain ceiling (pairs). Σ tiles-per-gaussian (pre-cull P) over the
// FIXED 30-view bicycle bench peaks at 3,700,450 (measured, iter-123). 4,718,592
// (= 4608*1024, 16-aligned) gives ~27% margin — a safe worst-case bound for the
// pair buffers (gid/tid/keep) + the sort bin work-split. Too small = overflow/
// corruption; too large = wasted DRAM + no-op work, so the margin is bounded.
inline constexpr unsigned int pair_ceiling() { return 4718592u; }

// M0: 32B per-entry record + pre-sized per-tile buckets.
inline constexpr bool l1_record_enabled() { return true; }         // L1_RECORD=1

// One-shot JIT compile of all device programs at scene open.
inline constexpr bool jit_warmup_enabled() { return true; }        // JIT_WARMUP=1

}  // namespace gsplat_tt::env_config

#include <cstdio>
#include <cstdlib>

namespace gsplat_tt::env_config {

// Task #100 sort_bucket_emit knobs (runtime env, read once; see sort_bin.cpp).
inline unsigned int env_uint(const char* name, unsigned int dflt) {
    const char* e = std::getenv(name);
    return (e != nullptr && *e != '\0') ? static_cast<unsigned int>(std::atoi(e)) : dflt;
}
// Pair pages per read batch (1 = one page per barrier, no prefetch; 2..16 =
// batched and double-buffered). Default 8 (t100 A/B winner).
inline unsigned int emit_pair_batch() {
    static const unsigned int v = [] {
        const unsigned int b = env_uint("GSPLAT_TT_EMIT_PB", 8u);
        return (b >= 1u && b <= 16u) ? b : 8u;
    }();
    return v;
}
// Records per per-tile L1 staging run (0 = one 32 B write per record; else a
// power of two <= 8 that divides the 64-record DRAM page). Default 8. 16 is
// capped to 8: its cb(15) ring (~520 KB per mover) overflows L1 with dual movers.
inline unsigned int emit_ring() {
    static const unsigned int v = [] {
        const unsigned int r = env_uint("GSPLAT_TT_EMIT_RING", 8u);
        if (r == 16u) {
            std::fprintf(stderr, "gsplat_tt: GSPLAT_TT_EMIT_RING=16 overflows L1; using 8\n");
            return 8u;
        }
        return (r == 2u || r == 4u || r == 8u) ? r : 0u;
    }();
    return v;
}
// The gather publishes the packed op/color words (blendrec[10], [11]) and the
// depth key (blendrec[12]); the emit copies them instead of packing them and
// writing them back, and reads no depth pages. Default on; 0 = legacy.
inline bool emit_puboc() {
    static const bool v = env_uint("GSPLAT_TT_EMIT_PUBOC", 1u) != 0u;
    return v;
}

// Task #124 one-launch sort v2 knobs (GSPLAT_TT_SORT_ONELAUNCH on, the default; see
// sort_bin_onelaunch.cpp and docs/sort-onelaunch-v2-t124.md). Kill switch for
// the v1 one-launch: GSPLAT_TT_OL_PB=1 GSPLAT_TT_OL_RING=0 GSPLAT_TT_OL_MAT_SELECT=0.
// Emit pair pages per batch: 1, 2, 4, 8 or 16. Default 8 (#100's EMIT_PB).
inline unsigned int ol_pair_batch() {
    static const unsigned int v = [] {
        const unsigned int b = env_uint("GSPLAT_TT_OL_PB", 8u);
        return (b != 0u && b <= 16u && (b & (b - 1u)) == 0u) ? b : 8u;
    }();
    return v;
}
// Emit records per per-tile run: 0 (one 32 B write per record), 2, 4 or 8.
// Default 8 (#100's EMIT_RING).
inline unsigned int ol_ring() {
    static const unsigned int v = [] {
        const unsigned int r = env_uint("GSPLAT_TT_OL_RING", 8u);
        return (r == 2u || r == 4u || r == 8u) ? r : 0u;
    }();
    return v;
}
// Big-tile materialize items sort only the depth bins holding their own ranks
// (sort_onelaunch_algo.h select_ranks), not the whole tile. Default off: task
// #121 measured it 0.15 ms/view slower on bicycle (24.69 vs 24.55, 3 rounds,
// yyzo-bh-07); select_ranks costs ~1.0 ms per 4096-record part.
inline bool ol_mat_select() {
    static const bool v = env_uint("GSPLAT_TT_OL_MAT_SELECT", 0u) != 0u;
    return v;
}
// Pair pages per mover kept in L1 from count to emit, a multiple of 32. Default
// 1536 (v1); 1024 with the emit rings, whose ~260 KB per mover must fit too.
inline unsigned int ol_win_pages() {
    static const unsigned int v = [] {
        const unsigned int d = ol_ring() != 0u ? 1024u : 1536u;
        const unsigned int w = env_uint("GSPLAT_TT_OL_WIN_PAGES", d);
        return (w >= 32u && w <= 2048u && w % 32u == 0u) ? w : d;
    }();
    return v;
}
// Task #202 (docs/emit-trisc-own-t200): the emit's per-record loop runs on the
// 3 TRISCs, tile-owned (sort_ol_town_compute.cpp). Needs the rings. Default on
// (gated -0.96 ms/view); GSPLAT_TT_OL_EMIT_TOWN=0 is the kill switch, same
// output either way.
inline bool ol_emit_town() {
    static const bool v = env_uint("GSPLAT_TT_OL_EMIT_TOWN", 1u) != 0u && ol_ring() != 0u;
    return v;
}


// Task #188 blend claim knobs (docs/blend-tail-t183). Late claim: the blend
// reader waits for a free bulk ring slot before it takes the next tile from the
// shared counter, so a core holds at most 2 tiles (compute + ring) instead of 3.
// Default on; 0 = claim as soon as the previous subchunk is pushed.
inline bool blend_late_claim() {
    static const bool v = env_uint("GSPLAT_TT_BLEND_LATE_CLAIM", 1u) != 0u;
    return v;
}
// Blend per-core lists dealt round-robin in record-count-descending order (core c
// gets ranks c, c+n, ...), so the reader's rank interleave claims tiles in
// global descending order. Default on; 0 = LPT lists (build_lpt).
inline bool blend_claim_desc() {
    static const bool v = env_uint("GSPLAT_TT_BLEND_CLAIM_DESC", 1u) != 0u;
    return v;
}

// Task #198 bridge hiding (docs/two-cq-t198). Early sort: the one-launch sort
// is enqueued right behind the fused K2, before the host knows P; the kernel
// reads P from ta_pairs_P and takes the K2's own page ranges, so the fold holds
// by construction. The host reads proj_M without draining the queue, skips the
// sort's P read and the M-float depths vector. Default on (task #198: with
// GSPLAT_TT_MAT_CQ1, -0.46 ms/view paired on yyzo-bh-07); 0 = off (also turns
// off GSPLAT_TT_MAT_CQ1).
inline bool sort_ol_early() {
    static const bool v = env_uint("GSPLAT_TT_SORT_OL_EARLY", 1u) != 0u;
    return v;
}
// Second command queue (needs sort_ol_early): CQ1 waits for the K2, reads
// proj_M and the K2 count rows; the host sums the bucket totals from the rows
// and uploads the bin layout and the mat work list on CQ1 while the sort runs
// on CQ0, then enqueues mat on CQ0 behind it. The device opens with 2 CQs.
// Default on; 0 = off.
inline bool mat_cq1() {
    static const bool v = sort_ol_early() && env_uint("GSPLAT_TT_MAT_CQ1", 1u) != 0u;
    return v;
}

}  // namespace gsplat_tt::env_config
