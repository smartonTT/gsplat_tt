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

// Blend writer fully overwrites res_out each frame — skip the zero H2D.
inline constexpr bool blend_skip_zero_out_enabled() { return true; }

// M0: 32B per-entry record + pre-sized per-tile buckets.
inline constexpr bool l1_record_enabled() { return true; }         // L1_RECORD=1

// One-shot JIT compile of all device programs at scene open.
inline constexpr bool jit_warmup_enabled() { return true; }        // JIT_WARMUP=1

}  // namespace gsplat_tt::env_config
