// SPDX-License-Identifier: Apache-2.0
//
// env_config.h — the production render configuration, baked to compile-time
// constants.
//
// In the original pipeline these were runtime env-flag probes that selected
// between many alternate code paths. The clean renderer hardcodes the single
// production configuration (the verify_cmd flag set that yields
// hero_vs_ref ~= 63.85 dB). Every function below returns a fixed constant; the
// dead alternate paths they used to gate have been (or are being) removed. See
// config.h for the human-readable summary of this configuration.
//
// Kept as `namespace gsplat_tt::env_config` with the original function names so
// the stage drivers read as drop-in: `if (env_config::tile_bucket_enabled())`
// is now `if (true)`.

#pragma once

namespace gsplat_tt::env_config {

// Production gather path: TILE_BUCKET + SFPU cull_masks (NOT per-candidate
// FUSED_TILE gather).
inline constexpr bool fused_tile_enabled() { return false; }       // FUSED_TILE=0

// L1-resident full-record bucket scatter.
inline constexpr bool tile_bucket_enabled() { return true; }       // TILE_BUCKET=1

inline constexpr bool proj_device_scan_enabled() { return true; }  // PROJ_DEVICE_SCAN=1

inline constexpr bool chunk_fusion_enabled() { return false; }     // CHUNK_FUSION unset

inline constexpr bool bucket_cb_fence_enabled() { return true; }   // BUCKET_CB_FENCE default-on

inline constexpr bool resident_blend_enabled() { return true; }    // RESIDENT_BLEND=1

inline constexpr bool sfpu_cull_enabled() { return true; }         // SFPU_CULL=1

// Overlap blend host setup with the SFPU cull device window (same in-order CQ).
inline constexpr bool cull_pipeline_enabled() { return true; }

// Chain sort publish -> cull -> blend on one CQ drain.
inline constexpr bool sort_blend_pipe_enabled() { return true; }

// On-device bin histogram layout: OFF — production uses the HOST bin layout.
inline constexpr bool sort_device_layout_enabled() { return false; }

// Blend writer fully overwrites res_out each frame — skip the zero H2D.
inline constexpr bool blend_skip_zero_out_enabled() { return true; }

// M0: 32B per-entry record + pre-sized per-tile buckets.
inline constexpr bool l1_record_enabled() { return true; }         // L1_RECORD=1

// L1 radix bit-order proof: diagnostic only, OFF in production.
inline constexpr bool l1_sort_verify_enabled() { return false; }

// One-shot JIT compile of all device programs at scene open.
inline constexpr bool jit_warmup_enabled() { return true; }        // JIT_WARMUP=1

}  // namespace gsplat_tt::env_config
