// SPDX-License-Identifier: Apache-2.0
//
// config.h — the baked-in production configuration for the clean renderer.
//
// The original pipeline selected its code path at runtime from ~40 GSPLAT_TT_*
// environment flags. This renderer hardcodes the single production
// configuration (the verify_cmd flag set that yields hero_vs_ref ~= 63.85 dB)
// as compile-time constants. There are no alternate paths, no CPU fallbacks and
// no debug toggles — what you read is exactly what runs.
//
// Production flag set (from ttw.toml verify_cmd):
//   BLEND_MODE=2  MB_KERNEL=1  DEVICE_PROJECT=1  RESIDENT_PROJECT=1
//   RESIDENT_GATHER=1  DEVICE_TILE_ASSIGN=1  RESIDENT_TA_IN=1  DEVICE_SORT=1
//   RESIDENT_PAIRS=1  RESIDENT_BLEND=1  SORT_DEVICE_PUBLISH=1  TA_DEVICE_SCAN=1
//   PROJ_DEVICE_SCAN=1  SFPU_CULL=1  TILE_BUCKET=1  BUCKET_FIT=8192
//   FUSED_TILE=0  L1_RECORD=1
//
// Dropped (dead) paths: BUCKET_MASK, SORT_DEVICE_LAYOUT, MB_DEVCULL (uploaded),
// BLEND_PAYLOAD, FUSED_TILE=1, CHUNK_FUSION, MB_BLEND_TAILSKIP,
// MB_BLEND_EARLYOUT, L1_SORT_VERIFY, all *_DEBUG/*_DUMP, MB_TILE_MASK_L1,
// L1_MASKS. (CULL_SPIN=512 and BUCKET_FIT=8192 are baked as kernel value
// defines, not dropped.)

#pragma once

#include <cstdint>

namespace render_config {

// Tile geometry (32x32 image tiles == one Tensix output tile).
inline constexpr uint32_t kTileSize = 32;

// L1-resident per-tile bucket capacity (GSPLAT_TT_BUCKET_FIT). Each tile owns a
// fixed-size bucket of 64B records; the SFPU cull + L1 radix sort + blend all
// operate inside this footprint.
inline constexpr uint32_t kBucketFit = 8192;

// Device microblock blend selector (GSPLAT_TT_BLEND_MODE). 2 == the TT device
// microblock-major SFPU kernel (the production blend).
inline constexpr int kBlendMode = 2;

// pfwc hardcodes the 3-sigma radius (k_cap == 3.0) and no isoellipse cull.
inline constexpr float kKCap = 3.0f;

}  // namespace render_config
