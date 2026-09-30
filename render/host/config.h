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

// iter-138 (Stage-2b overflow pre-pack): the largest overflow tile (count >
// kBucketFit) the materialize kernel will pre-pack + L1-radix in-place (the
// "coalesced bucket read + depth-permute" path that replaces the random
// blendrec gather). Tiles whose count exceeds this cap fall back to the
// existing per-subchunk blendrec gather.
//
// Sized to the EXISTING materialize CB budget so the L1 footprint barely moves:
// the in-budget path's CB_BUCKET is already allocated at kBucketFit*64 B = 512 KB
// but only ever uses kBucketFit*32 B (PACK2: kBucketFit/2 pages * 64 B) — 2x slack.
// cap = 2*kBucketFit makes the overflow bucket read reuse that exact 512 KB with
// ZERO CB_BUCKET growth; only CB_BSORT grows ((2*cap+256)*4 = 129 KB vs 65 KB).
// CB_SLAB stays kBucketFit*32 (the sorted slab is streamed to DRAM one subchunk
// at a time). Total materialize CB footprint ~0.9 MB — safely within L1.
// Overflow tiles with count > cap (the few heaviest, max_tile_n peaks ~26.5k)
// keep the legacy blendrec gather.
inline constexpr uint32_t kOverflowL1Cap = 2u * kBucketFit;  // 16384

// Device microblock blend selector (GSPLAT_TT_BLEND_MODE). 2 == the TT device
// microblock-major SFPU kernel (the production blend).
inline constexpr int kBlendMode = 2;

// pfwc hardcodes the 3-sigma radius (k_cap == 3.0) and no isoellipse cull.
inline constexpr float kKCap = 3.0f;

}  // namespace render_config
