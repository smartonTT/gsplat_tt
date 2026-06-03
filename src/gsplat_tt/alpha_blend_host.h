#pragma once
#include <cstdint>

namespace gsplat {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_BYTES_BF16 = TILE_H * TILE_W * 2;     // 2 KB
constexpr uint32_t TILE_BYTES_FP32 = TILE_H * TILE_W * 4;     // 4 KB
constexpr uint32_t SCALAR_PACK_BYTES = 9 * 4;                  // 9 fp32 scalars
constexpr uint32_t SCALAR_PACK_PAGE_BYTES = 64;                // padded for NoC alignment
constexpr uint32_t META_PAGE_BYTES = 64;                       // padded uint32 page

// CB indices
constexpr uint32_t CB_PX         = 0;
constexpr uint32_t CB_PY         = 1;
constexpr uint32_t CB_SCALARS    = 2;
constexpr uint32_t CB_TILE_META  = 3;

// Scratch CBs (per-Gaussian intermediate tiles; depth tuned for double-buffering)
constexpr uint32_t CB_DX         = 4;   // dx = px - mean_x
constexpr uint32_t CB_DY         = 5;   // dy = py - mean_y
constexpr uint32_t CB_DX2        = 6;   // dx²
constexpr uint32_t CB_DY2        = 7;   // dy²
constexpr uint32_t CB_DXDY       = 8;   // dx·dy
constexpr uint32_t CB_Q          = 9;   // [a·dx², c·dy², 2b·dx·dy] (3 tiles)
constexpr uint32_t CB_POWER      = 10;  // partial sum a·dx² + c·dy² before adding 2b·dx·dy
// CB 11 reserved (was CB_CONST_NEG88 for approx=false exp clamp; unused since
// switch to exp_tile<approx=true>). Slot kept to avoid renumbering downstream CBs.
constexpr uint32_t CB_ALPHA      = 12;  // alpha = min(0.99, opacity · exp(power))
constexpr uint32_t CB_CONTRIB    = 13;  // contrib = alpha · T_state · sat_mask
constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;  // (1 - alpha) for transmittance update
constexpr uint32_t CB_T_TMP      = 15;  // generic intermediate (D2 channel mul, E mul chain)

// Output CB: writer reads R/G/B tiles in batches of 3 per screen tile.
constexpr uint32_t CB_COLOR_OUT  = 16;

// Persistent per-tile running state (depth=1, swapped in-place each frame).
// These hold the alpha-blend accumulators across the per-Gaussian loop:
//   color_R/G/B_state: front-to-back composited color so far
//   T_state:           remaining transmittance per pixel (starts at 1.0)
//   sat_mask:          1.0 where T >= 1e-4, else 0.0 (refreshed every 16 Gaussians)
constexpr uint32_t CB_COLOR_R_STATE = 17;
constexpr uint32_t CB_COLOR_G_STATE = 18;
constexpr uint32_t CB_COLOR_B_STATE = 19;
constexpr uint32_t CB_T_STATE       = 20;
constexpr uint32_t CB_SAT_MASK      = 21;

// Pre-filled constant tiles (depth=1, never popped). Used by binary_min_tile
// against Dst slots when SFPU ops require a CB operand.
constexpr uint32_t CB_CONST_ZERO = 22;  // 0.0  (used to clamp power = min(power, 0))
constexpr uint32_t CB_CONST_099  = 23;  // 0.99 (used to clamp alpha = min(., 0.99))

// Sentinel-mask threshold: a pixel whose transmittance falls below this is
// "saturated" (further Gaussians contribute < 1/255 to it). Used by the Stage F
// sat_mask refresh to freeze saturated pixels in subsequent compositing steps.
constexpr float T_SAT_THRESHOLD = 1e-4f;

// ---------------------------------------------------------------------------
// Microblock-major blend (metal-iter-001) — host ↔ device agreement.
// See docs/optimization-log/microblock-kernel-design.md §3.4 + §6.3.
// ---------------------------------------------------------------------------
constexpr uint32_t NUM_MICROBLOCKS = 32;

constexpr uint32_t COEFF_LANES_PER_GAUSSIAN = 10;  // A..F, opacity, color_rgb
constexpr uint32_t COEFF_ROW_BYTES          = 48;  // 10 fp32 + 2 fp32 pad (NoC align)
constexpr uint32_t MB_HEADER_BYTES          = NUM_MICROBLOCKS * 8;  // 32 × (offset,count)

struct MicroblockHeader {
    uint32_t offset;
    uint32_t count;
};
static_assert(sizeof(MicroblockHeader) == 8, "8B per microblock header entry");

// Microblock m → DST addr offset within one 32×32 tile slot (§6.3 table).
constexpr uint32_t MB_TO_DST_ADDR[NUM_MICROBLOCKS] = {
    0,  2,  16, 18,  4,  6,  20, 22,
    8,  10, 24, 26, 12, 14, 28, 30,
    32, 34, 48, 50, 36, 38, 52, 54,
    40, 42, 56, 58, 44, 46, 60, 62,
};

// Planned CB indices when microblock reader lands (Stage 2+):
//   CB_COEFF_TABLE replaces CB_SCALARS
//   CB_MB_HEADER   replaces CB_TILE_META
//   CB_MB_STREAM   new
constexpr uint32_t CB_COEFF_TABLE = 2;
constexpr uint32_t CB_MB_HEADER   = 3;
constexpr uint32_t CB_MB_STREAM   = 4;

// Stage 2 shadow CBs (parallel legacy path; compute unchanged until Stage 3).
constexpr uint32_t CB_MB_COEFF_SHADOW  = 24;
constexpr uint32_t CB_MB_HEADER_SHADOW = 25;
constexpr uint32_t CB_MB_STREAM_SHADOW = 26;

}  // namespace gsplat
