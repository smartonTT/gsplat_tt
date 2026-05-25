#pragma once
#include <cstdint>

namespace gsplat {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_BYTES_BF16 = TILE_H * TILE_W * 2;     // 2 KB
constexpr uint32_t SCALAR_PACK_BYTES = 10 * 4;                 // 10 fp32 scalars (M1 basis form)
constexpr uint32_t SCALAR_PACK_PAGE_BYTES = 64;                // padded for NoC alignment (10*4=40 < 64)
constexpr uint32_t META_PAGE_BYTES = 64;                       // padded uint32 page

// CB indices
constexpr uint32_t CB_PX         = 0;
constexpr uint32_t CB_PY         = 1;
constexpr uint32_t CB_SCALARS    = 2;
constexpr uint32_t CB_TILE_META  = 3;

// Scratch CBs (per-Gaussian intermediate tiles; depth tuned for double-buffering)
constexpr uint32_t CB_DX         = 4;   // (unused in M1 — was dx = px - mean_x)
constexpr uint32_t CB_DY         = 5;   // (unused in M1 — was dy = py - mean_y)
constexpr uint32_t CB_DX2        = 6;   // (unused in M1 — was dx²)
constexpr uint32_t CB_DY2        = 7;   // (unused in M1 — was dy²)
constexpr uint32_t CB_DXDY       = 8;   // (unused in M1 — was dx·dy)
constexpr uint32_t CB_Q          = 9;   // Q accumulator spill (depth 1)
constexpr uint32_t CB_POWER      = 10;  // power = -0.5 * Q
// CB 11 reserved (was CB_CONST_NEG88; unused since exp_tile<approx=true>).
constexpr uint32_t CB_ALPHA      = 12;  // alpha = min(0.99, opacity · exp(power))
constexpr uint32_t CB_CONTRIB    = 13;  // contrib = alpha · T_state · sat_mask
constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;  // (1 - alpha) for transmittance update
constexpr uint32_t CB_T_TMP      = 15;  // generic intermediate (D2 channel mul, E mul chain)

// Output CB: writer reads R/G/B tiles in batches of 3 per screen tile.
constexpr uint32_t CB_COLOR_OUT  = 16;

// Persistent per-tile running state (depth=1, swapped in-place each frame).
constexpr uint32_t CB_COLOR_R_STATE = 17;
constexpr uint32_t CB_COLOR_G_STATE = 18;
constexpr uint32_t CB_COLOR_B_STATE = 19;
constexpr uint32_t CB_T_STATE       = 20;
constexpr uint32_t CB_SAT_MASK      = 21;

// Pre-filled constant tiles (depth=1, never popped).
constexpr uint32_t CB_CONST_ZERO = 22;  // 0.0  (Q init and power clamp)
constexpr uint32_t CB_CONST_099  = 23;  // 0.99 (alpha clamp)

// M1 basis tiles: precomputed once per kernel invocation, never popped.
// tile-local coords: x in [0.5, 31.5] (j + 0.5), y in [0.5, 31.5] (i + 0.5).
// Used by the Q basis-form accumulation: Q = A*x² + B*xy + C*y² + D*x + E*y + F
constexpr uint32_t CB_BASIS_X2   = 24;  // x² = (j + 0.5)²
constexpr uint32_t CB_BASIS_XY   = 25;  // x*y = (j + 0.5)*(i + 0.5)
constexpr uint32_t CB_BASIS_Y2   = 26;  // y² = (i + 0.5)²
constexpr uint32_t CB_BASIS_X    = 27;  // x  = j + 0.5
constexpr uint32_t CB_BASIS_Y    = 28;  // y  = i + 0.5
constexpr uint32_t CB_BASIS_ONE  = 29;  // 1.0 (for the F constant term)

// Sentinel-mask threshold: a pixel whose transmittance falls below this is
// "saturated" (further Gaussians contribute < 1/255 to it). Used by the Stage F
// sat_mask refresh to freeze saturated pixels in subsequent compositing steps.
constexpr float T_SAT_THRESHOLD = 1e-4f;

}  // namespace gsplat
