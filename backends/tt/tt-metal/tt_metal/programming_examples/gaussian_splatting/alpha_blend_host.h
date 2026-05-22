#pragma once
#include <cstdint>

namespace gsplat {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_BYTES_BF16 = TILE_H * TILE_W * 2;     // 2 KB
constexpr uint32_t SCALAR_PACK_BYTES = 10 * 4;                // 10 fp32 scalars (compute CB)
constexpr uint32_t SCALAR_PACK_PAGE_BYTES = 64;               // CB_SCALARS page (10 fp32 + pad)
constexpr uint32_t DYN_PACK_PAGE_BYTES = 32;                  // 6 fp32 basis coeffs per entry
constexpr uint32_t STATIC_COLOR_OPACITY_PAGE_BYTES = 32;      // 4 fp32 (R,G,B,opacity) per gid
constexpr uint32_t SORTED_GIDS_PAGE_BYTES = 64;                 // 16 uint32 gids per page
constexpr uint32_t META_PAGE_BYTES = 64;                        // padded uint32 page

// CB indices
constexpr uint32_t CB_PX         = 0;
constexpr uint32_t CB_PY         = 1;
constexpr uint32_t CB_SCALARS    = 2;
constexpr uint32_t CB_TILE_META  = 3;

// Per-tile basis tiles (computed once per screen tile from px/py)
constexpr uint32_t CB_PX2        = 4;   // px²
constexpr uint32_t CB_PY2        = 5;   // py²
constexpr uint32_t CB_PXPY       = 6;   // px·py
constexpr uint32_t CB_COEFF      = 7;   // 6 scalar-broadcast tiles per Gaussian
// Scratch CBs reused by Stage D2 (no overlap with per-tile basis CBs)
constexpr uint32_t CB_DXDY       = 8;   // D2 scratch R channel
constexpr uint32_t CB_Q          = 9;   // D2 scratch G channel
constexpr uint32_t CB_POWER      = 10;  // D2 scratch B channel
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

// IPC opcode for shared-memory handshake (iter 024).
// After READY + SCN1, Python can optionally send SHM1 to negotiate a
// POSIX shared-memory transport for frame data and image readback.
// The daemon responds with OK31 on success, ERR1 on failure.
// Python then sets the shm_flag field in subsequent FRM2 headers to 1
// to use the shared-memory path instead of the pipe.
constexpr uint32_t IPC_MAGIC_SHM1 = 0x53484D31;  // 'SHM1'

// Reader-only scratch CB: a dedicated L1 region used by the reader kernel
// for NoC-async-read destinations that need a stable, NoC-addressable
// location (the kernel stack lives in NCRISC IRAM and is NOT
// NoC-addressable; see watcher "Local L1 address overflow" if violated).
// depth=1, never push/pop — reader just reads `get_write_ptr` once and
// uses the L1 region for the per-frame inner-loop scratches:
//   [0,   64)  -> sorted_gids page cache (16 uint32 per page)
//   [64,  96)  -> static color/opacity scratch (8 fp32 per static gather)
//   [96, 128)  -> reserved
constexpr uint32_t CB_READER_SCRATCH = 24;
constexpr uint32_t READER_SCRATCH_PAGE_BYTES = 128;

// Sentinel-mask threshold: a pixel whose transmittance falls below this is
// "saturated" (further Gaussians contribute < 1/255 to it). Used by the Stage F
// sat_mask refresh to freeze saturated pixels in subsequent compositing steps.
constexpr float T_SAT_THRESHOLD = 1e-4f;

}  // namespace gsplat
