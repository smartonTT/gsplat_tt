#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {
class ThreadPool;
}

namespace gsplat_tt {

// ---------------------------------------------------------------------------
// amendment-003 step 3: microblock (4x8) alpha-blend payload.
//
// Mirrors the per-(tile, microblock) structure the CPU cull_and_blend builds
// internally, but materialized into flat buffers + converted to the
// basis-form quadratic the SFPU kernel evaluates:
//
//   power = A*x^2 + B*xy + C*y^2 + D*x + E*y + F     (x,y tile-local + 0.5)
//   weight = exp(min(power, 0));  alpha = min(opacity*weight, 0.99)
//
// with the -0.5 already folded into A..F (host side), exactly as the kernel
// design spec (opt/microblock-kernel-design.md sections 3, 5, 6).
//
// Microblock enumeration is the canonical raster order shared by CPU, host
// and device: m = row_band*4 + col_group, row_band in [0,8) (4px rows),
// col_group in [0,4) (8px cols). Microblock m covers tile-local rows
// [row_band*4, +4), cols [col_group*8, +8).
// ---------------------------------------------------------------------------

constexpr int kMbNumMicroblocks = 32;
constexpr int kMbCoeffLanes = 10;  // A,B,C,D,E,F,opacity,cr,cg,cb
constexpr int kMbGmLanes = 16;     // gaussian-major row: 10 coeff + mask + 5 pad
// Per-gaussian attribute row for the device-cull path: raw cov {a,b,c}, image-
// space center {mean_x,mean_y}, opacity, color {r,g,b} = 9 fp32, padded to a
// 16-float (64B DRAM-aligned) page.
constexpr int kMbAttrLanes = 16;

struct MbPayload {
    // Per-tile coefficient table: one row of kMbCoeffLanes floats per local
    // gaussian of the tile (the tile's full depth-sorted list). Concatenated
    // across tiles; coeff_tile_off[t] is the starting ROW of tile t.
    std::vector<float> coeff;             // (sum_t L_t) * kMbCoeffLanes
    std::vector<uint32_t> coeff_tile_off; // num_tiles + 1, in rows

    // Per-tile microblock header: 32 (offset, count) pairs per tile, flat.
    // mb_header[(t*32 + m)*2 + 0] = offset into THIS TILE's mb_stream slice,
    // mb_header[(t*32 + m)*2 + 1] = count.
    std::vector<uint32_t> mb_header;      // num_tiles * 32 * 2

    // Per-tile microblock stream: flat list of local_gaussian_idx (into the
    // tile's coeff rows), microblock-major, depth-sorted within a microblock.
    std::vector<uint32_t> mb_stream;          // sum_t L'_t
    std::vector<uint32_t> mb_stream_tile_off; // num_tiles + 1

    int num_tiles = 0;
    int tiles_x = 0;
    int tiles_y = 0;
    int tile_size = 32;
    int64_t pairs_in = 0;
    int64_t pairs_dropped_all_mb = 0;
    int64_t pairs_kept_per_mb = 0;
};

// Build the microblock basis payload from projected gaussians + the
// depth-sorted per-tile lists (same inputs as gsplat_cpu::cull_and_blend).
MbPayload build_mb_payload(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t M,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int image_height,
    int image_width,
    float mb_contrib_floor,
    gsplat_cpu::ThreadPool& pool,
    bool cull_disabled);

// ---------------------------------------------------------------------------
// Fused gaussian-major payload (MB_KERNEL=1 path only).
//
// Computes the gaussian-major rows DIRECTLY during the microblock cull,
// replacing the build_mb_payload + separate gaussian-major stream-build two-pass
// pipeline with a single fused pass. The per-tile cull is bit-identical to
// build_tile (same det/conic/centered-form coeff math, opacity floor drop, bbox
// computation, and constrained-min Mahalanobis m2_min logic); instead of
// per-microblock keep lists it accumulates a 32-bit microblock-coverage mask per
// local gaussian and emits ONE 16-word row per surviving gaussian (mask != 0):
//   words[0..9]  = the 10 basis coeffs (A,B,C,mx_local,my_local,0,opacity,r,g,b)
//   word[10]     = mask (uint32 bit-cast into the float array)
//   words[11..15]= 0
// Rows are emitted in depth/sorted order (l = 0..L-1) per tile, tiles
// concatenated in tile order.
struct GaussianMajorPayload {
    // counts[t*32 + 0] = surviving gaussian-row count for tile t (rest 0).
    std::vector<uint32_t> mb_counts;       // num_tiles * 32
    // ROW offsets into mb_coeff_stream (in 16-word rows), num_tiles + 1.
    std::vector<uint32_t> mb_coeff_off;    // num_tiles + 1
    // Gaussian-major rows: total_rows * kMbGmLanes floats.
    std::vector<float> mb_coeff_stream;    // total_rows * 16

    int64_t pairs_in = 0;
    int64_t pairs_dropped_all_mb = 0;
    int64_t pairs_kept_per_mb = 0;
};

// Fused cull + gaussian-major emit. Same projected inputs as build_mb_payload.
//
// When device_conic is true (GSPLAT_TT_MB_DEVCONIC gate), words[0..2] of each
// emitted row carry the RAW 2D covariance {cov_a,cov_b,cov_c} instead of the
// precomputed conic A,B,C; the device compute kernel derives det + A,B,C from
// them at row load. The host-side microblock cull (which still runs here)
// continues to use the conic internally, so the kept set is unchanged.
GaussianMajorPayload build_gaussian_major_payload(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size,
    float mb_contrib_floor,
    gsplat_cpu::ThreadPool& pool,
    bool cull_disabled,
    bool device_conic = false);

// CPU reference blend FROM the payload (basis form). Validates the payload +
// basis math against cull_and_blend before the device kernel consumes the
// same buffers. Writes H*W*3 row-major fp32 into image_out (pre-zeroed).
void blend_from_mb_payload_cpu(
    const MbPayload& p,
    int image_height,
    int image_width,
    float transmittance_threshold,
    float* image_out);

}  // namespace gsplat_tt
