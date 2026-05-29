#include "gsplat_tt/render_blend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "gsplat_cpu/cull_and_blend.h"
#include "gsplat_tt/blend.h"
#include "gsplat_tt/mb_payload.h"

namespace gsplat_tt {

namespace {

// Build the legacy 9-float-per-entry kernel payload (packs/offsets/px/py) the
// existing on-device blend kernel (blend_device.cpp process_frame) consumes,
// from the projected gaussians + depth-sorted per-tile lists. This is the
// plan-sanctioned full-32x32-tile device blend baseline (tt-001a): correct,
// proven on-device, ~47 dB. The mb-major 4x8 fp32 kernel supersedes it for
// the >=60 dB gate.
struct KernelInputs {
    std::vector<float> packs;    // P * 9
    std::vector<float> offsets;  // num_tiles + 1
    std::vector<float> px;       // num_tiles * 32 * 32
    std::vector<float> py;       // num_tiles * 32 * 32
};

KernelInputs build_kernel_inputs(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size) {
    const int num_tiles = tiles_x * tiles_y;
    KernelInputs k;
    k.packs.assign(P * 9, 0.0f);

    // Means are stored TILE-LOCAL (mean - tile_origin) and px/py are also
    // tile-local (0..31). The kernel stores px/py + mean as bf16; global
    // coords (up to ~5000 for bicycle) round to multiples of 16-32 in bf16,
    // destroying per-pixel dx. Tile-local coords stay in [0,32) where bf16 is
    // sub-0.25 accurate. dx = px_local - mean_local == global dx.
    for (int t = 0; t < num_tiles; ++t) {
        const int64_t start = tile_ranges[static_cast<std::size_t>(t) * 2 + 0];
        const int64_t end = tile_ranges[static_cast<std::size_t>(t) * 2 + 1];
        const float ox = static_cast<float>((t % tiles_x) * tile_size);
        const float oy = static_cast<float>((t / tiles_x) * tile_size);
        for (int64_t e = start; e < end; ++e) {
            const int64_t g = sorted_gaussian_ids[e];
            const float a = covs_2d[static_cast<std::size_t>(g) * 4 + 0];
            const float b = covs_2d[static_cast<std::size_t>(g) * 4 + 1];
            const float c = covs_2d[static_cast<std::size_t>(g) * 4 + 3];
            const float det = std::max(a * c - b * b, 1e-6f);
            const float ci_a = c / det;
            const float ci_b = -b / det;
            const float ci_c = a / det;
            float* row = &k.packs[static_cast<std::size_t>(e) * 9];
            row[0] = means_2d[static_cast<std::size_t>(g) * 2 + 0] - ox;
            row[1] = means_2d[static_cast<std::size_t>(g) * 2 + 1] - oy;
            row[2] = ci_a;
            row[3] = 2.0f * ci_b;
            row[4] = ci_c;
            row[5] = colors[static_cast<std::size_t>(g) * 3 + 0];
            row[6] = colors[static_cast<std::size_t>(g) * 3 + 1];
            row[7] = colors[static_cast<std::size_t>(g) * 3 + 2];
            row[8] = opacities[static_cast<std::size_t>(g)];
        }
    }

    k.offsets.assign(static_cast<std::size_t>(num_tiles) + 1, 0.0f);
    for (int t = 0; t < num_tiles; ++t) {
        k.offsets[static_cast<std::size_t>(t)] =
            static_cast<float>(tile_ranges[static_cast<std::size_t>(t) * 2 + 0]);
    }
    k.offsets[static_cast<std::size_t>(num_tiles)] = static_cast<float>(P);

    // Tile-local px/py grid is identical for every tile.
    const std::size_t ppt = static_cast<std::size_t>(tile_size) * tile_size;
    k.px.assign(static_cast<std::size_t>(num_tiles) * ppt, 0.0f);
    k.py.assign(static_cast<std::size_t>(num_tiles) * ppt, 0.0f);
    for (int t = 0; t < num_tiles; ++t) {
        float* pxt = &k.px[static_cast<std::size_t>(t) * ppt];
        float* pyt = &k.py[static_cast<std::size_t>(t) * ppt];
        for (int r = 0; r < tile_size; ++r) {
            for (int c = 0; c < tile_size; ++c) {
                const std::size_t i = static_cast<std::size_t>(r) * tile_size + c;
                pxt[i] = static_cast<float>(c) + 0.5f;
                pyt[i] = static_cast<float>(r) + 0.5f;
            }
        }
    }
    return k;
}

}  // namespace

gsplat_cpu::CullAndBlendResult render_blend_tt(
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
    float* image_out_external,
    bool cull_disabled,
    float transmittance_threshold,
    int blend_mode) {
    gsplat_cpu::CullAndBlendResult result;
    result.pairs_in = static_cast<int64_t>(P);

    // --- Synthetic single-splat debug injection (GSPLAT_TT_SYNTH=1) ---
    // Replaces the projected gaussians with ONE small gaussian in tile 0 so the
    // expected output is trivial to reason about (a small blob in the top-left,
    // everything else black). The identical tiny payload feeds both the device
    // mb kernel (GSPLAT_TT_MB_KERNEL=1) and the CPU oracle (GSPLAT_TT_DEVICE_BLEND=0),
    // isolating device-kernel bugs from the projection/sort/cull pipeline.
    //   GSPLAT_TT_SYNTH_X / _Y   : center in image pixels (default 4, 2)
    //   GSPLAT_TT_SYNTH_VAR      : isotropic 2D variance in px^2 (default 2.0)
    std::vector<float> syn_means, syn_covs, syn_colors, syn_op;
    std::vector<int64_t> syn_ids, syn_ranges;
    if (const char* s = std::getenv("GSPLAT_TT_SYNTH"); s != nullptr && s[0] == '1') {
        const int num_tiles = tiles_x * tiles_y;
        float cx = 4.0f, cy = 2.0f, var = 2.0f;
        if (const char* e = std::getenv("GSPLAT_TT_SYNTH_X")) cx = static_cast<float>(std::atof(e));
        if (const char* e = std::getenv("GSPLAT_TT_SYNTH_Y")) cy = static_cast<float>(std::atof(e));
        if (const char* e = std::getenv("GSPLAT_TT_SYNTH_VAR")) var = static_cast<float>(std::atof(e));
        syn_means  = {cx, cy};
        syn_covs   = {var, 0.0f, 0.0f, var};  // a, b, (unused), c
        syn_colors = {1.0f, 0.0f, 0.0f};      // red
        syn_op     = {0.9f};
        syn_ids    = {0};
        syn_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);
        syn_ranges[1] = 1;  // tile 0 owns gaussian 0: [0, 1); all other tiles empty
        means_2d = syn_means.data();
        covs_2d = syn_covs.data();
        colors = syn_colors.data();
        opacities = syn_op.data();
        sorted_gaussian_ids = syn_ids.data();
        tile_ranges = syn_ranges.data();
        M = 1;
        P = 1;
        result.pairs_in = 1;
    }

    float* image_out = image_out_external;
    if (image_out == nullptr) {
        result.image.assign(static_cast<std::size_t>(image_height) *
                                static_cast<std::size_t>(image_width) * 3,
                            0.0f);
        image_out = result.image.data();
    } else {
        std::memset(image_out, 0,
                    static_cast<std::size_t>(image_height) *
                        static_cast<std::size_t>(image_width) * 3 * sizeof(float));
    }

    if (blend_mode == 1 /* CPU reference blend from basis payload */) {
        MbPayload p = build_mb_payload(
            means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
            tile_ranges, M, P, tiles_x, tiles_y, tile_size, image_height,
            image_width, mb_contrib_floor, pool, cull_disabled);
        blend_from_mb_payload_cpu(p, image_height, image_width,
                                  transmittance_threshold, image_out);
        result.pairs_dropped_all_mb = p.pairs_dropped_all_mb;
        result.pairs_kept_per_mb = p.pairs_kept_per_mb;
        return result;
    }

    if (blend_mode == 2) {
        MbPayload p = build_mb_payload(
            means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
            tile_ranges, M, P, tiles_x, tiles_y, tile_size, image_height,
            image_width, mb_contrib_floor, pool, cull_disabled);
        result.pairs_dropped_all_mb = p.pairs_dropped_all_mb;
        result.pairs_kept_per_mb = p.pairs_kept_per_mb;

        // Microblock-major (4x8) device kernel path (amendment-003 step 3).
        // Opt in with GSPLAT_TT_MB_KERNEL=1. De-references the per-tile mb_stream
        // into a flat microblock-major coeff stream so the device reader streams
        // it linearly; the compute kernel processes one 4x8 microblock at a time.
        const char* mb_kernel = std::getenv("GSPLAT_TT_MB_KERNEL");
        if (mb_kernel != nullptr && mb_kernel[0] == '1') {
            const int num_tiles = tiles_x * tiles_y;
            std::vector<uint32_t> mb_counts(static_cast<std::size_t>(num_tiles) * 32, 0);
            std::vector<uint32_t> mb_coeff_off(static_cast<std::size_t>(num_tiles) + 1, 0);
            std::vector<float> mb_coeff_stream;
            mb_coeff_stream.reserve(static_cast<std::size_t>(p.pairs_kept_per_mb) * 10);
            uint32_t row = 0;
            for (int t = 0; t < num_tiles; ++t) {
                mb_coeff_off[static_cast<std::size_t>(t)] = row;
                const uint32_t coeff_base = p.coeff_tile_off[static_cast<std::size_t>(t)];
                const uint32_t stream_base = p.mb_stream_tile_off[static_cast<std::size_t>(t)];
                const uint32_t* hdr = &p.mb_header[static_cast<std::size_t>(t) * 32 * 2];
                for (int m = 0; m < 32; ++m) {
                    const uint32_t off = hdr[m * 2 + 0];
                    const uint32_t cnt = hdr[m * 2 + 1];
                    mb_counts[static_cast<std::size_t>(t) * 32 + m] = cnt;
                    for (uint32_t i = 0; i < cnt; ++i) {
                        const uint32_t lidx = p.mb_stream[stream_base + off + i];
                        const float* src =
                            &p.coeff[static_cast<std::size_t>(coeff_base + lidx) * 10];
                        mb_coeff_stream.insert(mb_coeff_stream.end(), src, src + 10);
                        ++row;
                    }
                }
            }
            mb_coeff_off[static_cast<std::size_t>(num_tiles)] = row;

            if (const char* d = std::getenv("GSPLAT_TT_MB_DUMP"); d && d[0] == '1') {
                std::fprintf(stderr, "[MB_DUMP] total_rows=%u coeff_off[0..2]=%u,%u,%u\n",
                             row, mb_coeff_off[0], mb_coeff_off.size() > 1 ? mb_coeff_off[1] : 0,
                             mb_coeff_off.size() > 2 ? mb_coeff_off[2] : 0);
                std::fprintf(stderr, "[MB_DUMP] tile0 counts:");
                for (int m = 0; m < 32; ++m) std::fprintf(stderr, " %u", mb_counts[m]);
                std::fprintf(stderr, "\n");
                for (uint32_t r2 = 0; r2 < row && r2 < 6; ++r2) {
                    std::fprintf(stderr, "[MB_DUMP] row%u: A=%.4f B=%.4f C=%.4f mx=%.3f my=%.3f f=%.1f op=%.3f rgb=%.3f,%.3f,%.3f\n",
                                 r2, mb_coeff_stream[r2*10+0], mb_coeff_stream[r2*10+1], mb_coeff_stream[r2*10+2],
                                 mb_coeff_stream[r2*10+3], mb_coeff_stream[r2*10+4], mb_coeff_stream[r2*10+5],
                                 mb_coeff_stream[r2*10+6], mb_coeff_stream[r2*10+7], mb_coeff_stream[r2*10+8], mb_coeff_stream[r2*10+9]);
                }
            }

            std::vector<float> img;
            blend_mb_from_payload(mb_counts, mb_coeff_off, mb_coeff_stream,
                                  num_tiles, tiles_x, image_height, image_width, img);
            const std::size_t n = static_cast<std::size_t>(image_height) *
                                  static_cast<std::size_t>(image_width) * 3;
            if (img.size() == n) {
                std::memcpy(image_out, img.data(), n * sizeof(float));
            }
            return result;
        }

        // blend_mode=2 runs the TT DEVICE blend by default (DST-persistent fp32
        // full-tile kernel: 56-64 dB vs cpu_cpp_mb, runs entirely on-device per
        // the "no CPU/Python in the render loop" goal). Set GSPLAT_TT_DEVICE_BLEND=0
        // to fall back to the C++ CPU basis oracle (validation only).
        const char* dev_blend = std::getenv("GSPLAT_TT_DEVICE_BLEND");
        const bool use_cpu_oracle = dev_blend != nullptr && dev_blend[0] == '0';
        if (!use_cpu_oracle) {
            KernelInputs k = build_kernel_inputs(
                means_2d, covs_2d, colors, opacities, sorted_gaussian_ids,
                tile_ranges, P, tiles_x, tiles_y, tile_size);
            setenv("GSPLAT_TT_DEVICE_KERNEL", "1", 1);
            std::vector<float> img;
            std::vector<float> empty_coeff;
            std::vector<uint32_t> empty_u32;
            blend_from_payload(k.packs, k.offsets, k.px, k.py, empty_coeff,
                               empty_u32, empty_u32, image_height, image_width,
                               img);
            const std::size_t n = static_cast<std::size_t>(image_height) *
                                  static_cast<std::size_t>(image_width) * 3;
            if (img.size() == n) {
                std::memcpy(image_out, img.data(), n * sizeof(float));
            }
        } else {
            blend_from_mb_payload_cpu(p, image_height, image_width,
                                      transmittance_threshold, image_out);
        }
        return result;
    }

    // Fallback: bit-identical CPU cull+blend.
    return gsplat_cpu::cull_and_blend(
        means_2d, covs_2d, colors, opacities, sorted_gaussian_ids, tile_ranges,
        M, P, tiles_x, tiles_y, tile_size, image_height, image_width,
        mb_contrib_floor, pool, image_out_external, cull_disabled,
        transmittance_threshold);
}

}  // namespace gsplat_tt
