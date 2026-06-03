// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-process host driver for gsplat_tt blend (amendment-002 tt-001a).
// Refactored from tt-metal alpha_blend.cpp: no daemon, no .npy IPC.

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/kernel_types.hpp"

#include "env_config.h"
#include "host_tracy.hpp"

#include "alpha_blend_host.h"
#include "blend.h"
#include "config.h"
#include "device_state.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace gsplat;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static std::vector<float> bf16_tile_to_fp32(const uint16_t* src) {
    std::vector<float> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &u, 4);
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Device + program reusable context
// ---------------------------------------------------------------------------

struct DeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle reader{};
    KernelHandle compute{};
    KernelHandle writer{};
    // Full compute grid (e.g. 8x8 on Wormhole N150 after harvesting).
    // CBs and kernels are allocated on every core in this range at init;
    // per-frame, split_work_to_cores carves [0, num_tiles) into a contiguous
    // slice per core via SetRuntimeArgs.
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    // RESIDENT blend scratch (GSPLAT_TT_RESIDENT_BLEND): persistent per-context
    // DRAM buffers reused across frames (the render intermediates themselves —
    // attrs/ids — live in device_state and are read over NoC, not reallocated).
    // xramp/yramp are constant; upload once. out/tile_ids grow on demand.
    std::shared_ptr<distributed::MeshBuffer> res_xramp;
    std::shared_ptr<distributed::MeshBuffer> res_yramp;
    std::shared_ptr<distributed::MeshBuffer> res_out;
    std::shared_ptr<distributed::MeshBuffer> res_tile_ids;
    uint32_t res_out_tiles = 0;
    size_t res_tile_ids_bytes = 0;
    bool res_ramp_uploaded = false;

    // SFPU microblock-cull (GSPLAT_TT_SFPU_CULL) persistent scratch: the
    // per-candidate 32-bit mask buffer (grow-on-demand) + a one-time-uploaded
    // flag for the box-origin ramps (res_xramp/res_yramp reused as box ramps in
    // the cull context).
    std::shared_ptr<distributed::MeshBuffer> res_masks;
    size_t res_masks_bytes = 0;
    // Per-tile PAGE-ALIGNED base offset (in mask elements) into res_masks. Cull
    // writer + blend reader index masks by cull_base[tile]+pos so every DRAM
    // write lands on a 16-element (64B) boundary.
    std::shared_ptr<distributed::MeshBuffer> res_mask_base;
    size_t res_mask_base_bytes = 0;

    // Post-sort subchunk table (iter 48): per-tile (num_subchunks, count) u32
    // pairs built on the host from sort_tile_ranges and uploaded once per frame.
    // Device readers index blend_subchunk_meta[tile_id*2] to dispatch cull/blend
    // in depth-ordered slices of at most kBucketFit candidates.
    std::shared_ptr<distributed::MeshBuffer> res_subchunk_meta;
    size_t res_subchunk_meta_bytes = 0;
};

// Host-built subchunk plan + [SUBCHUNK] stats (iter 48).
struct SubchunkPlan {
    std::vector<uint32_t> meta;  // num_tiles * 2: [num_subchunks, candidate_count]
    uint32_t tiles_split = 0;
    uint32_t total_subchunks = 0;
    uint32_t max_subchunks_per_tile = 0;
    uint64_t total_overflow_candidates = 0;
};

static SubchunkPlan build_subchunk_plan(
    distributed::MeshCommandQueue* cq, uint32_t num_tiles) {
    namespace ds = gsplat_tt::device_state;
    SubchunkPlan plan;
    plan.meta.assign(static_cast<std::size_t>(num_tiles) * 2u, 0u);
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    if (!buf_rng || num_tiles == 0) {
        return plan;
    }
    std::vector<uint32_t> ranges(static_cast<std::size_t>(num_tiles) * 2u, 0u);
    distributed::EnqueueReadMeshBuffer(*cq, ranges, buf_rng, /*blocking=*/true);
    constexpr uint32_t kFit = render_config::kBucketFit;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t start = ranges[static_cast<std::size_t>(t) * 2u + 0u];
        const uint32_t end = ranges[static_cast<std::size_t>(t) * 2u + 1u];
        const uint32_t count = end - start;
        const uint32_t num_sc =
            count == 0u ? 1u : (count + kFit - 1u) / kFit;
        plan.meta[static_cast<std::size_t>(t) * 2u + 0u] = num_sc;
        plan.meta[static_cast<std::size_t>(t) * 2u + 1u] = count;
        plan.total_subchunks += num_sc;
        if (num_sc > 1u) {
            plan.tiles_split += 1u;
            plan.total_overflow_candidates += count;
        }
        plan.max_subchunks_per_tile =
            std::max(plan.max_subchunks_per_tile, num_sc);
    }
    return plan;
}

static void log_subchunk_stats(const SubchunkPlan& plan) {
    std::fprintf(
        stderr,
        "[SUBCHUNK] tiles_split=%u total_subchunks=%u max_subchunks_per_tile=%u "
        "total_overflow_candidates=%llu\n",
        plan.tiles_split,
        plan.total_subchunks,
        plan.max_subchunks_per_tile,
        static_cast<unsigned long long>(plan.total_overflow_candidates));
}

// Host-side [L1LOAD] stats (iter 49): bulk subchunk payloads the reader loads
// into L1 (overflow tiles only; in-budget bucket tiles stay on the iter-48 path).
struct L1LoadStats {
    uint64_t bytes_bulk_loaded = 0;
    uint32_t subchunks_bulk = 0;
};

static L1LoadStats build_l1load_stats(const SubchunkPlan& plan) {
    L1LoadStats stats;
    constexpr uint32_t kFit = render_config::kBucketFit;
    constexpr uint64_t kRecBytes = 64u;
    const uint32_t num_tiles =
        static_cast<uint32_t>(plan.meta.size() / 2u);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t num_sc = plan.meta[static_cast<std::size_t>(t) * 2u + 0u];
        const uint32_t count = plan.meta[static_cast<std::size_t>(t) * 2u + 1u];
        const bool in_budget =
            num_sc == 1u && count > 0u && count <= kFit;
        if (in_budget) {
            continue;
        }
        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * kFit;
            if (sc_off >= count) {
                continue;
            }
            const uint32_t l_sub =
                (count - sc_off > kFit) ? kFit : (count - sc_off);
            stats.subchunks_bulk += 1u;
            stats.bytes_bulk_loaded += static_cast<uint64_t>(l_sub) * kRecBytes;
        }
    }
    return stats;
}

static void log_l1load_stats(const L1LoadStats& stats) {
    const uint64_t avg = stats.subchunks_bulk > 0u
        ? stats.bytes_bulk_loaded / stats.subchunks_bulk
        : 0u;
    std::fprintf(
        stderr,
        "[L1LOAD] bytes_bulk_loaded=%llu subchunks_bulk=%u "
        "avg_bytes_per_subchunk=%llu\n",
        static_cast<unsigned long long>(stats.bytes_bulk_loaded),
        stats.subchunks_bulk,
        static_cast<unsigned long long>(avg));
}

static bool upload_subchunk_meta(
    DeviceContext& ctx, const SubchunkPlan& plan, uint32_t num_tiles) {
    const size_t bytes = static_cast<size_t>(num_tiles) * 2u * sizeof(uint32_t);
    if (!ctx.res_subchunk_meta || ctx.res_subchunk_meta_bytes < bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{
            .page_size = 64, .buffer_type = BufferType::DRAM};
        ctx.res_subchunk_meta =
            distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
        ctx.res_subchunk_meta_bytes = bytes;
        gsplat_tt::device_state::register_buffer(
            "blend_subchunk_meta", ctx.res_subchunk_meta);
    }
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_subchunk_meta, plan.meta);
    return true;
}

// Pull the program out of the workload (it was moved in at init time) so we
// can refresh its runtime args before each frame.
static Program& get_program_for_workload(DeviceContext& ctx) {
    auto& programs = ctx.workload.get_programs();
    auto it = programs.find(distributed::MeshCoordinateRange(ctx.mesh_device->shape()));
    if (it == programs.end()) {
        throw std::runtime_error("workload missing program for device range");
    }
    return it->second;
}

// ===========================================================================
// Microblock-major (4x8) device blend path (amendment-003 step 3).
// Self-contained: its own program/CBs/kernels + context, selected at runtime.
// ===========================================================================

namespace mb {

constexpr uint32_t CB_XRAMP     = 0;   // fp32 tile-local x ramp
constexpr uint32_t CB_YRAMP     = 1;   // fp32 tile-local y ramp
constexpr uint32_t CB_MB_COEFF  = 2;   // 48B coeff row per gaussian (mb-major)
constexpr uint32_t CB_MB_COUNTS = 3;   // 128B = 32 uint32 per tile
constexpr uint32_t CB_OUT       = 16;  // 3 bf16 color tiles per screen tile

// (The M2 §6 tail-skip CB_HS handshake region is dropped: the production blend
// never raises the whole-tile saturation early-out.)

constexpr uint32_t COUNTS_PAGE_BYTES = 128;
// 10 real fp32 lanes (A,B,C,mx,my,f,op,r,g,b) padded to a 64B DRAM-aligned page.
// MUST be a multiple of the DRAM alignment (64B on Blackhole): an unaligned page
// size (e.g. 48B) makes the interleaved per-bank page stride diverge between the
// host EnqueueWriteBuffer layout and the device TensorAccessor, so only some rows
// land and the rest read as zero.
constexpr uint32_t COEFF_ROW_BYTES_MB = 64;
constexpr uint32_t RAMP_TILE_BYTES = TILE_H * TILE_W * 4;  // fp32 32x32

static void build_program_and_workload_mb(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    // Baked production blend-kernel configuration (the GSPLAT_TT verify flags,
    // resolved to constants): RESIDENT_BLEND + SFPU_CULL + TILE_BUCKET +
    // BLEND_AOS + DEVCONIC + L1_RECORD, BUCKET_FIT=8192, MB_CULL_SPIN=512,
    // MB_BUCKET_CB_FENCE on. No soft-float MB_DEVCULL, no payload pack pass, no
    // L1-mask handoff, no transmittance early-out / tail-skip. The reader reads
    // the SFPU-precomputed keep mask, fetches each candidate as one 64B AoS
    // record, and serves each tile from its dense L1-resident bucket (tiles
    // above FIT fall back to the per-candidate DRAM gather, still on-device);
    // the compute derives the conic on the SFPU.
    constexpr uint32_t kBucketFit = 8192u;

    cb_cfg(CB_XRAMP, RAMP_TILE_BYTES, 2, DataFormat::Float32);
    cb_cfg(CB_YRAMP, RAMP_TILE_BYTES, 2, DataFormat::Float32);
    cb_cfg(CB_MB_COEFF, COEFF_ROW_BYTES_MB, 8, DataFormat::Float32);
    // Depth must cover max subchunks per tile (overflow tiles can be >>2);
    // reader bulk path can push counts faster than compute pops.
    cb_cfg(CB_MB_COUNTS, COUNTS_PAGE_BYTES, 64, DataFormat::UInt32);
    cb_cfg(CB_OUT, TILE_BYTES_BF16, 6, DataFormat::Float16_b);

    // Resident devcull reader scratch CBs.
    constexpr uint32_t CB_SCR_IDS = 4;
    constexpr uint32_t CB_SCR_ATTR = 5;
    constexpr uint32_t CB_SCR_MASK = 6;
    constexpr uint32_t CB_CORE_TILES = 7;
    cb_cfg(CB_SCR_IDS, 64, 2, DataFormat::UInt32);
    // Pipelined double-buffered batched gather scratch: 2 chunks x 16 gaussians
    // x 9 SoA pages (64B each) = 288 pages. The reader issues a whole chunk's
    // reads ahead of one barrier and overlaps cull of chunk K with the in-flight
    // reads of chunk K+1.
    constexpr uint32_t GATHER_PAGES = 2u * 16u * 9u;  // 288 pages
    cb_cfg(CB_SCR_ATTR, 64, GATHER_PAGES, DataFormat::Float32);
    // Reader-private double-buffered cull_masks scratch (2 buffers x 2 pages x
    // 64B = 256B): each <=16-gaussian chunk's masks span at most two 64B pages.
    cb_cfg(CB_SCR_MASK, 256, 1, DataFormat::UInt32);
    cb_cfg(CB_CORE_TILES, 64, 1, DataFormat::UInt32);

    // CB_BUCKET/CB_BMASK: in-budget sort+emit scratch (push deferred until after
    // coeff stream). CB_BUCKET_BULK/CB_BMASK_BULK: overflow bulk L1 (iter 49) —
    // separate rings so reader bulk reserve does not block on in-budget scratch.
    constexpr uint32_t CB_BUCKET = 9;
    constexpr uint32_t CB_BSORT  = 10;
    constexpr uint32_t CB_BMASK  = 11;
    constexpr uint32_t CB_BUCKET_BULK = 12;
    constexpr uint32_t CB_BMASK_BULK  = 13;
    constexpr uint32_t rec_bytes = 64u;
    cb_cfg(CB_BUCKET, rec_bytes, kBucketFit, DataFormat::Float32);
    cb_cfg(CB_BSORT, 4, 2u * kBucketFit + 256u, DataFormat::UInt32);
    cb_cfg(CB_BMASK, 64, (kBucketFit + 15u) / 16u + 1u, DataFormat::UInt32);
    cb_cfg(CB_BUCKET_BULK, rec_bytes, kBucketFit, DataFormat::Float32);
    cb_cfg(CB_BMASK_BULK, 64, (kBucketFit + 15u) / 16u + 1u, DataFormat::UInt32);

    // The resident devcull reader binds 20 DRAM-interleaved accessors: proj_m
    // a/b/c/px/py/opacity/colors (7) + sort_sorted_ids + sort_tile_ranges +
    // xramp + yramp + tile_ids + lpt_meta (6) + cull_masks + cull_mask_base (2)
    // + proj_m_blendrec (AoS, 1) + buf_l1_recs (L1_RECORD, 1) + 2 bucket buffers
    // + blend_subchunk_meta (iter 48 post-sort subchunk table).
    constexpr int num_reader_accessors = 20;
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < num_reader_accessors; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    // The blend reader is single-path: all former on/off feature macros are
    // inlined in the kernel source. Only the two VALUE macros it reads remain.
    std::map<std::string, std::string> reader_defines = {
        {"MB_CULL_SPIN", "512"},      // per-candidate mask read-completion settle
        {"MB_BUCKET_FIT", "8192u"},   // L1 dense-record bucket capacity (slots/tile)
    };
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_alpha_blend_mb_devcull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
            .defines = reader_defines,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_XRAMP] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_YRAMP] = UnpackToDestMode::UnpackToDestFp32;

    // The blend compute kernel is single-path: every feature macro is inlined in
    // the kernel source (MB_BUCKET_FIT is a constexpr in-kernel), so no defines.
    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/alpha_blend_compute_mb.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
        });

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context_mb() {
    DeviceContext ctx;
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program_and_workload_mb(ctx);
    return ctx;
}

// Tile-local coordinate ramp (one 32x32 fp32 tile, row-major per-pixel,
// identical for every screen tile): xramp[r*32+c] = c + 0.5, yramp = r + 0.5.
// Uploaded the same per-pixel way the proven full-tile kernel uploads px/py:
// copy_tile + pack_tile apply an identical tilization permutation to input and
// output, so x = dst_reg[vector] yields the correct per-pixel coords for
// whatever raster pixels that SFPU vector covers (the permutation cancels).
// Fixed permutation tbl[devpos] = imgpos mapping a device tile-local raster
// position (row-major i*32+j, as produced/consumed by the standard tilize/
// untilize) to the image tile-local raster position it represents.
//
// WHY: a single SFPU 32-lane vector dst_reg[V] (fp32 dest) owns the 2 tile rows
// {2*(r/2), +1} at the 16 columns of parity (V&1) -- i.e. V = 2*(r/2)+(c&1) --
// NOT a contiguous 4x8 raster block. To run true 4x8-microblock culling we make
// vector V compute microblock V (host index m=(my<<2)|mx, region rows
// [my*4,+4) x cols [mx*8,+8)). The ramp upload puts microblock V's coords into
// vector V's real pixel slots P(V); the output is scattered back here. Because
// the device tilize (input) and untilize (output) are exact inverses at the
// pixel level (proven: the full-tile kernel renders pixel-correct with the
// natural ramp), the same table serves both directions.
static const std::array<uint32_t, TILE_H * TILE_W>& mb_perm_img_of_dev() {
    static const std::array<uint32_t, TILE_H * TILE_W> tbl = [] {
        std::array<uint32_t, TILE_H * TILE_W> t{};
        for (int V = 0; V < 32; ++V) {
            // P(V): device raster positions owned by vector V, row-major.
            std::vector<int> Pv;
            Pv.reserve(32);
            for (int r = 0; r < 32; ++r) {
                for (int c = 0; c < 32; ++c) {
                    if (2 * (r / 2) + (c & 1) == V) {
                        Pv.push_back(r * 32 + c);
                    }
                }
            }
            // T(V): microblock V's true 4x8 raster block, row-major.
            const int my = V >> 2;   // row-band (4 rows)
            const int mx = V & 3;    // col-group (8 cols)
            std::vector<int> Tv;
            Tv.reserve(32);
            for (int dr = 0; dr < 4; ++dr) {
                for (int dc = 0; dc < 8; ++dc) {
                    Tv.push_back((my * 4 + dr) * 32 + (mx * 8 + dc));
                }
            }
            for (int k = 0; k < 32; ++k) {
                t[static_cast<size_t>(Pv[k])] = static_cast<uint32_t>(Tv[k]);
            }
        }
        return t;
    }();
    return tbl;
}

static std::vector<uint32_t> make_ramp(bool is_x) {
    // Permuted ramp: at each device raster slot store the coordinate of the
    // microblock pixel that the SFPU vector owning that slot must compute.
    const auto& tbl = mb_perm_img_of_dev();
    std::vector<uint32_t> r(TILE_H * TILE_W);
    for (uint32_t dev = 0; dev < TILE_H * TILE_W; dev++) {
        const uint32_t img = tbl[dev];
        const uint32_t img_i = img / TILE_W;  // row
        const uint32_t img_j = img % TILE_W;  // col
        const float v = (is_x ? static_cast<float>(img_j) : static_cast<float>(img_i)) + 0.5f;
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        r[dev] = bits;
    }
    return r;
}

// Microblock-permuted variant of tiles_to_image: scatters each device tile-
// local raster slot back to its true microblock raster position via the fixed
// permutation, then places the tile into the full image.
static std::vector<float> tiles_to_image_mb(
    const std::vector<uint16_t>& result_bf16,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w) {
    const auto& tbl = mb_perm_img_of_dev();
    std::vector<float> img(static_cast<size_t>(image_h) * image_w * 3, 0.0f);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t ty = t / tiles_x;
        const uint32_t tx = t % tiles_x;
        for (uint32_t ch = 0; ch < 3; ch++) {
            const auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
            for (uint32_t dev = 0; dev < TILE_H * TILE_W; dev++) {
                const uint32_t imgpos = tbl[dev];
                const uint32_t i = imgpos / TILE_W;
                const uint32_t j = imgpos % TILE_W;
                const uint32_t y = ty * TILE_H + i;
                const uint32_t x = tx * TILE_W + j;
                if (y < image_h && x < image_w) {
                    img[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[dev];
                }
            }
        }
    }
    return img;
}

static void tiles_to_image_mb_into(
    const std::vector<uint16_t>& result_bf16,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    float* image_out) {
    const auto& tbl = mb_perm_img_of_dev();
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t ty = t / tiles_x;
        const uint32_t tx = t % tiles_x;
        for (uint32_t ch = 0; ch < 3; ch++) {
            const auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
            for (uint32_t dev = 0; dev < TILE_H * TILE_W; dev++) {
                const uint32_t imgpos = tbl[dev];
                const uint32_t i = imgpos / TILE_W;
                const uint32_t j = imgpos % TILE_W;
                const uint32_t y = ty * TILE_H + i;
                const uint32_t x = tx * TILE_W + j;
                if (y < image_h && x < image_w) {
                    image_out[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[dev];
                }
            }
        }
    }
}


namespace {

constexpr uint32_t SORT_META_ELEMS_PER_PAGE = 16;

struct ResidentSortLpt {
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;
    std::shared_ptr<distributed::MeshBuffer> tile_ids_buf;
    bool ok = false;
};

static ResidentSortLpt resident_sort_lpt_handles() {
    namespace ds = gsplat_tt::device_state;
    ResidentSortLpt r;
    r.tile_ids_buf = ds::get_buffer("sort_lpt_tile_ids");
    r.ok = static_cast<bool>(r.tile_ids_buf && ds::get_buffer("sort_lpt_meta"));
    return r;
}

// One D2H of sort_lpt_meta for kernels that still take (start,count) host args
// (pack pass only). Cull + resident blend read meta on-device.
static ResidentSortLpt load_resident_sort_lpt(
    distributed::MeshCommandQueue* cq, uint32_t num_cores) {
    namespace ds = gsplat_tt::device_state;
    ResidentSortLpt r = resident_sort_lpt_handles();
    if (!r.ok) {
        return r;
    }
    auto meta = ds::get_buffer("sort_lpt_meta");
    r.per_core_offset.assign(num_cores, 0);
    r.per_core_count.assign(num_cores, 0);
    const uint32_t meta_pad =
        ((num_cores * 2 + SORT_META_ELEMS_PER_PAGE - 1) / SORT_META_ELEMS_PER_PAGE) *
        SORT_META_ELEMS_PER_PAGE;
    std::vector<uint32_t> mbuf(meta_pad, 0);
    distributed::EnqueueReadMeshBuffer(*cq, mbuf, meta, true);
    for (uint32_t c = 0; c < num_cores; c++) {
        r.per_core_offset[c] = mbuf[c * 2 + 0];
        r.per_core_count[c] = mbuf[c * 2 + 1];
    }
    return r;
}

}  // namespace

// RESIDENT device-cull blend frame (GSPLAT_TT_RESIDENT_BLEND): reads the
// per-gaussian attributes from the device-resident per-component SoA proj_m_*
// buffers and the candidate ids/ranges from resident sort_sorted_ids /
// sort_tile_ranges (all registered in device_state by the on-device gather +
// sort stages). NOTHING about attrs/ids is uploaded per frame — LPT tile lists
// are reused from the sort stage (sort_lpt_*). Returns elapsed ms; sets *ok=false
// (no work done) if any required resident buffer is absent.

enum class ResidentBlendPhase {
    Complete,
    SetupRuntimeArgsOnly,
    DeviceLaunchAndReadback,
};

static double process_frame_mb_devcull_resident(
    DeviceContext& ctx,
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    float* image_out,
    bool* ok,
    ResidentBlendPhase phase = ResidentBlendPhase::Complete) {
    namespace ds = gsplat_tt::device_state;
    auto buf_a   = ds::get_buffer("proj_m_a");
    auto buf_b   = ds::get_buffer("proj_m_b");
    auto buf_c   = ds::get_buffer("proj_m_c");
    auto buf_px  = ds::get_buffer("proj_m_px");
    auto buf_py  = ds::get_buffer("proj_m_py");
    auto buf_op  = ds::get_buffer("proj_m_opacity");
    auto buf_col = ds::get_buffer("proj_m_colors");
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    if (!buf_a || !buf_b || !buf_c || !buf_px || !buf_py || !buf_op || !buf_col ||
        !buf_ids || !buf_rng) {
        if (ok) *ok = false;
        return 0.0;
    }
    if (ok) *ok = true;

    uint32_t cull_masks_addr = 0;
    const bool launch_only = (phase == ResidentBlendPhase::DeviceLaunchAndReadback);

    if (!launch_only) {
    // Post-sort subchunk table: host reads sort_tile_ranges, splits fat tiles
    // (count > kBucketFit) into depth-ordered subchunks, uploads metadata for
    // the blend reader dispatch loop (iter 48).
    const SubchunkPlan subchunk_plan = build_subchunk_plan(ctx.cq, num_tiles);
    log_subchunk_stats(subchunk_plan);
    log_l1load_stats(build_l1load_stats(subchunk_plan));
    if (!upload_subchunk_meta(ctx, subchunk_plan, num_tiles)) {
        if (ok) *ok = false;
        return 0.0;
    }
    const uint32_t subchunk_meta_addr =
        static_cast<uint32_t>(ctx.res_subchunk_meta->address());

    // SFPU cull: the blend reader reads the precomputed 32-bit mask from the
    // resident cull_masks buffer (registered by the cull pass) instead of
    // running the soft-float cull. Built with the matching MB_SFPU_CULL define.
    const bool sfpu_cull = true;      // SFPU_CULL=1
    const bool blend_aos = true;      // BLEND_AOS default-on
    const bool tile_bucket = true;    // TILE_BUCKET=1
    uint32_t tile_recs_addr = 0;
    uint32_t bucket_meta_addr = 0;
    if (tile_bucket) {
        auto buf_recs = ds::get_buffer("sort_tile_recs");
        auto buf_bm   = ds::get_buffer("sort_bucket_meta");
        if (!buf_recs || !buf_bm) {
            if (ok) *ok = false;
            return 0.0;
        }
        tile_recs_addr = static_cast<uint32_t>(buf_recs->address());
        bucket_meta_addr = static_cast<uint32_t>(buf_bm->address());
    }
    uint32_t blendrec_addr = 0;
    if (blend_aos) {
        auto buf_brec = gsplat_tt::device_state::get_buffer("proj_m_blendrec");
        if (!buf_brec) {
            if (ok) *ok = false;
            return 0.0;
        }
        blendrec_addr = static_cast<uint32_t>(buf_brec->address());
    }
    uint32_t cull_base_addr = 0;
    if (sfpu_cull) {
        auto buf_masks = gsplat_tt::device_state::get_buffer("cull_masks");
        if (buf_masks) {
            cull_masks_addr = static_cast<uint32_t>(buf_masks->address());
        }
        auto buf_base = gsplat_tt::device_state::get_buffer("cull_mask_base");
        if (buf_base) {
            cull_base_addr = static_cast<uint32_t>(buf_base->address());
        }
    }

    const ResidentSortLpt lpt = resident_sort_lpt_handles();
    if (!lpt.ok) {
        if (ok) *ok = false;
        return 0.0;
    }
    auto meta_buf = ds::get_buffer("sort_lpt_meta");
    const uint32_t lpt_meta_addr =
        meta_buf ? static_cast<uint32_t>(meta_buf->address()) : 0u;
    // Reader/compute/writer all read LPT on-device (no sort_lpt_meta D2H).

    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };

    // Persistent scratch: ramps (constant), output + tile-id list (grow-on-demand).
    if (!ctx.res_xramp) {
        ctx.res_xramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_yramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_ramp_uploaded = false;
    }
    if (!ctx.res_out || ctx.res_out_tiles < num_tiles) {
        ctx.res_out = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
        ctx.res_out_tiles = num_tiles;
    }

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t a_addr     = static_cast<uint32_t>(buf_a->address());
    const uint32_t b_addr     = static_cast<uint32_t>(buf_b->address());
    const uint32_t c_addr     = static_cast<uint32_t>(buf_c->address());
    const uint32_t px_addr    = static_cast<uint32_t>(buf_px->address());
    const uint32_t py_addr    = static_cast<uint32_t>(buf_py->address());
    const uint32_t op_addr    = static_cast<uint32_t>(buf_op->address());
    const uint32_t col_addr   = static_cast<uint32_t>(buf_col->address());
    const uint32_t ids_addr   = static_cast<uint32_t>(buf_ids->address());
    const uint32_t rng_addr   = static_cast<uint32_t>(buf_rng->address());
    const uint32_t xramp_addr = static_cast<uint32_t>(ctx.res_xramp->address());
    const uint32_t yramp_addr = static_cast<uint32_t>(ctx.res_yramp->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(lpt.tile_ids_buf->address());
    const uint32_t out_addr   = static_cast<uint32_t>(ctx.res_out->address());
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &contrib_floor, 4);
    {
        GSPLAT_HOST_ZONE("host_blend_setup");
        for (const auto& range : ctx.all_cores.ranges()) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    CoreCoord core{x, y};
                    std::vector<uint32_t> reader_args = {
                        a_addr, b_addr, c_addr, px_addr, py_addr, op_addr, col_addr,
                        ids_addr, rng_addr, xramp_addr, yramp_addr,
                        tile_ids_addr, lpt_meta_addr, core_index, tiles_x, floor_bits,
                        cull_disabled ? 1u : 0u,
                    };
                    {
                        if (sfpu_cull) {
                            reader_args.push_back(cull_masks_addr);  // arg 17
                            reader_args.push_back(cull_base_addr);   // arg 18
                            if (blend_aos) {
                                reader_args.push_back(blendrec_addr);  // arg 19
                                if (tile_bucket) {
                                    reader_args.push_back(tile_recs_addr);    // arg 20
                                    reader_args.push_back(bucket_meta_addr);  // arg 21
                                    auto buf_l1r = ds::get_buffer("sort_l1_recs");
                                    reader_args.push_back(
                                        (gsplat_tt::env_config::l1_record_enabled() && buf_l1r)
                                            ? static_cast<uint32_t>(buf_l1r->address())
                                            : 0u);  // arg 22
                                }
                            }
                        }
                    }
                    reader_args.push_back(subchunk_meta_addr);  // arg 23 (iter 48)
                    SetRuntimeArgs(program, ctx.reader, core, reader_args);
                    SetRuntimeArgs(program, ctx.compute, core, {0u});
                    SetRuntimeArgs(program, ctx.writer, core, {
                        out_addr, tile_ids_addr, lpt_meta_addr, core_index,
                    });
                    core_index++;
                }
            }
        }
    }

    if (phase == ResidentBlendPhase::SetupRuntimeArgsOnly) {
        return 0.0;
    }
    } else {
        if (!ctx.res_out || !ctx.res_xramp || !ctx.res_yramp) {
            if (ok) *ok = false;
            return 0.0;
        }
        if (auto buf_masks = ds::get_buffer("cull_masks")) {
            cull_masks_addr = static_cast<uint32_t>(buf_masks->address());
        }
    }

    const auto t_start = std::chrono::steady_clock::now();
    // Writer overwrites every LPT tile in res_out; skip the per-frame zero H2D.
    if (!gsplat_tt::env_config::blend_skip_zero_out_enabled()) {
        std::vector<uint16_t> output_zero(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_out, output_zero);
    }
    // Constant ramps: upload once, then reuse the resident copy every frame.
    if (!ctx.res_ramp_uploaded) {
        auto xramp = make_ramp(/*is_x=*/true);
        auto yramp = make_ramp(/*is_x=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_xramp, xramp);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_yramp, yramp);
        ctx.res_ramp_uploaded = true;
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    {
        GSPLAT_HOST_ZONE("host_finish_blend");
        distributed::Finish(*ctx.cq);
    }
    gsplat_tt::device_state::clear_sort_publish_pending();
    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, ctx.res_out, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();

    tiles_to_image_mb_into(result_bf16, num_tiles, tiles_x, image_h, image_w, image_out);
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

// ===========================================================================
// SFPU microblock-cull pass (GSPLAT_TT_SFPU_CULL).
//
// A dedicated 3-kernel program (reader_microblock_cull / microblock_cull_compute
// / writer_microblock_cull) that runs BEFORE the resident blend. It computes the
// 32-bit microblock mask of every depth-sorted candidate ON THE SFPU (one 32-lane
// vector == all 32 microblocks of a gaussian) and stores it resident in
// cull_masks (registered in device_state). The blend reader then reads the mask
// with a pure integer load -> the expensive soft-float constrained-min cull is
// gone from the data mover.
// ===========================================================================
namespace cull {

constexpr uint32_t CB_BOX_OX     = 0;   // fp32 box x-origin ramp tile (constant)
constexpr uint32_t CB_BOX_OY     = 1;   // fp32 box y-origin ramp tile (constant)
constexpr uint32_t CB_CULL_COEFF = 2;   // 64B coeff row per gaussian (6 used)
constexpr uint32_t CB_CULL_COUNTS= 3;   // per-tile [L, tx_pix, ty_pix]
constexpr uint32_t CB_SCR_IDS    = 4;   // reader-private ids/ranges scratch
constexpr uint32_t CB_SCR_ATTR   = 5;   // reader-private gather scratch
constexpr uint32_t CB_MASK_SCR   = 6;   // writer-private mask packing scratch
constexpr uint32_t CB_KEEP       = 16;  // compute -> writer keep tiles

constexpr uint32_t COEFF_ROW_BYTES = 64;
constexpr uint32_t COUNTS_PAGE_BYTES = 64;
constexpr uint32_t MASKS_PAGE_BYTES = 64;   // 16 u32 per page
constexpr uint32_t GATHER_FIELDS = 6;

// Intra-vector CB-linear position for (gaussian g == SFPU vector, microblock m).
// MUST match perm() in writer_microblock_cull.cpp exactly.
inline uint32_t perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

// Constant box-origin ramp tile: at CB-linear position perm(g,m) store
// microblock m's tile-local box origin ((m&3)*8 for x, (m>>2)*4 for y). After
// the compute kernel's copy_tile->math->pack_tile round-trip (CB-linear-
// identity), the keep flag for (vector g, microblock m) lands back at the same
// position, which the writer reads to assemble the 32-bit mask.
static std::vector<uint32_t> make_box_ramp(bool is_x) {
    std::vector<uint32_t> r(TILE_H * TILE_W, 0);
    for (uint32_t g = 0; g < 32; ++g) {
        for (uint32_t m = 0; m < 32; ++m) {
            const uint32_t dev = perm(g, m);
            const float v = is_x ? static_cast<float>((m & 3u) * 8u)
                                 : static_cast<float>((m >> 2) * 4u);
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            r[dev] = bits;
        }
    }
    return r;
}

static void build_program_and_workload(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_cfg(CB_BOX_OX, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
    cb_cfg(CB_BOX_OY, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
    cb_cfg(CB_CULL_COEFF, COEFF_ROW_BYTES, 32, DataFormat::Float32);
    cb_cfg(CB_CULL_COUNTS, COUNTS_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_cfg(CB_SCR_IDS, 64, 2, DataFormat::UInt32);
    cb_cfg(CB_SCR_ATTR, 64, 16u * GATHER_FIELDS, DataFormat::Float32);  // 96 pages
    cb_cfg(CB_MASK_SCR, 128, 1, DataFormat::UInt32);
    cb_cfg(CB_KEEP, mb::RAMP_TILE_BYTES, 4, DataFormat::Float32);
    constexpr uint32_t CB_CORE_TILES = 7;
    cb_cfg(CB_CORE_TILES, 64, 1, DataFormat::UInt32);

    // Reader: 12 DRAM-interleaved accessors (a,b,c,px,py,op, ids, ranges,
    // box_ox, box_oy, tile_ids, sort_lpt_meta).
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 12; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_microblock_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_BOX_OX] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_BOX_OY] = UnpackToDestMode::UnpackToDestFp32;
    // Single-path cull compute kernel: CULL_LPT_CB is inlined in the source.
    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/microblock_cull_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
        });

    // Writer: 5 accessors (cull_masks, ranges, tile_ids, sort_lpt_meta,
    // cull_mask_base). cull_masks (index 0) is L1-interleaved under the L1
    // mask handoff so the writer's masks land in the same resident-L1 buffer
    // the blend reader pops.
    std::vector<uint32_t> writer_ct;
    for (int i = 0; i < 5; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    }
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_microblock_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context() {
    DeviceContext ctx;
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    cull::build_program_and_workload(ctx);
    return ctx;
}

// Allocate cull_masks + cull_mask_base from `ctx`'s allocator (the BLEND
// context g_ctx_mb) and upload the per-tile PAGE-ALIGNED base prefix. MUST be
// called before the cull pass runs. Allocating from the blend context (rather
// than the separate cull context) is REQUIRED for correctness: the two
// contexts' DRAM allocations are not mutually tracked, so cull-context buffers
// overlapped the blend's res_out and were clobbered mid-blend. base[t] is the
// prefix sum of round_up(per_tile_count[i], 16); each tile is padded to a whole
// 64B/16-elem page so every writer DRAM write is 16-element aligned.
static bool ensure_resident_buffers(
    DeviceContext& ctx,
    uint32_t num_tiles) {
    namespace ds = gsplat_tt::device_state;
    auto published_base = ds::get_buffer("cull_mask_base");
    if (!published_base) {
        return false;
    }
    uint32_t pipe_p = 0;
    uint32_t pipe_mask = 0;
    size_t total_elems = 16;
    if (ds::get_sort_blend_pipe_scalars(&pipe_p, &pipe_mask)) {
        total_elems = std::max<size_t>(16, static_cast<size_t>(pipe_mask));
    } else {
        auto buf_pk = ds::get_buffer("sort_P_kept");
        if (!buf_pk) {
            return false;
        }
        std::vector<uint32_t> pkept(16, 0);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, pkept, buf_pk, true);
        total_elems = std::max<size_t>(16, static_cast<size_t>(pkept[1]));
        (void)pipe_p;
    }
    const size_t masks_bytes = (total_elems / 16) * MASKS_PAGE_BYTES;
    // Production cull_masks live in DRAM (the L1-mask handoff experiment is
    // dropped; see the iter-15 finding that the per-candidate settle is a read-
    // completion window, not a DRAM write-settle artifact). Same 64B/16-elem
    // page layout and per-tile page-aligned base either way.
    const BufferType masks_bt = BufferType::DRAM;
    auto make_masks_buf = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = masks_bt};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };
    if (!ctx.res_masks || ctx.res_masks_bytes < masks_bytes) {
        ctx.res_masks = make_masks_buf(masks_bytes, MASKS_PAGE_BYTES);
        ctx.res_masks_bytes = masks_bytes;
        ds::register_buffer("cull_masks", ctx.res_masks);
    }
    // cull_mask_base is published by sort (H2D at sort time); blend only aliases it.
    ctx.res_mask_base = published_base;
    ctx.res_mask_base_bytes = static_cast<size_t>(published_base->size());
    return true;
}

// Run the SFPU cull pass over all candidates and register the resident
// cull_masks buffer in device_state. Returns elapsed ms (or 0 if a required
// resident buffer is missing, leaving *ok=false).
static double process_frame(
    DeviceContext& ctx,
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    bool* ok,
    bool defer_cq_finish = false) {
    namespace ds = gsplat_tt::device_state;
    auto buf_a   = ds::get_buffer("proj_m_a");
    auto buf_b   = ds::get_buffer("proj_m_b");
    auto buf_c   = ds::get_buffer("proj_m_c");
    auto buf_px  = ds::get_buffer("proj_m_px");
    auto buf_py  = ds::get_buffer("proj_m_py");
    auto buf_op  = ds::get_buffer("proj_m_opacity");
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    if (!buf_a || !buf_b || !buf_c || !buf_px || !buf_py || !buf_op || !buf_ids || !buf_rng) {
        if (ok) *ok = false;
        return 0.0;
    }
    if (ok) *ok = true;

    uint64_t total_candidates = 0;
    uint64_t total_mask_elems = 16;
    uint32_t pipe_p = 0;
    uint32_t pipe_mask = 0;
    if (ds::get_sort_blend_pipe_scalars(&pipe_p, &pipe_mask)) {
        total_candidates = pipe_p;
        total_mask_elems = std::max<uint64_t>(16, pipe_mask);
    } else {
        auto buf_pk = ds::get_buffer("sort_P_kept");
        if (!buf_pk) {
            if (ok) *ok = false;
            return 0.0;
        }
        std::vector<uint32_t> pkept(16, 0);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, pkept, buf_pk, true);
        total_candidates = pkept[0];
        total_mask_elems = std::max<uint64_t>(16, pkept[1]);
    }

    const ResidentSortLpt lpt = resident_sort_lpt_handles();
    if (!lpt.ok) {
        if (ok) *ok = false;
        return 0.0;
    }
    auto meta_buf = ds::get_buffer("sort_lpt_meta");
    const uint32_t lpt_meta_addr =
        meta_buf ? static_cast<uint32_t>(meta_buf->address()) : 0u;
    // Reader/compute/writer read LPT on-device (no sort_lpt_meta D2H).

    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };

    // Persistent box-origin ramps (constant) + grow-on-demand tile-id list,
    // reused across frames in the cull context.
    if (!ctx.res_xramp) {
        ctx.res_xramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_yramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_ramp_uploaded = false;
    }
    // cull_masks + cull_mask_base are allocated by ensure_resident_buffers() on
    // the BLEND context (g_ctx_mb) so they live in the SAME allocator that owns
    // the blend's res_out / ramps / tile_ids. Allocating them from this separate
    // cull context produced DRAM that the blend allocator did not track, so
    // res_out overlapped cull_masks and clobbered the masks mid-blend (the bug:
    // hero_vs_ref stuck ~29 dB). Here we ONLY read their addresses.
    auto buf_masks_res = ds::get_buffer("cull_masks");
    auto buf_base_res  = ds::get_buffer("cull_mask_base");
    if (!buf_masks_res || !buf_base_res) {
        if (ok) *ok = false;
        return 0.0;
    }
    const size_t masks_bytes = (static_cast<size_t>(total_mask_elems) / 16) * MASKS_PAGE_BYTES;

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t a_addr      = static_cast<uint32_t>(buf_a->address());
    const uint32_t b_addr      = static_cast<uint32_t>(buf_b->address());
    const uint32_t c_addr      = static_cast<uint32_t>(buf_c->address());
    const uint32_t px_addr     = static_cast<uint32_t>(buf_px->address());
    const uint32_t py_addr     = static_cast<uint32_t>(buf_py->address());
    const uint32_t op_addr     = static_cast<uint32_t>(buf_op->address());
    const uint32_t ids_addr    = static_cast<uint32_t>(buf_ids->address());
    const uint32_t rng_addr    = static_cast<uint32_t>(buf_rng->address());
    const uint32_t box_ox_addr = static_cast<uint32_t>(ctx.res_xramp->address());
    const uint32_t box_oy_addr = static_cast<uint32_t>(ctx.res_yramp->address());
    const uint32_t masks_addr  = static_cast<uint32_t>(buf_masks_res->address());
    const uint32_t base_addr   = static_cast<uint32_t>(buf_base_res->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(lpt.tile_ids_buf->address());
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &contrib_floor, 4);
    {
        GSPLAT_HOST_ZONE("host_cull_setup");
        for (const auto& range : ctx.all_cores.ranges()) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    CoreCoord core{x, y};
                    SetRuntimeArgs(program, ctx.reader, core, {
                        a_addr, b_addr, c_addr, px_addr, py_addr, op_addr,
                        ids_addr, rng_addr, box_ox_addr, box_oy_addr,
                        tile_ids_addr, lpt_meta_addr, core_index, tiles_x, floor_bits,
                    });
                    SetRuntimeArgs(program, ctx.compute, core, {
                        0u, floor_bits, cull_disabled ? 1u : 0u,
                    });
                    SetRuntimeArgs(program, ctx.writer, core, {
                        masks_addr, rng_addr, tile_ids_addr,
                        lpt_meta_addr, core_index, base_addr,
                    });
                    core_index++;
                }
            }
        }
    }

    // CULL_PIPELINE: do NOT Finish between cull and blend on the shared in-order CQ.
    // defer_cq_finish chains cull+blend into one drain (also drains sort publish).
    const bool pipeline =
        defer_cq_finish || gsplat_tt::env_config::cull_pipeline_enabled();
    const auto t_start = std::chrono::steady_clock::now();
    if (!ctx.res_ramp_uploaded) {
        auto bx = make_box_ramp(/*is_x=*/true);
        auto by = make_box_ramp(/*is_x=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_xramp, bx);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_yramp, by);
        ctx.res_ramp_uploaded = true;
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    if (!pipeline) {
        distributed::Finish(*ctx.cq);
    }
    const auto t_end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

}  // namespace cull

}  // namespace mb

namespace gsplat_tt {

namespace {
std::unique_ptr<DeviceContext> g_ctx_mb;
std::unique_ptr<DeviceContext> g_ctx_cull;   // SFPU microblock-cull pass
}  // namespace

void blend_warmup_resident_contexts() {
    (void)gsplat_tt::device_state::get_device();
    if (!g_ctx_mb) {
        g_ctx_mb = std::make_unique<DeviceContext>(::mb::init_device_context_mb());
    }
    // SFPU_CULL=1 (production): always create the SFPU cull context.
    if (!g_ctx_cull) {
        g_ctx_cull = std::make_unique<DeviceContext>(::mb::cull::init_device_context());
    }
}

double blend_mb_devcull_resident(
    float contrib_floor,
    bool cull_disabled,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    float* image_out,
    bool* device_ok,
    double* cull_ms_out,
    double* blend_ms_out) {
    if (cull_ms_out) *cull_ms_out = 0.0;
    if (blend_ms_out) *blend_ms_out = 0.0;
    if (!g_ctx_mb) {
        (void)gsplat_tt::device_state::get_device();
        g_ctx_mb = std::make_unique<DeviceContext>(::mb::init_device_context_mb());
    }

    // SFPU microblock-cull pass (GSPLAT_TT_SFPU_CULL): precompute the 32-bit
    // masks on the SFPU into the resident cull_masks buffer BEFORE the blend.
    // The blend reader then reads the mask (pure integer) instead of running
    // the soft-float constrained-min cull.
    double cull_ms = 0.0;
    const bool sfpu_cull = true;  // SFPU_CULL=1 (production)
    const bool chain_cull_blend = sfpu_cull && gsplat_tt::env_config::cull_pipeline_enabled();
    if (sfpu_cull) {
        if (!g_ctx_cull) {
            g_ctx_cull = std::make_unique<DeviceContext>(::mb::cull::init_device_context());
        }
        if (!::mb::cull::ensure_resident_buffers(*g_ctx_mb, static_cast<uint32_t>(num_tiles))) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
        if (chain_cull_blend) {
            // Overlap: set all blend runtime args before enqueuing cull so the
            // ~110-core host setup runs while the SFPU cull pass executes.
            {
                GSPLAT_HOST_ZONE("host_blend_setup");
                ::mb::process_frame_mb_devcull_resident(
                    *g_ctx_mb, contrib_floor, cull_disabled,
                    static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
                    static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
                    image_out, device_ok, ::mb::ResidentBlendPhase::SetupRuntimeArgsOnly);
            }
            if (device_ok && !*device_ok) {
                return 0.0;
            }
        }
        bool cull_ok = false;
        cull_ms = ::mb::cull::process_frame(
            *g_ctx_cull, contrib_floor, cull_disabled,
            static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x), &cull_ok,
            chain_cull_blend);
        if (!cull_ok) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
    }
    const double blend_ms = chain_cull_blend
        ? ::mb::process_frame_mb_devcull_resident(
              *g_ctx_mb, contrib_floor, cull_disabled,
              static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
              static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
              image_out, device_ok, ::mb::ResidentBlendPhase::DeviceLaunchAndReadback)
        : ::mb::process_frame_mb_devcull_resident(
              *g_ctx_mb, contrib_floor, cull_disabled,
              static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
              static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
              image_out, device_ok);
    if (cull_ms_out) *cull_ms_out = cull_ms;
    if (blend_ms_out) *blend_ms_out = blend_ms;
    return cull_ms + blend_ms;
}

void device_shutdown() {
    // Leak contexts on shutdown — see sort_device_shutdown().
    (void)g_ctx_mb.release();
    (void)g_ctx_cull.release();
}

}  // namespace gsplat_tt
