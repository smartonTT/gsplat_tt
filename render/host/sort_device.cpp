// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt sort — amendment-002 tt-003.
//
// See sort.h for the staged design (S0 host-sort-to-resident, S1 device radix).
//
// S1 keeps the host binning (Pass1 counts + Pass2 stable scatter, identical to
// gsplat_cpu::sort.cpp) but lays the (key, id) pairs out PAGE-ALIGNED per tile
// in DRAM so the device radix kernel can read/write each tile's exclusive 64B
// pages without cross-tile races. After the kernel, the host compacts the
// aligned per-tile segments back into the CPU-contiguous order (Pass4) and
// widens ids to int64 so the returned SortResult is byte-identical to the CPU.
//
// Both stages publish the CONTIGUOUS outputs into device_state under
// "sort_sorted_ids" (uint32, P) and "sort_tile_ranges" (uint32, num_tiles*2)
// so a future device blend can consume them resident.

#include "blend.h"
#include "config.h"
#include "env_config.h"
#include "sort.h"
#include "device_state.h"
#include "host_tracy.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/kernel_types.hpp"

#include "gsplat_cpu/thread_pool.h"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace gsplat_tt {
namespace {

constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t PAGE_BYTES = ELEMS_PER_PAGE * 4;  // 64

// iter 110 (A2): the depth-sorted PACK2 slab (sort_subchunk_payload) uses a
// LARGE DRAM interleave page so each subchunk loads in ceil(L/64) big NoC
// transfers instead of ~4096 per-64B-page reads. The in-record layout is
// UNCHANGED — the slab is still a contiguous array of 32B records (record g at
// byte g*32); only the DRAM interleave granularity grows. SLAB_RECS_PER_PAGE
// = SLAB_PAGE_BYTES / 32. Subchunk bases are page-aligned in these units.
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / 32u;  // 64

// ROUTE C bucket-cull (GSPLAT_TT_BUCKET_MASK): the SFPU microblock cull runs in
// the sort stage over the dense record bucket so the keep mask is a SORT-STAGE
// write baked into record word 10 (read back spin-free by the L1 blend).
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t RAMP_TILE_BYTES = TILE_DIM * TILE_DIM * 4;  // 4096 (fp32 32x32)

// Intra-vector CB-linear position for (gaussian g == SFPU vector, microblock m).
// MUST match perm() in writer_bucket_cull.cpp / microblock_cull_compute.cpp.
inline uint32_t cull_perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

// Constant box-origin ramp: at CB-linear cull_perm(g,m) store microblock m's
// tile-local box origin ((m&3)*8 for x, (m>>2)*4 for y). Identical to the
// blend-side cull's make_box_ramp.
static std::vector<uint32_t> make_box_ramp(bool is_x) {
    std::vector<uint32_t> r(TILE_DIM * TILE_DIM, 0);
    for (uint32_t g = 0; g < 32; ++g) {
        for (uint32_t m = 0; m < 32; ++m) {
            const uint32_t dev = cull_perm(g, m);
            const float v = is_x ? static_cast<float>((m & 3u) * 8u)
                                 : static_cast<float>((m >> 2) * 4u);
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            r[dev] = bits;
        }
    }
    return r;
}

static bool bucket_mask_enabled() { return false; }  // BUCKET_MASK unset

// L1 scratch budget per ping/pong key+id buffer. The worst hero-scene tile is
// ~25k entries; 32768 leaves headroom. 4 CBs * 32768 * 4B = 512 KB, well under
// Blackhole's ~1.4 MB per-core L1. If a tile exceeds this the host
// transparently falls back to the CPU sort (device_ok = false).
constexpr uint32_t MAX_TILE_ENTRIES = 32768;
constexpr uint32_t MAX_TILE_PAGES = MAX_TILE_ENTRIES / ELEMS_PER_PAGE;  // 2048
constexpr uint32_t SCRATCH_BYTES = MAX_TILE_ENTRIES * 4;  // 128 KB per CB

// Device-binning: max tiles the per-core L1 row / cursor / offset CBs hold
// (hero is 1024 tiles). Larger inputs are unsupported and hard-fail.
constexpr uint32_t MAX_BIN_TILES = 2048;
constexpr uint32_t BIN_ROW_BYTES = MAX_BIN_TILES * 4;  // 8 KB
// Max kept pairs a single core counting-sorts in L1. Pairs are split evenly by
// page across cores, so per-core load ~= P_kept / num_cores (~25k on hero);
// 65536 (256 KB per ks/is CB) leaves >2x headroom. Larger inputs hard-fail.
constexpr uint32_t BIN_LOCAL_MAX = 65536;
constexpr uint32_t BIN_LOCAL_BYTES = BIN_LOCAL_MAX * 4;  // 256 KB

inline uint32_t round_up(uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; }

struct SortDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    distributed::MeshWorkload workload;
    KernelHandle kernel{};

    // On-device compact+publish (buf_out -> sort_sorted_ids).
    distributed::MeshWorkload wl_publish;
    KernelHandle kpublish{};

    // ROUTE C bucket-cull program (reader_bucket_cull / microblock_cull_compute
    // / writer_bucket_cull): SFPU microblock cull over the dense record bucket,
    // baking the keep mask into record word 10 (GSPLAT_TT_BUCKET_MASK).
    distributed::MeshWorkload wl_cull;
    KernelHandle kc_reader{};
    KernelHandle kc_compute{};
    KernelHandle kc_writer{};
    bool cull_built = false;
    std::shared_ptr<distributed::MeshBuffer> buf_box_ox;
    std::shared_ptr<distributed::MeshBuffer> buf_box_oy;
    bool box_ramp_uploaded = false;

    // R4/R5 device-binning program (count + scatter) for resident pairs.
    distributed::MeshWorkload wl_bin;
    KernelHandle kbin{};
    // Post-count layout + LPT (single core, GSPLAT_TT_SORT_DEVICE_LAYOUT).
    distributed::MeshWorkload wl_bin_layout;
    KernelHandle kbin_layout{};
    std::shared_ptr<distributed::MeshBuffer> buf_bin_ctrl;  // 1-page control out
    // S5.4 (iter-125): parallel Pass-2 base emit. The coordinator (wl_bin_layout)
    // publishes per-worker running-prefix checkpoints into buf_layout_ckpt; this
    // multi-core workload emits the bin2d + l1_rec_base rows in parallel.
    distributed::MeshWorkload wl_bin_layout_emit;
    KernelHandle kbin_layout_emit{};
    std::shared_ptr<distributed::MeshBuffer> buf_layout_ckpt;  // num_workers × 2 rows
    std::size_t cap_layout_ckpt_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bin2d;  // per-core 2D hist/base
    std::size_t cap_bin2d_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bin_dbg;  // core0 scatter dump

    // Cached DRAM buffers (grow-on-demand).
    std::shared_ptr<distributed::MeshBuffer> buf_keys;     // aligned packed keys
    std::shared_ptr<distributed::MeshBuffer> buf_ids;      // aligned packed ids
    std::shared_ptr<distributed::MeshBuffer> buf_out;      // aligned sorted ids
    std::size_t cap_aligned_bytes = 0;

    // T1/T2 (GSPLAT_TT_TILE_BUCKET): per-tile contiguous full-record bucket
    // scattered by the bin in arbitrary order. DENSE layout: tile t occupies
    // record-pages [starts[t], starts[t]+counts[t]) (1 record == 1 page == 64B),
    // no padding. buf_bin2d_rec holds the per-(core,tile) DENSE base (mirrors
    // bin2d); buf_bucket_meta publishes (start,count) per tile for the reader.
    std::shared_ptr<distributed::MeshBuffer> buf_tile_recs;
    std::size_t cap_tile_recs_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bin2d_rec;
    std::size_t cap_bin2d_rec_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bucket_meta;
    std::size_t cap_bucket_meta_bytes = 0;

    // M0 (GSPLAT_TT_L1_RECORD): pre-sized 32B per-entry L1 record bucket.
    // buf_l1_recs: BUCKET_FIT * num_tiles records × 32B each (pre-sized; tile t
    //   at slot range [t*BUCKET_FIT, (t+1)*BUCKET_FIT)).
    // buf_l1_rec_base: per-(core,tile) slot index start within buf_l1_recs —
    //   l1_base[c][t] = t*BUCKET_FIT + sum_{c'<c} count[c'][t].
    //   Same shape as buf_bin2d_rec but in 32B record slot units.
    std::shared_ptr<distributed::MeshBuffer> buf_l1_recs;
    std::size_t cap_l1_recs_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_l1_rec_base;
    std::size_t cap_l1_rec_base_bytes = 0;

    // iter-138 (Stage-2b overflow pre-pack): separate COMPACT PACK2 region holding
    // the FULL records of overflow tiles (kBucketFit < count <= kOverflowL1Cap),
    // prefix-allocated over those tiles only. buf_l1_ov: the records (64B PACK2
    // pages, same layout as buf_l1_recs). buf_l1_ov_base: per-(core,tile) start
    // slot in the region (sentinel for non-overflow tiles) — consumed by the emit
    // (bin) kernel. buf_tile_ov_base: per-TILE start slot (sentinel otherwise) —
    // consumed by the materialize kernel to coalesced-read + L1-radix the bucket.
    std::shared_ptr<distributed::MeshBuffer> buf_l1_ov;
    std::size_t cap_l1_ov_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_l1_ov_base;
    std::size_t cap_l1_ov_base_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_ov_base;
    std::size_t cap_tile_ov_base_bytes = 0;

    std::shared_ptr<distributed::MeshBuffer> buf_tile_ids;  // LPT tile-id list
    std::size_t cap_tile_ids_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tmeta;     // (pstart_page, n)
    std::size_t cap_tmeta_bytes = 0;

    // Resident contiguous outputs (published for downstream device consumers).
    std::shared_ptr<distributed::MeshBuffer> buf_sorted_ids;   // contiguous, P
    std::size_t cap_sorted_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_ranges;  // num_tiles*2
    std::size_t cap_ranges_bytes = 0;

    // Downstream blend/cull LPT + per-tile kept counts (published resident).
    std::shared_ptr<distributed::MeshBuffer> buf_lpt_meta;  // num_cores*2 u32
    std::size_t cap_lpt_meta_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_counts;  // num_tiles u32
    std::size_t cap_tile_counts_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_P_kept;  // 1-page scalar
    std::shared_ptr<distributed::MeshBuffer> buf_cull_mask_base;  // per-tile page-aligned mask offset
    std::size_t cap_cull_mask_base_bytes = 0;

    // Iter 53+: post-radix PACK2 subchunk payloads + device-resident directory.
    distributed::MeshWorkload wl_subchunk;
    KernelHandle ksubchunk{};
    distributed::MeshWorkload wl_subchunk_dir;
    KernelHandle ksubchunk_dir{};
    std::shared_ptr<distributed::MeshBuffer> buf_blend_subchunk_meta;
    std::size_t cap_blend_subchunk_meta_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_subchunk_payload;
    std::size_t cap_subchunk_payload_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_subchunk_dir;
    std::size_t cap_subchunk_dir_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_subchunk_prefix;
    std::size_t cap_subchunk_prefix_bytes = 0;
    // iter 130: per-core materialize WORK-ITEM list (flat u32: tile_id, sc, ...)
    // — subchunk-granular, gather-cost-weighted balance (see build_mat_worklist).
    std::shared_ptr<distributed::MeshBuffer> buf_mat_work;
    std::size_t cap_mat_work_bytes = 0;
};

static std::shared_ptr<distributed::MeshBuffer> make_dram(
    distributed::MeshDevice* dev, std::size_t bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = PAGE_BYTES, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

static std::shared_ptr<distributed::MeshBuffer> make_dram_paged(
    distributed::MeshDevice* dev, std::size_t bytes, std::size_t page_bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = page_bytes, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

static void build_program(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto big_cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    big_cb(0, SCRATCH_BYTES);  // CB_KIN
    big_cb(1, SCRATCH_BYTES);  // CB_IIN
    big_cb(2, SCRATCH_BYTES);  // CB_KOUT
    big_cb(3, SCRATCH_BYTES);  // CB_IOUT
    big_cb(4, PAGE_BYTES);     // CB_TIDS
    big_cb(5, PAGE_BYTES);     // CB_META

    std::vector<uint32_t> ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_radix_tile.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

struct LptAssignment {
    std::vector<uint32_t> flat_tile_ids;
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;
};

struct SubchunkLayout {
    uint32_t total_subchunks = 0;
    uint64_t total_payload_pages = 0;
    uint32_t tiles_split = 0;
    uint32_t max_subchunks_per_tile = 0;
    std::vector<uint32_t> tile_meta;   // num_tiles*2: [dir_base, num_subchunks]
    std::vector<uint32_t> prefix;      // num_tiles: payload page offset
    std::vector<uint32_t> dir;         // total_subchunks*4: page,L,flags,0
};

static SubchunkLayout build_subchunk_layout(
    const std::vector<int64_t>& counts, uint32_t num_tiles, uint32_t bucket_fit) {
    SubchunkLayout layout;
    layout.tile_meta.assign(static_cast<std::size_t>(num_tiles) * 2u, 0u);
    layout.prefix.assign(num_tiles, 0u);
    uint32_t dir_cursor = 0;
    uint64_t page_cursor = 0;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t count = static_cast<uint32_t>(counts[t]);
        const uint32_t num_sc =
            count == 0u ? 1u : (count + bucket_fit - 1u) / bucket_fit;
        layout.tile_meta[static_cast<std::size_t>(t) * 2u + 0u] = dir_cursor;
        layout.tile_meta[static_cast<std::size_t>(t) * 2u + 1u] = num_sc;
        layout.prefix[t] = static_cast<uint32_t>(page_cursor);
        layout.total_subchunks += num_sc;
        if (num_sc > 1u) {
            layout.tiles_split += 1u;
        }
        layout.max_subchunks_per_tile =
            std::max(layout.max_subchunks_per_tile, num_sc);
        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * bucket_fit;
            const uint32_t l_sub = (sc_off >= count) ? 0u
                : ((count - sc_off > bucket_fit) ? bucket_fit : (count - sc_off));
            const uint32_t flags =
                ((sc > 0u) ? 2u : 0u) | ((sc + 1u == num_sc) ? 1u : 0u);
            layout.dir.push_back(static_cast<uint32_t>(page_cursor));
            layout.dir.push_back(l_sub);
            layout.dir.push_back(flags);
            layout.dir.push_back(0u);
            if (l_sub > 0u) {
                page_cursor += (static_cast<uint64_t>(l_sub) + SLAB_RECS_PER_PAGE - 1u) /
                               SLAB_RECS_PER_PAGE;
            }
            dir_cursor += 1u;
        }
    }
    layout.total_payload_pages = page_cursor;
    return layout;
}

static void log_subchunk_layout_stats(const SubchunkLayout& layout) {
    std::fprintf(
        stderr,
        "[SUBCHUNK] tiles_split=%u total_subchunks=%u max_subchunks_per_tile=%u "
        "payload_pages=%llu\n",
        layout.tiles_split,
        layout.total_subchunks,
        layout.max_subchunks_per_tile,
        static_cast<unsigned long long>(layout.total_payload_pages));
}

static void build_program_subchunk(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    const uint32_t bucket_fit = render_config::kBucketFit;
    auto page_cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    // iter-138 (Stage-2b overflow pre-pack): the in-cap overflow path reads the
    // WHOLE overflow tile (up to kOverflowL1Cap records) into CB_BUCKET and L1-
    // radix-sorts it in place. CB_BUCKET already holds bucket_fit*64 B = cap*32 B
    // (cap = 2*bucket_fit) with ZERO growth — the in-budget path only used half of
    // it (bucket_fit/2 PACK2 pages). CB_BSORT (the radix idx/cnt scratch) grows to
    // (2*cap+256) u32 to index cap records. CB_SLAB stays bucket_fit-sized: the
    // sorted slab is streamed to DRAM ONE subchunk (<=bucket_fit recs) at a time.
    const uint32_t ov_cap = render_config::kOverflowL1Cap;
    page_cb(0, PAGE_BYTES);
    page_cb(1, PAGE_BYTES);
    page_cb(2, 32u * PAGE_BYTES);  // REC_BATCH=32 blendrec gather ring (iter 76)
    page_cb(3, 32u);
    page_cb(4, std::max(bucket_fit * 64u, ov_cap * 32u));  // CB_BUCKET (holds cap recs)
    page_cb(5, (2u * ov_cap + 256u) * 4u);                 // CB_BSORT (radix over cap recs)
    // iter 113 (sort Stage 1): CB_SLAB — contiguous L1 scratch the in-budget
    // depth permutation lands in (bucket_fit * 32B records) so the slab is
    // emitted in coalesced SLAB_PAGE_BYTES writes, not per-record DRAM scatter.
    page_cb(6, bucket_fit * 32u);

    std::vector<uint32_t> ct;
    // 9 base accessors + iter-138 {overflow region, per-tile overflow base}.
    for (int i = 0; i < 11; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    }
    ctx.ksubchunk = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_subchunk_materialize.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_subchunk.add_program(device_range, std::move(program));
}

// Iter 55 / step B: device writes blend meta + dir + prefix at sort publish.
static void build_program_subchunk_directory(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRange core0({0, 0}, {0, 0});
    CircularBufferConfig c(PAGE_BYTES, {{0, DataFormat::UInt32}});
    c.set_page_size(0, PAGE_BYTES);
    CreateCircularBuffer(program, core0, c);

    std::vector<uint32_t> ct;
    for (int i = 0; i < 4; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    }
    ctx.ksubchunk_dir = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_subchunk_directory.cpp",
        core0,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_subchunk_dir.add_program(device_range, std::move(program));
}

static bool launch_subchunk_directory(
    SortDeviceContext* ctx,
    uint32_t num_tiles,
    uint32_t bucket_fit) {
    auto brng = device_state::get_buffer("sort_tile_ranges");
    if (!brng || !ctx->buf_blend_subchunk_meta || !ctx->buf_subchunk_dir ||
        !ctx->buf_subchunk_prefix) {
        return false;
    }
    Program& prog = ctx->wl_subchunk_dir.get_programs().begin()->second;
    SetRuntimeArgs(prog, ctx->ksubchunk_dir, CoreCoord{0, 0}, {
        static_cast<uint32_t>(brng->address()),
        static_cast<uint32_t>(ctx->buf_blend_subchunk_meta->address()),
        static_cast<uint32_t>(ctx->buf_subchunk_dir->address()),
        static_cast<uint32_t>(ctx->buf_subchunk_prefix->address()),
        num_tiles,
        bucket_fit,
    });
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_subchunk_dir, false);
    return true;
}

// Host layout sizing + DRAM alloc; device fills meta/dir/prefix at publish.
static bool prepare_subchunk_buffers(
    SortDeviceContext* ctx,
    const SubchunkLayout& layout,
    uint32_t num_tiles) {
    const std::size_t payload_bytes =
        std::max<std::size_t>(SLAB_PAGE_BYTES, layout.total_payload_pages * SLAB_PAGE_BYTES);
    if (!ctx->buf_subchunk_payload ||
        ctx->cap_subchunk_payload_bytes < payload_bytes) {
        ctx->buf_subchunk_payload =
            make_dram_paged(ctx->mesh_device.get(), payload_bytes, SLAB_PAGE_BYTES);
        ctx->cap_subchunk_payload_bytes = payload_bytes;
        device_state::register_buffer("sort_subchunk_payload", ctx->buf_subchunk_payload);
    }
    const std::size_t dir_bytes = round_up(
        std::max<std::size_t>(PAGE_BYTES, layout.dir.size() * 4u), PAGE_BYTES);
    if (!ctx->buf_subchunk_dir || ctx->cap_subchunk_dir_bytes < dir_bytes) {
        ctx->buf_subchunk_dir = make_dram(ctx->mesh_device.get(), dir_bytes);
        ctx->cap_subchunk_dir_bytes = dir_bytes;
        device_state::register_buffer("sort_subchunk_dir", ctx->buf_subchunk_dir);
    }
    const std::size_t prefix_bytes = round_up(
        static_cast<std::size_t>(round_up(num_tiles, ELEMS_PER_PAGE)) * 4u, PAGE_BYTES);
    if (!ctx->buf_subchunk_prefix ||
        ctx->cap_subchunk_prefix_bytes < prefix_bytes) {
        ctx->buf_subchunk_prefix = make_dram(ctx->mesh_device.get(), prefix_bytes);
        ctx->cap_subchunk_prefix_bytes = prefix_bytes;
        device_state::register_buffer("sort_subchunk_prefix", ctx->buf_subchunk_prefix);
    }
    const std::size_t blend_meta_bytes = round_up(
        static_cast<std::size_t>(round_up(num_tiles, ELEMS_PER_PAGE)) * 8u, PAGE_BYTES);
    if (!ctx->buf_blend_subchunk_meta ||
        ctx->cap_blend_subchunk_meta_bytes < blend_meta_bytes) {
        ctx->buf_blend_subchunk_meta =
            make_dram(ctx->mesh_device.get(), blend_meta_bytes);
        ctx->cap_blend_subchunk_meta_bytes = blend_meta_bytes;
        device_state::register_buffer(
            "blend_subchunk_meta", ctx->buf_blend_subchunk_meta);
    }
    return true;
}

// iter 130: materialize work-item assignment — balance at (tile, subchunk)
// granularity. iter-130 MEASURED the dominant materialize cost as the OVERFLOW
// gather (24.6 ms/view busiest-core vs the in-budget permute's 1.7 ms), and the
// shared per-tile count-LPT overloads cores owning big overflow tiles (max 27.1
// vs the 17.0 ms balanced floor). Each (tile, sc) item is independent and writes
// byte-identical output regardless of which core runs it (in-budget reads
// buf_l1_recs by tile / writes payload by (tile,sc); gather reads sorted_ids +
// blendrec by global id / writes payload by (tile,sc)). So greedily LPT-balance
// all items, weighting gather subchunks GATHER_WEIGHT x their record count.
struct MatWorkAssignment {
    std::vector<uint32_t> flat;             // 2 u32 / item: {tile_id, sc}
    std::vector<uint32_t> per_core_offset;  // in ITEMS
    std::vector<uint32_t> per_core_count;   // in ITEMS
    uint32_t max_items_per_core = 0;
};

static MatWorkAssignment build_mat_worklist(
    const std::vector<int64_t>& counts,
    uint32_t num_tiles,
    uint32_t num_cores,
    uint32_t bucket_fit) {
    constexpr uint64_t GATHER_WEIGHT = 8;  // gather ~8-10x an in-budget record
    // iter-138: overflow tiles within the L1 cap are pre-packed at emit; the
    // materialize path reads the WHOLE tile coalesced + L1-radix-permutes it in a
    // SINGLE work item (sc==0, processes every subchunk internally) — like the
    // in-budget permute, ~1x per record (NOT the GATHER_WEIGHT random gather).
    const uint32_t ov_cap = render_config::kOverflowL1Cap;
    struct Item { uint32_t tile; uint32_t sc; uint64_t cost; };
    std::vector<Item> items;
    items.reserve(static_cast<std::size_t>(num_tiles) + 256u);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t cnt = static_cast<uint32_t>(counts[t]);
        if (cnt == 0u) continue;
        const bool inbudget = (cnt <= bucket_fit);
        const bool prepack_ov = (cnt > bucket_fit && cnt <= ov_cap);
        if (inbudget || prepack_ov) {
            // ONE whole-tile item: coalesced bucket read + L1 depth permute.
            items.push_back({t, 0u, static_cast<uint64_t>(cnt)});
            continue;
        }
        // Over-cap overflow tile: legacy per-subchunk blendrec gather.
        const uint32_t num_sc = (cnt + bucket_fit - 1u) / bucket_fit;
        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * bucket_fit;
            const uint32_t l_sub = (sc_off >= cnt) ? 0u
                : ((cnt - sc_off > bucket_fit) ? bucket_fit : (cnt - sc_off));
            if (l_sub == 0u) continue;
            items.push_back({t, sc, static_cast<uint64_t>(l_sub) * GATHER_WEIGHT});
        }
    }
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.cost > b.cost; });
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> per_core(num_cores);
    std::vector<uint64_t> load(num_cores, 0);
    for (const auto& it : items) {
        const auto m = std::min_element(load.begin(), load.end());
        const uint32_t c = static_cast<uint32_t>(std::distance(load.begin(), m));
        per_core[c].emplace_back(it.tile, it.sc);
        load[c] += it.cost;
    }
    MatWorkAssignment a;
    a.per_core_offset.assign(num_cores, 0);
    a.per_core_count.assign(num_cores, 0);
    for (uint32_t c = 0; c < num_cores; ++c) {
        a.per_core_offset[c] = static_cast<uint32_t>(a.flat.size() / 2u);
        a.per_core_count[c] = static_cast<uint32_t>(per_core[c].size());
        a.max_items_per_core = std::max(a.max_items_per_core, a.per_core_count[c]);
        for (const auto& pr : per_core[c]) {
            a.flat.push_back(pr.first);
            a.flat.push_back(pr.second);
        }
    }
    return a;
}

// Device post-radix PACK2 materialize (enqueue only; caller Finish()).
// Kernel: in-budget sc==0 uses buf_l1_recs bulk; overflow sc==0/sc>=1 use sorted_ids gather.
static bool launch_subchunk_materialize(
    SortDeviceContext* ctx,
    const MatWorkAssignment& work,
    uint32_t num_cores,
    uint32_t tiles_x,
    uint32_t bucket_fit) {
    if (work.flat.empty()) {
        return true;
    }
    auto bbrec = device_state::get_buffer("proj_m_blendrec");
    auto bl1 = device_state::get_buffer("sort_l1_recs");
    auto bsids = device_state::get_buffer("sort_sorted_ids");
    auto brng = device_state::get_buffer("sort_tile_ranges");
    if (!bbrec || !bl1 || !bsids || !brng || !ctx->buf_blend_subchunk_meta ||
        !ctx->buf_subchunk_dir) {
        return false;
    }
    // Grow-only work buffer; EnqueueWriteMeshBuffer writes the WHOLE buffer, so
    // the host vector is padded to capacity (mirrors buf_sorted_ids).
    const std::size_t work_bytes =
        round_up(std::max<std::size_t>(work.flat.size() * 4u, PAGE_BYTES), PAGE_BYTES);
    if (!ctx->buf_mat_work || ctx->cap_mat_work_bytes < work_bytes) {
        ctx->buf_mat_work = make_dram(ctx->mesh_device.get(), work_bytes);
        ctx->cap_mat_work_bytes = work_bytes;
    }
    const uint32_t cap_elems = static_cast<uint32_t>(ctx->cap_mat_work_bytes / 4);
    std::vector<uint32_t> wbuf(cap_elems, 0);
    for (std::size_t i = 0; i < work.flat.size(); ++i) wbuf[i] = work.flat[i];
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_mat_work, wbuf, false);

    // iter-138: overflow region + per-tile overflow base for the coalesced path.
    // Both 0 ⇒ no in-cap overflow tiles this view (kernel keeps gather/in-budget).
    const uint32_t ov_addr = ctx->buf_l1_ov
        ? static_cast<uint32_t>(ctx->buf_l1_ov->address()) : 0u;
    const uint32_t ov_base_addr = ctx->buf_tile_ov_base
        ? static_cast<uint32_t>(ctx->buf_tile_ov_base->address()) : 0u;
    Program& prog = ctx->wl_subchunk.get_programs().begin()->second;
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
        SetRuntimeArgs(prog, ctx->ksubchunk, core, {
            static_cast<uint32_t>(bsids->address()),
            static_cast<uint32_t>(brng->address()),
            static_cast<uint32_t>(bbrec->address()),
            static_cast<uint32_t>(bl1->address()),
            static_cast<uint32_t>(ctx->buf_subchunk_payload->address()),
            static_cast<uint32_t>(ctx->buf_blend_subchunk_meta->address()),
            static_cast<uint32_t>(ctx->buf_subchunk_dir->address()),
            static_cast<uint32_t>(ctx->buf_mat_work->address()),
            work.per_core_offset[c],
            work.per_core_count[c],
            tiles_x,
            bucket_fit,
            ov_addr,
            ov_base_addr,
            render_config::kOverflowL1Cap,
        });
    }
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_subchunk, false);
    return true;
}

static void build_program_publish(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto page_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    page_cb(0);  // CB_TIDS
    page_cb(1);  // CB_META
    page_cb(2);  // CB_SCRATCH

    std::vector<uint32_t> ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kpublish = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_publish.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_publish.add_program(device_range, std::move(program));
}

static bool fused_tile_enabled() { return false; }  // FUSED_TILE=0

static bool tile_bucket_enabled() { return true; }  // TILE_BUCKET=1

// R4/R5 binning program: one data-movement kernel (count + scatter modes).
static void build_program_bin(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    cb(0, PAGE_BYTES);        // gid_in
    cb(1, PAGE_BYTES);        // tid_in
    cb(2, PAGE_BYTES);        // keep_in
    cb(3, PAGE_BYTES);        // depth
    cb(4, BIN_ROW_BYTES);     // row (hist out / base in)
    cb(5, BIN_ROW_BYTES);     // cur (local per-tile cursor)
    cb(6, BIN_ROW_BYTES);     // off (local per-tile L1 offset)
    cb(7, BIN_LOCAL_BYTES);   // ksort (L1 counting-sort keys)
    cb(8, BIN_LOCAL_BYTES);   // isort (L1 counting-sort ids)

    const bool tile_bucket = tile_bucket_enabled();
    const bool l1_record = gsplat_tt::env_config::l1_record_enabled();
    if (tile_bucket) {
        cb(9, 16u * PAGE_BYTES);  // rec staging ring (REC_BATCH=16 blendrec pages)
        cb(10, BIN_ROW_BYTES);    // recrow (per-(core,tile) DENSE record base)
    }
    if (l1_record) {
        // cb(11): per-(core,tile) L1 slot base (same shape as recrow/BIN_ROW_BYTES)
        cb(11, BIN_ROW_BYTES);
        // cb(12): REC_BATCH × 32B staging area for packing L1 records before write
        // (512B), plus REC_BATCH × 4B for the optional M1 gid stash (L1_SORT_VERIFY).
        cb(12, 16u * 32u + 16u * 4u);  // 512B staging + 64B gid scratch
        // iter-138: cb(14) per-(core,tile) overflow-region slot base row (sentinel
        // = tile is not a pre-packed overflow tile). Same shape as cb(11).
        cb(14, BIN_ROW_BYTES);
    }
    // iter 132: cb(13) — PACKOC_BATCH × 16B ring staging the per-gaussian blendrec
    // chunk [words 8,9,10,11] (orig cb/depth + packed op/color) written back
    // 16B-aligned (offset 32) to blendrec for materialize. 16B is the natural DRAM
    // write granule (sub-16B / unaligned writes to blendrec do not land on this BH).
    // Sized to the kernel's PACKOC_BATCH=16 (256B). Allocated unconditionally (used
    // whenever the blendrec record is read, i.e. tile_bucket).
    cb(13, 16u * 16u);

    std::vector<uint32_t> ct;
    // Accessors: 7 base + 3 tile_bucket + 2 l1_record + 2 l1_overflow (iter-138)
    int bin_accessors = 7;
    if (tile_bucket) bin_accessors += 3;  // blendrec, tile_recs, recbase
    if (l1_record)   bin_accessors += 2;  // l1_recs, l1_rec_base
    if (l1_record)   bin_accessors += 2;  // iter-138: l1_overflow, l1_overflow_base
    for (int i = 0; i < bin_accessors; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    // Single-path bin kernel: BIN_EMIT_REC / L1_BUCKET_REC are inlined in the
    // kernel source; the debug/verify defines (BIN_NO_DEPTH, BIN_DUMP,
    // L1_SORT_VERIFY) were removed with their kernel branches.
    std::map<std::string, std::string> defines;
    ctx.kbin = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
            .defines = defines,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_bin.add_program(device_range, std::move(program));
}

// S5.4 (iter-125): number of cores the parallel Pass-2 base emit is spread over.
// The coordinator publishes one prefix checkpoint per worker (W × 2 × row_span u32
// of serial DRAM writes), so W trades emit parallelism against checkpoint-publish
// cost — ~16 is near the makespan minimum (the single serial Pass-1 count read
// dominates the coordinator either way). Clamped to num_cores at enqueue time.
constexpr uint32_t kLayoutEmitWorkers = 16u;

static void build_program_bin_layout(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreCoord core0{0, 0};
    const CoreRangeSet cores(core0);
    // Bulk-row layout kernel L1 budget (single core). MAX_TILES/MAX_CORES mirror
    // the kernel's compile-time bounds.
    constexpr uint32_t KMAX_TILES = 2048;
    constexpr uint32_t KMAX_CORES = 128;
    constexpr uint32_t scratch_bytes = (5u * KMAX_TILES + 5u * KMAX_CORES) * 4u;
    constexpr uint32_t row_bytes = KMAX_TILES * 4u;       // one core row (hist/bin2d/l1base)
    constexpr uint32_t out_bytes = 2u * KMAX_TILES * 4u;  // per-tile staging (tmeta/ranges/...)
    // S5.2 histogram cache: hold the whole per-core histogram in L1 so the layout
    // kernel streams DRAM once (Pass 1) and the base-emit pass reads from L1. Cap
    // = 128*1024 u32 = 512 KB (fits hero's 110*1024 ≈ 450 KB); the kernel falls
    // back to a 2nd DRAM read pass if num_cores*row_span exceeds this. MUST match
    // the kernel's HCACHE_CAP constant in sort_bin_layout.cpp.
    constexpr uint32_t KHCACHE_CAP = 128u * 1024u;
    constexpr uint32_t hcache_bytes = KHCACHE_CAP * 4u;
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    cb(0, PAGE_BYTES);      // CB_CTRL
    cb(1, scratch_bytes);   // CB_SCRATCH
    cb(2, row_bytes);       // CB_ROW (fallback row buffer)
    cb(3, row_bytes);       // CB_BIN
    cb(4, row_bytes);       // CB_L1B
    cb(5, out_bytes);       // CB_OUT
    cb(6, hcache_bytes);    // CB_HCACHE (full per-core histogram cache)

    std::vector<uint32_t> ct;
    for (int i = 0; i < 12; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kbin_layout = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin_layout.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_bin_layout.add_program(device_range, std::move(program));
}

// S5.4 (iter-125): parallel Pass-2 base emit. Each worker core emits the bin2d
// base + l1_rec_base rows for a contiguous range of source-cores, seeded from the
// coordinator's per-worker prefix checkpoint. Created on the full grid; cores
// without an assigned range get core_count=0 and no-op.
static void build_program_bin_layout_emit(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    constexpr uint32_t KMAX_TILES = 2048;
    constexpr uint32_t row_bytes = KMAX_TILES * 4u;  // one row (page_acc/rec_acc/hist/bin/l1b)
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    cb(0, PAGE_BYTES);  // CB_CTRL (status)
    cb(1, row_bytes);   // CB_PAGE (page_acc)
    cb(2, row_bytes);   // CB_REC  (rec_acc)
    cb(3, row_bytes);   // CB_HIST (source-core hist row)
    cb(4, row_bytes);   // CB_BIN  (bin2d base out)
    cb(5, row_bytes);   // CB_L1B  (l1_rec_base out)

    std::vector<uint32_t> ct;
    // 4 DRAM-interleaved accessors: bin2d, l1_base, ckpt, ctrl.
    for (int i = 0; i < 4; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kbin_layout_emit = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin_emit.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_bin_layout_emit.add_program(device_range, std::move(program));
}

// ROUTE C: 3-kernel bucket-cull program. reader_bucket_cull streams each LPT
// tile's dense records into the SAME microblock_cull_compute SFPU kernel the
// blend-side cull used; writer_bucket_cull packs the 32-bit keep mask and RMWs
// it into record word 10. Because it is dispatched in the SORT stage, the mask
// reads back spin-free in the downstream L1 blend (no cull_masks DRAM, no spin).
static void build_program_bucket_cull(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };
    // CB ids must match the three kernels exactly:
    //   reader: 0,1 (box ramps) 2 (coeff) 3 (counts) 4 (record/ids/meta scratch)
    //   compute: 0,1,2,3,16
    //   writer: 6 (RMW scratch) 16 (keep)
    cb_cfg(0, RAMP_TILE_BYTES, 1, DataFormat::Float32);   // CB_BOX_OX
    cb_cfg(1, RAMP_TILE_BYTES, 1, DataFormat::Float32);   // CB_BOX_OY
    cb_cfg(2, PAGE_BYTES, 32, DataFormat::Float32);       // CB_CULL_COEFF (7 used)
    cb_cfg(3, PAGE_BYTES, 2, DataFormat::UInt32);         // CB_CULL_COUNTS
    cb_cfg(4, 16u * PAGE_BYTES, 1, DataFormat::UInt32);   // reader scratch (16 records)
    cb_cfg(6, 32u * PAGE_BYTES, 1, DataFormat::UInt32);   // writer RMW scratch (32 records)
    cb_cfg(16, RAMP_TILE_BYTES, 4, DataFormat::Float32);  // CB_KEEP

    // Reader: 5 DRAM-interleaved accessors (tile_recs, bucket_meta, box_ox,
    // box_oy, tile_ids).
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    ctx.kc_reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_bucket_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[0] = UnpackToDestMode::UnpackToDestFp32;
    u2d[1] = UnpackToDestMode::UnpackToDestFp32;
    ctx.kc_compute = CreateKernel(
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

    // Writer: 3 DRAM-interleaved accessors (tile_recs, bucket_meta, tile_ids).
    std::vector<uint32_t> writer_ct;
    for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.kc_writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_bucket_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_cull.add_program(device_range, std::move(program));
    ctx.cull_built = true;
}

static SortDeviceContext init_context() {
    SortDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    build_program_bin(ctx);
    build_program_bin_layout(ctx);
    build_program_bin_layout_emit(ctx);
    build_program_publish(ctx);
    build_program_subchunk(ctx);
    build_program_subchunk_directory(ctx);
    if (bucket_mask_enabled()) {
        build_program_bucket_cull(ctx);
    }
    return ctx;
}

static std::unique_ptr<SortDeviceContext>& context_slot() {
    static std::unique_ptr<SortDeviceContext> ctx;
    return ctx;
}

static SortDeviceContext* ensure_context() {
    auto& slot = context_slot();
    if (!slot) {
        try {
            slot = std::make_unique<SortDeviceContext>(init_context());
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::sort] device init failed: " << e.what() << "\n";
            slot.reset();
        }
    }
    return slot.get();
}

// LPT (longest-processing-time) tile->core assignment over non-empty tiles.
// Mirrors blend_device.cpp compute_lpt_assignment: heaviest tiles first, each
// onto the currently-least-loaded core. Empty tiles never reach the kernel.

// Contiguous page-range split of num_pages over num_cores (matches the
// gather/tile_assign convention so count + scatter use identical ranges).
struct PageSplit {
    std::vector<uint32_t> start;
    std::vector<uint32_t> count;
};
static PageSplit split_pages(uint32_t num_pages, uint32_t num_cores) {
    PageSplit ws;
    ws.start.assign(num_cores, 0);
    ws.count.assign(num_cores, 0);
    const uint32_t base = num_pages / num_cores;
    const uint32_t rem = num_pages % num_cores;
    uint32_t cursor = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        const uint32_t cnt = base + (c < rem ? 1u : 0u);
        ws.start[c] = cursor;
        ws.count[c] = cnt;
        cursor += cnt;
    }
    return ws;
}

static LptAssignment build_lpt(
    const std::vector<int64_t>& counts, uint32_t num_tiles, uint32_t num_cores) {
    std::vector<std::pair<uint32_t, uint32_t>> cost_id;
    cost_id.reserve(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t c = static_cast<uint32_t>(counts[t]);
        if (c > 0) cost_id.emplace_back(c, t);
    }
    std::sort(cost_id.begin(), cost_id.end(), std::greater<>());

    std::vector<std::vector<uint32_t>> per_core(num_cores);
    std::vector<uint64_t> load(num_cores, 0);
    for (const auto& [cost, id] : cost_id) {
        const auto it = std::min_element(load.begin(), load.end());
        const uint32_t c = static_cast<uint32_t>(std::distance(load.begin(), it));
        per_core[c].push_back(id);
        load[c] += cost;
    }

    LptAssignment a;
    a.per_core_offset.assign(num_cores, 0);
    a.per_core_count.assign(num_cores, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        a.per_core_offset[c] = static_cast<uint32_t>(a.flat_tile_ids.size());
        a.per_core_count[c] = static_cast<uint32_t>(per_core[c].size());
        a.flat_tile_ids.insert(
            a.flat_tile_ids.end(), per_core[c].begin(), per_core[c].end());
    }

    return a;
}

struct BinLayoutResult {
    std::vector<uint32_t> hist;  // page-aligned bases in bin2d layout
    std::vector<int64_t> counts;
    std::vector<int64_t> starts;
    std::vector<uint32_t> pstart_page;
    std::vector<uint32_t> pstart_elem;
    std::vector<uint32_t> tile_pad;
    std::vector<uint32_t> histrec;
    std::vector<uint32_t> bucket_meta;
    // M0: per-(core,tile) 32B record slot base in buf_l1_recs.
    // l1_base[c*stride+t] = t*bucket_fit + sum_{c'<c} count[c'][t].
    // Populated when l1_record is true.
    std::vector<uint32_t> histrec_l1;
    // iter-138: per-(core,tile) start slot in the compact overflow region
    // (0xFFFFFFFF for tiles that are NOT pre-packed overflow tiles), and the
    // per-TILE start slot (even-aligned for PACK2; 0xFFFFFFFF otherwise). Both
    // populated when l1_record is true. ov_total_slots = region size in 32B slots.
    std::vector<uint32_t> histrec_overflow;
    std::vector<uint32_t> tile_ov_base;
    uint64_t ov_total_slots = 0;
    uint32_t ov_tiles = 0;       // # tiles routed to the pre-pack path
    uint64_t ov_records = 0;     // # records in those tiles (the coalesced win)
    LptAssignment lpt;
    uint32_t P_kept = 0;
    uint32_t P_aligned = 0;
    uint32_t max_pad_n = 0;
    uint32_t status = 0;  // 0=ok, 1=BIN_LOCAL_MAX, 2=MAX_TILE_ENTRIES
};

static BinLayoutResult host_bin_layout_from_hist(
    const std::vector<uint32_t>& hist_in,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool tile_bucket,
    bool l1_record = false,
    uint32_t bucket_fit = 8192u) {
    BinLayoutResult r;
    r.hist = hist_in;
    r.counts.assign(num_tiles, 0);
    for (uint32_t t = 0; t < num_tiles; t++) {
        uint64_t s = 0;
        for (uint32_t c = 0; c < num_cores; c++)
            s += r.hist[static_cast<std::size_t>(c) * stride + t];
        r.counts[t] = static_cast<int64_t>(s);
    }
    uint32_t max_core_padded = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        uint64_t s = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint64_t h = r.hist[static_cast<std::size_t>(c) * stride + t];
            s += ((h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE) * ELEMS_PER_PAGE;
        }
        if (s > max_core_padded) max_core_padded = static_cast<uint32_t>(s);
    }
    if (max_core_padded > BIN_LOCAL_MAX) {
        r.status = 1;
        return r;
    }
    r.starts.assign(num_tiles, 0);
    r.pstart_page.assign(num_tiles, 0);
    r.pstart_elem.assign(num_tiles, 0);
    r.tile_pad.assign(num_tiles, 0);
    if (tile_bucket) {
        r.histrec.assign(static_cast<std::size_t>(num_cores) * stride, 0u);
        r.bucket_meta.assign(static_cast<std::size_t>(num_tiles) * 2u, 0u);
    }
    if (l1_record) {
        r.histrec_l1.assign(static_cast<std::size_t>(num_cores) * stride, 0u);
        // iter-138: prefix-allocate the COMPACT overflow region over in-cap
        // overflow tiles only (kBucketFit < count <= kOverflowL1Cap). Each such
        // tile's base is EVEN-aligned so its PACK2 page run starts at half 0
        // (the materialize reader indexes record g at page base/2 + g/2, half g&1).
        r.histrec_overflow.assign(
            static_cast<std::size_t>(num_cores) * stride, 0xFFFFFFFFu);
        r.tile_ov_base.assign(num_tiles, 0xFFFFFFFFu);
        const uint32_t ov_cap = render_config::kOverflowL1Cap;
        uint64_t ov_cursor = 0;  // in 32B slots; kept even per tile for PACK2
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const uint64_t cnt = static_cast<uint64_t>(r.counts[t]);
            if (cnt > bucket_fit && cnt <= ov_cap) {
                r.tile_ov_base[t] = static_cast<uint32_t>(ov_cursor);
                ov_cursor += (cnt + 1u) & ~static_cast<uint64_t>(1u);  // round up to even
                r.ov_tiles += 1u;
                r.ov_records += cnt;
            }
        }
        r.ov_total_slots = ov_cursor;
    }
    int64_t cstart = 0;
    uint32_t apage = 0;
    uint32_t max_pad_n = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        const int64_t creal = r.counts[t];
        r.starts[t] = cstart;
        cstart += creal;
        r.pstart_page[t] = apage;
        r.pstart_elem[t] = apage * ELEMS_PER_PAGE;
        const uint32_t tile_start_page = apage;
        uint32_t rec_run = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            const std::size_t idx = static_cast<std::size_t>(c) * stride + t;
            const uint32_t h = r.hist[idx];
            if (tile_bucket) {
                r.histrec[idx] = static_cast<uint32_t>(r.starts[t]) + rec_run;
            }
            if (l1_record) {
                // M0: pre-sized bucket; tile t starts at slot t*bucket_fit.
                // Per-core base = t*bucket_fit + prefix of cores before this one.
                r.histrec_l1[idx] = t * bucket_fit + rec_run;
                // iter-138: overflow tiles also pre-pack the FULL tile into the
                // compact region at tile_ov_base[t] + (core prefix). Sentinel ⇒
                // non-overflow tile (emit keeps the buf_l1_recs bucket clamp path).
                if (r.tile_ov_base[t] != 0xFFFFFFFFu) {
                    r.histrec_overflow[idx] = r.tile_ov_base[t] + rec_run;
                }
            }
            if (tile_bucket || l1_record) {
                rec_run += h;
            }
            r.hist[idx] = apage * ELEMS_PER_PAGE;
            if (h > 0) apage += (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
        }
        const uint32_t pad_pages = apage - tile_start_page;
        r.tile_pad[t] = pad_pages * ELEMS_PER_PAGE;
        if (tile_bucket) {
            r.bucket_meta[static_cast<std::size_t>(t) * 2 + 0] =
                static_cast<uint32_t>(r.starts[t]);
            r.bucket_meta[static_cast<std::size_t>(t) * 2 + 1] = static_cast<uint32_t>(creal);
        }
        if (r.tile_pad[t] > max_pad_n) max_pad_n = r.tile_pad[t];
    }
    if (max_pad_n > MAX_TILE_ENTRIES) {
        r.status = 2;
        return r;
    }
    r.P_kept = static_cast<uint32_t>(cstart);
    r.max_pad_n = max_pad_n;
    r.P_aligned = std::max<uint32_t>(apage, 1u) * ELEMS_PER_PAGE;
    if (l1_record) {
        // iter-138 feasibility diagnostic: how the GATHERED records (all records of
        // tiles with count > bucket_fit) split across cap buckets. The pre-pack path
        // captures the (bucket_fit, kOverflowL1Cap] band; the rest still gathers.
        uint64_t gathered_total = 0, in_cap = 0, over_cap = 0;
        uint32_t over_cap_tiles = 0, max_tile = 0;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const uint64_t cnt = static_cast<uint64_t>(r.counts[t]);
            if (cnt > max_tile) max_tile = static_cast<uint32_t>(cnt);
            if (cnt > bucket_fit) {
                gathered_total += cnt;
                if (cnt <= render_config::kOverflowL1Cap) in_cap += cnt;
                else { over_cap += cnt; over_cap_tiles += 1u; }
            }
        }
        std::fprintf(stderr,
            "[OVERFLOW-DIST] bucket_fit=%u cap=%u num_tiles=%u P_kept=%u max_tile=%u "
            "| overflow_tiles=%u(prepack)+%u(gather) gathered_recs=%llu = in_cap=%llu "
            "(%.1f%% prepacked) + over_cap=%llu | ov_region_slots=%llu (~%.1f MB DRAM)\n",
            bucket_fit, render_config::kOverflowL1Cap, num_tiles, r.P_kept, max_tile,
            r.ov_tiles, over_cap_tiles,
            static_cast<unsigned long long>(gathered_total),
            static_cast<unsigned long long>(in_cap),
            gathered_total ? 100.0 * static_cast<double>(in_cap) /
                             static_cast<double>(gathered_total) : 0.0,
            static_cast<unsigned long long>(over_cap),
            static_cast<unsigned long long>(r.ov_total_slots),
            static_cast<double>(r.ov_total_slots) * 32.0 / (1024.0 * 1024.0));
    }
    std::vector<int64_t> pad_counts(num_tiles, 0);
    for (uint32_t t = 0; t < num_tiles; t++)
        pad_counts[t] = static_cast<int64_t>(r.tile_pad[t]);
    r.lpt = build_lpt(pad_counts, num_tiles, num_cores);
    return r;
}

// S5.4 (iter-125): enqueue the parallel Pass-2 base emit. Splits the num_cores
// source rows into W contiguous ranges (must match the coordinator's checkpoint
// split exactly), one per worker core; grid cores beyond W get core_count=0.
static void enqueue_bin_layout_emit_kernel(
    SortDeviceContext* ctx,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool l1_record,
    uint32_t bucket_fit,
    uint32_t ckpt_addr,
    uint32_t W) {
    Program& prog = ctx->wl_bin_layout_emit.get_programs().begin()->second;
    const uint32_t grid_cores = ctx->grid.x * ctx->grid.y;
    const uint32_t base_c = num_cores / W;
    const uint32_t rem_c = num_cores % W;
    const uint32_t l1_base_addr = (l1_record && ctx->buf_l1_rec_base)
        ? static_cast<uint32_t>(ctx->buf_l1_rec_base->address())
        : 0u;
    uint32_t cc = 0;
    for (uint32_t g = 0; g < grid_cores; g++) {
        CoreCoord core{g % ctx->grid.x, g / ctx->grid.x};
        uint32_t cstart = 0, ccount = 0, widx = 0;
        if (g < W) {
            const uint32_t cnt = base_c + (g < rem_c ? 1u : 0u);
            cstart = cc;
            ccount = cnt;
            widx = g;
            cc += cnt;
        }
        SetRuntimeArgs(prog, ctx->kbin_layout_emit, core, {
            static_cast<uint32_t>(ctx->buf_bin2d->address()),
            l1_base_addr,
            ckpt_addr,
            static_cast<uint32_t>(ctx->buf_bin_ctrl->address()),
            num_tiles,
            stride,
            bucket_fit,
            cstart,
            ccount,
            widx,
        });
    }
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin_layout_emit, false);
}

static void enqueue_bin_layout_kernel(
    SortDeviceContext* ctx,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool tile_bucket,
    uint32_t tile_ranges_addr,
    bool l1_record,
    uint32_t bucket_fit) {
    if (!ctx->buf_bin_ctrl) {
        ctx->buf_bin_ctrl = make_dram(ctx->mesh_device.get(), PAGE_BYTES);
    }
    // Checkpoint buffer: W slots × {page_acc row, rec_acc row}, each row_pages
    // 64B pages. W is the emit worker count (clamped to num_cores).
    const uint32_t W = std::min(num_cores, kLayoutEmitWorkers);
    const uint32_t row_pages = (num_tiles + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
    const uint32_t row_span = row_pages * ELEMS_PER_PAGE;
    const std::size_t ckpt_bytes =
        static_cast<std::size_t>(W) * 2u * row_span * 4u;
    if (!ctx->buf_layout_ckpt || ctx->cap_layout_ckpt_bytes < ckpt_bytes) {
        ctx->buf_layout_ckpt = make_dram(ctx->mesh_device.get(), ckpt_bytes);
        ctx->cap_layout_ckpt_bytes = ckpt_bytes;
    }
    const uint32_t ckpt_addr = static_cast<uint32_t>(ctx->buf_layout_ckpt->address());

    Program& prog = ctx->wl_bin_layout.get_programs().begin()->second;
    CoreCoord core0{0, 0};
    SetRuntimeArgs(prog, ctx->kbin_layout, core0, {
        static_cast<uint32_t>(ctx->buf_bin2d->address()),
        static_cast<uint32_t>(ctx->buf_tmeta->address()),
        static_cast<uint32_t>(ctx->buf_tile_ids->address()),
        static_cast<uint32_t>(ctx->buf_lpt_meta->address()),
        static_cast<uint32_t>(ctx->buf_tile_counts->address()),
        static_cast<uint32_t>(ctx->buf_bin_ctrl->address()),
        num_cores,
        num_tiles,
        stride,
        tile_bucket && ctx->buf_bin2d_rec
            ? static_cast<uint32_t>(ctx->buf_bin2d_rec->address())
            : 0u,
        tile_bucket && ctx->buf_bucket_meta
            ? static_cast<uint32_t>(ctx->buf_bucket_meta->address())
            : 0u,
        tile_ranges_addr,
        bucket_fit,
        l1_record && ctx->buf_l1_rec_base
            ? static_cast<uint32_t>(ctx->buf_l1_rec_base->address())
            : 0u,
        W,
        ckpt_addr,
    });
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin_layout, false);
    // Parallel Pass-2 base emit (CQ-ordered after the coordinator: it sees the
    // coordinator's checkpoint writes and the still-intact histogram in bin2d).
    enqueue_bin_layout_emit_kernel(
        ctx, num_cores, num_tiles, stride, l1_record, bucket_fit, ckpt_addr, W);
}

static bool read_bin_layout_ctrl(
    SortDeviceContext* ctx,
    uint32_t& P_kept,
    uint32_t& P_aligned,
    uint32_t& max_pad_n,
    uint32_t& status) {
    std::vector<uint32_t> ctrl(ELEMS_PER_PAGE, 0);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, ctrl, ctx->buf_bin_ctrl, true);
    P_kept = ctrl[0];
    P_aligned = ctrl[1];
    max_pad_n = ctrl[2];
    status = ctrl[3];
    return status == 0;
}

// Publish the contiguous (sorted_ids, tile_ranges) into device_state as uint32
// DRAM buffers so downstream device stages can read them resident.
static bool resident_blend_chain_enabled() { return true; }  // RESIDENT_BLEND=1

static bool sort_device_publish_enabled();

// Drop the sort-publish Finish() and blend's blocking sort_P_kept D2H so the
// publish kernel chains into FUSED_TILE cull+blend with one CQ drain.
static bool sort_blend_pipe_enabled() {
    return gsplat_tt::env_config::sort_blend_pipe_enabled();
}

// Chain upload+scatter+radix(+publish) into the blend/cull CQ drain — drops the
// per-stage Finish locks between sort kernels when publish pipes to blend.
static bool sort_stage_defer_finish() {
    return sort_blend_pipe_enabled();
}

static void finish_sort_cq_if_needed(SortDeviceContext* ctx) {
    if (device_state::sort_publish_pending()) {
        GSPLAT_HOST_ZONE("host_finish_sort");
        distributed::Finish(*ctx->cq);
        device_state::clear_sort_publish_pending();
    }
}

// Publish LPT tile-id list + per-core (offset,count) and per-tile kept counts
// so blend/cull never scan host tile_ranges or rebuild LPT from host vectors.
static void publish_sort_downstream_metadata(
    SortDeviceContext* ctx,
    const LptAssignment& lpt,
    const std::vector<int64_t>& counts,
    uint32_t num_tiles,
    uint32_t num_cores) {
    device_state::register_buffer("sort_lpt_tile_ids", ctx->buf_tile_ids);

    const uint32_t meta_elems = num_cores * 2;
    const uint32_t meta_pad = round_up(std::max(meta_elems, 1u), ELEMS_PER_PAGE);
    const std::size_t meta_bytes = static_cast<std::size_t>(meta_pad) * 4;
    if (!ctx->buf_lpt_meta || ctx->cap_lpt_meta_bytes < meta_bytes) {
        ctx->buf_lpt_meta = make_dram(ctx->mesh_device.get(), meta_bytes);
        ctx->cap_lpt_meta_bytes = meta_bytes;
        device_state::register_buffer("sort_lpt_meta", ctx->buf_lpt_meta);
    }
    const uint32_t cap_meta = static_cast<uint32_t>(ctx->cap_lpt_meta_bytes / 4);
    std::vector<uint32_t> meta(cap_meta, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        meta[c * 2 + 0] = lpt.per_core_offset[c];
        meta[c * 2 + 1] = lpt.per_core_count[c];
    }
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_lpt_meta, meta, false);

    const uint32_t counts_pad = round_up(std::max(num_tiles, 1u), ELEMS_PER_PAGE);
    const std::size_t counts_bytes = static_cast<std::size_t>(counts_pad) * 4;
    if (!ctx->buf_tile_counts || ctx->cap_tile_counts_bytes < counts_bytes) {
        ctx->buf_tile_counts = make_dram(ctx->mesh_device.get(), counts_bytes);
        ctx->cap_tile_counts_bytes = counts_bytes;
        device_state::register_buffer("sort_tile_counts", ctx->buf_tile_counts);
    }
    const uint32_t cap_counts = static_cast<uint32_t>(ctx->cap_tile_counts_bytes / 4);
    std::vector<uint32_t> cu32(cap_counts, 0);
    for (uint32_t t = 0; t < num_tiles; t++) {
        cu32[t] = static_cast<uint32_t>(counts[t]);
    }
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_counts, cu32, false);

    // Scalar P_kept + padded cull-mask footprint for downstream SFPU cull (blend
    // must not D2H sort_tile_counts to build cull_mask_base).
    if (!ctx->buf_P_kept) {
        ctx->buf_P_kept = make_dram(ctx->mesh_device.get(), PAGE_BYTES);
        device_state::register_buffer("sort_P_kept", ctx->buf_P_kept);
    }
    std::vector<uint32_t> pkept_buf(ELEMS_PER_PAGE, 0);
    uint64_t acc = 0;
    uint64_t mask_elems = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        acc += static_cast<uint64_t>(counts[t]);
        mask_elems += (static_cast<uint64_t>(counts[t]) + 15u) & ~static_cast<uint64_t>(15u);
    }
    pkept_buf[0] = static_cast<uint32_t>(acc);
    pkept_buf[1] = static_cast<uint32_t>(mask_elems);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_P_kept, pkept_buf, false);
    if (sort_blend_pipe_enabled()) {
        device_state::set_sort_blend_pipe_scalars(
            static_cast<uint32_t>(acc), static_cast<uint32_t>(mask_elems));
    }

    if (resident_blend_chain_enabled()) {
        std::vector<uint32_t> mask_base(counts_pad, 0u);
        uint64_t off = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            mask_base[t] = static_cast<uint32_t>(off);
            off += (static_cast<uint64_t>(counts[t]) + 15u) & ~static_cast<uint64_t>(15u);
        }
        const std::size_t base_bytes = static_cast<std::size_t>(counts_pad) * 4;
        if (!ctx->buf_cull_mask_base || ctx->cap_cull_mask_base_bytes < base_bytes) {
            ctx->buf_cull_mask_base = make_dram(ctx->mesh_device.get(), base_bytes);
            ctx->cap_cull_mask_base_bytes = base_bytes;
            device_state::register_buffer("cull_mask_base", ctx->buf_cull_mask_base);
        }
        distributed::EnqueueWriteMeshBuffer(
            *ctx->cq, ctx->buf_cull_mask_base, mask_base, false);
    }
}

static void publish_resident(
    SortDeviceContext* ctx,
    const std::vector<int64_t>& sorted_ids,
    const std::vector<int64_t>& tile_ranges,
    double* publish_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t P = static_cast<uint32_t>(sorted_ids.size());
    const uint32_t P_pad = round_up(std::max<uint32_t>(P, 1), ELEMS_PER_PAGE);
    const std::size_t sorted_bytes = static_cast<std::size_t>(P_pad) * 4;
    if (!ctx->buf_sorted_ids || ctx->cap_sorted_bytes < sorted_bytes) {
        ctx->buf_sorted_ids = make_dram(ctx->mesh_device.get(), sorted_bytes);
        ctx->cap_sorted_bytes = sorted_bytes;
        device_state::register_buffer("sort_sorted_ids", ctx->buf_sorted_ids);
    }
    // buf_sorted_ids is grow-only: a smaller-P frame keeps a larger (hero)
    // capacity. EnqueueWriteMeshBuffer writes the WHOLE buffer, so the host
    // vector must be sized to capacity (not the current P_pad) or tt-metal
    // asserts "source vector too small" and crashes the 30-view sweep.
    const uint32_t cap_sorted_elems = static_cast<uint32_t>(ctx->cap_sorted_bytes / 4);
    std::vector<uint32_t> sids(cap_sorted_elems, 0);
    for (uint32_t i = 0; i < P; i++)
        sids[i] = static_cast<uint32_t>(sorted_ids[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_sorted_ids, sids, false);

    const uint32_t R = static_cast<uint32_t>(tile_ranges.size());
    const uint32_t R_pad = round_up(std::max<uint32_t>(R, 1), ELEMS_PER_PAGE);
    const std::size_t ranges_bytes = static_cast<std::size_t>(R_pad) * 4;
    if (!ctx->buf_tile_ranges || ctx->cap_ranges_bytes < ranges_bytes) {
        ctx->buf_tile_ranges = make_dram(ctx->mesh_device.get(), ranges_bytes);
        ctx->cap_ranges_bytes = ranges_bytes;
        device_state::register_buffer("sort_tile_ranges", ctx->buf_tile_ranges);
    }
    const uint32_t cap_ranges_elems = static_cast<uint32_t>(ctx->cap_ranges_bytes / 4);
    std::vector<uint32_t> ranges(cap_ranges_elems, 0);
    for (uint32_t i = 0; i < R; i++)
        ranges[i] = static_cast<uint32_t>(tile_ranges[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ranges, ranges, false);
    distributed::Finish(*ctx->cq);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (publish_ms) *publish_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Grow-only resident sort_sorted_ids buffer (no host id upload).
static void ensure_resident_sorted_buffer(SortDeviceContext* ctx, uint32_t P_kept) {
    const uint32_t P_pad = round_up(std::max<uint32_t>(P_kept, 1), ELEMS_PER_PAGE);
    const std::size_t sorted_bytes = static_cast<std::size_t>(P_pad) * 4;
    if (!ctx->buf_sorted_ids || ctx->cap_sorted_bytes < sorted_bytes) {
        ctx->buf_sorted_ids = make_dram(ctx->mesh_device.get(), sorted_bytes);
        ctx->cap_sorted_bytes = sorted_bytes;
        device_state::register_buffer("sort_sorted_ids", ctx->buf_sorted_ids);
    }
}

// Upload tile_ranges only (grow-on-demand whole-buffer write).
static void upload_resident_tile_ranges(
    SortDeviceContext* ctx, const std::vector<int64_t>& tile_ranges) {
    const uint32_t R = static_cast<uint32_t>(tile_ranges.size());
    const uint32_t R_pad = round_up(std::max<uint32_t>(R, 1), ELEMS_PER_PAGE);
    const std::size_t ranges_bytes = static_cast<std::size_t>(R_pad) * 4;
    if (!ctx->buf_tile_ranges || ctx->cap_ranges_bytes < ranges_bytes) {
        ctx->buf_tile_ranges = make_dram(ctx->mesh_device.get(), ranges_bytes);
        ctx->cap_ranges_bytes = ranges_bytes;
        device_state::register_buffer("sort_tile_ranges", ctx->buf_tile_ranges);
    }
    const uint32_t cap_ranges_elems = static_cast<uint32_t>(ctx->cap_ranges_bytes / 4);
    std::vector<uint32_t> ranges(cap_ranges_elems, 0);
    for (uint32_t i = 0; i < R; i++)
        ranges[i] = static_cast<uint32_t>(tile_ranges[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ranges, ranges, false);
}

static bool sort_device_publish_enabled() { return true; }  // SORT_DEVICE_PUBLISH=1

static void maybe_run_sort_blend_continuation(
    SortBlendContinuation* cont, int tiles_x, uint32_t num_tiles) {
    if (cont == nullptr || cont->image_out == nullptr) {
        return;
    }
    if (!sort_blend_pipe_enabled() || !resident_blend_chain_enabled()) {
        return;
    }
    bool blend_ok = false;
    double cull_ms = 0.0, blend_ms = 0.0;
    (void)blend_mb_devcull_resident(
        cont->mb_contrib_floor,
        cont->cull_disabled,
        static_cast<int>(num_tiles),
        tiles_x,
        cont->image_height,
        cont->image_width,
        cont->image_out,
        &blend_ok,
        &cull_ms,
        &blend_ms,
        cont->transmittance_threshold);
    cont->cull_ms = cull_ms;
    cont->blend_ms = blend_ms;
    cont->invoked = true;
    if (cont->blend_ok != nullptr) {
        *cont->blend_ok = blend_ok;
    }
}

// ── R4/R5 resident-pairs device binning ─────────────────────────────────
// Bins the resident full-P (gid,tid) pairs + keep mask into the page-aligned
// per-tile (key,id) layout on-device, runs the radix kernel, compacts, and
// publishes. Returns a SortResult identical in shape to the host path.
static gsplat_cpu::SortResult sort_resident_pairs(
    SortDeviceContext* ctx,
    uint32_t num_tiles,
    int tiles_x,
    int tiles_y,
    std::size_t M,
    bool need_host_sorted_ids,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    SortCallTimings& T,
    SortBlendContinuation* sort_blend) {
    using clk = std::chrono::high_resolution_clock;
    const auto t_total0_rp = clk::now();
    (void)pool;  // resident binning needs no host thread pool; kept for ABI
    auto fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::SortResult{};
    };
    gsplat_cpu::SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);

    if (num_tiles > MAX_BIN_TILES) {
        std::cerr << "[gsplat_tt::sort] num_tiles=" << num_tiles
                  << " > MAX_BIN_TILES=" << MAX_BIN_TILES
                  << "; unsupported (render_clean is single-path TT, no host "
                     "fallback) — hard fail\n";
        return fail();
    }

    auto bgid = device_state::get_buffer("ta_pairs_gid");
    auto btid = device_state::get_buffer("ta_pairs_tid");
    auto bkeep = device_state::get_buffer("ta_pairs_keep");
    auto bP = device_state::get_buffer("ta_pairs_P");
    auto bdep = device_state::get_buffer("proj_m_depth");
    if (!bgid || !btid || !bkeep || !bP || !bdep) {
        std::cerr << "[gsplat_tt::sort] resident pairs / proj_m_depth missing; "
                     "the upstream resident stages did not run — hard fail "
                     "(render_clean is single-path TT, no host fallback)\n";
        return fail();
    }

    try {
        // Read full P + P_pad published by tile_assign.
        std::vector<uint32_t> pbuf(ELEMS_PER_PAGE);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, pbuf, bP, true);
        const uint32_t P_full = pbuf[0];
        const uint32_t P_pad = pbuf[1];
        // S5.3 host-free overflow guard: tile_assign's scan_bases CLAMPS the
        // published P (pbuf[0]) to the static pair ceiling so neither it nor K2
        // ever indexes past the p_max-sized pair buffers (no silent memory
        // corruption). pbuf[2]=overflow / pbuf[3]=P_true surface a too-small
        // ceiling here — the one place that would read OOB — as a hard fail
        // (render_clean is single-path TT, no fallback). This reuses the P read
        // already on this path, so it adds no new mid-frame drain.
        const uint32_t p_overflow = pbuf[2];
        const uint32_t P_true = pbuf[3];
        if (p_overflow != 0) {
            std::cerr << "[gsplat_tt::sort] PAIR-CEILING OVERFLOW: pre-cull P_true="
                      << P_true << " exceeds the static pair_ceiling()=" << P_full
                      << " — pairs were clamped (output would be corrupt). Raise "
                         "env_config::pair_ceiling() above " << P_true
                      << " and rebuild. Hard fail (single-path TT, no fallback).\n";
            return fail();
        }
        if (P_full == 0) {
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);
            if (device_ok) *device_ok = true;
            return result;
        }

        const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
        const uint32_t dump_tile = 0;
        const uint32_t stride = round_up(num_tiles, ELEMS_PER_PAGE);
        const uint32_t total_p_pages = P_pad / ELEMS_PER_PAGE;
        const PageSplit ws = split_pages(total_p_pages, num_cores);

        // ── bin2d (per-core hist / base rows) ───────────────────────────
        const std::size_t bin2d_bytes =
            static_cast<std::size_t>(num_cores) * stride * 4;
        if (!ctx->buf_bin2d || ctx->cap_bin2d_bytes < bin2d_bytes) {
            ctx->buf_bin2d = make_dram(ctx->mesh_device.get(), bin2d_bytes);
            ctx->cap_bin2d_bytes = bin2d_bytes;
        }

        // T1 (GSPLAT_TT_TILE_BUCKET): scatter full records into per-tile buckets.
        const bool tile_bucket = tile_bucket_enabled();
        auto bbrec = tile_bucket ? device_state::get_buffer("proj_m_blendrec") : nullptr;
        if (tile_bucket && !bbrec) {
            std::cerr << "[gsplat_tt::sort] proj_m_blendrec missing; the gather "
                         "stage did not run — hard fail (render_clean is "
                         "single-path TT, no host fallback)\n";
            return fail();
        }
        const uint32_t blendrec_addr = bbrec ? static_cast<uint32_t>(bbrec->address()) : 0u;
        uint32_t tile_recs_addr = 0u;  // real address set after P_aligned is known
        uint32_t recbase_addr = 0u;    // dense per-(core,tile) record base (set w/ buf)

        // Metadata buffers (layout kernel writes these resident on device).
        const uint32_t tmeta_pad = round_up(std::max(num_tiles * 2u, 1u), ELEMS_PER_PAGE);
        const std::size_t tmeta_bytes = static_cast<std::size_t>(tmeta_pad) * 4;
        if (!ctx->buf_tmeta || ctx->cap_tmeta_bytes < tmeta_bytes) {
            ctx->buf_tmeta = make_dram(ctx->mesh_device.get(), tmeta_bytes);
            ctx->cap_tmeta_bytes = tmeta_bytes;
        }
        const uint32_t tile_ids_pad = round_up(std::max(num_tiles, 1u), ELEMS_PER_PAGE);
        const std::size_t tile_ids_bytes = static_cast<std::size_t>(tile_ids_pad) * 4;
        if (!ctx->buf_tile_ids || ctx->cap_tile_ids_bytes < tile_ids_bytes) {
            ctx->buf_tile_ids = make_dram(ctx->mesh_device.get(), tile_ids_bytes);
            ctx->cap_tile_ids_bytes = tile_ids_bytes;
        }
        device_state::register_buffer("sort_lpt_tile_ids", ctx->buf_tile_ids);
        const uint32_t meta_elems = num_cores * 2;
        const uint32_t meta_pad = round_up(std::max(meta_elems, 1u), ELEMS_PER_PAGE);
        const std::size_t meta_bytes = static_cast<std::size_t>(meta_pad) * 4;
        if (!ctx->buf_lpt_meta || ctx->cap_lpt_meta_bytes < meta_bytes) {
            ctx->buf_lpt_meta = make_dram(ctx->mesh_device.get(), meta_bytes);
            ctx->cap_lpt_meta_bytes = meta_bytes;
        }
        device_state::register_buffer("sort_lpt_meta", ctx->buf_lpt_meta);
        const uint32_t counts_pad = round_up(std::max(num_tiles, 1u), ELEMS_PER_PAGE);
        const std::size_t counts_bytes = static_cast<std::size_t>(counts_pad) * 4;
        if (!ctx->buf_tile_counts || ctx->cap_tile_counts_bytes < counts_bytes) {
            ctx->buf_tile_counts = make_dram(ctx->mesh_device.get(), counts_bytes);
            ctx->cap_tile_counts_bytes = counts_bytes;
        }
        device_state::register_buffer("sort_tile_counts", ctx->buf_tile_counts);
        const std::size_t ranges_bytes =
            static_cast<std::size_t>(round_up(num_tiles * 2u, ELEMS_PER_PAGE)) * 4;
        if (!ctx->buf_tile_ranges || ctx->cap_ranges_bytes < ranges_bytes) {
            ctx->buf_tile_ranges = make_dram(ctx->mesh_device.get(), ranges_bytes);
            ctx->cap_ranges_bytes = ranges_bytes;
        }
        device_state::register_buffer("sort_tile_ranges", ctx->buf_tile_ranges);
        if (tile_bucket) {
            const std::size_t rec_base_bytes =
                static_cast<std::size_t>(num_cores) * stride * 4;
            if (!ctx->buf_bin2d_rec || ctx->cap_bin2d_rec_bytes < rec_base_bytes) {
                ctx->buf_bin2d_rec = make_dram(ctx->mesh_device.get(), rec_base_bytes);
                ctx->cap_bin2d_rec_bytes = rec_base_bytes;
            }
            recbase_addr = static_cast<uint32_t>(ctx->buf_bin2d_rec->address());
            const uint32_t bm_pad = round_up(num_tiles * 2u, ELEMS_PER_PAGE);
            const std::size_t bm_bytes = static_cast<std::size_t>(bm_pad) * 4;
            if (!ctx->buf_bucket_meta || ctx->cap_bucket_meta_bytes < bm_bytes) {
                ctx->buf_bucket_meta = make_dram(ctx->mesh_device.get(), bm_bytes);
                ctx->cap_bucket_meta_bytes = bm_bytes;
                device_state::register_buffer("sort_bucket_meta", ctx->buf_bucket_meta);
            }
        }

        // M0: l1_record buffers.
        const bool l1_record_early = gsplat_tt::env_config::l1_record_enabled();
        const uint32_t bucket_fit = render_config::kBucketFit;
        uint32_t l1_recs_addr = 0u;
        uint32_t l1_base_addr = 0u;
        // iter-138: overflow pre-pack region + per-(core,tile) base row. Sized and
        // filled AFTER the host layout pass (count-dependent), so they stay 0 for
        // the count pass (launch_bin(0)) and the kernel's overflow path is disabled
        // there; set to real addresses before the scatter pass (launch_bin(1)).
        uint32_t l1_ov_addr = 0u;
        uint32_t l1_ov_base_addr = 0u;
        if (l1_record_early) {
            // M0/iter50: two 32B splats per 64B DRAM page (PACK2). Sub-64B paging
            // is unreliable; 64B pages hold low/high splat at +0/+32. kBucketFit
            // logical slots => bucket_fit/2 pages per tile.
            const std::size_t l1_rec_bytes =
                static_cast<std::size_t>(num_tiles) * bucket_fit * 32u;
            if (!ctx->buf_l1_recs || ctx->cap_l1_recs_bytes < l1_rec_bytes) {
                ctx->buf_l1_recs = make_dram_paged(ctx->mesh_device.get(), l1_rec_bytes, 64u);
                ctx->cap_l1_recs_bytes = l1_rec_bytes;
                device_state::register_buffer("sort_l1_recs", ctx->buf_l1_recs);
            }
            l1_recs_addr = static_cast<uint32_t>(ctx->buf_l1_recs->address());
            const std::size_t l1_base_bytes =
                static_cast<std::size_t>(num_cores) * stride * 4u;
            if (!ctx->buf_l1_rec_base || ctx->cap_l1_rec_base_bytes < l1_base_bytes) {
                ctx->buf_l1_rec_base = make_dram(ctx->mesh_device.get(), l1_base_bytes);
                ctx->cap_l1_rec_base_bytes = l1_base_bytes;
                device_state::register_buffer("sort_l1_rec_base", ctx->buf_l1_rec_base);
            }
            l1_base_addr = static_cast<uint32_t>(ctx->buf_l1_rec_base->address());
        }

        const bool device_layout = gsplat_tt::env_config::sort_device_layout_enabled();
        const bool layout_verify = false;

        // ── Pass A: per-core histogram (count) ──────────────────────────
        const auto t_bin0 = clk::now();
        auto launch_bin = [&](uint32_t mode, bool finish_cq) {
            Program& prog = ctx->wl_bin.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                std::vector<uint32_t> args = {
                    static_cast<uint32_t>(bgid->address()),
                    static_cast<uint32_t>(btid->address()),
                    static_cast<uint32_t>(bkeep->address()),
                    static_cast<uint32_t>(bdep->address()),
                    static_cast<uint32_t>(ctx->buf_bin2d->address()),
                    ctx->buf_keys ? static_cast<uint32_t>(ctx->buf_keys->address()) : 0u,
                    ctx->buf_ids ? static_cast<uint32_t>(ctx->buf_ids->address()) : 0u,
                    ws.start[c], ws.count[c], P_full, num_tiles, stride, c, mode,
                    dump_tile,
                };
                if (tile_bucket) {
                    args.push_back(blendrec_addr);
                    args.push_back(tile_recs_addr);
                    args.push_back(recbase_addr);
                }
                if (l1_record_early) {
                    args.push_back(l1_recs_addr);
                    args.push_back(l1_base_addr);
                    args.push_back(bucket_fit);  // per-tile bucket slot count (clamp)
                    args.push_back(static_cast<uint32_t>(tiles_x));  // tile-local mean
                    args.push_back(l1_ov_addr);       // iter-138: overflow region (0=off)
                    args.push_back(l1_ov_base_addr);  // iter-138: per-(core,tile) ov base
                }
                SetRuntimeArgs(prog, ctx->kbin, core, args);
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin, false);
            if (finish_cq) {
                if (mode == 0) {
                    GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
                } else {
                    GSPLAT_HOST_ZONE("host_finish_sort_bin_scat");
                }
                distributed::Finish(*ctx->cq);
            }
        };
        launch_bin(0, !device_layout);
        const auto t_cnt = clk::now();

        std::vector<int64_t> counts(num_tiles, 0);
        std::vector<int64_t> starts(num_tiles, 0);
        std::vector<uint32_t> pstart_page(num_tiles, 0);
        std::vector<uint32_t> pstart_elem(num_tiles, 0);
        std::vector<uint32_t> tile_pad(num_tiles, 0);
        LptAssignment lpt;
        uint32_t P_kept = 0;
        uint32_t P_aligned = 0;
        uint32_t max_n = 0;
        clk::time_point t_d2h = t_cnt;
        clk::time_point t_bin1 = t_cnt;

        std::optional<BinLayoutResult> layout_verify_ref;
        if (layout_verify) {
            if (!device_layout) {
                GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
                distributed::Finish(*ctx->cq);
            }
            std::vector<uint32_t> hist_ref(static_cast<std::size_t>(num_cores) * stride);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, hist_ref, ctx->buf_bin2d, true);
            t_d2h = clk::now();
            layout_verify_ref =
                host_bin_layout_from_hist(hist_ref, num_cores, num_tiles, stride, tile_bucket);
            if (layout_verify_ref->status != 0) {
                std::cerr << "[gsplat_tt::sort] layout verify: host ref status="
                          << layout_verify_ref->status << "\n";
            }
            enqueue_bin_layout_kernel(
                ctx,
                num_cores,
                num_tiles,
                stride,
                tile_bucket,
                static_cast<uint32_t>(ctx->buf_tile_ranges->address()),
                l1_record_early,
                bucket_fit);
            GSPLAT_HOST_ZONE("host_finish_sort_bin_layout");
            distributed::Finish(*ctx->cq);
            if (layout_verify_ref->status == 0) {
                const BinLayoutResult& href = *layout_verify_ref;
                uint32_t dev_P_kept = 0, dev_P_aligned = 0, dev_max_n = 0, dev_status = 0;
                read_bin_layout_ctrl(ctx, dev_P_kept, dev_P_aligned, dev_max_n, dev_status);
                std::vector<uint32_t> hist_dev(hist_ref.size());
                std::vector<uint32_t> tmeta_d(tmeta_pad, 0);
                std::vector<uint32_t> tids_d(tile_ids_pad, 0);
                std::vector<uint32_t> lptm(meta_pad, 0);
                std::vector<uint32_t> cnts(counts_pad, 0);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, hist_dev, ctx->buf_bin2d, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, tmeta_d, ctx->buf_tmeta, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, tids_d, ctx->buf_tile_ids, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, lptm, ctx->buf_lpt_meta, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, cnts, ctx->buf_tile_counts, true);
                std::size_t mism_hist = 0, mism_cnt = 0, mism_tids = 0, mism_lpt = 0, mism_tmeta = 0;
                for (std::size_t i = 0; i < hist_dev.size(); i++) {
                    if (hist_dev[i] != href.hist[i]) {
                        mism_hist++;
                        if (mism_hist == 1) {
                            std::fprintf(stderr,
                                "[SORT_LAYOUT_VERIFY] first hist mism i=%zu dev=%u ref=%u\n",
                                i, hist_dev[i], href.hist[i]);
                        }
                    }
                }
                for (uint32_t t = 0; t < num_tiles; t++) {
                    if (cnts[t] != static_cast<uint32_t>(href.counts[t])) mism_cnt++;
                }
                const uint32_t nflat =
                    static_cast<uint32_t>(href.lpt.flat_tile_ids.size());
                for (uint32_t i = 0; i < nflat; i++) {
                    if (tids_d[i] != href.lpt.flat_tile_ids[i]) mism_tids++;
                }
                for (uint32_t c = 0; c < num_cores; c++) {
                    if (lptm[c * 2 + 0] != href.lpt.per_core_offset[c] ||
                        lptm[c * 2 + 1] != href.lpt.per_core_count[c])
                        mism_lpt++;
                }
                for (uint32_t t = 0; t < num_tiles; t++) {
                    if (tmeta_d[t * 2 + 0] != href.pstart_page[t] ||
                        tmeta_d[t * 2 + 1] != href.tile_pad[t])
                        mism_tmeta++;
                }
                const std::size_t mism =
                    mism_hist + mism_cnt + mism_tids + mism_lpt + mism_tmeta;
                std::fprintf(stderr,
                    "[SORT_LAYOUT_VERIFY] P_kept dev=%u ref=%u P_aligned dev=%u ref=%u "
                    "status=%u hist=%zu cnt=%zu tids=%zu lpt=%zu tmeta=%zu %s\n",
                    dev_P_kept, href.P_kept, dev_P_aligned, href.P_aligned, dev_status,
                    mism_hist, mism_cnt, mism_tids, mism_lpt, mism_tmeta,
                    mism == 0 ? "IDENTICAL" : "FAIL");
                if (mism != 0 || dev_P_kept != href.P_kept || dev_P_aligned != href.P_aligned) {
                    return fail();
                }
            }
            t_bin1 = clk::now();
        }

        if (device_layout) {
            if (!layout_verify) {
                enqueue_bin_layout_kernel(
                    ctx,
                    num_cores,
                    num_tiles,
                    stride,
                    tile_bucket,
                    static_cast<uint32_t>(ctx->buf_tile_ranges->address()),
                    l1_record_early,
                    bucket_fit);
                GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
                distributed::Finish(*ctx->cq);
                t_bin1 = clk::now();
            }
            uint32_t layout_status = 0;
            if (!read_bin_layout_ctrl(ctx, P_kept, P_aligned, max_n, layout_status)) {
                std::cerr << "[gsplat_tt::sort] device layout status=" << layout_status
                          << "; tile exceeds device sort capacity — hard fail "
                             "(render_clean is single-path TT, no host fallback)\n";
                return fail();
            }
            counts.assign(num_tiles, 0);
            std::vector<uint32_t> cnts_u(counts_pad, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, cnts_u, ctx->buf_tile_counts, true);
            for (uint32_t t = 0; t < num_tiles; t++)
                counts[t] = static_cast<int64_t>(cnts_u[t]);
            std::vector<uint32_t> ranges_u(
                static_cast<std::size_t>(ctx->cap_ranges_bytes / 4), 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, ranges_u, ctx->buf_tile_ranges, true);
            for (uint32_t t = 0; t < num_tiles; t++) {
                starts[t] = static_cast<int64_t>(ranges_u[t * 2 + 0]);
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                    static_cast<int64_t>(ranges_u[t * 2 + 1]);
            }
            std::vector<uint32_t> tmeta_d(tmeta_pad, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tmeta_d, ctx->buf_tmeta, true);
            for (uint32_t t = 0; t < num_tiles; t++) {
                pstart_page[t] = tmeta_d[t * 2 + 0];
                tile_pad[t] = tmeta_d[t * 2 + 1];
                pstart_elem[t] = pstart_page[t] * ELEMS_PER_PAGE;
            }
            const uint32_t cap_meta = static_cast<uint32_t>(ctx->cap_lpt_meta_bytes / 4);
            std::vector<uint32_t> lptm(cap_meta, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, lptm, ctx->buf_lpt_meta, true);
            lpt.per_core_offset.assign(num_cores, 0);
            lpt.per_core_count.assign(num_cores, 0);
            for (uint32_t c = 0; c < num_cores; c++) {
                lpt.per_core_offset[c] = lptm[c * 2 + 0];
                lpt.per_core_count[c] = lptm[c * 2 + 1];
            }
            const uint32_t cap_tids = static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
            std::vector<uint32_t> tids_d(cap_tids, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tids_d, ctx->buf_tile_ids, true);
            uint32_t flat_n = 0;
            for (uint32_t c = 0; c < num_cores; c++) flat_n += lpt.per_core_count[c];
            lpt.flat_tile_ids.assign(tids_d.begin(), tids_d.begin() + flat_n);
            device_state::register_buffer("sort_lpt_tile_ids", ctx->buf_tile_ids);
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            T.upload_ms = 0.0;
        } else if (layout_verify_ref) {
            BinLayoutResult bl = std::move(*layout_verify_ref);
            if (bl.status == 1 || bl.status == 2) {
                return fail();
            }
            counts = std::move(bl.counts);
            starts = std::move(bl.starts);
            pstart_page = std::move(bl.pstart_page);
            pstart_elem = std::move(bl.pstart_elem);
            tile_pad = std::move(bl.tile_pad);
            lpt = std::move(bl.lpt);
            P_kept = bl.P_kept;
            P_aligned = bl.P_aligned;
            max_n = bl.max_pad_n;
            for (uint32_t t = 0; t < num_tiles; t++) {
                if (counts[t] > 0) {
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                        starts[t] + counts[t];
                }
            }
            t_bin1 = clk::now();
            if (tile_bucket) {
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bin2d_rec, bl.histrec, false);
                recbase_addr = static_cast<uint32_t>(ctx->buf_bin2d_rec->address());
                bl.bucket_meta.resize(
                    static_cast<std::size_t>(ctx->cap_bucket_meta_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bucket_meta, bl.bucket_meta, false);
            }
            if (l1_record_early && !bl.histrec_l1.empty()) {
                bl.histrec_l1.resize(
                    static_cast<std::size_t>(ctx->cap_l1_rec_base_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_l1_rec_base, bl.histrec_l1, false);
                l1_base_addr = static_cast<uint32_t>(ctx->buf_l1_rec_base->address());
            }
            std::vector<uint32_t> tmeta(tmeta_pad, 0);
            for (uint32_t t = 0; t < num_tiles; t++) {
                tmeta[t * 2 + 0] = pstart_page[t];
                tmeta[t * 2 + 1] = tile_pad[t];
            }
            const uint32_t cap_tile_ids_elems =
                static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
            std::vector<uint32_t> tile_ids_flat(cap_tile_ids_elems, 0);
            std::copy(
                lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            const auto t_up0 = clk::now();
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d, bl.hist, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
            distributed::EnqueueWriteMeshBuffer(
                *ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
            if (!sort_stage_defer_finish()) {
                GSPLAT_HOST_ZONE("host_finish_sort_upload");
                distributed::Finish(*ctx->cq);
            }
            T.upload_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_up0).count();
        } else {
            // Host bridge: D2H histogram + page layout + LPT + H2D metadata.
            std::vector<uint32_t> hist(static_cast<std::size_t>(num_cores) * stride);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, hist, ctx->buf_bin2d, true);
            t_d2h = clk::now();

            BinLayoutResult bl =
                host_bin_layout_from_hist(hist, num_cores, num_tiles, stride, tile_bucket,
                                          l1_record_early, bucket_fit);
            if (bl.status == 1) {
                std::cerr << "[gsplat_tt::sort] per-core padded run > BIN_LOCAL_MAX; "
                             "exceeds device sort capacity — hard fail "
                             "(render_clean is single-path TT, no host fallback)\n";
                return fail();
            }
            if (bl.status == 2) {
                std::cerr << "[gsplat_tt::sort] padded tile exceeds MAX_TILE_ENTRIES; "
                             "exceeds device sort capacity — hard fail "
                             "(render_clean is single-path TT, no host fallback)\n";
                return fail();
            }
            counts = std::move(bl.counts);
            starts = std::move(bl.starts);
            pstart_page = std::move(bl.pstart_page);
            pstart_elem = std::move(bl.pstart_elem);
            tile_pad = std::move(bl.tile_pad);
            lpt = std::move(bl.lpt);
            P_kept = bl.P_kept;
            P_aligned = bl.P_aligned;
            max_n = bl.max_pad_n;
            hist = std::move(bl.hist);
            for (uint32_t t = 0; t < num_tiles; t++) {
                if (counts[t] > 0) {
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                        starts[t] + counts[t];
                }
            }
            t_bin1 = clk::now();

            if (tile_bucket) {
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bin2d_rec, bl.histrec, false);
                recbase_addr = static_cast<uint32_t>(ctx->buf_bin2d_rec->address());
                bl.bucket_meta.resize(
                    static_cast<std::size_t>(ctx->cap_bucket_meta_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bucket_meta, bl.bucket_meta, false);
            }
            if (l1_record_early && !bl.histrec_l1.empty()) {
                bl.histrec_l1.resize(
                    static_cast<std::size_t>(ctx->cap_l1_rec_base_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_l1_rec_base, bl.histrec_l1, false);
                l1_base_addr = static_cast<uint32_t>(ctx->buf_l1_rec_base->address());
            }
            // iter-138: allocate + upload the compact overflow region and its base
            // rows. Region is sized to the actual in-cap overflow record count
            // (even-padded per tile); base rows are sentinel-filled for tiles that
            // are not pre-packed. Always allocated (>=1 page) so the kernel reads
            // valid addresses even when this view has no in-cap overflow tile.
            if (l1_record_early && !bl.histrec_overflow.empty()) {
                const std::size_t ov_region_bytes = std::max<std::size_t>(
                    PAGE_BYTES, static_cast<std::size_t>(bl.ov_total_slots) * 32u);
                if (!ctx->buf_l1_ov || ctx->cap_l1_ov_bytes < ov_region_bytes) {
                    ctx->buf_l1_ov =
                        make_dram_paged(ctx->mesh_device.get(), ov_region_bytes, 64u);
                    ctx->cap_l1_ov_bytes = ov_region_bytes;
                    device_state::register_buffer("sort_l1_overflow", ctx->buf_l1_ov);
                }
                l1_ov_addr = static_cast<uint32_t>(ctx->buf_l1_ov->address());

                const std::size_t ov_base_bytes =
                    static_cast<std::size_t>(num_cores) * stride * 4u;
                if (!ctx->buf_l1_ov_base || ctx->cap_l1_ov_base_bytes < ov_base_bytes) {
                    ctx->buf_l1_ov_base =
                        make_dram(ctx->mesh_device.get(), ov_base_bytes);
                    ctx->cap_l1_ov_base_bytes = ov_base_bytes;
                }
                bl.histrec_overflow.resize(
                    static_cast<std::size_t>(ctx->cap_l1_ov_base_bytes / 4), 0xFFFFFFFFu);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_l1_ov_base, bl.histrec_overflow, false);
                l1_ov_base_addr = static_cast<uint32_t>(ctx->buf_l1_ov_base->address());

                const uint32_t tov_pad = round_up(num_tiles, ELEMS_PER_PAGE);
                const std::size_t tov_bytes = static_cast<std::size_t>(tov_pad) * 4u;
                if (!ctx->buf_tile_ov_base ||
                    ctx->cap_tile_ov_base_bytes < tov_bytes) {
                    ctx->buf_tile_ov_base =
                        make_dram(ctx->mesh_device.get(), tov_bytes);
                    ctx->cap_tile_ov_base_bytes = tov_bytes;
                    device_state::register_buffer(
                        "sort_tile_ov_base", ctx->buf_tile_ov_base);
                }
                bl.tile_ov_base.resize(
                    static_cast<std::size_t>(ctx->cap_tile_ov_base_bytes / 4), 0xFFFFFFFFu);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_tile_ov_base, bl.tile_ov_base, false);
            }
            std::vector<uint32_t> tmeta(tmeta_pad, 0);
            for (uint32_t t = 0; t < num_tiles; t++) {
                tmeta[t * 2 + 0] = pstart_page[t];
                tmeta[t * 2 + 1] = tile_pad[t];
            }
            const uint32_t cap_tile_ids_elems =
                static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
            std::vector<uint32_t> tile_ids_flat(cap_tile_ids_elems, 0);
            std::copy(
                lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            const auto t_up0 = clk::now();
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d, hist, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
            distributed::EnqueueWriteMeshBuffer(
                *ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
            if (!sort_stage_defer_finish()) {
                GSPLAT_HOST_ZONE("host_finish_sort_upload");
                distributed::Finish(*ctx->cq);
            }
            T.upload_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_up0).count();
        }
        T.bin_ms = std::chrono::duration<double, std::milli>(t_bin1 - t_bin0).count();

        // ── Allocate aligned keys/ids/out ───────────────────────────────
        const std::size_t aligned_bytes = static_cast<std::size_t>(P_aligned) * 4;
        if (!ctx->buf_keys || ctx->cap_aligned_bytes < aligned_bytes) {
            ctx->buf_keys = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_ids  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_out  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->cap_aligned_bytes = aligned_bytes;
        }
        if (tile_bucket) {
            const std::size_t recs_bytes =
                static_cast<std::size_t>(std::max<uint32_t>(P_kept, 1)) * PAGE_BYTES;
            if (!ctx->buf_tile_recs || ctx->cap_tile_recs_bytes < recs_bytes) {
                ctx->buf_tile_recs = make_dram(ctx->mesh_device.get(), recs_bytes);
                ctx->cap_tile_recs_bytes = recs_bytes;
                device_state::register_buffer("sort_tile_recs", ctx->buf_tile_recs);
            }
            tile_recs_addr = static_cast<uint32_t>(ctx->buf_tile_recs->address());
        }

        // ── Pass B: device scatter into aligned (key,id) layout ─────────
        const auto t_sc0 = clk::now();
        launch_bin(1, true);
        const auto t_sc1 = clk::now();
        T.bin_ms += std::chrono::duration<double, std::milli>(t_sc1 - t_sc0).count();

        // ── ROUTE C: SFPU microblock cull over the dense bucket ─────────
        // Records are now scattered (launch_bin(1) Finished) and bucket_meta /
        // tile_ids are uploaded. Run the 3-kernel cull program over the SAME
        // LPT tile assignment, baking each candidate's 32-bit keep mask into
        // record word 10. This is a SORT-STAGE write -> the downstream L1 blend
        // reads it back spin-free (GSPLAT_TT_BUCKET_MASK path). Fully overlaps
        // no DRAM round-trip beyond the in-place record RMW.
        if (tile_bucket && bucket_mask_enabled() && ctx->cull_built && P_kept > 0) {
            const auto t_cl0 = clk::now();
            if (!ctx->buf_box_ox) {
                ctx->buf_box_ox = make_dram_paged(ctx->mesh_device.get(), RAMP_TILE_BYTES, RAMP_TILE_BYTES);
                ctx->buf_box_oy = make_dram_paged(ctx->mesh_device.get(), RAMP_TILE_BYTES, RAMP_TILE_BYTES);
                ctx->box_ramp_uploaded = false;
            }
            if (!ctx->box_ramp_uploaded) {
                std::vector<uint32_t> bx = make_box_ramp(/*is_x=*/true);
                std::vector<uint32_t> by = make_box_ramp(/*is_x=*/false);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_box_ox, bx, false);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_box_oy, by, false);
                ctx->box_ramp_uploaded = true;
            }
            float floor = 1.0f / 16384.0f;
            bool cull_disabled = false;
            device_state::get_bucket_cull_params(&floor, &cull_disabled);
            uint32_t floor_bits;
            std::memcpy(&floor_bits, &floor, 4);
            const uint32_t recs_addr = tile_recs_addr;
            const uint32_t meta_addr = static_cast<uint32_t>(ctx->buf_bucket_meta->address());
            const uint32_t box_ox_addr = static_cast<uint32_t>(ctx->buf_box_ox->address());
            const uint32_t box_oy_addr = static_cast<uint32_t>(ctx->buf_box_oy->address());
            const uint32_t tids_addr = static_cast<uint32_t>(ctx->buf_tile_ids->address());
            Program& cprog = ctx->wl_cull.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                const uint32_t start = lpt.per_core_offset[c];
                const uint32_t count = lpt.per_core_count[c];
                SetRuntimeArgs(cprog, ctx->kc_reader, core, {
                    recs_addr, meta_addr, box_ox_addr, box_oy_addr,
                    tids_addr, start, count, static_cast<uint32_t>(tiles_x), floor_bits,
                });
                SetRuntimeArgs(cprog, ctx->kc_compute, core, {
                    count, floor_bits, cull_disabled ? 1u : 0u,
                });
                SetRuntimeArgs(cprog, ctx->kc_writer, core, {
                    recs_addr, meta_addr, tids_addr, start, count,
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_cull, false);
            distributed::Finish(*ctx->cq);
        }

        // ── Device radix kernel (per-tile stable depth sort) ────────────
        const auto t_k0 = clk::now();
        Program& prog = ctx->workload.get_programs().begin()->second;
        for (uint32_t c = 0; c < num_cores; c++) {
            CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
            SetRuntimeArgs(prog, ctx->kernel, core, {
                static_cast<uint32_t>(ctx->buf_keys->address()),
                static_cast<uint32_t>(ctx->buf_ids->address()),
                static_cast<uint32_t>(ctx->buf_out->address()),
                static_cast<uint32_t>(ctx->buf_tile_ids->address()),
                static_cast<uint32_t>(ctx->buf_tmeta->address()),
                lpt.per_core_offset[c],
                lpt.per_core_count[c],
            });
        }
        distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, false);
        if (!sort_stage_defer_finish()) {
            GSPLAT_HOST_ZONE("host_finish_sort_radix");
            distributed::Finish(*ctx->cq);
        }
        const auto t_k1 = clk::now();
        T.kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

        const bool dev_publish = sort_device_publish_enabled();
        std::vector<uint32_t> out_aligned;  // only populated for host-id readback / BIN_DEBUG
        SubchunkLayout sc_layout{};
        MatWorkAssignment mat_work;  // iter 130: subchunk-balanced materialize work
        bool subchunk_materialize = false;

        if (dev_publish) {
            // ── Device compact+publish (skip D2H buf_out + host Pass4) ────
            // sort_publish.cpp copies WHOLE 16-elem pages from the page-aligned
            // radix output into sort_sorted_ids at dst_base = range_start/16, which
            // is only bit-correct when each tile's dst slice is page-aligned. So we
            // publish a PADDED layout: every tile starts on a 16-elem boundary. The
            // resident sort_tile_ranges carries the padded [start, start+count) so
            // all downstream resident readers (cull/blend) index the right slice.
            const auto t_pub0 = clk::now();
            std::vector<int64_t> padded_ranges(result.tile_ranges.size(), 0);
            uint32_t padded_cursor = 0;
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t s = static_cast<uint32_t>(result.tile_ranges[2 * t]);
                const uint32_t e = static_cast<uint32_t>(result.tile_ranges[2 * t + 1]);
                const uint32_t cnt = (e > s) ? (e - s) : 0u;
                padded_ranges[2 * t] = static_cast<int64_t>(padded_cursor);
                padded_ranges[2 * t + 1] = static_cast<int64_t>(padded_cursor + cnt);
                padded_cursor += round_up(cnt, ELEMS_PER_PAGE);
            }
            ensure_resident_sorted_buffer(ctx, padded_cursor);
            upload_resident_tile_ranges(ctx, padded_ranges);
            Program& pub_prog = ctx->wl_publish.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(pub_prog, ctx->kpublish, core, {
                    static_cast<uint32_t>(ctx->buf_out->address()),
                    static_cast<uint32_t>(ctx->buf_sorted_ids->address()),
                    static_cast<uint32_t>(ctx->buf_tile_ranges->address()),
                    static_cast<uint32_t>(ctx->buf_tile_ids->address()),
                    static_cast<uint32_t>(ctx->buf_tmeta->address()),
                    lpt.per_core_offset[c],
                    lpt.per_core_count[c],
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_publish, false);
            sc_layout = build_subchunk_layout(counts, num_tiles, bucket_fit);
            log_subchunk_layout_stats(sc_layout);
            mat_work = build_mat_worklist(counts, num_tiles, num_cores, bucket_fit);
            if (!prepare_subchunk_buffers(ctx, sc_layout, num_tiles)) {
                std::cerr << "[gsplat_tt::sort] subchunk buffer setup failed\n";
                return fail();
            }
            if (!launch_subchunk_directory(ctx, num_tiles, bucket_fit)) {
                std::cerr << "[gsplat_tt::sort] subchunk directory launch failed\n";
                return fail();
            }
            if (sort_blend_pipe_enabled()) {
                // C1: materialize reads prefix/dir written by directory — drain dir
                // before enqueueing mat on the piped CQ (not between mat and blend).
                GSPLAT_HOST_ZONE("host_finish_sort_subchunk_dir");
                distributed::Finish(*ctx->cq);
            }
            if (mat_work.max_items_per_core > 1024u) {
                std::cerr << "[gsplat_tt::sort] materialize work items/core "
                          << mat_work.max_items_per_core << " > MAX_WORK=1024\n";
                return fail();
            }
            subchunk_materialize = true;
            T.publish_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_pub0).count();
            if (sort_blend_pipe_enabled()) {
                device_state::mark_sort_publish_pending();
            } else {
                const auto t_mat0 = clk::now();
                if (!launch_subchunk_materialize(
                        ctx, mat_work, num_cores,
                        static_cast<uint32_t>(tiles_x), bucket_fit)) {
                    std::cerr << "[gsplat_tt::sort] subchunk materialize launch failed\n";
                    return fail();
                }
                GSPLAT_HOST_ZONE("host_finish_sort_materialize");
                distributed::Finish(*ctx->cq);
                T.materialize_ms =
                    std::chrono::duration<double, std::milli>(clk::now() - t_mat0).count();
            }
            // Resident blend reads sort_sorted_ids over NoC — skip the large ids
            // D2H + host Pass4 unless the caller needs the dense host vector.
            const bool need_host_ids = need_host_sorted_ids ||
                                       !resident_blend_chain_enabled();
            if (need_host_ids) {
                finish_sort_cq_if_needed(ctx);
                const uint32_t cap_sorted_elems =
                    static_cast<uint32_t>(ctx->cap_sorted_bytes / 4);
                const auto t_d0 = clk::now();
                std::vector<uint32_t> sids(cap_sorted_elems);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, sids, ctx->buf_sorted_ids, true);
                T.d2h_ms = std::chrono::duration<double, std::milli>(clk::now() - t_d0).count();
                result.sorted_gaussian_ids.assign(P_kept, 0);
                for (uint32_t t = 0; t < num_tiles; t++) {
                    const uint32_t ds = static_cast<uint32_t>(result.tile_ranges[2 * t]);
                    const uint32_t de = static_cast<uint32_t>(result.tile_ranges[2 * t + 1]);
                    const uint32_t ps = static_cast<uint32_t>(padded_ranges[2 * t]);
                    for (uint32_t k = 0; k + ds < de && (ds + k) < P_kept; k++) {
                        result.sorted_gaussian_ids[ds + k] =
                            static_cast<int64_t>(sids[ps + k]);
                    }
                }
            } else {
                T.d2h_ms = 0.0;
                // tile_ranges kept for stats/timing only; blend uses resident DRAM.
            }
        } else {
            // ── D2H aligned sorted ids + Pass4 compact -> contiguous ────
            const uint32_t cap_aligned_elems =
                static_cast<uint32_t>(ctx->cap_aligned_bytes / 4);
            const auto t_d0 = clk::now();
            out_aligned.resize(cap_aligned_elems);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, out_aligned, ctx->buf_out, true);
            const auto t_d1 = clk::now();
            T.d2h_ms = std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();

            const auto t_c0 = clk::now();
            result.sorted_gaussian_ids.resize(P_kept);
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t n = static_cast<uint32_t>(counts[t]);
                if (n == 0) continue;
                const uint32_t src = pstart_elem[t];
                const std::size_t dst = static_cast<std::size_t>(starts[t]);
                for (uint32_t k = 0; k < n; k++)
                    result.sorted_gaussian_ids[dst + k] =
                        static_cast<int64_t>(out_aligned[src + k]);
            }
            const auto t_c1 = clk::now();
            T.compact_ms = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
        }

        if (!dev_publish)
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);

        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0_rp).count();
        std::fprintf(stderr,
            "[SORT] stage=RP P=%u P_kept=%u num_tiles=%u max_tile_n=%u bin=%.2f "
            "up=%.2f kernel=%.2f d2h=%.2f compact=%.2f publish=%.2f mat=%.2f total=%.2fms\n",
            P_full, P_kept, num_tiles, max_n, T.bin_ms, T.upload_ms, T.kernel_ms,
            T.d2h_ms, T.compact_ms, T.publish_ms, T.materialize_ms, T.total_ms);
        if (device_ok) *device_ok = true;
        // Step C1: materialize before blend on the piped CQ (no Finish here —
        // sort_publish_pending: one drain at blend readback; iter-58/83).
        if (subchunk_materialize && sort_blend_pipe_enabled()) {
            const auto t_mat0 = clk::now();
            if (!launch_subchunk_materialize(
                    ctx, mat_work, num_cores,
                    static_cast<uint32_t>(tiles_x), bucket_fit)) {
                std::cerr << "[gsplat_tt::sort] subchunk materialize launch failed\n";
                return fail();
            }
            T.materialize_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_mat0).count();
            std::fprintf(
                stderr, "[SUBCHUNK] materialize_ms=%.2f (piped pre-blend)\n",
                T.materialize_ms);
        }
        maybe_run_sort_blend_continuation(sort_blend, tiles_x, num_tiles);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::sort] resident-pairs path failed: " << e.what() << "\n";
        finish_sort_cq_if_needed(ctx);
        return fail();
    }
}

}  // namespace

bool sort_device_ready() { return ensure_context() != nullptr; }

void sort_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        // Intentionally leak the context: MeshWorkload::~MeshWorkload can
        // SIGSEGV in tt_metal if ProgramImpl runs after MeshDevice::close().
        (void)slot.release();
    }
}

gsplat_cpu::SortResult sort_and_bin_tt(
    const int64_t* gaussian_ids,
    const int64_t* tile_ids,
    const float* depths,
    const std::size_t P,
    const std::size_t M,
    const int tiles_x,
    const int tiles_y,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    SortCallTimings* timings,
    const bool need_host_sorted_ids,
    SortBlendContinuation* sort_blend) {
    auto set_fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::SortResult{};
    };

    // render_clean is single-path: the resident-pairs device binning stage reads
    // the resident full-P (gid,tid) pairs + keep mask that tile_assign left in
    // device_state, plus the resident proj_m_depth, and bins them on-device into
    // the page-aligned per-tile (key,id) layout the radix kernel consumes. The
    // host depth/id/tile_id arguments are unused (kept for ABI). The legacy
    // host-sort (S0) and host-binning (S1) fallbacks were removed; unsupported
    // input hard-fails via set_fail() (render.cpp turns that into a throw).
    (void)gaussian_ids;
    (void)tile_ids;
    (void)depths;
    (void)P;

    auto* ctx = ensure_context();
    if (ctx == nullptr) return set_fail();

    SortCallTimings tlocal;
    auto& T = (timings ? *timings : tlocal);
    T.stage = 1;

    const uint32_t num_tiles =
        static_cast<uint32_t>(tiles_x) * static_cast<uint32_t>(tiles_y);

    return sort_resident_pairs(
        ctx, num_tiles, tiles_x, tiles_y, M, need_host_sorted_ids, pool,
        device_ok, T, sort_blend);
}

}  // namespace gsplat_tt
