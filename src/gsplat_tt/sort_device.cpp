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

#include "gsplat_tt/env_config.h"
#include "gsplat_tt/sort.h"
#include "gsplat_tt/device_state.h"
#include "gsplat_tt/host_tracy.hpp"

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

static bool bucket_mask_enabled() {
    const char* tb = std::getenv("GSPLAT_TT_TILE_BUCKET");
    const char* bm = std::getenv("GSPLAT_TT_BUCKET_MASK");
    return tb && tb[0] == '1' && bm && bm[0] == '1';
}

// L1 scratch budget per ping/pong key+id buffer. The worst hero-scene tile is
// ~25k entries; 32768 leaves headroom. 4 CBs * 32768 * 4B = 512 KB, well under
// Blackhole's ~1.4 MB per-core L1. If a tile exceeds this the host
// transparently falls back to the CPU sort (device_ok = false).
constexpr uint32_t MAX_TILE_ENTRIES = 32768;
constexpr uint32_t MAX_TILE_PAGES = MAX_TILE_ENTRIES / ELEMS_PER_PAGE;  // 2048
constexpr uint32_t SCRATCH_BYTES = MAX_TILE_ENTRIES * 4;  // 128 KB per CB

// R4/R5 device-binning (GSPLAT_TT_RESIDENT_PAIRS): max tiles the per-core L1
// row / cursor / offset CBs hold (hero is 1024 tiles). Larger -> host fallback.
constexpr uint32_t MAX_BIN_TILES = 2048;
constexpr uint32_t BIN_ROW_BYTES = MAX_BIN_TILES * 4;  // 8 KB
// Max kept pairs a single core counting-sorts in L1. Pairs are split evenly by
// page across cores, so per-core load ~= P_kept / num_cores (~25k on hero);
// 65536 (256 KB per ks/is CB) leaves >2x headroom. Larger -> host fallback.
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

static bool fused_tile_enabled() {
    const char* v = std::getenv("GSPLAT_TT_FUSED_TILE");
    return v != nullptr && v[0] == '1';
}

static bool tile_bucket_enabled() {
    const char* v = std::getenv("GSPLAT_TT_TILE_BUCKET");
    if (v != nullptr) {
        return v[0] == '1';
    }
    if (fused_tile_enabled()) {
        return false;
    }
    const char* sfpu = std::getenv("GSPLAT_TT_SFPU_CULL");
    const char* rb = std::getenv("GSPLAT_TT_RESIDENT_BLEND");
    return sfpu != nullptr && sfpu[0] == '1' && rb != nullptr && rb[0] == '1';
}

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
    if (tile_bucket) {
        cb(9, 16u * PAGE_BYTES);  // rec staging ring (REC_BATCH=16 blendrec pages)
        cb(10, BIN_ROW_BYTES);    // recrow (per-(core,tile) DENSE record base)
    }

    std::vector<uint32_t> ct;
    const int bin_accessors = tile_bucket ? 10 : 7;  // +blendrec, +tile_recs, +recbase
    for (int i = 0; i < bin_accessors; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    std::map<std::string, std::string> defines;
    if (const char* v = std::getenv("GSPLAT_TT_BIN_NODEPTH"); v && v[0] == '1')
        defines["BIN_NO_DEPTH"] = "1";
    if (const char* v = std::getenv("GSPLAT_TT_BIN_DUMP"); v && v[0] == '1')
        defines["BIN_DUMP"] = "1";
    if (tile_bucket) defines["BIN_EMIT_REC"] = "1";
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
    build_program_publish(ctx);
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
struct LptAssignment {
    std::vector<uint32_t> flat_tile_ids;     // concatenated per-core lists
    std::vector<uint32_t> per_core_offset;   // element offset into flat list
    std::vector<uint32_t> per_core_count;
};

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

    // GSPLAT_TT_LPT_STATS: dump the realized LPT load distribution so the
    // tile->core balance (the SFPU cull/blend makespan driver) can be quantified
    // without re-deriving it from the device profiler. Host-only, default OFF,
    // no effect on the assignment. Reports the theoretical makespan-vs-mean gap
    // (= LPT headroom) and the heaviest single tile (the LPT lower bound:
    // makespan >= max(mean, heaviest_tile)).
    if (std::getenv("GSPLAT_TT_LPT_STATS") != nullptr && !cost_id.empty()) {
        uint64_t total = 0;
        for (const auto& [cost, id] : cost_id) total += cost;
        const uint64_t heaviest_tile = cost_id.front().first;  // sorted descending
        std::vector<uint64_t> sorted_load = load;
        std::sort(sorted_load.begin(), sorted_load.end());
        const uint64_t max_load = sorted_load.back();
        const uint64_t min_load = sorted_load.front();
        const uint64_t med_load = sorted_load[num_cores / 2];
        const double mean_load = static_cast<double>(total) / num_cores;
        std::fprintf(stderr,
            "[LPT_STATS] tiles_nonempty=%zu cores=%u total_cand=%llu "
            "mean/core=%.0f median/core=%llu max/core=%llu min/core=%llu "
            "heaviest_tile=%llu | makespan/mean=%.4f makespan-mean=%.0f cand "
            "(=%.1f%% headroom) heaviest/mean=%.4f\n",
            cost_id.size(), num_cores,
            static_cast<unsigned long long>(total), mean_load,
            static_cast<unsigned long long>(med_load),
            static_cast<unsigned long long>(max_load),
            static_cast<unsigned long long>(min_load),
            static_cast<unsigned long long>(heaviest_tile),
            mean_load > 0 ? max_load / mean_load : 0.0,
            max_load - mean_load,
            mean_load > 0 ? 100.0 * (max_load - mean_load) / mean_load : 0.0,
            mean_load > 0 ? heaviest_tile / mean_load : 0.0);
    }
    return a;
}

// Publish the contiguous (sorted_ids, tile_ranges) into device_state as uint32
// DRAM buffers so downstream device stages can read them resident.
static bool resident_blend_chain_enabled() {
    const char* v = std::getenv("GSPLAT_TT_RESIDENT_BLEND");
    return v != nullptr && v[0] == '1';
}

static bool sort_device_publish_enabled();

// Drop the sort-publish Finish() and blend's blocking sort_P_kept D2H so the
// publish kernel chains into FUSED_TILE cull+blend with one CQ drain.
static bool sort_blend_pipe_enabled() {
    return gsplat_tt::env_config::sort_blend_pipe_enabled();
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

static bool sort_device_publish_enabled() {
    const char* v = std::getenv("GSPLAT_TT_SORT_DEVICE_PUBLISH");
    return v != nullptr && v[0] == '1';
}

// Compare device result against gsplat_cpu::sort_and_bin. Prints a SORT line;
// aborts on any byte mismatch when GSPLAT_TT_SORT_VERIFY=1.
static void verify_vs_cpu(
    const gsplat_cpu::SortResult& dev,
    const int64_t* gaussian_ids, const int64_t* tile_ids, const float* depths,
    std::size_t P, std::size_t M, int tiles_x, int tiles_y,
    gsplat_cpu::ThreadPool* pool, int stage) {
    const gsplat_cpu::SortResult cpu = gsplat_cpu::sort_and_bin(
        gaussian_ids, tile_ids, depths, P, M, tiles_x, tiles_y, pool);
    bool ids_match = (cpu.sorted_gaussian_ids.size() == dev.sorted_gaussian_ids.size());
    std::size_t id_mism = 0, first_id = 0;
    if (ids_match) {
        for (std::size_t i = 0; i < cpu.sorted_gaussian_ids.size(); i++) {
            if (cpu.sorted_gaussian_ids[i] != dev.sorted_gaussian_ids[i]) {
                if (id_mism == 0) first_id = i;
                id_mism++;
            }
        }
    }
    bool rng_match = (cpu.tile_ranges.size() == dev.tile_ranges.size());
    std::size_t rng_mism = 0;
    if (rng_match) {
        for (std::size_t i = 0; i < cpu.tile_ranges.size(); i++) {
            if (cpu.tile_ranges[i] != dev.tile_ranges[i]) rng_mism++;
        }
    }
    const bool ok = ids_match && rng_match && id_mism == 0 && rng_mism == 0;
    std::fprintf(stderr,
        "[SORT verify] stage=S%d P=%zu cpu_P=%zu size_match=%d id_mismatch=%zu "
        "(first@%zu) range_mismatch=%zu -> %s\n",
        stage, dev.sorted_gaussian_ids.size(), cpu.sorted_gaussian_ids.size(),
        (int)(ids_match && rng_match), id_mism, first_id, rng_mism,
        ok ? "IDENTICAL" : "MISMATCH");
    if (!ok) {
        if (ids_match && id_mism) {
            std::size_t bt = 0, blo = 0, bhi = 0;
            for (std::size_t t = 0; t * 2 + 1 < cpu.tile_ranges.size(); t++) {
                const std::size_t lo = static_cast<std::size_t>(cpu.tile_ranges[2 * t]);
                const std::size_t hi = static_cast<std::size_t>(cpu.tile_ranges[2 * t + 1]);
                if (first_id >= lo && first_id < hi) { bt = t; blo = lo; bhi = hi; break; }
            }
            std::fprintf(stderr,
                "[SORT verify] first_id@%zu in tile=%zu off=%zu count=%zu\n",
                first_id, bt, first_id - blo, bhi - blo);
            std::fprintf(stderr, "  cpu:");
            for (std::size_t k = (first_id > 4 ? first_id - 4 : 0); k < first_id + 6 && k < cpu.sorted_gaussian_ids.size(); k++)
                std::fprintf(stderr, " %lld", (long long)cpu.sorted_gaussian_ids[k]);
            std::fprintf(stderr, "\n  dev:");
            for (std::size_t k = (first_id > 4 ? first_id - 4 : 0); k < first_id + 6 && k < dev.sorted_gaussian_ids.size(); k++)
                std::fprintf(stderr, " %lld", (long long)dev.sorted_gaussian_ids[k]);
            std::fprintf(stderr, "\n");
        }
        std::fprintf(stderr, "[SORT verify] FATAL: device sort not byte-identical to CPU\n");
        std::abort();
    }
}

// ── R4/R5 resident-pairs device binning ─────────────────────────────────
// Bins the resident full-P (gid,tid) pairs + keep mask into the page-aligned
// per-tile (key,id) layout on-device, runs the radix kernel, compacts, and
// publishes. Returns a SortResult identical in shape to the host path.
static gsplat_cpu::SortResult sort_resident_pairs(
    SortDeviceContext* ctx, uint32_t num_tiles, int tiles_x, int tiles_y,
    std::size_t M, bool verify, bool need_host_sorted_ids,
    gsplat_cpu::ThreadPool* pool, bool* device_ok, SortCallTimings& T) {
    using clk = std::chrono::high_resolution_clock;
    const auto t_total0_rp = clk::now();
    auto fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::SortResult{};
    };
    gsplat_cpu::SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);

    if (num_tiles > MAX_BIN_TILES) {
        std::cerr << "[gsplat_tt::sort] num_tiles=" << num_tiles
                  << " > MAX_BIN_TILES=" << MAX_BIN_TILES << "; host fallback\n";
        return fail();
    }

    auto bgid = device_state::get_buffer("ta_pairs_gid");
    auto btid = device_state::get_buffer("ta_pairs_tid");
    auto bkeep = device_state::get_buffer("ta_pairs_keep");
    auto bP = device_state::get_buffer("ta_pairs_P");
    auto bdep = device_state::get_buffer("proj_m_depth");
    if (!bgid || !btid || !bkeep || !bP || !bdep) {
        std::cerr << "[gsplat_tt::sort] RESIDENT_PAIRS set but resident pairs / "
                     "proj_m_depth missing; host fallback\n";
        return fail();
    }

    try {
        // Read full P + P_pad published by tile_assign.
        std::vector<uint32_t> pbuf(ELEMS_PER_PAGE);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, pbuf, bP, true);
        const uint32_t P_full = pbuf[0];
        const uint32_t P_pad = pbuf[1];
        if (P_full == 0) {
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);
            if (device_ok) *device_ok = true;
            return result;
        }

        uint32_t num_cores = ctx->grid.x * ctx->grid.y;
        if (const char* v = std::getenv("GSPLAT_TT_BIN_1CORE"); v && v[0] == '1')
            num_cores = 1;  // diagnostic: serialize binning to one core
        uint32_t dump_tile = 0;
        if (const char* v = std::getenv("GSPLAT_TT_BIN_DUMP_TILE"); v)
            dump_tile = static_cast<uint32_t>(std::atoi(v));
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
            std::cerr << "[gsplat_tt::sort] TILE_BUCKET set but proj_m_blendrec missing; "
                         "host fallback\n";
            return fail();
        }
        const uint32_t blendrec_addr = bbrec ? static_cast<uint32_t>(bbrec->address()) : 0u;
        uint32_t tile_recs_addr = 0u;  // real address set after P_aligned is known
        uint32_t recbase_addr = 0u;    // dense per-(core,tile) record base (set w/ buf)

        // ── Pass A: per-core histogram (count) ──────────────────────────
        const auto t_bin0 = clk::now();
        auto launch_bin = [&](uint32_t mode) {
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
                    dump_tile,  // arg14: tile to dump under BIN_DUMP
                };
                if (tile_bucket) {
                    args.push_back(blendrec_addr);   // arg15
                    args.push_back(tile_recs_addr);  // arg16 (0 for mode 0; real for mode 1)
                    args.push_back(recbase_addr);    // arg17 (0 for mode 0; real for mode 1)
                }
                SetRuntimeArgs(prog, ctx->kbin, core, args);
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin, false);
            if (mode == 0) {
                GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
            } else {
                GSPLAT_HOST_ZONE("host_finish_sort_bin_scat");
            }
            distributed::Finish(*ctx->cq);
        };
        launch_bin(0);
        const auto t_cnt = clk::now();  // DIAG: count kernel (+Finish) done

        // D2H the per-core histograms.
        std::vector<uint32_t> hist(static_cast<std::size_t>(num_cores) * stride);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, hist, ctx->buf_bin2d, true);
        const auto t_d2h = clk::now();  // DIAG: histogram D2H done

        // Host: per-tile totals + page-aligned starts + per-core base offsets.
        std::vector<int64_t> counts(num_tiles, 0);
        for (uint32_t t = 0; t < num_tiles; t++) {
            uint64_t s = 0;
            for (uint32_t c = 0; c < num_cores; c++)
                s += hist[static_cast<std::size_t>(c) * stride + t];
            counts[t] = static_cast<int64_t>(s);
        }
        // The kernel page-pads each tile's per-core L1 block (ceil(count/16)*16),
        // so the per-core L1 counting-sort buffer is bounded by the PADDED run,
        // not the raw kept count. If any core's padded run exceeds BIN_LOCAL_MAX,
        // fall back to the host binning path.
        uint32_t max_core_padded = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            uint64_t s = 0;
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint64_t h = hist[static_cast<std::size_t>(c) * stride + t];
                s += ((h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE) * ELEMS_PER_PAGE;
            }
            if (s > max_core_padded) max_core_padded = static_cast<uint32_t>(s);
        }
        if (max_core_padded > BIN_LOCAL_MAX) {
            std::cerr << "[gsplat_tt::sort] per-core padded run=" << max_core_padded
                      << " > BIN_LOCAL_MAX=" << BIN_LOCAL_MAX << "; host fallback\n";
            return fail();
        }
        // Page-aligned PER-CORE layout: within each tile, every contributing
        // core gets a PAGE-ALIGNED block (ceil(count/16) pages). No two cores
        // ever share a DRAM page, eliminating the multi-writer sub-page write
        // race. The unused tail slots of each block are padded by the kernel
        // with a max key (0xFFFFFFFF) that the stable radix sorts to the end of
        // the tile; host compaction keeps only the real count.
        //   counts[t]       : real per-tile element count (compaction + ranges)
        //   starts[t]       : real exclusive prefix (compaction dst)
        //   pstart_page[t]  : tile region start page (radix input)
        //   tile_pad[t]     : padded per-tile element count (radix sorts this)
        //   hist[c*stride+t]: page-aligned global base for core c's block (scatter)
        std::vector<int64_t> starts(num_tiles, 0);
        std::vector<uint32_t> pstart_page(num_tiles, 0);
        std::vector<uint32_t> pstart_elem(num_tiles, 0);
        std::vector<uint32_t> tile_pad(num_tiles, 0);
        // T2: DENSE per-(core,tile) record base (no page padding — each record is
        // its own 64B page so cores never share a page even when dense). Tile t's
        // records occupy pages [starts[t], starts[t]+counts[t]); core c's block
        // starts at starts[t] + (sum of kept counts of cores < c for tile t).
        std::vector<uint32_t> histrec;
        std::vector<uint32_t> bucket_meta;  // (start,count) per tile
        if (tile_bucket) {
            histrec.assign(static_cast<std::size_t>(num_cores) * stride, 0u);
            bucket_meta.assign(static_cast<std::size_t>(num_tiles) * 2u, 0u);
        }
        int64_t cstart = 0;
        uint32_t apage = 0;
        uint32_t max_pad_n = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const int64_t creal = counts[t];
            starts[t] = cstart;
            cstart += creal;
            pstart_page[t] = apage;
            pstart_elem[t] = apage * ELEMS_PER_PAGE;
            const uint32_t tile_start_page = apage;
            uint32_t rec_run = 0;  // dense prefix over cores within this tile
            for (uint32_t c = 0; c < num_cores; c++) {
                const std::size_t idx = static_cast<std::size_t>(c) * stride + t;
                const uint32_t h = hist[idx];
                if (tile_bucket) {
                    histrec[idx] = static_cast<uint32_t>(starts[t]) + rec_run;
                    rec_run += h;
                }
                hist[idx] = apage * ELEMS_PER_PAGE;  // page-aligned base for (c,t)
                if (h > 0) apage += (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
            }
            const uint32_t pad_pages = apage - tile_start_page;
            tile_pad[t] = pad_pages * ELEMS_PER_PAGE;
            if (creal > 0) {
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] = starts[t] + creal;
            }
            if (tile_bucket) {
                bucket_meta[static_cast<std::size_t>(t) * 2 + 0] = static_cast<uint32_t>(starts[t]);
                bucket_meta[static_cast<std::size_t>(t) * 2 + 1] = static_cast<uint32_t>(creal);
            }
            if (tile_pad[t] > max_pad_n) max_pad_n = tile_pad[t];
        }
        if (max_pad_n > MAX_TILE_ENTRIES) {
            std::cerr << "[gsplat_tt::sort] padded tile n=" << max_pad_n
                      << " exceeds MAX_TILE_ENTRIES; host fallback\n";
            return fail();
        }
        const uint32_t P_kept = static_cast<uint32_t>(cstart);
        const uint32_t total_pages = std::max<uint32_t>(apage, 1);
        const uint32_t P_aligned = total_pages * ELEMS_PER_PAGE;
        const uint32_t max_n = max_pad_n;
        const auto t_bin1 = clk::now();
        T.bin_ms = std::chrono::duration<double, std::milli>(t_bin1 - t_bin0).count();

        // ── Allocate aligned keys/ids/out + tmeta + tile_ids ────────────
        const std::size_t aligned_bytes = static_cast<std::size_t>(P_aligned) * 4;
        if (!ctx->buf_keys || ctx->cap_aligned_bytes < aligned_bytes) {
            ctx->buf_keys = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_ids  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_out  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->cap_aligned_bytes = aligned_bytes;
        }
        if (tile_bucket) {
            // DENSE record bucket: one 64B record per kept candidate (P_kept pages).
            const std::size_t recs_bytes =
                static_cast<std::size_t>(std::max<uint32_t>(P_kept, 1)) * PAGE_BYTES;
            if (!ctx->buf_tile_recs || ctx->cap_tile_recs_bytes < recs_bytes) {
                ctx->buf_tile_recs = make_dram(ctx->mesh_device.get(), recs_bytes);
                ctx->cap_tile_recs_bytes = recs_bytes;
                device_state::register_buffer("sort_tile_recs", ctx->buf_tile_recs);
            }
            tile_recs_addr = static_cast<uint32_t>(ctx->buf_tile_recs->address());
            // Per-(core,tile) DENSE base for the record scatter (mirrors bin2d).
            const std::size_t rec_base_bytes =
                static_cast<std::size_t>(num_cores) * stride * 4;
            if (!ctx->buf_bin2d_rec || ctx->cap_bin2d_rec_bytes < rec_base_bytes) {
                ctx->buf_bin2d_rec = make_dram(ctx->mesh_device.get(), rec_base_bytes);
                ctx->cap_bin2d_rec_bytes = rec_base_bytes;
            }
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d_rec, histrec, false);
            recbase_addr = static_cast<uint32_t>(ctx->buf_bin2d_rec->address());
            // Publish per-tile (start,count) bucket meta for the L1-sort reader.
            const uint32_t bm_pad = round_up(num_tiles * 2u, ELEMS_PER_PAGE);
            const std::size_t bm_bytes = static_cast<std::size_t>(bm_pad) * 4;
            if (!ctx->buf_bucket_meta || ctx->cap_bucket_meta_bytes < bm_bytes) {
                ctx->buf_bucket_meta = make_dram(ctx->mesh_device.get(), bm_bytes);
                ctx->cap_bucket_meta_bytes = bm_bytes;
                device_state::register_buffer("sort_bucket_meta", ctx->buf_bucket_meta);
            }
            bucket_meta.resize(static_cast<std::size_t>(ctx->cap_bucket_meta_bytes / 4), 0u);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bucket_meta, bucket_meta, false);
        }
        const uint32_t tmeta_count = num_tiles * 2;
        const uint32_t tmeta_pad = round_up(std::max<uint32_t>(tmeta_count, 1), ELEMS_PER_PAGE);
        const std::size_t tmeta_bytes = static_cast<std::size_t>(tmeta_pad) * 4;
        if (!ctx->buf_tmeta || ctx->cap_tmeta_bytes < tmeta_bytes) {
            ctx->buf_tmeta = make_dram(ctx->mesh_device.get(), tmeta_bytes);
            ctx->cap_tmeta_bytes = tmeta_bytes;
        }
        std::vector<uint32_t> tmeta(tmeta_pad, 0);
        std::vector<int64_t> pad_counts(num_tiles, 0);
        for (uint32_t t = 0; t < num_tiles; t++) {
            tmeta[t * 2 + 0] = pstart_page[t];
            tmeta[t * 2 + 1] = tile_pad[t];  // radix sorts the padded region
            pad_counts[t] = static_cast<int64_t>(tile_pad[t]);
        }
        LptAssignment lpt;
        {
            GSPLAT_HOST_ZONE("host_lpt_build");
            lpt = build_lpt(pad_counts, num_tiles, num_cores);
        }
        const uint32_t tile_ids_count = static_cast<uint32_t>(lpt.flat_tile_ids.size());
        const uint32_t tile_ids_pad = round_up(std::max<uint32_t>(tile_ids_count, 1), ELEMS_PER_PAGE);
        const std::size_t tile_ids_bytes = static_cast<std::size_t>(tile_ids_pad) * 4;
        if (!ctx->buf_tile_ids || ctx->cap_tile_ids_bytes < tile_ids_bytes) {
            ctx->buf_tile_ids = make_dram(ctx->mesh_device.get(), tile_ids_bytes);
            ctx->cap_tile_ids_bytes = tile_ids_bytes;
        }
        // Grow-only buffer: size the upload to capacity (whole-buffer write).
        const uint32_t cap_tile_ids_elems =
            static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
        std::vector<uint32_t> tile_ids_flat(cap_tile_ids_elems, 0);
        std::copy(lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());
        publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);

        // ── Upload: base offsets (into bin2d), tmeta, tile_ids ──────────
        const auto t_up0 = clk::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d, hist, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
        GSPLAT_HOST_ZONE("host_finish_sort_upload");
        distributed::Finish(*ctx->cq);
        const auto t_up1 = clk::now();
        T.upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

        // ── Pass B: device scatter into aligned (key,id) layout ─────────
        const auto t_sc0 = clk::now();
        launch_bin(1);
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
            if (std::getenv("GSPLAT_TT_MB_TIMING") != nullptr) {
                const double cms =
                    std::chrono::duration<double, std::milli>(clk::now() - t_cl0).count();
                std::fprintf(stderr,
                    "[BUCKET_CULL] P_kept=%u tiles=%u floor=%.6g cull_disabled=%d exec=%.2fms\n",
                    P_kept, num_tiles, floor, (int)cull_disabled, cms);
            }
        }

        if (tile_bucket && std::getenv("GSPLAT_TT_BUCKET_DUMP")) {
            distributed::Finish(*ctx->cq);
            const uint32_t ndump = std::min<uint32_t>(P_kept, 6u);
            std::vector<uint32_t> recs(static_cast<std::size_t>(ndump) * ELEMS_PER_PAGE, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, recs, ctx->buf_tile_recs, true);
            std::vector<uint32_t> bm(static_cast<std::size_t>(ctx->cap_bucket_meta_bytes / 4), 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, bm, ctx->buf_bucket_meta, true);
            std::fprintf(stderr, "[BUCKET_DUMP] P_kept=%u recbase_addr=%u meta[t0]=(%u,%u) meta[t1]=(%u,%u)\n",
                         P_kept, recbase_addr, bm[0], bm[1], bm[2], bm[3]);
            for (uint32_t s = 0; s < ndump; s++) {
                const uint32_t* r = &recs[static_cast<std::size_t>(s) * ELEMS_PER_PAGE];
                float a, b, c, px, py, op, dep;
                std::memcpy(&a, &r[0], 4); std::memcpy(&b, &r[1], 4); std::memcpy(&c, &r[2], 4);
                std::memcpy(&px, &r[3], 4); std::memcpy(&py, &r[4], 4); std::memcpy(&op, &r[5], 4);
                std::memcpy(&dep, &r[9], 4);
                std::fprintf(stderr, "[BUCKET_DUMP] slot%u a=%.3f b=%.3f c=%.3f px=%.2f py=%.2f op=%.3f depbits=%08x dep=%.4f\n",
                             s, a, b, c, px, py, op, r[9], dep);
            }
        }

        // ── DEBUG (GSPLAT_TT_BUCKET_VERIFY): A/B the DENSE record bucket vs the
        // PAGE-ALIGNED keys layout, both written by THIS scatter kernel from the
        // same data. The keys/ids layout is the production (gather) path and is
        // known bit-correct, so comparing the per-tile key MULTISET isolates a
        // dense-assembly overlap/skip (the suspected Lb>64 bug) precisely.
        if (tile_bucket && std::getenv("GSPLAT_TT_BUCKET_VERIFY")) {
            distributed::Finish(*ctx->cq);
            std::vector<uint32_t> recs(static_cast<std::size_t>(std::max<uint32_t>(P_kept, 1)) *
                                       ELEMS_PER_PAGE, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, recs, ctx->buf_tile_recs, true);
            std::vector<uint32_t> kbuf(static_cast<std::size_t>(P_aligned), 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, kbuf, ctx->buf_keys, true);
            // Rank tiles by density; probe the densest handful with count>64.
            std::vector<uint32_t> order(num_tiles);
            for (uint32_t t = 0; t < num_tiles; t++) order[t] = t;
            std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
                return counts[x] > counts[y];
            });
            uint32_t probed = 0, mismatched = 0;
            for (uint32_t oi = 0; oi < num_tiles && probed < 8u; oi++) {
                const uint32_t t = order[oi];
                const uint32_t creal = static_cast<uint32_t>(counts[t]);
                if (creal <= 64u) break;  // densest first; nothing past here is >64
                probed++;
                // Dense bucket key multiset: recs[starts[t]+i][9], i in [0,creal).
                std::vector<uint32_t> bk(creal);
                uint32_t zero_slots = 0;
                for (uint32_t i = 0; i < creal; i++) {
                    const std::size_t pg = static_cast<std::size_t>(starts[t]) + i;
                    const uint32_t key = recs[pg * ELEMS_PER_PAGE + 9];
                    bk[i] = key;
                    if (recs[pg * ELEMS_PER_PAGE + 0] == 0u && key == 0u) zero_slots++;
                }
                // Canonical key multiset from the page-aligned keys region (drop
                // 0xffffffff padding the kernel pre-fills into tail page slots).
                std::vector<uint32_t> ck;
                ck.reserve(creal);
                const std::size_t kstart = static_cast<std::size_t>(pstart_elem[t]);
                const std::size_t kend = kstart + tile_pad[t];
                for (std::size_t i = kstart; i < kend && i < kbuf.size(); i++) {
                    if (kbuf[i] != 0xffffffffu) ck.push_back(kbuf[i]);
                }
                std::sort(bk.begin(), bk.end());
                std::sort(ck.begin(), ck.end());
                const bool eq = (bk.size() == ck.size()) &&
                                std::equal(bk.begin(), bk.end(), ck.begin());
                if (!eq) mismatched++;
                uint32_t first_div = 0xffffffffu;
                const std::size_t n = std::min(bk.size(), ck.size());
                for (std::size_t i = 0; i < n; i++) {
                    if (bk[i] != ck[i]) { first_div = static_cast<uint32_t>(i); break; }
                }
                std::fprintf(stderr,
                    "[BUCKET_VERIFY] t=%u creal=%u start=%lld bucket_keys=%zu canon_keys=%zu "
                    "zero_slots=%u %s first_div=%d bk[0]=%08x ck[0]=%08x bk[last]=%08x ck[last]=%08x\n",
                    t, creal, (long long)starts[t], bk.size(), ck.size(), zero_slots,
                    eq ? "MATCH" : "MISMATCH", (int)first_div,
                    bk.empty() ? 0 : bk.front(), ck.empty() ? 0 : ck.front(),
                    bk.empty() ? 0 : bk.back(), ck.empty() ? 0 : ck.back());
            }
            std::fprintf(stderr, "[BUCKET_VERIFY] probed=%u dense(>64) tiles, mismatched=%u\n",
                         probed, mismatched);
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
        const bool skip_radix = [] {
            const char* v = std::getenv("GSPLAT_TT_BIN_NORADIX");
            return v && v[0] == '1';
        }();
        if (!skip_radix) {
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, false);
            GSPLAT_HOST_ZONE("host_finish_sort_radix");
            distributed::Finish(*ctx->cq);
        }
        const auto t_k1 = clk::now();
        T.kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

        const bool dev_publish = sort_device_publish_enabled();
        std::vector<uint32_t> out_aligned;  // host fallback / BIN_DEBUG only

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
            if (sort_blend_pipe_enabled()) {
                device_state::mark_sort_publish_pending();
            } else {
                distributed::Finish(*ctx->cq);
            }
            T.publish_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_pub0).count();
            // Resident blend reads sort_sorted_ids over NoC — skip the large ids
            // D2H + host Pass4 unless verify/debug needs the dense host vector.
            const bool need_host_ids = verify || need_host_sorted_ids ||
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

        // ── Optional binning self-check (GSPLAT_TT_BIN_DEBUG=1) ─────────
        // D2H the device-filled (pre-radix) keys/ids and compare to a host
        // binning of the reconstructed resident pairs. Localizes binning bugs
        // (keys vs ids vs placement) independent of the radix sort.
        if (const char* bd = std::getenv("GSPLAT_TT_BIN_DEBUG"); bd && bd[0] == '1') {
            finish_sort_cq_if_needed(ctx);
            std::vector<uint32_t> dkeys(P_aligned), dids(P_aligned);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, dkeys, ctx->buf_keys, true);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, dids, ctx->buf_ids, true);
            std::vector<uint32_t> gz(P_pad), tz(P_pad), kz(P_pad);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, gz, bgid, true);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tz, btid, true);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, kz, bkeep, true);
            const uint32_t Mpad = round_up(static_cast<uint32_t>(std::max<std::size_t>(M, 1)),
                                           ELEMS_PER_PAGE);
            std::vector<uint32_t> dz(Mpad);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, dz, bdep, true);
            // Host reference replicating the EXACT device padded layout: per
            // (core,tile) page-aligned blocks at base=hist[c*stride+t] holding
            // that core's kept pairs in gaussian-major order, tails padded with
            // max key (0xffffffff)/0. hist holds the per-(core,tile) bases.
            std::vector<uint32_t> hkeys(P_aligned, 0xffffffffu), hids(P_aligned, 0);
            std::vector<uint32_t> bcur(num_tiles, 0);
            for (uint32_t c = 0; c < num_cores; c++) {
                for (uint32_t t = 0; t < num_tiles; t++)
                    bcur[t] = hist[static_cast<std::size_t>(c) * stride + t];
                const uint32_t plo = ws.start[c] * ELEMS_PER_PAGE;
                const uint32_t phi = (ws.start[c] + ws.count[c]) * ELEMS_PER_PAGE;
                for (uint32_t p = plo; p < phi && p < P_full; p++) {
                    if (!kz[p]) continue;
                    const uint32_t g = static_cast<uint32_t>(static_cast<int32_t>(gz[p]));
                    const uint32_t t = static_cast<uint32_t>(static_cast<int32_t>(tz[p]));
                    const uint32_t pos = bcur[t]++;
                    hkeys[pos] = dz[g];
                    hids[pos] = g;
                }
            }
            std::size_t key_mism = 0, id_mism = 0, first_key = 0, first_id = 0;
            for (uint32_t i = 0; i < P_aligned; i++) {
                if (dkeys[i] != hkeys[i]) { if (!key_mism) first_key = i; key_mism++; }
                if (dids[i]  != hids[i])  { if (!id_mism)  first_id  = i; id_mism++; }
            }
            std::fprintf(stderr,
                "[BIN debug] P_aligned=%u key_mismatch=%zu (first@%zu) "
                "id_mismatch=%zu (first@%zu)\n",
                P_aligned, key_mism, first_key, id_mism, first_id);
            if (id_mism) {
                // Locate which tile + offset the first id mismatch falls in.
                for (uint32_t t = 0; t < num_tiles; t++) {
                    const uint32_t lo = pstart_elem[t];
                    const uint32_t hi = lo + tile_pad[t];
                    if (first_id >= lo && first_id < hi) {
                        std::fprintf(stderr,
                            "[BIN debug] first_id@%zu in tile=%u off=%u "
                            "pstart_elem=%u tile_pad=%u count_real=%lld\n",
                            first_id, t, (unsigned)(first_id - lo), lo,
                            tile_pad[t], (long long)counts[t]);
                        const uint32_t pg = (first_id / ELEMS_PER_PAGE) * ELEMS_PER_PAGE;
                        for (int dp = -16; dp <= 16; dp += 16) {
                            const long base = (long)pg + dp;
                            if (base < 0) continue;
                            std::fprintf(stderr, "  page@%ld dev:", base);
                            for (uint32_t k = 0; k < 16; k++)
                                std::fprintf(stderr, " %d", (int)dids[base + k]);
                            std::fprintf(stderr, "\n  page@%ld hst:", base);
                            for (uint32_t k = 0; k < 16; k++)
                                std::fprintf(stderr, " %d", (int)hids[base + k]);
                            std::fprintf(stderr, "\n");
                        }
                        break;
                    }
                }
            }
            if (const char* du = std::getenv("GSPLAT_TT_BIN_DUMP"); du && du[0] == '1') {
                std::vector<uint32_t> b2(static_cast<std::size_t>(num_cores) * stride);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, b2, ctx->buf_bin2d, true);
                // Per-core tile-0 summary at b2[c*stride + 0..7].
                struct Row { uint32_t base, cnt, g0, cid, ps, pc, off0, g1; };
                std::vector<Row> rows;
                for (uint32_t c = 0; c < num_cores; c++) {
                    const std::size_t o = static_cast<std::size_t>(c) * stride;
                    Row r{b2[o+0], b2[o+1], b2[o+2], b2[o+3], b2[o+4], b2[o+5], b2[o+6], b2[o+7]};
                    if (r.cnt > 0) rows.push_back(r);
                }
                std::sort(rows.begin(), rows.end(),
                          [](const Row& a, const Row& b){ return a.base < b.base; });
                std::fprintf(stderr, "[BIN dump] tile%u cross-core (sorted by base), %zu cores:\n",
                             dump_tile, rows.size());
                for (std::size_t i = 0; i < rows.size() && i < 16; i++) {
                    const Row& r = rows[i];
                    std::fprintf(stderr, "  core=%u base=%u cnt=%u off0=%u g0=%d g1=%d ps=%u pc=%u\n",
                        r.cid, r.base, r.cnt, r.off0, (int)r.g0, (int)r.g1, r.ps, r.pc);
                }
                // Write-out view (captured inside the loop) at b2[c*stride+16..].
                for (uint32_t c = 0; c < num_cores && c < 8; c++) {
                    const std::size_t o = static_cast<std::size_t>(c) * stride + 16;
                    if (b2[o + 1] == 0) continue;  // n==0 -> not captured
                    std::fprintf(stderr,
                        "  [wview] core=%u base=%u n=%u src=%u isp[src]=%d isp[src+1]=%d "
                        "offp=%u curp=%u\n",
                        b2[o+7], b2[o+0], b2[o+1], b2[o+2], (int)b2[o+3], (int)b2[o+4],
                        b2[o+5], b2[o+6]);
                }
            }
            if (key_mism && first_key < P_aligned) {
                std::fprintf(stderr, "[BIN debug] @%zu dev_key=%08x host_key=%08x "
                    "dev_id=%u host_id=%u\n", first_key, dkeys[first_key],
                    hkeys[first_key], dids[first_key], hids[first_key]);
                std::fprintf(stderr, "[BIN debug] tile0 dev_ids:");
                for (uint32_t i = 0; i < 10 && i < P_aligned; i++)
                    std::fprintf(stderr, " %u", dids[i]);
                std::fprintf(stderr, "\n[BIN debug] tile0 host_ids:");
                for (uint32_t i = 0; i < 10 && i < P_aligned; i++)
                    std::fprintf(stderr, " %u", hids[i]);
                std::fprintf(stderr, "\n[BIN debug] tile0 count=%lld pstart_elem0=%u\n",
                    (long long)counts[0], pstart_elem[0]);
            }
        }

        if (!dev_publish)
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);

        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0_rp).count();
        std::fprintf(stderr,
            "[SORT] stage=RP P=%u P_kept=%u num_tiles=%u max_tile_n=%u bin=%.2f "
            "up=%.2f kernel=%.2f d2h=%.2f compact=%.2f publish=%.2f total=%.2fms\n",
            P_full, P_kept, num_tiles, max_n, T.bin_ms, T.upload_ms, T.kernel_ms,
            T.d2h_ms, T.compact_ms, T.publish_ms, T.total_ms);
        if (const char* st = std::getenv("GSPLAT_TT_SORT_TIMING"); st && st[0] == '1') {
            // STEP-4 diag: split bin into the device count kernel(+Finish), the
            // 450KB histogram D2H, and the host per-tile/page-layout/LPT build —
            // i.e. how much of bin is a HOST bridge (the on-device-LPT target).
            const double count_ms =
                std::chrono::duration<double, std::milli>(t_cnt - t_bin0).count();
            const double histd2h_ms =
                std::chrono::duration<double, std::milli>(t_d2h - t_cnt).count();
            const double hostbuild_ms =
                std::chrono::duration<double, std::milli>(t_bin1 - t_d2h).count();
            std::fprintf(stderr,
                "[SORT_TIMING] count_kernel=%.3f hist_d2h=%.3f host_build=%.3f "
                "h2d_up=%.3f | host_bridge(d2h+build+up)=%.3f\n",
                count_ms, histd2h_ms, hostbuild_ms, T.upload_ms,
                histd2h_ms + hostbuild_ms + T.upload_ms);
        }

        if (verify) {
            finish_sort_cq_if_needed(ctx);
            // Reconstruct host pairs from the resident buffers for the CPU
            // reference comparison (gaussian-major compaction over keep[]).
            std::vector<uint32_t> gz(P_pad), tz(P_pad), kz(P_pad), dz;
            distributed::EnqueueReadMeshBuffer(*ctx->cq, gz, bgid, true);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tz, btid, true);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, kz, bkeep, true);
            const uint32_t Mpad = round_up(static_cast<uint32_t>(std::max<std::size_t>(M, 1)),
                                           ELEMS_PER_PAGE);
            dz.resize(Mpad);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, dz, bdep, true);
            std::vector<int64_t> hg, ht;
            hg.reserve(P_kept);
            ht.reserve(P_kept);
            for (uint32_t p = 0; p < P_full; p++) {
                if (kz[p]) {
                    hg.push_back(static_cast<int64_t>(static_cast<int32_t>(gz[p])));
                    ht.push_back(static_cast<int64_t>(static_cast<int32_t>(tz[p])));
                }
            }
            std::vector<float> hd(M);
            for (std::size_t m = 0; m < M; m++)
                std::memcpy(&hd[m], &dz[m], 4);
            verify_vs_cpu(result, hg.data(), ht.data(), hd.data(), hg.size(), M,
                          tiles_x, tiles_y, pool, /*stage=*/2);
        }

        if (device_ok) *device_ok = true;
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
    const bool need_host_sorted_ids) {
    auto set_fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::SortResult{};
    };

    // Stage selection: STAGE=0 -> S0 (host sort, resident only). Otherwise S1.
    int stage = 1;
    if (const char* s = std::getenv("GSPLAT_TT_SORT_STAGE"); s != nullptr) {
        stage = std::atoi(s);
    }
    const bool verify = [] {
        const char* v = std::getenv("GSPLAT_TT_SORT_VERIFY");
        return v != nullptr && v[0] == '1';
    }();

    auto* ctx = ensure_context();
    if (ctx == nullptr) return set_fail();

    SortCallTimings tlocal;
    auto& T = (timings ? *timings : tlocal);
    T.stage = stage;
    using clk = std::chrono::high_resolution_clock;
    const auto t_total0 = clk::now();

    const uint32_t num_tiles = static_cast<uint32_t>(tiles_x) * static_cast<uint32_t>(tiles_y);

    // ── R4/R5: resident-pairs device binning (GSPLAT_TT_RESIDENT_PAIRS=1) ─
    // Reads the resident full-P (gid,tid) pairs + keep mask tile_assign left in
    // device_state, plus the resident proj_m_depth, and bins them on-device
    // into the page-aligned per-tile (key,id) layout the radix kernel consumes.
    // Eliminates the host pair D2H, the host compaction, the host Pass1/Pass2
    // binning, and the keys/ids re-upload. depths/gaussian_ids/tile_ids host
    // arguments are ignored (may be empty).
    const bool resident_pairs = [] {
        const char* v = std::getenv("GSPLAT_TT_RESIDENT_PAIRS");
        return v != nullptr && v[0] == '1';
    }();
    if (resident_pairs && stage >= 1) {
        gsplat_cpu::SortResult rr = sort_resident_pairs(
            ctx, num_tiles, tiles_x, tiles_y, M, verify, need_host_sorted_ids,
            pool, device_ok, T);
        return rr;
    }

    // ── S0: run CPU sort, publish contiguous outputs resident. ───────────
    if (stage <= 0) {
        const gsplat_cpu::SortResult cpu = gsplat_cpu::sort_and_bin(
            gaussian_ids, tile_ids, depths, P, M, tiles_x, tiles_y, pool);
        try {
            publish_resident(ctx, cpu.sorted_gaussian_ids, cpu.tile_ranges, &T.publish_ms);
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::sort] S0 publish failed: " << e.what() << "\n";
            return set_fail();
        }
        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();
        std::fprintf(stderr, "[SORT] stage=S0 P=%zu num_tiles=%u publish=%.2fms total=%.2fms\n",
                     cpu.sorted_gaussian_ids.size(), num_tiles, T.publish_ms, T.total_ms);
        if (verify) {
            verify_vs_cpu(cpu, gaussian_ids, tile_ids, depths, P, M, tiles_x, tiles_y, pool, 0);
        }
        if (device_ok) *device_ok = true;
        return cpu;
    }

    // ── S1: host binning -> device radix -> host compaction. ─────────────
    gsplat_cpu::SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);
    if (P == 0) {
        try {
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);
        } catch (...) {}
        if (device_ok) *device_ok = true;
        return result;
    }

    try {
        const auto t_bin0 = clk::now();
        // Pass 1: per-tile counts.
        std::vector<int64_t> counts(num_tiles, 0);
        for (std::size_t i = 0; i < P; i++) {
            counts[static_cast<std::size_t>(tile_ids[i])]++;
        }
        // Contiguous exclusive starts (== CPU tile_ranges) + page-aligned starts.
        std::vector<int64_t> starts(num_tiles, 0);
        std::vector<uint32_t> pstart_page(num_tiles, 0);
        std::vector<uint32_t> pstart_elem(num_tiles, 0);
        int64_t cstart = 0;
        uint32_t apage = 0;
        uint32_t max_n = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const int64_t c = counts[t];
            starts[t] = cstart;
            cstart += c;
            if (c > 0) {
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] = starts[t] + c;
                pstart_page[t] = apage;
                pstart_elem[t] = apage * ELEMS_PER_PAGE;
                const uint32_t npages =
                    (static_cast<uint32_t>(c) + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
                apage += npages;
                if (static_cast<uint32_t>(c) > max_n) max_n = static_cast<uint32_t>(c);
            }
        }
        // Tile too large for the L1 scratch budget -> CPU fallback.
        if (max_n > MAX_TILE_ENTRIES) {
            std::cerr << "[gsplat_tt::sort] tile n=" << max_n
                      << " exceeds MAX_TILE_ENTRIES=" << MAX_TILE_ENTRIES
                      << "; falling back to CPU\n";
            return set_fail();
        }

        const uint32_t total_pages = std::max<uint32_t>(apage, 1);
        const uint32_t P_aligned = total_pages * ELEMS_PER_PAGE;

        // Pass 2: stable sequential scatter into the aligned layout. Iterating
        // i in input order keeps within-tile order == CPU's stable order.
        std::vector<uint32_t> keys(P_aligned, 0);
        std::vector<uint32_t> ids(P_aligned, 0);
        std::vector<uint32_t> cursor(num_tiles, 0);
        for (std::size_t i = 0; i < P; i++) {
            const std::size_t t = static_cast<std::size_t>(tile_ids[i]);
            const int64_t g = gaussian_ids[i];
            const uint32_t pos = pstart_elem[t] + cursor[t]++;
            keys[pos] = std::bit_cast<uint32_t>(depths[static_cast<std::size_t>(g)]);
            ids[pos] = static_cast<uint32_t>(g);
        }
        const auto t_bin1 = clk::now();
        T.bin_ms = std::chrono::duration<double, std::milli>(t_bin1 - t_bin0).count();

        // ── Allocate / grow DRAM buffers ────────────────────────────────
        const std::size_t aligned_bytes = static_cast<std::size_t>(P_aligned) * 4;
        if (!ctx->buf_keys || ctx->cap_aligned_bytes < aligned_bytes) {
            ctx->buf_keys = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_ids  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_out  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->cap_aligned_bytes = aligned_bytes;
        }

        // tmeta: (pstart_page, n) per tile, uint32 SoA, 64B pages.
        const uint32_t tmeta_count = num_tiles * 2;
        const uint32_t tmeta_pad = round_up(std::max<uint32_t>(tmeta_count, 1), ELEMS_PER_PAGE);
        const std::size_t tmeta_bytes = static_cast<std::size_t>(tmeta_pad) * 4;
        if (!ctx->buf_tmeta || ctx->cap_tmeta_bytes < tmeta_bytes) {
            ctx->buf_tmeta = make_dram(ctx->mesh_device.get(), tmeta_bytes);
            ctx->cap_tmeta_bytes = tmeta_bytes;
        }
        std::vector<uint32_t> tmeta(tmeta_pad, 0);
        for (uint32_t t = 0; t < num_tiles; t++) {
            tmeta[t * 2 + 0] = pstart_page[t];
            tmeta[t * 2 + 1] = static_cast<uint32_t>(counts[t]);
        }

        // LPT tile->core assignment over non-empty tiles.
        const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
        const LptAssignment lpt = build_lpt(counts, num_tiles, num_cores);
        const uint32_t tile_ids_count = static_cast<uint32_t>(lpt.flat_tile_ids.size());
        const uint32_t tile_ids_pad = round_up(std::max<uint32_t>(tile_ids_count, 1), ELEMS_PER_PAGE);
        const std::size_t tile_ids_bytes = static_cast<std::size_t>(tile_ids_pad) * 4;
        if (!ctx->buf_tile_ids || ctx->cap_tile_ids_bytes < tile_ids_bytes) {
            ctx->buf_tile_ids = make_dram(ctx->mesh_device.get(), tile_ids_bytes);
            ctx->cap_tile_ids_bytes = tile_ids_bytes;
        }
        // Grow-only buffers: keys/ids/tile_ids whole-buffer writes must be sized
        // to capacity (a smaller-P frame keeps a larger hero-frame allocation).
        const uint32_t cap_aligned_elems =
            static_cast<uint32_t>(ctx->cap_aligned_bytes / 4);
        const uint32_t cap_tile_ids_elems =
            static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
        keys.resize(cap_aligned_elems, 0);
        ids.resize(cap_aligned_elems, 0);
        std::vector<uint32_t> tile_ids_flat(cap_tile_ids_elems, 0);
        std::copy(lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());

        // ── Upload ──────────────────────────────────────────────────────
        const auto t_up0 = clk::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_keys, keys, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_ids, ids, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
        GSPLAT_HOST_ZONE("host_finish_sort_upload");
        distributed::Finish(*ctx->cq);
        const auto t_up1 = clk::now();
        T.upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

        // ── Device radix kernel ─────────────────────────────────────────
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
        distributed::Finish(*ctx->cq);
        const auto t_k1 = clk::now();
        T.kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

        // ── D2H aligned sorted ids ──────────────────────────────────────
        // buf_out is grow-only: read the whole buffer -> size dst to capacity.
        const auto t_d0 = clk::now();
        std::vector<uint32_t> out_aligned(cap_aligned_elems);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, out_aligned, ctx->buf_out, true);
        const auto t_d1 = clk::now();
        T.d2h_ms = std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();

        // ── Pass 4: compact aligned segments -> contiguous, widen to int64.
        const auto t_c0 = clk::now();
        result.sorted_gaussian_ids.resize(P);
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t n = static_cast<uint32_t>(counts[t]);
            if (n == 0) continue;
            const uint32_t src = pstart_elem[t];
            const std::size_t dst = static_cast<std::size_t>(starts[t]);
            for (uint32_t k = 0; k < n; k++) {
                result.sorted_gaussian_ids[dst + k] =
                    static_cast<int64_t>(out_aligned[src + k]);
            }
        }
        const auto t_c1 = clk::now();
        T.compact_ms = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();

        // ── Publish contiguous resident outputs ─────────────────────────
        publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);

        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();
        std::fprintf(stderr,
            "[SORT] stage=S1 P=%zu num_tiles=%u max_tile_n=%u bin=%.2f up=%.2f "
            "kernel=%.2f d2h=%.2f compact=%.2f publish=%.2f total=%.2fms\n",
            P, num_tiles, max_n, T.bin_ms, T.upload_ms, T.kernel_ms, T.d2h_ms,
            T.compact_ms, T.publish_ms, T.total_ms);

        if (verify) {
            verify_vs_cpu(result, gaussian_ids, tile_ids, depths, P, M, tiles_x, tiles_y, pool, 1);
        }

        if (device_ok) *device_ok = true;
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::sort] S1 call failed: " << e.what() << "\n";
        return set_fail();
    }
}

}  // namespace gsplat_tt
