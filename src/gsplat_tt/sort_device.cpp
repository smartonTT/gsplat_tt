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

#include "gsplat_tt/sort.h"
#include "gsplat_tt/device_state.h"

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

    std::shared_ptr<distributed::MeshBuffer> buf_tile_ids;  // LPT tile-id list
    std::size_t cap_tile_ids_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tmeta;     // (pstart_page, n)
    std::size_t cap_tmeta_bytes = 0;

    // Resident contiguous outputs (published for downstream device consumers).
    std::shared_ptr<distributed::MeshBuffer> buf_sorted_ids;   // contiguous, P
    std::size_t cap_sorted_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_ranges;  // num_tiles*2
    std::size_t cap_ranges_bytes = 0;
};

static std::shared_ptr<distributed::MeshBuffer> make_dram(
    distributed::MeshDevice* dev, std::size_t bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = PAGE_BYTES, .buffer_type = BufferType::DRAM};
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

    std::vector<uint32_t> ct;
    for (int i = 0; i < 7; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    std::map<std::string, std::string> defines;
    if (const char* v = std::getenv("GSPLAT_TT_BIN_NODEPTH"); v && v[0] == '1')
        defines["BIN_NO_DEPTH"] = "1";
    if (const char* v = std::getenv("GSPLAT_TT_BIN_DUMP"); v && v[0] == '1')
        defines["BIN_DUMP"] = "1";
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

static SortDeviceContext init_context() {
    SortDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    build_program_bin(ctx);
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
    return a;
}

// Publish the contiguous (sorted_ids, tile_ranges) into device_state as uint32
// DRAM buffers so downstream device stages can read them resident.
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
    std::vector<uint32_t> sids(P_pad, 0);
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
    std::vector<uint32_t> ranges(R_pad, 0);
    for (uint32_t i = 0; i < R; i++)
        ranges[i] = static_cast<uint32_t>(tile_ranges[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ranges, ranges, false);
    distributed::Finish(*ctx->cq);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (publish_ms) *publish_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
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
    std::size_t M, bool verify, gsplat_cpu::ThreadPool* pool,
    bool* device_ok, SortCallTimings& T) {
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

        // ── Pass A: per-core histogram (count) ──────────────────────────
        const auto t_bin0 = clk::now();
        auto launch_bin = [&](uint32_t mode) {
            Program& prog = ctx->wl_bin.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(prog, ctx->kbin, core, {
                    static_cast<uint32_t>(bgid->address()),
                    static_cast<uint32_t>(btid->address()),
                    static_cast<uint32_t>(bkeep->address()),
                    static_cast<uint32_t>(bdep->address()),
                    static_cast<uint32_t>(ctx->buf_bin2d->address()),
                    ctx->buf_keys ? static_cast<uint32_t>(ctx->buf_keys->address()) : 0u,
                    ctx->buf_ids ? static_cast<uint32_t>(ctx->buf_ids->address()) : 0u,
                    ws.start[c], ws.count[c], P_full, num_tiles, stride, c, mode,
                    dump_tile,  // arg14: tile to dump under BIN_DUMP
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin, false);
            distributed::Finish(*ctx->cq);
        };
        launch_bin(0);

        // D2H the per-core histograms.
        std::vector<uint32_t> hist(static_cast<std::size_t>(num_cores) * stride);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, hist, ctx->buf_bin2d, true);

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
            for (uint32_t c = 0; c < num_cores; c++) {
                const std::size_t idx = static_cast<std::size_t>(c) * stride + t;
                const uint32_t h = hist[idx];
                hist[idx] = apage * ELEMS_PER_PAGE;  // page-aligned base for (c,t)
                if (h > 0) apage += (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
            }
            const uint32_t pad_pages = apage - tile_start_page;
            tile_pad[t] = pad_pages * ELEMS_PER_PAGE;
            if (creal > 0) {
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] = starts[t] + creal;
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
        const LptAssignment lpt = build_lpt(pad_counts, num_tiles, num_cores);
        const uint32_t tile_ids_count = static_cast<uint32_t>(lpt.flat_tile_ids.size());
        const uint32_t tile_ids_pad = round_up(std::max<uint32_t>(tile_ids_count, 1), ELEMS_PER_PAGE);
        const std::size_t tile_ids_bytes = static_cast<std::size_t>(tile_ids_pad) * 4;
        if (!ctx->buf_tile_ids || ctx->cap_tile_ids_bytes < tile_ids_bytes) {
            ctx->buf_tile_ids = make_dram(ctx->mesh_device.get(), tile_ids_bytes);
            ctx->cap_tile_ids_bytes = tile_ids_bytes;
        }
        std::vector<uint32_t> tile_ids_flat(tile_ids_pad, 0);
        std::copy(lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());

        // ── Upload: base offsets (into bin2d), tmeta, tile_ids ──────────
        const auto t_up0 = clk::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d, hist, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
        distributed::Finish(*ctx->cq);
        const auto t_up1 = clk::now();
        T.upload_ms = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

        // ── Pass B: device scatter into aligned (key,id) layout ─────────
        const auto t_sc0 = clk::now();
        launch_bin(1);
        const auto t_sc1 = clk::now();
        T.bin_ms += std::chrono::duration<double, std::milli>(t_sc1 - t_sc0).count();

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
            distributed::Finish(*ctx->cq);
        }
        const auto t_k1 = clk::now();
        T.kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

        // ── D2H aligned sorted ids + Pass4 compact -> contiguous ────────
        const auto t_d0 = clk::now();
        std::vector<uint32_t> out_aligned(P_aligned);
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

        // ── Optional binning self-check (GSPLAT_TT_BIN_DEBUG=1) ─────────
        // D2H the device-filled (pre-radix) keys/ids and compare to a host
        // binning of the reconstructed resident pairs. Localizes binning bugs
        // (keys vs ids vs placement) independent of the radix sort.
        if (const char* bd = std::getenv("GSPLAT_TT_BIN_DEBUG"); bd && bd[0] == '1') {
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

        publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);

        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0_rp).count();
        std::fprintf(stderr,
            "[SORT] stage=RP P=%u P_kept=%u num_tiles=%u max_tile_n=%u bin=%.2f "
            "up=%.2f kernel=%.2f d2h=%.2f compact=%.2f publish=%.2f total=%.2fms\n",
            P_full, P_kept, num_tiles, max_n, T.bin_ms, T.upload_ms, T.kernel_ms,
            T.d2h_ms, T.compact_ms, T.publish_ms, T.total_ms);

        if (verify) {
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
        return fail();
    }
}

}  // namespace

bool sort_device_ready() { return ensure_context() != nullptr; }

void sort_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        slot->buf_keys.reset();
        slot->buf_ids.reset();
        slot->buf_out.reset();
        slot->buf_tile_ids.reset();
        slot->buf_tmeta.reset();
        slot->buf_sorted_ids.reset();
        slot->buf_tile_ranges.reset();
        slot->buf_bin2d.reset();
        // NOTE: do NOT close mesh_device — device_state owns it.
        slot.reset();
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
    SortCallTimings* timings) {
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
        gsplat_cpu::SortResult rr =
            sort_resident_pairs(ctx, num_tiles, tiles_x, tiles_y, M, verify,
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
        std::vector<uint32_t> tile_ids_flat(tile_ids_pad, 0);
        std::copy(lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());

        // ── Upload ──────────────────────────────────────────────────────
        const auto t_up0 = clk::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_keys, keys, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_ids, ids, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
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
        const auto t_d0 = clk::now();
        std::vector<uint32_t> out_aligned(P_aligned);
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
