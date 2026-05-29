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
#include <memory>
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

inline uint32_t round_up(uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; }

struct SortDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    distributed::MeshWorkload workload;
    KernelHandle kernel{};

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

static SortDeviceContext init_context() {
    SortDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
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
