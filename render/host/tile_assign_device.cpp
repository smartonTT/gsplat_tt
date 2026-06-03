// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt tile_assign — amendment-002 tt-006.
//
// STAGE S1: K1 (per-Gaussian AABB / tiles_per_gaussian) + H1 (host exclusive
// prefix-sum) + K2 (pair-centric scatter). Produces the AABB-only (gid, tid)
// pair set in gaussian-major order, bit-identical to gsplat_cpu::tile_assign
// with the per-pair Mahalanobis cull disabled. The cull (K3/K4 + compaction)
// lands in S2.
//
// Layout: every per-Gaussian / per-pair buffer is SoA int32/fp32 with 64-byte
// pages (16 elements). Kernels read/write whole pages so the interleaved DRAM
// page stride stays aligned (a 48B layout previously caused a silent zero-row
// bug). Pair buffers are sized to the exact padded P each call.

#include "tile_assign.h"
#include "device_state.h"
#include "env_config.h"
#include "host_tracy.hpp"
#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/kernel_types.hpp"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace gsplat_tt {
namespace {

constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t PAGE_BYTES = ELEMS_PER_PAGE * 4;  // 64

inline uint32_t round_up(uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; }

// Persistent host pool for the K3 per-Gaussian m2_thresh/opacity precompute.
// The loop is embarrassingly parallel and dominated by std::log over M (~9ms
// single-threaded on the hero frame). Each element is computed independently
// and identically (same std::log, same memcpy bit-pattern) so the result is
// byte-identical regardless of thread count / split — no reduction, no
// ordering dependence. Lazily constructed once (hardware_concurrency threads).
static gsplat_cpu::ThreadPool& k3_pool() {
    static gsplat_cpu::ThreadPool pool(0);
    return pool;
}

struct TileAssignDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    distributed::MeshWorkload wl_k1;
    distributed::MeshWorkload wl_k2;
    distributed::MeshWorkload wl_cull;
    distributed::MeshWorkload wl_scan1;
    distributed::MeshWorkload wl_scan_bases;
    distributed::MeshWorkload wl_scan2;
    distributed::MeshWorkload wl_m2thr;
    KernelHandle k1{};
    KernelHandle k2{};
    KernelHandle k4{};
    KernelHandle ks1{};
    KernelHandle ks_bases{};
    KernelHandle ks2{};
    KernelHandle km3{};

    // Cached DRAM buffers (grow-on-demand).
    std::shared_ptr<distributed::MeshBuffer> buf_px;
    std::shared_ptr<distributed::MeshBuffer> buf_py;
    std::shared_ptr<distributed::MeshBuffer> buf_rx;
    std::shared_ptr<distributed::MeshBuffer> buf_ry;
    std::shared_ptr<distributed::MeshBuffer> buf_tpg;
    // cull inputs (M-sized).
    std::shared_ptr<distributed::MeshBuffer> buf_a;
    std::shared_ptr<distributed::MeshBuffer> buf_b;
    std::shared_ptr<distributed::MeshBuffer> buf_c;
    std::shared_ptr<distributed::MeshBuffer> buf_m2thr;
    std::size_t cap_m_bytes = 0;  // capacity of the M-sized buffers (bytes)

    std::shared_ptr<distributed::MeshBuffer> buf_offs;
    std::size_t cap_offs_bytes = 0;

    // On-device exclusive scan (GSPLAT_TT_TA_DEVICE_SCAN): per-core partial
    // totals, one dedicated 64B page per core (sized at init from num_cores).
    std::shared_ptr<distributed::MeshBuffer> buf_core_total;
    // Exclusive bases from scan_bases (one page per core); scan2 reads its slot.
    std::shared_ptr<distributed::MeshBuffer> buf_core_base;

    std::shared_ptr<distributed::MeshBuffer> buf_gids;
    std::shared_ptr<distributed::MeshBuffer> buf_tids;
    std::shared_ptr<distributed::MeshBuffer> buf_keep;
    std::size_t cap_p_bytes = 0;
    // True once buf_keep holds all-1s up to its full capacity. In the
    // GSPLAT_TT_TA_NO_CULL default path the keep mask is a constant all-ones
    // array (the per-pair cull is off; the blend's microblock cull rejects the
    // empty-corner pairs downstream). The all-ones fill only has to happen ONCE
    // per (re)allocation of buf_keep — DRAM persists across frames and nothing
    // overwrites keep when the cull is off — so we drop the ~13.5 MB/frame H2D
    // of constant 1s + its Finish() stage-lock from every steady-state frame.
    // Reset to false whenever buf_keep is (re)allocated or K4 overwrites it.
    bool buf_keep_all_ones = false;

    // R4/R5: tiny 1-page buffer publishing the full P (pre-cull pair count) so
    // the resident-pairs sort path knows how many pairs to bin.
    std::shared_ptr<distributed::MeshBuffer> buf_pairs_P;
};

static std::shared_ptr<distributed::MeshBuffer> make_dram(
    distributed::MeshDevice* dev, std::size_t bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = PAGE_BYTES, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

static void build_program_k1(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    for (uint32_t id = 0; id < 5; id++) scratch_cb(id);  // px,py,rx,ry,out

    std::vector<uint32_t> ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.k1 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_bbox.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_k1.add_program(device_range, std::move(program));
}

static void build_program_k2(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    for (uint32_t id = 0; id < 7; id++) scratch_cb(id);  // offs,px,py,rx,ry,gid,tid

    std::vector<uint32_t> ct;
    for (int i = 0; i < 7; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.k2 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scatter.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_k2.add_program(device_range, std::move(program));
}

static void build_program_cull(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    // gid,tid,a,b,c,px,py,m2thr,keep (opacok folded into the m2thr sentinel)
    for (uint32_t id = 0; id < 9; id++) scratch_cb(id);

    std::vector<uint32_t> ct;
    for (int i = 0; i < 9; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.k4 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_cull.add_program(device_range, std::move(program));
}

static void build_program_scan_reduce(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    for (uint32_t id = 0; id < 2; id++) scratch_cb(id);  // tpg, total

    std::vector<uint32_t> ct;
    for (int i = 0; i < 2; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.ks1 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scan_reduce.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_scan1.add_program(device_range, std::move(program));
}

static void build_program_m2thr(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    scratch_cb(0);  // op
    scratch_cb(1);  // m2thr out

    std::vector<uint32_t> ct;
    for (int i = 0; i < 2; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.km3 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_m2thr.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_m2thr.add_program(device_range, std::move(program));
}

static void build_program_scan_bases(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreCoord core0{0, 0};
    const CoreRangeSet cores(core0);
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    scratch_cb(0);
    scratch_cb(1);

    std::vector<uint32_t> ct;
    for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.ks_bases = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scan_bases.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_scan_bases.add_program(device_range, std::move(program));
}

static void build_program_scan_add(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    for (uint32_t id = 0; id < 3; id++) scratch_cb(id);  // tpg, offs, base

    std::vector<uint32_t> ct;
    for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.ks2 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scan_add.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_scan2.add_program(device_range, std::move(program));
}

static TileAssignDeviceContext init_context() {
    TileAssignDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program_k1(ctx);
    build_program_k2(ctx);
    build_program_cull(ctx);
    build_program_scan_reduce(ctx);
    build_program_scan_bases(ctx);
    build_program_scan_add(ctx);
    build_program_m2thr(ctx);
    // core_total / core_base: one dedicated 64B page per core.
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;
    const std::size_t cores_bytes = static_cast<std::size_t>(num_cores) * PAGE_BYTES;
    ctx.buf_core_total = make_dram(ctx.mesh_device.get(), cores_bytes);
    ctx.buf_core_base  = make_dram(ctx.mesh_device.get(), cores_bytes);
    ctx.buf_pairs_P = make_dram(ctx.mesh_device.get(), PAGE_BYTES);
    device_state::register_buffer("ta_pairs_P", ctx.buf_pairs_P);
    return ctx;
}

static std::unique_ptr<TileAssignDeviceContext>& context_slot() {
    static std::unique_ptr<TileAssignDeviceContext> ctx;
    return ctx;
}

static TileAssignDeviceContext* ensure_context() {
    auto& slot = context_slot();
    if (!slot) {
        try {
            slot = std::make_unique<TileAssignDeviceContext>(init_context());
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::tile_assign] device init failed: "
                      << e.what() << "\n";
            slot.reset();
        }
    }
    return slot.get();
}

struct WorkSplit {
    std::vector<uint32_t> start;
    std::vector<uint32_t> count;
};

static WorkSplit split_pages(uint32_t num_pages, uint32_t num_cores) {
    WorkSplit ws;
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

// GSPLAT_TT_TA_STATS: host-only diagnostic of the K2 scatter per-core work
// distribution, computed purely from the prefix-sum offs[] (no device change,
// no effect on the run). Reports two proxies for the contiguous page split:
//   * pairs/core   == the WRITE work (every core writes its 16-pair pages).
//   * gspan/core   == the gaussian boundaries crossed == set_g calls (each a
//     binary-search + 4 NoC attr reads + AABB recompute) == the READ/COMPUTE
//     work. The scene is spatially clustered (tiles-per-gaussian varies), so a
//     contiguous page chunk hands a core a non-representative gspan -> makespan
//     skew lands on gspan, not on the (uniform) pair writes.
// Also projects the gspan skew a BLOCKED-CYCLIC (strided) split of block size
// B would give, so the fix's headroom is visible before it ships.
inline bool ta_stats_enabled() {
    const char* v = std::getenv("GSPLAT_TT_TA_STATS");
    return v != nullptr && v[0] == '1';
}

static uint32_t ta_block_pages() {
    const char* v = std::getenv("GSPLAT_TT_TA_BLOCK");
    if (v == nullptr || v[0] == '\0') return 8u;  // default block size (pages)
    const long b = std::strtol(v, nullptr, 10);
    return (b <= 0) ? 1u : static_cast<uint32_t>(b);
}

// Largest g in [0, M-1] with offs[g] <= p (the gaussian owning pair p).
static uint32_t owner_of(const std::vector<uint32_t>& offs, uint32_t M, uint32_t p) {
    uint32_t lo = 0, hi = (M ? M - 1 : 0);
    while (lo < hi) {
        const uint32_t mid = (lo + hi + 1) >> 1;
        if (offs[mid] <= p) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Sum the gaussian-boundary crossings (== set_g calls) over a contiguous page
// range [pg_lo, pg_hi): one binary search at pg_lo*16 then linear advance to the
// owner of the last real pair. Padding pairs (>= P) cost no advance.
static uint32_t gspan_of_range(const std::vector<uint32_t>& offs, uint32_t M,
                               uint32_t P, uint32_t pg_lo, uint32_t pg_hi) {
    const uint32_t p0 = pg_lo * ELEMS_PER_PAGE;
    uint32_t p1 = pg_hi * ELEMS_PER_PAGE;
    if (p1 > P) p1 = P;
    if (p1 <= p0) return 0;
    const uint32_t g0 = owner_of(offs, M, p0);
    const uint32_t g1 = owner_of(offs, M, p1 - 1);
    return g1 - g0 + 1;
}

static void ta_dump_k2_stats(const std::vector<uint32_t>& offs, uint32_t M,
                             uint32_t P, uint32_t k2_pages, uint32_t num_cores) {
    auto skew = [](const std::vector<double>& v) {
        double mn = 1e30, mx = 0, sum = 0;
        for (double x : v) { mn = std::min(mn, x); mx = std::max(mx, x); sum += x; }
        const double mean = v.empty() ? 0.0 : sum / v.size();
        return std::array<double, 4>{mn, mx, mean, mean > 0 ? mx / mean : 0.0};
    };
    // Contiguous split (production K2 today).
    const WorkSplit wc = split_pages(k2_pages, num_cores);
    std::vector<double> c_pairs(num_cores), c_gspan(num_cores);
    for (uint32_t c = 0; c < num_cores; c++) {
        const uint32_t lo = wc.start[c], hi = wc.start[c] + wc.count[c];
        uint32_t p0 = lo * ELEMS_PER_PAGE, p1 = hi * ELEMS_PER_PAGE;
        if (p1 > P) p1 = P;
        if (p1 < p0) p1 = p0;
        c_pairs[c] = static_cast<double>(p1 - p0);
        c_gspan[c] = gspan_of_range(offs, M, P, lo, hi);
    }
    // Blocked-cyclic (strided) split: core c owns blocks c, c+nc, c+2nc, ...
    const uint32_t B = ta_block_pages();
    const uint32_t num_blocks = (k2_pages + B - 1) / B;
    std::vector<double> s_pairs(num_cores, 0), s_gspan(num_cores, 0);
    for (uint32_t bk = 0; bk < num_blocks; bk++) {
        const uint32_t c = bk % num_cores;
        const uint32_t lo = bk * B;
        uint32_t hi = lo + B;
        if (hi > k2_pages) hi = k2_pages;
        uint32_t p0 = lo * ELEMS_PER_PAGE, p1 = hi * ELEMS_PER_PAGE;
        if (p1 > P) p1 = P;
        if (p1 < p0) p1 = p0;
        s_pairs[c] += static_cast<double>(p1 - p0);
        s_gspan[c] += gspan_of_range(offs, M, P, lo, hi);
    }
    const auto cp = skew(c_pairs), cg = skew(c_gspan);
    const auto sp = skew(s_pairs), sg = skew(s_gspan);
    std::fprintf(stderr,
        "[TA_STATS] M=%u P=%u k2_pages=%u cores=%u block_B=%u num_blocks=%u\n"
        "[TA_STATS] CONTIG pairs(write): min=%.0f max=%.0f mean=%.0f max/mean=%.4f\n"
        "[TA_STATS] CONTIG gspan(set_g): min=%.0f max=%.0f mean=%.0f max/mean=%.4f\n"
        "[TA_STATS] STRIDED pairs(write): min=%.0f max=%.0f mean=%.0f max/mean=%.4f\n"
        "[TA_STATS] STRIDED gspan(set_g): min=%.0f max=%.0f mean=%.0f max/mean=%.4f "
        "(+%u binsearch/core)\n",
        M, P, k2_pages, num_cores, B, num_blocks,
        cp[0], cp[1], cp[2], cp[3], cg[0], cg[1], cg[2], cg[3],
        sp[0], sp[1], sp[2], sp[3], sg[0], sg[1], sg[2], sg[3],
        num_blocks / num_cores);
}

}  // namespace

bool tile_assign_device_ready() { return ensure_context() != nullptr; }

void tile_assign_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        (void)slot.release();
    }
}

gsplat_cpu::TileAssignResult tile_assign_tt(
    const float* means_2d,
    const float* radii,
    std::size_t M,
    int image_height,
    int image_width,
    int tile_size,
    const float* covs_2d,
    const float* opacities,
    float contrib_floor,
    bool* device_ok,
    TileAssignCallTimings* timings) {
    auto set_fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::TileAssignResult{};
    };
    if (M == 0) {
        if (device_ok) *device_ok = true;
        return gsplat_cpu::TileAssignResult{};
    }
    auto* ctx = ensure_context();
    if (ctx == nullptr) return set_fail();

    TileAssignCallTimings tlocal;
    auto& T = (timings ? *timings : tlocal);
    using clk = std::chrono::high_resolution_clock;
    const auto t_total0 = clk::now();

    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;

    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t M_pad = round_up(Mu, ELEMS_PER_PAGE);
    const uint32_t num_cores = ctx->grid.x * ctx->grid.y;

    // ── R3: resident tile_assign inputs (GSPLAT_TT_RESIDENT_TA_IN=1) ──────
    // When on, K1/K2/K4 read the M-compact proj_m_* buffers the gather stage
    // left resident in device DRAM (SoA px/py/rx/ry means+radii, cov a/b/c)
    // over NoC instead of us re-uploading the host arrays each frame. The
    // kernels already consume per-component SoA, so the resident format drops
    // in with no repack. The m2_thresh/opacok precompute (K3), exclusive
    // prefix-sum (H1), compaction (H2) and the D2H of the final pairs stay
    // host-side bridges (removed in R4). Requires RESIDENT_PROJECT+
    // RESIDENT_GATHER; falls back to CPU (device_ok=false) if proj_m_* absent.
    const bool resident_in = true;  // RESIDENT_TA_IN=1
    // ── R4/R5: resident-pairs handoff (GSPLAT_TT_RESIDENT_PAIRS=1) ────────
    // When on (and the per-pair cull runs), TA keeps the full-P gaussian-major
    // (gid,tid) pairs + keep mask RESIDENT in DRAM and registers them in
    // device_state for the sort stage to bin on-device. The host D2H of the
    // pairs (28ms) and the sequential gaussian-major compaction (37ms) are
    // dropped — sort's device binning reads keep[] and compacts implicitly.
    const bool resident_pairs = true;  // RESIDENT_PAIRS=1
    // GSPLAT_TT_TA_TIMING=1: print a per-call host/device sub-stage breakdown
    // to stderr (works in resident-pairs mode, unlike GSPLAT_TT_TA_DEBUG). To
    // attribute the H2D bridges in isolation it inserts an extra Finish after
    // the offs upload and after the cull H2D — these syncs are ONLY added when
    // timing is enabled so the production path is unperturbed.
    const bool ta_timing = false;
    // GSPLAT_TT_TA_DEVICE_SCAN=1: exclusive prefix-sum on-device (scan_reduce ->
    // scan_bases -> scan_add). Eliminates full-M D2H(tpg)+H2D(offs) and the
    // host scan of num_cores partials; only a 64B read of P remains for early
    // exit + pair-buffer sizing. Integer = byte-exact.
    const bool device_scan = true;  // TA_DEVICE_SCAN=1
    // GSPLAT_TT_TA_NO_CULL (R1): skip the ta-stage per-pair Mahalanobis cull —
    // both K3 (per-Gaussian m2_thresh log) and K4 (per-pair cull). The cull
    // removes only ~4.6% of pairs (P 3.37M -> P_kept 3.21M on hero) while
    // costing ~80% of the stage (~103 ms of soft-float on the data-mover RISC).
    // Blend runs its OWN finer per-microblock Mahalanobis cull downstream
    // (GSPLAT_TT_SFPU_CULL / MB_DEVCULL), which rejects the empty-corner pairs
    // this skips. When set we publish the resident pairs with an all-ones keep
    // mask (P_kept == P) so sort passes every AABB pair through. kTaNoCullDefault
    // is the shipped default when the env var is unset; gate stays env-flippable
    // (set GSPLAT_TT_TA_NO_CULL=0 to force the old per-pair cull back on).
    //
    // SHIPPED DEFAULT ON (R1, iter 20): measured on hero/bicycle (yyzo-bh-03,
    // 1 view, all-resident): hero_vs_ref 63.85 dB == baseline (NO regression —
    // blend's microblock cull rejects the corner pairs), ta 128.7 -> 36.9 ms,
    // ms/view 391.4 -> 306.0. See opt/ta-stage-analysis.md §8.
    const bool ta_no_cull = true;
    std::shared_ptr<distributed::MeshBuffer> res_px, res_py, res_rx, res_ry,
        res_a, res_b, res_c;
    std::shared_ptr<distributed::MeshBuffer> res_op;

    try {
        if (resident_in) {
            res_px = device_state::get_buffer("proj_m_px");
            res_py = device_state::get_buffer("proj_m_py");
            res_rx = device_state::get_buffer("proj_m_rx");
            res_ry = device_state::get_buffer("proj_m_ry");
            res_a  = device_state::get_buffer("proj_m_a");
            res_b  = device_state::get_buffer("proj_m_b");
            res_c  = device_state::get_buffer("proj_m_c");
            res_op = device_state::get_buffer("proj_m_opacity");
            auto res_M = device_state::get_buffer("proj_M");
            if (!res_px || !res_py || !res_rx || !res_ry || !res_a || !res_b ||
                !res_c || !res_M) {
                std::cerr << "[gsplat_tt::tile_assign] RESIDENT_TA_IN set but "
                             "proj_m_* not resident; needs RESIDENT_PROJECT+"
                             "RESIDENT_GATHER\n";
                return set_fail();
            }
            // proj_M is published by gather on-device; host M is the same value
            // (read back in gather's minimal path). Skip the per-frame proj_M D2H.
            (void)res_M;
        }

        // Resident device K3 (m2thr): independent of K1/scan/K2. In production
        // overlap it with scan1 (+ scan2/K2 when slow) so m2_thresh does not sit
        // on the critical path before K4. TA_TIMING keeps the serial schedule so
        // per-stage numbers stay attributable.
        const bool k3_on_device = resident_in && static_cast<bool>(res_op);
        // R1: when the cull is disabled, K3 (m2_thresh) is dead work — never
        // enqueue it (nor overlap it with the scans).
        const bool k3_pipeline = k3_on_device && !ta_timing && !ta_no_cull;
        const uint32_t m3_pages = M_pad / ELEMS_PER_PAGE;
        const WorkSplit ws_m3 = split_pages(m3_pages, num_cores);
        auto enqueue_k3_device = [&]() {
            uint32_t floor_bits = 0;
            std::memcpy(&floor_bits, &contrib_floor, 4);
            Program& progm3 = ctx->wl_m2thr.get_programs().begin()->second;
            const uint32_t op_addr = static_cast<uint32_t>(res_op->address());
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(progm3, ctx->km3, core, {
                    op_addr,
                    static_cast<uint32_t>(ctx->buf_m2thr->address()),
                    ws_m3.start[c], ws_m3.count[c], Mu, floor_bits,
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_m2thr, false);
        };
        clk::time_point k3_t0{};
        bool k3_pipelined = false;

        // ── Allocate / grow M-sized buffers ─────────────────────────────
        const std::size_t m_bytes = static_cast<std::size_t>(M_pad) * 4;
        if (!ctx->buf_tpg || ctx->cap_m_bytes < m_bytes) {
            ctx->buf_tpg    = make_dram(ctx->mesh_device.get(), m_bytes);
            ctx->buf_m2thr  = make_dram(ctx->mesh_device.get(), m_bytes);
            // Host-array input buffers: only needed when NOT reading resident.
            // In resident mode K1/K2/K4 read proj_m_* directly over NoC, so we
            // neither allocate nor H2D-upload these.
            if (!resident_in) {
                ctx->buf_px = make_dram(ctx->mesh_device.get(), m_bytes);
                ctx->buf_py = make_dram(ctx->mesh_device.get(), m_bytes);
                ctx->buf_rx = make_dram(ctx->mesh_device.get(), m_bytes);
                ctx->buf_ry = make_dram(ctx->mesh_device.get(), m_bytes);
                ctx->buf_a  = make_dram(ctx->mesh_device.get(), m_bytes);
                ctx->buf_b  = make_dram(ctx->mesh_device.get(), m_bytes);
                ctx->buf_c  = make_dram(ctx->mesh_device.get(), m_bytes);
            }
            ctx->cap_m_bytes = m_bytes;
        }
        // The M-sized DRAM buffers are grow-only: a smaller-M frame keeps the
        // larger capacity allocated by an earlier (e.g. hero) frame. Every
        // EnqueueRead/WriteMeshBuffer writes/reads the WHOLE buffer, so the host
        // vector MUST be sized to the buffer capacity, not the current M_pad —
        // otherwise tt-metal asserts "source vector too small" (and crashes the
        // 30-view sweep). Kernel work-splits still use the real M_pad so only
        // the current data is processed; the capacity tail stays zero-padded and
        // unread. (gather's readback already follows this cap-sizing pattern.)
        const uint32_t cap_m_elems = static_cast<uint32_t>(ctx->cap_m_bytes / 4);
        const uint32_t offs_count = Mu + 1;
        const uint32_t offs_pad = round_up(offs_count, ELEMS_PER_PAGE);
        const std::size_t offs_bytes = static_cast<std::size_t>(offs_pad) * 4;
        if (!ctx->buf_offs || ctx->cap_offs_bytes < offs_bytes) {
            ctx->buf_offs = make_dram(ctx->mesh_device.get(), offs_bytes);
            ctx->cap_offs_bytes = offs_bytes;
        }
        const uint32_t cap_offs_elems = static_cast<uint32_t>(ctx->cap_offs_bytes / 4);

        // ── Pack inputs (SoA px,py,rx,ry), zero-padded tail ─────────────
        // Resident mode skips this H2D entirely — K1/K2 read proj_m_* over NoC.
        if (!resident_in) {
            std::vector<uint32_t> px(cap_m_elems, 0), py(cap_m_elems, 0),
                rx(cap_m_elems, 0), ry(cap_m_elems, 0);
            for (uint32_t m = 0; m < Mu; m++) {
                std::memcpy(&px[m], &means_2d[m * 2 + 0], 4);
                std::memcpy(&py[m], &means_2d[m * 2 + 1], 4);
                std::memcpy(&rx[m], &radii[m * 2 + 0], 4);
                std::memcpy(&ry[m], &radii[m * 2 + 1], 4);
            }
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_px, px, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_py, py, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_rx, rx, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_ry, ry, false);
        }

        // Reader input addresses: resident proj_m_* (NoC) or the host arrays we
        // just uploaded. K1/K2 use means+radii (px,py,rx,ry); K4 adds cov a,b,c.
        const uint32_t in_px = static_cast<uint32_t>(
            (resident_in ? res_px : ctx->buf_px)->address());
        const uint32_t in_py = static_cast<uint32_t>(
            (resident_in ? res_py : ctx->buf_py)->address());
        const uint32_t in_rx = static_cast<uint32_t>(
            (resident_in ? res_rx : ctx->buf_rx)->address());
        const uint32_t in_ry = static_cast<uint32_t>(
            (resident_in ? res_ry : ctx->buf_ry)->address());
        const uint32_t in_a = static_cast<uint32_t>(
            (resident_in ? res_a : ctx->buf_a)->address());
        const uint32_t in_b = static_cast<uint32_t>(
            (resident_in ? res_b : ctx->buf_b)->address());
        const uint32_t in_c = static_cast<uint32_t>(
            (resident_in ? res_c : ctx->buf_c)->address());

        // ── K1: per-Gaussian AABB -> tiles_per_gaussian ─────────────────
        // Phase B (GSPLAT_TT_CHUNK_FUSION): gather scatter already wrote tpg.
        const bool chunk_fusion_k1 =
            env_config::chunk_fusion_enabled() && resident_in;
        std::shared_ptr<distributed::MeshBuffer> fused_tpg;
        if (chunk_fusion_k1) {
            fused_tpg = device_state::get_buffer("ta_tiles_per_gaussian");
        }
        const bool skip_k1 = chunk_fusion_k1 && static_cast<bool>(fused_tpg);
        const auto t_k1_0 = clk::now();
        if (skip_k1) {
            ctx->buf_tpg = fused_tpg;
        } else {
            if (chunk_fusion_k1 && !fused_tpg) {
                std::cerr << "[gsplat_tt::tile_assign] CHUNK_FUSION set but "
                             "ta_tiles_per_gaussian missing; running K1\n";
            }
            const uint32_t k1_pages = M_pad / ELEMS_PER_PAGE;
            const WorkSplit ws1 = split_pages(k1_pages, num_cores);
            Program& prog1 = ctx->wl_k1.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(prog1, ctx->k1, core, {
                    in_px,
                    in_py,
                    in_rx,
                    in_ry,
                    static_cast<uint32_t>(ctx->buf_tpg->address()),
                    ws1.start[c], ws1.count[c], Mu,
                    static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y),
                    static_cast<uint32_t>(tile_size),
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_k1, false);
        }
        // K1 -> scan chain on one in-order CQ; scan Finish drains K1 (drops k1-only lock).
        const auto t_k1_1 = clk::now();
        T.k1_ms = skip_k1 ? 0.0
                          : std::chrono::duration<double, std::milli>(t_k1_1 - t_k1_0).count();

        gsplat_cpu::TileAssignResult result;
        uint32_t P = 0;
        std::vector<uint32_t> offs;  // host-scan path only
        clk::time_point t_s2_0{};

        if (device_scan) {
            // ── On-device exclusive scan (two-phase) ────────────────────
            // Cover offs_pad pages so offs[M] (read by K2) is produced even
            // when M is a multiple of 16 (then offs lives one page past the
            // K1-written tpg range); the kernels' g0>=M guard makes those
            // padding pages = P without reading tpg out of bounds.
            const uint32_t scan_pages = offs_pad / ELEMS_PER_PAGE;
            const WorkSplit wss = split_pages(scan_pages, num_cores);

            // Phase 1: per-core reduce of tpg -> core_total partials.
            if (k3_pipeline) {
                k3_t0 = clk::now();
                enqueue_k3_device();
                k3_pipelined = true;
            }
            const auto t_s1_0 = clk::now();
            Program& progs1 = ctx->wl_scan1.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(progs1, ctx->ks1, core, {
                    static_cast<uint32_t>(ctx->buf_tpg->address()),
                    static_cast<uint32_t>(ctx->buf_core_total->address()),
                    wss.start[c], wss.count[c], Mu, c,
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan1, false);
            // scan_bases chains on scan1 output — one Finish for both (in-order CQ).
            Program& progb = ctx->wl_scan_bases.get_programs().begin()->second;
            CoreCoord core0{0, 0};
            SetRuntimeArgs(progb, ctx->ks_bases, core0, {
                static_cast<uint32_t>(ctx->buf_core_total->address()),
                static_cast<uint32_t>(ctx->buf_core_base->address()),
                static_cast<uint32_t>(ctx->buf_pairs_P->address()),
                num_cores,
            });
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan_bases, false);
            {
                GSPLAT_HOST_ZONE("host_finish_ta_scan");
                distributed::Finish(*ctx->cq);
            }
            const auto t_scan_done = clk::now();
            T.scan1_ms =
                std::chrono::duration<double, std::milli>(t_scan_done - t_s1_0).count();
            T.prefix_ms = 0.0;  // scan_bases merged into scan1 (one Finish)
            if (k3_pipelined) {
                T.k3_compute_ms =
                    std::chrono::duration<double, std::milli>(t_scan_done - k3_t0).count();
                T.k3_h2d_ms = 0.0;
            }
            T.d2h_tpg_ms = 0.0;
            T.h2d_offs_ms = 0.0;

            // Control-only read of P (64B page); pairs already resident for sort.
            const auto t_p0 = clk::now();
            std::vector<uint32_t> pbuf(ELEMS_PER_PAGE, 0);
            {
                GSPLAT_HOST_ZONE("host_ta_d2h_p");
                distributed::EnqueueReadMeshBuffer(*ctx->cq, pbuf, ctx->buf_pairs_P, true);
            }
            P = pbuf[0];
            T.d2h_tpg_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_p0).count();

            if (P == 0) {
                if (k3_pipelined) {
                    GSPLAT_HOST_ZONE("host_finish_ta_drain");
                    distributed::Finish(*ctx->cq);
                }
                if (device_ok) *device_ok = true;
                T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();
                return result;
            }

            // Phase 2: per-core exclusive prefix-add seeded by core_base -> offs.
            t_s2_0 = clk::now();
            Program& progs2 = ctx->wl_scan2.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(progs2, ctx->ks2, core, {
                    static_cast<uint32_t>(ctx->buf_tpg->address()),
                    static_cast<uint32_t>(ctx->buf_offs->address()),
                    wss.start[c], wss.count[c], Mu,
                    static_cast<uint32_t>(ctx->buf_core_base->address()),
                    c,
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan2, false);
            // When not pipelined with K3, scan2 Finish merges with K2 below.
            if (k3_pipelined) {
                T.scan2_ms = 0.0;  // attributed in the scan2+K2 barrier below
            }
            // result.tiles_per_gaussian intentionally left empty: render_full's
            // fast path never reads it (sort/blend don't need it).
        } else {
            // D2H tiles_per_gaussian (read whole grow-only buffer -> size to cap).
            const auto t_d2htpg0 = clk::now();
            std::vector<uint32_t> tpg(cap_m_elems);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tpg, ctx->buf_tpg, true);
            T.d2h_tpg_ms = std::chrono::duration<double, std::milli>(clk::now() - t_d2htpg0).count();

            // ── H1: host exclusive prefix-sum ───────────────────────────
            const auto t_pre0 = clk::now();
            offs.assign(cap_offs_elems, 0);
            uint64_t acc = 0;
            for (uint32_t m = 0; m < Mu; m++) {
                offs[m] = static_cast<uint32_t>(acc);
                acc += tpg[m];
            }
            P = static_cast<uint32_t>(acc);
            for (uint32_t m = Mu; m < cap_offs_elems; m++) offs[m] = P;  // offs[M..] = P
            T.prefix_ms = std::chrono::duration<double, std::milli>(clk::now() - t_pre0).count();

            result.tiles_per_gaussian.assign(M, 0);
            for (uint32_t m = 0; m < Mu; m++)
                result.tiles_per_gaussian[m] = static_cast<int64_t>(tpg[m]);

            if (ta_stats_enabled() && P > 0) {
                ta_dump_k2_stats(offs, Mu, P, round_up(P, ELEMS_PER_PAGE) / ELEMS_PER_PAGE,
                                 num_cores);
            }

            if (P == 0) {
                if (device_ok) *device_ok = true;
                T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();
                return result;
            }

            const auto t_h2doffs0 = clk::now();
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_offs, offs, false);
            if (ta_timing) {
                GSPLAT_HOST_ZONE("host_finish_ta_h2d_offs");
                distributed::Finish(*ctx->cq);
            }
            T.h2d_offs_ms = std::chrono::duration<double, std::milli>(clk::now() - t_h2doffs0).count();
        }

        // ── Allocate / grow pair buffers ────────────────────────────────
        const uint32_t P_pad = round_up(P, ELEMS_PER_PAGE);
        const std::size_t p_bytes = static_cast<std::size_t>(P_pad) * 4;
        if (!ctx->buf_gids || ctx->cap_p_bytes < p_bytes) {
            ctx->buf_gids = make_dram(ctx->mesh_device.get(), p_bytes);
            ctx->buf_tids = make_dram(ctx->mesh_device.get(), p_bytes);
            ctx->buf_keep = make_dram(ctx->mesh_device.get(), p_bytes);
            ctx->cap_p_bytes = p_bytes;
            ctx->buf_keep_all_ones = false;  // fresh DRAM: needs the all-ones fill
        }
        const uint32_t cap_p_elems = static_cast<uint32_t>(ctx->cap_p_bytes / 4);

        // ── K2: pair-centric scatter ────────────────────────────────────
        const auto t_k2_0 = clk::now();
        const uint32_t k2_pages = P_pad / ELEMS_PER_PAGE;
        const WorkSplit ws2 = split_pages(k2_pages, num_cores);
        Program& prog2 = ctx->wl_k2.get_programs().begin()->second;
        for (uint32_t c = 0; c < num_cores; c++) {
            CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
            SetRuntimeArgs(prog2, ctx->k2, core, {
                static_cast<uint32_t>(ctx->buf_offs->address()),
                in_px,
                in_py,
                in_rx,
                in_ry,
                static_cast<uint32_t>(ctx->buf_gids->address()),
                static_cast<uint32_t>(ctx->buf_tids->address()),
                ws2.start[c], ws2.count[c], P, Mu,
                static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y),
                static_cast<uint32_t>(tile_size),
            });
        }
        distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_k2, false);
        if (k3_pipelined) {
            // scan2 + K2 share one barrier with any in-flight K3 (started before scan1).
            GSPLAT_HOST_ZONE("host_finish_ta_k2");
            distributed::Finish(*ctx->cq);
            const auto t_barrier = clk::now();
            if (device_scan) {
                T.scan2_ms =
                    std::chrono::duration<double, std::milli>(t_barrier - t_s2_0).count();
            }
            T.k2_ms = std::chrono::duration<double, std::milli>(t_barrier - t_k2_0).count();
            if (T.k3_compute_ms == 0.0) {
                T.k3_compute_ms =
                    std::chrono::duration<double, std::milli>(t_barrier - k3_t0).count();
            }
        } else {
            // scan2 + K2 on one in-order CQ — single Finish (drops scan2-only drain).
            GSPLAT_HOST_ZONE("host_finish_ta_k2");
            distributed::Finish(*ctx->cq);
            const auto t_k2_1 = clk::now();
            T.k2_ms = std::chrono::duration<double, std::milli>(t_k2_1 - t_k2_0).count();
            if (device_scan) {
                T.scan2_ms =
                    std::chrono::duration<double, std::milli>(t_k2_1 - t_s2_0).count();
            }
        }

        // The per-pair Mahalanobis cull needs the per-Gaussian cov (a,b,c) and
        // opacities. In resident mode the cov comes from the resident
        // proj_m_a/b/c (read over NoC by K4), so the host covs_2d pointer is
        // intentionally absent — gate the cull on the resident cov buffers
        // (already validated present above) + the host opacities the m2_thresh
        // precompute still consumes. (When the caller disables the cull it
        // passes opacities==nullptr in both modes.) Without this, an M-only
        // ProjectResult (empty host covs_2d) would silently drop the cull and
        // the resident-pairs publish, breaking the resident sort/blend handoff.
        const bool do_cull = resident_in
            ? static_cast<bool>(res_op)
            : ((covs_2d != nullptr) && (opacities != nullptr));
        // Resident-pairs is only valid when the cull runs (keep mask is the
        // implicit compaction). Otherwise fall through to the host path.
        const bool resident_pairs_active = resident_pairs && do_cull;

        // ── D2H pairs ───────────────────────────────────────────────────
        // Skipped in resident-pairs mode: the (gid,tid) pairs stay resident and
        // sort's device binning consumes them directly.
        if (!resident_pairs_active) {
            const auto t_d2h0 = clk::now();
            std::vector<uint32_t> gids(cap_p_elems), tids(cap_p_elems);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, gids, ctx->buf_gids, true);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tids, ctx->buf_tids, true);
            const auto t_d2h1 = clk::now();
            T.d2h_ms = std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count();

            result.gaussian_ids.resize(P);
            result.tile_ids.resize(P);
            for (uint32_t p = 0; p < P; p++) {
                result.gaussian_ids[p] = static_cast<int64_t>(static_cast<int32_t>(gids[p]));
                result.tile_ids[p] = static_cast<int64_t>(static_cast<int32_t>(tids[p]));
            }
        }

        // ── Phase 4: per-pair Mahalanobis cull (K3 host precompute + K4) ──
        if (do_cull) {
            const auto t_cull0 = clk::now();
          if (ta_no_cull) {
            // R1: skip K3 + K4. Publish an all-ones keep so sort keeps every
            // AABB pair (P_kept == P); blend's microblock cull rejects the
            // empty-corner pairs downstream. sort_bin treats keep==0 as a drop
            // while a MISSING ta_pairs_keep buffer forces a host fallback — so
            // the mask MUST be present and all-1s for [0, P).
            //
            // iter-35: the all-ones mask is CONSTANT, so fill buf_keep ONCE per
            // (re)allocation instead of re-uploading ~13.5 MB of 1s every frame.
            // DRAM persists across frames and nothing overwrites keep while the
            // cull is off, so once buf_keep_all_ones is set the per-frame H2D +
            // Finish() are pure dead host work on the critical path. We fill to
            // the full buffer CAPACITY, which always covers the current P_pad
            // (the buffer only grows), so a later smaller-P frame is still
            // all-1s over its [0, P). Bit-identical: same bytes sort reads, just
            // written on the alloc frame instead of every frame.
            const auto t_keep0 = clk::now();
            if (!ctx->buf_keep_all_ones) {
                std::vector<uint32_t> keep_ones(cap_p_elems, 1u);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_keep, keep_ones, false);
                GSPLAT_HOST_ZONE("host_finish_ta_keep_fill");
                distributed::Finish(*ctx->cq);
                ctx->buf_keep_all_ones = true;
            }
            T.k3_compute_ms = 0.0;
            T.k3_h2d_ms = 0.0;
            T.k3_ms = 0.0;
            T.k4_ms = std::chrono::duration<double, std::milli>(clk::now() - t_keep0).count();
            T.cull_ms = std::chrono::duration<double, std::milli>(clk::now() - t_cull0).count();
          } else {
            std::vector<uint32_t> a_v, b_v, c_v;
            const auto t_k3c0 = clk::now();
            if (resident_in && res_op) {
                if (!k3_pipelined) {
                    // K3 on device: m2_thresh from resident proj_m_opacity (no host
                    // opacities[] loop, no m2thr H2D).
                    enqueue_k3_device();
                    GSPLAT_HOST_ZONE("host_finish_ta_k3");
                    distributed::Finish(*ctx->cq);
                    T.k3_compute_ms =
                        std::chrono::duration<double, std::milli>(clk::now() - t_k3c0).count();
                    T.k3_h2d_ms = 0.0;
                }
            } else {
                // K3 host path (non-resident TA inputs only).
                if (!resident_in) {
                    a_v.assign(cap_m_elems, 0);
                    b_v.assign(cap_m_elems, 0);
                    c_v.assign(cap_m_elems, 0);
                }
                std::vector<uint32_t> m2t_v(cap_m_elems, 0);
                auto k3_range = [&](uint32_t lo, uint32_t hi) {
                    for (uint32_t m = lo; m < hi; m++) {
                        if (!resident_in) {
                            const float a = covs_2d[m * 4 + 0];
                            const float b = covs_2d[m * 4 + 1];
                            const float c = covs_2d[m * 4 + 3];
                            std::memcpy(&a_v[m], &a, 4);
                            std::memcpy(&b_v[m], &b, 4);
                            std::memcpy(&c_v[m], &c, 4);
                        }
                        const float op = opacities[m];
                        float m2t = -1.0f;
                        if (op > contrib_floor) {
                            m2t = -2.0f * std::log(contrib_floor / op);
                        }
                        std::memcpy(&m2t_v[m], &m2t, 4);
                    }
                };
                auto& pool = k3_pool();
                const uint32_t W =
                    std::max<uint32_t>(1, static_cast<uint32_t>(pool.size()));
                const uint32_t chunk_pages = (M_pad / ELEMS_PER_PAGE + W - 1) / W;
                const uint32_t chunk = chunk_pages * ELEMS_PER_PAGE;
                if (W <= 1 || Mu <= ELEMS_PER_PAGE) {
                    k3_range(0, Mu);
                } else {
                    for (uint32_t w = 0; w < W; w++) {
                        const uint32_t lo = w * chunk;
                        if (lo >= Mu) break;
                        const uint32_t hi = std::min(lo + chunk, Mu);
                        pool.submit([k3_range, lo, hi]() { k3_range(lo, hi); });
                    }
                    pool.wait();
                }
                T.k3_compute_ms =
                    std::chrono::duration<double, std::milli>(clk::now() - t_k3c0).count();
                const auto t_k3h0 = clk::now();
                if (!resident_in) {
                    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_a, a_v, false);
                    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_b, b_v, false);
                    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c, c_v, false);
                }
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_m2thr, m2t_v, false);
                if (ta_timing) {
                    GSPLAT_HOST_ZONE("host_finish_ta_k3_h2d");
                    distributed::Finish(*ctx->cq);
                }
                T.k3_h2d_ms =
                    std::chrono::duration<double, std::milli>(clk::now() - t_k3h0).count();
            }
            const auto t_k3_1 = clk::now();
            T.k3_ms = std::chrono::duration<double, std::milli>(t_k3_1 - t_cull0).count();

            // K4: per-pair cull -> keep_mask. K4 overwrites buf_keep with
            // per-pair 0/1 values, so the cached "all-ones" invariant no longer
            // holds (defensive: production is cull-off and never reaches here).
            ctx->buf_keep_all_ones = false;
            Program& progc = ctx->wl_cull.get_programs().begin()->second;
            for (uint32_t cc = 0; cc < num_cores; cc++) {
                CoreCoord core{cc % ctx->grid.x, cc / ctx->grid.x};
                SetRuntimeArgs(progc, ctx->k4, core, {
                    static_cast<uint32_t>(ctx->buf_gids->address()),
                    static_cast<uint32_t>(ctx->buf_tids->address()),
                    in_a,
                    in_b,
                    in_c,
                    in_px,
                    in_py,
                    static_cast<uint32_t>(ctx->buf_m2thr->address()),
                    static_cast<uint32_t>(ctx->buf_keep->address()),
                    ws2.start[cc], ws2.count[cc], P,
                    static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tile_size),
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_cull, false);
            GSPLAT_HOST_ZONE("host_finish_ta_k4");
            distributed::Finish(*ctx->cq);
            const auto t_cull1 = clk::now();
            T.k4_ms = std::chrono::duration<double, std::milli>(t_cull1 - t_k3_1).count();
            T.cull_ms = std::chrono::duration<double, std::milli>(t_cull1 - t_cull0).count();
          }  // end !ta_no_cull (K3 + K4)

            if (resident_pairs_active) {
                const auto t_pub0 = clk::now();
                // R4/R5: publish the full-P resident pairs + keep mask for the
                // device-binning sort path. No keep D2H, no host compaction —
                // sort reads keep[] and compacts implicitly while binning.
                device_state::register_buffer("ta_pairs_gid", ctx->buf_gids);
                device_state::register_buffer("ta_pairs_tid", ctx->buf_tids);
                device_state::register_buffer("ta_pairs_keep", ctx->buf_keep);
                if (!device_scan) {
                    // Host-scan path: publish P via H2D (device_scan writes P on-device).
                    std::vector<uint32_t> pbuf(ELEMS_PER_PAGE, 0);
                    pbuf[0] = P;
                    pbuf[1] = P_pad;
                    distributed::EnqueueWriteMeshBuffer(
                        *ctx->cq, ctx->buf_pairs_P, pbuf, false);
                }
                // Sort's next stage Finish() drains the CQ; no extra stage-lock here.
                T.publish_ms = std::chrono::duration<double, std::milli>(clk::now() - t_pub0).count();
                // result.gaussian_ids / tile_ids intentionally left empty: the
                // resident-pairs sort path does not read them.
            } else {
                // D2H keep_mask.
                const auto t_kd0 = clk::now();
                std::vector<uint32_t> keep(cap_p_elems);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, keep, ctx->buf_keep, true);
                const auto t_kd1 = clk::now();
                T.d2h_ms += std::chrono::duration<double, std::milli>(t_kd1 - t_kd0).count();

                // H2 (host bridge): sequential gaussian-major compaction.
                // Identical ordering to gsplat_cpu::tile_assign's chunked
                // compaction (which tiles [0,P) in order, preserving order).
                const auto t_comp0 = clk::now();
                std::vector<int64_t> kg, kt;
                kg.reserve(P);
                kt.reserve(P);
                for (uint32_t p = 0; p < P; p++) {
                    if (keep[p]) {
                        kg.push_back(result.gaussian_ids[p]);
                        kt.push_back(result.tile_ids[p]);
                    }
                }
                result.gaussian_ids = std::move(kg);
                result.tile_ids = std::move(kt);
                const auto t_comp1 = clk::now();
                T.compact_ms = std::chrono::duration<double, std::milli>(t_comp1 - t_comp0).count();
            }
        }

        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();

        // GSPLAT_TT_TA_TIMING: per-call host/device sub-stage breakdown. Works
        // in every mode (incl. resident-pairs, where TA_DEBUG is skipped).
        if (ta_timing) {
            std::fprintf(stderr,
                "[TA] M=%u P=%u resident_in=%d resident_pairs=%d cull=%d "
                "dev_scan=%d | k1=%.2f scan1=%.2f d2h_tpg=%.2f prefix=%.2f "
                "scan2=%.2f h2d_offs=%.2f k2=%.2f k3=%.2f(c=%.2f h2d=%.2f) "
                "k4=%.2f publish=%.2f compact=%.2f d2h=%.2f total=%.2fms\n",
                Mu, P, (int)resident_in, (int)resident_pairs_active, (int)do_cull,
                (int)device_scan, T.k1_ms, T.scan1_ms, T.d2h_tpg_ms, T.prefix_ms,
                T.scan2_ms, T.h2d_offs_ms, T.k2_ms, T.k3_ms, T.k3_compute_ms,
                T.k3_h2d_ms, T.k4_ms, T.publish_ms, T.compact_ms, T.d2h_ms,
                T.total_ms);
        }

        // Optional self-check vs the matching CPU reference (AABB-only when
        // cull disabled, full Phase-4 cull otherwise). Skipped in resident-pairs
        // mode (host pairs intentionally absent — correctness is checked
        // end-to-end via the sort verify / PSNR).
        if (const char* dbg = std::getenv("GSPLAT_TT_TA_DEBUG");
            dbg && dbg[0] == '1' && !resident_pairs_active) {
            const gsplat_cpu::TileAssignResult cpu = gsplat_cpu::tile_assign(
                means_2d, radii, M, image_height, image_width, tile_size,
                covs_2d, opacities, contrib_floor,
                /*pool=*/nullptr, /*recompute=*/false);
            const std::size_t cpu_Pp = cpu.gaussian_ids.size();
            const std::size_t dev_Pp = result.gaussian_ids.size();
            bool set_match = (cpu_Pp == dev_Pp);
            std::size_t mism = 0;
            if (set_match) {
                for (std::size_t p = 0; p < cpu_Pp; p++) {
                    if (cpu.gaussian_ids[p] != result.gaussian_ids[p] ||
                        cpu.tile_ids[p] != result.tile_ids[p]) {
                        if (mism < 5) {
                            std::fprintf(stderr,
                                "[TA mismatch] p=%zu cpu(g=%lld,t=%lld) dev(g=%lld,t=%lld)\n",
                                p, (long long)cpu.gaussian_ids[p], (long long)cpu.tile_ids[p],
                                (long long)result.gaussian_ids[p], (long long)result.tile_ids[p]);
                        }
                        mism++;
                    }
                }
            }
            std::fprintf(stderr,
                "[TA assert] cull=%d AABB_P=%u device_Pprime=%zu cpu_Pprime=%zu "
                "count_match=%d pair_mismatches=%zu k1=%.2f k2=%.2f cull=%.2f "
                "compact=%.2f d2h=%.2f total=%.2fms\n",
                (int)do_cull, P, dev_Pp, cpu_Pp, (int)set_match, mism,
                T.k1_ms, T.k2_ms, T.cull_ms, T.compact_ms, T.d2h_ms, T.total_ms);
        }

        if (device_ok) *device_ok = true;
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::tile_assign] call failed: " << e.what() << "\n";
        return set_fail();
    }
}

}  // namespace gsplat_tt
