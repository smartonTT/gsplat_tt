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
#include "stage_timers.h"
#include "gather_visible.h"
#include "vis_mode.h"
#include "sort_mover_speed.h"
#include "sort.h"
#include "../kernels/dataflow/pfwc_fuse.h"
#include "../kernels/dataflow/vis_tile.h"
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
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_event.hpp>
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
    // Dual data mover (task #34): BRISC instances of K1 / K2 (valid iff dual).
    bool dual = false;
    KernelHandle k1b{};
    KernelHandle k2b{};
    // Lever 2 (task #99): K2 built with TA_K2_AABB (offs + packed rectangle
    // from the gather, no K1 / scans). Built on first use.
    bool k2v_built = false;
    distributed::MeshWorkload wl_k2v;
    KernelHandle k2v{};
    KernelHandle k2vb{};
    // Lever B (task #125): segment K2 (built on first use for k2seg_nseg
    // segments) and the hand-off from tile_assign_fused_k2 to tile_assign_tt.
    uint32_t k2seg_nseg = 0;
    distributed::MeshWorkload wl_k2seg;
    KernelHandle k2s{};
    KernelHandle k2sb{};
    bool fused_ready = false;
    uint32_t fused_P = 0;
    // Task #170 K2 fold: per-mover per-tile count rows (2 per core).
    std::shared_ptr<distributed::MeshBuffer> buf_k2_rows;
    std::size_t cap_k2_rows_bytes = 0;

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

// Dual data mover (task #34). K1 (ta_gauss_aabb) and K2 (ta_bucket_scatter)
// run on NCRISC and BRISC: BRISC takes the first TA_SPLIT/1000 of each core's
// page range, NCRISC the rest. Both kernels map every output page from its own
// inputs only, so any split is byte-identical. BRISC uses private CBs at
// id + TA_MOVER0_CB_OFFSET. GSPLAT_TT_TA_MOVERS=1 keeps NCRISC alone (A/B);
// GSPLAT_TT_TA_SPLIT=<permille> sets BRISC's share (default 500). Read once.
constexpr uint32_t TA_MOVER0_CB_OFFSET = 16;

static bool ta_dual_mover() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_TA_MOVERS");
        return !(e != nullptr && std::atoi(e) == 1);
    }();
    return v;
}
static uint32_t ta_split_permille() {
    static const uint32_t v = [] {
        const char* e = std::getenv("GSPLAT_TT_TA_SPLIT");
        const int x = (e != nullptr) ? std::atoi(e) : 500;
        return static_cast<uint32_t>(std::clamp(x, 0, 1000));
    }();
    return v;
}

// Task #170 (pair stage diet). GSPLAT_TT_K2_DIET=0 is the kill switch of the
// segment K2's diet loop (tile_assign_scatter_seg.cpp K2_DIET, default on).
// GSPLAT_TT_K2_FOLD=0 keeps the diet but not its per-tile count rows, so the
// one-launch sort runs its own count pass again. Read once.
constexpr uint32_t K2_FOLD_TILES = 1024;  // == tile_assign_scatter_seg.cpp
static bool k2_diet_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_K2_DIET");
        return !(e != nullptr && std::atoi(e) == 0);
    }();
    return v;
}
static bool k2_fold_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_K2_FOLD");
        return k2_diet_enabled() && !(e != nullptr && std::atoi(e) == 0);
    }();
    return v;
}

// BRISC (mover 0) kernel instance of a single-mover RISCV_1 kernel.
static KernelHandle create_mover0_kernel(
    Program& program, const char* path, const CoreRangeSet& cores,
    const std::vector<uint32_t>& ct) {
    return CreateKernel(
        program, path, cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
            .defines = {{"TA_CB_OFFSET", std::to_string(TA_MOVER0_CB_OFFSET)}},
        });
}

// iter-137: ta_gauss_aabb (tile_assign_bbox.cpp) batches MULTIBUF_PAGES pages of
// input reads before each NoC read barrier to overlap DRAM read latency (the
// kernel is NoC-READ-bound). Its scratch CBs 0..4 (px,py,rx,ry,out) must hold
// MULTIBUF_PAGES 64B pages each instead of one. Keep in sync with the
// MULTIBUF_PAGES constant in render/kernels/dataflow/tile_assign_bbox.cpp.
constexpr uint32_t K1_MULTIBUF_PAGES = 8;

static void build_program_k1(TileAssignDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        const uint32_t bytes = K1_MULTIBUF_PAGES * PAGE_BYTES;
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    for (uint32_t id = 0; id < 5; id++) scratch_cb(id);  // px,py,rx,ry,out (MULTIBUF_PAGES-deep)
    if (ctx.dual) {
        for (uint32_t id = 0; id < 5; id++) scratch_cb(TA_MOVER0_CB_OFFSET + id);
    }

    std::vector<uint32_t> ct;
    // 5 input/output accessors (px,py,rx,ry,tpg). proj_M is read in-kernel via a
    // runtime InterleavedAddrGen (no extra accessor / CT arg), S5.3.
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
    if (ctx.dual) {
        ctx.k1b = create_mover0_kernel(
            program, OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_bbox.cpp",
            cores, ct);
    }
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_k1.add_program(device_range, std::move(program));
}

static void build_program_k2(TileAssignDeviceContext& ctx, bool aabb = false) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto scratch_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    for (uint32_t id = 0; id < 7; id++) scratch_cb(id);  // offs,px,py,rx,ry,gid,tid
    if (ctx.dual) {
        for (uint32_t id = 0; id < 7; id++) scratch_cb(TA_MOVER0_CB_OFFSET + id);
    }

    std::vector<uint32_t> ct;
    for (int i = 0; i < 7; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    std::map<std::string, std::string> defines;
    if (aabb) defines["TA_K2_AABB"] = "1";
    const KernelHandle k = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scatter.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
            .defines = defines,
        });
    KernelHandle kb{};
    if (ctx.dual) {
        std::map<std::string, std::string> d0 = defines;
        d0["TA_CB_OFFSET"] = std::to_string(TA_MOVER0_CB_OFFSET);
        kb = CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scatter.cpp", cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = ct,
                .defines = d0,
            });
    }
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    if (aabb) {
        ctx.k2v = k;
        ctx.k2vb = kb;
        ctx.wl_k2v.add_program(device_range, std::move(program));
        ctx.k2v_built = true;
    } else {
        ctx.k2 = k;
        ctx.k2b = kb;
        ctx.wl_k2.add_program(device_range, std::move(program));
    }
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
    // tpg + total accessors. proj_M read in-kernel via InterleavedAddrGen (S5.3).
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
    // tpg + offs + base accessors. proj_M read in-kernel via InterleavedAddrGen (S5.3).
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

// Pair buffers (gids / tids / keep), grown to at least p_bytes. Size to at
// least the static pair ceiling (task #85). A view with a larger P than any
// before it used to regrow these buffers, and each regrowth refilled the
// all-ones keep mask (~19 MB H2D + Finish, 9-12 ms). On the bicycle orbit that
// hit 4 of 30 views (1.3 ms/view). A P above the ceiling still grows the buffers.
static void ensure_pair_buffers(TileAssignDeviceContext& ctx, std::size_t p_bytes) {
    if (ctx.buf_gids && ctx.cap_p_bytes >= p_bytes) return;
    const std::size_t ceil_bytes =
        static_cast<std::size_t>(round_up(env_config::pair_ceiling(), ELEMS_PER_PAGE)) * 4;
    const std::size_t alloc_bytes = std::max(p_bytes, ceil_bytes);
    ctx.buf_gids = make_dram(ctx.mesh_device.get(), alloc_bytes);
    ctx.buf_tids = make_dram(ctx.mesh_device.get(), alloc_bytes);
    ctx.buf_keep = make_dram(ctx.mesh_device.get(), alloc_bytes);
    ctx.cap_p_bytes = alloc_bytes;
    ctx.buf_keep_all_ones = false;  // fresh DRAM: needs the all-ones fill
}

// Lever B (task #125): K2 over the fused pfwc writer's segments
// (tile_assign_scatter_seg.cpp), NCRISC + BRISC like build_program_k2. One raw
// scratch CB per mover: the nseg-page segment table + 5 pages (+ alignment);
// K2_DIET (task #170) adds the read-ahead ring, the pair staging and a count
// row.
static void build_program_k2seg(TileAssignDeviceContext& ctx, uint32_t nseg) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    const bool diet = k2_diet_enabled();
    const uint32_t diet_pages =
        diet ? 2u * pfwc_fuse::RA_SLOTS + 2u * pfwc_fuse::OUT_SLOTS + K2_FOLD_TILES / ELEMS_PER_PAGE
             : 0u;
    const uint32_t cb_bytes = (nseg + 6 + diet_pages) * PAGE_BYTES;
    auto raw_cb = [&](uint32_t id) {
        CircularBufferConfig c(cb_bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, cb_bytes);
        CreateCircularBuffer(program, cores, c);
    };
    raw_cb(0);
    if (ctx.dual) raw_cb(TA_MOVER0_CB_OFFSET);
    std::vector<uint32_t> ct;
    for (int i = 0; i < 8; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    std::map<std::string, std::string> defines;
    if (diet) defines["K2_DIET"] = "1";
    // Task #274: GSPLAT_TT_K2_PROF=1 (profiling only) records the K2 movers'
    // per-part cycle totals as Tracy "k2p_*" markers. Unset: no define.
    if (const char* e = std::getenv("GSPLAT_TT_K2_PROF"); e != nullptr && std::atoi(e) != 0)
        defines["K2_PROF"] = "1";
    ctx.k2s = CreateKernel(program,
                           OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scatter_seg.cpp",
                           cores,
                           DataMovementConfig{
                               .processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = ct,
                               .defines = defines,
                           });
    if (ctx.dual) {
        std::map<std::string, std::string> defines0 = defines;
        defines0["TA_CB_OFFSET"] = std::to_string(TA_MOVER0_CB_OFFSET);
        ctx.k2sb = CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "kernels/dataflow/tile_assign_scatter_seg.cpp", cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = ct,
                .defines = defines0,
            });
    }
    ctx.wl_k2seg = distributed::MeshWorkload();
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_k2seg.add_program(device_range, std::move(program));
    ctx.k2seg_nseg = nseg;
}

static TileAssignDeviceContext init_context() {
    TileAssignDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    ctx.dual = ta_dual_mover();
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

// Pages of a core's range given to BRISC (mover 0) under the dual-mover split.
static uint32_t mover0_pages(uint32_t count) {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(count) * ta_split_permille()) / 1000u);
}

// GSPLAT_TT_SFPU_VIS=2 cross-check (task #99): legacy K1 + scan1 + scan_bases +
// scan2 into ctx.buf_tpg / ctx.buf_offs, then compare with the gather's offs
// and check every packed rectangle against the K1 formula on the host. Note
// scan_bases also rewrites ta_pairs_P (same values when the paths agree).
static void vis_check_legacy(
    TileAssignDeviceContext& ctx, uint32_t Mu, uint32_t n_ceil, uint32_t offs_pad,
    uint32_t num_cores, uint32_t in_px, uint32_t in_py, uint32_t in_rx, uint32_t in_ry,
    int tiles_x, int tiles_y, int tile_size,
    const std::shared_ptr<distributed::MeshBuffer>& vis_offs,
    const std::shared_ptr<distributed::MeshBuffer>& vis_aabb,
    const std::shared_ptr<distributed::MeshBuffer>& res_px,
    const std::shared_ptr<distributed::MeshBuffer>& res_py,
    const std::shared_ptr<distributed::MeshBuffer>& res_rx,
    const std::shared_ptr<distributed::MeshBuffer>& res_ry) {
    const WorkSplit ws1 = split_pages(n_ceil / ELEMS_PER_PAGE, num_cores);
    Program& prog1 = ctx.wl_k1.get_programs().begin()->second;
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord core{c % ctx.grid.x, c / ctx.grid.x};
        const uint32_t n0 = ctx.dual ? mover0_pages(ws1.count[c]) : 0u;
        auto args = [&](uint32_t start, uint32_t count) -> std::vector<uint32_t> {
            return {in_px, in_py, in_rx, in_ry, static_cast<uint32_t>(ctx.buf_tpg->address()),
                    start, count, Mu, static_cast<uint32_t>(tiles_x),
                    static_cast<uint32_t>(tiles_y), static_cast<uint32_t>(tile_size), 0u};
        };
        SetRuntimeArgs(prog1, ctx.k1, core, args(ws1.start[c] + n0, ws1.count[c] - n0));
        if (ctx.dual) SetRuntimeArgs(prog1, ctx.k1b, core, args(ws1.start[c], n0));
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.wl_k1, false);
    const WorkSplit wss = split_pages(offs_pad / ELEMS_PER_PAGE, num_cores);
    Program& progs1 = ctx.wl_scan1.get_programs().begin()->second;
    Program& progs2 = ctx.wl_scan2.get_programs().begin()->second;
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord core{c % ctx.grid.x, c / ctx.grid.x};
        SetRuntimeArgs(progs1, ctx.ks1, core, {
            static_cast<uint32_t>(ctx.buf_tpg->address()),
            static_cast<uint32_t>(ctx.buf_core_total->address()),
            wss.start[c], wss.count[c], Mu, c, 0u});
        SetRuntimeArgs(progs2, ctx.ks2, core, {
            static_cast<uint32_t>(ctx.buf_tpg->address()),
            static_cast<uint32_t>(ctx.buf_offs->address()),
            wss.start[c], wss.count[c], Mu,
            static_cast<uint32_t>(ctx.buf_core_base->address()), c, 0u});
    }
    Program& progb = ctx.wl_scan_bases.get_programs().begin()->second;
    SetRuntimeArgs(progb, ctx.ks_bases, CoreCoord{0, 0}, {
        static_cast<uint32_t>(ctx.buf_core_total->address()),
        static_cast<uint32_t>(ctx.buf_core_base->address()),
        static_cast<uint32_t>(ctx.buf_pairs_P->address()), num_cores, 0u});
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.wl_scan1, false);
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.wl_scan_bases, false);
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.wl_scan2, false);
    distributed::Finish(*ctx.cq);

    auto rd = [&](std::shared_ptr<distributed::MeshBuffer> b) {
        std::vector<uint32_t> v(b->size() / 4);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, v, b, true);
        return v;
    };
    const std::vector<uint32_t> lo = rd(ctx.buf_offs), no = rd(vis_offs), box = rd(vis_aabb);
    const std::vector<uint32_t> px = rd(res_px), py = rd(res_py), rx = rd(res_rx),
                                ry = rd(res_ry);
    uint64_t bad_offs = 0, bad_box = 0;
    long first_offs = -1, first_box = -1;
    for (uint32_t g = 0; g <= Mu; g++)
        if (lo[g] != no[g]) {
            if (first_offs < 0) first_offs = static_cast<long>(g);
            bad_offs++;
        }
    vis_tile::Params p;
    p.tiles_x = static_cast<uint32_t>(tiles_x);
    p.tiles_y = static_cast<uint32_t>(tiles_y);
    p.tile_shift = dm_fp32::pow2_shift(static_cast<uint32_t>(tile_size));
    p.inv_tile = 1.0f / static_cast<float>(tile_size);
    for (uint32_t g = 0; g < Mu; g++) {
        using dm_fp32::SIGN;
        const int tx1 = tiles_x - 1, ty1 = tiles_y - 1;
        const int x0 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(px[g], rx[g] ^ SIGN, p.tile_shift, p.inv_tile), 0, tx1);
        const int x1 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(px[g], rx[g], p.tile_shift, p.inv_tile), 0, tx1);
        const int y0 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(py[g], ry[g] ^ SIGN, p.tile_shift, p.inv_tile), 0, ty1);
        const uint32_t want = vis_tile::aabb_pack(static_cast<uint32_t>(x0), static_cast<uint32_t>(y0),
                                                  static_cast<uint32_t>(x1 - x0 + 1));
        if (box[g] != want) {
            if (first_box < 0) first_box = static_cast<long>(g);
            bad_box++;
        }
    }
    std::fprintf(stderr,
                 "[VIS-CHECK] tile_assign M=%u offs_mismatches=%llu first=%ld P_legacy=%u "
                 "P=%u aabb_mismatches=%llu first=%ld %s\n",
                 Mu, static_cast<unsigned long long>(bad_offs), first_offs, lo[Mu], no[Mu],
                 static_cast<unsigned long long>(bad_box), first_box,
                 (bad_offs == 0 && bad_box == 0) ? "OK" : "MISMATCH");
}

}  // namespace

bool tile_assign_fused_k2(uint32_t nseg, uint32_t num_tiles, uint32_t tiles_x,
                          uint32_t screen_tiles, uint32_t* M, uint32_t* P) {
    auto* ctx = ensure_context();
    if (ctx == nullptr) return false;
    ctx->fused_ready = false;
    device_state::clear_k2_count_rows("fused_k2");
    try {
        if (nseg == 0 || nseg > pfwc_fuse::MAX_SEG)
            throw std::runtime_error("segment count " + std::to_string(nseg) + " out of range");
        auto need = [](const char* name) {
            auto b = device_state::get_buffer(name);
            if (!b) throw std::runtime_error(std::string("missing resident buffer ") + name);
            return b;
        };
        auto offs = need("proj_m_offs");
        auto aabb = need("proj_m_aabb");
        auto cnt = need("pfwc_fuse_counts");
        auto projM = need("proj_M");
        if (ctx->k2seg_nseg != nseg) build_program_k2seg(*ctx, nseg);
        ensure_pair_buffers(*ctx, 0);
        const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
        const uint32_t permille = ta_split_permille();
        // Task #170 fold: count rows for the one-launch sort (both movers, one
        // local-memory row per mover). num_tiles counts gaussian tiles (the
        // segment table); the rows are per screen tile.
        const bool fold = k2_fold_enabled() && ctx->dual && screen_tiles != 0 &&
                          screen_tiles <= K2_FOLD_TILES;
        const uint32_t row_pages =
            fold ? (screen_tiles + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE : 0u;
        // Under the sort's speed-proportional split (task #174) each mover
        // takes that range: running speed sums acc[2c + mover], acc[... + 1].
        std::vector<uint64_t> acc;
        if (fold && gsplat_tt::sort_split::ol_mover_speed_enabled()) {
            std::vector<uint32_t> noc_xy(num_cores);
            for (uint32_t c = 0; c < num_cores; c++) {
                const CoreCoord v = ctx->mesh_device->worker_core_from_logical_core(
                    CoreCoord{c % ctx->grid.x, c / ctx->grid.x});
                noc_xy[c] = static_cast<uint32_t>(v.x) | (static_cast<uint32_t>(v.y) << 16);
            }
            const std::vector<uint32_t> speed = gsplat_tt::sort_split::mover_speeds(noc_xy);
            acc.assign(speed.size() + 1u, 0u);
            for (std::size_t k = 0; k < speed.size(); k++) acc[k + 1] = acc[k] + speed[k];
        }
        {
            static int logged = -1;
            if (logged != static_cast<int>(fold)) {
                logged = static_cast<int>(fold);
                std::fprintf(stderr,
                             "[TA] K2 fold=%d (enabled %d dual %d screen tiles %u speed split %d)\n",
                             static_cast<int>(fold), static_cast<int>(k2_fold_enabled()),
                             static_cast<int>(ctx->dual), screen_tiles,
                             static_cast<int>(!acc.empty()));
            }
        }
        if (fold) {
            const std::size_t rows_bytes =
                static_cast<std::size_t>(num_cores) * 2u * row_pages * PAGE_BYTES;
            if (!ctx->buf_k2_rows || ctx->cap_k2_rows_bytes < rows_bytes) {
                ctx->buf_k2_rows = make_dram(ctx->mesh_device.get(), rows_bytes);
                ctx->cap_k2_rows_bytes = rows_bytes;
            }
        }
        const uint32_t rows_addr = fold ? static_cast<uint32_t>(ctx->buf_k2_rows->address()) : 0u;
        // Task #198: the one-launch sort goes right behind the K2 (it reads P on
        // the device); with GSPLAT_TT_MAT_CQ1, CQ1 waits for the K2 and reads
        // proj_M, else proj_M is read without draining CQ0.
        const bool want_early = env_config::sort_ol_early() && fold && !acc.empty();
        distributed::MeshCommandQueue* cq1 =
            want_early && env_config::mat_cq1() ? device_state::command_queue1() : nullptr;
        bool early = false;
        uint32_t p_cap = 0;
        std::vector<uint32_t> mread(ELEMS_PER_PAGE, 0);
        // Normally one pass; a P over the pair capacity publishes overflow,
        // then the buffers grow and the K2 reruns (the segments are intact).
        for (int pass = 0; pass < 2; pass++) {
            // Host-free P (S5.3): clamp to the static pair ceiling like the
            // legacy scan; an overflow stays published and sort hard-fails.
            const bool host_free = env_config::host_free_mp_enabled();
            p_cap = host_free ? std::min<uint32_t>(env_config::pair_ceiling(),
                                                   static_cast<uint32_t>(ctx->cap_p_bytes / 4))
                              : static_cast<uint32_t>(ctx->cap_p_bytes / 4);
            Program& prog = ctx->wl_k2seg.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                const CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                auto args = [&](uint32_t mover) -> std::vector<uint32_t> {
                    return {
                        static_cast<uint32_t>(offs->address()),
                        static_cast<uint32_t>(aabb->address()),
                        static_cast<uint32_t>(cnt->address()),
                        static_cast<uint32_t>(ctx->buf_gids->address()),
                        static_cast<uint32_t>(ctx->buf_tids->address()),
                        static_cast<uint32_t>(ctx->buf_pairs_P->address()),
                        static_cast<uint32_t>(projM->address()),
                        nseg, num_tiles, c, num_cores, mover, ctx->dual ? 1u : 0u, permille,
                        p_cap, tiles_x, rows_addr, row_pages,
                        acc.empty() ? 0u : static_cast<uint32_t>(acc[2u * c + mover]),
                        acc.empty() ? 0u : static_cast<uint32_t>(acc[2u * c + mover + 1u]),
                        acc.empty() ? 0u : static_cast<uint32_t>(acc.back()),
                    };
                };
                SetRuntimeArgs(prog, ctx->k2s, core, args(1));
                if (ctx->dual) SetRuntimeArgs(prog, ctx->k2sb, core, args(0));
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_k2seg, false);
            if (pass == 0 && want_early) {
                auto enqueue_sort = [&]() {
                    return sort_onelaunch_enqueue_early(
                        screen_tiles, tiles_x, row_pages, rows_addr,
                        static_cast<uint32_t>(ctx->buf_gids->address()),
                        static_cast<uint32_t>(ctx->buf_tids->address()),
                        static_cast<uint32_t>(ctx->buf_keep->address()),
                        static_cast<uint32_t>(ctx->buf_pairs_P->address()), acc);
                };
                if (cq1 != nullptr) {
                    const distributed::MeshEvent k2_done = ctx->cq->enqueue_record_event();
                    early = enqueue_sort();
                    cq1->enqueue_wait_for_event(k2_done);
                    GSPLAT_HOST_ZONE("host_cq1_proj_m");
                    distributed::EnqueueReadMeshBuffer(*cq1, mread, projM, true);
                } else {
                    distributed::ReadShard(*ctx->cq, mread, projM, distributed::MeshCoordinate(0, 0),
                                           false);
                    const distributed::MeshEvent read_done =
                        ctx->cq->enqueue_record_event_to_host();
                    early = enqueue_sort();
                    GSPLAT_HOST_ZONE("host_wait_proj_m");
                    distributed::EventSynchronize(read_done);
                }
                if (early && mread[2] != 0) {
                    // The early sort took the clamped pairs: drain it, then regrow
                    // and rerun the K2; the sort runs at its usual place.
                    distributed::Finish(*ctx->cq);
                    early = false;
                }
            } else {
                distributed::EnqueueReadMeshBuffer(*ctx->cq, mread, projM, true);
            }
            if (mread[2] == 0 || host_free) break;
            if (pass == 1) throw std::runtime_error("pair overflow after regrow");
            ensure_pair_buffers(*ctx, static_cast<std::size_t>(round_up(mread[1], ELEMS_PER_PAGE)) * 4);
        }
        *M = mread[0];
        *P = mread[1];
        ctx->fused_P = mread[1];
        ctx->fused_ready = true;
        if (fold) {
            device_state::K2CountRows rows;
            rows.buf = ctx->buf_k2_rows;
            rows.num_cores = num_cores;
            rows.row_pages = row_pages;
            rows.num_tiles = screen_tiles;
            rows.P_pub = std::min(mread[1], p_cap);
            // The ranges the kernel took (pfwc_fuse::k2_range[_speed]).
            rows.bounds.assign(2u * num_cores + 1u, 0u);
            for (uint32_t c = 0; c < num_cores; c++) {
                for (uint32_t mover = 0; mover < 2u; mover++) {
                    uint32_t s0 = 0, n0 = 0;
                    if (acc.empty())
                        pfwc_fuse::k2_range(rows.P_pub, num_cores, c, mover, 1u, permille, &s0, &n0);
                    else
                        pfwc_fuse::k2_range_speed(rows.P_pub,
                                                  static_cast<uint32_t>(acc[2u * c + mover]),
                                                  static_cast<uint32_t>(acc[2u * c + mover + 1u]),
                                                  static_cast<uint32_t>(acc.back()), &s0, &n0);
                    rows.bounds[2u * c + mover] = s0;
                    rows.bounds[2u * c + mover + 1u] = s0 + n0;
                }
            }
            rows.bytes = ctx->cap_k2_rows_bytes;
            rows.P_true = mread[1];
            rows.overflow = mread[2];
            rows.early = early;
            rows.cq1 = early && cq1 != nullptr;
            device_state::set_k2_count_rows(rows);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::tile_assign] fused K2 failed: " << e.what() << "\n";
        return false;
    }
}

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
    // Stage-timer sub-buckets of `tile_assign` (stage_timers.h).
    auto& st_acc = stagetimers::acc();
    stagetimers::Span setup_span(st_acc.tile_assign_setup);
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

    // S5.3 (host-free M/P): over-provision the M-domain (K1/scan) work-split +
    // buffers to the static padded_n ceiling = the resident proj_m_* capacity
    // (host-known, view-independent — gather sized it to ceil(N/1024)*1024 >= M).
    // The kernels read the REAL M from the resident proj_M control page and guard
    // g >= M, so the extra padding pages are exact no-ops. n_ceil / mctrl_addr are
    // populated from the resident buffers below (host_free path); when off they
    // stay at the dynamic M_pad / 0 (kernels fall back to the host M arg).
    const bool host_free = env_config::host_free_mp_enabled();
    uint32_t n_ceil = M_pad;       // M-domain page-split ceiling (elems, page-aligned)
    uint32_t mctrl_addr = 0;       // resident proj_M base (0 => kernels use host M arg)
    // S5.3 host-free P-domain: static pair ceiling. When host_free we over-
    // provision the pair buffers + K2 work-split to this fixed, view-independent
    // count and DELETE the mid-frame host P-read — scan_bases clamps the
    // published P to this ceiling (so kernels can never index past the buffers)
    // and K2/sort read the clamped P from the resident ta_pairs_P ctrl page. 0
    // disables the clamp (legacy host-read path). pair_ceiling() carries ~27%
    // margin over the measured 30-view pre-cull P max (3.70M); an overflow is a
    // post-frame hard-fail (sort throws on the overflow flag), never silent.
    const uint32_t p_max = host_free ? env_config::pair_ceiling() : 0u;
    const uint32_t p_max_pad = host_free
        ? round_up(std::max<uint32_t>(p_max, ELEMS_PER_PAGE), ELEMS_PER_PAGE)
        : 0u;

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
            if (host_free && res_M && res_px) {
                // Static M-domain ceiling = the resident proj_m_* capacity
                // (= padded_n = ceil(N/1024)*1024 >= M, fixed across views). The
                // kernels read the real M from proj_M and guard g >= M.
                const uint32_t cap = static_cast<uint32_t>(res_px->size() / 4);
                n_ceil = round_up(std::max<uint32_t>(cap, ELEMS_PER_PAGE), ELEMS_PER_PAGE);
                mctrl_addr = static_cast<uint32_t>(res_M->address());
            }
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
        // host_free: size to the static padded_n ceiling (n_ceil) so the buffers
        // + work-split never depend on this frame's M.
        const std::size_t m_bytes = static_cast<std::size_t>(n_ceil) * 4;
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
        // host_free: offs spans the static ceiling (+1 for offs[M] read by K2).
        const uint32_t offs_count = (host_free ? n_ceil : Mu) + 1;
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

        // Lever 2 (task #99, GSPLAT_TT_SFPU_VIS): the gather already wrote the
        // exclusive pair offsets (proj_m_offs, offs[M] = P) and the packed tile
        // rectangles (proj_m_aabb), and published P (ta_pairs_P, host copy via
        // gather_visible_last_pairs). K1 and the three scans are skipped; K2
        // reads the rectangle instead of recomputing it (TA_K2_AABB).
        uint32_t vis_P = 0;
        const bool vis_path = sfpu_vis_mode() != 0 && !host_free && resident_in &&
                              gather_visible_last_pairs(&vis_P);
        // Lever B (task #125): the gather ran tile_assign_fused_k2, so the pairs
        // are already in buf_gids / buf_tids and ta_pairs_P is published.
        const bool fused_k2 = ctx->fused_ready;
        ctx->fused_ready = false;
        if (!fused_k2) device_state::clear_k2_count_rows("ta_unfused");  // rows of an older frame
        if (fused_k2 && (!vis_path || vis_P != ctx->fused_P)) {
            std::cerr << "[gsplat_tt::tile_assign] PFWC_FUSE: pairs of the fused K2 not usable\n";
            return set_fail();
        }
        std::shared_ptr<distributed::MeshBuffer> vis_offs, vis_aabb;
        if (vis_path) {
            vis_offs = device_state::get_buffer("proj_m_offs");
            vis_aabb = device_state::get_buffer("proj_m_aabb");
            if (!vis_offs || !vis_aabb) {
                std::cerr << "[gsplat_tt::tile_assign] SFPU_VIS: proj_m_offs/aabb missing\n";
                return set_fail();
            }
            if (!ctx->k2v_built && !fused_k2) build_program_k2(*ctx, /*aabb=*/true);
        }

        // ── K1: per-Gaussian AABB -> tiles_per_gaussian ─────────────────
        // Phase B (GSPLAT_TT_CHUNK_FUSION): gather scatter already wrote tpg.
        const bool chunk_fusion_k1 =
            env_config::chunk_fusion_enabled() && resident_in;
        std::shared_ptr<distributed::MeshBuffer> fused_tpg;
        if (chunk_fusion_k1) {
            fused_tpg = device_state::get_buffer("ta_tiles_per_gaussian");
        }
        const bool skip_k1 = (chunk_fusion_k1 && static_cast<bool>(fused_tpg)) || vis_path;
        const auto t_k1_0 = clk::now();
        if (skip_k1) {
            if (!vis_path) ctx->buf_tpg = fused_tpg;
        } else {
            if (chunk_fusion_k1 && !fused_tpg) {
                std::cerr << "[gsplat_tt::tile_assign] CHUNK_FUSION set but "
                             "ta_tiles_per_gaussian missing; running K1\n";
            }
            setup_span.stop();
            stagetimers::Span rtargs_span(st_acc.tile_assign_rtargs);
            const uint32_t k1_pages = n_ceil / ELEMS_PER_PAGE;
            const WorkSplit ws1 = split_pages(k1_pages, num_cores);
            Program& prog1 = ctx->wl_k1.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                // Dual mover: BRISC [start, start+n0), NCRISC [start+n0, end).
                const uint32_t n0 = ctx->dual ? mover0_pages(ws1.count[c]) : 0u;
                auto args = [&](uint32_t start, uint32_t count) -> std::vector<uint32_t> {
                    return {
                        in_px,
                        in_py,
                        in_rx,
                        in_ry,
                        static_cast<uint32_t>(ctx->buf_tpg->address()),
                        start, count, Mu,
                        static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y),
                        static_cast<uint32_t>(tile_size),
                        mctrl_addr,  // arg 11: resident proj_M (real M); 0 = use Mu
                    };
                };
                SetRuntimeArgs(prog1, ctx->k1, core, args(ws1.start[c] + n0, ws1.count[c] - n0));
                if (ctx->dual) SetRuntimeArgs(prog1, ctx->k1b, core, args(ws1.start[c], n0));
            }
            rtargs_span.stop();
            stagetimers::Span enq_span(st_acc.tile_assign_enqueue);
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_k1, false);
        }
        setup_span.stop();  // no-op unless K1 was skipped
        // K1 -> scan chain on one in-order CQ; scan Finish drains K1 (drops k1-only lock).
        const auto t_k1_1 = clk::now();
        T.k1_ms = skip_k1 ? 0.0
                          : std::chrono::duration<double, std::milli>(t_k1_1 - t_k1_0).count();

        gsplat_cpu::TileAssignResult result;
        uint32_t P = 0;
        std::vector<uint32_t> offs;  // host-scan path only
        clk::time_point t_s2_0{};

        if (vis_path) {
            P = vis_P;
            T.scan1_ms = T.prefix_ms = T.d2h_tpg_ms = T.h2d_offs_ms = 0.0;
            if (P == 0) {
                if (device_ok) *device_ok = true;
                T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();
                return result;
            }
        } else if (device_scan) {
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
            stagetimers::Span s1_rt_span(st_acc.tile_assign_rtargs);
            Program& progs1 = ctx->wl_scan1.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(progs1, ctx->ks1, core, {
                    static_cast<uint32_t>(ctx->buf_tpg->address()),
                    static_cast<uint32_t>(ctx->buf_core_total->address()),
                    wss.start[c], wss.count[c], Mu, c,
                    mctrl_addr,  // arg 6: resident proj_M (real M); 0 = use Mu
                });
            }
            s1_rt_span.stop();
            {
                stagetimers::Span s(st_acc.tile_assign_enqueue);
                distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan1, false);
            }
            // scan_bases chains on scan1 output — one Finish for both (in-order CQ).
            stagetimers::Span sb_rt_span(st_acc.tile_assign_rtargs);
            Program& progb = ctx->wl_scan_bases.get_programs().begin()->second;
            CoreCoord core0{0, 0};
            SetRuntimeArgs(progb, ctx->ks_bases, core0, {
                static_cast<uint32_t>(ctx->buf_core_total->address()),
                static_cast<uint32_t>(ctx->buf_core_base->address()),
                static_cast<uint32_t>(ctx->buf_pairs_P->address()),
                num_cores,
                p_max,  // arg 4: static pair ceiling (0 => no clamp), S5.3
            });
            sb_rt_span.stop();
            {
                stagetimers::Span s(st_acc.tile_assign_enqueue);
                distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan_bases, false);
            }
            {
                GSPLAT_HOST_ZONE("host_finish_ta_scan");
                stagetimers::Span s(st_acc.tile_assign_scan_finish);
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

            // S5.3 host-free: DELETE the mid-frame control-read of P. The pair
            // buffers + K2 work-split are over-provisioned to the static p_max
            // ceiling and K2 reads the (clamped) P from the resident ta_pairs_P
            // ctrl page, so the host never needs P here. The P==0 early-exit is
            // also dropped: K2 self-skips (p_start >= P) and sort early-exits on
            // a published P of 0, so a zero-visibility frame is a device no-op.
            if (!host_free) {
                // Control-only read of P (64B page); pairs already resident for sort.
                const auto t_p0 = clk::now();
                std::vector<uint32_t> pbuf(ELEMS_PER_PAGE, 0);
                {
                    GSPLAT_HOST_ZONE("host_ta_d2h_p");
                    stagetimers::Span s(st_acc.tile_assign_p_d2h);
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
            }

            // Phase 2: per-core exclusive prefix-add seeded by core_base -> offs.
            t_s2_0 = clk::now();
            stagetimers::Span s2_rt_span(st_acc.tile_assign_rtargs);
            Program& progs2 = ctx->wl_scan2.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(progs2, ctx->ks2, core, {
                    static_cast<uint32_t>(ctx->buf_tpg->address()),
                    static_cast<uint32_t>(ctx->buf_offs->address()),
                    wss.start[c], wss.count[c], Mu,
                    static_cast<uint32_t>(ctx->buf_core_base->address()),
                    c,
                    mctrl_addr,  // arg 7: resident proj_M (real M); 0 = use Mu
                });
            }
            s2_rt_span.stop();
            {
                stagetimers::Span s(st_acc.tile_assign_enqueue);
                distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan2, false);
            }
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
        // S5.3 host-free: size to the static p_max ceiling (view-independent) so
        // the allocation is fixed and the host needs no per-frame P. Legacy path
        // grows to the dynamic P_pad read back above.
        stagetimers::Span palloc_span(st_acc.tile_assign_setup);
        const uint32_t P_pad = host_free ? p_max_pad : round_up(P, ELEMS_PER_PAGE);
        const std::size_t p_bytes = static_cast<std::size_t>(P_pad) * 4;
        ensure_pair_buffers(*ctx, p_bytes);
        const uint32_t cap_p_elems = static_cast<uint32_t>(ctx->cap_p_bytes / 4);

        // ── K2: pair-centric scatter ────────────────────────────────────
        palloc_span.stop();
        const uint32_t k2_pages = P_pad / ELEMS_PER_PAGE;
        const WorkSplit ws2 = split_pages(k2_pages, num_cores);  // K2 and K4 page split
        // Lever B: tile_assign_fused_k2 already built the pairs (seg K2).
        if (!fused_k2) {
            const auto t_k2_0 = clk::now();
            stagetimers::Span k2_rt_span(st_acc.tile_assign_rtargs);
            Program& prog2 = (vis_path ? ctx->wl_k2v : ctx->wl_k2).get_programs().begin()->second;
            const KernelHandle k2_nc = vis_path ? ctx->k2v : ctx->k2;
            const KernelHandle k2_br = vis_path ? ctx->k2vb : ctx->k2b;
            const uint32_t k2_offs = static_cast<uint32_t>((vis_path ? vis_offs : ctx->buf_offs)->address());
            // TA_K2_AABB reads the packed rectangle through the px slot (args 2..4 unused).
            const uint32_t k2_box = vis_path ? static_cast<uint32_t>(vis_aabb->address()) : 0u;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                // Dual mover: BRISC [start, start+n0), NCRISC [start+n0, end).
                const uint32_t n0 = ctx->dual ? mover0_pages(ws2.count[c]) : 0u;
                auto args = [&](uint32_t start, uint32_t count) -> std::vector<uint32_t> {
                    return {
                        k2_offs,
                        vis_path ? k2_box : in_px,
                        vis_path ? k2_box : in_py,
                        vis_path ? k2_box : in_rx,
                        vis_path ? k2_box : in_ry,
                        static_cast<uint32_t>(ctx->buf_gids->address()),
                        static_cast<uint32_t>(ctx->buf_tids->address()),
                        start, count,
                        host_free ? 0u : P,  // arg 9: host P (host_free => read resident)
                        Mu,
                        static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y),
                        static_cast<uint32_t>(tile_size),
                        // arg 14: resident ta_pairs_P ctrl page (0 => use host P), S5.3
                        host_free ? static_cast<uint32_t>(ctx->buf_pairs_P->address()) : 0u,
                    };
                };
                SetRuntimeArgs(prog2, k2_nc, core, args(ws2.start[c] + n0, ws2.count[c] - n0));
                if (ctx->dual) SetRuntimeArgs(prog2, k2_br, core, args(ws2.start[c], n0));
            }
            k2_rt_span.stop();
            {
                stagetimers::Span s(st_acc.tile_assign_enqueue);
                distributed::EnqueueMeshWorkload(*ctx->cq, vis_path ? ctx->wl_k2v : ctx->wl_k2, false);
            }
            if (k3_pipelined) {
                // scan2 + K2 share one barrier with any in-flight K3 (started before scan1).
                GSPLAT_HOST_ZONE("host_finish_ta_k2");
                {
                    stagetimers::Span s(st_acc.tile_assign_k2_finish);
                    distributed::Finish(*ctx->cq);
                }
                const auto t_barrier = clk::now();
                if (device_scan && !vis_path) {
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
                {
                    stagetimers::Span s(st_acc.tile_assign_k2_finish);
                    distributed::Finish(*ctx->cq);
                }
                const auto t_k2_1 = clk::now();
                T.k2_ms = std::chrono::duration<double, std::milli>(t_k2_1 - t_k2_0).count();
                if (device_scan && !vis_path) {
                    T.scan2_ms =
                        std::chrono::duration<double, std::milli>(t_k2_1 - t_s2_0).count();
                }
            }
        }

        // GSPLAT_TT_SFPU_VIS=2: rerun the legacy K1 + scans on the compact
        // px/py/rx/ry the check-mode scatter also wrote, and compare offs[0..M]
        // and the packed rectangles (host K1 formula) with the gather's.
        if (vis_path && !fused_k2 && sfpu_vis_mode() == 2) {
            vis_check_legacy(*ctx, Mu, n_ceil, offs_pad, num_cores, in_px, in_py, in_rx, in_ry,
                             tiles_x, tiles_y, tile_size, vis_offs, vis_aabb, res_px, res_py,
                             res_rx, res_ry);
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
        stagetimers::Span publish_span(st_acc.tile_assign_publish);
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
            // empty-corner pairs downstream. sort treats keep==0 as a drop while
            // a MISSING ta_pairs_keep buffer hard-fails the device sort — so the
            // mask MUST be present and all-1s for [0, P).
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
            device_state::clear_k2_count_rows("ta_k4");  // the K2's rows count culled pairs
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

        if (device_ok) *device_ok = true;
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::tile_assign] call failed: " << e.what() << "\n";
        return set_fail();
    }
}

}  // namespace gsplat_tt
