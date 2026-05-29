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

#include "gsplat_tt/tile_assign.h"
#include "gsplat_tt/device_state.h"

#include <algorithm>
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

struct TileAssignDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    distributed::MeshWorkload wl_k1;
    distributed::MeshWorkload wl_k2;
    distributed::MeshWorkload wl_cull;
    KernelHandle k1{};
    KernelHandle k2{};
    KernelHandle k4{};

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
    std::shared_ptr<distributed::MeshBuffer> buf_opacok;
    std::size_t cap_m_bytes = 0;  // capacity of the M-sized buffers (bytes)

    std::shared_ptr<distributed::MeshBuffer> buf_offs;
    std::size_t cap_offs_bytes = 0;

    std::shared_ptr<distributed::MeshBuffer> buf_gids;
    std::shared_ptr<distributed::MeshBuffer> buf_tids;
    std::shared_ptr<distributed::MeshBuffer> buf_keep;
    std::size_t cap_p_bytes = 0;

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
    // gid,tid,a,b,c,px,py,m2thr,opacok,keep
    for (uint32_t id = 0; id < 10; id++) scratch_cb(id);

    std::vector<uint32_t> ct;
    for (int i = 0; i < 10; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
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

}  // namespace

bool tile_assign_device_ready() { return ensure_context() != nullptr; }

void tile_assign_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        slot->buf_px.reset();
        slot->buf_py.reset();
        slot->buf_rx.reset();
        slot->buf_ry.reset();
        slot->buf_tpg.reset();
        slot->buf_a.reset();
        slot->buf_b.reset();
        slot->buf_c.reset();
        slot->buf_m2thr.reset();
        slot->buf_opacok.reset();
        slot->buf_offs.reset();
        slot->buf_gids.reset();
        slot->buf_tids.reset();
        slot->buf_keep.reset();
        slot->buf_pairs_P.reset();
        // NOTE: do NOT close mesh_device — device_state owns it.
        slot.reset();
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
    const bool resident_in = [] {
        const char* v = std::getenv("GSPLAT_TT_RESIDENT_TA_IN");
        return v != nullptr && v[0] == '1';
    }();
    // ── R4/R5: resident-pairs handoff (GSPLAT_TT_RESIDENT_PAIRS=1) ────────
    // When on (and the per-pair cull runs), TA keeps the full-P gaussian-major
    // (gid,tid) pairs + keep mask RESIDENT in DRAM and registers them in
    // device_state for the sort stage to bin on-device. The host D2H of the
    // pairs (28ms) and the sequential gaussian-major compaction (37ms) are
    // dropped — sort's device binning reads keep[] and compacts implicitly.
    const bool resident_pairs = [] {
        const char* v = std::getenv("GSPLAT_TT_RESIDENT_PAIRS");
        return v != nullptr && v[0] == '1';
    }();
    std::shared_ptr<distributed::MeshBuffer> res_px, res_py, res_rx, res_ry,
        res_a, res_b, res_c;

    try {
        if (resident_in) {
            res_px = device_state::get_buffer("proj_m_px");
            res_py = device_state::get_buffer("proj_m_py");
            res_rx = device_state::get_buffer("proj_m_rx");
            res_ry = device_state::get_buffer("proj_m_ry");
            res_a  = device_state::get_buffer("proj_m_a");
            res_b  = device_state::get_buffer("proj_m_b");
            res_c  = device_state::get_buffer("proj_m_c");
            auto res_M = device_state::get_buffer("proj_M");
            if (!res_px || !res_py || !res_rx || !res_ry || !res_a || !res_b ||
                !res_c || !res_M) {
                std::cerr << "[gsplat_tt::tile_assign] RESIDENT_TA_IN set but "
                             "proj_m_* not resident; needs RESIDENT_PROJECT+"
                             "RESIDENT_GATHER\n";
                return set_fail();
            }
            // M from the resident scalar (proj_M) must equal the host M.
            std::vector<uint32_t> mbuf(ELEMS_PER_PAGE);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, mbuf, res_M, true);
            if (mbuf[0] != Mu) {
                std::fprintf(stderr,
                    "[TA resident] proj_M=%u != host M=%u — abort resident\n",
                    mbuf[0], Mu);
                return set_fail();
            }
        }

        // ── Allocate / grow M-sized buffers ─────────────────────────────
        const std::size_t m_bytes = static_cast<std::size_t>(M_pad) * 4;
        if (!ctx->buf_tpg || ctx->cap_m_bytes < m_bytes) {
            ctx->buf_tpg    = make_dram(ctx->mesh_device.get(), m_bytes);
            ctx->buf_m2thr  = make_dram(ctx->mesh_device.get(), m_bytes);
            ctx->buf_opacok = make_dram(ctx->mesh_device.get(), m_bytes);
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
        const uint32_t offs_count = Mu + 1;
        const uint32_t offs_pad = round_up(offs_count, ELEMS_PER_PAGE);
        const std::size_t offs_bytes = static_cast<std::size_t>(offs_pad) * 4;
        if (!ctx->buf_offs || ctx->cap_offs_bytes < offs_bytes) {
            ctx->buf_offs = make_dram(ctx->mesh_device.get(), offs_bytes);
            ctx->cap_offs_bytes = offs_bytes;
        }

        // ── Pack inputs (SoA px,py,rx,ry), zero-padded tail ─────────────
        // Resident mode skips this H2D entirely — K1/K2 read proj_m_* over NoC.
        if (!resident_in) {
            std::vector<uint32_t> px(M_pad, 0), py(M_pad, 0), rx(M_pad, 0), ry(M_pad, 0);
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
        const auto t_k1_0 = clk::now();
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
        distributed::Finish(*ctx->cq);
        const auto t_k1_1 = clk::now();
        T.k1_ms = std::chrono::duration<double, std::milli>(t_k1_1 - t_k1_0).count();

        // D2H tiles_per_gaussian.
        std::vector<uint32_t> tpg(M_pad);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, tpg, ctx->buf_tpg, true);

        // ── H1: host exclusive prefix-sum ───────────────────────────────
        const auto t_pre0 = clk::now();
        std::vector<uint32_t> offs(offs_pad, 0);
        uint64_t acc = 0;
        for (uint32_t m = 0; m < Mu; m++) {
            offs[m] = static_cast<uint32_t>(acc);
            acc += tpg[m];
        }
        const uint32_t P = static_cast<uint32_t>(acc);
        for (uint32_t m = Mu; m < offs_pad; m++) offs[m] = P;  // offs[M..] = P
        const auto t_pre1 = clk::now();
        T.prefix_ms = std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();

        gsplat_cpu::TileAssignResult result;
        result.tiles_per_gaussian.assign(M, 0);
        for (uint32_t m = 0; m < Mu; m++)
            result.tiles_per_gaussian[m] = static_cast<int64_t>(tpg[m]);

        if (P == 0) {
            if (device_ok) *device_ok = true;
            T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0).count();
            return result;
        }

        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_offs, offs, false);

        // ── Allocate / grow pair buffers ────────────────────────────────
        const uint32_t P_pad = round_up(P, ELEMS_PER_PAGE);
        const std::size_t p_bytes = static_cast<std::size_t>(P_pad) * 4;
        if (!ctx->buf_gids || ctx->cap_p_bytes < p_bytes) {
            ctx->buf_gids = make_dram(ctx->mesh_device.get(), p_bytes);
            ctx->buf_tids = make_dram(ctx->mesh_device.get(), p_bytes);
            ctx->buf_keep = make_dram(ctx->mesh_device.get(), p_bytes);
            ctx->cap_p_bytes = p_bytes;
        }

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
        distributed::Finish(*ctx->cq);
        const auto t_k2_1 = clk::now();
        T.k2_ms = std::chrono::duration<double, std::milli>(t_k2_1 - t_k2_0).count();

        const bool do_cull = (covs_2d != nullptr) && (opacities != nullptr);
        // Resident-pairs is only valid when the cull runs (keep mask is the
        // implicit compaction). Otherwise fall through to the host path.
        const bool resident_pairs_active = resident_pairs && do_cull;

        // ── D2H pairs ───────────────────────────────────────────────────
        // Skipped in resident-pairs mode: the (gid,tid) pairs stay resident and
        // sort's device binning consumes them directly.
        if (!resident_pairs_active) {
            const auto t_d2h0 = clk::now();
            std::vector<uint32_t> gids(P_pad), tids(P_pad);
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
            // K3 (host bridge): per-Gaussian m2_thresh + opacity-floor.
            // Bit-exact match to gsplat_cpu::tile_assign lines 174-182.
            // The cov a/b/c packing+upload is only needed when NOT resident:
            // in resident mode K4 reads proj_m_a/b/c over NoC. m2_thresh/opacok
            // remain a host bridge (computed from the host opacities array).
            std::vector<uint32_t> a_v, b_v, c_v;
            if (!resident_in) {
                a_v.assign(M_pad, 0);
                b_v.assign(M_pad, 0);
                c_v.assign(M_pad, 0);
            }
            std::vector<uint32_t> m2t_v(M_pad, 0);
            std::vector<uint32_t> ok_v(M_pad, 0);
            for (uint32_t m = 0; m < Mu; m++) {
                if (!resident_in) {
                    const float a = covs_2d[m * 4 + 0];
                    const float b = covs_2d[m * 4 + 1];
                    const float c = covs_2d[m * 4 + 3];
                    std::memcpy(&a_v[m], &a, 4);
                    std::memcpy(&b_v[m], &b, 4);
                    std::memcpy(&c_v[m], &c, 4);
                }
                const float op = opacities[m];
                float m2t = 0.0f;
                int32_t ok = 0;
                if (op > contrib_floor) {
                    m2t = -2.0f * std::log(contrib_floor / op);
                    ok = 1;
                }
                std::memcpy(&m2t_v[m], &m2t, 4);
                ok_v[m] = static_cast<uint32_t>(ok);
            }
            if (!resident_in) {
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_a, a_v, false);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_b, b_v, false);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c, c_v, false);
            }
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_m2thr, m2t_v, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_opacok, ok_v, false);

            // K4: per-pair cull -> keep_mask.
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
                    static_cast<uint32_t>(ctx->buf_opacok->address()),
                    static_cast<uint32_t>(ctx->buf_keep->address()),
                    ws2.start[cc], ws2.count[cc], P,
                    static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tile_size),
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_cull, false);
            distributed::Finish(*ctx->cq);
            const auto t_cull1 = clk::now();
            T.cull_ms = std::chrono::duration<double, std::milli>(t_cull1 - t_cull0).count();

            if (resident_pairs_active) {
                // R4/R5: publish the full-P resident pairs + keep mask for the
                // device-binning sort path. No keep D2H, no host compaction —
                // sort reads keep[] and compacts implicitly while binning.
                device_state::register_buffer("ta_pairs_gid", ctx->buf_gids);
                device_state::register_buffer("ta_pairs_tid", ctx->buf_tids);
                device_state::register_buffer("ta_pairs_keep", ctx->buf_keep);
                if (!ctx->buf_pairs_P) {
                    ctx->buf_pairs_P = make_dram(ctx->mesh_device.get(), PAGE_BYTES);
                    device_state::register_buffer("ta_pairs_P", ctx->buf_pairs_P);
                }
                std::vector<uint32_t> pbuf(ELEMS_PER_PAGE, 0);
                pbuf[0] = P;        // full pre-cull pair count
                pbuf[1] = P_pad;    // padded count (page-aligned)
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_pairs_P, pbuf, false);
                distributed::Finish(*ctx->cq);
                // result.gaussian_ids / tile_ids intentionally left empty: the
                // resident-pairs sort path does not read them.
            } else {
                // D2H keep_mask.
                const auto t_kd0 = clk::now();
                std::vector<uint32_t> keep(P_pad);
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
