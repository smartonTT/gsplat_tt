// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt pfwc — amendment-002 tt-008a.
//
// Companion to project_device.cpp; runs after the project kernel has
// populated means_cam_x/y/z (registered in gsplat_tt::device_state by
// the project context). This kernel reads those device-resident buffers
// via NoC (no host→device round-trip for means_cam), plus a one-time
// cached cov3d upload, and produces:
//   mean_2d (N×2), depth (N), cov_cam_unique (N×6).
// These outputs are downloaded so the existing host pfwc finisher (cov2d /
// radii / valid_mask) can consume them. The host pfwc finisher is a new
// pybind entry point — pfwc_finish_with_precomp — that skips the perspective
// and cov_cam compute (now done on device).

#include "gsplat_tt/pfwc.h"
#include "gsplat_tt/device_state.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
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

#include "gsplat_cpu/thread_pool.h"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace gsplat_tt {
namespace {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t ELEMS_PER_TILE = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES_FP32 = ELEMS_PER_TILE * sizeof(float);

// CB layout — must match pfwc_compute.cpp.
constexpr uint32_t CB_MCX = 0;
constexpr uint32_t CB_MCY = 1;
constexpr uint32_t CB_MCZ = 2;
constexpr uint32_t CB_C00 = 3;
constexpr uint32_t CB_C01 = 4;
constexpr uint32_t CB_C02 = 5;
constexpr uint32_t CB_C11 = 6;
constexpr uint32_t CB_C12 = 7;
constexpr uint32_t CB_C22 = 8;
constexpr uint32_t CB_M2X  = 9;
constexpr uint32_t CB_M2Y  = 10;
constexpr uint32_t CB_DEP  = 11;
constexpr uint32_t CB_CC00 = 12;
constexpr uint32_t CB_CC01 = 13;
constexpr uint32_t CB_CC02 = 14;
constexpr uint32_t CB_CC11 = 15;
constexpr uint32_t CB_CC12 = 16;
constexpr uint32_t CB_CC22 = 17;
constexpr uint32_t CB_TMP_TX = 18;
constexpr uint32_t CB_TMP_TY = 19;
constexpr uint32_t CB_TMP_TZ = 20;

struct PfwcDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle reader{};
    KernelHandle compute{};
    KernelHandle writer{};
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    // Cached cov3d input buffers (6 SoA streams of unique entries).
    std::shared_ptr<distributed::MeshBuffer> buf_c00;
    std::shared_ptr<distributed::MeshBuffer> buf_c01;
    std::shared_ptr<distributed::MeshBuffer> buf_c02;
    std::shared_ptr<distributed::MeshBuffer> buf_c11;
    std::shared_ptr<distributed::MeshBuffer> buf_c12;
    std::shared_ptr<distributed::MeshBuffer> buf_c22;
    std::size_t cov3d_cached_bytes = 0;
    const float* uploaded_cov3d_ptr = nullptr;
    std::size_t uploaded_cov3d_N = 0;

    // Output buffers — published to device_state for downstream stages.
    std::shared_ptr<distributed::MeshBuffer> buf_m2x;
    std::shared_ptr<distributed::MeshBuffer> buf_m2y;
    std::shared_ptr<distributed::MeshBuffer> buf_dep;
    std::shared_ptr<distributed::MeshBuffer> buf_cc00;
    std::shared_ptr<distributed::MeshBuffer> buf_cc01;
    std::shared_ptr<distributed::MeshBuffer> buf_cc02;
    std::shared_ptr<distributed::MeshBuffer> buf_cc11;
    std::shared_ptr<distributed::MeshBuffer> buf_cc12;
    std::shared_ptr<distributed::MeshBuffer> buf_cc22;
    std::size_t output_cached_bytes = 0;
};

static gsplat_cpu::ThreadPool& soa_pool() {
    static gsplat_cpu::ThreadPool pool(
        static_cast<std::size_t>(std::max(2u, std::thread::hardware_concurrency())));
    return pool;
}

struct Cov3dSoa {
    std::vector<float> c00, c01, c02, c11, c12, c22;
    uint32_t num_tiles = 0;
    uint32_t padded_n = 0;
};

static Cov3dSoa pack_cov3d_soa(const float* cov3d_unique, std::size_t N) {
    const uint32_t num_tiles =
        static_cast<uint32_t>((N + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE);
    const uint32_t padded_n = num_tiles * ELEMS_PER_TILE;
    Cov3dSoa s;
    s.num_tiles = num_tiles;
    s.padded_n = padded_n;
    s.c00.resize(padded_n);
    s.c01.resize(padded_n);
    s.c02.resize(padded_n);
    s.c11.resize(padded_n);
    s.c12.resize(padded_n);
    s.c22.resize(padded_n);

    auto& pool = soa_pool();
    const std::size_t W = pool.size();
    const std::size_t chunk = (N + W - 1) / W;
    for (std::size_t w = 0; w < W; ++w) {
        pool.submit([w, chunk, N, cov3d_unique, &s]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            for (std::size_t i = lo; i < hi; ++i) {
                s.c00[i] = cov3d_unique[i * 6 + 0];
                s.c01[i] = cov3d_unique[i * 6 + 1];
                s.c02[i] = cov3d_unique[i * 6 + 2];
                s.c11[i] = cov3d_unique[i * 6 + 3];
                s.c12[i] = cov3d_unique[i * 6 + 4];
                s.c22[i] = cov3d_unique[i * 6 + 5];
            }
        });
    }
    pool.wait();
    // Zero-pad tail.
    for (std::size_t i = N; i < padded_n; ++i) {
        s.c00[i] = 0.0f; s.c01[i] = 0.0f; s.c02[i] = 0.0f;
        s.c11[i] = 0.0f; s.c12[i] = 0.0f; s.c22[i] = 0.0f;
    }
    return s;
}

static inline uint32_t fp32_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(uint32_t));
    return u;
}

struct WorkSplit {
    std::vector<uint32_t> chunk_start;
    std::vector<uint32_t> num_chunks;
};

static WorkSplit split_chunks(uint32_t num_tiles, uint32_t num_cores) {
    WorkSplit ws;
    ws.chunk_start.assign(num_cores, 0);
    ws.num_chunks.assign(num_cores, 0);
    const uint32_t base = num_tiles / num_cores;
    const uint32_t rem = num_tiles % num_cores;
    uint32_t cursor = 0;
    for (uint32_t c = 0; c < num_cores; ++c) {
        const uint32_t cnt = base + (c < rem ? 1U : 0U);
        ws.chunk_start[c] = cursor;
        ws.num_chunks[c] = cnt;
        cursor += cnt;
    }
    return ws;
}

static void build_program(PfwcDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_fp32 = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_FP32, {{id, DataFormat::Float32}});
        c.set_page_size(id, TILE_BYTES_FP32);
        CreateCircularBuffer(program, cores, c);
    };

    // 9 input CBs (means_cam + cov3d)
    cb_fp32(CB_MCX, 2); cb_fp32(CB_MCY, 2); cb_fp32(CB_MCZ, 2);
    cb_fp32(CB_C00, 2); cb_fp32(CB_C01, 2); cb_fp32(CB_C02, 2);
    cb_fp32(CB_C11, 2); cb_fp32(CB_C12, 2); cb_fp32(CB_C22, 2);
    // 9 output CBs
    cb_fp32(CB_M2X, 2); cb_fp32(CB_M2Y, 2); cb_fp32(CB_DEP, 2);
    cb_fp32(CB_CC00, 2); cb_fp32(CB_CC01, 2); cb_fp32(CB_CC02, 2);
    cb_fp32(CB_CC11, 2); cb_fp32(CB_CC12, 2); cb_fp32(CB_CC22, 2);
    // 3 scratch CBs (tx, ty, tz)
    cb_fp32(CB_TMP_TX, 2); cb_fp32(CB_TMP_TY, 2); cb_fp32(CB_TMP_TZ, 2);

    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 9; ++i) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_pfwc.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    // tt-007 fp32 unpack-to-DEST for ALL input CBs.
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    for (uint32_t cb : {CB_MCX, CB_MCY, CB_MCZ,
                        CB_C00, CB_C01, CB_C02,
                        CB_C11, CB_C12, CB_C22,
                        CB_TMP_TX, CB_TMP_TY, CB_TMP_TZ}) {
        u2d[cb] = UnpackToDestMode::UnpackToDestFp32;
    }

    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/pfwc_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
        });

    std::vector<uint32_t> writer_ct;
    for (int i = 0; i < 9; ++i) {
        TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    }
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_pfwc.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static PfwcDeviceContext init_context() {
    PfwcDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    return ctx;
}

static std::unique_ptr<PfwcDeviceContext>& context_slot() {
    static std::unique_ptr<PfwcDeviceContext> ctx;
    return ctx;
}

static PfwcDeviceContext* ensure_context() {
    auto& slot = context_slot();
    if (!slot) {
        try {
            slot = std::make_unique<PfwcDeviceContext>(init_context());
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::pfwc] device init failed: " << e.what() << "\n";
            slot.reset();
        }
    }
    return slot.get();
}

// Pack 36 cov_cam scale floats for a given row-major R (extrinsics 3×3).
// Order: 6 entries × 6 scales per entry, in the order the kernel expects:
//   entry = [cc00, cc01, cc02, cc11, cc12, cc22]
//   scale = [c00, c11, c22, c01, c02, c12]
// where the dot-product expansion is:
//   cc_ij = c00*r(i,0)*r(j,0)
//         + c11*r(i,1)*r(j,1)
//         + c22*r(i,2)*r(j,2)
//         + c01*(r(i,0)*r(j,1) + r(i,1)*r(j,0))
//         + c02*(r(i,0)*r(j,2) + r(i,2)*r(j,0))
//         + c12*(r(i,1)*r(j,2) + r(i,2)*r(j,1))
static void pack_cc_scales(const float r[9], std::vector<uint32_t>& out) {
    const std::pair<int, int> entries[6] = {
        {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    out.clear();
    out.reserve(36);
    for (const auto& [i, j] : entries) {
        const float ri0 = r[i * 3 + 0], ri1 = r[i * 3 + 1], ri2 = r[i * 3 + 2];
        const float rj0 = r[j * 3 + 0], rj1 = r[j * 3 + 1], rj2 = r[j * 3 + 2];
        const float s00 = ri0 * rj0;
        const float s11 = ri1 * rj1;
        const float s22 = ri2 * rj2;
        const float s01 = ri0 * rj1 + ri1 * rj0;
        const float s02 = ri0 * rj2 + ri2 * rj0;
        const float s12 = ri1 * rj2 + ri2 * rj1;
        out.push_back(fp32_bits(s00));
        out.push_back(fp32_bits(s11));
        out.push_back(fp32_bits(s22));
        out.push_back(fp32_bits(s01));
        out.push_back(fp32_bits(s02));
        out.push_back(fp32_bits(s12));
    }
}

}  // namespace

bool pfwc_device_ready() {
    return ensure_context() != nullptr;
}

void pfwc_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        slot->buf_c00.reset(); slot->buf_c01.reset(); slot->buf_c02.reset();
        slot->buf_c11.reset(); slot->buf_c12.reset(); slot->buf_c22.reset();
        slot->buf_m2x.reset(); slot->buf_m2y.reset(); slot->buf_dep.reset();
        slot->buf_cc00.reset(); slot->buf_cc01.reset(); slot->buf_cc02.reset();
        slot->buf_cc11.reset(); slot->buf_cc12.reset(); slot->buf_cc22.reset();
        slot->cov3d_cached_bytes = 0;
        slot->output_cached_bytes = 0;
        slot->uploaded_cov3d_ptr = nullptr;
        slot->uploaded_cov3d_N = 0;
        slot.reset();
    }
}

double pfwc_tt(
    const float* cov3d_unique,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    float* mean_2d_out,
    float* depth_out,
    float* cov_cam_out,
    PfwcCallTimings* timings_out) {
    if (N == 0) {
        return 0.0;
    }
    auto* ctx = ensure_context();
    if (ctx == nullptr) {
        return -1.0;
    }
    PfwcCallTimings tlocal;
    auto& T = (timings_out ? *timings_out : tlocal);

    // Extrinsics row-major 4×4. R is rows 0..2, cols 0..2; t is the 4th col.
    const float r[9] = {
        extrinsics[0],  extrinsics[1],  extrinsics[2],
        extrinsics[4],  extrinsics[5],  extrinsics[6],
        extrinsics[8],  extrinsics[9],  extrinsics[10],
    };
    const float t0 = extrinsics[3];
    const float t1 = extrinsics[7];
    const float t2 = extrinsics[11];

    // Intrinsics 3×3 row-major.
    const float fx = intrinsics[0];
    const float fy = intrinsics[4];
    const float cx = intrinsics[2];
    const float cy = intrinsics[5];

    const uint32_t num_tiles =
        static_cast<uint32_t>((N + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE);
    const uint32_t padded_n = num_tiles * ELEMS_PER_TILE;
    const std::size_t soa_bytes = static_cast<std::size_t>(padded_n) * sizeof(float);

    // Allocate or grow cov3d / output DRAM buffers.
    if (!ctx->buf_c00 || ctx->cov3d_cached_bytes < soa_bytes) {
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = TILE_BYTES_FP32, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig rep_cfg{.size = soa_bytes};
        ctx->buf_c00 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_c01 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_c02 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_c11 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_c12 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_c22 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->cov3d_cached_bytes = soa_bytes;
        ctx->uploaded_cov3d_ptr = nullptr;
        ctx->uploaded_cov3d_N = 0;
    }
    if (!ctx->buf_m2x || ctx->output_cached_bytes < soa_bytes) {
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = TILE_BYTES_FP32, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig rep_cfg{.size = soa_bytes};
        ctx->buf_m2x  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_m2y  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_dep  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_cc00 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_cc01 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_cc02 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_cc11 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_cc12 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_cc22 = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->output_cached_bytes = soa_bytes;

        device_state::register_buffer("pfwc_m2x", ctx->buf_m2x);
        device_state::register_buffer("pfwc_m2y", ctx->buf_m2y);
        device_state::register_buffer("pfwc_depth", ctx->buf_dep);
        device_state::register_buffer("pfwc_cc00", ctx->buf_cc00);
        device_state::register_buffer("pfwc_cc01", ctx->buf_cc01);
        device_state::register_buffer("pfwc_cc02", ctx->buf_cc02);
        device_state::register_buffer("pfwc_cc11", ctx->buf_cc11);
        device_state::register_buffer("pfwc_cc12", ctx->buf_cc12);
        device_state::register_buffer("pfwc_cc22", ctx->buf_cc22);
    }

    // cov3d upload (cached across views).
    T.cache_hit =
        ctx->uploaded_cov3d_ptr == cov3d_unique && ctx->uploaded_cov3d_N == N;
    if (!T.cache_hit) {
        const auto t_pack0 = std::chrono::high_resolution_clock::now();
        const Cov3dSoa soa = pack_cov3d_soa(cov3d_unique, N);
        const auto t_pack1 = std::chrono::high_resolution_clock::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c00, soa.c00, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c01, soa.c01, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c02, soa.c02, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c11, soa.c11, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c12, soa.c12, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_c22, soa.c22, /*blocking=*/false);
        const auto t_pack2 = std::chrono::high_resolution_clock::now();
        T.pack_ms = std::chrono::duration<double, std::milli>(t_pack1 - t_pack0).count();
        T.upload_ms = std::chrono::duration<double, std::milli>(t_pack2 - t_pack1).count();
        ctx->uploaded_cov3d_ptr = cov3d_unique;
        ctx->uploaded_cov3d_N = N;
    }

    // Pull mc{x,y,z} addresses from the project context's device-registered
    // buffers. If the project kernel hasn't been invoked this view (or for
    // this scene), bail back to CPU pfwc.
    auto buf_mcx = device_state::get_buffer("means_cam_x");
    auto buf_mcy = device_state::get_buffer("means_cam_y");
    auto buf_mcz = device_state::get_buffer("means_cam_z");
    if (!buf_mcx || !buf_mcy || !buf_mcz) {
        std::cerr << "[gsplat_tt::pfwc] means_cam_* not registered — call project_tt first\n";
        return -1.0;
    }

    std::vector<uint32_t> cc_scales;
    pack_cc_scales(r, cc_scales);  // 36 fp32 bits

    const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
    const WorkSplit ws = split_chunks(num_tiles, num_cores);

    Program& program = ctx->workload.get_programs().begin()->second;

    const auto t_launch0 = std::chrono::high_resolution_clock::now();
    for (uint32_t c = 0; c < num_cores; ++c) {
        CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
        const uint32_t chunk_start = ws.chunk_start[c];
        const uint32_t num_chunks  = ws.num_chunks[c];

        SetRuntimeArgs(
            program, ctx->reader, core,
            {static_cast<uint32_t>(buf_mcx->address()),
             static_cast<uint32_t>(buf_mcy->address()),
             static_cast<uint32_t>(buf_mcz->address()),
             static_cast<uint32_t>(ctx->buf_c00->address()),
             static_cast<uint32_t>(ctx->buf_c01->address()),
             static_cast<uint32_t>(ctx->buf_c02->address()),
             static_cast<uint32_t>(ctx->buf_c11->address()),
             static_cast<uint32_t>(ctx->buf_c12->address()),
             static_cast<uint32_t>(ctx->buf_c22->address()),
             chunk_start, num_chunks});

        std::vector<uint32_t> compute_args;
        compute_args.reserve(53);
        compute_args.push_back(num_chunks);
        for (int k = 0; k < 9; ++k) compute_args.push_back(fp32_bits(r[k]));
        compute_args.push_back(fp32_bits(t0));
        compute_args.push_back(fp32_bits(t1));
        compute_args.push_back(fp32_bits(t2));
        compute_args.push_back(fp32_bits(fx));
        compute_args.push_back(fp32_bits(fy));
        compute_args.push_back(fp32_bits(cx));
        compute_args.push_back(fp32_bits(cy));
        for (uint32_t s : cc_scales) compute_args.push_back(s);
        SetRuntimeArgs(program, ctx->compute, core, compute_args);

        SetRuntimeArgs(
            program, ctx->writer, core,
            {static_cast<uint32_t>(ctx->buf_m2x->address()),
             static_cast<uint32_t>(ctx->buf_m2y->address()),
             static_cast<uint32_t>(ctx->buf_dep->address()),
             static_cast<uint32_t>(ctx->buf_cc00->address()),
             static_cast<uint32_t>(ctx->buf_cc01->address()),
             static_cast<uint32_t>(ctx->buf_cc02->address()),
             static_cast<uint32_t>(ctx->buf_cc11->address()),
             static_cast<uint32_t>(ctx->buf_cc12->address()),
             static_cast<uint32_t>(ctx->buf_cc22->address()),
             chunk_start, num_chunks});
    }
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, /*blocking=*/false);
    const auto t_launch1 = std::chrono::high_resolution_clock::now();
    distributed::Finish(*ctx->cq);
    const auto t_launch_end = std::chrono::high_resolution_clock::now();
    T.launch_ms = std::chrono::duration<double, std::milli>(t_launch1 - t_launch0).count();
    T.compute_ms = std::chrono::duration<double, std::milli>(t_launch_end - t_launch1).count();

    // Download outputs into per-stream scratch buffers.
    const auto t_dl0 = std::chrono::high_resolution_clock::now();
    std::vector<float> m2x_out(padded_n);
    std::vector<float> m2y_out(padded_n);
    std::vector<float> dep_out(padded_n);
    std::vector<float> cc00_out(padded_n);
    std::vector<float> cc01_out(padded_n);
    std::vector<float> cc02_out(padded_n);
    std::vector<float> cc11_out(padded_n);
    std::vector<float> cc12_out(padded_n);
    std::vector<float> cc22_out(padded_n);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, m2x_out,  ctx->buf_m2x,  /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, m2y_out,  ctx->buf_m2y,  /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, dep_out,  ctx->buf_dep,  /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, cc00_out, ctx->buf_cc00, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, cc01_out, ctx->buf_cc01, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, cc02_out, ctx->buf_cc02, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, cc11_out, ctx->buf_cc11, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, cc12_out, ctx->buf_cc12, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, cc22_out, ctx->buf_cc22, /*blocking=*/true);
    const auto t_dl1 = std::chrono::high_resolution_clock::now();
    T.download_ms = std::chrono::duration<double, std::milli>(t_dl1 - t_dl0).count();

    // Unpack SoA → caller-supplied AoS layouts:
    //   mean_2d_out [N, 2]
    //   depth_out   [N]
    //   cov_cam_out [N, 6]   (order: cc00, cc01, cc02, cc11, cc12, cc22)
    {
        auto& pool = soa_pool();
        const std::size_t W = pool.size();
        const std::size_t chunk = (N + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool.submit([w, chunk, N,
                         &m2x_out, &m2y_out, &dep_out,
                         &cc00_out, &cc01_out, &cc02_out,
                         &cc11_out, &cc12_out, &cc22_out,
                         mean_2d_out, depth_out, cov_cam_out]() {
                const std::size_t lo = std::min(w * chunk, N);
                const std::size_t hi = std::min(lo + chunk, N);
                for (std::size_t i = lo; i < hi; ++i) {
                    mean_2d_out[i * 2 + 0] = m2x_out[i];
                    mean_2d_out[i * 2 + 1] = m2y_out[i];
                    depth_out[i] = dep_out[i];
                    cov_cam_out[i * 6 + 0] = cc00_out[i];
                    cov_cam_out[i * 6 + 1] = cc01_out[i];
                    cov_cam_out[i * 6 + 2] = cc02_out[i];
                    cov_cam_out[i * 6 + 3] = cc11_out[i];
                    cov_cam_out[i * 6 + 4] = cc12_out[i];
                    cov_cam_out[i * 6 + 5] = cc22_out[i];
                }
            });
        }
        pool.wait();
    }
    const auto t_unp = std::chrono::high_resolution_clock::now();
    T.unpack_ms = std::chrono::duration<double, std::milli>(t_unp - t_dl1).count();

    return std::chrono::duration<double, std::milli>(t_launch_end - t_launch0).count();
}

}  // namespace gsplat_tt
