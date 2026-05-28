// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt project — amendment-002 tt-005.
//
// Scope (this shift): bounded hotspot port of transform_means_cam — the
// world→camera matrix-vec stage of project. Per-Gaussian math:
//
//   means_cam[i] = R @ means[i]
//
// where R is the extrinsics rotation (rows 0..2 cols 0..2). The translation
// `+t` is intentionally NOT applied here — it is added in the per-Gaussian
// inner loop of project_full_fused (see project.cpp:546-549), so this device
// output is a drop-in replacement for the CPU `means_cam` intermediate. On
// linux without Accelerate this is a single-threaded scalar loop in
// src/gsplat_cpu/project.cpp:528-535 costing ~30-50 ms / frame at N=6.13M
// Gaussians.
//
// Device strategy: SoA tile layout, 1024 Gaussians per tile, 3 input tiles
// (mx, my, mz) + 3 output tiles (mcx, mcy, mcz) per chunk. SFPU executes
// per-tile mul_unary + add_tiles + add_unary chains broadcasting R/t across
// the 1024 lanes. Work splits over the full compute grid via simple
// round-robin (each core gets a contiguous chunk range).

#include "gsplat_tt/project.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
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
constexpr uint32_t ELEMS_PER_TILE = TILE_H * TILE_W;        // 1024
constexpr uint32_t TILE_BYTES_FP32 = ELEMS_PER_TILE * sizeof(float);  // 4096

// CB indices — must match project_means_cam_compute.cpp.
constexpr uint32_t CB_MX    = 0;
constexpr uint32_t CB_MY    = 1;
constexpr uint32_t CB_MZ    = 2;
constexpr uint32_t CB_MCX   = 3;
constexpr uint32_t CB_MCY   = 4;
constexpr uint32_t CB_MCZ   = 5;
constexpr uint32_t CB_TMP_A = 6;
constexpr uint32_t CB_TMP_B = 7;

struct ProjectDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle reader{};
    KernelHandle compute{};
    KernelHandle writer{};
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    // Per-frame DRAM buffer cache. Initial allocation cost is paid once;
    // subsequent frames with N <= cached capacity reuse the same buffers.
    // Tenstorrent MeshBuffer::create has substantial per-call overhead
    // (~5-30 ms on Blackhole at frame-size buffers); per-frame realloc was
    // the dominant per-frame cost in the iter-04 micro-profiling run.
    std::shared_ptr<distributed::MeshBuffer> buf_mx;
    std::shared_ptr<distributed::MeshBuffer> buf_my;
    std::shared_ptr<distributed::MeshBuffer> buf_mz;
    std::shared_ptr<distributed::MeshBuffer> buf_mcx;
    std::shared_ptr<distributed::MeshBuffer> buf_mcy;
    std::shared_ptr<distributed::MeshBuffer> buf_mcz;
    std::size_t cached_bytes = 0;
};

static void build_program(ProjectDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_fp32 = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_FP32, {{id, DataFormat::Float32}});
        c.set_page_size(id, TILE_BYTES_FP32);
        CreateCircularBuffer(program, cores, c);
    };

    // Inputs: depth 2 for double-buffering reader vs compute.
    cb_fp32(CB_MX, 2);
    cb_fp32(CB_MY, 2);
    cb_fp32(CB_MZ, 2);
    // Outputs: depth 2 for double-buffering compute vs writer.
    cb_fp32(CB_MCX, 2);
    cb_fp32(CB_MCY, 2);
    cb_fp32(CB_MCZ, 2);
    // Compute-local scratch CBs for partial products / sums.
    cb_fp32(CB_TMP_A, 2);
    cb_fp32(CB_TMP_B, 2);

    // Reader: 3 DRAM-interleaved TensorAccessorArgs (mx, my, mz).
    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_project_means_cam.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/project_means_cam_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
        });

    // Writer: 3 DRAM-interleaved TensorAccessorArgs (mcx, mcy, mcz).
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_project_means_cam.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static ProjectDeviceContext init_context() {
    ProjectDeviceContext ctx;
    constexpr int device_id = 0;
    ctx.mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    ctx.cq = &ctx.mesh_device->mesh_command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    return ctx;
}

// Lazy-initialised singleton: same lifetime pattern as blend_device.cpp
// (ensure_device()). We keep this in a unique_ptr inside a function-local
// static so test teardown can shut it down cleanly via project_device_shutdown.
static std::unique_ptr<ProjectDeviceContext>& context_slot() {
    static std::unique_ptr<ProjectDeviceContext> ctx;
    return ctx;
}

static ProjectDeviceContext* ensure_context() {
    auto& slot = context_slot();
    if (!slot) {
        try {
            slot = std::make_unique<ProjectDeviceContext>(init_context());
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::project] device init failed: " << e.what() << "\n";
            slot.reset();
        }
    }
    return slot.get();
}

// Pack N x 3 row-major means into 3 SoA buffers of size ceil(N/1024)*1024 fp32.
// Tail padding is zero-filled (transforms harmlessly).
struct SoaInputs {
    std::vector<float> mx;
    std::vector<float> my;
    std::vector<float> mz;
    uint32_t num_tiles = 0;
    uint32_t padded_n = 0;
};

// Get a process-shared CPU thread pool for the host-side pack/unpack work.
// Inlined here to avoid a static dependency dance with backends/cpu_cpp's
// pybind module — using gsplat_cpu::ThreadPool directly with a fresh
// instance the first time we hit this code path.
static gsplat_cpu::ThreadPool& soa_pool() {
    static gsplat_cpu::ThreadPool pool(
        static_cast<std::size_t>(std::max(2u, std::thread::hardware_concurrency())));
    return pool;
}

static SoaInputs pack_means_soa(const float* means, std::size_t N) {
    const uint32_t num_tiles =
        static_cast<uint32_t>((N + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE);
    const uint32_t padded_n = num_tiles * ELEMS_PER_TILE;

    SoaInputs out;
    out.num_tiles = num_tiles;
    out.padded_n = padded_n;
    out.mx.resize(padded_n);
    out.my.resize(padded_n);
    out.mz.resize(padded_n);

    auto& pool = soa_pool();
    const std::size_t W = pool.size();
    const std::size_t chunk = (N + W - 1) / W;
    for (std::size_t w = 0; w < W; ++w) {
        pool.submit([w, W, chunk, N, means, &out]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            for (std::size_t i = lo; i < hi; ++i) {
                out.mx[i] = means[i * 3 + 0];
                out.my[i] = means[i * 3 + 1];
                out.mz[i] = means[i * 3 + 2];
            }
        });
    }
    pool.wait();

    // Zero the tail padding (only matters for the last tile that's
    // partially filled). Single-threaded — only at most (padded_n - N) <=
    // 1023 elements per buffer, ~12 KB total.
    for (std::size_t i = N; i < padded_n; ++i) {
        out.mx[i] = 0.0f;
        out.my[i] = 0.0f;
        out.mz[i] = 0.0f;
    }

    return out;
}

static uint32_t fp32_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(uint32_t));
    return u;
}

// Round-robin contiguous chunk assignment across the compute grid.
// Each core gets [chunk_start, chunk_start + num_chunks) tile IDs.
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

}  // namespace

bool project_device_ready() {
    return ensure_context() != nullptr;
}

void project_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        // Release cached MeshBuffers before tearing down the device.
        slot->buf_mx.reset();
        slot->buf_my.reset();
        slot->buf_mz.reset();
        slot->buf_mcx.reset();
        slot->buf_mcy.reset();
        slot->buf_mcz.reset();
        slot->cached_bytes = 0;
        slot.reset();
    }
}

double transform_means_cam_tt(
    const float* means,
    const float* extrinsics,
    std::size_t N,
    float* means_cam_out) {
    if (N == 0) {
        return 0.0;
    }
    auto* ctx = ensure_context();
    if (ctx == nullptr) {
        // Device init failure — caller should fall back to CPU.
        return -1.0;
    }

    const float r00 = extrinsics[0], r01 = extrinsics[1], r02 = extrinsics[2];
    const float r10 = extrinsics[4], r11 = extrinsics[5], r12 = extrinsics[6];
    const float r20 = extrinsics[8], r21 = extrinsics[9], r22 = extrinsics[10];

    const SoaInputs soa = pack_means_soa(means, N);
    const uint32_t num_tiles = soa.num_tiles;
    const std::size_t soa_bytes = static_cast<std::size_t>(soa.padded_n) * sizeof(float);

    // (Re)allocate cached DRAM buffers when N grows past current capacity.
    // First frame pays full allocation cost; subsequent frames re-use the
    // same MeshBuffer instances and only pay per-frame upload/download.
    if (!ctx->buf_mx || ctx->cached_bytes < soa_bytes) {
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = TILE_BYTES_FP32,
            .buffer_type = BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig rep_cfg{.size = soa_bytes};
        ctx->buf_mx  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_my  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_mz  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_mcx = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_mcy = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_mcz = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->cached_bytes = soa_bytes;
    }
    auto& buf_mx  = ctx->buf_mx;
    auto& buf_my  = ctx->buf_my;
    auto& buf_mz  = ctx->buf_mz;
    auto& buf_mcx = ctx->buf_mcx;
    auto& buf_mcy = ctx->buf_mcy;
    auto& buf_mcz = ctx->buf_mcz;

    // Upload inputs.
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, buf_mx, soa.mx, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, buf_my, soa.my, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, buf_mz, soa.mz, /*blocking=*/false);

    // Work split across compute grid.
    const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
    const WorkSplit ws = split_chunks(num_tiles, num_cores);

    // Per-core runtime args. All cores share the same R/t scalars (broadcast).
    for (uint32_t c = 0; c < num_cores; ++c) {
        CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
        const uint32_t chunk_start = ws.chunk_start[c];
        const uint32_t num_chunks  = ws.num_chunks[c];

        Program& program = ctx->workload.get_programs().begin()->second;

        SetRuntimeArgs(
            program, ctx->reader, core,
            {static_cast<uint32_t>(buf_mx->address()),
             static_cast<uint32_t>(buf_my->address()),
             static_cast<uint32_t>(buf_mz->address()),
             chunk_start,
             num_chunks});

        SetRuntimeArgs(
            program, ctx->compute, core,
            {num_chunks,
             fp32_bits(r00), fp32_bits(r01), fp32_bits(r02),
             fp32_bits(r10), fp32_bits(r11), fp32_bits(r12),
             fp32_bits(r20), fp32_bits(r21), fp32_bits(r22)});

        SetRuntimeArgs(
            program, ctx->writer, core,
            {static_cast<uint32_t>(buf_mcx->address()),
             static_cast<uint32_t>(buf_mcy->address()),
             static_cast<uint32_t>(buf_mcz->address()),
             chunk_start,
             num_chunks});
    }

    const auto t_launch_start = std::chrono::high_resolution_clock::now();
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, /*blocking=*/false);
    distributed::Finish(*ctx->cq);
    const auto t_launch_end = std::chrono::high_resolution_clock::now();

    // Download outputs.
    std::vector<float> mcx_out(soa.padded_n);
    std::vector<float> mcy_out(soa.padded_n);
    std::vector<float> mcz_out(soa.padded_n);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, mcx_out, buf_mcx, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, mcy_out, buf_mcy, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, mcz_out, buf_mcz, /*blocking=*/true);

    // Unpack SoA -> AoS into caller's buffer (parallel; ~24 MB per channel).
    {
        auto& pool = soa_pool();
        const std::size_t W = pool.size();
        const std::size_t chunk = (N + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool.submit([w, W, chunk, N, &mcx_out, &mcy_out, &mcz_out, means_cam_out]() {
                const std::size_t lo = std::min(w * chunk, N);
                const std::size_t hi = std::min(lo + chunk, N);
                for (std::size_t i = lo; i < hi; ++i) {
                    means_cam_out[i * 3 + 0] = mcx_out[i];
                    means_cam_out[i * 3 + 1] = mcy_out[i];
                    means_cam_out[i * 3 + 2] = mcz_out[i];
                }
            });
        }
        pool.wait();
    }

    const auto kernel_ms =
        std::chrono::duration<double, std::milli>(t_launch_end - t_launch_start).count();
    return kernel_ms;
}

}  // namespace gsplat_tt
