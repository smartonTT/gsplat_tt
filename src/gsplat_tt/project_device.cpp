// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt project — amendment-002 tt-005.
//
// tt-005 (iter-04): bounded hotspot port of transform_means_cam — world→
// camera matrix-vec. Per-Gaussian math:
//
//   means_cam[i] = R @ means[i]
//
// where R is the extrinsics rotation (rows 0..2 cols 0..2). Translation +t
// is intentionally NOT applied here — it is added in the per-Gaussian inner
// loop of project_full_fused (see project.cpp:546-549), so this device
// output is a drop-in replacement for the CPU `means_cam` intermediate.
//
// tt-005b (iter-05) — PERF (super-supervisor 2026-05-28):
//   * Per-frame R↔W means_3d round-trip eats ~85 ms/frame at N=6.13M (the
//     super-supervisor's measurement vs cpu_cpp_mb). means_3d is invariant
//     across views — only extrinsics changes. Skip pack + upload when the
//     input pointer/N hasn't changed.
//   * Register the 3 fp32 means_cam DRAM buffers in gsplat_tt::device_state
//     so future device stages (tile_assign / cov2d / blend) can NoC-read
//     them directly instead of pulling host-resident copies. Stage-A
//     device-resident buffer registry.

#include "gsplat_tt/project.h"
#include "gsplat_tt/device_state.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <thread>
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
constexpr uint32_t ELEMS_PER_TILE = TILE_H * TILE_W;                  // 1024
constexpr uint32_t TILE_BYTES_FP32 = ELEMS_PER_TILE * sizeof(float);  // 4096

// CB indices — must match project_means_cam_compute.cpp (HEAD layout).
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

    // Cached DRAM buffers — shared across frames. Outputs (mcx/mcy/mcz)
    // are also published in gsplat_tt::device_state under the keys
    // "means_cam_x/y/z" for downstream stages.
    std::shared_ptr<distributed::MeshBuffer> buf_mx;
    std::shared_ptr<distributed::MeshBuffer> buf_my;
    std::shared_ptr<distributed::MeshBuffer> buf_mz;
    std::shared_ptr<distributed::MeshBuffer> buf_mcx;
    std::shared_ptr<distributed::MeshBuffer> buf_mcy;
    std::shared_ptr<distributed::MeshBuffer> buf_mcz;
    std::size_t cached_bytes = 0;

    // Pointer-keyed means cache: skip pack+upload when the caller hands us
    // the same means pointer with the same N. In the 30-view bench this
    // fires on frame 0 only and saves ~85 ms/frame for the remaining 29
    // frames (3×24 MB H2D + host-side SoA pack work).
    const float* uploaded_means_ptr = nullptr;
    std::size_t uploaded_means_N = 0;
};

static void build_program(ProjectDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_fp32 = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_FP32, {{id, DataFormat::Float32}});
        c.set_page_size(id, TILE_BYTES_FP32);
        CreateCircularBuffer(program, cores, c);
    };

    cb_fp32(CB_MX, 2);
    cb_fp32(CB_MY, 2);
    cb_fp32(CB_MZ, 2);
    cb_fp32(CB_MCX, 2);
    cb_fp32(CB_MCY, 2);
    cb_fp32(CB_MCZ, 2);
    cb_fp32(CB_TMP_A, 2);
    cb_fp32(CB_TMP_B, 2);

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

    // tt-007: enable UnpackToDestFp32 for every CB so 32-bit data lands in
    // DEST directly (bypassing SrcA/SrcB). Without this, copy_tile / add_tiles
    // route through SrcA/SrcB which truncate to bf16 — that is the 34.5 dB
    // hero-PSNR regression observed at iter-06e. With unpack-to-dest enabled
    // plus dest-slot accumulation via add_binary_tile (SFPU), every operation
    // stays in fp32 from CB read through compute through CB write.
    //
    // The vector index matches CB id. Our kernel uses CBs 0..7 (MX,MY,MZ,
    // MCX,MCY,MCZ,TMP_A,TMP_B). tt-metal requires the vector length to match
    // its full buf_formats table — runtime check enforces size() == 64.
    std::vector<UnpackToDestMode> u2d_modes(64, UnpackToDestMode::Default);
    u2d_modes[CB_MX] = UnpackToDestMode::UnpackToDestFp32;
    u2d_modes[CB_MY] = UnpackToDestMode::UnpackToDestFp32;
    u2d_modes[CB_MZ] = UnpackToDestMode::UnpackToDestFp32;

    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/project_means_cam_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = u2d_modes,
            .math_approx_mode = false,
        });

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
    // Use the shared MeshDevice owned by gsplat_tt::device_state. This makes
    // blend and project share the same handle, eliminating the double-close
    // observed at iter-04 (ShmResourceTracker::cleanup_all double-free).
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    return ctx;
}

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

struct SoaInputs {
    std::vector<float> mx;
    std::vector<float> my;
    std::vector<float> mz;
    uint32_t num_tiles = 0;
    uint32_t padded_n = 0;
};

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
        pool.submit([w, chunk, N, means, &out]() {
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

    // Zero tail padding (only the last partial tile, <=12 KB total).
    for (std::size_t i = N; i < padded_n; ++i) {
        out.mx[i] = 0.0f;
        out.my[i] = 0.0f;
        out.mz[i] = 0.0f;
    }
    return out;
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

}  // namespace

bool project_device_ready() {
    return ensure_context() != nullptr;
}

void project_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        slot->buf_mx.reset();
        slot->buf_my.reset();
        slot->buf_mz.reset();
        slot->buf_mcx.reset();
        slot->buf_mcy.reset();
        slot->buf_mcz.reset();
        slot->cached_bytes = 0;
        slot->uploaded_means_ptr = nullptr;
        slot->uploaded_means_N = 0;
        // NOTE: do NOT close mesh_device here. device_state owns it.
        slot.reset();
    }
}

double transform_means_cam_tt(
    const float* means,
    const float* extrinsics,
    std::size_t N,
    float* means_cam_out,
    ProjectCallTimings* timings_out) {
    if (N == 0) {
        return 0.0;
    }
    auto* ctx = ensure_context();
    if (ctx == nullptr) {
        return -1.0;
    }
    ProjectCallTimings tlocal;
    auto& T = (timings_out ? *timings_out : tlocal);

    const float r00 = extrinsics[0], r01 = extrinsics[1], r02 = extrinsics[2];
    const float r10 = extrinsics[4], r11 = extrinsics[5], r12 = extrinsics[6];
    const float r20 = extrinsics[8], r21 = extrinsics[9], r22 = extrinsics[10];

    const uint32_t num_tiles =
        static_cast<uint32_t>((N + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE);
    const uint32_t padded_n = num_tiles * ELEMS_PER_TILE;
    const std::size_t soa_bytes = static_cast<std::size_t>(padded_n) * sizeof(float);

    // (Re)allocate cached DRAM buffers when N outgrows current capacity.
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
        // Buffers fresh — uploaded_means_ptr/N are stale.
        ctx->uploaded_means_ptr = nullptr;
        ctx->uploaded_means_N = 0;

        // Publish output handles for downstream device stages. Re-registers
        // on grow; keys are stable.
        device_state::register_buffer("means_cam_x", ctx->buf_mcx);
        device_state::register_buffer("means_cam_y", ctx->buf_mcy);
        device_state::register_buffer("means_cam_z", ctx->buf_mcz);
    }

    // ── Per-frame data path ─────────────────────────────────────────────
    // tt-005b: only pack + upload means when the input pointer has changed.
    // In the 30-view bench, means_3d is a single torch tensor reused across
    // all views, so this fires on frame 0 only (microbench-confirmed:
    // call 0 pack=49.8 ms + upload=7.8 ms; calls 1+ skipped).
    T.cache_hit = !(ctx->uploaded_means_ptr != means || ctx->uploaded_means_N != N);
    if (!T.cache_hit) {
        const auto t_pack0 = std::chrono::high_resolution_clock::now();
        const SoaInputs soa = pack_means_soa(means, N);
        const auto t_pack1 = std::chrono::high_resolution_clock::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_mx, soa.mx, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_my, soa.my, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_mz, soa.mz, /*blocking=*/false);
        const auto t_pack2 = std::chrono::high_resolution_clock::now();
        T.pack_ms = std::chrono::duration<double, std::milli>(t_pack1 - t_pack0).count();
        T.upload_ms = std::chrono::duration<double, std::milli>(t_pack2 - t_pack1).count();
        ctx->uploaded_means_ptr = means;
        ctx->uploaded_means_N = N;
    }

    auto& buf_mx  = ctx->buf_mx;
    auto& buf_my  = ctx->buf_my;
    auto& buf_mz  = ctx->buf_mz;
    auto& buf_mcx = ctx->buf_mcx;
    auto& buf_mcy = ctx->buf_mcy;
    auto& buf_mcz = ctx->buf_mcz;

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
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, /*blocking=*/false);
    const auto t_launch1 = std::chrono::high_resolution_clock::now();
    distributed::Finish(*ctx->cq);
    const auto t_launch_end = std::chrono::high_resolution_clock::now();
    T.launch_ms = std::chrono::duration<double, std::milli>(t_launch1 - t_launch0).count();
    T.compute_ms = std::chrono::duration<double, std::milli>(t_launch_end - t_launch1).count();

    // tt-008b: skip D2H when caller passed nullptr — means_cam stays purely
    // device-resident for downstream stages (pfwc_tt etc.) to consume via NoC.
    // The means_cam_{x,y,z} buffers remain registered in device_state.
    if (means_cam_out == nullptr) {
        T.download_ms = 0.0;
        T.unpack_ms = 0.0;
        return std::chrono::duration<double, std::milli>(t_launch_end - t_launch0).count();
    }

    // Download outputs. Until downstream stages also live on device the
    // host still consumes means_cam in project_full_with_cov3d. Once
    // tile_assign (tt-006) is on device it will NoC-read these same
    // mc{x,y,z} buffers directly via device_state and this D2H drops.
    const auto t_dl0 = std::chrono::high_resolution_clock::now();
    std::vector<float> mcx_out(padded_n);
    std::vector<float> mcy_out(padded_n);
    std::vector<float> mcz_out(padded_n);
    // tt-metal mesh_command_queue_base.cpp:217 enforces blocking=true on
    // EnqueueReadMeshBuffer; non-blocking reads must go through
    // enqueue_read_shards. Keep blocking and time individually for profiling.
    distributed::EnqueueReadMeshBuffer(*ctx->cq, mcx_out, buf_mcx, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, mcy_out, buf_mcy, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, mcz_out, buf_mcz, /*blocking=*/true);
    const auto t_dl1 = std::chrono::high_resolution_clock::now();
    T.download_ms = std::chrono::duration<double, std::milli>(t_dl1 - t_dl0).count();

    {
        auto& pool = soa_pool();
        const std::size_t W = pool.size();
        const std::size_t chunk = (N + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool.submit([w, chunk, N, &mcx_out, &mcy_out, &mcz_out, means_cam_out]() {
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
    const auto t_unp = std::chrono::high_resolution_clock::now();
    T.unpack_ms = std::chrono::duration<double, std::milli>(t_unp - t_dl1).count();

    return std::chrono::duration<double, std::milli>(t_launch_end - t_launch0).count();
}

double transform_means_cam_tt_no_download(
    const float* means,
    const float* extrinsics,
    std::size_t N,
    ProjectCallTimings* timings_out) {
    // Convenience wrapper — nullptr means_cam_out tells the main path to
    // skip the D2H readback and SoA→AoS unpack work. Saves 25-30 ms/view
    // when downstream stages (e.g. pfwc_tt) consume means_cam directly
    // from device-resident buffers via NoC.
    return transform_means_cam_tt(means, extrinsics, N, /*means_cam_out=*/nullptr, timings_out);
}

}  // namespace gsplat_tt
