// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt FUSED project(means_cam)+pfwc —
// amendment-002 tt-008c / iter-133 (program fusion #1).
//
// iter-133: the former standalone project_means_cam program is fused into this
// ONE program (single CreateProgram / EnqueueMeshWorkload). The reader streams
// world-space means (mx,my,mz) + cov3d unique; the compute kernel runs the
// world→camera means transform in L1 (no means_cam DRAM round-trip) and then the
// pfwc body; the writer emits 8 per-Gaussian outputs:
//   mean_2d.{x,y}, depth, cov2d.{a,b,c}, radii.{x,y}
// All 8 outputs are registered in device_state under
//   pfwc_m2x / pfwc_m2y / pfwc_depth / pfwc_a / pfwc_b / pfwc_c /
//   pfwc_rx / pfwc_ry
// so downstream device kernels (tt-006 tile_assign) can consume them
// directly without going through host memory.
//
// When mean_2d_out / depth_out / cov2d_out / radii_out are non-null, the
// matching streams are still D2H'd + unpacked SoA→AoS into the caller-
// provided buffers (legacy host-finisher path; this is the tt-008b
// behavior, just with cov2d/radii instead of cov_cam).

#include "pfwc.h"
#include "device_state.h"
#include "env_config.h"
#include "gather_visible.h"
#include "../kernels/dataflow/pfwc_fuse.h"
#include "../kernels/dataflow/pfwc_wsplit.h"
#include "host_profile.h"
#include "stage_timers.h"
#include "vis_mode.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
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

// CB layout — must match project_pfwc_compute.cpp and writer_pfwc.cpp.
// iter-133 fusion: CBs 0,1,2 now carry the WORLD-space means (mx,my,mz); the
// fused compute kernel applies the means_cam transform + translation in L1.
constexpr uint32_t CB_MX  = 0;
constexpr uint32_t CB_MY  = 1;
constexpr uint32_t CB_MZ  = 2;
constexpr uint32_t CB_C00 = 3;
constexpr uint32_t CB_C01 = 4;
constexpr uint32_t CB_C02 = 5;
constexpr uint32_t CB_C11 = 6;
constexpr uint32_t CB_C12 = 7;
constexpr uint32_t CB_C22 = 8;
constexpr uint32_t CB_M2X = 9;
constexpr uint32_t CB_M2Y = 10;
constexpr uint32_t CB_DEP = 11;
constexpr uint32_t CB_A   = 12;
constexpr uint32_t CB_B   = 13;
constexpr uint32_t CB_C   = 14;
constexpr uint32_t CB_RX  = 15;
constexpr uint32_t CB_RY  = 16;
constexpr uint32_t CB_TMP_TX     = 17;
constexpr uint32_t CB_TMP_TY     = 18;
constexpr uint32_t CB_TMP_TZ     = 19;
constexpr uint32_t CB_TMP_INV_TZ = 20;
constexpr uint32_t CB_TMP_CC00   = 21;
constexpr uint32_t CB_TMP_CC01   = 22;
constexpr uint32_t CB_TMP_CC02   = 23;
constexpr uint32_t CB_TMP_CC11   = 24;
constexpr uint32_t CB_TMP_CC12   = 25;
constexpr uint32_t CB_TMP_CC22   = 26;
// A1 (iter 111): per-gaussian conic scratch (a,b,c stashed before the conic fold).
constexpr uint32_t CB_TMP_A      = 27;
constexpr uint32_t CB_TMP_B      = 28;
constexpr uint32_t CB_TMP_C      = 29;
// Lever 2 (task #99, PFWC_VIS program only): opacity input, scratch copies of
// mean_x/mean_y/rx/ry, the two word tiles, and writer staging (vis_tile.h).
constexpr uint32_t CB_OP     = 30;
constexpr uint32_t CB_TMP_MX = 31;
constexpr uint32_t CB_TMP_MY = 32;
constexpr uint32_t CB_TMP_RX = 33;
constexpr uint32_t CB_TMP_RY = 34;
constexpr uint32_t CB_TPG    = 35;
constexpr uint32_t CB_AABB   = 36;
constexpr uint32_t CB_VMASK  = 37;  // 128 B mask staging
constexpr uint32_t CB_VCNT   = 38;  // per-tile counts staging
constexpr uint32_t CB_VOP    = 39;  // opacity tile for RECHECK
constexpr uint32_t VIS_MASK_BYTES = 128;
constexpr uint32_t VIS_COUNTS_PAGE = 1024;  // vis_tile::COUNTS_PAGE_BYTES
constexpr uint32_t VIS_CNT_STAGING = 16 * 1024;  // up to ~1900 tiles per core
// Lever B (task #125) writer staging (writer_pfwc_fuse.cpp): col r/g/b and the
// two puboc tiles (task #122), 8 banks x 16 records (task #122), the dep /
// offs / aabb pages, the counts page, 64 B alignment.
constexpr uint32_t CB_FUSE = 40;
constexpr uint32_t FUSE_CB_BYTES = 5 * TILE_BYTES_FP32 + 8 * 16 * 64 + 4 * 64 + 64 + 64;

struct PfwcDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle reader{};
    KernelHandle compute{};
    KernelHandle writer{};
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    // Cached world-space means input buffers (3 SoA streams). iter-133 fusion:
    // the fused kernel reads means directly and runs the means_cam transform in
    // L1, so there is no longer a separate project program / means_cam DRAM
    // round-trip. Pointer-keyed upload cache (fires on frame 0 only).
    std::shared_ptr<distributed::MeshBuffer> buf_mx;
    std::shared_ptr<distributed::MeshBuffer> buf_my;
    std::shared_ptr<distributed::MeshBuffer> buf_mz;
    std::size_t means_cached_bytes = 0;
    const float* uploaded_means_ptr = nullptr;
    std::size_t uploaded_means_N = 0;

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
    std::shared_ptr<distributed::MeshBuffer> buf_a;
    std::shared_ptr<distributed::MeshBuffer> buf_b;
    std::shared_ptr<distributed::MeshBuffer> buf_c;
    std::shared_ptr<distributed::MeshBuffer> buf_rx;
    std::shared_ptr<distributed::MeshBuffer> buf_ry;
    std::size_t output_cached_bytes = 0;

    // Lever 2 (task #99): the PFWC_VIS program variant (built on first use) and
    // its extra outputs.
    bool vis_built = false;
    distributed::MeshWorkload wl_vis;
    KernelHandle vreader{};
    KernelHandle vcompute{};
    KernelHandle vwriter{};
    std::shared_ptr<distributed::MeshBuffer> buf_vtpg;
    std::shared_ptr<distributed::MeshBuffer> buf_vaabb;
    std::shared_ptr<distributed::MeshBuffer> buf_vmask;
    std::shared_ptr<distributed::MeshBuffer> buf_vcounts;
    uint32_t vis_cap_tiles = 0;

    // Lever B (task #125): the PFWC_VIS program with the fused compaction
    // writer (writer_pfwc_fuse.cpp). Built on first use.
    bool fuse_built = false;
    distributed::MeshWorkload wl_fuse;
    KernelHandle freader{};
    KernelHandle fcompute{};
    KernelHandle fwriter{};
    // Task #207 (GSPLAT_TT_PFWC_WRITER_SPLIT): freader / fwriter are
    // writer_pfwc_split.cpp (NCRISC / BRISC); its mailbox semaphores.
    bool fuse_wsplit = false;
    uint32_t fsem[pfwc_wsplit::NUM_SEMS] = {};
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
    for (std::size_t i = N; i < padded_n; ++i) {
        s.c00[i] = 0.0f; s.c01[i] = 0.0f; s.c02[i] = 0.0f;
        s.c11[i] = 0.0f; s.c12[i] = 0.0f; s.c22[i] = 0.0f;
    }
    return s;
}

struct MeansSoa {
    std::vector<float> mx, my, mz;
    uint32_t num_tiles = 0;
    uint32_t padded_n = 0;
};

static MeansSoa pack_means_soa(const float* means, std::size_t N) {
    const uint32_t num_tiles =
        static_cast<uint32_t>((N + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE);
    const uint32_t padded_n = num_tiles * ELEMS_PER_TILE;
    MeansSoa s;
    s.num_tiles = num_tiles;
    s.padded_n = padded_n;
    s.mx.resize(padded_n);
    s.my.resize(padded_n);
    s.mz.resize(padded_n);

    auto& pool = soa_pool();
    const std::size_t W = pool.size();
    const std::size_t chunk = (N + W - 1) / W;
    for (std::size_t w = 0; w < W; ++w) {
        pool.submit([w, chunk, N, means, &s]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            for (std::size_t i = lo; i < hi; ++i) {
                s.mx[i] = means[i * 3 + 0];
                s.my[i] = means[i * 3 + 1];
                s.mz[i] = means[i * 3 + 2];
            }
        });
    }
    pool.wait();
    for (std::size_t i = N; i < padded_n; ++i) {
        s.mx[i] = 0.0f; s.my[i] = 0.0f; s.mz[i] = 0.0f;
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

static void build_program(PfwcDeviceContext& ctx, bool vis = false, bool fuse = false) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    // Task #207: with the split writer (fused program only) NCRISC reads between
    // its own chunks, so the 10 inputs get a third slot.
    const bool wsplit = fuse && env_config::pfwc_writer_split();
    const uint32_t in_depth = wsplit ? 3u : 2u;

    auto cb_fp32 = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_FP32, {{id, DataFormat::Float32}});
        c.set_page_size(id, TILE_BYTES_FP32);
        CreateCircularBuffer(program, cores, c);
    };

    // 9 input CBs (world means + cov3d unique)
    cb_fp32(CB_MX, in_depth); cb_fp32(CB_MY, in_depth); cb_fp32(CB_MZ, in_depth);
    cb_fp32(CB_C00, in_depth); cb_fp32(CB_C01, in_depth); cb_fp32(CB_C02, in_depth);
    cb_fp32(CB_C11, in_depth); cb_fp32(CB_C12, in_depth); cb_fp32(CB_C22, in_depth);
    // 8 output CBs (mean_2d, depth, cov2d, radii)
    cb_fp32(CB_M2X, 2); cb_fp32(CB_M2Y, 2); cb_fp32(CB_DEP, 2);
    cb_fp32(CB_A, 2);   cb_fp32(CB_B, 2);   cb_fp32(CB_C, 2);
    cb_fp32(CB_RX, 2);  cb_fp32(CB_RY, 2);
    // 10 scratch CBs (tx, ty, tz, inv_tz, cc00..cc22)
    cb_fp32(CB_TMP_TX, 2);     cb_fp32(CB_TMP_TY, 2);     cb_fp32(CB_TMP_TZ, 2);
    cb_fp32(CB_TMP_INV_TZ, 2);
    cb_fp32(CB_TMP_CC00, 2);   cb_fp32(CB_TMP_CC01, 2);   cb_fp32(CB_TMP_CC02, 2);
    cb_fp32(CB_TMP_CC11, 2);   cb_fp32(CB_TMP_CC12, 2);   cb_fp32(CB_TMP_CC22, 2);
    // 3 conic scratch CBs (A1): cov2d a,b,c stashed, then folded to A,B,C.
    cb_fp32(CB_TMP_A, 2);      cb_fp32(CB_TMP_B, 2);      cb_fp32(CB_TMP_C, 2);
    std::map<std::string, std::string> vis_defines;
    if (vis) {
        vis_defines["PFWC_VIS"] = "1";
        if (gsplat_tt::precull_mode() >= 1 && gsplat_tt::sfpu_vis_mode() == 1)
            vis_defines["PFWC_PRECULL"] = "1";  // lever C (task #140), args 65..66
        if (gsplat_tt::precull_mode() == 2 && gsplat_tt::sfpu_vis_mode() == 1)
            vis_defines["PRECULL_PC"] = "1";  // pixel-centre rect (task #157)
        cb_fp32(CB_OP, in_depth);
        cb_fp32(CB_TMP_MX, 2);     cb_fp32(CB_TMP_MY, 2);
        cb_fp32(CB_TMP_RX, 2);     cb_fp32(CB_TMP_RY, 2);
        cb_fp32(CB_TPG, 2);        cb_fp32(CB_AABB, 2);
        auto cb_raw = [&](uint32_t id, uint32_t bytes) {
            CircularBufferConfig c(bytes, {{id, DataFormat::Float32}});
            c.set_page_size(id, bytes);
            CreateCircularBuffer(program, cores, c);
        };
        cb_raw(CB_VMASK, VIS_MASK_BYTES);
        if (!fuse) cb_raw(CB_VCNT, VIS_CNT_STAGING + 64);
        cb_raw(CB_VOP, TILE_BYTES_FP32);
        if (fuse) cb_raw(CB_FUSE, wsplit ? pfwc_wsplit::STG_BYTES : FUSE_CB_BYTES);
        if (wsplit) {
            // The odd chunks' outputs (project_pfwc_compute.cpp OCB), NCRISC's
            // staging and the chunk handoff mailbox (pfwc_wsplit.h).
            for (uint32_t cb : {CB_M2X, CB_M2Y, CB_DEP, CB_A, CB_B, CB_C, CB_RX, CB_RY, CB_TPG, CB_AABB})
                cb_fp32(pfwc_wsplit::odd_cb(cb), 2);
            cb_raw(pfwc_wsplit::CB_STG_ODD, pfwc_wsplit::STG_BYTES);
            cb_raw(pfwc_wsplit::CB_MBX, pfwc_wsplit::MBX_BYTES);
            for (uint32_t i = 0; i < pfwc_wsplit::NUM_SEMS; ++i) ctx.fsem[i] = CreateSemaphore(program, cores, 0);
        }
    }
    // Task #197: GSPLAT_TT_PFWC_STEPCYC=1|2 records per-step wall cycles (profiler builds) from the
    // reader, the three TRISCs and the fused writer (targeted profiling, default OFF).
    // GSPLAT_TT_PFWC_STEPRISC (0..2 TRISC0..2, 3 reader, 4 writer, 9 all; default 9)
    // picks the instrumented RISC(s); all five need GSPLAT_TT_KCFG_EXTRA_KB headroom.
    const uint32_t stepcyc = vis_env_u32("GSPLAT_TT_PFWC_STEPCYC", 0);
    const uint32_t steprisc = vis_env_u32("GSPLAT_TT_PFWC_STEPRISC", 9);
    std::map<std::string, std::string> reader_defines = vis_defines;
    if (stepcyc != 0) {
        vis_defines["PFWC_STEPCYC"] = std::to_string(stepcyc);
        vis_defines["PFWC_STEPRISC"] = std::to_string(steprisc);
        if (steprisc == 3 || steprisc == 9) reader_defines["PFWC_STEPCYC"] = std::to_string(stepcyc);
    }
    // Task #207: writer_pfwc_split.cpp on NCRISC (role 1: odd chunks + the reader).
    std::map<std::string, std::string> split_defines;
    if (wsplit) {
        vis_defines["PFWC_WSPLIT"] = "1";
        split_defines["WSPLIT_ROLE"] = "1";
        if (env_config::emit_puboc()) split_defines["EMIT_PUBOC"] = "1";
        if (stepcyc != 0 && (steprisc == 3 || steprisc == 9))
            split_defines["PFWC_STEPCYC"] = std::to_string(stepcyc);
    }
    // Task #206: GSPLAT_TT_PFWC_COVCAM_SFPU=1 runs cov_cam as one SFPU pass over the six
    // cov3d tiles in DEST (6 copy_tile instead of 36, no mul_unary/add_binary), same
    // rounding order, so bit-identical. Default on since task #221 (=0 off).
    if (env_config::pfwc_covcam_sfpu()) vis_defines["PFWC_COVCAM_SFPU"] = "1";

    // Reader: 9 input streams (mx,my,mz + cov3d). Same 9-stream DRAM-interleaved
    // layout as before; the fused kernel just reads world means in slots 0..2
    // (instead of the project program's means_cam output), so reader_pfwc.cpp is
    // reused verbatim — only the runtime base addresses change.
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < (wsplit ? 19 : (vis ? 10 : 9)); ++i) {  // split: writer's 9 + reader's 10
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    const KernelHandle reader = CreateKernel(
        program,
        wsplit ? OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_pfwc_split.cpp"
               : OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_pfwc.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
            .defines = wsplit ? split_defines : reader_defines,
        });

    // tt-007 fp32 unpack-to-DEST for every FP32 CB the compute kernel reads
    // (inputs + scratch). Outputs are pack-only, no unpack mode needed.
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    for (uint32_t cb : {CB_MX, CB_MY, CB_MZ,
                        CB_C00, CB_C01, CB_C02,
                        CB_C11, CB_C12, CB_C22,
                        CB_TMP_TX, CB_TMP_TY, CB_TMP_TZ, CB_TMP_INV_TZ,
                        CB_TMP_CC00, CB_TMP_CC01, CB_TMP_CC02,
                        CB_TMP_CC11, CB_TMP_CC12, CB_TMP_CC22,
                        CB_TMP_A, CB_TMP_B, CB_TMP_C}) {
        u2d[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (vis) {
        for (uint32_t cb : {CB_OP, CB_TMP_MX, CB_TMP_MY, CB_TMP_RX, CB_TMP_RY}) {
            u2d[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    // dst_full_sync_en = true disables double-buffering, which is the only way
    // to get 8 fp32 DEST tile slots (vs the default 4 with double-buffering).
    // tt-008c's cov2d/radii kernel uses 6 DEST slots in a single acquire block
    // (j00, j02, j11, j12, accumulator, scratch) so we MUST disable
    // double-buffering. Trade-off: math and pack can't overlap, but the kernel
    // is dominated by math anyway.
    const KernelHandle compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/project_pfwc_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
            .defines = vis_defines,
        });

    // Writer: 8 output streams (13 accessors for the PFWC_VIS writer: + tpg,
    // aabb, mask, counts, opacity; 9 for the lever B writer: opacity, colors,
    // the 4 compact outputs and the counts table).
    std::vector<uint32_t> writer_ct;
    for (int i = 0; i < (fuse ? 9 : (vis ? 13 : 8)); ++i) {
        TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    }
    std::map<std::string, std::string> writer_defines;
    if (fuse && env_config::emit_puboc()) writer_defines["EMIT_PUBOC"] = "1";
    if (fuse && stepcyc != 0 && (steprisc == 4 || steprisc == 9)) writer_defines["PFWC_STEPCYC"] = std::to_string(stepcyc);
    // Targeted profiling only (task #122): writer_pfwc_fuse.cpp FUSE_ABL bits.
    if (fuse && vis_env_u32("GSPLAT_TT_FUSE_ABL", 0) != 0)
        writer_defines["FUSE_ABL"] = std::to_string(vis_env_u32("GSPLAT_TT_FUSE_ABL", 0)) + "u";
    if (wsplit) {
        if (writer_defines.erase("FUSE_ABL") != 0)
            std::cerr << "[gsplat_tt::pfwc] GSPLAT_TT_FUSE_ABL is ignored with GSPLAT_TT_PFWC_WRITER_SPLIT=1\n";
        writer_defines["WSPLIT_ROLE"] = "0";
    }
    const KernelHandle writer = CreateKernel(
        program,
        wsplit  ? OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_pfwc_split.cpp"
        : fuse  ? OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_pfwc_fuse.cpp"
        : vis ? OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_pfwc_vis.cpp"
              : OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_pfwc.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
            .defines = writer_defines,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    if (fuse) {
        ctx.freader = reader;
        ctx.fcompute = compute;
        ctx.fwriter = writer;
        ctx.fuse_wsplit = wsplit;
        ctx.wl_fuse.add_program(device_range, std::move(program));
        ctx.fuse_built = true;
    } else if (vis) {
        ctx.vreader = reader;
        ctx.vcompute = compute;
        ctx.vwriter = writer;
        ctx.wl_vis.add_program(device_range, std::move(program));
        ctx.vis_built = true;
    } else {
        ctx.reader = reader;
        ctx.compute = compute;
        ctx.writer = writer;
        ctx.workload.add_program(device_range, std::move(program));
    }
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

static bool g_pfwc_ran_vis = false;
static bool g_pfwc_ran_fused = false;
static PfwcFuseInfo g_fuse_info;

bool pfwc_ran_vis() { return g_pfwc_ran_vis; }

bool pfwc_ran_fused(PfwcFuseInfo* info) {
    if (g_pfwc_ran_fused && info != nullptr) *info = g_fuse_info;
    return g_pfwc_ran_fused;
}

bool pfwc_device_ready() {
    return ensure_context() != nullptr;
}

void pfwc_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        (void)slot.release();
    }
}

double pfwc_tt(
    const float* means,
    const float* cov3d_unique,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    float* mean_2d_out,
    float* depth_out,
    float* cov2d_out,
    float* radii_out,
    PfwcCallTimings* timings_out,
    const PfwcVisParams* vis) {
    if (N == 0) {
        return 0.0;
    }
    // Stage-timer sub-buckets of `project` (stage_timers.h): setup runs until
    // the per-core arg loop, then rtargs / enqueue / finish.
    auto& st_acc = stagetimers::acc();
    stagetimers::Span setup_span(st_acc.project_pfwc_setup);
    g_pfwc_ran_vis = false;
    g_pfwc_ran_fused = false;
    auto* ctx = ensure_context();
    if (ctx == nullptr) {
        return -1.0;
    }
    PfwcCallTimings tlocal;
    auto& T = (timings_out ? *timings_out : tlocal);

    const float r[9] = {
        extrinsics[0],  extrinsics[1],  extrinsics[2],
        extrinsics[4],  extrinsics[5],  extrinsics[6],
        extrinsics[8],  extrinsics[9],  extrinsics[10],
    };
    const float t0 = extrinsics[3];
    const float t1 = extrinsics[7];
    const float t2 = extrinsics[11];

    const float fx = intrinsics[0];
    const float fy = intrinsics[4];
    const float cx = intrinsics[2];
    const float cy = intrinsics[5];

    const uint32_t num_tiles =
        static_cast<uint32_t>((N + ELEMS_PER_TILE - 1) / ELEMS_PER_TILE);
    const uint32_t padded_n = num_tiles * ELEMS_PER_TILE;
    const std::size_t soa_bytes = static_cast<std::size_t>(padded_n) * sizeof(float);

    // Allocate or grow world-means DRAM buffers (iter-133 fusion input).
    if (!ctx->buf_mx || ctx->means_cached_bytes < soa_bytes) {
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = TILE_BYTES_FP32, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig rep_cfg{.size = soa_bytes};
        ctx->buf_mx = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_my = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_mz = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->means_cached_bytes = soa_bytes;
        ctx->uploaded_means_ptr = nullptr;
        ctx->uploaded_means_N = 0;
    }

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
        ctx->buf_m2x = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_m2y = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_dep = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_a   = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_b   = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_c   = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_rx  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->buf_ry  = distributed::MeshBuffer::create(rep_cfg, dram_cfg, ctx->mesh_device.get());
        ctx->output_cached_bytes = soa_bytes;

        device_state::register_buffer("pfwc_m2x", ctx->buf_m2x);
        device_state::register_buffer("pfwc_m2y", ctx->buf_m2y);
        device_state::register_buffer("pfwc_depth", ctx->buf_dep);
        device_state::register_buffer("pfwc_a", ctx->buf_a);
        device_state::register_buffer("pfwc_b", ctx->buf_b);
        device_state::register_buffer("pfwc_c", ctx->buf_c);
        device_state::register_buffer("pfwc_rx", ctx->buf_rx);
        device_state::register_buffer("pfwc_ry", ctx->buf_ry);
    }

    // world-means upload (cached across views — fires on frame 0 only, since
    // means_3d is a single tensor reused for all 30 views).
    const bool means_hit =
        ctx->uploaded_means_ptr == means && ctx->uploaded_means_N == N;
    if (!means_hit) {
        const auto m_pack0 = std::chrono::high_resolution_clock::now();
        const MeansSoa msoa = pack_means_soa(means, N);
        const auto m_pack1 = std::chrono::high_resolution_clock::now();
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_mx, msoa.mx, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_my, msoa.my, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_mz, msoa.mz, /*blocking=*/false);
        const auto m_pack2 = std::chrono::high_resolution_clock::now();
        T.pack_ms += std::chrono::duration<double, std::milli>(m_pack1 - m_pack0).count();
        T.upload_ms += std::chrono::duration<double, std::milli>(m_pack2 - m_pack1).count();
        ctx->uploaded_means_ptr = means;
        ctx->uploaded_means_N = N;
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
        T.pack_ms += std::chrono::duration<double, std::milli>(t_pack1 - t_pack0).count();
        T.upload_ms += std::chrono::duration<double, std::milli>(t_pack2 - t_pack1).count();
        ctx->uploaded_cov3d_ptr = cov3d_unique;
        ctx->uploaded_cov3d_N = N;
    }

    // iter-133 fusion: world means feed the reader directly (the fused compute
    // kernel runs the means_cam transform in L1). No separate project program /
    // means_cam DRAM round-trip.
    auto& buf_mx = ctx->buf_mx;
    auto& buf_my = ctx->buf_my;
    auto& buf_mz = ctx->buf_mz;

    std::vector<uint32_t> cc_scales;
    pack_cc_scales(r, cc_scales);  // 36 fp32 bits

    const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
    const WorkSplit ws = split_chunks(num_tiles, num_cores);

    // Lever 2 (task #99): the PFWC_VIS variant also needs the resident scene
    // opacities and its four extra outputs.
    std::shared_ptr<distributed::MeshBuffer> vis_op;
    if (vis != nullptr) {
        vis_op = device_state::get_buffer("scene_opacities");
        if (!vis_op) {
            // Keep the frame correct: run the legacy program (gather then takes
            // its legacy path too, pfwc_ran_vis() == false).
            std::cerr << "[gsplat_tt::pfwc] PFWC_VIS needs the resident scene_opacities "
                         "(gather_visible_upload_scene); running the legacy program\n";
            vis = nullptr;
        }
    }
    // Lever B (task #125, GSPLAT_TT_PFWC_FUSE=1): the fused writer compacts
    // straight into the gather's outputs (pfwc_fuse.h). Anything missing before
    // the enqueue drops back to the lever 2 program.
    bool fuse_on = false;
    if (vis != nullptr && vis->fuse) {
        if (!gather_visible_fuse_prepare(N, num_cores) || !device_state::get_buffer("scene_col_r") ||
            !device_state::get_buffer("scene_col_g") || !device_state::get_buffer("scene_col_b")) {
            std::cerr << "[gsplat_tt::pfwc] GSPLAT_TT_PFWC_FUSE=1: gather outputs not ready; "
                         "running the PFWC_VIS program\n";
        } else if (num_cores > pfwc_fuse::MAX_SEG) {
            std::cerr << "[gsplat_tt::pfwc] GSPLAT_TT_PFWC_FUSE=1: " << num_cores
                      << " cores over the segment table; running the PFWC_VIS program\n";
        } else {
            try {
                if (!ctx->fuse_built) build_program(*ctx, /*vis=*/true, /*fuse=*/true);
                fuse_on = true;
            } catch (const std::exception& e) {
                std::cerr << "[gsplat_tt::pfwc] fused program build failed (" << e.what()
                          << "); running the PFWC_VIS program\n";
            }
        }
    }
    if (vis != nullptr && !fuse_on) {
        // The PFWC_VIS writer stages the 1 KB count pages spanned by its tile
        // range in CB_VCNT (task #102); a range that outgrows it would overrun L1.
        constexpr uint32_t CTILES = VIS_COUNTS_PAGE / (2 * sizeof(uint32_t));  // COUNT_WORDS = 2
        for (uint32_t c = 0; c < num_cores; ++c) {
            if (ws.num_chunks[c] == 0) continue;
            const uint32_t p0 = ws.chunk_start[c] / CTILES;
            const uint32_t p1 = (ws.chunk_start[c] + ws.num_chunks[c] - 1) / CTILES;
            if ((p1 - p0 + 1) * VIS_COUNTS_PAGE > VIS_CNT_STAGING)
                throw std::runtime_error(
                    "[gsplat_tt::pfwc] PFWC_VIS core " + std::to_string(c) + " spans " +
                    std::to_string(p1 - p0 + 1) + " count pages (" +
                    std::to_string(ws.num_chunks[c]) + " tiles), over the CB_VCNT staging of " +
                    std::to_string(VIS_CNT_STAGING) + " B");
        }
        if (!ctx->vis_built) build_program(*ctx, /*vis=*/true);
        if (!ctx->buf_vtpg || ctx->vis_cap_tiles < num_tiles) {
            distributed::DeviceLocalBufferConfig tile_cfg{
                .page_size = TILE_BYTES_FP32, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig tile_rep{.size = soa_bytes};
            ctx->buf_vtpg = distributed::MeshBuffer::create(tile_rep, tile_cfg, ctx->mesh_device.get());
            ctx->buf_vaabb = distributed::MeshBuffer::create(tile_rep, tile_cfg, ctx->mesh_device.get());
            distributed::DeviceLocalBufferConfig mask_cfg{
                .page_size = VIS_MASK_BYTES, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig mask_rep{
                .size = static_cast<std::size_t>(num_tiles) * VIS_MASK_BYTES};
            ctx->buf_vmask = distributed::MeshBuffer::create(mask_rep, mask_cfg, ctx->mesh_device.get());
            const uint32_t cpages = (num_tiles * 8u + VIS_COUNTS_PAGE - 1) / VIS_COUNTS_PAGE;
            distributed::DeviceLocalBufferConfig cnt_cfg{
                .page_size = VIS_COUNTS_PAGE, .buffer_type = BufferType::DRAM};
            distributed::ReplicatedBufferConfig cnt_rep{
                .size = static_cast<std::size_t>(cpages) * VIS_COUNTS_PAGE};
            ctx->buf_vcounts = distributed::MeshBuffer::create(cnt_rep, cnt_cfg, ctx->mesh_device.get());
            ctx->vis_cap_tiles = num_tiles;
            device_state::register_buffer("pfwc_tpg", ctx->buf_vtpg);
            device_state::register_buffer("pfwc_aabb", ctx->buf_vaabb);
            device_state::register_buffer("pfwc_vis_mask", ctx->buf_vmask);
            device_state::register_buffer("pfwc_tile_counts", ctx->buf_vcounts);
        }
    }
    const bool vis_on = (vis != nullptr);
    g_pfwc_ran_vis = vis_on && !fuse_on;
    distributed::MeshWorkload& wl =
        fuse_on ? ctx->wl_fuse : (vis_on ? ctx->wl_vis : ctx->workload);
    Program& program = wl.get_programs().begin()->second;
    const KernelHandle k_reader = fuse_on ? ctx->freader : (vis_on ? ctx->vreader : ctx->reader);
    const KernelHandle k_compute =
        fuse_on ? ctx->fcompute : (vis_on ? ctx->vcompute : ctx->compute);
    const KernelHandle k_writer = fuse_on ? ctx->fwriter : (vis_on ? ctx->vwriter : ctx->writer);
    // Fused: tiles dealt strided (core c owns c, c + C, ...), the legacy
    // compaction order (vis_tile::SeqMap).
    vis_tile::SeqMap seq;
    seq.init(num_tiles, num_cores);
    auto fbuf = [](const char* name) {
        auto b = device_state::get_buffer(name);
        if (!b) throw std::runtime_error(std::string("[gsplat_tt::pfwc] missing ") + name);
        return static_cast<uint32_t>(b->address());
    };
    std::vector<uint32_t> fuse_addr;
    if (fuse_on) {
        fuse_addr = {static_cast<uint32_t>(vis_op->address()), fbuf("scene_col_r"),
                     fbuf("scene_col_g"),  fbuf("scene_col_b"),
                     fbuf("proj_m_depth"), fbuf("proj_m_blendrec"),
                     fbuf("proj_m_offs"),  fbuf("proj_m_aabb"),
                     fbuf("pfwc_fuse_counts")};
    }
    // Per-scene UNORM16 packs (task #122); absent = the writer packs on device.
    uint32_t fuse_pub[2] = {0u, 0u};
    if (fuse_on && env_config::emit_puboc()) {
        const auto q01 = device_state::get_buffer("scene_puboc01");
        const auto q23 = device_state::get_buffer("scene_puboc23");
        if (q01 && q23) {
            fuse_pub[0] = static_cast<uint32_t>(q01->address());
            fuse_pub[1] = static_cast<uint32_t>(q23->address());
        }
    }
    std::vector<uint32_t> vis_bits;
    if (vis_on) {
        vis_bits = {
            fp32_bits(vis->k_near),
            fp32_bits(vis->min_opacity),
            fp32_bits(vis->image_width),
            fp32_bits(vis->image_height),
            fp32_bits(vis->max_radius),
            fp32_bits(1.0f / static_cast<float>(vis->tile_size)),
            fp32_bits(static_cast<float>(vis->tiles_x - 1)),
            fp32_bits(static_cast<float>(vis->tiles_y - 1)),
            fp32_bits(vis->edge_tau),
        };
        // Lever C (task #140): args 65 (c0 = 2 ln(1 / floor) + margin, plus
        // 2 ln 2 (0.0860713 + 2^-14 - 127), the constant of the kernel's log2
        // upper bound (task #142; tests/unit/test_precull.cpp)) and 66 (radius
        // limit; -1 = no shrink on any lane). Read only by PFWC_PRECULL.
        constexpr float PRECULL_T_MARGIN = 0.25f;  // vs the band cull's 0.05
        constexpr float PRECULL_RMAX = 4096.0f;
        const bool pc = vis->precull_floor > 0.0f && vis->precull_floor < 1.0f;
        const double c0 = 2.0 * std::log(1.0 / vis->precull_floor) + PRECULL_T_MARGIN +
                          2.0 * std::log(2.0) * (0.0860713 + 1.0 / 16384.0 - 127.0);
        vis_bits.push_back(fp32_bits(pc ? static_cast<float>(c0) : 0.0f));
        vis_bits.push_back(fp32_bits(pc ? std::min(vis->max_radius, PRECULL_RMAX) : -1.0f));
    }

    // Pre-pack k = 3.0, -fx, -fy as fp32 bits — match project_full_fused k_cap.
    constexpr float K_RADII = 3.0f;
    const uint32_t k_bits      = fp32_bits(K_RADII);
    const uint32_t neg_fx_bits = fp32_bits(-fx);
    const uint32_t neg_fy_bits = fp32_bits(-fy);

    setup_span.stop();
    gsplat_tt::hostprof::on_pfwc_dispatch_start();
    const auto t_launch0 = std::chrono::high_resolution_clock::now();
    stagetimers::Span rtargs_span(st_acc.project_pfwc_rtargs);
    for (uint32_t c = 0; c < num_cores; ++c) {
        CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
        const uint32_t chunk_start = fuse_on ? c : ws.chunk_start[c];
        const uint32_t num_chunks  = fuse_on ? seq.count(c) : ws.num_chunks[c];

        std::vector<uint32_t> reader_args = {
             static_cast<uint32_t>(buf_mx->address()),
             static_cast<uint32_t>(buf_my->address()),
             static_cast<uint32_t>(buf_mz->address()),
             static_cast<uint32_t>(ctx->buf_c00->address()),
             static_cast<uint32_t>(ctx->buf_c01->address()),
             static_cast<uint32_t>(ctx->buf_c02->address()),
             static_cast<uint32_t>(ctx->buf_c11->address()),
             static_cast<uint32_t>(ctx->buf_c12->address()),
             static_cast<uint32_t>(ctx->buf_c22->address()),
             chunk_start, num_chunks};
        if (vis_on) {
            reader_args.push_back(static_cast<uint32_t>(vis_op->address()));  // arg 11
            reader_args.push_back(fuse_on ? num_cores : 1u);  // arg 12: tile stride
        }
        const bool wsplit_on = fuse_on && ctx->fuse_wsplit;
        if (!wsplit_on) SetRuntimeArgs(program, k_reader, core, reader_args);

        std::vector<uint32_t> compute_args;
        compute_args.reserve(56);
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
        compute_args.push_back(k_bits);        // arg 53
        compute_args.push_back(neg_fx_bits);   // arg 54
        compute_args.push_back(neg_fy_bits);   // arg 55
        for (uint32_t b : vis_bits) compute_args.push_back(b);  // args 56..66 (PFWC_VIS)
        SetRuntimeArgs(program, k_compute, core, compute_args);

        if (fuse_on) {
            // writer_pfwc_fuse.cpp args 0..24.
            std::vector<uint32_t> fw = fuse_addr;
            fw.push_back(c);
            fw.push_back(num_chunks);
            fw.push_back(num_cores);
            fw.push_back(static_cast<uint32_t>(N));
            for (uint32_t k = 0; k < 5; ++k) fw.push_back(vis_bits[k]);
            fw.push_back(static_cast<uint32_t>(vis->tiles_x));
            fw.push_back(static_cast<uint32_t>(vis->tiles_y));
            fw.push_back(static_cast<uint32_t>(vis->tile_size));
            fw.push_back(pfwc_fuse::seg_base(num_tiles, num_cores, c));
            fw.push_back(c);
            fw.push_back(fuse_pub[0]);  // scene_puboc01 (0 = pack on device)
            fw.push_back(fuse_pub[1]);  // scene_puboc23
            if (wsplit_on) {
                // writer_pfwc_split.cpp: 25..28 the mailbox semaphores; NCRISC
                // (the reader too): 29..38 the reader's 9 bases and the opacity.
                for (uint32_t sid : ctx->fsem) fw.push_back(sid);
                std::vector<uint32_t> nw = fw;
                for (uint32_t k = 0; k < 9; ++k) nw.push_back(reader_args[k]);
                nw.push_back(static_cast<uint32_t>(vis_op->address()));
                SetRuntimeArgs(program, k_reader, core, nw);
            }
            SetRuntimeArgs(program, k_writer, core, fw);
            continue;
        }
        std::vector<uint32_t> writer_args = {
             static_cast<uint32_t>(ctx->buf_m2x->address()),
             static_cast<uint32_t>(ctx->buf_m2y->address()),
             static_cast<uint32_t>(ctx->buf_dep->address()),
             static_cast<uint32_t>(ctx->buf_a->address()),
             static_cast<uint32_t>(ctx->buf_b->address()),
             static_cast<uint32_t>(ctx->buf_c->address()),
             static_cast<uint32_t>(ctx->buf_rx->address()),
             static_cast<uint32_t>(ctx->buf_ry->address())};
        if (vis_on) {
            // writer_pfwc_vis.cpp: + tpg, aabb, mask, counts, opacity, then the
            // chunk range, N and the predicate / grid parameters.
            writer_args.push_back(static_cast<uint32_t>(ctx->buf_vtpg->address()));
            writer_args.push_back(static_cast<uint32_t>(ctx->buf_vaabb->address()));
            writer_args.push_back(static_cast<uint32_t>(ctx->buf_vmask->address()));
            writer_args.push_back(static_cast<uint32_t>(ctx->buf_vcounts->address()));
            writer_args.push_back(static_cast<uint32_t>(vis_op->address()));
            writer_args.push_back(chunk_start);
            writer_args.push_back(num_chunks);
            writer_args.push_back(static_cast<uint32_t>(N));
            for (uint32_t k = 0; k < 5; ++k) writer_args.push_back(vis_bits[k]);
            writer_args.push_back(static_cast<uint32_t>(vis->tiles_x));
            writer_args.push_back(static_cast<uint32_t>(vis->tiles_y));
            writer_args.push_back(static_cast<uint32_t>(vis->tile_size));
        } else {
            writer_args.push_back(chunk_start);
            writer_args.push_back(num_chunks);
        }
        SetRuntimeArgs(program, k_writer, core, writer_args);
    }
    rtargs_span.stop();
    {
        stagetimers::Span s(st_acc.project_pfwc_enqueue);
        distributed::EnqueueMeshWorkload(*ctx->cq, wl, /*blocking=*/false);
    }
    gsplat_tt::hostprof::on_pfwc_enqueued();
    const auto t_launch1 = std::chrono::high_resolution_clock::now();
    if (fuse_on) {
        // No host sync: the segment K2 (tile_assign_fused_k2) queues behind
        // this program and its proj_M read is the stage's one blocking read.
        g_pfwc_ran_fused = true;
        g_fuse_info.num_cores = num_cores;
        g_fuse_info.num_tiles = num_tiles;
        g_fuse_info.tiles_x = static_cast<uint32_t>(vis->tiles_x);
        g_fuse_info.screen_tiles = static_cast<uint32_t>(vis->tiles_x * vis->tiles_y);
        T.launch_ms = std::chrono::duration<double, std::milli>(t_launch1 - t_launch0).count();
        T.compute_ms = 0.0;
        T.download_ms = 0.0;
        T.unpack_ms = 0.0;
        if (mean_2d_out || depth_out || cov2d_out || radii_out)
            throw std::runtime_error("[gsplat_tt::pfwc] the fused program writes no pfwc outputs");
        return T.launch_ms;
    }
    {
        stagetimers::Span s(st_acc.project_pfwc_finish);
        distributed::Finish(*ctx->cq);
    }
    const auto t_launch_end = std::chrono::high_resolution_clock::now();
    T.launch_ms = std::chrono::duration<double, std::milli>(t_launch1 - t_launch0).count();
    T.compute_ms = std::chrono::duration<double, std::milli>(t_launch_end - t_launch1).count();

    // Skip D2H + unpack when all caller pointers are null (full device-resident
    // mode for tt-006 tile_assign etc.).
    const bool any_download =
        (mean_2d_out != nullptr) ||
        (depth_out   != nullptr) ||
        (cov2d_out   != nullptr) ||
        (radii_out   != nullptr);
    if (!any_download) {
        T.download_ms = 0.0;
        T.unpack_ms = 0.0;
        return std::chrono::duration<double, std::milli>(t_launch_end - t_launch0).count();
    }

    // Selective D2H: each stream only downloads when its caller buffer is non-null.
    // Outputs already in SoA → unpack to AoS per stream group.
    const auto t_dl0 = std::chrono::high_resolution_clock::now();
    std::vector<float> m2x_out, m2y_out, dep_out;
    std::vector<float> a_out, b_out, c_out;
    std::vector<float> rx_out, ry_out;
    if (mean_2d_out != nullptr) {
        m2x_out.resize(padded_n);
        m2y_out.resize(padded_n);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, m2x_out, ctx->buf_m2x, /*blocking=*/true);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, m2y_out, ctx->buf_m2y, /*blocking=*/true);
    }
    if (depth_out != nullptr) {
        dep_out.resize(padded_n);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, dep_out, ctx->buf_dep, /*blocking=*/true);
    }
    if (cov2d_out != nullptr) {
        a_out.resize(padded_n);
        b_out.resize(padded_n);
        c_out.resize(padded_n);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, a_out, ctx->buf_a, /*blocking=*/true);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, b_out, ctx->buf_b, /*blocking=*/true);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, c_out, ctx->buf_c, /*blocking=*/true);
    }
    if (radii_out != nullptr) {
        rx_out.resize(padded_n);
        ry_out.resize(padded_n);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, rx_out, ctx->buf_rx, /*blocking=*/true);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, ry_out, ctx->buf_ry, /*blocking=*/true);
    }
    const auto t_dl1 = std::chrono::high_resolution_clock::now();
    T.download_ms = std::chrono::duration<double, std::milli>(t_dl1 - t_dl0).count();

    // Unpack SoA → caller AoS.
    {
        auto& pool = soa_pool();
        const std::size_t W = pool.size();
        const std::size_t chunk = (N + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool.submit([w, chunk, N,
                         &m2x_out, &m2y_out, &dep_out,
                         &a_out, &b_out, &c_out,
                         &rx_out, &ry_out,
                         mean_2d_out, depth_out, cov2d_out, radii_out]() {
                const std::size_t lo = std::min(w * chunk, N);
                const std::size_t hi = std::min(lo + chunk, N);
                for (std::size_t i = lo; i < hi; ++i) {
                    if (mean_2d_out != nullptr) {
                        mean_2d_out[i * 2 + 0] = m2x_out[i];
                        mean_2d_out[i * 2 + 1] = m2y_out[i];
                    }
                    if (depth_out != nullptr) {
                        depth_out[i] = dep_out[i];
                    }
                    if (cov2d_out != nullptr) {
                        cov2d_out[i * 3 + 0] = a_out[i];
                        cov2d_out[i * 3 + 1] = b_out[i];
                        cov2d_out[i * 3 + 2] = c_out[i];
                    }
                    if (radii_out != nullptr) {
                        radii_out[i * 2 + 0] = rx_out[i];
                        radii_out[i * 2 + 1] = ry_out[i];
                    }
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
