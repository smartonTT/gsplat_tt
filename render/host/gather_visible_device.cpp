// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt gather_visible — residency pass R2.
//
// See gather_visible.h for the contract. This driver:
//   * uploads the scene colors (SoA r/g/b) + opacities once per scene and
//     registers them resident (scene_col_r/g/b, scene_opacities);
//   * allocates + registers the M-compact proj_m_* DRAM buffers;
//   * runs a multi-core device gather: a count pass (sizes M exactly) then a
//     scatter kernel that fills proj_m_* on-device, reads M back, and
//     reassembles the M-compact ProjectResult. Single-path: there is no host
//     gather and no CPU-reference verify (those were removed for render_clean).

#include "gather_visible.h"
#include "device_state.h"
#include "env_config.h"
#include "host_tracy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <thread>
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

constexpr uint32_t TILE_ELEMS = 1024;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4;   // 4096
constexpr uint32_t PAGE_ELEMS = 16;
constexpr uint32_t PAGE_BYTES = PAGE_ELEMS * 4;   // 64
constexpr uint32_t COLOR_ACC_BYTES = PAGE_ELEMS * 3 * 4;  // 192

inline uint32_t round_up(uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; }

// S1 (GSPLAT_TT_BLEND_AOS): also emit a contiguous Array-of-Structs blend
// record (proj_m_blendrec) so the blend reader fetches a candidate with ONE
// 64B read instead of 7-9 random SoA pages. Default ON (iter21: measured
// gate-clean ~6ms blend drop, bit-identical PSNR); set GSPLAT_TT_BLEND_AOS=0
// to fall back to the per-component SoA gather.
inline bool blend_aos_enabled() { return true; }  // BLEND_AOS default-on

// GSPLAT_TT_PROJ_BALANCE: replace the contiguous N-tile split of the gather
// passes (core c owns tiles [c*K, (c+1)*K)) with an INTERLEAVED/STRIDED split
// (core c owns tiles c, c+num_cores, c+2*num_cores, ...). The gather reads are
// whole 4096B tile pages either way (no DRAM-coalescing loss from striding —
// each tile read is already one page), but the per-core SCATTER write work
// scales with that core's visible-fraction. The .ply gaussian order is
// spatially clustered, so contiguous tile chunks have very different visible
// fractions (sky chunks ~0 writes, foliage chunks dense) -> makespan skew:
// measured min/core 9 235, max/core 36 300, mean 17 126 = max/mean 2.12.
// Striding hands every core a representative spatial sample -> balanced writes
// (skew drops to ~1.21x, gather kernel 51.6->36.6 ms, proj -4.6 ms @8v) at
// identical hero PSNR. Reordering the compact gid space is safe: downstream
// tile_assign/sort/blend index proj_m_* by the compact id and order by depth,
// so the absolute compaction order is an arbitrary relabel. DEFAULT ON
// (verified net win on a shared production stage); disable with =0.
inline bool proj_balance_enabled() { return true; }  // PROJ_BALANCE default-on

// GSPLAT_TT_PROJ_DEVICE_SCAN: fuse the project stage's count->scan->scatter into
// ONE in-order CQ submission with NO inter-kernel Finish and NO mid-chain host M
// read. An on-device single-core exclusive prefix-sum (gather_scan_bases.cpp)
// computes the per-core compact base offsets + publishes proj_M, so the host
// never D2H's the 110 per-core counts, never scans on the host, and never writes
// proj_M. The outputs are pre-sized to the per-frame safe ceiling padded_n (M <=
// N), so the scatter can be dispatched WITHOUT first reading M; the project
// stage's only host sync is one post-chain 1-page M read (which, in the resident
// chain, also replaces the unused cap-sized depth D2H — see
// readback_proj_m_count_only). This collapses project's 3 Finish drains + 2 D2H
// + host scan to a single 1-page read. DEFAULT ON (iter-37: verified net win on
// the shared production stage — proj 32.4->28.3 ms, ms/view 173.2->169.6 @8v,
// 63.85 dB bit-identical); disable with GSPLAT_TT_PROJ_DEVICE_SCAN=0.
inline bool proj_device_scan_enabled() { return true; }  // PROJ_DEVICE_SCAN=1

inline uint32_t fp32_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(uint32_t));
    return u;
}

struct WorkSplit {
    std::vector<uint32_t> chunk_start;  // first tile per core
    std::vector<uint32_t> num_chunks;   // tile count per core
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

// Strided (interleaved) split: core c owns tiles {c, c+num_cores, c+2*nc, ...}.
// chunk_start[c] = c (first tile), num_chunks[c] = how many strided tiles it
// owns. Paired with t_stride = num_cores in the kernel loop.
static WorkSplit split_strided(uint32_t num_tiles, uint32_t num_cores) {
    WorkSplit ws;
    ws.chunk_start.assign(num_cores, 0);
    ws.num_chunks.assign(num_cores, 0);
    for (uint32_t c = 0; c < num_cores; ++c) {
        ws.chunk_start[c] = c;  // first strided tile (== c, harmless if c>=num_tiles: count 0)
        ws.num_chunks[c] =
            (c < num_tiles) ? ((num_tiles - 1 - c) / num_cores + 1) : 0U;
    }
    return ws;
}

struct GatherDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle kernel{};
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;
    uint32_t num_cores = 0;

    // GSPLAT_TT_PROJ_DEVICE_SCAN: single-core on-device exclusive prefix-sum of
    // the per-core counts (scan_bases-style). Replaces the host D2H(counts) +
    // host scan + host proj_M write between the count and scatter passes.
    distributed::MeshWorkload wl_scan;
    KernelHandle kscan{};
    // Per-core exclusive bases ([0]=base, [1]=is_last); one 64B page per core.
    std::shared_ptr<distributed::MeshBuffer> buf_core_base;

    // Per-core visible counts (one 64B page per core; grow-on-demand never:
    // num_cores is fixed for the device).
    std::shared_ptr<distributed::MeshBuffer> buf_counts;

    // Scene inputs (uploaded once per scene, cached by pointer + N).
    std::shared_ptr<distributed::MeshBuffer> buf_cr;
    std::shared_ptr<distributed::MeshBuffer> buf_cg;
    std::shared_ptr<distributed::MeshBuffer> buf_cb;
    std::shared_ptr<distributed::MeshBuffer> buf_op;
    std::size_t scene_cached_bytes = 0;
    const float* uploaded_colors_ptr = nullptr;
    const float* uploaded_opacities_ptr = nullptr;
    std::size_t uploaded_N = 0;

    // M-compact resident outputs (grow-on-demand).
    std::shared_ptr<distributed::MeshBuffer> buf_px;
    std::shared_ptr<distributed::MeshBuffer> buf_py;
    std::shared_ptr<distributed::MeshBuffer> buf_rx;
    std::shared_ptr<distributed::MeshBuffer> buf_ry;
    std::shared_ptr<distributed::MeshBuffer> buf_a;
    std::shared_ptr<distributed::MeshBuffer> buf_b;
    std::shared_ptr<distributed::MeshBuffer> buf_c;
    std::shared_ptr<distributed::MeshBuffer> buf_depth;
    std::shared_ptr<distributed::MeshBuffer> buf_opacity;
    std::shared_ptr<distributed::MeshBuffer> buf_colors;  // AoS M*3
    std::shared_ptr<distributed::MeshBuffer> buf_blendrec; // S1: AoS blend record, 64B/gaussian
    std::shared_ptr<distributed::MeshBuffer> buf_M;       // uint32 count
    std::size_t cap_m_elems = 0;
};

static std::shared_ptr<distributed::MeshBuffer> make_dram(
    distributed::MeshDevice* dev, std::size_t bytes, uint32_t page_bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = page_bytes, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

static void build_program(GatherDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb = [&](uint32_t id, uint32_t total_bytes, uint32_t page_bytes) {
        CircularBufferConfig c(total_bytes, {{id, DataFormat::Float32}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };
    // 12 input tile CBs.
    for (uint32_t id = 0; id < 12; id++) cb(id, TILE_BYTES, TILE_BYTES);
    // 9 SoA output accumulators (64B), colors accumulator (192B), M (64B).
    for (uint32_t id = 12; id < 21; id++) cb(id, PAGE_BYTES, PAGE_BYTES);
    cb(21, COLOR_ACC_BYTES, COLOR_ACC_BYTES);
    cb(22, PAGE_BYTES, PAGE_BYTES);

    // S1 AoS blend-record staging: 16 records x 64B = 1024B. The kernel always
    // emits the AoS record (GATHER_EMIT_BLENDREC is inlined), so this CB and the
    // matching accessor are unconditional.
    cb(23, PAGE_ELEMS * PAGE_BYTES, PAGE_ELEMS * PAGE_BYTES);

    std::vector<uint32_t> ct;
    constexpr int n_acc = 25;  // 24 base + 1 AoS blend-record accessor
    for (int i = 0; i < n_acc; i++)
        TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    std::map<std::string, std::string> defines;
    ctx.kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/gather_visible_scatter.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
            .defines = defines,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

// Single-core on-device exclusive prefix-sum of the per-core counts. Two 64B CBs
// (in/out staging) + 3 DRAM-interleaved accessors (counts, base, proj_M).
static void build_program_scan(GatherDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet core0(CoreRange({0, 0}, {0, 0}));
    auto cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::Float32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, core0, c);
    };
    cb(0);
    cb(1);
    std::vector<uint32_t> ct;
    for (int i = 0; i < 3; i++)
        TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kscan = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/gather_scan_bases.cpp",
        core0,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_scan.add_program(device_range, std::move(program));
}

static GatherDeviceContext init_context() {
    GatherDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.num_cores = ctx.grid.x * ctx.grid.y;
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    build_program_scan(ctx);
    return ctx;
}

static std::unique_ptr<GatherDeviceContext>& context_slot() {
    static std::unique_ptr<GatherDeviceContext> ctx;
    return ctx;
}

static GatherDeviceContext* ensure_context() {
    auto& slot = context_slot();
    if (!slot) {
        try {
            slot = std::make_unique<GatherDeviceContext>(init_context());
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::gather] device init failed: " << e.what() << "\n";
            slot.reset();
        }
    }
    return slot.get();
}

static gsplat_cpu::ThreadPool& soa_pool() {
    static gsplat_cpu::ThreadPool pool(
        static_cast<std::size_t>(std::max(2u, std::thread::hardware_concurrency())));
    return pool;
}

// Upload scene colors (SoA r/g/b) + opacities tile-padded; cache by pointer.
static void ensure_scene_uploaded(
    GatherDeviceContext* ctx, const float* colors, const float* opacities,
    std::size_t N, uint32_t num_tiles, double* upload_ms, bool* cache_hit) {
    const uint32_t padded_n = num_tiles * TILE_ELEMS;
    const std::size_t soa_bytes = static_cast<std::size_t>(padded_n) * 4;
    if (!ctx->buf_cr || ctx->scene_cached_bytes < soa_bytes) {
        ctx->buf_cr = make_dram(ctx->mesh_device.get(), soa_bytes, TILE_BYTES);
        ctx->buf_cg = make_dram(ctx->mesh_device.get(), soa_bytes, TILE_BYTES);
        ctx->buf_cb = make_dram(ctx->mesh_device.get(), soa_bytes, TILE_BYTES);
        ctx->buf_op = make_dram(ctx->mesh_device.get(), soa_bytes, TILE_BYTES);
        ctx->scene_cached_bytes = soa_bytes;
        ctx->uploaded_colors_ptr = nullptr;
        ctx->uploaded_opacities_ptr = nullptr;
        device_state::register_buffer("scene_col_r", ctx->buf_cr);
        device_state::register_buffer("scene_col_g", ctx->buf_cg);
        device_state::register_buffer("scene_col_b", ctx->buf_cb);
        device_state::register_buffer("scene_opacities", ctx->buf_op);
    }
    const bool hit = (ctx->uploaded_colors_ptr == colors) &&
                     (ctx->uploaded_opacities_ptr == opacities) &&
                     (ctx->uploaded_N == N);
    if (cache_hit) *cache_hit = hit;
    if (hit) return;

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> cr(padded_n, 0.0f), cg(padded_n, 0.0f),
        cbv(padded_n, 0.0f), op(padded_n, 0.0f);
    {
        auto& pool = soa_pool();
        const std::size_t W = pool.size();
        const std::size_t chunk = (N + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool.submit([w, chunk, N, colors, opacities, &cr, &cg, &cbv, &op]() {
                const std::size_t lo = std::min(w * chunk, N);
                const std::size_t hi = std::min(lo + chunk, N);
                for (std::size_t i = lo; i < hi; ++i) {
                    cr[i] = colors[i * 3 + 0];
                    cg[i] = colors[i * 3 + 1];
                    cbv[i] = colors[i * 3 + 2];
                    op[i] = opacities[i];
                }
            });
        }
        pool.wait();
    }
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_cr, cr, false);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_cg, cg, false);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_cb, cbv, false);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_op, op, false);
    distributed::Finish(*ctx->cq);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (upload_ms) *upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ctx->uploaded_colors_ptr = colors;
    ctx->uploaded_opacities_ptr = opacities;
    ctx->uploaded_N = N;
}

// Allocate/grow + (re)register the M-compact output buffers to hold cap_elems.
static void ensure_outputs(GatherDeviceContext* ctx, uint32_t cap_elems) {
    if (!ctx->buf_M) {
        ctx->buf_M = make_dram(ctx->mesh_device.get(), PAGE_BYTES, PAGE_BYTES);
        device_state::register_buffer("proj_M", ctx->buf_M);
    }
    if (ctx->buf_px && ctx->cap_m_elems >= cap_elems) return;
    const std::size_t m_bytes = static_cast<std::size_t>(cap_elems) * 4;
    const std::size_t col_bytes = static_cast<std::size_t>(cap_elems) * 3 * 4;
    ctx->buf_px      = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_py      = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_rx      = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_ry      = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_a       = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_b       = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_c       = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_depth   = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_opacity = make_dram(ctx->mesh_device.get(), m_bytes, PAGE_BYTES);
    ctx->buf_colors  = make_dram(ctx->mesh_device.get(), col_bytes, PAGE_BYTES);
    {
        // One 64B page per record (page index == compact gaussian id g).
        const std::size_t rec_bytes = static_cast<std::size_t>(cap_elems) * PAGE_BYTES;
        ctx->buf_blendrec = make_dram(ctx->mesh_device.get(), rec_bytes, PAGE_BYTES);
        device_state::register_buffer("proj_m_blendrec", ctx->buf_blendrec);
    }
    ctx->cap_m_elems = cap_elems;
    device_state::register_buffer("proj_m_px", ctx->buf_px);
    device_state::register_buffer("proj_m_py", ctx->buf_py);
    device_state::register_buffer("proj_m_rx", ctx->buf_rx);
    device_state::register_buffer("proj_m_ry", ctx->buf_ry);
    device_state::register_buffer("proj_m_a", ctx->buf_a);
    device_state::register_buffer("proj_m_b", ctx->buf_b);
    device_state::register_buffer("proj_m_c", ctx->buf_c);
    device_state::register_buffer("proj_m_depth", ctx->buf_depth);
    device_state::register_buffer("proj_m_opacity", ctx->buf_opacity);
    device_state::register_buffer("proj_m_colors", ctx->buf_colors);
}

// Allocate the per-core counts buffer once (one 64B page per core).
static void ensure_counts(GatherDeviceContext* ctx) {
    if (ctx->buf_counts) return;
    const std::size_t bytes = static_cast<std::size_t>(ctx->num_cores) * PAGE_BYTES;
    ctx->buf_counts = make_dram(ctx->mesh_device.get(), bytes, PAGE_BYTES);
    // Device-scan output bases (one 64B page per core); allocated alongside.
    ctx->buf_core_base = make_dram(ctx->mesh_device.get(), bytes, PAGE_BYTES);
}

// Effective max_radius matching project_finish_with_cov2d_radii.
static float effective_max_radius(int max_radius_param, int H, int W) {
    if (max_radius_param < 0) return static_cast<float>(std::numeric_limits<int>::max());
    if (max_radius_param > 0) return static_cast<float>(max_radius_param);
    return static_cast<float>(std::min(H, W) / 2);
}

// A1 (iter-111): pfwc now stores the PRE-FOLDED conic {A,B,C} in the three
// covariance words of the per-Gaussian record (A=-0.5*cov_c*inv, B=cov_b*inv,
// C=-0.5*cov_a*inv, inv ~= 1/det). The device blend/cull consume that conic
// directly. The host-side CPU consumers (project_finish, tile_assign per-pair
// Mahalanobis cull, cull_and_blend) still expect the RAW 2D covariance {a,b,c}
// and re-derive the conic themselves. So when those words are read back to the
// host (the cpu_cpp_mb CPU-blend reference path; the device resident-blend path
// never reads cov2d back), invert the fold so proj.covs_2d keeps its raw-cov
// contract and every downstream CPU stage is unchanged.
//   Sigma^-1 = [[-2A,-B],[-B,-2C]];  det(Sigma^-1) = 4AC - B^2
//   Sigma = adj(Sigma^-1)/det(Sigma^-1):  a=-2C/den, b=B/den, c=-2A/den.
// den = inv^2 * det > 0 always (pfwc floors det at 1e-6), so the divide is safe.
static inline void conic_to_cov2d(
    float A, float B, float C, float& a, float& b, float& c) {
    const float den = 4.0f * A * C - B * B;
    const float inv = 1.0f / den;
    a = -2.0f * C * inv;
    b = B * inv;
    c = -2.0f * A * inv;
}

// Read the resident pfwc_* SoA buffers (padded_n) back to host AoS arrays.
static bool readback_pfwc(
    GatherDeviceContext* ctx, std::size_t N, uint32_t padded_n,
    std::vector<float>& mean_2d, std::vector<float>& depth,
    std::vector<float>& cov2d, std::vector<float>& radii) {
    auto bm2x = device_state::get_buffer("pfwc_m2x");
    auto bm2y = device_state::get_buffer("pfwc_m2y");
    auto bdep = device_state::get_buffer("pfwc_depth");
    auto ba = device_state::get_buffer("pfwc_a");
    auto bb = device_state::get_buffer("pfwc_b");
    auto bc = device_state::get_buffer("pfwc_c");
    auto brx = device_state::get_buffer("pfwc_rx");
    auto bry = device_state::get_buffer("pfwc_ry");
    if (!bm2x || !bm2y || !bdep || !ba || !bb || !bc || !brx || !bry) return false;
    std::vector<float> m2x(padded_n), m2y(padded_n), dep(padded_n),
        a(padded_n), b(padded_n), c(padded_n), rx(padded_n), ry(padded_n);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, m2x, bm2x, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, m2y, bm2y, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, dep, bdep, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, a, ba, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, b, bb, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, c, bc, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, rx, brx, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, ry, bry, true);
    mean_2d.resize(N * 2);
    depth.resize(N);
    cov2d.resize(N * 3);
    radii.resize(N * 2);
    for (std::size_t i = 0; i < N; ++i) {
        mean_2d[i * 2 + 0] = m2x[i];
        mean_2d[i * 2 + 1] = m2y[i];
        depth[i] = dep[i];
        // pfwc stores conic {A,B,C} here; recover raw cov {a,b,c} for the host
        // finisher / CPU cull+blend (see conic_to_cov2d).
        conic_to_cov2d(a[i], b[i], c[i],
                       cov2d[i * 3 + 0], cov2d[i * 3 + 1], cov2d[i * 3 + 2]);
        radii[i * 2 + 0] = rx[i];
        radii[i * 2 + 1] = ry[i];
    }
    return true;
}

// Read the M-compact proj_m_* buffers back and assemble a ProjectResult.
static gsplat_cpu::ProjectResult readback_proj_m(
    GatherDeviceContext* ctx, std::size_t M, double* readback_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    // The read vector must match the buffer element count; buffers are sized to
    // capacity (>= M_pad), so size the read vectors to capacity and slice M.
    const uint32_t cap = static_cast<uint32_t>(ctx->cap_m_elems);
    std::vector<float> px(cap), py(cap), rx(cap), ry(cap),
        a(cap), b(cap), c(cap), dep(cap), op(cap);
    std::vector<float> col(static_cast<std::size_t>(cap) * 3);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, px, ctx->buf_px, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, py, ctx->buf_py, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, rx, ctx->buf_rx, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, ry, ctx->buf_ry, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, a, ctx->buf_a, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, b, ctx->buf_b, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, c, ctx->buf_c, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, dep, ctx->buf_depth, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, op, ctx->buf_opacity, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, col, ctx->buf_colors, true);

    gsplat_cpu::ProjectResult proj;
    proj.means_2d.resize(M * 2);
    proj.covs_2d.resize(M * 4);
    proj.depths.resize(M);
    proj.radii.resize(M * 2);
    proj.colors.resize(M * 3);
    proj.opacities.resize(M);
    for (std::size_t m = 0; m < M; ++m) {
        proj.means_2d[m * 2 + 0] = px[m];
        proj.means_2d[m * 2 + 1] = py[m];
        // pfwc stores conic {A,B,C} in the cov words; recover raw {a,b,c} and
        // expand to [a, b, b, c] to match the unchanged downstream layout.
        float ca, cb, cc;
        conic_to_cov2d(a[m], b[m], c[m], ca, cb, cc);
        proj.covs_2d[m * 4 + 0] = ca;
        proj.covs_2d[m * 4 + 1] = cb;
        proj.covs_2d[m * 4 + 2] = cb;
        proj.covs_2d[m * 4 + 3] = cc;
        proj.depths[m] = dep[m];
        proj.radii[m * 2 + 0] = rx[m];
        proj.radii[m * 2 + 1] = ry[m];
        proj.colors[m * 3 + 0] = col[m * 3 + 0];
        proj.colors[m * 3 + 1] = col[m * 3 + 1];
        proj.colors[m * 3 + 2] = col[m * 3 + 2];
        proj.opacities[m] = op[m];
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (readback_ms) *readback_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return proj;
}

// True when the full on-device resident downstream chain is active, i.e. when
// tile_assign reads the resident proj_m_* over NoC, sort reads resident
// proj_m_depth + the resident pairs, and blend reads resident proj_m_* attrs.
// In that configuration the host-side proj.{means_2d,covs_2d,radii,colors}
// arrays are NEVER dereferenced downstream — only proj.opacities (the
// tile_assign host-side m2_thresh/opacity-floor cull precompute) and
// proj.depths (its size == M; values feed only the CPU sort fallback) are
// consumed. So we can skip the bulk proj_m_* D2H readback (~the dominant
// ~150ms/hero, ~75-90ms median sub-component of project_ms) and return an
// M-only-plus-opacities/depths ProjectResult — bit-identical, since the copy
// being removed is unused in this path.
static bool downstream_chain_resident() {
    // Production: the full resident chain (RESIDENT_TA_IN + DEVICE_TILE_ASSIGN +
    // DEVICE_SORT + RESIDENT_PAIRS + RESIDENT_BLEND + SFPU cull_masks) consumes
    // proj_m_* over NoC, so the gather returns the minimal M-only ProjectResult.
    return true;
}

// Read back ONLY the host-consumed compact attrs (opacity + depth) and assemble
// a metadata-only ProjectResult. means_2d/covs_2d/radii/colors are left EMPTY:
// in the resident downstream chain those are read from DRAM over NoC and never
// touched on host. Drops 10 of the 12 cap-sized D2H reads (px,py,rx,ry,a,b,c
// and the 3-wide colors) that readback_proj_m would otherwise copy and waste.
static gsplat_cpu::ProjectResult readback_proj_m_minimal(
    GatherDeviceContext* ctx, std::size_t M, double* readback_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t cap = static_cast<uint32_t>(ctx->cap_m_elems);
    std::vector<float> dep(cap), op(cap);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, dep, ctx->buf_depth, true);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, op, ctx->buf_opacity, true);

    gsplat_cpu::ProjectResult proj;
    proj.depths.resize(M);
    proj.opacities.resize(M);
    for (std::size_t m = 0; m < M; ++m) {
        proj.depths[m] = dep[m];
        proj.opacities[m] = op[m];
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (readback_ms) *readback_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return proj;
}

// Depth-only readback for the full resident downstream chain: tile_assign K3
// reads proj_m_opacity on device; sort/blend never touch host opacities.
static gsplat_cpu::ProjectResult readback_proj_m_depth_only(
    GatherDeviceContext* ctx, std::size_t M, double* readback_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t cap = static_cast<uint32_t>(ctx->cap_m_elems);
    std::vector<float> dep(cap);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, dep, ctx->buf_depth, true);

    gsplat_cpu::ProjectResult proj;
    proj.depths.resize(M);
    for (std::size_t m = 0; m < M; ++m) {
        proj.depths[m] = dep[m];
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (readback_ms) *readback_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return proj;
}

// Count-only "readback" for the FUSED device-scan + full-resident chain
// (GSPLAT_TT_PROJ_DEVICE_SCAN): M is already known from the on-device scan's
// proj_M (read as the single post-chain host sync), and in the host-free render
// the host consumes NONE of the compact attr VALUES — tile_assign/sort/blend all
// read proj_m_* resident over NoC, and a device-stage failure std::abort()s the
// host-free path rather than falling back to the host (which is the only place
// the host depth values would be used). So there is NOTHING to D2H here: return
// a metadata-only ProjectResult with depths.size()==M. Bit-identical to
// readback_proj_m_depth_only in the resident chain, minus that unused ~M*4B
// depth D2H (one more host bridge removed from the project stage).
static gsplat_cpu::ProjectResult readback_proj_m_count_only(
    std::size_t M, double* readback_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    gsplat_cpu::ProjectResult proj;
    proj.depths.assign(M, 0.0f);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (readback_ms) *readback_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return proj;
}

// Launch one pass of the multi-core gather kernel. count_only=true tallies each
// core's visible quota into buf_counts. count_only=false scatters; `bases`
// holds each core's exclusive-prefix-sum global output offset and `last_core`
// is the core that writes the final element (zero-pads the tail).
static void launch_pass(
    GatherDeviceContext* ctx, const WorkSplit& ws, std::size_t N,
    uint32_t num_tiles, float min_opacity, int H, int W, float max_radius,
    bool count_only, const std::vector<uint32_t>* bases, int last_core,
    uint32_t t_stride, bool do_finish = true, bool device_scan = false) {
    Program& program = ctx->workload.get_programs().begin()->second;
    auto bm2x = device_state::get_buffer("pfwc_m2x");
    auto bm2y = device_state::get_buffer("pfwc_m2y");
    auto bdep = device_state::get_buffer("pfwc_depth");
    auto ba = device_state::get_buffer("pfwc_a");
    auto bb = device_state::get_buffer("pfwc_b");
    auto bc = device_state::get_buffer("pfwc_c");
    auto brx = device_state::get_buffer("pfwc_rx");
    auto bry = device_state::get_buffer("pfwc_ry");

    const uint32_t common_min_op = fp32_bits(min_opacity);
    const uint32_t common_knear  = fp32_bits(0.2f);
    const uint32_t common_w      = fp32_bits(static_cast<float>(W));
    const uint32_t common_h      = fp32_bits(static_cast<float>(H));
    const uint32_t common_maxr   = fp32_bits(max_radius);
    // In the device-scan SCATTER pass the kernel reads its (base, is_last) from
    // the base buffer via the counts accessor, so repoint counts_addr at it.
    // The count pass still writes its quota into buf_counts.
    const uint32_t counts_addr =
        (device_scan && !count_only)
            ? static_cast<uint32_t>(ctx->buf_core_base->address())
            : static_cast<uint32_t>(ctx->buf_counts->address());
    const uint32_t blendrec_addr =
        ctx->buf_blendrec ? static_cast<uint32_t>(ctx->buf_blendrec->address()) : 0u;

    for (uint32_t c = 0; c < ctx->num_cores; ++c) {
        CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
        const uint32_t base = bases ? (*bases)[c] : 0u;
        const uint32_t is_last =
            (bases && static_cast<int>(c) == last_core) ? 1u : 0u;
        std::vector<uint32_t> args = {
            static_cast<uint32_t>(bm2x->address()),
            static_cast<uint32_t>(bm2y->address()),
            static_cast<uint32_t>(bdep->address()),
            static_cast<uint32_t>(ba->address()),
            static_cast<uint32_t>(bb->address()),
            static_cast<uint32_t>(bc->address()),
            static_cast<uint32_t>(brx->address()),
            static_cast<uint32_t>(bry->address()),
            static_cast<uint32_t>(ctx->buf_cr->address()),
            static_cast<uint32_t>(ctx->buf_cg->address()),
            static_cast<uint32_t>(ctx->buf_cb->address()),
            static_cast<uint32_t>(ctx->buf_op->address()),
            static_cast<uint32_t>(ctx->buf_px->address()),
            static_cast<uint32_t>(ctx->buf_py->address()),
            static_cast<uint32_t>(ctx->buf_rx->address()),
            static_cast<uint32_t>(ctx->buf_ry->address()),
            static_cast<uint32_t>(ctx->buf_a->address()),
            static_cast<uint32_t>(ctx->buf_b->address()),
            static_cast<uint32_t>(ctx->buf_c->address()),
            static_cast<uint32_t>(ctx->buf_depth->address()),
            static_cast<uint32_t>(ctx->buf_opacity->address()),
            static_cast<uint32_t>(ctx->buf_colors->address()),
            static_cast<uint32_t>(ctx->buf_M->address()),
            static_cast<uint32_t>(N),
            num_tiles,
            common_min_op,
            common_knear,
            common_w,
            common_h,
            common_maxr,
            count_only ? 1u : 0u,
            ws.chunk_start[c],
            ws.num_chunks[c],
            base,
            is_last,
            c,
            counts_addr,
        };
        args.push_back(blendrec_addr);             // arg 37: AoS blend-record base
        args.push_back(t_stride);                  // arg 38: tile stride
        args.push_back(device_scan ? 1u : 0u);     // arg 39: device-scan flag
        SetRuntimeArgs(program, ctx->kernel, core, args);
    }
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, false);
    if (do_finish) distributed::Finish(*ctx->cq);
}

}  // namespace

bool readback_pfwc_resident(
    std::size_t N, float* mean_2d, float* depth, float* cov2d, float* radii) {
    if (N == 0) return true;
    auto* ctx = ensure_context();
    if (ctx == nullptr) return false;
    const uint32_t num_tiles =
        static_cast<uint32_t>((N + TILE_ELEMS - 1) / TILE_ELEMS);
    const uint32_t padded_n = num_tiles * TILE_ELEMS;
    std::vector<float> m2d, dep, cov, rad;
    if (!readback_pfwc(ctx, N, padded_n, m2d, dep, cov, rad)) return false;
    std::memcpy(mean_2d, m2d.data(), N * 2 * sizeof(float));
    std::memcpy(depth, dep.data(), N * sizeof(float));
    std::memcpy(cov2d, cov.data(), N * 3 * sizeof(float));
    std::memcpy(radii, rad.data(), N * 2 * sizeof(float));
    return true;
}

bool gather_visible_device_ready() { return ensure_context() != nullptr; }

void gather_visible_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        (void)slot.release();
    }
}

gsplat_cpu::ProjectResult gather_visible_tt(
    const float* scene_colors,
    const float* scene_opacities,
    std::size_t N,
    int image_height,
    int image_width,
    float min_opacity,
    int max_radius_param,
    bool host_gather,
    bool verify,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    GatherCallTimings* timings,
    bool downstream_resident) {
    auto set_fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::ProjectResult{};
    };
    if (N == 0) {
        if (device_ok) *device_ok = true;
        return gsplat_cpu::ProjectResult{};
    }
    auto* ctx = ensure_context();
    if (ctx == nullptr) return set_fail();

    GatherCallTimings tlocal;
    auto& T = (timings ? *timings : tlocal);
    T.host_path = host_gather;

    const uint32_t num_tiles =
        static_cast<uint32_t>((N + TILE_ELEMS - 1) / TILE_ELEMS);
    const uint32_t padded_n = num_tiles * TILE_ELEMS;
    const float max_radius =
        effective_max_radius(max_radius_param, image_height, image_width);

    try {
        ensure_scene_uploaded(ctx, scene_colors, scene_opacities, N, num_tiles,
                              &T.upload_ms, &T.cache_hit);

        // ── R2: multi-core device gather kernel ──────────────────────────
        // Pass 1 (count): each core tallies its visible quota over its tile
        // range into buf_counts[core]. Pass 2 (scatter): the host computes the
        // exclusive prefix-sum of those counts to give each core its global
        // compact base offset; cores scatter their visible elements at those
        // offsets, preserving the single-core stable compaction order.
        ensure_outputs(ctx, PAGE_ELEMS);  // ensure buf_M + placeholder bufs exist
        ensure_counts(ctx);
        const bool balance = proj_balance_enabled();
        const WorkSplit ws = balance
            ? split_strided(num_tiles, ctx->num_cores)
            : split_chunks(num_tiles, ctx->num_cores);
        const uint32_t t_stride = balance ? ctx->num_cores : 1u;

        const auto t_k0 = std::chrono::high_resolution_clock::now();
        std::size_t M = 0;

        if (proj_device_scan_enabled()) {
            // ── STEP-4 FUSION (GSPLAT_TT_PROJ_DEVICE_SCAN) ───────────────────
            // Enqueue count -> scan_bases -> scatter as ONE in-order CQ
            // submission with NO inter-kernel Finish AND NO mid-chain host M
            // read. Previously the host had to D2H M between scan and scatter
            // just to size proj_m_* before the scatter wrote into it — a full
            // count+scan CQ drain plus the scatter-dispatch idle bubble. Instead
            // pre-size the outputs to the per-frame safe ceiling padded_n: the
            // compact visible count M is always <= N <= padded_n, so the scatter
            // (which writes only the M densely-packed [0,M) compact slots) can
            // never overflow regardless of this frame's M, and the host no longer
            // needs M before dispatching the scatter. The in-order CQ guarantees
            // count's counts[] are visible to scan and scan's base[]/proj_M are
            // visible to scatter (the same guarantee FUSED_TILE relies on across
            // sort-publish+cull+blend). The project stage's ONLY host sync is the
            // single post-chain 1-page M read below.
            ensure_outputs(ctx, padded_n);

            launch_pass(ctx, ws, N, num_tiles, min_opacity, image_height,
                        image_width, max_radius, /*count_only=*/true,
                        /*bases=*/nullptr, /*last_core=*/-1, t_stride,
                        /*do_finish=*/false, /*device_scan=*/true);

            // scan_bases: counts -> buf_core_base ([0]=base,[1]=is_last) + proj_M.
            {
                Program& sp = ctx->wl_scan.get_programs().begin()->second;
                SetRuntimeArgs(sp, ctx->kscan, CoreCoord{0, 0}, {
                    static_cast<uint32_t>(ctx->buf_counts->address()),
                    static_cast<uint32_t>(ctx->buf_core_base->address()),
                    static_cast<uint32_t>(ctx->buf_M->address()),
                    ctx->num_cores,
                });
                distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_scan, false);
            }

            // Scatter immediately (no Finish): each core reads its (base,is_last)
            // from buf_core_base over NoC — no host args, no host M needed.
            launch_pass(ctx, ws, N, num_tiles, min_opacity, image_height,
                        image_width, max_radius, /*count_only=*/false,
                        /*bases=*/nullptr, /*last_core=*/-1, t_stride,
                        /*do_finish=*/false, /*device_scan=*/true);

            // The project stage's ONLY host sync: one blocking 1-page read of the
            // on-device M. It drains count+scan+scatter together (they ran serially
            // on the in-order CQ) and replaces the old count-Finish + per-core
            // counts D2H + host exclusive scan + host proj_M write + proj_M Finish
            // + scatter Finish (and, in the resident chain below, the cap-sized
            // depth D2H). proj_M is already published on-device for the resident
            // downstream consumers.
            std::vector<uint32_t> mread(PAGE_ELEMS, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, mread, ctx->buf_M, true);
            M = mread[0];
        } else {
            launch_pass(ctx, ws, N, num_tiles, min_opacity, image_height,
                        image_width, max_radius, /*count_only=*/true,
                        /*bases=*/nullptr, /*last_core=*/-1, t_stride);

            // Read per-core counts (each occupies the first uint32 of its 64B page).
            std::vector<uint32_t> craw(
                static_cast<std::size_t>(ctx->num_cores) * PAGE_ELEMS);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, craw, ctx->buf_counts, true);

            std::vector<uint32_t> bases(ctx->num_cores, 0);
            int last_core = -1;
            for (uint32_t c = 0; c < ctx->num_cores; ++c) {
                const uint32_t cnt = craw[static_cast<std::size_t>(c) * PAGE_ELEMS];
                bases[c] = static_cast<uint32_t>(M);
                M += cnt;
                if (cnt > 0) last_core = static_cast<int>(c);
            }

            const uint32_t M_pad =
                round_up(static_cast<uint32_t>(std::max<std::size_t>(M, 1)), PAGE_ELEMS);
            ensure_outputs(ctx, M_pad);

            // Publish the authoritative M into proj_M for downstream consumers.
            {
                std::vector<uint32_t> mbuf(PAGE_ELEMS, 0);
                mbuf[0] = static_cast<uint32_t>(M);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_M, mbuf, false);
                GSPLAT_HOST_ZONE("host_finish_proj");
                distributed::Finish(*ctx->cq);
            }

            // Scatter pass: fill proj_m_* on-device in increasing source order.
            launch_pass(ctx, ws, N, num_tiles, min_opacity, image_height,
                        image_width, max_radius, /*count_only=*/false, &bases,
                        last_core, t_stride);
        }
        const auto t_k1 = std::chrono::high_resolution_clock::now();
        T.kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

        // PERF: skip the wasted bulk proj_m_* D2H iff (a) the caller confirms
        // THIS render's downstream blend is the device resident blend
        // (downstream_resident, i.e. blend_mode>=1 — NOT the cpu_cpp_mb CPU-blend
        // reference render), (b) the full resident chain env gates are on, and
        // (c) we are not verifying. Then tile_assign/sort/blend read the
        // resident DRAM over NoC; only opacity (TA cull) + depth (M / CPU-sort
        // fallback) are consumed host-side. Bit-identical: the dropped copy is
        // never read in this path.
        const bool minimal_readback =
            downstream_resident && downstream_chain_resident() && !verify;
        // FUSED device-scan + full-resident chain: M came back from proj_M and
        // the host needs none of the compact attr values -> skip the D2H entirely
        // (count-only). Falls back to the existing readbacks for the non-fused,
        // non-resident, or verify configs (which DO consume host-side proj_m_*).
        const bool fused_count_only =
            proj_device_scan_enabled() && minimal_readback;
        gsplat_cpu::ProjectResult proj =
            fused_count_only
                ? readback_proj_m_count_only(M, &T.readback_ms)
                : (minimal_readback
                       ? (downstream_chain_resident()
                              ? readback_proj_m_depth_only(ctx, M, &T.readback_ms)
                              : readback_proj_m_minimal(ctx, M, &T.readback_ms))
                       : readback_proj_m(ctx, M, &T.readback_ms));

        if (device_ok) *device_ok = true;
        return proj;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::gather] call failed: " << e.what() << "\n";
        return set_fail();
    }
}

}  // namespace gsplat_tt
