// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-process host driver for gsplat_tt blend (amendment-002 tt-001a).
// Refactored from tt-metal alpha_blend.cpp: no daemon, no .npy IPC.

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
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

#include "alpha_blend_host.h"
#include "blend.h"
#include "gsplat_cpu/blend_microblock.h"
#include "gsplat_cpu/thread_pool.h"
#include "gsplat_tt/device_state.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace gsplat;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// ---------------------------------------------------------------------------
// .npy I/O helpers
// ---------------------------------------------------------------------------

static std::vector<float> load_npy_f32(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[6];
    f.read(magic, 6);
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, ' ');
    f.read(header.data(), header_len);
    auto start = header.find('(') + 1;
    auto end = header.find(')');
    std::string shape_str = header.substr(start, end - start);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        if (comma == std::string::npos) {
            comma = shape_str.size();
        }
        std::string num = shape_str.substr(pos, comma - pos);
        while (!num.empty() && (num.front() == ' ' || num.front() == '\t')) {
            num.erase(0, 1);
        }
        while (!num.empty() && (num.back() == ' ' || num.back() == '\t')) {
            num.pop_back();
        }
        if (!num.empty()) {
            shape.push_back(std::stoul(num));
        }
        pos = comma + 1;
    }
    size_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return data;
}

static std::vector<uint32_t> load_npy_u32(const std::string& path, std::vector<size_t>& shape) {
    auto f32 = load_npy_f32(path, shape);
    std::vector<uint32_t> out(f32.size());
    for (size_t i = 0; i < f32.size(); i++) {
        out[i] = static_cast<uint32_t>(f32[i]);
    }
    return out;
}

static void save_npy_f32(const std::string& path, const std::vector<float>& data, const std::vector<size_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    f.write(reinterpret_cast<char*>(&major), 1);
    f.write(reinterpret_cast<char*>(&minor), 1);
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) {
            shape_str += ", ";
        }
    }
    shape_str += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    while ((10 + header.size() + 1) % 64 != 0) {
        header += ' ';
    }
    header += '\n';
    uint16_t header_len = header.size();
    f.write(reinterpret_cast<char*>(&header_len), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

static std::vector<uint16_t> fp32_tile_to_bf16(const float* src) {
    std::vector<uint16_t> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u;
        std::memcpy(&u, &src[i], 4);
        dst[i] = static_cast<uint16_t>(u >> 16);
    }
    return dst;
}

static std::vector<float> bf16_tile_to_fp32(const uint16_t* src) {
    std::vector<float> dst(TILE_H * TILE_W);
    for (size_t i = 0; i < TILE_H * TILE_W; i++) {
        uint32_t u = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &u, 4);
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Device + program reusable context
// ---------------------------------------------------------------------------

struct DeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    distributed::MeshWorkload workload;
    KernelHandle reader{};
    KernelHandle compute{};
    KernelHandle writer{};
    // Full compute grid (e.g. 8x8 on Wormhole N150 after harvesting).
    // CBs and kernels are allocated on every core in this range at init;
    // per-frame, split_work_to_cores carves [0, num_tiles) into a contiguous
    // slice per core via SetRuntimeArgs.
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;
};

// Build a Program with all CBs allocated and the 3 kernels compiled.
// TensorAccessorArgs for DRAM-interleaved buffers only encode (IsDram,
// aligned_page_size) at compile time; both are buffer-instance-independent
// for our use, so we can reuse this Program across frames whose inputs have
// different sizes — only DRAM addresses + num_tiles change per frame and are
// passed via SetRuntimeArgs.
//
// CBs and kernels are created on the full compute grid (`ctx.all_cores`).
// Each core gets its own independent copy of every CB (state, scratch, etc.).
// Per-frame work distribution happens in process_frame() via
// split_work_to_cores + SetRuntimeArgs.
static void build_program_and_workload(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    // Mixed-precision layout (amendment-002 tt-001b precision fix, 2026-05-28):
    //   * Mahalanobis chain (PX/PY/DX/DY/DX2/DY2/DXDY/Q/POWER) stays bf16 —
    //     these are bounded geometric quantities; bf16 gives 47+ dB on cpu_cpp_mb.
    //   * Per-pixel STATE (R/G/B/T) + intermediates that fold into state
    //     (ALPHA, CONTRIB, T_TMP, ONE_MINUS_ALPHA) move to fp32. Same playbook
    //     as tt-007 project/pfwc: state mutation via SFPU on DEST, fp32 CBs
    //     with UnpackToDestFp32 round-trip. This kills the bf16 accumulator
    //     swamping that caps current PSNR at 39 dB.
    //   * Earlier "whole-pipeline fp32" attempts (per the original baseline
    //     comment) used FPU mul_tiles/add_tiles which still truncate via
    //     SrcA/SrcB regardless of CB format — that's what regressed to 34 dB.
    //     The fix is fp32 CB + SFPU ops, not fp32 CB alone.
    auto cb_tile_bf16 = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_BF16, {{id, DataFormat::Float16_b}});
        c.set_page_size(id, TILE_BYTES_BF16);
        CreateCircularBuffer(program, cores, c);
    };
    auto cb_tile_fp32 = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_FP32, {{id, DataFormat::Float32}});
        c.set_page_size(id, TILE_BYTES_FP32);
        CreateCircularBuffer(program, cores, c);
    };
    auto cb_small = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_tile_bf16(CB_PX, 2);
    cb_tile_bf16(CB_PY, 2);
    cb_small(CB_SCALARS, SCALAR_PACK_PAGE_BYTES, 4, DataFormat::Float32);
    cb_small(CB_TILE_META, META_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_tile_bf16(CB_COLOR_OUT, 6);  // writer DMAs bf16 to DRAM; final pack converts.

    cb_tile_bf16(CB_DX, 2);
    cb_tile_bf16(CB_DY, 2);
    cb_tile_bf16(CB_DX2, 2);
    cb_tile_bf16(CB_DY2, 2);
    cb_tile_bf16(CB_DXDY, 2);
    {
        CircularBufferConfig c(3 * TILE_BYTES_BF16, {{CB_Q, DataFormat::Float16_b}});
        c.set_page_size(CB_Q, TILE_BYTES_BF16);
        CreateCircularBuffer(program, cores, c);
    }
    cb_tile_bf16(CB_POWER, 2);

    // metal-iter-000 baseline: all tile CBs bf16 + no fp32_dest_acc. The tt-001b
    // fp32 state/alpha CB experiment corrupts rows >= 8 on Blackhole and is
    // opt-in via GSPLAT_TT_BLEND_FP32=1.
    const char* blend_fp32_env = std::getenv("GSPLAT_TT_BLEND_FP32");
    const bool blend_fp32 = blend_fp32_env != nullptr && blend_fp32_env[0] == '1';

    if (blend_fp32) {
        cb_tile_fp32(CB_ALPHA, 2);
        cb_tile_fp32(CB_CONTRIB, 1);
        cb_tile_fp32(CB_ONE_MINUS_ALPHA, 1);
        cb_tile_fp32(CB_T_TMP, 1);
        cb_tile_fp32(CB_COLOR_R_STATE, 1);
        cb_tile_fp32(CB_COLOR_G_STATE, 1);
        cb_tile_fp32(CB_COLOR_B_STATE, 1);
        cb_tile_fp32(CB_T_STATE, 1);
    } else {
        cb_tile_bf16(CB_ALPHA, 2);
        cb_tile_bf16(CB_CONTRIB, 1);
        cb_tile_bf16(CB_ONE_MINUS_ALPHA, 1);
        cb_tile_bf16(CB_T_TMP, 1);
        cb_tile_bf16(CB_COLOR_R_STATE, 1);
        cb_tile_bf16(CB_COLOR_G_STATE, 1);
        cb_tile_bf16(CB_COLOR_B_STATE, 1);
        cb_tile_bf16(CB_T_STATE, 1);
    }

    cb_tile_bf16(CB_SAT_MASK, 1);  // 0/1 mask; bf16 exact.

    cb_tile_bf16(CB_CONST_ZERO, 1);
    cb_tile_bf16(CB_CONST_099, 1);
    // Stage 2 shadow CBs: reader fills these in parallel with legacy scalars;
    // compute ignores them until Stage 3 microblock-major rewrite.
    cb_small(CB_MB_COEFF_SHADOW, COEFF_ROW_BYTES, 4, DataFormat::Float32);
    cb_small(CB_MB_HEADER_SHADOW, META_PAGE_BYTES, 4, DataFormat::UInt32);
    cb_small(CB_MB_STREAM_SHADOW, META_PAGE_BYTES, 4, DataFormat::UInt32);
    // CB_CONST_NEG88 (index 11) is reserved but unused now that the kernel
    // uses exp_tile<approx=true>, which clamps negative inputs internally.
    // Slot kept reserved to avoid renumbering downstream CBs.

    // Reader: 5 DRAM-interleaved TensorAccessorArgs for
    // packs/offsets/px/py/tile_ids. For non-sharded interleaved buffers, the
    // compile-time args reduce to (IsDram flag, aligned_page_size). Page sizes
    // are compile-time constants, so this is independent of any specific
    // buffer instance.
    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    // tt-001b precision fix: enable UnpackToDestFp32 for every CB that holds
    // (or is unpacked into a DEST slot that holds) state-precision fp32 data.
    // Without this, copy_tile routes through SrcA/SrcB which truncate fp32 to
    // bf16 — same root cause as the iter-06e regression in project_device.cpp.
    // The vector must be sized to the full buf_formats table (64); tt-metal
    // asserts on a mismatch at program build time.
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    if (blend_fp32) {
        for (uint32_t cb : {CB_ALPHA, CB_CONTRIB, CB_ONE_MINUS_ALPHA, CB_T_TMP,
                            CB_COLOR_R_STATE, CB_COLOR_G_STATE, CB_COLOR_B_STATE,
                            CB_T_STATE}) {
            u2d[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    // DST-persistent fp32 kernel is the default device blend: keeps R/G/B/T in
    // fp32 DEST across the Gaussian loop -> 56-64 dB vs cpu_cpp_mb (vs ~26 dB
    // for the legacy bf16-CB-spill kernel). Opt out with GSPLAT_TT_DST_PERSISTENT=0.
    const char* dst_persist = std::getenv("GSPLAT_TT_DST_PERSISTENT");
    const bool use_dst_persistent =
        (dst_persist == nullptr) || (dst_persist[0] != '0');
    const char* compute_kernel = use_dst_persistent
        ? OVERRIDE_KERNEL_PREFIX
          "kernels/compute/alpha_blend_compute_dst_persistent.cpp"
        : OVERRIDE_KERNEL_PREFIX "kernels/compute/alpha_blend_compute.cpp";

    ctx.compute = CreateKernel(
        program,
        compute_kernel,
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            // DST-persistent kernel keeps R/G/B/T in fp32 DEST across the whole
            // Gaussian loop, so it ALWAYS needs fp32 dest-acc (+ full-sync for 8
            // fp32 DEST tiles), independent of the legacy fp32-state-CB toggle.
            .fp32_dest_acc_en = blend_fp32 || use_dst_persistent,
            .dst_full_sync_en = use_dst_persistent,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
        });

    // Writer: 2 TensorAccessorArgs for out + tile_ids.
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context() {
    DeviceContext ctx;
    // Use the shared MeshDevice owned by gsplat_tt::device_state. This makes
    // blend and project share a single device handle, eliminating the
    // double-close that happened when each module had its own
    // create_unit_mesh(0) + close() pair.
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program_and_workload(ctx);
    return ctx;
}

// ---------------------------------------------------------------------------
// Per-frame work
// ---------------------------------------------------------------------------

struct FrameBufferInputs {
    const std::vector<float>& packs_f32;
    const std::vector<float>& offsets_f32;
    const std::vector<float>& px_f32;
    const std::vector<float>& py_f32;
    const std::vector<float>* coeff_f32 = nullptr;
    const std::vector<uint32_t>* mb_header_u32 = nullptr;
    const std::vector<uint32_t>* mb_stream_u32 = nullptr;
    uint32_t image_h = 0;
    uint32_t image_w = 0;
    std::vector<float>& image_out;
    bool use_microblock_payload() const {
        return coeff_f32 != nullptr && mb_header_u32 != nullptr && mb_stream_u32 != nullptr;
    }
};

// Result of LPT load balancing for one frame.
struct TileAssignment {
    // Concatenated per-core tile ID slices, padded up to a multiple of
    // TILE_IDS_PAGE_BYTES so the DRAM write matches the buffer page size.
    std::vector<uint32_t> tile_id_buffer_padded;
    std::vector<uint32_t> per_core_offset;  // size num_cores
    std::vector<uint32_t> per_core_count;   // size num_cores
    size_t tile_id_buffer_bytes_padded = 0;
};

constexpr size_t TILE_IDS_PAGE_BYTES = 64;

// LPT (Longest Processing Time first): sort tiles descending by Gaussian
// count, then greedily assign each tile to the currently least-loaded core.
// 4/3-approximation of the optimal makespan; usually within a few percent of
// perfect on our workloads. Returns a per-core list of tile IDs.
//
// Empty tiles (cost == 0) are filtered out and never reach the kernel. This
// is correctness-critical, not just a perf optimization: dispatching the full
// CB pump (TILE_META + PX + PY + state CBs + COLOR_OUT) for thousands of
// empty tiles per frame triggers a producer/consumer CB deadlock at >= 16
// active cores. The host pre-zeros the output buffer in process_frame() so
// skipped tile slots show as black instead of stale DRAM content.
static std::vector<std::vector<uint32_t>> compute_lpt_assignment(
    const std::vector<float>& offsets_f32, uint32_t num_tiles, uint32_t num_cores) {
    std::vector<std::pair<uint32_t, uint32_t>> cost_id;
    cost_id.reserve(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t cost = static_cast<uint32_t>(offsets_f32[t + 1] - offsets_f32[t]);
        if (cost > 0) {
            cost_id.emplace_back(cost, t);
        }
    }
    std::sort(cost_id.begin(), cost_id.end(), std::greater<>());

    std::vector<std::vector<uint32_t>> per_core_tile_ids(num_cores);
    std::vector<uint64_t> core_load(num_cores, 0);
    for (const auto& [cost, id] : cost_id) {
        const auto min_it = std::min_element(core_load.begin(), core_load.end());
        const uint32_t c = static_cast<uint32_t>(std::distance(core_load.begin(), min_it));
        per_core_tile_ids[c].push_back(id);
        core_load[c] += cost;
    }
    return per_core_tile_ids;
}

// Concatenate per-core tile-ID lists into one buffer with per-core (offset,
// count) bookkeeping, padded so DRAM write covers a whole multiple of pages.
// (tt-metal asserts size % page_size == 0; std::max alone only guarantees
// >= one page, not multiple-of-page.)
static TileAssignment build_tile_assignment(
    const std::vector<float>& offsets_f32, uint32_t num_tiles, uint32_t num_cores) {
    auto per_core = compute_lpt_assignment(offsets_f32, num_tiles, num_cores);

    TileAssignment a;
    a.per_core_offset.assign(num_cores, 0);
    a.per_core_count.assign(num_cores, 0);

    std::vector<uint32_t> flat;
    flat.reserve(num_tiles);
    for (uint32_t c = 0; c < num_cores; c++) {
        a.per_core_offset[c] = static_cast<uint32_t>(flat.size());
        a.per_core_count[c]  = static_cast<uint32_t>(per_core[c].size());
        flat.insert(flat.end(), per_core[c].begin(), per_core[c].end());
    }

    const size_t bytes_payload = flat.size() * sizeof(uint32_t);
    const size_t bytes_min = std::max<size_t>(bytes_payload, TILE_IDS_PAGE_BYTES);
    a.tile_id_buffer_bytes_padded =
        ((bytes_min + TILE_IDS_PAGE_BYTES - 1) / TILE_IDS_PAGE_BYTES) * TILE_IDS_PAGE_BYTES;
    a.tile_id_buffer_padded.assign(a.tile_id_buffer_bytes_padded / sizeof(uint32_t), 0);
    std::copy(flat.begin(), flat.end(), a.tile_id_buffer_padded.begin());
    return a;
}

// Pack N x 9 fp32 attribute rows into 64-byte pages
// (9 fp32 = 36 bytes payload, 28 bytes zero-padded per row).
static std::vector<uint32_t> encode_attribute_packs(
    const std::vector<float>& packs_f32, uint32_t total_entries) {
    std::vector<uint32_t> packs_payload(
        (static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES) / 4, 0);
    constexpr size_t row_payload_bytes = 9 * sizeof(float);
    for (uint32_t e = 0; e < total_entries; e++) {
        std::memcpy(
            reinterpret_cast<uint8_t*>(packs_payload.data())
                + static_cast<size_t>(e) * SCALAR_PACK_PAGE_BYTES,
            &packs_f32[e * 9],
            row_payload_bytes);
    }
    return packs_payload;
}

static std::vector<uint32_t> encode_coeff_table(
    const std::vector<float>& coeff_f32, uint32_t total_entries) {
    std::vector<uint32_t> payload(
        (static_cast<size_t>(total_entries) * COEFF_ROW_BYTES) / 4, 0);
    constexpr size_t row_bytes = COEFF_LANES_PER_GAUSSIAN * sizeof(float);
    for (uint32_t e = 0; e < total_entries; e++) {
        std::memcpy(
            reinterpret_cast<uint8_t*>(payload.data()) + static_cast<size_t>(e) * COEFF_ROW_BYTES,
            &coeff_f32[e * COEFF_LANES_PER_GAUSSIAN],
            row_bytes);
    }
    return payload;
}

// Encode (num_tiles, 32, 32) fp32 input as bf16 tile-major bytes.
static std::vector<uint16_t> encode_tiles_to_bf16(
    const std::vector<float>& f32, uint32_t num_tiles) {
    std::vector<uint16_t> bf16(static_cast<size_t>(num_tiles) * TILE_H * TILE_W);
    for (uint32_t t = 0; t < num_tiles; t++) {
        auto tile = fp32_tile_to_bf16(&f32[t * TILE_H * TILE_W]);
        std::memcpy(&bf16[t * TILE_H * TILE_W], tile.data(), TILE_BYTES_BF16);
    }
    return bf16;
}

// Convert tile-major (num_tiles, 3, 32, 32) bf16 result into a row-major
// (image_h, image_w, 3) fp32 image, cropping to the requested image dims.
static std::vector<float> tiles_to_image(
    const std::vector<uint16_t>& result_bf16,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w) {
    std::vector<float> img(static_cast<size_t>(image_h) * image_w * 3, 0.0f);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t ty = t / tiles_x;
        const uint32_t tx = t % tiles_x;
        for (uint32_t ch = 0; ch < 3; ch++) {
            const auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
            for (uint32_t i = 0; i < TILE_H; i++) {
                for (uint32_t j = 0; j < TILE_W; j++) {
                    const uint32_t y = ty * TILE_H + i;
                    const uint32_t x = tx * TILE_W + j;
                    if (y < image_h && x < image_w) {
                        img[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[i * TILE_W + j];
                    }
                }
            }
        }
    }
    return img;
}

struct FrameDramBuffers {
    std::shared_ptr<distributed::MeshBuffer> packs;
    std::shared_ptr<distributed::MeshBuffer> offsets;
    std::shared_ptr<distributed::MeshBuffer> px;
    std::shared_ptr<distributed::MeshBuffer> py;
    std::shared_ptr<distributed::MeshBuffer> output;
    std::shared_ptr<distributed::MeshBuffer> tile_ids;
    std::shared_ptr<distributed::MeshBuffer> coeff_table;
    std::shared_ptr<distributed::MeshBuffer> mb_header;
    std::shared_ptr<distributed::MeshBuffer> mb_stream;
};

// Allocate the 6 DRAM buffers a frame needs. Sizes are derived from the
// scene's total_entries + tile count + the LPT-balanced tile-id list.
// All buffers are RAII via shared_ptr; they free on scope exit.
static FrameDramBuffers allocate_frame_buffers(
    DeviceContext& ctx,
    uint32_t total_entries,
    uint32_t num_tiles,
    size_t offsets_count,
    size_t tile_ids_bytes,
    bool use_mb,
    size_t mb_header_bytes,
    size_t mb_stream_bytes) {
    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{
            .page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };
    FrameDramBuffers b;
    b.packs    = make_dram(static_cast<size_t>(total_entries) * SCALAR_PACK_PAGE_BYTES, SCALAR_PACK_PAGE_BYTES);
    b.offsets  = make_dram(offsets_count * sizeof(uint32_t), sizeof(uint32_t));
    b.px       = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
    b.py       = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
    b.output   = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
    b.tile_ids = make_dram(tile_ids_bytes, TILE_IDS_PAGE_BYTES);
    if (use_mb) {
        b.coeff_table = make_dram(
            static_cast<size_t>(total_entries) * COEFF_ROW_BYTES, COEFF_ROW_BYTES);
        b.mb_header = make_dram(mb_header_bytes, META_PAGE_BYTES);
        b.mb_stream = make_dram(mb_stream_bytes, META_PAGE_BYTES);
    }
    return b;
}

// Set per-core runtime args for reader/compute/writer. Each core's slice of
// the concatenated tile_id_buffer is identified by (per_core_offset[c],
// per_core_count[c]); reader/writer kernels look up their tile IDs at runtime
// via this slice.
static void set_per_core_runtime_args(
    Program& program,
    const DeviceContext& ctx,
    const FrameDramBuffers& bufs,
    const TileAssignment& assign,
    uint32_t num_tiles) {
    const uint32_t packs_addr    = static_cast<uint32_t>(bufs.packs->address());
    const uint32_t offsets_addr  = static_cast<uint32_t>(bufs.offsets->address());
    const uint32_t px_addr       = static_cast<uint32_t>(bufs.px->address());
    const uint32_t py_addr       = static_cast<uint32_t>(bufs.py->address());
    const uint32_t out_addr      = static_cast<uint32_t>(bufs.output->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(bufs.tile_ids->address());
    const uint32_t coeff_addr =
        bufs.coeff_table ? static_cast<uint32_t>(bufs.coeff_table->address()) : 0;
    const uint32_t mb_header_addr =
        bufs.mb_header ? static_cast<uint32_t>(bufs.mb_header->address()) : 0;
    const uint32_t mb_stream_addr =
        bufs.mb_stream ? static_cast<uint32_t>(bufs.mb_stream->address()) : 0;
    const bool use_mb = bufs.coeff_table != nullptr;

    uint32_t core_index = 0;
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                const uint32_t start = assign.per_core_offset[core_index];
                const uint32_t count = assign.per_core_count[core_index];
                SetRuntimeArgs(program, ctx.reader, core, {
                    packs_addr, offsets_addr, px_addr, py_addr,
                    tile_ids_addr, start, count,
                    coeff_addr, mb_header_addr, mb_stream_addr, num_tiles,
                });
                SetRuntimeArgs(program, ctx.compute, core, {
                    count, use_mb ? 1u : 0u,
                });
                SetRuntimeArgs(program, ctx.writer, core, {
                    out_addr, tile_ids_addr, start, count,
                });
                core_index++;
            }
        }
    }
}

// Pull the program out of the workload (it was moved in at init time) so we
// can refresh its runtime args before each frame.
static Program& get_program_for_workload(DeviceContext& ctx) {
    auto& programs = ctx.workload.get_programs();
    auto it = programs.find(distributed::MeshCoordinateRange(ctx.mesh_device->shape()));
    if (it == programs.end()) {
        throw std::runtime_error("workload missing program for device range");
    }
    return it->second;
}

// Returns the kernel-only elapsed time (EnqueueWriteBuffer start ->
// EnqueueReadBuffer end) in milliseconds.
static double process_frame(DeviceContext& ctx, const FrameBufferInputs& f) {
    const uint32_t image_h = f.image_h;
    const uint32_t image_w = f.image_w;
    const uint32_t tiles_x = (image_w + TILE_W - 1) / TILE_W;
    const uint32_t tiles_y = (image_h + TILE_H - 1) / TILE_H;
    const uint32_t num_tiles = tiles_x * tiles_y;
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;

    const auto& packs_f32 = f.packs_f32;
    const auto& offsets_f32 = f.offsets_f32;
    const auto& px_f32 = f.px_f32;
    const auto& py_f32 = f.py_f32;
    if (packs_f32.empty() || offsets_f32.size() < static_cast<size_t>(num_tiles + 1)) {
        throw std::runtime_error("invalid blend payload sizes");
    }
    const uint32_t total_entries = static_cast<uint32_t>(packs_f32.size() / 9);

    const bool use_mb = f.use_microblock_payload();
    size_t mb_header_bytes = 0;
    size_t mb_stream_bytes = 0;
    if (use_mb) {
        if (f.coeff_f32->size() < static_cast<size_t>(total_entries) * COEFF_LANES_PER_GAUSSIAN) {
            throw std::runtime_error("coeff_table rows != packs rows");
        }
        mb_header_bytes = f.mb_header_u32->size() * sizeof(uint32_t);
        mb_stream_bytes = f.mb_stream_u32->size() * sizeof(uint32_t);
        mb_header_bytes = ((mb_header_bytes + META_PAGE_BYTES - 1) / META_PAGE_BYTES) * META_PAGE_BYTES;
        mb_stream_bytes = ((mb_stream_bytes + META_PAGE_BYTES - 1) / META_PAGE_BYTES) * META_PAGE_BYTES;
    }

    // 2. LPT-balanced tile-to-core assignment.
    const TileAssignment assign = build_tile_assignment(offsets_f32, num_tiles, num_cores);

    // 3. Allocate per-frame DRAM buffers and prepare upload payloads.
    // (Non-const because EnqueueWrite/ReadMeshBuffer takes non-const lvalue
    // refs to shared_ptr<MeshBuffer>.)
    FrameDramBuffers bufs = allocate_frame_buffers(
        ctx, total_entries, num_tiles, offsets_f32.size(),
        assign.tile_id_buffer_bytes_padded, use_mb, mb_header_bytes, mb_stream_bytes);
    auto packs_payload = encode_attribute_packs(packs_f32, total_entries);
    std::vector<uint32_t> coeff_payload;
    if (use_mb) {
        coeff_payload = encode_coeff_table(*f.coeff_f32, total_entries);
    }
    auto px_bf16 = encode_tiles_to_bf16(px_f32, num_tiles);
    auto py_bf16 = encode_tiles_to_bf16(py_f32, num_tiles);
    std::vector<uint32_t> offsets_u32(offsets_f32.size());
    for (size_t i = 0; i < offsets_f32.size(); i++) {
        offsets_u32[i] = static_cast<uint32_t>(offsets_f32[i]);
    }
    std::vector<uint32_t> mb_header_payload;
    std::vector<uint32_t> mb_stream_payload;
    if (use_mb) {
        mb_header_payload.assign(mb_header_bytes / 4, 0);
        std::memcpy(
            mb_header_payload.data(), f.mb_header_u32->data(),
            f.mb_header_u32->size() * sizeof(uint32_t));
        mb_stream_payload.assign(mb_stream_bytes / 4, 0);
        std::memcpy(
            mb_stream_payload.data(), f.mb_stream_u32->data(),
            f.mb_stream_u32->size() * sizeof(uint32_t));
    }

    // 4. Refresh runtime args for this frame.
    Program& program = get_program_for_workload(ctx);
    set_per_core_runtime_args(program, ctx, bufs, assign, num_tiles);

    // 5. Kernel timing window: DRAM upload start -> output readback end.
    const auto t_start = std::chrono::steady_clock::now();
    // Zero-fill the output buffer. compute_lpt_assignment() filters out
    // empty tiles, so their slots are never written by the writer kernel.
    // The DRAM allocator may reuse addresses across frames in daemon mode,
    // so without this fill, empty regions would show stale pixels.
    // Output is fp32 tile-major now; size matches CB_COLOR_OUT page (4 KB).
    std::vector<uint16_t> output_zero(
        static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.output,   output_zero);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.packs,    packs_payload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.offsets,  offsets_u32);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.px,       px_bf16);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.py,       py_bf16);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.tile_ids, assign.tile_id_buffer_padded);
    if (use_mb) {
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.coeff_table, coeff_payload);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.mb_header, mb_header_payload);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.mb_stream, mb_stream_payload);
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::vector<uint16_t> result_bf16(
        static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, bufs.output, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();
    const double kernel_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 6. Tile-major bf64 output -> row-major fp32 image.
    f.image_out = tiles_to_image(result_bf16, num_tiles, tiles_x, image_h, image_w);
    return kernel_ms;
}

// ===========================================================================
// Microblock-major (4x8) device blend path (amendment-003 step 3).
// Self-contained: its own program/CBs/kernels + context, selected at runtime.
// ===========================================================================

namespace mb {

constexpr uint32_t CB_XRAMP     = 0;   // fp32 tile-local x ramp
constexpr uint32_t CB_YRAMP     = 1;   // fp32 tile-local y ramp
constexpr uint32_t CB_MB_COEFF  = 2;   // 48B coeff row per gaussian (mb-major)
constexpr uint32_t CB_MB_COUNTS = 3;   // 128B = 32 uint32 per tile
constexpr uint32_t CB_OUT       = 16;  // 3 bf16 color tiles per screen tile

constexpr uint32_t COUNTS_PAGE_BYTES = 128;
// 10 real fp32 lanes (A,B,C,mx,my,f,op,r,g,b) padded to a 64B DRAM-aligned page.
// MUST be a multiple of the DRAM alignment (64B on Blackhole): an unaligned page
// size (e.g. 48B) makes the interleaved per-bank page stride diverge between the
// host EnqueueWriteBuffer layout and the device TensorAccessor, so only some rows
// land and the rest read as zero.
constexpr uint32_t COEFF_ROW_BYTES_MB = 64;
constexpr uint32_t RAMP_TILE_BYTES = TILE_H * TILE_W * 4;  // fp32 32x32

static void build_program_and_workload_mb(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_cfg(CB_XRAMP, RAMP_TILE_BYTES, 2, DataFormat::Float32);
    cb_cfg(CB_YRAMP, RAMP_TILE_BYTES, 2, DataFormat::Float32);
    cb_cfg(CB_MB_COEFF, COEFF_ROW_BYTES_MB, 8, DataFormat::Float32);
    cb_cfg(CB_MB_COUNTS, COUNTS_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_cfg(CB_OUT, TILE_BYTES_BF16, 6, DataFormat::Float16_b);

    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 6; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_alpha_blend_mb.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_XRAMP] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_YRAMP] = UnpackToDestMode::UnpackToDestFp32;

    std::map<std::string, std::string> mb_defines;
    if (const char* dbg = std::getenv("GSPLAT_TT_MB_DEBUG")) {
        // e.g. GSPLAT_TT_MB_DEBUG=ANALYTIC -> -DMB_DEBUG_ANALYTIC
        mb_defines["MB_DEBUG_" + std::string(dbg)] = "1";
    }
    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/alpha_blend_compute_mb.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
            .defines = mb_defines,
        });

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context_mb() {
    DeviceContext ctx;
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program_and_workload_mb(ctx);
    return ctx;
}

// Tile-local coordinate ramp (one 32x32 fp32 tile, row-major per-pixel,
// identical for every screen tile): xramp[r*32+c] = c + 0.5, yramp = r + 0.5.
// Uploaded the same per-pixel way the proven full-tile kernel uploads px/py:
// copy_tile + pack_tile apply an identical tilization permutation to input and
// output, so x = dst_reg[vector] yields the correct per-pixel coords for
// whatever raster pixels that SFPU vector covers (the permutation cancels).
// Fixed permutation tbl[devpos] = imgpos mapping a device tile-local raster
// position (row-major i*32+j, as produced/consumed by the standard tilize/
// untilize) to the image tile-local raster position it represents.
//
// WHY: a single SFPU 32-lane vector dst_reg[V] (fp32 dest) owns the 2 tile rows
// {2*(r/2), +1} at the 16 columns of parity (V&1) -- i.e. V = 2*(r/2)+(c&1) --
// NOT a contiguous 4x8 raster block. To run true 4x8-microblock culling we make
// vector V compute microblock V (host index m=(my<<2)|mx, region rows
// [my*4,+4) x cols [mx*8,+8)). The ramp upload puts microblock V's coords into
// vector V's real pixel slots P(V); the output is scattered back here. Because
// the device tilize (input) and untilize (output) are exact inverses at the
// pixel level (proven: the full-tile kernel renders pixel-correct with the
// natural ramp), the same table serves both directions.
static const std::array<uint32_t, TILE_H * TILE_W>& mb_perm_img_of_dev() {
    static const std::array<uint32_t, TILE_H * TILE_W> tbl = [] {
        std::array<uint32_t, TILE_H * TILE_W> t{};
        for (int V = 0; V < 32; ++V) {
            // P(V): device raster positions owned by vector V, row-major.
            std::vector<int> Pv;
            Pv.reserve(32);
            for (int r = 0; r < 32; ++r) {
                for (int c = 0; c < 32; ++c) {
                    if (2 * (r / 2) + (c & 1) == V) {
                        Pv.push_back(r * 32 + c);
                    }
                }
            }
            // T(V): microblock V's true 4x8 raster block, row-major.
            const int my = V >> 2;   // row-band (4 rows)
            const int mx = V & 3;    // col-group (8 cols)
            std::vector<int> Tv;
            Tv.reserve(32);
            for (int dr = 0; dr < 4; ++dr) {
                for (int dc = 0; dc < 8; ++dc) {
                    Tv.push_back((my * 4 + dr) * 32 + (mx * 8 + dc));
                }
            }
            for (int k = 0; k < 32; ++k) {
                t[static_cast<size_t>(Pv[k])] = static_cast<uint32_t>(Tv[k]);
            }
        }
        return t;
    }();
    return tbl;
}

static std::vector<uint32_t> make_ramp(bool is_x) {
    // Permuted ramp: at each device raster slot store the coordinate of the
    // microblock pixel that the SFPU vector owning that slot must compute.
    const auto& tbl = mb_perm_img_of_dev();
    std::vector<uint32_t> r(TILE_H * TILE_W);
    for (uint32_t dev = 0; dev < TILE_H * TILE_W; dev++) {
        const uint32_t img = tbl[dev];
        const uint32_t img_i = img / TILE_W;  // row
        const uint32_t img_j = img % TILE_W;  // col
        const float v = (is_x ? static_cast<float>(img_j) : static_cast<float>(img_i)) + 0.5f;
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        r[dev] = bits;
    }
    return r;
}

// Microblock-permuted variant of tiles_to_image: scatters each device tile-
// local raster slot back to its true microblock raster position via the fixed
// permutation, then places the tile into the full image.
static std::vector<float> tiles_to_image_mb(
    const std::vector<uint16_t>& result_bf16,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w) {
    const auto& tbl = mb_perm_img_of_dev();
    std::vector<float> img(static_cast<size_t>(image_h) * image_w * 3, 0.0f);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t ty = t / tiles_x;
        const uint32_t tx = t % tiles_x;
        for (uint32_t ch = 0; ch < 3; ch++) {
            const auto fp = bf16_tile_to_fp32(&result_bf16[(3 * t + ch) * TILE_H * TILE_W]);
            for (uint32_t dev = 0; dev < TILE_H * TILE_W; dev++) {
                const uint32_t imgpos = tbl[dev];
                const uint32_t i = imgpos / TILE_W;
                const uint32_t j = imgpos % TILE_W;
                const uint32_t y = ty * TILE_H + i;
                const uint32_t x = tx * TILE_W + j;
                if (y < image_h && x < image_w) {
                    img[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[dev];
                }
            }
        }
    }
    return img;
}

static double process_frame_mb(
    DeviceContext& ctx,
    const std::vector<uint32_t>& mb_counts,
    const std::vector<uint32_t>& mb_coeff_off,
    const std::vector<float>& mb_coeff_stream,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    std::vector<float>& image_out) {
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;
    const uint32_t total_pairs = static_cast<uint32_t>(mb_coeff_stream.size() / COEFF_LANES_PER_GAUSSIAN);

    // LPT cost = pairs per tile = coeff_off[t+1] - coeff_off[t].
    std::vector<float> cost_f32(num_tiles + 1);
    for (uint32_t t = 0; t <= num_tiles; t++) {
        cost_f32[t] = static_cast<float>(mb_coeff_off[t]);
    }
    const TileAssignment assign = build_tile_assignment(cost_f32, num_tiles, num_cores);

    // Allocate DRAM buffers.
    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };
    const size_t counts_bytes = static_cast<size_t>(num_tiles) * mb::COUNTS_PAGE_BYTES;
    const size_t coeff_bytes = std::max<size_t>(1, total_pairs) * mb::COEFF_ROW_BYTES_MB;
    auto buf_counts    = make_dram(counts_bytes, mb::COUNTS_PAGE_BYTES);
    auto buf_coeff_off = make_dram((static_cast<size_t>(num_tiles) + 1) * sizeof(uint32_t), sizeof(uint32_t));
    auto buf_coeff     = make_dram(coeff_bytes, mb::COEFF_ROW_BYTES_MB);
    auto buf_xramp     = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
    auto buf_yramp     = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
    auto buf_out       = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
    auto buf_tile_ids  = make_dram(assign.tile_id_buffer_bytes_padded, TILE_IDS_PAGE_BYTES);

    // Encode payloads.
    std::vector<uint32_t> counts_payload(counts_bytes / 4, 0);  // 32 uint32 per 128B page
    for (uint32_t t = 0; t < num_tiles; t++) {
        for (uint32_t m = 0; m < NUM_MICROBLOCKS; m++) {
            counts_payload[t * (mb::COUNTS_PAGE_BYTES / 4) + m] =
                mb_counts[static_cast<size_t>(t) * NUM_MICROBLOCKS + m];
        }
    }
    std::vector<uint32_t> coeff_payload(coeff_bytes / 4, 0);
    for (uint32_t e = 0; e < total_pairs; e++) {
        std::memcpy(
            reinterpret_cast<uint8_t*>(coeff_payload.data()) + static_cast<size_t>(e) * mb::COEFF_ROW_BYTES_MB,
            &mb_coeff_stream[static_cast<size_t>(e) * COEFF_LANES_PER_GAUSSIAN],
            COEFF_LANES_PER_GAUSSIAN * sizeof(float));
    }
    std::vector<uint32_t> coeff_off_u32(mb_coeff_off);
    auto xramp = make_ramp(/*is_x=*/true);
    auto yramp = make_ramp(/*is_x=*/false);

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t counts_addr     = static_cast<uint32_t>(buf_counts->address());
    const uint32_t coeff_addr      = static_cast<uint32_t>(buf_coeff->address());
    const uint32_t coeff_off_addr  = static_cast<uint32_t>(buf_coeff_off->address());
    const uint32_t xramp_addr      = static_cast<uint32_t>(buf_xramp->address());
    const uint32_t yramp_addr      = static_cast<uint32_t>(buf_yramp->address());
    const uint32_t out_addr        = static_cast<uint32_t>(buf_out->address());
    const uint32_t tile_ids_addr   = static_cast<uint32_t>(buf_tile_ids->address());
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                const uint32_t start = assign.per_core_offset[core_index];
                const uint32_t count = assign.per_core_count[core_index];
                SetRuntimeArgs(program, ctx.reader, core, {
                    counts_addr, coeff_addr, coeff_off_addr, xramp_addr, yramp_addr,
                    tile_ids_addr, start, count, num_tiles,
                });
                SetRuntimeArgs(program, ctx.compute, core, {count});
                SetRuntimeArgs(program, ctx.writer, core, {
                    out_addr, tile_ids_addr, start, count,
                });
                core_index++;
            }
        }
    }

    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> output_zero(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_out,       output_zero);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_counts,    counts_payload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_coeff_off, coeff_off_u32);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_coeff,     coeff_payload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_xramp,     xramp);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_yramp,     yramp);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_tile_ids,  assign.tile_id_buffer_padded);
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, buf_out, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();

    image_out = tiles_to_image_mb(result_bf16, num_tiles, tiles_x, image_h, image_w);
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

}  // namespace mb

namespace gsplat_tt {

namespace {
std::unique_ptr<DeviceContext> g_ctx;
std::unique_ptr<DeviceContext> g_ctx_mb;
}  // namespace

bool device_ready() { return g_ctx != nullptr || g_ctx_mb != nullptr; }

double blend_mb_from_payload(
    const std::vector<uint32_t>& mb_counts,
    const std::vector<uint32_t>& mb_coeff_off,
    const std::vector<float>& mb_coeff_stream,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    std::vector<float>& image_out) {
    if (!g_ctx_mb) {
        // Initialize the shared MeshDevice BEFORE constructing the context: the
        // DeviceContext default-constructs a MeshWorkload member, which would
        // otherwise create an uninitialized MetalContext (root_dir unset) if no
        // device stage ran first.
        (void)gsplat_tt::device_state::get_device();
        g_ctx_mb = std::make_unique<DeviceContext>(::mb::init_device_context_mb());
    }
    return ::mb::process_frame_mb(
        *g_ctx_mb, mb_counts, mb_coeff_off, mb_coeff_stream,
        static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
        static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
        image_out);
}

static DeviceContext& ensure_device() {
    if (!g_ctx) {
        g_ctx = std::make_unique<DeviceContext>(init_device_context());
    }
    return *g_ctx;
}

static bool use_device_blend_kernel() {
    const char* v = std::getenv("GSPLAT_TT_DEVICE_KERNEL");
    return v != nullptr && v[0] == '1';
}

// tt-001b host path: microblock-major blend from attribute_packs + mb_stream_local.
// Matches cpu_cpp_mb algorithm until device kernel Stage 3 lands.
static double process_frame_cpu_host(const FrameBufferInputs& f) {
    const auto t_start = std::chrono::steady_clock::now();

    if (!f.use_microblock_payload()) {
        throw std::runtime_error("host blend requires microblock payload");
    }
    if (f.packs_f32.size() % 9 != 0) {
        throw std::runtime_error("invalid attribute_packs size");
    }

    const int image_h = static_cast<int>(f.image_h);
    const int image_w = static_cast<int>(f.image_w);
    const std::size_t num_entries = f.packs_f32.size() / 9;

    std::vector<int64_t> mb_header_i64(f.mb_header_u32->begin(), f.mb_header_u32->end());
    std::vector<int64_t> mb_stream_local_i64(f.mb_stream_u32->begin(), f.mb_stream_u32->end());

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    gsplat_cpu::ThreadPool pool(hw);
    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock_from_packs(
        f.packs_f32.data(),
        mb_header_i64.data(),
        mb_stream_local_i64.data(),
        num_entries,
        mb_stream_local_i64.size(),
        image_h,
        image_w,
        32,
        pool);

    f.image_out = std::move(result.image);

    const auto t_end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

double blend_from_payload(
    const std::vector<float>& packs_f32,
    const std::vector<float>& offsets_f32,
    const std::vector<float>& px_f32,
    const std::vector<float>& py_f32,
    const std::vector<float>& coeff_f32,
    const std::vector<uint32_t>& mb_header_u32,
    const std::vector<uint32_t>& mb_stream_u32,
    int image_height,
    int image_width,
    std::vector<float>& image_out) {
    // Empty coeff/mb payloads mean "no microblock shadow" (full-tile blend);
    // pass nullptr so use_microblock_payload() reports false.
    const bool has_mb = !coeff_f32.empty() && !mb_header_u32.empty() &&
                        !mb_stream_u32.empty();
    FrameBufferInputs f{
        packs_f32,
        offsets_f32,
        px_f32,
        py_f32,
        has_mb ? &coeff_f32 : nullptr,
        has_mb ? &mb_header_u32 : nullptr,
        has_mb ? &mb_stream_u32 : nullptr,
        static_cast<uint32_t>(image_height),
        static_cast<uint32_t>(image_width),
        image_out,
    };
    if (use_device_blend_kernel()) {
        return process_frame(ensure_device(), f);
    }
    return process_frame_cpu_host(f);
}

void device_shutdown() {
    // NOTE: do NOT close mesh_device here. gsplat_tt::device_state owns the
    // single MeshDevice instance; it closes once in
    // device_state::shutdown() (called from tt_device_shutdown after every
    // per-stage shutdown). Closing here too would double-free the SHM
    // tracker entries (observed 2026-05-28 as "double free or corruption
    // (fasttop)" during process exit).
    g_ctx.reset();
    g_ctx_mb.reset();
}

}  // namespace gsplat_tt
