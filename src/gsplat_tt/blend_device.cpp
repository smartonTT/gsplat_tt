// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-process host driver for gsplat_tt blend (amendment-002 tt-001a).
// Refactored from tt-metal alpha_blend.cpp: no daemon, no .npy IPC.

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <cstdio>
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

#include "gsplat_tt/host_tracy.hpp"

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

    // RESIDENT blend scratch (GSPLAT_TT_RESIDENT_BLEND): persistent per-context
    // DRAM buffers reused across frames (the render intermediates themselves —
    // attrs/ids — live in device_state and are read over NoC, not reallocated).
    // xramp/yramp are constant; upload once. out/tile_ids grow on demand.
    std::shared_ptr<distributed::MeshBuffer> res_xramp;
    std::shared_ptr<distributed::MeshBuffer> res_yramp;
    std::shared_ptr<distributed::MeshBuffer> res_out;
    std::shared_ptr<distributed::MeshBuffer> res_tile_ids;
    uint32_t res_out_tiles = 0;
    size_t res_tile_ids_bytes = 0;
    bool res_ramp_uploaded = false;

    // SFPU microblock-cull (GSPLAT_TT_SFPU_CULL) persistent scratch: the
    // per-candidate 32-bit mask buffer (grow-on-demand) + a one-time-uploaded
    // flag for the box-origin ramps (res_xramp/res_yramp reused as box ramps in
    // the cull context).
    std::shared_ptr<distributed::MeshBuffer> res_masks;
    size_t res_masks_bytes = 0;
    // Per-tile PAGE-ALIGNED base offset (in mask elements) into res_masks. Cull
    // writer + blend reader index masks by cull_base[tile]+pos so every DRAM
    // write lands on a 16-element (64B) boundary.
    std::shared_ptr<distributed::MeshBuffer> res_mask_base;
    size_t res_mask_base_bytes = 0;

    // Stage C2 (GSPLAT_TT_BLEND_PAYLOAD): contiguous per-candidate blend payload
    // (64B rows, global-candidate-index) produced by the pack pass and streamed
    // sequentially by the payload blend reader. Grow-on-demand like res_masks.
    std::shared_ptr<distributed::MeshBuffer> res_payload;
    size_t res_payload_bytes = 0;
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

// §8.4 Lever 2 (iter 15): hand the per-tile SFPU cull masks from the cull
// program to the blend program through a RESIDENT-L1 buffer instead of the
// cull_masks DRAM round-trip. cull_masks becomes an L1-interleaved buffer
// (same 64B/16-elem page layout + per-tile page-aligned base), the cull writer
// (writer_microblock_cull / fused_tile_writer) writes masks into it, a Finish()
// is placed between the cull and blend enqueues, and the blend reader pops them.
//
// GATED OFF BY DEFAULT (opt in with GSPLAT_TT_L1_MASKS=1). The masks land in L1
// BIT-IDENTICALLY (GSPLAT_TT_SFPU_CULL_DEBUG: 0 mismatches, full 63.85 dB), but
// the iteration's premise — that the per-candidate MB_CULL_SPIN is a DRAM-bank
// write-settle artifact removable by an L1 handoff — is FALSIFIED on device:
//   spin   0    256   384    448    512    (GSPLAT_TT_CULL_SPIN, L1, 1 view)
//   dB    30.1  40.4  43.7   50.9   63.85
//   blend 136   143   167    178    191  ms
// The gate (>=63.6) is only met at spin≈512, where blend (190.9 ms) EQUALS the
// DRAM baseline (191.4 ms): the per-candidate settle window is required for the
// reader to consume the freshly NoC-read mask page (read-completion after
// noc_async_read_barrier), NOT for DRAM write-settle — so it is unchanged by
// moving masks DRAM->L1, and the round-trip removal yields no blend win (mask
// DRAM traffic ~0.4 GB was already hidden behind the spin + the 1.9 GB random
// attr gather). Kept in-tree gated for future investigation; see the iter-15
// report. Next lever: Stage C2 contiguous payload (kills the random gather and
// lets masks ride the sequential stream), GSPLAT_TT_BLEND_PAYLOAD scaffold.
inline bool l1_masks_enabled() {
    const char* v = std::getenv("GSPLAT_TT_L1_MASKS");
    return v != nullptr && v[0] == '1';
}

// Stage C2 (GSPLAT_TT_BLEND_PAYLOAD): a separate pack pass writes a contiguous
// per-candidate blend payload (attrs + microblock mask) and the blend reader
// streams it SEQUENTIALLY, killing the ~1.9 GB random attr gather + the
// cull_masks DRAM round-trip + the per-candidate MB_CULL_SPIN. The pack pass
// does the gather + scalar cull once; the SFPU cull pass is bypassed.
inline bool payload_enabled() {
    const char* v = std::getenv("GSPLAT_TT_BLEND_PAYLOAD");
    return v != nullptr && v[0] == '1';
}

static void build_program_and_workload_mb(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    // DEVCULL: kernel computes conic + microblock mask on-core; needs two
    // reader-private scratch CBs (ids page + attr page gather). Determined by
    // env at program-build time and consumed only by the devcull reader.
    auto env_on = [](const char* n) {
        const char* v = std::getenv(n);
        return v != nullptr && v[0] == '1';
    };
    const bool resident_blend = env_on("GSPLAT_TT_RESIDENT_BLEND");
    // Soft-float inline cull on BRISC is NOT the ideal path — only when explicitly
    // requested (FUSED_TILE gather production). Ideal path uses SFPU cull_masks.
    const bool dev_cull = resident_blend && env_on("GSPLAT_TT_MB_DEVCULL");
    // Resident reader always ships raw 2D cov in coeff rows 0..2; compute must
    // derive the conic on SFPU (MB_DEVCONIC). Without this, SFPU-only ideal path
    // mis-interprets raw cov as pre-folded A,B,C (~15 dB).
    const bool dev_conic =
        dev_cull || env_on("GSPLAT_TT_MB_DEVCONIC") || (resident_blend && env_on("GSPLAT_TT_SFPU_CULL"));
    // SFPU CULL: the 32-bit microblock mask is precomputed on the SFPU (separate
    // cull pass) and kept resident in cull_masks; the blend reader then just
    // reads the mask (pure integer) instead of running the soft-float cull.
    const bool sfpu_cull = resident_blend && env_on("GSPLAT_TT_SFPU_CULL");
    const bool resident_reader = resident_blend && (dev_cull || sfpu_cull);
    // S1 (GSPLAT_TT_BLEND_AOS): the blend reader fetches each candidate's attrs
    // as ONE contiguous 64B AoS record (proj_m_blendrec, emitted by the gather
    // stage) instead of 7-9 random per-component SoA pages. Layers on top of the
    // SFPU-cull reader (mask still read from cull_masks); byte-identical fields.
    // Default ON; set GSPLAT_TT_BLEND_AOS=0 to fall back to the SoA gather.
    auto env_off = [](const char* n) {
        const char* v = std::getenv(n);
        return v != nullptr && v[0] == '0';
    };
    const bool blend_aos = sfpu_cull && !env_off("GSPLAT_TT_BLEND_AOS");
    // T2/T3 (GSPLAT_TT_TILE_BUCKET): serve each tile from its DENSE L1-resident
    // record bucket (sort_tile_recs) — bulk-load once, STABLE depth-sort in L1,
    // emit coeff rows from L1 (NO per-candidate attr gather). Layers on the AoS
    // SFPU-cull reader; masks still read from cull_masks (same depth order).
    const bool tile_bucket = blend_aos && [] {
        const char* v = std::getenv("GSPLAT_TT_TILE_BUCKET");
        if (v != nullptr) {
            return v[0] == '1';
        }
        if (const char* ft = std::getenv("GSPLAT_TT_FUSED_TILE"); ft && ft[0] == '1') {
            return false;
        }
        return std::getenv("GSPLAT_TT_SFPU_CULL") != nullptr &&
               std::getenv("GSPLAT_TT_SFPU_CULL")[0] == '1' &&
               std::getenv("GSPLAT_TT_RESIDENT_BLEND") != nullptr &&
               std::getenv("GSPLAT_TT_RESIDENT_BLEND")[0] == '1';
    }();
    // GSPLAT_TT_TILE_L1 (default OFF): bulk-load each tile's whole cull_masks
    // region into L1 ONCE per tile (one barrier), read masks from L1 in the
    // candidate loop, and DROP the per-candidate MB_CULL_SPIN. The spin (~89 ms
    // of blend) is a per-candidate read-completion settle for the per-chunk mask
    // NoC read; a per-tile bulk read with a single barrier removes the need.
    const bool tile_mask_l1 = sfpu_cull && env_on("GSPLAT_TT_TILE_L1");
    // Stage C2: the blend reader streams a pre-packed contiguous payload instead
    // of gathering attrs + reading cull_masks. Takes precedence over sfpu_cull
    // for reader selection (the mask rides the payload row).
    const bool payload = resident_blend && payload_enabled();

    cb_cfg(CB_XRAMP, RAMP_TILE_BYTES, 2, DataFormat::Float32);
    cb_cfg(CB_YRAMP, RAMP_TILE_BYTES, 2, DataFormat::Float32);
    cb_cfg(CB_MB_COEFF, COEFF_ROW_BYTES_MB, 8, DataFormat::Float32);
    cb_cfg(CB_MB_COUNTS, COUNTS_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_cfg(CB_OUT, TILE_BYTES_BF16, 6, DataFormat::Float16_b);
    if (resident_reader) {
        constexpr uint32_t CB_SCR_IDS = 4;
        constexpr uint32_t CB_SCR_ATTR = 5;
        cb_cfg(CB_SCR_IDS, 64, 2, DataFormat::UInt32);
        if (resident_blend) {
            // Pipelined double-buffered batched gather scratch: 2 chunks x 16
            // gaussians x 9 SoA pages (64B each) = 18432B. The resident reader
            // issues a whole chunk's reads ahead of one barrier and overlaps
            // cull of chunk K with the in-flight reads of chunk K+1.
            constexpr uint32_t GATHER_PAGES = 2u * 16u * 9u;  // 288 pages
            cb_cfg(CB_SCR_ATTR, 64, GATHER_PAGES, DataFormat::Float32);
        } else {
            cb_cfg(CB_SCR_ATTR, COEFF_ROW_BYTES_MB, 2, DataFormat::Float32);
        }
        if (sfpu_cull) {
            // Reader-private double-buffered cull_masks scratch (2 buffers x 2
            // pages x 64B = 256B): the 2-page mask window for each ids chunk is
            // prefetched alongside the chunk's attr reads and read back with a
            // pure integer load. cull_masks is per-tile page-aligned, so a <=16
            // chunk's masks span at most two 64B pages.
            constexpr uint32_t CB_SCR_MASK = 6;
            cb_cfg(CB_SCR_MASK, 256, 1, DataFormat::UInt32);
        }
        if (resident_blend) {
            constexpr uint32_t CB_CORE_TILES = 7;
            cb_cfg(CB_CORE_TILES, 64, 1, DataFormat::UInt32);
        }
        if (tile_mask_l1) {
            // Whole-tile cull_masks region resident in L1. Max hero tile is
            // ~26.5k candidates; size to MAX_TILE_ENTRIES (32768) uint32 = 128 KB
            // (2048 x 64B pages), well under per-core L1. Used as flat scratch.
            constexpr uint32_t CB_TILE_MASKS = 8;
            constexpr uint32_t MAX_TILE_MASK_PAGES = 32768u / 16u;  // 2048
            cb_cfg(CB_TILE_MASKS, 64, MAX_TILE_MASK_PAGES, DataFormat::UInt32);
        }
        if (tile_bucket) {
            // L1-resident dense record bucket (CB_BUCKET) + sort scratch (CB_BSORT:
            // in_idx[FIT] + out_idx[FIT] + counts[256]). FIT tunable; default 8192
            // candidates (512 KB bucket + ~66 KB scratch). Tiles above FIT fall
            // back to the gather path.
            constexpr uint32_t CB_BUCKET = 9;
            constexpr uint32_t CB_BSORT  = 10;
            uint32_t fit = 8192u;
            if (const char* f = std::getenv("GSPLAT_TT_BUCKET_FIT")) {
                const uint32_t v = static_cast<uint32_t>(std::atoi(f));
                if (v >= 16u) fit = v;
            }
            constexpr uint32_t CB_BMASK = 11;
            cb_cfg(CB_BUCKET, 64, fit, DataFormat::Float32);              // fit x 64B records
            cb_cfg(CB_BSORT, 4, 2u * fit + 256u, DataFormat::UInt32);     // idxA+idxB+counts
            cb_cfg(CB_BMASK, 64, (fit + 15u) / 16u + 1u, DataFormat::UInt32);  // whole-tile masks
        }
    }

    // Accessor count: resident devcull reader binds 12 DRAM-interleaved buffers
    // (proj_m_a/b/c/px/py/opacity/colors + sort_sorted_ids + sort_tile_ranges +
    // xramp/yramp/tile_ids); +2 (cull_masks, cull_mask_base) under SFPU cull;
    // uploaded paths 6.
    // Stage C2 sequential payload reader: 6 DRAM-interleaved accessors (ranges,
    // xramp, yramp, tile_ids, lpt_meta, payload). No SoA gather, no cull_masks.
    // +1 accessor (proj_m_blendrec, index 15) under S1 AoS.
    const int num_reader_accessors =
        payload ? 6
                : (resident_reader ? (sfpu_cull ? (blend_aos ? (tile_bucket ? 18 : 16) : 15) : 13)
                                   : 6);
    // cull_masks is reader accessor index 13 (after a,b,c,px,py,op,col, ids,
    // ranges, xramp,yramp,tile_ids, lpt_meta). Under the L1 mask handoff it is
    // an L1-interleaved buffer; everything else stays DRAM-interleaved.
    const bool l1_masks = sfpu_cull && !payload && l1_masks_enabled();
    constexpr int kCullMasksReaderIdx = 13;
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < num_reader_accessors; i++) {
        if (l1_masks && i == kCullMasksReaderIdx) {
            TensorAccessorArgs::create_l1_interleaved().append_to(reader_ct);
        } else {
            TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
        }
    }
    const char* reader_src =
        payload  ? OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_alpha_blend_mb_payload.cpp"
        : resident_reader
            ? OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_alpha_blend_mb_devcull.cpp"
            : OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_alpha_blend_mb.cpp";
    std::map<std::string, std::string> reader_defines;
    if (resident_blend) {
        reader_defines["MB_RESIDENT"] = "1";
    }
    if (payload) {
        reader_defines["MB_PAYLOAD"] = "1";
        if (std::getenv("GSPLAT_TT_PAYLOAD_DEBUG") != nullptr) {
            reader_defines["MB_PAYLOAD_DEBUG"] = "1";
        }
        // Optional per-chunk settle spin for the sequential-read spin/PSNR sweep
        // (default OFF; sequential reads with a chunk-covering barrier need none).
        const char* sp = std::getenv("GSPLAT_TT_CULL_SPIN");
        if (sp != nullptr && sp[0] != '\0' && std::string(sp) != "0") {
            reader_defines["MB_PAY_SPIN"] = std::string(sp);
        }
    }
    if (sfpu_cull && !payload) {
        reader_defines["MB_SFPU_CULL"] = "1";
        if (std::getenv("GSPLAT_TT_SFPU_CULL_DEBUG") != nullptr) {
            reader_defines["MB_SFPU_CULL_DEBUG"] = "1";
        }
        if (std::getenv("GSPLAT_TT_FUSE_AB") != nullptr) {
            reader_defines["FUSE_AB"] = "1";
        }
        if (tile_mask_l1) {
            // L1-resident whole-tile masks: the candidate loop reads masks from
            // L1, so there is NO per-candidate NoC read. The masks still cross the
            // NoC ONCE per tile (bulk DRAM->L1); the bulk-read barrier can return
            // before the DMA physically lands all pages in L1, so a ONE-TIME
            // per-tile settle (MB_TILE_MASK_SETTLE) after the barrier lets them
            // land before the candidate loop consumes them. This is ~30x cheaper
            // than the old per-candidate spin (1 settle/tile vs 1/candidate).
            reader_defines["MB_TILE_MASK_L1"] = "1";
            const char* st = std::getenv("GSPLAT_TT_TILE_L1_SETTLE");
            std::string settle = st ? std::string(st) : std::string("4096");
            if (settle != "0") {
                reader_defines["MB_TILE_MASK_SETTLE"] = settle;
            }
            if (const char* bt = std::getenv("GSPLAT_TT_TILE_L1_BATCH")) {
                if (bt[0] != '\0') reader_defines["MB_TILE_MASK_BATCH"] = std::string(bt) + "u";
            }
        } else {
            // Per-candidate settle spin after the mask-page read barrier. The
            // mask page lands in scratch, but consuming it immediately after
            // noc_async_read_barrier() races the read landing -> dropped
            // microblocks (PSNR ~30). The default 512 is past the single-view
            // knee (63.85 dB). This is a READ-COMPLETION window, NOT a DRAM
            // write-settle artifact: an L1 mask buffer (GSPLAT_TT_L1_MASKS=1)
            // needs the SAME ~512 to hold the gate. Override with
            // GSPLAT_TT_CULL_SPIN ("0" disables the spin entirely).
            const char* sp = std::getenv("GSPLAT_TT_CULL_SPIN");
            std::string spin = sp ? std::string(sp) : std::string("512");
            if (spin != "0") {
                reader_defines["MB_CULL_SPIN"] = spin;
            }
        }
        if (blend_aos) {
            reader_defines["MB_BLEND_AOS"] = "1";
        }
        if (tile_bucket) {
            reader_defines["MB_TILE_BUCKET"] = "1";
            if (const char* f = std::getenv("GSPLAT_TT_BUCKET_FIT")) {
                if (f[0] != '\0') reader_defines["MB_BUCKET_FIT"] = std::string(f) + "u";
            }
            if (env_on("GSPLAT_TT_BUCKET_DBG_INLINE")) {
                reader_defines["MB_BUCKET_DBG_INLINE"] = "1";
            }
            if (env_on("GSPLAT_TT_BUCKET_DBG_NOSORT")) {
                reader_defines["MB_BUCKET_DBG_NOSORT"] = "1";
            }
            if (env_on("GSPLAT_TT_BUCKET_NO_FALLBACK_SPIN")) {
                reader_defines["MB_BUCKET_NO_FALLBACK_SPIN"] = "1";
            }
            if (env_on("GSPLAT_TT_BUCKET_PREFETCH_MASK")) {
                reader_defines["MB_BUCKET_PREFETCH_MASK"] = "1";
            }
            // ROUTE C: the sort-stage cull baked the keep mask into record word
            // 10, so the L1 bucket path reads it directly (spin-free, no
            // cull_masks DRAM round-trip). Disables both the per-candidate spin
            // and the CB_BMASK bulk load on the bucket path.
            if (env_on("GSPLAT_TT_BUCKET_MASK")) {
                reader_defines["MB_BUCKET_MASK"] = "1";
            }
            if (env_on("GSPLAT_TT_BUCKET_MASK_DEBUG")) {
                reader_defines["MB_BUCKET_MASK_DEBUG"] = "1";
            }
            // Diagnostic: keep the bucket-cull RMW running, but make the blend
            // recompute the mask inline instead of reading the baked recp[10].
            if (env_on("GSPLAT_TT_BUCKET_FORCE_INLINE")) {
                reader_defines["MB_BUCKET_FORCE_INLINE"] = "1";
            }
            if (const char* s = std::getenv("GSPLAT_TT_BUCKET_EMIT_SPIN")) {
                if (s[0] != '\0') reader_defines["MB_BUCKET_EMIT_SPIN"] = s;
            }
            // A/B probe: reader stamps seq+checksum into the coeff row (row[11/12])
            // so the compute can prove whether a fast-fed row is read STALE/TORN
            // (HYPOTHESIS A) or delivered intact (HYPOTHESIS B).
            if (env_on("GSPLAT_TT_BUCKET_AB_PROBE")) {
                reader_defines["MB_BUCKET_AB_PROBE"] = "1";
            }
            // Principled fix for the bucket emit->blend producer/consumer L1
            // visibility race: a `fence` orders the row stores before cb_push_back
            // (producer) and invalidates the stale cached CB-slot line before the
            // blend reads it (consumer). Replaces the fragile MB_BUCKET_EMIT_SPIN.
            // PROVEN NECESSARY (iter-29) for the L1-resident bucket path to hit the
            // gate (without it the fast-producer race caps at ~42 dB), so it is
            // DEFAULT-ON whenever the bucket path is active; opt out with
            // GSPLAT_TT_BUCKET_CB_FENCE=0. (Never set for production's FUSED_TILE
            // path — this whole block is inside `if (tile_bucket)`.)
            if (const char* cf = std::getenv("GSPLAT_TT_BUCKET_CB_FENCE");
                cf == nullptr || cf[0] != '0') {
                reader_defines["MB_BUCKET_CB_FENCE"] = "1";
            }
        }
    }
    ctx.reader = CreateKernel(
        program,
        reader_src,
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
            .defines = reader_defines,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_XRAMP] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_YRAMP] = UnpackToDestMode::UnpackToDestFp32;

    std::map<std::string, std::string> mb_defines;
    if (const char* dbg = std::getenv("GSPLAT_TT_MB_DEBUG")) {
        // e.g. GSPLAT_TT_MB_DEBUG=ANALYTIC -> -DMB_DEBUG_ANALYTIC
        mb_defines["MB_DEBUG_" + std::string(dbg)] = "1";
    }
    // DEVCONIC: kernel derives the conic A,B,C from raw cov at row load.
    // DEVCULL (implies DEVCONIC): the device reader computes the microblock mask.
    if (dev_conic) {
        mb_defines["MB_DEVCONIC"] = "1";
    }
    if (resident_reader) {
        mb_defines["MB_DEVCULL"] = "1";
    }
    if (resident_blend) {
        mb_defines["MB_RESIDENT"] = "1";
    }
    if (std::getenv("GSPLAT_TT_COEFF_DEBUG") != nullptr) {
        mb_defines["MB_COEFF_DEBUG"] = "1";
    }
    // A/B probe + CB-fence fix on the compute (consumer) side — must mirror the
    // reader defines so the seq/checksum verify and the read-side fence compile in.
    if (env_on("GSPLAT_TT_BUCKET_AB_PROBE")) {
        mb_defines["MB_BUCKET_AB_PROBE"] = "1";
    }
    // Mirror the reader's DEFAULT-ON CB_FENCE for the bucket path (consumer side
    // mailbox ack). Only when the bucket path is active; opt out with
    // GSPLAT_TT_BUCKET_CB_FENCE=0. Production (FUSED_TILE, tile_bucket=false) is
    // never affected.
    if (const char* cf = std::getenv("GSPLAT_TT_BUCKET_CB_FENCE");
        tile_bucket && (cf == nullptr || cf[0] != '0')) {
        mb_defines["MB_BUCKET_CB_FENCE"] = "1";
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
    if (resident_blend) {
        TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    }
    std::map<std::string, std::string> writer_defines;
    if (resident_blend) {
        writer_defines["MB_RESIDENT"] = "1";
    }
    if (std::getenv("GSPLAT_TT_FUSE_AB") != nullptr) {
        writer_defines["FUSE_AB"] = "1";
    }
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
            .defines = writer_defines,
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

static void tiles_to_image_mb_into(
    const std::vector<uint16_t>& result_bf16,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    float* image_out) {
    const auto& tbl = mb_perm_img_of_dev();
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
                    image_out[(static_cast<size_t>(y) * image_w + x) * 3 + ch] = fp[dev];
                }
            }
        }
    }
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
    // Gaussian-major rows are a full 64B page each (16 words: 10 coeff + mask + pad).
    constexpr uint32_t GM_LANES = mb::COEFF_ROW_BYTES_MB / 4;  // 16
    const uint32_t total_pairs = static_cast<uint32_t>(mb_coeff_stream.size() / GM_LANES);

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
    // mb_coeff_stream ALREADY has the exact 64B-per-row (GM_LANES-word) byte
    // layout the device expects, so upload it directly. The old code zero-init'd
    // a 200MB coeff_payload and memcpy'd the whole stream into it -- a redundant
    // ~100ms (50ms zero-init + 50ms copy) for no reason. Only the empty-frame
    // edge case needs a placeholder page (buf_coeff is sized max(1,total_pairs)).
    static const std::vector<float> kEmptyCoeffPage(GM_LANES, 0.0f);
    const std::vector<float>& coeff_upload =
        mb_coeff_stream.empty() ? kEmptyCoeffPage : mb_coeff_stream;
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

    const bool timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> output_zero(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_out,       output_zero);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_counts,    counts_payload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_coeff_off, coeff_off_u32);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_coeff,     coeff_upload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_xramp,     xramp);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_yramp,     yramp);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_tile_ids,  assign.tile_id_buffer_padded);
    std::chrono::steady_clock::time_point t_upload = t_start;
    if (timing) {
        distributed::Finish(*ctx.cq);  // force all H2D uploads to complete
        t_upload = std::chrono::steady_clock::now();
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::chrono::steady_clock::time_point t_exec = t_upload;
    if (timing) {
        distributed::Finish(*ctx.cq);  // force workload (blend kernels) to complete
        t_exec = std::chrono::steady_clock::now();
    }
    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, buf_out, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();

    if (timing) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double coeff_mb = static_cast<double>(coeff_bytes) / (1024.0 * 1024.0);
        std::fprintf(stderr,
            "[BLEND_SPLIT] upload=%.1f (coeff=%.1fMB rows=%u) exec=%.1f readback=%.1f total=%.1f ms\n",
            ms(t_start, t_upload), coeff_mb, total_pairs,
            ms(t_upload, t_exec), ms(t_exec, t_end), ms(t_start, t_end));
    }

    image_out = tiles_to_image_mb(result_bf16, num_tiles, tiles_x, image_h, image_w);
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

// Device-cull blend frame: uploads compact per-gaussian attrs + per-tile
// candidate id lists; the reader computes the conic + microblock mask on-core.
constexpr uint32_t ATTR_PAGE_BYTES = 64;  // kMbAttrLanes (16 fp32), 9 used
constexpr uint32_t IDS_PAGE_BYTES = 64;   // 16 uint32 ids per page

static double process_frame_mb_devcull(
    DeviceContext& ctx,
    const std::vector<float>& attrs,         // M * 16
    const std::vector<uint32_t>& ids,        // P
    const std::vector<uint32_t>& ids_off,    // num_tiles + 1
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    std::vector<float>& image_out) {
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;

    // LPT cost = candidate count per tile = ids_off[t+1] - ids_off[t].
    std::vector<float> cost_f32(num_tiles + 1);
    for (uint32_t t = 0; t <= num_tiles; t++) {
        cost_f32[t] = static_cast<float>(ids_off[t]);
    }
    const TileAssignment assign = build_tile_assignment(cost_f32, num_tiles, num_cores);

    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };

    const size_t attrs_bytes = std::max<size_t>(ATTR_PAGE_BYTES, attrs.size() * sizeof(float));
    // Pad ids up to whole 64B pages (TensorAccessor reads whole pages).
    const size_t ids_count = std::max<size_t>(1, ids.size());
    const size_t ids_pages = (ids_count + (IDS_PAGE_BYTES / 4) - 1) / (IDS_PAGE_BYTES / 4);
    const size_t ids_bytes = ids_pages * IDS_PAGE_BYTES;

    auto buf_attrs    = make_dram(attrs_bytes, ATTR_PAGE_BYTES);
    auto buf_ids      = make_dram(ids_bytes, IDS_PAGE_BYTES);
    auto buf_ids_off  = make_dram((static_cast<size_t>(num_tiles) + 1) * sizeof(uint32_t), sizeof(uint32_t));
    auto buf_xramp    = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
    auto buf_yramp    = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
    auto buf_out      = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
    auto buf_tile_ids = make_dram(assign.tile_id_buffer_bytes_padded, TILE_IDS_PAGE_BYTES);

    // Pad payloads to their page multiples.
    static const std::vector<float> kEmptyAttr(ATTR_PAGE_BYTES / 4, 0.0f);
    const std::vector<float>& attrs_upload = attrs.empty() ? kEmptyAttr : attrs;
    std::vector<uint32_t> ids_padded(ids_bytes / 4, 0);
    std::copy(ids.begin(), ids.end(), ids_padded.begin());
    std::vector<uint32_t> ids_off_u32(ids_off);
    auto xramp = make_ramp(/*is_x=*/true);
    auto yramp = make_ramp(/*is_x=*/false);

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t attrs_addr    = static_cast<uint32_t>(buf_attrs->address());
    const uint32_t ids_addr      = static_cast<uint32_t>(buf_ids->address());
    const uint32_t ids_off_addr  = static_cast<uint32_t>(buf_ids_off->address());
    const uint32_t xramp_addr    = static_cast<uint32_t>(buf_xramp->address());
    const uint32_t yramp_addr    = static_cast<uint32_t>(buf_yramp->address());
    const uint32_t out_addr      = static_cast<uint32_t>(buf_out->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(buf_tile_ids->address());
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &contrib_floor, 4);
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                const uint32_t start = assign.per_core_offset[core_index];
                const uint32_t count = assign.per_core_count[core_index];
                SetRuntimeArgs(program, ctx.reader, core, {
                    attrs_addr, ids_addr, ids_off_addr, xramp_addr, yramp_addr,
                    tile_ids_addr, start, count, tiles_x, floor_bits,
                    cull_disabled ? 1u : 0u,
                });
                SetRuntimeArgs(program, ctx.compute, core, {count});
                SetRuntimeArgs(program, ctx.writer, core, {
                    out_addr, tile_ids_addr, start, count,
                });
                core_index++;
            }
        }
    }

    const bool timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> output_zero(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_out,      output_zero);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_attrs,    attrs_upload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_ids,      ids_padded);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_ids_off,  ids_off_u32);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_xramp,    xramp);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_yramp,    yramp);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, buf_tile_ids, assign.tile_id_buffer_padded);
    std::chrono::steady_clock::time_point t_upload = t_start;
    if (timing) {
        distributed::Finish(*ctx.cq);
        t_upload = std::chrono::steady_clock::now();
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::chrono::steady_clock::time_point t_exec = t_upload;
    if (timing) {
        distributed::Finish(*ctx.cq);
        t_exec = std::chrono::steady_clock::now();
    }
    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, buf_out, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();

    if (timing) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double attr_mb = static_cast<double>(attrs_bytes) / (1024.0 * 1024.0);
        const double ids_mb = static_cast<double>(ids_bytes) / (1024.0 * 1024.0);
        std::fprintf(stderr,
            "[BLEND_SPLIT] upload=%.1f (attrs=%.1fMB ids=%.1fMB) exec=%.1f readback=%.1f total=%.1f ms\n",
            ms(t_start, t_upload), attr_mb, ids_mb,
            ms(t_upload, t_exec), ms(t_exec, t_end), ms(t_start, t_end));
    }

    image_out = tiles_to_image_mb(result_bf16, num_tiles, tiles_x, image_h, image_w);
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

namespace {

constexpr uint32_t SORT_META_ELEMS_PER_PAGE = 16;

struct ResidentSortLpt {
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;
    std::shared_ptr<distributed::MeshBuffer> tile_ids_buf;
    bool ok = false;
};

static ResidentSortLpt resident_sort_lpt_handles() {
    namespace ds = gsplat_tt::device_state;
    ResidentSortLpt r;
    r.tile_ids_buf = ds::get_buffer("sort_lpt_tile_ids");
    r.ok = static_cast<bool>(r.tile_ids_buf && ds::get_buffer("sort_lpt_meta"));
    return r;
}

// One D2H of sort_lpt_meta for kernels that still take (start,count) host args
// (compute, writer, cull). Blend reader reads meta on-device; this shrinks the
// hot path and will go away when those kernels are updated (plan C2+D+E fuse).
static ResidentSortLpt load_resident_sort_lpt(
    distributed::MeshCommandQueue* cq, uint32_t num_cores) {
    namespace ds = gsplat_tt::device_state;
    ResidentSortLpt r = resident_sort_lpt_handles();
    if (!r.ok) {
        return r;
    }
    auto meta = ds::get_buffer("sort_lpt_meta");
    r.per_core_offset.assign(num_cores, 0);
    r.per_core_count.assign(num_cores, 0);
    const uint32_t meta_pad =
        ((num_cores * 2 + SORT_META_ELEMS_PER_PAGE - 1) / SORT_META_ELEMS_PER_PAGE) *
        SORT_META_ELEMS_PER_PAGE;
    std::vector<uint32_t> mbuf(meta_pad, 0);
    distributed::EnqueueReadMeshBuffer(*cq, mbuf, meta, true);
    for (uint32_t c = 0; c < num_cores; c++) {
        r.per_core_offset[c] = mbuf[c * 2 + 0];
        r.per_core_count[c] = mbuf[c * 2 + 1];
    }
    return r;
}

}  // namespace

// RESIDENT device-cull blend frame (GSPLAT_TT_RESIDENT_BLEND): reads the
// per-gaussian attributes from the device-resident per-component SoA proj_m_*
// buffers and the candidate ids/ranges from resident sort_sorted_ids /
// sort_tile_ranges (all registered in device_state by the on-device gather +
// sort stages). NOTHING about attrs/ids is uploaded per frame — LPT tile lists
// are reused from the sort stage (sort_lpt_*). Returns elapsed ms; sets *ok=false
// (no work done) if any required resident buffer is absent.
static double process_frame_mb_devcull_resident(
    DeviceContext& ctx,
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    float* image_out,
    bool* ok) {
    namespace ds = gsplat_tt::device_state;
    auto buf_a   = ds::get_buffer("proj_m_a");
    auto buf_b   = ds::get_buffer("proj_m_b");
    auto buf_c   = ds::get_buffer("proj_m_c");
    auto buf_px  = ds::get_buffer("proj_m_px");
    auto buf_py  = ds::get_buffer("proj_m_py");
    auto buf_op  = ds::get_buffer("proj_m_opacity");
    auto buf_col = ds::get_buffer("proj_m_colors");
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    if (!buf_a || !buf_b || !buf_c || !buf_px || !buf_py || !buf_op || !buf_col ||
        !buf_ids || !buf_rng) {
        if (ok) *ok = false;
        return 0.0;
    }
    if (ok) *ok = true;

    // SFPU cull: the blend reader reads the precomputed 32-bit mask from the
    // resident cull_masks buffer (registered by the cull pass) instead of
    // running the soft-float cull. Built with the matching MB_SFPU_CULL define.
    const bool payload = mb::payload_enabled();
    const char* sfpu_cull_env = std::getenv("GSPLAT_TT_SFPU_CULL");
    const bool sfpu_cull = !payload && sfpu_cull_env != nullptr && sfpu_cull_env[0] == '1';
    const char* blend_aos_env = std::getenv("GSPLAT_TT_BLEND_AOS");
    const bool blend_aos = sfpu_cull && !(blend_aos_env != nullptr && blend_aos_env[0] == '0');
    const bool tile_bucket = blend_aos && sfpu_cull && [] {
        const char* v = std::getenv("GSPLAT_TT_TILE_BUCKET");
        if (v != nullptr) {
            return v[0] == '1';
        }
        const char* ft = std::getenv("GSPLAT_TT_FUSED_TILE");
        return !(ft != nullptr && ft[0] == '1');
    }();
    uint32_t tile_recs_addr = 0;
    uint32_t bucket_meta_addr = 0;
    if (tile_bucket) {
        auto buf_recs = ds::get_buffer("sort_tile_recs");
        auto buf_bm   = ds::get_buffer("sort_bucket_meta");
        if (!buf_recs || !buf_bm) {
            if (ok) *ok = false;
            return 0.0;
        }
        tile_recs_addr = static_cast<uint32_t>(buf_recs->address());
        bucket_meta_addr = static_cast<uint32_t>(buf_bm->address());
    }
    uint32_t blendrec_addr = 0;
    if (blend_aos) {
        auto buf_brec = gsplat_tt::device_state::get_buffer("proj_m_blendrec");
        if (!buf_brec) {
            if (ok) *ok = false;
            return 0.0;
        }
        blendrec_addr = static_cast<uint32_t>(buf_brec->address());
    }
    uint32_t payload_addr = 0;
    if (payload) {
        auto buf_payload = gsplat_tt::device_state::get_buffer("blend_payload");
        if (!buf_payload) {
            if (ok) *ok = false;
            return 0.0;
        }
        payload_addr = static_cast<uint32_t>(buf_payload->address());
    }
    uint32_t cull_masks_addr = 0;
    uint32_t cull_base_addr = 0;
    if (sfpu_cull) {
        auto buf_masks = gsplat_tt::device_state::get_buffer("cull_masks");
        if (buf_masks) {
            cull_masks_addr = static_cast<uint32_t>(buf_masks->address());
        }
        auto buf_base = gsplat_tt::device_state::get_buffer("cull_mask_base");
        if (buf_base) {
            cull_base_addr = static_cast<uint32_t>(buf_base->address());
        }
    }

    const ResidentSortLpt lpt = resident_sort_lpt_handles();
    if (!lpt.ok) {
        if (ok) *ok = false;
        return 0.0;
    }
    auto meta_buf = ds::get_buffer("sort_lpt_meta");
    const uint32_t lpt_meta_addr =
        meta_buf ? static_cast<uint32_t>(meta_buf->address()) : 0u;
    // Reader/compute/writer all read LPT on-device (no sort_lpt_meta D2H).

    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };

    // Persistent scratch: ramps (constant), output + tile-id list (grow-on-demand).
    if (!ctx.res_xramp) {
        ctx.res_xramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_yramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_ramp_uploaded = false;
    }
    if (!ctx.res_out || ctx.res_out_tiles < num_tiles) {
        ctx.res_out = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
        ctx.res_out_tiles = num_tiles;
    }

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t a_addr     = static_cast<uint32_t>(buf_a->address());
    const uint32_t b_addr     = static_cast<uint32_t>(buf_b->address());
    const uint32_t c_addr     = static_cast<uint32_t>(buf_c->address());
    const uint32_t px_addr    = static_cast<uint32_t>(buf_px->address());
    const uint32_t py_addr    = static_cast<uint32_t>(buf_py->address());
    const uint32_t op_addr    = static_cast<uint32_t>(buf_op->address());
    const uint32_t col_addr   = static_cast<uint32_t>(buf_col->address());
    const uint32_t ids_addr   = static_cast<uint32_t>(buf_ids->address());
    const uint32_t rng_addr   = static_cast<uint32_t>(buf_rng->address());
    const uint32_t xramp_addr = static_cast<uint32_t>(ctx.res_xramp->address());
    const uint32_t yramp_addr = static_cast<uint32_t>(ctx.res_yramp->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(lpt.tile_ids_buf->address());
    const uint32_t out_addr   = static_cast<uint32_t>(ctx.res_out->address());
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &contrib_floor, 4);
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                std::vector<uint32_t> reader_args;
                if (payload) {
                    reader_args = {
                        rng_addr, xramp_addr, yramp_addr, tile_ids_addr,
                        lpt_meta_addr, core_index, payload_addr,
                    };
                } else {
                    reader_args = {
                        a_addr, b_addr, c_addr, px_addr, py_addr, op_addr, col_addr,
                        ids_addr, rng_addr, xramp_addr, yramp_addr,
                        tile_ids_addr, lpt_meta_addr, core_index, tiles_x, floor_bits,
                        cull_disabled ? 1u : 0u,
                    };
                    if (sfpu_cull) {
                        reader_args.push_back(cull_masks_addr);  // arg 17
                        reader_args.push_back(cull_base_addr);   // arg 18
                        if (blend_aos) {
                            reader_args.push_back(blendrec_addr);  // arg 19
                            if (tile_bucket) {
                                reader_args.push_back(tile_recs_addr);    // arg 20
                                reader_args.push_back(bucket_meta_addr);  // arg 21
                            }
                        }
                    }
                }
                SetRuntimeArgs(program, ctx.reader, core, reader_args);
                SetRuntimeArgs(program, ctx.compute, core, {0u});
                SetRuntimeArgs(program, ctx.writer, core, {
                    out_addr, tile_ids_addr, lpt_meta_addr, core_index,
                });
                core_index++;
            }
        }
    }

    const bool timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> output_zero(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_out, output_zero);
    // Constant ramps: upload once, then reuse the resident copy every frame.
    // DIAG(iter18): GSPLAT_TT_FORCE_RAMP=1 re-uploads every frame to test whether
    // the payload pack pass perturbs res_xramp/res_yramp (would recover the gate).
    if (std::getenv("GSPLAT_TT_FORCE_RAMP") != nullptr) {
        ctx.res_ramp_uploaded = false;
    }
    if (!ctx.res_ramp_uploaded) {
        auto xramp = make_ramp(/*is_x=*/true);
        auto yramp = make_ramp(/*is_x=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_xramp, xramp);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_yramp, yramp);
        ctx.res_ramp_uploaded = true;
    }
    std::chrono::steady_clock::time_point t_upload = t_start;
    if (timing) {
        GSPLAT_HOST_ZONE("host_finish_blend");
        distributed::Finish(*ctx.cq);
        t_upload = std::chrono::steady_clock::now();
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::chrono::steady_clock::time_point t_exec = t_upload;
    if (timing) {
        GSPLAT_HOST_ZONE("host_finish_blend");
        distributed::Finish(*ctx.cq);
        t_exec = std::chrono::steady_clock::now();
    }
    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, ctx.res_out, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();

    if (timing) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr,
            "[BLEND_SPLIT] upload=%.1f (attrs=0.0MB ids=0.0MB resident) exec=%.1f readback=%.1f total=%.1f ms\n",
            ms(t_start, t_upload), ms(t_upload, t_exec), ms(t_exec, t_end), ms(t_start, t_end));
        unsigned long pay_a = 0, pay_sz = 0;
        if (auto bp = gsplat_tt::device_state::get_buffer("blend_payload")) {
            pay_a = static_cast<unsigned long>(bp->address());
            pay_sz = static_cast<unsigned long>(bp->size());
        }
        std::fprintf(stderr,
            "[BLEND_ADDR] cull_masks=0x%lx out=0x%lx(sz=0x%lx) tile_ids=0x%lx xramp=0x%lx yramp=0x%lx payload=0x%lx(sz=0x%lx)\n",
            static_cast<unsigned long>(cull_masks_addr),
            static_cast<unsigned long>(ctx.res_out->address()),
            static_cast<unsigned long>(ctx.res_out->size()),
            static_cast<unsigned long>(tile_ids_addr),
            static_cast<unsigned long>(ctx.res_xramp->address()),
            static_cast<unsigned long>(ctx.res_yramp->address()),
            pay_a, pay_sz);
    }

    tiles_to_image_mb_into(result_bf16, num_tiles, tiles_x, image_h, image_w, image_out);
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

// ===========================================================================
// SFPU microblock-cull pass (GSPLAT_TT_SFPU_CULL).
//
// A dedicated 3-kernel program (reader_microblock_cull / microblock_cull_compute
// / writer_microblock_cull) that runs BEFORE the resident blend. It computes the
// 32-bit microblock mask of every depth-sorted candidate ON THE SFPU (one 32-lane
// vector == all 32 microblocks of a gaussian) and stores it resident in
// cull_masks (registered in device_state). The blend reader then reads the mask
// with a pure integer load -> the expensive soft-float constrained-min cull is
// gone from the data mover.
// ===========================================================================
namespace cull {

constexpr uint32_t CB_BOX_OX     = 0;   // fp32 box x-origin ramp tile (constant)
constexpr uint32_t CB_BOX_OY     = 1;   // fp32 box y-origin ramp tile (constant)
constexpr uint32_t CB_CULL_COEFF = 2;   // 64B coeff row per gaussian (6 used)
constexpr uint32_t CB_CULL_COUNTS= 3;   // per-tile [L, tx_pix, ty_pix]
constexpr uint32_t CB_SCR_IDS    = 4;   // reader-private ids/ranges scratch
constexpr uint32_t CB_SCR_ATTR   = 5;   // reader-private gather scratch
constexpr uint32_t CB_MASK_SCR   = 6;   // writer-private mask packing scratch
constexpr uint32_t CB_KEEP       = 16;  // compute -> writer keep tiles

constexpr uint32_t COEFF_ROW_BYTES = 64;
constexpr uint32_t COUNTS_PAGE_BYTES = 64;
constexpr uint32_t MASKS_PAGE_BYTES = 64;   // 16 u32 per page
constexpr uint32_t GATHER_FIELDS = 6;

// Intra-vector CB-linear position for (gaussian g == SFPU vector, microblock m).
// MUST match perm() in writer_microblock_cull.cpp exactly.
inline uint32_t perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

// Constant box-origin ramp tile: at CB-linear position perm(g,m) store
// microblock m's tile-local box origin ((m&3)*8 for x, (m>>2)*4 for y). After
// the compute kernel's copy_tile->math->pack_tile round-trip (CB-linear-
// identity), the keep flag for (vector g, microblock m) lands back at the same
// position, which the writer reads to assemble the 32-bit mask.
static std::vector<uint32_t> make_box_ramp(bool is_x) {
    std::vector<uint32_t> r(TILE_H * TILE_W, 0);
    for (uint32_t g = 0; g < 32; ++g) {
        for (uint32_t m = 0; m < 32; ++m) {
            const uint32_t dev = perm(g, m);
            const float v = is_x ? static_cast<float>((m & 3u) * 8u)
                                 : static_cast<float>((m >> 2) * 4u);
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            r[dev] = bits;
        }
    }
    return r;
}

static void build_program_and_workload(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_cfg(CB_BOX_OX, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
    cb_cfg(CB_BOX_OY, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
    cb_cfg(CB_CULL_COEFF, COEFF_ROW_BYTES, 32, DataFormat::Float32);
    cb_cfg(CB_CULL_COUNTS, COUNTS_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_cfg(CB_SCR_IDS, 64, 2, DataFormat::UInt32);
    cb_cfg(CB_SCR_ATTR, 64, 16u * GATHER_FIELDS, DataFormat::Float32);  // 96 pages
    cb_cfg(CB_MASK_SCR, 128, 1, DataFormat::UInt32);
    cb_cfg(CB_KEEP, mb::RAMP_TILE_BYTES, 4, DataFormat::Float32);

    // Reader: 11 DRAM-interleaved accessors (a,b,c,px,py,op, ids, ranges,
    // box_ox, box_oy, tile_ids).
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 11; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    std::map<std::string, std::string> cull_reader_defines;
    if (std::getenv("GSPLAT_TT_CULL_VALS") != nullptr) {
        cull_reader_defines["CULL_READER_DEBUG"] = "1";
    }
    const bool cull_selfcheck = std::getenv("GSPLAT_TT_CULL_SELFCHECK") != nullptr;
    if (cull_selfcheck) {
        cull_reader_defines["CULL_DEBUG_REF"] = "1";
    }
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_microblock_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
            .defines = cull_reader_defines,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_BOX_OX] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_BOX_OY] = UnpackToDestMode::UnpackToDestFp32;
    std::map<std::string, std::string> cull_compute_defines;
    if (std::getenv("GSPLAT_TT_CULL_KEEPALL") != nullptr) {
        cull_compute_defines["CULL_DEBUG_KEEPALL"] = "1";
    }
    if (std::getenv("GSPLAT_TT_CULL_RAMP") != nullptr) {
        cull_compute_defines["CULL_DEBUG_RAMP"] = "1";
    }
    if (std::getenv("GSPLAT_TT_CULL_RECIP") != nullptr) {
        cull_compute_defines["CULL_DEBUG_RECIP"] = "1";
    }
    if (std::getenv("GSPLAT_TT_CULL_RAMPY") != nullptr) {
        cull_compute_defines["CULL_DEBUG_RAMPY"] = "1";
    }
    const bool cull_emit_m2 = std::getenv("GSPLAT_TT_CULL_EMIT_M2") != nullptr || cull_selfcheck;
    if (cull_emit_m2) {
        cull_compute_defines["CULL_DEBUG_EMIT_M2"] = "1";
    }
    if (std::getenv("GSPLAT_TT_CULL_EMIT_M2") != nullptr) {
        cull_compute_defines["CULL_DEBUG_VALS"] = "1";
    }
    const bool cull_dumpbox = std::getenv("GSPLAT_TT_CULL_DUMPBOX") != nullptr;
    if (cull_dumpbox) {
        cull_compute_defines["CULL_DEBUG_EMIT_BOX"] = "1";
    }
    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/microblock_cull_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
            .defines = cull_compute_defines,
        });

    // Writer: 4 accessors (cull_masks, ranges, tile_ids, cull_mask_base).
    // cull_masks (index 0) is L1-interleaved under the L1 mask handoff so the
    // writer's masks land in the same resident-L1 buffer the blend reader pops.
    std::vector<uint32_t> writer_ct;
    const bool l1_masks = mb::l1_masks_enabled();
    for (int i = 0; i < 4; i++) {
        if (l1_masks && i == 0) {
            TensorAccessorArgs::create_l1_interleaved().append_to(writer_ct);
        } else {
            TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
        }
    }
    std::map<std::string, std::string> cull_writer_defines;
    if (cull_emit_m2) {
        cull_writer_defines["CULL_DEBUG_EMIT_M2"] = "1";
    }
    if (std::getenv("GSPLAT_TT_CULL_WDUMP") != nullptr) {
        cull_writer_defines["CULL_DEBUG_WRITE"] = "1";
    }
    if (std::getenv("GSPLAT_TT_CULL_KVAL") != nullptr) {
        cull_writer_defines["CULL_KVAL"] = "1";
    }
    if (cull_dumpbox) {
        cull_writer_defines["CULL_DEBUG_DUMPBOX"] = "1";
    }
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_microblock_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
            .defines = cull_writer_defines,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context() {
    DeviceContext ctx;
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    cull::build_program_and_workload(ctx);
    return ctx;
}

// Allocate cull_masks + cull_mask_base from `ctx`'s allocator (the BLEND
// context g_ctx_mb) and upload the per-tile PAGE-ALIGNED base prefix. MUST be
// called before the cull pass runs. Allocating from the blend context (rather
// than the separate cull context) is REQUIRED for correctness: the two
// contexts' DRAM allocations are not mutually tracked, so cull-context buffers
// overlapped the blend's res_out and were clobbered mid-blend. base[t] is the
// prefix sum of round_up(per_tile_count[i], 16); each tile is padded to a whole
// 64B/16-elem page so every writer DRAM write is 16-element aligned.
static bool ensure_resident_buffers(
    DeviceContext& ctx,
    uint32_t num_tiles) {
    namespace ds = gsplat_tt::device_state;
    auto published_base = ds::get_buffer("cull_mask_base");
    if (!published_base) {
        return false;
    }
    uint32_t pipe_p = 0;
    uint32_t pipe_mask = 0;
    size_t total_elems = 16;
    if (ds::get_sort_blend_pipe_scalars(&pipe_p, &pipe_mask)) {
        total_elems = std::max<size_t>(16, static_cast<size_t>(pipe_mask));
    } else {
        auto buf_pk = ds::get_buffer("sort_P_kept");
        if (!buf_pk) {
            return false;
        }
        std::vector<uint32_t> pkept(16, 0);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, pkept, buf_pk, true);
        total_elems = std::max<size_t>(16, static_cast<size_t>(pkept[1]));
        (void)pipe_p;
    }
    const size_t masks_bytes = (total_elems / 16) * MASKS_PAGE_BYTES;
    // L1 (iter 15) vs DRAM cull_masks. Interleaved L1 spreads the ~12 MB of
    // masks across the grid's L1 banks (~95 KB/core over ~130 cores), well
    // within the ~1.5 MB/core L1 alongside the cull+blend CBs. Same 64B/16-elem
    // page layout and per-tile page-aligned base as DRAM, so all indexing
    // (cull_mask_base[tile] + p) is byte-identical — only the bank kind changes.
    const BufferType masks_bt = mb::l1_masks_enabled() ? BufferType::L1 : BufferType::DRAM;
    auto make_masks_buf = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = masks_bt};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };
    if (!ctx.res_masks || ctx.res_masks_bytes < masks_bytes) {
        ctx.res_masks = make_masks_buf(masks_bytes, MASKS_PAGE_BYTES);
        ctx.res_masks_bytes = masks_bytes;
        ds::register_buffer("cull_masks", ctx.res_masks);
    }
    // cull_mask_base is published by sort (H2D at sort time); blend only aliases it.
    ctx.res_mask_base = published_base;
    ctx.res_mask_base_bytes = static_cast<size_t>(published_base->size());
    return true;
}

// Run the SFPU cull pass over all candidates and register the resident
// cull_masks buffer in device_state. Returns elapsed ms (or 0 if a required
// resident buffer is missing, leaving *ok=false).
static double process_frame(
    DeviceContext& ctx,
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    bool* ok) {
    namespace ds = gsplat_tt::device_state;
    auto buf_a   = ds::get_buffer("proj_m_a");
    auto buf_b   = ds::get_buffer("proj_m_b");
    auto buf_c   = ds::get_buffer("proj_m_c");
    auto buf_px  = ds::get_buffer("proj_m_px");
    auto buf_py  = ds::get_buffer("proj_m_py");
    auto buf_op  = ds::get_buffer("proj_m_opacity");
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    if (!buf_a || !buf_b || !buf_c || !buf_px || !buf_py || !buf_op || !buf_ids || !buf_rng) {
        if (ok) *ok = false;
        return 0.0;
    }
    if (ok) *ok = true;

    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;

    auto buf_pk = ds::get_buffer("sort_P_kept");
    if (!buf_pk) {
        if (ok) *ok = false;
        return 0.0;
    }
    std::vector<uint32_t> pkept(16, 0);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, pkept, buf_pk, true);
    const uint64_t total_candidates = pkept[0];
    const uint64_t total_mask_elems = std::max<uint64_t>(16, pkept[1]);

    const ResidentSortLpt lpt = load_resident_sort_lpt(ctx.cq, num_cores);
    if (!lpt.ok) {
        if (ok) *ok = false;
        return 0.0;
    }

    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };

    // Persistent box-origin ramps (constant) + grow-on-demand tile-id list,
    // reused across frames in the cull context.
    if (!ctx.res_xramp) {
        ctx.res_xramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_yramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx.res_ramp_uploaded = false;
    }
    // cull_masks + cull_mask_base are allocated by ensure_resident_buffers() on
    // the BLEND context (g_ctx_mb) so they live in the SAME allocator that owns
    // the blend's res_out / ramps / tile_ids. Allocating them from this separate
    // cull context produced DRAM that the blend allocator did not track, so
    // res_out overlapped cull_masks and clobbered the masks mid-blend (the bug:
    // hero_vs_ref stuck ~29 dB). Here we ONLY read their addresses.
    auto buf_masks_res = ds::get_buffer("cull_masks");
    auto buf_base_res  = ds::get_buffer("cull_mask_base");
    if (!buf_masks_res || !buf_base_res) {
        if (ok) *ok = false;
        return 0.0;
    }
    const size_t masks_bytes = (static_cast<size_t>(total_mask_elems) / 16) * MASKS_PAGE_BYTES;

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t a_addr      = static_cast<uint32_t>(buf_a->address());
    const uint32_t b_addr      = static_cast<uint32_t>(buf_b->address());
    const uint32_t c_addr      = static_cast<uint32_t>(buf_c->address());
    const uint32_t px_addr     = static_cast<uint32_t>(buf_px->address());
    const uint32_t py_addr     = static_cast<uint32_t>(buf_py->address());
    const uint32_t op_addr     = static_cast<uint32_t>(buf_op->address());
    const uint32_t ids_addr    = static_cast<uint32_t>(buf_ids->address());
    const uint32_t rng_addr    = static_cast<uint32_t>(buf_rng->address());
    const uint32_t box_ox_addr = static_cast<uint32_t>(ctx.res_xramp->address());
    const uint32_t box_oy_addr = static_cast<uint32_t>(ctx.res_yramp->address());
    const uint32_t masks_addr  = static_cast<uint32_t>(buf_masks_res->address());
    const uint32_t base_addr   = static_cast<uint32_t>(buf_base_res->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(lpt.tile_ids_buf->address());
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &contrib_floor, 4);
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                const uint32_t start = lpt.per_core_offset[core_index];
                const uint32_t count = lpt.per_core_count[core_index];
                SetRuntimeArgs(program, ctx.reader, core, {
                    a_addr, b_addr, c_addr, px_addr, py_addr, op_addr,
                    ids_addr, rng_addr, box_ox_addr, box_oy_addr,
                    tile_ids_addr, start, count, tiles_x, floor_bits,
                });
                SetRuntimeArgs(program, ctx.compute, core, {
                    count, floor_bits, cull_disabled ? 1u : 0u,
                });
                SetRuntimeArgs(program, ctx.writer, core, {
                    masks_addr, rng_addr, tile_ids_addr, start, count, base_addr,
                });
                core_index++;
            }
        }
    }

    const bool timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;
    // GSPLAT_TT_CULL_PIPELINE: do NOT block the host on the cull's own Finish.
    // The cull and the resident blend share ONE in-order command queue
    // (device_state::command_queue() -> dev->mesh_command_queue()), so the cull
    // program fully executes — writing cull_masks — before the blend program is
    // launched regardless. The Finish here was a pure HOST-side bubble: it
    // serialized the blend's ~110-core host program/runtime-arg setup AFTER the
    // cull's ~80 ms device execution, in a timeline that is ~87% device-idle.
    // Skipping it lets that blend host setup overlap the cull device window.
    // Correctness is unchanged (same CQ, in-order; the blend's first Finish /
    // blocking readback still drains the cull before any host readback).
    const bool pipeline = std::getenv("GSPLAT_TT_CULL_PIPELINE") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    if (!ctx.res_ramp_uploaded) {
        auto bx = make_box_ramp(/*is_x=*/true);
        auto by = make_box_ramp(/*is_x=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_xramp, bx);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, ctx.res_yramp, by);
        ctx.res_ramp_uploaded = true;
    }
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    if (!pipeline) {
        distributed::Finish(*ctx.cq);
    }
    const auto t_end = std::chrono::steady_clock::now();

    const double cull_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (timing && pipeline) {
        std::fprintf(stderr, "[CULL_SPLIT] pipelined (no separate Finish; cull device "
                     "exec folded under the blend host-setup window; candidates=%u)\n",
                     static_cast<unsigned>(total_candidates));
    } else if (timing) {
        std::fprintf(stderr, "[CULL_SPLIT] candidates=%u masks=%.2fMB exec=%.1f ms "
                     "masks_addr=0x%lx end=0x%lx base_addr=0x%lx\n",
                     total_candidates,
                     static_cast<double>(masks_bytes) / (1024.0 * 1024.0), cull_ms,
                     static_cast<unsigned long>(buf_masks_res->address()),
                     static_cast<unsigned long>(buf_masks_res->address() + masks_bytes),
                     static_cast<unsigned long>(buf_base_res->address()));
    }
    return cull_ms;
}

}  // namespace cull

// ===========================================================================
// Stage C2 payload-pack pass (GSPLAT_TT_BLEND_PAYLOAD).
//
// A single dataflow kernel (payload_pack.cpp) that runs BEFORE the blend. Per
// LPT tile it streams the depth-sorted candidate ids, gathers the proj_m_* SoA
// attrs (the SAME random gather the blend reader used to do), runs the scalar
// microblock cull, and writes one CONTIGUOUS 64B row per candidate into
// blend_payload (indexed by global candidate index). The blend reader then
// STREAMS those rows sequentially -> the ~1.9 GB random gather + cull_masks
// round-trip + MB_CULL_SPIN are all gone from the blend. This pass REPLACES the
// separate SFPU cull pass (it bakes the mask into the payload row).
// ===========================================================================
namespace pack {

constexpr uint32_t CB_SCR_IDS  = 0;   // ids / ranges / tile-id scratch
constexpr uint32_t CB_SCR_ATTR = 1;   // double-buffered 9-field gather scratch
constexpr uint32_t CB_SCR_ROW  = 2;   // 64B row assembly scratch
constexpr uint32_t GATHER_FIELDS = 9;

static void build_program_and_workload(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_cfg(CB_SCR_IDS, 64, 2, DataFormat::UInt32);
    // 2 chunks x 16 gaussians x 9 SoA pages (64B each) = 288 pages.
    cb_cfg(CB_SCR_ATTR, 64, 2u * 16u * GATHER_FIELDS, DataFormat::Float32);
    cb_cfg(CB_SCR_ROW, 64, 1, DataFormat::UInt32);

    // 11 DRAM-interleaved accessors: a,b,c,px,py,op,col, ids, ranges, payload,
    // tile_ids.
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 11; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/payload_pack.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context() {
    DeviceContext ctx;
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    pack::build_program_and_workload(ctx);
    return ctx;
}

// Allocate blend_payload from the BLEND context allocator (g_ctx_mb) so it lives
// in the same allocator that owns res_out / ramps (mirroring cull_masks; a
// buffer from a different context's allocator is not mutually tracked and can be
// clobbered). Sized to round_up(P_kept,16) rows of 64B (global-candidate-index).
static bool ensure_payload_buffer(DeviceContext& ctx, uint32_t num_tiles) {
    namespace ds = gsplat_tt::device_state;
    (void)num_tiles;
    // ROOT CAUSE (iter 17-19): the pack/reader index the payload by the GLOBAL
    // candidate index id_start[t] + p, where id_start comes from the resident
    // sort_tile_ranges. The resident sort publishes a PAGE-ALIGNED PADDED layout
    // (sort_device.cpp: padded_cursor += round_up(cnt, 16) per tile), so the max
    // id_end == padded_cursor, which is LARGER than the dense candidate count
    // (sort_P_kept[0]) by up to ~num_tiles*15. Sizing the payload to P_kept (even
    // +pad) left it short by the per-tile padding sum, so pack wrote payload pages
    // PAST the buffer end. Those OOB pages land (via the interleaved accessor) in
    // the next-allocated buffers (res_xramp/res_yramp/res_out), corrupting the
    // coordinate ramps every timed frame -> vertical-stripe + block corruption.
    //
    // The correct, exact index space is sort_sorted_ids itself: it is allocated to
    // EXACTLY padded_cursor elements (ensure_resident_sorted_buffer) and the
    // payload row index == the sort_sorted_ids element index. So size the payload
    // to one 64B row per sort_sorted_ids element (grow-only buffer => its element
    // count is >= the current frame's padded_cursor).
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    if (!buf_ids) {
        return false;
    }
    const size_t rows = std::max<size_t>(16, buf_ids->size() / sizeof(uint32_t));
    const size_t payload_bytes = rows * mb::COEFF_ROW_BYTES_MB;
    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };
    if (!ctx.res_payload || ctx.res_payload_bytes < payload_bytes) {
        ctx.res_payload = make_dram(payload_bytes, mb::COEFF_ROW_BYTES_MB);
        ctx.res_payload_bytes = payload_bytes;
        ds::register_buffer("blend_payload", ctx.res_payload);
    }
    return true;
}

// Run the payload-pack pass over all candidates. Returns elapsed ms (or 0 with
// *ok=false if a required resident buffer is missing).
static double process_frame(
    DeviceContext& ctx,
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    bool* ok) {
    namespace ds = gsplat_tt::device_state;
    auto buf_a   = ds::get_buffer("proj_m_a");
    auto buf_b   = ds::get_buffer("proj_m_b");
    auto buf_c   = ds::get_buffer("proj_m_c");
    auto buf_px  = ds::get_buffer("proj_m_px");
    auto buf_py  = ds::get_buffer("proj_m_py");
    auto buf_op  = ds::get_buffer("proj_m_opacity");
    auto buf_col = ds::get_buffer("proj_m_colors");
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    auto buf_pay = ds::get_buffer("blend_payload");
    if (!buf_a || !buf_b || !buf_c || !buf_px || !buf_py || !buf_op || !buf_col ||
        !buf_ids || !buf_rng || !buf_pay) {
        if (ok) *ok = false;
        return 0.0;
    }
    if (ok) *ok = true;

    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;
    const ResidentSortLpt lpt = load_resident_sort_lpt(ctx.cq, num_cores);
    if (!lpt.ok) {
        if (ok) *ok = false;
        return 0.0;
    }

    Program& program = get_program_for_workload(ctx);
    uint32_t core_index = 0;
    const uint32_t a_addr   = static_cast<uint32_t>(buf_a->address());
    const uint32_t b_addr   = static_cast<uint32_t>(buf_b->address());
    const uint32_t c_addr   = static_cast<uint32_t>(buf_c->address());
    const uint32_t px_addr  = static_cast<uint32_t>(buf_px->address());
    const uint32_t py_addr  = static_cast<uint32_t>(buf_py->address());
    const uint32_t op_addr  = static_cast<uint32_t>(buf_op->address());
    const uint32_t col_addr = static_cast<uint32_t>(buf_col->address());
    const uint32_t ids_addr = static_cast<uint32_t>(buf_ids->address());
    const uint32_t rng_addr = static_cast<uint32_t>(buf_rng->address());
    const uint32_t pay_addr = static_cast<uint32_t>(buf_pay->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(lpt.tile_ids_buf->address());
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &contrib_floor, 4);
    for (const auto& range : ctx.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                const uint32_t start = lpt.per_core_offset[core_index];
                const uint32_t count = lpt.per_core_count[core_index];
                SetRuntimeArgs(program, ctx.reader, core, {
                    a_addr, b_addr, c_addr, px_addr, py_addr, op_addr, col_addr,
                    ids_addr, rng_addr, pay_addr, tile_ids_addr, start, count,
                    tiles_x, floor_bits, cull_disabled ? 1u : 0u,
                });
                core_index++;
            }
        }
    }

    const bool timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    distributed::Finish(*ctx.cq);
    const auto t_end = std::chrono::steady_clock::now();
    const double pack_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (timing) {
        std::fprintf(stderr, "[PACK_SPLIT] exec=%.1f ms payload=0x%lx\n", pack_ms,
                     static_cast<unsigned long>(pay_addr));
    }
    // Diagnostic: read back the packed payload to host and dump the first few
    // rows (global-index 0..N) so the pack write-order can be byte-verified
    // against the consumption layout without DPRINT backpressure. Gated by env.
    if (std::getenv("GSPLAT_TT_PAYLOAD_READBACK") != nullptr) {
        namespace ds = gsplat_tt::device_state;
        auto u2f = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
        auto f2b = [](float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; };
        const size_t rows_total = buf_pay->size() / mb::COEFF_ROW_BYTES_MB;
        std::vector<uint32_t> rb(rows_total * (mb::COEFF_ROW_BYTES_MB / 4), 0xCDCDCDCDu);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, rb, buf_pay, /*blocking=*/true);

        // DIAG(iter18): DETERMINISTIC host cross-check of EVERY packed row field
        // against proj_m_*[gid] (gid = sort_sorted_ids[id_start+p]). The pack and
        // the proven devcull reader derive gid identically, so a correct pack must
        // reproduce every field byte-for-byte. Also histogram the mask popcount.
        auto rdbuf = [&](const char* name) {
            auto b = ds::get_buffer(name);
            std::vector<uint32_t> v(b ? b->size() / 4 : 0, 0u);
            if (b) distributed::EnqueueReadMeshBuffer(*ctx.cq, v, b, /*blocking=*/true);
            return v;
        };
        std::vector<uint32_t> sids(buf_ids->size() / 4, 0u);
        std::vector<uint32_t> rng(buf_rng->size() / 4, 0u);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, sids, buf_ids, /*blocking=*/true);
        distributed::EnqueueReadMeshBuffer(*ctx.cq, rng, buf_rng, /*blocking=*/true);
        const std::vector<uint32_t> pma = rdbuf("proj_m_a");
        const std::vector<uint32_t> pmb = rdbuf("proj_m_b");
        const std::vector<uint32_t> pmc = rdbuf("proj_m_c");
        const std::vector<uint32_t> pmpx = rdbuf("proj_m_px");
        const std::vector<uint32_t> pmpy = rdbuf("proj_m_py");
        const std::vector<uint32_t> pmop = rdbuf("proj_m_opacity");
        const std::vector<uint32_t> pmcol = rdbuf("proj_m_colors");

        size_t checked = 0, gid_mm = 0, oob = 0;
        // per-field mismatch counts: a,b,c,mxl,myl,op,cr,cg,cb
        size_t fmm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        size_t first_field_mm = SIZE_MAX;
        size_t mask_zero = 0, mask_full = 0, pop_sum = 0;
        int dumped = 0;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            if (static_cast<size_t>(t) * 2u + 1u >= rng.size()) break;
            const uint32_t id_start = rng[t * 2u + 0u];
            const uint32_t id_end = rng[t * 2u + 1u];
            if (id_end <= id_start) continue;
            const uint32_t L = id_end - id_start;
            const uint32_t tx = t % tiles_x;
            const uint32_t ty = t / tiles_x;
            const float tx_tile = static_cast<float>(tx * 32u);
            const float ty_tile = static_cast<float>(ty * 32u);
            for (uint32_t p = 0; p < L; ++p) {
                const uint32_t gidx = id_start + p;
                if (static_cast<size_t>(gidx) >= rows_total || static_cast<size_t>(gidx) >= sids.size()) {
                    oob++;
                    continue;
                }
                const uint32_t* row = rb.data() + static_cast<size_t>(gidx) * 16;
                const uint32_t g = sids[gidx];
                checked++;
                if (row[11] != g) {
                    if (first_field_mm == SIZE_MAX) first_field_mm = gidx;
                    gid_mm++;
                }
                const uint32_t e0 = g * 3u;
                const uint32_t exp_mxl = (g < pmpx.size()) ? f2b(u2f(pmpx[g]) - tx_tile) : 0u;
                const uint32_t exp_myl = (g < pmpy.size()) ? f2b(u2f(pmpy[g]) - ty_tile) : 0u;
                const uint32_t exp[9] = {
                    (g < pma.size()) ? pma[g] : 0u,
                    (g < pmb.size()) ? pmb[g] : 0u,
                    (g < pmc.size()) ? pmc[g] : 0u,
                    exp_mxl, exp_myl,
                    (g < pmop.size()) ? pmop[g] : 0u,
                    (e0 + 0u < pmcol.size()) ? pmcol[e0 + 0u] : 0u,
                    (e0 + 1u < pmcol.size()) ? pmcol[e0 + 1u] : 0u,
                    (e0 + 2u < pmcol.size()) ? pmcol[e0 + 2u] : 0u,
                };
                // row indices for the 9 fields: a=0,b=1,c=2,mxl=3,myl=4,op=6,cr=7,cg=8,cb=9
                const int ridx[9] = {0, 1, 2, 3, 4, 6, 7, 8, 9};
                bool any = false;
                for (int f = 0; f < 9; ++f) {
                    if (row[ridx[f]] != exp[f]) {
                        fmm[f]++;
                        any = true;
                    }
                }
                if (any && first_field_mm == SIZE_MAX) first_field_mm = gidx;
                if (any && dumped < 8) {
                    std::fprintf(stderr,
                        "[PAYLOAD_XCHK] MM gidx=%u g=%u | a %.3f/%.3f b %.3f/%.3f c %.3f/%.3f "
                        "mxl %.3f/%.3f op %.4f/%.4f cr %.4f/%.4f mask=0x%x\n",
                        gidx, g, u2f(row[0]), u2f(exp[0]), u2f(row[1]), u2f(exp[1]),
                        u2f(row[2]), u2f(exp[2]), u2f(row[3]), u2f(exp[3]),
                        u2f(row[6]), u2f(exp[5]), u2f(row[7]), u2f(exp[6]), row[10]);
                    dumped++;
                }
                const uint32_t mask = row[10];
                if (mask == 0u) mask_zero++;
                if (mask == 0xFFFFFFFFu) mask_full++;
                pop_sum += static_cast<size_t>(__builtin_popcount(mask));
            }
        }
        std::fprintf(stderr,
            "[PAYLOAD_XCHK] rows_total=%zu checked=%zu gid_mm=%zu oob=%zu first_mm=%zd | "
            "fmm a=%zu b=%zu c=%zu mxl=%zu myl=%zu op=%zu cr=%zu cg=%zu cb=%zu | "
            "mask_zero=%zu mask_full=%zu avg_pop=%.2f\n",
            rows_total, checked, gid_mm, oob, static_cast<ssize_t>(first_field_mm),
            fmm[0], fmm[1], fmm[2], fmm[3], fmm[4], fmm[5], fmm[6], fmm[7], fmm[8],
            mask_zero, mask_full,
            checked ? static_cast<double>(pop_sum) / static_cast<double>(checked) : 0.0);
    }
    return pack_ms;
}

}  // namespace pack

// ===========================================================================
// FUSED_TILE: one command-queue Finish for cull+blend (default ON for resident
// SFPU stack; GSPLAT_TT_FUSED_TILE=0 to opt out). Scaffold: fused_tile_* cull
// triple (on-device sort_lpt_meta) + resident mb blend, back-to-back workloads.
// Next: single program, L1 mask handoff (no cull_masks DRAM), C2 payload fuse.
// ===========================================================================
namespace fused {

static bool enabled() {
    auto env_on = [](const char* n) {
        const char* e = std::getenv(n);
        return e != nullptr && e[0] == '1';
    };
    // FUSED_TILE gather path: opt-in only (GSPLAT_TT_FUSED_TILE=1). Default OFF
    // so the ideal TILE_BUCKET + SFPU cull_masks path is not shadowed.
    const char* v = std::getenv("GSPLAT_TT_FUSED_TILE");
    if (v == nullptr || v[0] != '1') {
        return false;
    }
    return env_on("GSPLAT_TT_MB_DEVCULL") && env_on("GSPLAT_TT_RESIDENT_BLEND") &&
           env_on("GSPLAT_TT_SFPU_CULL");
}

// §8.4 L1 mask handoff: collapse cull+blend into ONE program where each core
// runs cull-then-blend per tile and the 32-bit masks travel writer->reader via
// the L1 CB_TILE_MASKS instead of the cull_masks DRAM round-trip (which removes
// the per-candidate MB_CULL_SPIN).
//
// DEFAULT OFF (opt in with GSPLAT_TT_FUSE_BLEND=1). The single-program fusion is
// implemented end-to-end (host CBs/program + reader/compute/writer kernels) and
// JIT-compiles + binds correctly, but it currently HARD-FAULTS at
// EnqueueMeshWorkload with "Program size (92688) too large for kernel config
// buffer (70656) on TENSIX": merging the full cull SFPU (~38 KB trisc1) and the
// blend SFPU (~22 KB trisc1) into one compute kernel overflows the per-core
// kernel-config ring buffer (= l1_unreserved_base - KERNEL_CONFIG, a fixed HAL
// ceiling). The cull math is `noinline` templates taking uint32 bit-args (vFloat
// can't cross that ABI), so its conic can't be cheaply hoisted, and the proven
// cull regime must not be retuned. Until the merged compute binary is shrunk
// (or masks are handed via a fixed resident-L1 region across TWO small programs
// with a Finish between), the default stays the working two-program scaffold.
static bool fuse_blend() {
    const char* v = std::getenv("GSPLAT_TT_FUSE_BLEND");
    return (v != nullptr && v[0] == '1');
}

// Blend-side CB ids in the fused program (disjoint from the cull set 0-7,16).
namespace fb {
constexpr uint32_t CB_XRAMP      = 8;
constexpr uint32_t CB_YRAMP      = 9;
constexpr uint32_t CB_MB_COEFF   = 10;
constexpr uint32_t CB_MB_COUNTS  = 11;
constexpr uint32_t CB_SCR_ATTR_B = 12;
constexpr uint32_t CB_COLOR_OUT  = 17;
constexpr uint32_t CB_TILE_MASKS = 18;
constexpr uint32_t BLEND_GATHER_PAGES = 16u * 8u;  // 16 gaussians x 8 SoA pages
}  // namespace fb

static void build_fused_program_and_workload(DeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    const bool fuse = fuse_blend();

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_cfg(cull::CB_BOX_OX, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
    cb_cfg(cull::CB_BOX_OY, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
    cb_cfg(cull::CB_CULL_COEFF, cull::COEFF_ROW_BYTES, 32, DataFormat::Float32);
    cb_cfg(cull::CB_CULL_COUNTS, cull::COUNTS_PAGE_BYTES, 2, DataFormat::UInt32);
    cb_cfg(cull::CB_SCR_IDS, 64, 2, DataFormat::UInt32);
    cb_cfg(cull::CB_SCR_ATTR, 64, 16u * cull::GATHER_FIELDS, DataFormat::Float32);
    cb_cfg(cull::CB_MASK_SCR, 128, 1, DataFormat::UInt32);
    cb_cfg(cull::CB_KEEP, mb::RAMP_TILE_BYTES, 4, DataFormat::Float32);
    constexpr uint32_t CB_CORE_TILES = 7;
    cb_cfg(CB_CORE_TILES, 64, 1, DataFormat::UInt32);

    if (fuse) {
        // Blend CBs. CB_TILE_MASKS must buffer a WHOLE tile's masks (⌈L/32⌉
        // pages): the merged compute does cull-all-then-blend-all per tile (cull
        // 5 + blend 6 DST tiles can't be co-resident in 8), so a tile's masks
        // are produced during its cull phase with no consumer until its blend
        // phase. max_tile_n≈25.9k ⇒ ≈809 pages; size generously (override via
        // GSPLAT_TT_FUSE_MASK_PAGES). A too-small depth deadlocks at exec.
        uint32_t mask_pages = 1024;
        if (const char* mp = std::getenv("GSPLAT_TT_FUSE_MASK_PAGES")) {
            const uint32_t v = static_cast<uint32_t>(std::atoi(mp));
            if (v > 0) mask_pages = v;
        }
        cb_cfg(fb::CB_XRAMP, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
        cb_cfg(fb::CB_YRAMP, mb::RAMP_TILE_BYTES, 1, DataFormat::Float32);
        cb_cfg(fb::CB_MB_COEFF, mb::COEFF_ROW_BYTES_MB, 8, DataFormat::Float32);
        cb_cfg(fb::CB_MB_COUNTS, mb::COUNTS_PAGE_BYTES, 2, DataFormat::UInt32);
        cb_cfg(fb::CB_SCR_ATTR_B, 64, fb::BLEND_GATHER_PAGES, DataFormat::Float32);
        cb_cfg(fb::CB_COLOR_OUT, TILE_BYTES_BF16, 6, DataFormat::Float16_b);
        cb_cfg(fb::CB_TILE_MASKS, 128, mask_pages, DataFormat::UInt32);
    }

    std::map<std::string, std::string> fuse_defines;
    if (fuse) fuse_defines["FUSE_BLEND"] = "1";
    if (std::getenv("GSPLAT_TT_FUSE_AB") != nullptr) fuse_defines["FUSE_AB"] = "1";

    std::vector<uint32_t> reader_ct;
    const int reader_accessors = fuse ? 15 : 12;  // +colors,+xramp,+yramp
    for (int i = 0; i < reader_accessors; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    }
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/fused_tile_render.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
            .defines = fuse_defines,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[cull::CB_BOX_OX] = UnpackToDestMode::UnpackToDestFp32;
    u2d[cull::CB_BOX_OY] = UnpackToDestMode::UnpackToDestFp32;
    if (fuse) {
        u2d[fb::CB_XRAMP] = UnpackToDestMode::UnpackToDestFp32;
        u2d[fb::CB_YRAMP] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::map<std::string, std::string> compute_defines = {{"CULL_LPT_CB", "1"}};
    if (fuse) compute_defines["FUSE_BLEND"] = "1";
    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/microblock_cull_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
            .defines = compute_defines,
        });

    std::vector<uint32_t> writer_ct;
    const int writer_accessors = fuse ? 6 : 5;  // +res_out
    // cull_masks (writer accessor index 0) is L1-interleaved under the L1 mask
    // handoff (non-fused path). In the single-program FUSE_BLEND path the masks
    // travel via CB_TILE_MASKS and cull_masks is unused, so the kind is moot.
    const bool l1_masks = mb::l1_masks_enabled();
    for (int i = 0; i < writer_accessors; i++) {
        if (l1_masks && i == 0) {
            TensorAccessorArgs::create_l1_interleaved().append_to(writer_ct);
        } else {
            TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
        }
    }
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/fused_tile_writer.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
            .defines = fuse_defines,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

static DeviceContext init_device_context() {
    DeviceContext ctx;
    ctx.mesh_device = gsplat_tt::device_state::get_device();
    ctx.cq = gsplat_tt::device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_fused_program_and_workload(ctx);
    return ctx;
}

// Cull triple + resident blend with a single Finish() between upload and readback.
static double process_frame_resident(
    DeviceContext& ctx_fused,
    DeviceContext& ctx_blend,
    float contrib_floor,
    bool cull_disabled,
    uint32_t num_tiles,
    uint32_t tiles_x,
    uint32_t image_h,
    uint32_t image_w,
    float* image_out,
    bool* ok) {
    namespace ds = gsplat_tt::device_state;
    auto buf_a   = ds::get_buffer("proj_m_a");
    auto buf_b   = ds::get_buffer("proj_m_b");
    auto buf_c   = ds::get_buffer("proj_m_c");
    auto buf_px  = ds::get_buffer("proj_m_px");
    auto buf_py  = ds::get_buffer("proj_m_py");
    auto buf_op  = ds::get_buffer("proj_m_opacity");
    auto buf_col = ds::get_buffer("proj_m_colors");
    auto buf_ids = ds::get_buffer("sort_sorted_ids");
    auto buf_rng = ds::get_buffer("sort_tile_ranges");
    if (!buf_a || !buf_b || !buf_c || !buf_px || !buf_py || !buf_op || !buf_col ||
        !buf_ids || !buf_rng) {
        if (ok) *ok = false;
        return 0.0;
    }
    auto buf_masks = ds::get_buffer("cull_masks");
    auto buf_base  = ds::get_buffer("cull_mask_base");
    if (!buf_masks || !buf_base) {
        if (ok) *ok = false;
        return 0.0;
    }
    // S1 (GSPLAT_TT_BLEND_AOS): the devcull blend reader (built with MB_BLEND_AOS
    // by build_program) reads one contiguous proj_m_blendrec record/candidate; it
    // needs the buffer's address as reader arg 19. This fused path always runs the
    // sfpu-cull reader, so mirror build_program's default-ON gate (=0 disables).
    const char* blend_aos_env = std::getenv("GSPLAT_TT_BLEND_AOS");
    const bool blend_aos = !(blend_aos_env != nullptr && blend_aos_env[0] == '0');
    uint32_t blendrec_addr = 0;
    if (blend_aos) {
        auto buf_brec = ds::get_buffer("proj_m_blendrec");
        if (!buf_brec) {
            if (ok) *ok = false;
            return 0.0;
        }
        blendrec_addr = static_cast<uint32_t>(buf_brec->address());
    }
    if (ok) *ok = true;

    const ResidentSortLpt lpt = resident_sort_lpt_handles();
    if (!lpt.ok) {
        if (ok) *ok = false;
        return 0.0;
    }
    auto meta_buf = ds::get_buffer("sort_lpt_meta");
    const uint32_t lpt_meta_addr =
        meta_buf ? static_cast<uint32_t>(meta_buf->address()) : 0u;

    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx_blend.mesh_device.get());
    };

    if (!ctx_blend.res_xramp) {
        ctx_blend.res_xramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx_blend.res_yramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx_blend.res_ramp_uploaded = false;
    }
    if (!ctx_blend.res_out || ctx_blend.res_out_tiles < num_tiles) {
        ctx_blend.res_out = make_dram(static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
        ctx_blend.res_out_tiles = num_tiles;
    }
    if (!ctx_fused.res_xramp) {
        ctx_fused.res_xramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx_fused.res_yramp = make_dram(mb::RAMP_TILE_BYTES, mb::RAMP_TILE_BYTES);
        ctx_fused.res_ramp_uploaded = false;
    }

    const uint32_t lpt_tile_ids_addr = static_cast<uint32_t>(lpt.tile_ids_buf->address());
    const uint32_t cull_masks_addr = static_cast<uint32_t>(buf_masks->address());
    const uint32_t cull_base_addr  = static_cast<uint32_t>(buf_base->address());
    const uint32_t floor_bits = [&]() {
        uint32_t bits;
        std::memcpy(&bits, &contrib_floor, 4);
        return bits;
    }();

    const bool fuse = fuse_blend();
    // Blend pixel-center ramps + bf16 output (allocated/uploaded on ctx_blend).
    const uint32_t bxramp_addr = static_cast<uint32_t>(ctx_blend.res_xramp->address());
    const uint32_t byramp_addr = static_cast<uint32_t>(ctx_blend.res_yramp->address());
    const uint32_t out_addr   = static_cast<uint32_t>(ctx_blend.res_out->address());

    Program& prog_fused = get_program_for_workload(ctx_fused);
    uint32_t core_index = 0;
    for (const auto& range : ctx_fused.all_cores.ranges()) {
        for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                CoreCoord core{x, y};
                std::vector<uint32_t> r_args = {
                    static_cast<uint32_t>(buf_a->address()),
                    static_cast<uint32_t>(buf_b->address()),
                    static_cast<uint32_t>(buf_c->address()),
                    static_cast<uint32_t>(buf_px->address()),
                    static_cast<uint32_t>(buf_py->address()),
                    static_cast<uint32_t>(buf_op->address()),
                    static_cast<uint32_t>(buf_ids->address()),
                    static_cast<uint32_t>(buf_rng->address()),
                    static_cast<uint32_t>(ctx_fused.res_xramp->address()),  // box ox ramp
                    static_cast<uint32_t>(ctx_fused.res_yramp->address()),  // box oy ramp
                    lpt_tile_ids_addr,
                    lpt_meta_addr, core_index, tiles_x, floor_bits,
                };
                if (fuse) {
                    r_args.push_back(static_cast<uint32_t>(buf_col->address()));  // arg15 colors
                    r_args.push_back(bxramp_addr);  // arg16 pixel-center x ramp
                    r_args.push_back(byramp_addr);  // arg17 pixel-center y ramp
                }
                SetRuntimeArgs(prog_fused, ctx_fused.reader, core, r_args);
                SetRuntimeArgs(prog_fused, ctx_fused.compute, core, {
                    0u, floor_bits, cull_disabled ? 1u : 0u,
                });
                std::vector<uint32_t> w_args = {
                    cull_masks_addr,
                    static_cast<uint32_t>(buf_rng->address()),
                    lpt_tile_ids_addr,
                    lpt_meta_addr, core_index, cull_base_addr,
                };
                if (fuse) {
                    w_args.push_back(out_addr);  // arg6 res_out
                }
                SetRuntimeArgs(prog_fused, ctx_fused.writer, core, w_args);
                core_index++;
            }
        }
    }

    if (!fuse) {
        Program& prog_blend = get_program_for_workload(ctx_blend);
        core_index = 0;
        const uint32_t a_addr     = static_cast<uint32_t>(buf_a->address());
        const uint32_t b_addr     = static_cast<uint32_t>(buf_b->address());
        const uint32_t c_addr     = static_cast<uint32_t>(buf_c->address());
        const uint32_t px_addr    = static_cast<uint32_t>(buf_px->address());
        const uint32_t py_addr    = static_cast<uint32_t>(buf_py->address());
        const uint32_t op_addr    = static_cast<uint32_t>(buf_op->address());
        const uint32_t col_addr   = static_cast<uint32_t>(buf_col->address());
        const uint32_t ids_addr   = static_cast<uint32_t>(buf_ids->address());
        const uint32_t rng_addr   = static_cast<uint32_t>(buf_rng->address());
        const uint32_t tile_ids_addr = lpt_tile_ids_addr;
        for (const auto& range : ctx_blend.all_cores.ranges()) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; x++) {
                for (auto y = range.start_coord.y; y <= range.end_coord.y; y++) {
                    CoreCoord core{x, y};
                    std::vector<uint32_t> blend_r_args = {
                        a_addr, b_addr, c_addr, px_addr, py_addr, op_addr, col_addr,
                        ids_addr, rng_addr, bxramp_addr, byramp_addr,
                        tile_ids_addr, lpt_meta_addr, core_index, tiles_x, floor_bits,
                        cull_disabled ? 1u : 0u,
                        cull_masks_addr, cull_base_addr,
                    };
                    if (blend_aos) {
                        blend_r_args.push_back(blendrec_addr);  // arg 19
                    }
                    SetRuntimeArgs(prog_blend, ctx_blend.reader, core, blend_r_args);
                    SetRuntimeArgs(prog_blend, ctx_blend.compute, core, {0u});
                    SetRuntimeArgs(prog_blend, ctx_blend.writer, core, {
                        out_addr, tile_ids_addr, lpt_meta_addr, core_index,
                    });
                    core_index++;
                }
            }
        }
    }

    const bool timing = std::getenv("GSPLAT_TT_MB_TIMING") != nullptr;
    const auto t_start = std::chrono::steady_clock::now();
    // res_out is fully overwritten by the blend writer for every tile in the LPT
    // assignment (all num_tiles). Skip the ~6MB zero H2D each frame.
    if (!ctx_blend.res_ramp_uploaded) {
        auto xramp = make_ramp(/*is_x=*/true);
        auto yramp = make_ramp(/*is_x=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx_blend.cq, ctx_blend.res_xramp, xramp);
        distributed::EnqueueWriteMeshBuffer(*ctx_blend.cq, ctx_blend.res_yramp, yramp);
        ctx_blend.res_ramp_uploaded = true;
    }
    if (!ctx_fused.res_ramp_uploaded) {
        auto bx = cull::make_box_ramp(/*is_x=*/true);
        auto by = cull::make_box_ramp(/*is_x=*/false);
        distributed::EnqueueWriteMeshBuffer(*ctx_fused.cq, ctx_fused.res_xramp, bx);
        distributed::EnqueueWriteMeshBuffer(*ctx_fused.cq, ctx_fused.res_yramp, by);
        ctx_fused.res_ramp_uploaded = true;
    }
    const bool sort_publish_piped = ds::sort_publish_pending();
    // §8.4: when fused, cull+blend are ONE program (masks via L1 CB_TILE_MASKS,
    // no cull_masks DRAM round-trip / spin) — enqueue only the single workload.
    distributed::EnqueueMeshWorkload(*ctx_blend.cq, ctx_fused.workload, /*blocking=*/false);
    if (!fuse) {
        // §8.4 Lever 2 (iter 15): with the resident-L1 mask handoff, place a
        // Finish between the cull (ctx_fused) and blend (ctx_blend) programs so
        // the cull writer's L1 mask writes are guaranteed settled before the
        // blend reader pops them. This single Finish/frame REPLACES the ~3.2 M
        // per-candidate MB_CULL_SPIN busy-waits (the spin existed only to let
        // DRAM mask pages settle; L1 is coherent after the writer barrier +
        // this Finish). The DRAM-masks fallback keeps the prior single-Finish
        // back-to-back enqueue (its per-candidate spin handles settle).
        // GSPLAT_TT_TILE_L1: the blend reader bulk-loads each tile's whole
        // cull_masks region DRAM->L1 once and reads masks from L1 (no per-candidate
        // NoC read). The bulk read still crosses the NoC from DRAM, so the cull's
        // freshly written mask pages must be GUARANTEED settled first — a single
        // Finish/frame here does that deterministically and REPLACES the per-
        // candidate MB_CULL_SPIN (which only ever existed to let those DRAM pages
        // settle). One device sync/frame << ~3.2 M per-candidate busy-waits.
        const char* tl1 = std::getenv("GSPLAT_TT_TILE_L1");
        const bool tile_mask_l1 = (tl1 != nullptr && tl1[0] == '1');
        const char* bf = std::getenv("GSPLAT_TT_BUCKET_FINISH");
        const bool bucket_finish = (bf != nullptr && bf[0] == '1');
        if (mb::l1_masks_enabled() || tile_mask_l1 || bucket_finish) {
            distributed::Finish(*ctx_blend.cq);
        }
        distributed::EnqueueMeshWorkload(*ctx_blend.cq, ctx_blend.workload, /*blocking=*/false);
    }
    distributed::Finish(*ctx_blend.cq);
    ds::clear_sort_publish_pending();

    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx_blend.cq, result_bf16, ctx_blend.res_out, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();

    if (timing) {
        std::fprintf(stderr,
            "[FUSED_TILE] upload+exec+readback=%.1f ms (single Finish%s)\n",
            std::chrono::duration<double, std::milli>(t_end - t_start).count(),
            sort_publish_piped ? " sort-publish+cull+blend" : " cull+blend");
    }

    tiles_to_image_mb_into(result_bf16, num_tiles, tiles_x, image_h, image_w, image_out);
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

}  // namespace fused

}  // namespace mb

namespace gsplat_tt {

namespace {
std::unique_ptr<DeviceContext> g_ctx;
std::unique_ptr<DeviceContext> g_ctx_mb;
std::unique_ptr<DeviceContext> g_ctx_cull;   // SFPU microblock-cull pass
std::unique_ptr<DeviceContext> g_ctx_fused;  // GSPLAT_TT_FUSED_TILE cull triple
std::unique_ptr<DeviceContext> g_ctx_pack;   // GSPLAT_TT_BLEND_PAYLOAD pack pass
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

double blend_mb_devcull_from_payload(
    const std::vector<float>& attrs,
    const std::vector<uint32_t>& ids,
    const std::vector<uint32_t>& ids_off,
    float contrib_floor,
    bool cull_disabled,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    std::vector<float>& image_out) {
    if (!g_ctx_mb) {
        (void)gsplat_tt::device_state::get_device();
        g_ctx_mb = std::make_unique<DeviceContext>(::mb::init_device_context_mb());
    }
    return ::mb::process_frame_mb_devcull(
        *g_ctx_mb, attrs, ids, ids_off, contrib_floor, cull_disabled,
        static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
        static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
        image_out);
}

double blend_mb_devcull_resident(
    float contrib_floor,
    bool cull_disabled,
    int num_tiles,
    int tiles_x,
    int image_height,
    int image_width,
    float* image_out,
    bool* device_ok) {
    if (!g_ctx_mb) {
        (void)gsplat_tt::device_state::get_device();
        g_ctx_mb = std::make_unique<DeviceContext>(::mb::init_device_context_mb());
    }
    // Stage C2 (GSPLAT_TT_BLEND_PAYLOAD): pack pass writes a contiguous payload,
    // then the blend reader streams it sequentially. Takes precedence over the
    // fused cull+blend path (the pack bakes the mask into the payload row, so the
    // SFPU cull pass is bypassed). A Finish between pack and blend settles the
    // payload DRAM writes -> the sequential reads need no per-candidate spin.
    if (::mb::payload_enabled()) {
        if (!g_ctx_pack) {
            g_ctx_pack = std::make_unique<DeviceContext>(::mb::pack::init_device_context());
        }
        if (!::mb::pack::ensure_payload_buffer(*g_ctx_mb, static_cast<uint32_t>(num_tiles))) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
        bool pack_ok = false;
        const double pack_ms = ::mb::pack::process_frame(
            *g_ctx_pack, contrib_floor, cull_disabled,
            static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x), &pack_ok);
        if (!pack_ok) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
        const double blend_ms = ::mb::process_frame_mb_devcull_resident(
            *g_ctx_mb, contrib_floor, cull_disabled,
            static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
            static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
            image_out, device_ok);
        return pack_ms + blend_ms;
    }
    if (::mb::fused::enabled()) {
        if (!g_ctx_fused) {
            g_ctx_fused = std::make_unique<DeviceContext>(::mb::fused::init_device_context());
        }
        if (!::mb::cull::ensure_resident_buffers(*g_ctx_mb, static_cast<uint32_t>(num_tiles))) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
        return ::mb::fused::process_frame_resident(
            *g_ctx_fused, *g_ctx_mb, contrib_floor, cull_disabled,
            static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
            static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
            image_out, device_ok);
    }

    // SFPU microblock-cull pass (GSPLAT_TT_SFPU_CULL): precompute the 32-bit
    // masks on the SFPU into the resident cull_masks buffer BEFORE the blend.
    // The blend reader then reads the mask (pure integer) instead of running
    // the soft-float constrained-min cull.
    double cull_ms = 0.0;
    const char* sfpu_cull_env = std::getenv("GSPLAT_TT_SFPU_CULL");
    if (sfpu_cull_env != nullptr && sfpu_cull_env[0] == '1') {
        if (!g_ctx_cull) {
            g_ctx_cull = std::make_unique<DeviceContext>(::mb::cull::init_device_context());
        }
        if (!::mb::cull::ensure_resident_buffers(*g_ctx_mb, static_cast<uint32_t>(num_tiles))) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
        bool cull_ok = false;
        cull_ms = ::mb::cull::process_frame(
            *g_ctx_cull, contrib_floor, cull_disabled,
            static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x), &cull_ok);
        if (!cull_ok) {
            if (device_ok) *device_ok = false;
            return 0.0;
        }
    }
    const double blend_ms = ::mb::process_frame_mb_devcull_resident(
        *g_ctx_mb, contrib_floor, cull_disabled,
        static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tiles_x),
        static_cast<uint32_t>(image_height), static_cast<uint32_t>(image_width),
        image_out, device_ok);
    return cull_ms + blend_ms;
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
    // Leak contexts on shutdown — see sort_device_shutdown().
    (void)g_ctx.release();
    (void)g_ctx_mb.release();
    (void)g_ctx_cull.release();
    (void)g_ctx_fused.release();
    (void)g_ctx_pack.release();
}

}  // namespace gsplat_tt
