// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host driver for the gaussian_splatting alpha-blend kernel.
//
// CLI signatures:
//   metal_example_gaussian_splatting packs.npy offsets.npy px.npy py.npy output.npy [H] [W]
//   metal_example_gaussian_splatting --daemon
//
// Single-shot mode: loads four .npy fixtures, opens the device, JIT-compiles
// kernels, runs once, writes the output, exits. Used by tests and benchmarks.
//
// Daemon mode: opens the device + JIT-compiles kernels once, then reads
// binary FRM1 frame requests from stdin in a loop (24-byte header + 4 fp32
// payloads), writes OK11/ERR1 binary responses on stdout, and exits on
// EOF or "QUIT". Reuses cached DRAM MeshBuffers across frames. Used by the
// interactive viewer to keep the ~3s device init + JIT cost off the per-frame
// path.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
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

using namespace tt;
using namespace tt::tt_metal;
using namespace gsplat;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Binary IPC magic (little-endian on the wire).
constexpr uint32_t IPC_MAGIC_FRM1 = 0x46524D31;  // 'FRM1'
constexpr uint32_t IPC_MAGIC_OK11 = 0x4F4B3131;  // 'OK11'
constexpr uint32_t IPC_MAGIC_ERR1 = 0x45525231;  // 'ERR1'

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

// Cached DRAM buffers reused across daemon frames. Grows geometrically (×1.5)
// when total_entries / offsets / tile_ids exceed current caps; px/py/output
// are reallocated only when (image_h, image_w) changes.
struct BufferCache {
    uint32_t image_h = 0;
    uint32_t image_w = 0;
    uint32_t num_tiles = 0;
    uint32_t max_entries = 0;
    uint32_t max_offsets = 0;
    size_t max_tile_id_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> packs;
    std::shared_ptr<distributed::MeshBuffer> offsets;
    std::shared_ptr<distributed::MeshBuffer> px;
    std::shared_ptr<distributed::MeshBuffer> py;
    std::shared_ptr<distributed::MeshBuffer> output;
    std::shared_ptr<distributed::MeshBuffer> tile_ids;
    // Tracks which tiles had Gaussians last frame for selective output zero-fill.
    std::vector<uint8_t> last_frame_tile_nonempty;
};

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
    BufferCache cache;
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

    auto cb_tile = [&](uint32_t id, uint32_t depth) {
        CircularBufferConfig c(depth * TILE_BYTES_BF16, {{id, DataFormat::Float16_b}});
        c.set_page_size(id, TILE_BYTES_BF16);
        CreateCircularBuffer(program, cores, c);
    };
    auto cb_small = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };

    cb_tile(CB_PX, 2);
    cb_tile(CB_PY, 2);
    cb_small(CB_SCALARS, SCALAR_PACK_PAGE_BYTES, 4, DataFormat::Float32);
    cb_small(CB_TILE_META, META_PAGE_BYTES, 2, DataFormat::UInt32);
    // Depth must be a multiple of 3 (the per-tile batch size) so no
    // single push-of-3 ever straddles the CB wrap. Picking 6 keeps two
    // batches in flight (parity with the previous double-buffering depth).
    cb_tile(CB_COLOR_OUT, 6);

    cb_tile(CB_DX, 2);
    cb_tile(CB_DY, 2);
    cb_tile(CB_DX2, 2);
    cb_tile(CB_DY2, 2);
    cb_tile(CB_DXDY, 2);
    {
        CircularBufferConfig c(3 * TILE_BYTES_BF16, {{CB_Q, DataFormat::Float16_b}});
        c.set_page_size(CB_Q, TILE_BYTES_BF16);
        CreateCircularBuffer(program, cores, c);
    }
    cb_tile(CB_POWER, 2);
    cb_tile(CB_ALPHA, 2);

    cb_tile(CB_CONTRIB, 1);
    cb_tile(CB_ONE_MINUS_ALPHA, 1);
    cb_tile(CB_T_TMP, 1);

    cb_tile(CB_COLOR_R_STATE, 1);
    cb_tile(CB_COLOR_G_STATE, 1);
    cb_tile(CB_COLOR_B_STATE, 1);
    cb_tile(CB_T_STATE, 1);
    cb_tile(CB_SAT_MASK, 1);

    cb_tile(CB_CONST_ZERO, 1);
    cb_tile(CB_CONST_099, 1);
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
    ctx.reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    ctx.compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/compute/alpha_blend_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
        });

    // Writer: 2 TensorAccessorArgs for out + tile_ids.
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "gaussian_splatting/kernels/dataflow/writer_alpha_blend.cpp",
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
    constexpr int device_id = 0;
    ctx.mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    ctx.cq = &ctx.mesh_device->mesh_command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores = CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program_and_workload(ctx);
    return ctx;
}

// ---------------------------------------------------------------------------
// Per-frame work
// ---------------------------------------------------------------------------

struct FrameInputs {
    std::string packs_path;
    std::string offsets_path;
    std::string px_path;
    std::string py_path;
    std::string out_path;
    uint32_t image_h;
    uint32_t image_w;
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
};

// Allocate the 6 DRAM buffers a frame needs. Sizes are derived from the
// scene's total_entries + tile count + the LPT-balanced tile-id list.
// All buffers are RAII via shared_ptr; they free on scope exit.
static FrameDramBuffers allocate_frame_buffers(
    DeviceContext& ctx,
    uint32_t total_entries,
    uint32_t num_tiles,
    size_t offsets_count,
    size_t tile_ids_bytes) {
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
    return b;
}

static uint32_t grow_cap_u32(uint32_t current, uint32_t needed) {
    if (needed <= current) {
        return current;
    }
    if (current == 0) {
        return needed;
    }
    return std::max(needed, current + current / 2);
}

static size_t grow_cap_size(size_t current, size_t needed) {
    if (needed <= current) {
        return current;
    }
    if (current == 0) {
        return needed;
    }
    return std::max(needed, current + current / 2);
}

// Same geometric growth, but rounded up to the buffer's page size so the
// result is a valid MeshBuffer size (`size % page_size == 0`). Geometric
// growth on raw byte counts can land on `current + current/2` that is
// page-aligned for `current` but not for the buffer's `page_size`
// (e.g. current=320, needed=448, page=64 -> grow=480, 480%64=32 -> crash).
static size_t grow_cap_bytes_page_aligned(size_t current, size_t needed, size_t page_size) {
    size_t grown = grow_cap_size(current, needed);
    if (page_size <= 1) {
        return grown;
    }
    return ((grown + page_size - 1) / page_size) * page_size;
}

// Grow or create cached DRAM buffers. Sets *output_reallocated when px/py/output
// were (re)allocated due to a resolution change.
static void ensure_buffer_cache(
    DeviceContext& ctx,
    BufferCache& cache,
    uint32_t image_h,
    uint32_t image_w,
    uint32_t total_entries,
    uint32_t offsets_count,
    size_t tile_ids_bytes,
    uint32_t num_tiles,
    bool& output_reallocated) {
    output_reallocated = false;
    auto make_dram = [&](size_t bytes, size_t page_bytes) {
        distributed::ReplicatedBufferConfig rc{.size = bytes};
        distributed::DeviceLocalBufferConfig lc{
            .page_size = page_bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(rc, lc, ctx.mesh_device.get());
    };

    if (total_entries > cache.max_entries) {
        cache.max_entries = grow_cap_u32(cache.max_entries, total_entries);
        cache.packs = make_dram(
            static_cast<size_t>(cache.max_entries) * SCALAR_PACK_PAGE_BYTES, SCALAR_PACK_PAGE_BYTES);
    }
    if (offsets_count > cache.max_offsets) {
        cache.max_offsets = grow_cap_u32(cache.max_offsets, offsets_count);
        cache.offsets = make_dram(
            static_cast<size_t>(cache.max_offsets) * sizeof(uint32_t), sizeof(uint32_t));
    }
    if (tile_ids_bytes > cache.max_tile_id_bytes) {
        cache.max_tile_id_bytes = grow_cap_bytes_page_aligned(
            cache.max_tile_id_bytes, tile_ids_bytes, TILE_IDS_PAGE_BYTES);
        cache.tile_ids = make_dram(cache.max_tile_id_bytes, TILE_IDS_PAGE_BYTES);
    }
    if (image_h != cache.image_h || image_w != cache.image_w) {
        cache.image_h = image_h;
        cache.image_w = image_w;
        cache.num_tiles = num_tiles;
        cache.px = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        cache.py = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        cache.output = make_dram(
            static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
        cache.last_frame_tile_nonempty.assign(num_tiles, 0);
        output_reallocated = true;
    }

    if (!cache.packs) {
        cache.max_entries = total_entries;
        cache.packs = make_dram(
            static_cast<size_t>(cache.max_entries) * SCALAR_PACK_PAGE_BYTES, SCALAR_PACK_PAGE_BYTES);
    }
    if (!cache.offsets) {
        cache.max_offsets = offsets_count;
        cache.offsets = make_dram(
            static_cast<size_t>(cache.max_offsets) * sizeof(uint32_t), sizeof(uint32_t));
    }
    if (!cache.px) {
        cache.image_h = image_h;
        cache.image_w = image_w;
        cache.num_tiles = num_tiles;
        cache.px = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        cache.py = make_dram(static_cast<size_t>(num_tiles) * TILE_BYTES_BF16, TILE_BYTES_BF16);
        cache.output = make_dram(
            static_cast<size_t>(num_tiles) * 3 * TILE_BYTES_BF16, TILE_BYTES_BF16);
        cache.last_frame_tile_nonempty.assign(num_tiles, 0);
        output_reallocated = true;
    }
    if (!cache.tile_ids) {
        cache.max_tile_id_bytes = tile_ids_bytes;
        cache.tile_ids = make_dram(cache.max_tile_id_bytes, TILE_IDS_PAGE_BYTES);
    }
}

static std::vector<uint8_t> compute_tile_nonempty(
    const std::vector<float>& offsets_f32, uint32_t num_tiles) {
    std::vector<uint8_t> mask(num_tiles, 0);
    for (uint32_t t = 0; t < num_tiles; t++) {
        if (offsets_f32[t + 1] - offsets_f32[t] > 0.0f) {
            mask[t] = 1;
        }
    }
    return mask;
}

// Zero output slots for tiles that were nonempty last frame but empty this frame.
// On resolution change the caller zero-fills the entire output buffer instead.
static void zero_flipped_empty_tiles(
    DeviceContext& ctx,
    const BufferCache& cache,
    const std::vector<uint8_t>& last_nonempty,
    const std::vector<uint8_t>& this_nonempty) {
    const distributed::MeshCoordinate coord(0, 0);
    std::vector<uint16_t> zeros(3 * TILE_H * TILE_W, 0);
    std::vector<distributed::ShardDataTransfer> transfers;
    for (uint32_t t = 0; t < this_nonempty.size(); t++) {
        if (last_nonempty[t] && !this_nonempty[t]) {
            const DeviceAddr offset = static_cast<DeviceAddr>(t) * 3 * TILE_BYTES_BF16;
            transfers.push_back(
                distributed::ShardDataTransfer(coord)
                    .host_data(zeros.data())
                    .region(BufferRegion{offset, 3 * TILE_BYTES_BF16}));
        }
    }
    if (!transfers.empty()) {
        ctx.cq->enqueue_write_shards(cache.output, transfers, /*blocking=*/false);
    }
}

// EnqueueWriteMeshBuffer requires host data >= buffer size; pad with zeros when
// the cached buffer is larger than this frame's payload.
template <typename T>
static void enqueue_write_cached_buffer(
    DeviceContext& ctx,
    std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    std::vector<T>& payload) {
    const size_t need_elems = mesh_buffer->size() / sizeof(T);
    if (payload.size() < need_elems) {
        payload.resize(need_elems, 0);
    }
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, mesh_buffer, payload);
}

// Set per-core runtime args for reader/compute/writer. Each core's slice of
// the concatenated tile_id_buffer is identified by (per_core_offset[c],
// per_core_count[c]); reader/writer kernels look up their tile IDs at runtime
// via this slice.
static void set_per_core_runtime_args(
    Program& program,
    const DeviceContext& ctx,
    const FrameDramBuffers& bufs,
    const TileAssignment& assign) {
    const uint32_t packs_addr    = static_cast<uint32_t>(bufs.packs->address());
    const uint32_t offsets_addr  = static_cast<uint32_t>(bufs.offsets->address());
    const uint32_t px_addr       = static_cast<uint32_t>(bufs.px->address());
    const uint32_t py_addr       = static_cast<uint32_t>(bufs.py->address());
    const uint32_t out_addr      = static_cast<uint32_t>(bufs.output->address());
    const uint32_t tile_ids_addr = static_cast<uint32_t>(bufs.tile_ids->address());

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
                });
                SetRuntimeArgs(program, ctx.compute, core, {count});
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
static double process_frame(DeviceContext& ctx, const FrameInputs& f) {
    const uint32_t image_h = f.image_h;
    const uint32_t image_w = f.image_w;
    const uint32_t tiles_x = (image_w + TILE_W - 1) / TILE_W;
    const uint32_t tiles_y = (image_h + TILE_H - 1) / TILE_H;
    const uint32_t num_tiles = tiles_x * tiles_y;
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;

    // 1. Load .npy fixtures.
    std::vector<size_t> packs_shape, offsets_shape, px_shape, py_shape;
    auto packs_f32   = load_npy_f32(f.packs_path,   packs_shape);
    auto offsets_f32 = load_npy_f32(f.offsets_path, offsets_shape);
    auto px_f32      = load_npy_f32(f.px_path,      px_shape);
    auto py_f32      = load_npy_f32(f.py_path,      py_shape);
    const uint32_t total_entries = static_cast<uint32_t>(packs_shape[0]);

    // 2. LPT-balanced tile-to-core assignment.
    const TileAssignment assign = build_tile_assignment(offsets_f32, num_tiles, num_cores);

    // 3. Allocate per-frame DRAM buffers and prepare upload payloads.
    // (Non-const because EnqueueWrite/ReadMeshBuffer takes non-const lvalue
    // refs to shared_ptr<MeshBuffer>.)
    FrameDramBuffers bufs = allocate_frame_buffers(
        ctx, total_entries, num_tiles, offsets_f32.size(),
        assign.tile_id_buffer_bytes_padded);
    auto packs_payload = encode_attribute_packs(packs_f32, total_entries);
    auto px_bf16 = encode_tiles_to_bf16(px_f32, num_tiles);
    auto py_bf16 = encode_tiles_to_bf16(py_f32, num_tiles);
    std::vector<uint32_t> offsets_u32(offsets_f32.size());
    for (size_t i = 0; i < offsets_f32.size(); i++) {
        offsets_u32[i] = static_cast<uint32_t>(offsets_f32[i]);
    }

    // 4. Refresh runtime args for this frame.
    Program& program = get_program_for_workload(ctx);
    set_per_core_runtime_args(program, ctx, bufs, assign);

    // 5. Kernel timing window: DRAM upload start -> output readback end.
    const auto t_start = std::chrono::steady_clock::now();
    // Zero-fill the output buffer. compute_lpt_assignment() filters out
    // empty tiles, so their slots are never written by the writer kernel.
    // The DRAM allocator may reuse addresses across frames in daemon mode,
    // so without this fill, empty regions would show stale pixels.
    std::vector<uint16_t> output_zero(
        static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W, 0);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.output,   output_zero);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.packs,    packs_payload);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.offsets,  offsets_u32);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.px,       px_bf16);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.py,       py_bf16);
    distributed::EnqueueWriteMeshBuffer(*ctx.cq, bufs.tile_ids, assign.tile_id_buffer_padded);
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::vector<uint16_t> result_bf16(
        static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, bufs.output, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();
    const double kernel_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 6. Tile-major bf16 output -> row-major fp32 image; save .npy.
    const auto img = tiles_to_image(result_bf16, num_tiles, tiles_x, image_h, image_w);
    save_npy_f32(f.out_path, img, {image_h, image_w, 3});
    return kernel_ms;
}

// Daemon path: reuse cached DRAM buffers and return the fp32 image in memory.
static std::vector<float> process_frame_cached(
    DeviceContext& ctx,
    uint32_t image_h,
    uint32_t image_w,
    const std::vector<float>& packs_f32,
    const std::vector<float>& offsets_f32,
    const std::vector<float>& px_f32,
    const std::vector<float>& py_f32,
    double& kernel_ms_out) {
    const uint32_t tiles_x = (image_w + TILE_W - 1) / TILE_W;
    const uint32_t tiles_y = (image_h + TILE_H - 1) / TILE_H;
    const uint32_t num_tiles = tiles_x * tiles_y;
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;
    const uint32_t total_entries = static_cast<uint32_t>(packs_f32.size() / 9);

    const TileAssignment assign = build_tile_assignment(offsets_f32, num_tiles, num_cores);

    bool output_reallocated = false;
    ensure_buffer_cache(
        ctx,
        ctx.cache,
        image_h,
        image_w,
        total_entries,
        static_cast<uint32_t>(offsets_f32.size()),
        assign.tile_id_buffer_bytes_padded,
        num_tiles,
        output_reallocated);
    BufferCache& cache = ctx.cache;

    auto packs_payload = encode_attribute_packs(packs_f32, total_entries);
    auto px_bf16 = encode_tiles_to_bf16(px_f32, num_tiles);
    auto py_bf16 = encode_tiles_to_bf16(py_f32, num_tiles);
    std::vector<uint32_t> offsets_u32(offsets_f32.size());
    for (size_t i = 0; i < offsets_f32.size(); i++) {
        offsets_u32[i] = static_cast<uint32_t>(offsets_f32[i]);
    }

    const std::vector<uint8_t> this_nonempty = compute_tile_nonempty(offsets_f32, num_tiles);

    FrameDramBuffers bufs{
        cache.packs, cache.offsets, cache.px, cache.py, cache.output, cache.tile_ids};
    Program& program = get_program_for_workload(ctx);
    set_per_core_runtime_args(program, ctx, bufs, assign);

    const auto t_start = std::chrono::steady_clock::now();
    if (output_reallocated) {
        std::vector<uint16_t> output_zero(cache.output->size() / sizeof(uint16_t), 0);
        distributed::EnqueueWriteMeshBuffer(*ctx.cq, cache.output, output_zero);
    } else {
        zero_flipped_empty_tiles(ctx, cache, cache.last_frame_tile_nonempty, this_nonempty);
    }
    enqueue_write_cached_buffer(ctx, cache.packs, packs_payload);
    enqueue_write_cached_buffer(ctx, cache.offsets, offsets_u32);
    enqueue_write_cached_buffer(ctx, cache.px, px_bf16);
    enqueue_write_cached_buffer(ctx, cache.py, py_bf16);
    std::vector<uint32_t> tile_ids_payload = assign.tile_id_buffer_padded;
    enqueue_write_cached_buffer(ctx, cache.tile_ids, tile_ids_payload);
    distributed::EnqueueMeshWorkload(*ctx.cq, ctx.workload, /*blocking=*/false);
    std::vector<uint16_t> result_bf16(static_cast<size_t>(num_tiles) * 3 * TILE_H * TILE_W);
    distributed::EnqueueReadMeshBuffer(*ctx.cq, result_bf16, cache.output, /*blocking=*/true);
    const auto t_end = std::chrono::steady_clock::now();
    kernel_ms_out = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    cache.last_frame_tile_nonempty = this_nonempty;
    return tiles_to_image(result_bf16, num_tiles, tiles_x, image_h, image_w);
}

// ---------------------------------------------------------------------------
// Binary IPC helpers (daemon mode)
// ---------------------------------------------------------------------------

static bool read_exact_stdin(void* dst, size_t nbytes) {
    char* out = static_cast<char*>(dst);
    size_t got = 0;
    while (got < nbytes) {
        std::cin.read(out + got, static_cast<std::streamsize>(nbytes - got));
        const auto n = static_cast<size_t>(std::cin.gcount());
        if (n == 0) {
            return false;
        }
        got += n;
    }
    return true;
}

static void write_ok_response(
    const std::vector<float>& img, uint32_t image_h, uint32_t image_w, double kernel_ms) {
    const uint32_t image_bytes = image_h * image_w * 3 * sizeof(float);
    const uint32_t kernel_us = static_cast<uint32_t>(kernel_ms * 1000.0 + 0.5);
    const uint32_t hdr[4] = {IPC_MAGIC_OK11, image_bytes, kernel_us, 0};
    std::cout.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    std::cout.write(reinterpret_cast<const char*>(img.data()), image_bytes);
    std::cout.flush();
}

static void write_err_response(const std::string& msg) {
    const uint32_t hdr[4] = {IPC_MAGIC_ERR1, 0, 0, static_cast<uint32_t>(msg.size())};
    std::cout.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    std::cout.write(msg.data(), static_cast<std::streamsize>(msg.size()));
    std::cout.flush();
}

// ---------------------------------------------------------------------------
// Daemon mode
// ---------------------------------------------------------------------------

static int run_daemon() {
    DeviceContext ctx;
    try {
        ctx = init_device_context();
    } catch (const std::exception& e) {
        std::cerr << "daemon init failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "READY" << std::endl;
    std::cout.flush();

    while (true) {
        char prefix[4];
        if (!read_exact_stdin(prefix, 4)) {
            break;
        }

        if ((prefix[0] == 'Q' || prefix[0] == 'q') && (prefix[1] == 'U' || prefix[1] == 'u') &&
            (prefix[2] == 'I' || prefix[2] == 'i') && (prefix[3] == 'T' || prefix[3] == 't')) {
            char nl = '\0';
            read_exact_stdin(&nl, 1);
            break;
        }

        uint32_t magic = 0;
        std::memcpy(&magic, prefix, 4);
        if (magic != IPC_MAGIC_FRM1) {
            write_err_response("expected FRM1 magic or QUIT");
            continue;
        }

        uint32_t hdr_rest[5] = {};
        if (!read_exact_stdin(hdr_rest, sizeof(hdr_rest))) {
            break;
        }
        const uint32_t image_h = hdr_rest[0];
        const uint32_t image_w = hdr_rest[1];
        const uint32_t total_entries = hdr_rest[2];
        const uint32_t offsets_count = hdr_rest[3];

        const uint32_t tiles_x = (image_w + TILE_W - 1) / TILE_W;
        const uint32_t tiles_y = (image_h + TILE_H - 1) / TILE_H;
        const uint32_t num_tiles = tiles_x * tiles_y;
        const size_t packs_floats = static_cast<size_t>(total_entries) * 9;
        const size_t px_floats = static_cast<size_t>(num_tiles) * TILE_H * TILE_W;

        std::vector<float> packs_f32(packs_floats);
        std::vector<float> offsets_f32(offsets_count);
        std::vector<float> px_f32(px_floats);
        std::vector<float> py_f32(px_floats);
        if (!read_exact_stdin(packs_f32.data(), packs_floats * sizeof(float)) ||
            !read_exact_stdin(offsets_f32.data(), offsets_count * sizeof(float)) ||
            !read_exact_stdin(px_f32.data(), px_floats * sizeof(float)) ||
            !read_exact_stdin(py_f32.data(), px_floats * sizeof(float))) {
            break;
        }

        try {
            double kernel_ms = 0.0;
            auto img = process_frame_cached(
                ctx, image_h, image_w, packs_f32, offsets_f32, px_f32, py_f32, kernel_ms);
            write_ok_response(img, image_h, image_w, kernel_ms);
        } catch (const std::exception& e) {
            write_err_response(e.what());
        }
    }

    bool ok = true;
    if (ctx.mesh_device) {
        ok = ctx.mesh_device->close();
    }
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Single-shot mode (original CLI)
// ---------------------------------------------------------------------------

static int run_single_shot(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " packs.npy offsets.npy px.npy py.npy output.npy [H] [W]\n"
                  << "       " << argv[0] << " --daemon\n";
        return 1;
    }
    FrameInputs f;
    f.packs_path = argv[1];
    f.offsets_path = argv[2];
    f.px_path = argv[3];
    f.py_path = argv[4];
    f.out_path = argv[5];
    f.image_h = argc > 6 ? static_cast<uint32_t>(std::stoi(argv[6])) : 32;
    f.image_w = argc > 7 ? static_cast<uint32_t>(std::stoi(argv[7])) : 32;

    bool pass = true;
    DeviceContext ctx;
    try {
        ctx = init_device_context();
        process_frame(ctx, f);
        std::cout << "Wrote " << f.out_path << std::endl;
        if (ctx.mesh_device) {
            pass &= ctx.mesh_device->close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Run failed with exception: " << e.what() << std::endl;
        throw;
    }
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--daemon") {
        return run_daemon();
    }
    return run_single_shot(argc, argv);
}
