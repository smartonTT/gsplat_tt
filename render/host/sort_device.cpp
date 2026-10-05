// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// In-process host driver for gsplat_tt sort — amendment-002 tt-003.
//
// See sort.h for the staged design (S0 host-sort-to-resident, S1 device radix).
//
// S1 keeps the host binning (Pass1 counts + Pass2 stable scatter, identical to
// gsplat_cpu::sort.cpp) but lays the (key, id) pairs out PAGE-ALIGNED per tile
// in DRAM so the device radix kernel can read/write each tile's exclusive 64B
// pages without cross-tile races. After the kernel, the host compacts the
// aligned per-tile segments back into the CPU-contiguous order (Pass4) and
// widens ids to int64 so the returned SortResult is byte-identical to the CPU.
//
// Both stages publish the CONTIGUOUS outputs into device_state under
// "sort_sorted_ids" (uint32, P) and "sort_tile_ranges" (uint32, num_tiles*2)
// so a future device blend can consume them resident.

#include "blend.h"
#include "config.h"
#include "env_config.h"
#include "sort.h"
#include "sort_mover_speed.h"
#include "blend_claim_order.h"
#include "sort_mover_split.h"
#include "sort_onelaunch_layout.h"
#include "device_state.h"
#include "../kernels/dataflow/pfwc_fuse.h"
#include "host_tracy.hpp"
#include "stage_timers.h"
#include "vis_mode.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
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
using namespace gsplat_tt::sort_split;

constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t PAGE_BYTES = ELEMS_PER_PAGE * 4;  // 64

// iter 110 (A2): the depth-sorted PACK2 slab (sort_subchunk_payload) uses a
// LARGE DRAM interleave page so each subchunk loads in ceil(L/64) big NoC
// transfers instead of ~4096 per-64B-page reads. The in-record layout is
// UNCHANGED — the slab is still a contiguous array of 32B records (record g at
// byte g*32); only the DRAM interleave granularity grows. SLAB_RECS_PER_PAGE
// = SLAB_PAGE_BYTES / 32. Subchunk bases are page-aligned in these units.
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / 32u;  // 64

// ROUTE C bucket-cull (GSPLAT_TT_BUCKET_MASK): the SFPU microblock cull runs in
// the sort stage over the dense record bucket so the keep mask is a SORT-STAGE
// write baked into record word 10 (read back spin-free by the L1 blend).
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t RAMP_TILE_BYTES = TILE_DIM * TILE_DIM * 4;  // 4096 (fp32 32x32)

// Intra-vector CB-linear position for (gaussian g == SFPU vector, microblock m).
// MUST match perm() in writer_bucket_cull.cpp / microblock_cull_compute.cpp.
inline uint32_t cull_perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

// Constant box-origin ramp: at CB-linear cull_perm(g,m) store microblock m's
// tile-local box origin ((m&3)*8 for x, (m>>2)*4 for y). Identical to the
// blend-side cull's make_box_ramp.
static std::vector<uint32_t> make_box_ramp(bool is_x) {
    std::vector<uint32_t> r(TILE_DIM * TILE_DIM, 0);
    for (uint32_t g = 0; g < 32; ++g) {
        for (uint32_t m = 0; m < 32; ++m) {
            const uint32_t dev = cull_perm(g, m);
            // Task #44: pixel-centre box origin (+0.5); extent 7x3 in the kernel.
            const float v = is_x ? static_cast<float>((m & 3u) * 8u) + 0.5f
                                 : static_cast<float>((m >> 2) * 4u) + 0.5f;
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            r[dev] = bits;
        }
    }
    return r;
}

static bool bucket_mask_enabled() { return false; }  // BUCKET_MASK unset

// L1 scratch budget per ping/pong key+id buffer. The worst hero-scene tile is
// ~25k entries; 32768 leaves headroom. 4 CBs * 32768 * 4B = 512 KB, well under
// Blackhole's ~1.4 MB per-core L1. If a tile exceeds this the host
// transparently falls back to the CPU sort (device_ok = false).
constexpr uint32_t MAX_TILE_ENTRIES = 32768;
constexpr uint32_t MAX_TILE_PAGES = MAX_TILE_ENTRIES / ELEMS_PER_PAGE;  // 2048
constexpr uint32_t SCRATCH_BYTES = MAX_TILE_ENTRIES * 4;  // 128 KB per CB

// Device-binning: max tiles the per-core L1 row / cursor / offset CBs hold
// (hero is 1024 tiles). Larger inputs are unsupported and hard-fail.
constexpr uint32_t MAX_BIN_TILES = 2048;
constexpr uint32_t BIN_ROW_BYTES = MAX_BIN_TILES * 4;  // 8 KB
// Max kept pairs a single core counting-sorts in L1. Pairs are split evenly by
// page across cores, so per-core load ~= P_kept / num_cores (~25k on hero);
// 65536 (256 KB per ks/is CB) leaves >2x headroom. Larger inputs hard-fail.
constexpr uint32_t BIN_LOCAL_MAX = 65536;
constexpr uint32_t BIN_LOCAL_BYTES = BIN_LOCAL_MAX * 4;  // 256 KB

inline uint32_t round_up(uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; }

struct SortDeviceContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    distributed::MeshCommandQueue* cq = nullptr;
    CoreCoord grid{0, 0};
    CoreRangeSet all_cores;

    distributed::MeshWorkload workload;
    KernelHandle kernel{};
    KernelHandle kernel_m0{};  // radix on BRISC (mover 0), CBs at id + 16

    // On-device compact+publish (buf_out -> sort_sorted_ids).
    distributed::MeshWorkload wl_publish;
    KernelHandle kpublish{};

    // ROUTE C bucket-cull program (reader_bucket_cull / microblock_cull_compute
    // / writer_bucket_cull): SFPU microblock cull over the dense record bucket,
    // baking the keep mask into record word 10 (GSPLAT_TT_BUCKET_MASK).
    distributed::MeshWorkload wl_cull;
    KernelHandle kc_reader{};
    KernelHandle kc_compute{};
    KernelHandle kc_writer{};
    bool cull_built = false;
    std::shared_ptr<distributed::MeshBuffer> buf_box_ox;
    std::shared_ptr<distributed::MeshBuffer> buf_box_oy;
    bool box_ramp_uploaded = false;

    // R4/R5 device-binning program (count + scatter) for resident pairs.
    distributed::MeshWorkload wl_bin;
    KernelHandle kbin{};
    // T-C: the same kernel on BRISC (mover 0) of the same cores, and the two
    // per-core fill-done semaphores of the dual-mover emit (see sort_bin.cpp).
    KernelHandle kbin0{};
    uint32_t bin_sem[2] = {0, 0};
    // Post-count layout + LPT (single core, GSPLAT_TT_SORT_DEVICE_LAYOUT).
    distributed::MeshWorkload wl_bin_layout;
    KernelHandle kbin_layout{};
    std::shared_ptr<distributed::MeshBuffer> buf_bin_ctrl;  // 1-page control out
    // S5.4 (iter-125): parallel Pass-2 base emit. The coordinator (wl_bin_layout)
    // publishes per-worker running-prefix checkpoints into buf_layout_ckpt; this
    // multi-core workload emits the bin2d + l1_rec_base rows in parallel.
    distributed::MeshWorkload wl_bin_layout_emit;
    KernelHandle kbin_layout_emit{};
    std::shared_ptr<distributed::MeshBuffer> buf_layout_ckpt;  // num_workers × 2 rows
    std::size_t cap_layout_ckpt_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bin2d;  // per-core 2D hist/base
    std::size_t cap_bin2d_bytes = 0;
    // Host-bridge path: the count pass writes its per-(core,tile) histogram here
    // (not into bin2d, which receives the host's page bases), so the emit pass
    // can read its own row back instead of recounting.
    std::shared_ptr<distributed::MeshBuffer> buf_bin_hist;
    std::size_t cap_bin_hist_bytes = 0;
    // T-C: the count pass's per-core histogram of the first (mover 0) half of
    // each core's page range; mover 1 of the emit starts its cursors there.
    std::shared_ptr<distributed::MeshBuffer> buf_bin_h0;
    std::size_t cap_bin_h0_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bin_dbg;  // core0 scatter dump

    // Cached DRAM buffers (grow-on-demand).
    std::shared_ptr<distributed::MeshBuffer> buf_keys;     // aligned packed keys
    std::shared_ptr<distributed::MeshBuffer> buf_ids;      // aligned packed ids
    std::shared_ptr<distributed::MeshBuffer> buf_out;      // aligned sorted ids
    std::size_t cap_aligned_bytes = 0;

    // T1/T2 (GSPLAT_TT_TILE_BUCKET): per-tile contiguous full-record bucket
    // scattered by the bin in arbitrary order. DENSE layout: tile t occupies
    // record-pages [starts[t], starts[t]+counts[t]) (1 record == 1 page == 64B),
    // no padding. buf_bin2d_rec holds the per-(core,tile) DENSE base (mirrors
    // bin2d); buf_bucket_meta publishes (start,count) per tile for the reader.
    std::shared_ptr<distributed::MeshBuffer> buf_tile_recs;
    std::size_t cap_tile_recs_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bin2d_rec;
    std::size_t cap_bin2d_rec_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_bucket_meta;
    std::size_t cap_bucket_meta_bytes = 0;

    // M0 (GSPLAT_TT_L1_RECORD): pre-sized 32B per-entry L1 record bucket.
    // buf_l1_recs: BUCKET_FIT * num_tiles records × 32B each (pre-sized; tile t
    //   at slot range [t*BUCKET_FIT, (t+1)*BUCKET_FIT)).
    // buf_l1_rec_base: per-(core,tile) slot index start within buf_l1_recs —
    //   l1_base[c][t] = t*BUCKET_FIT + sum_{c'<c} count[c'][t].
    //   Same shape as buf_bin2d_rec but in 32B record slot units.
    std::shared_ptr<distributed::MeshBuffer> buf_l1_recs;
    std::size_t cap_l1_recs_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_l1_rec_base;
    std::size_t cap_l1_rec_base_bytes = 0;

    // iter-138 (Stage-2b overflow pre-pack): separate COMPACT PACK2 region holding
    // the FULL records of overflow tiles (kBucketFit < count <= kOverflowL1Cap),
    // prefix-allocated over those tiles only. buf_l1_ov: the records (64B PACK2
    // pages, same layout as buf_l1_recs). buf_l1_ov_base: per-(core,tile) start
    // slot in the region (sentinel for non-overflow tiles) — consumed by the emit
    // (bin) kernel. buf_tile_ov_base: per-TILE start slot (sentinel otherwise) —
    // consumed by the materialize kernel to coalesced-read + L1-radix the bucket.
    std::shared_ptr<distributed::MeshBuffer> buf_l1_ov;
    std::size_t cap_l1_ov_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_l1_ov_base;
    std::size_t cap_l1_ov_base_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_ov_base;
    std::size_t cap_tile_ov_base_bytes = 0;

    std::shared_ptr<distributed::MeshBuffer> buf_tile_ids;  // LPT tile-id list
    std::size_t cap_tile_ids_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tmeta;     // (pstart_page, n)
    std::size_t cap_tmeta_bytes = 0;

    // Resident contiguous outputs (published for downstream device consumers).
    std::shared_ptr<distributed::MeshBuffer> buf_sorted_ids;   // contiguous, P
    std::size_t cap_sorted_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_ranges;  // num_tiles*2
    std::size_t cap_ranges_bytes = 0;

    // Downstream blend/cull LPT + per-tile kept counts (published resident).
    std::shared_ptr<distributed::MeshBuffer> buf_lpt_meta;  // num_cores*2 u32
    std::size_t cap_lpt_meta_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_tile_counts;  // num_tiles u32
    std::size_t cap_tile_counts_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_P_kept;  // 1-page scalar
    std::shared_ptr<distributed::MeshBuffer> buf_cull_mask_base;  // per-tile page-aligned mask offset
    std::size_t cap_cull_mask_base_bytes = 0;

    // Iter 53+: post-radix PACK2 subchunk payloads + device-resident directory.
    distributed::MeshWorkload wl_subchunk;
    KernelHandle ksubchunk{};
    KernelHandle ksubchunk_m0{};  // materialize on BRISC (mover 0), CBs at id + 16
    KernelHandle kmatcull{};      // task #90: fused SFPU cull (mat_cull_compute.cpp)
    std::shared_ptr<distributed::MeshBuffer> buf_blend_subchunk_meta;
    std::size_t cap_blend_subchunk_meta_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_subchunk_payload;
    std::size_t cap_subchunk_payload_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_subchunk_dir;
    std::size_t cap_subchunk_dir_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_subchunk_prefix;
    std::size_t cap_subchunk_prefix_bytes = 0;
    // iter 130: per-core materialize WORK-ITEM list (flat u32: tile_id, sc, ...)
    // — subchunk-granular, gather-cost-weighted balance (see build_mat_worklist).
    std::shared_ptr<distributed::MeshBuffer> buf_mat_work;
    std::size_t cap_mat_work_bytes = 0;

    // Task #106: one-launch sort (sort_bin_onelaunch.cpp, GSPLAT_TT_SORT_ONELAUNCH, default on).
    distributed::MeshWorkload wl_onelaunch;
    KernelHandle kol{};   // NCRISC (mover 1)
    KernelHandle kol0{};  // BRISC (mover 0)
    uint32_t ol_sem[6] = {0, 0, 0, 0, 0, 0};  // counted, based, arrive1, release1, arrive2, release2
    bool ol_built = false;
    bool ol_frame = false;  // this frame's materialize reads the tile buckets
    std::shared_ptr<distributed::MeshBuffer> buf_ol_bucket;  // tile t at slot t * tile cap
    std::size_t cap_ol_bucket_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_ol_counts;  // per-core count rows
    std::shared_ptr<distributed::MeshBuffer> buf_ol_bases;   // per-core base rows
    std::size_t cap_ol_rows_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> buf_ol_totals;  // totals row, padded-totals row
    std::size_t cap_ol_totals_bytes = 0;
};

static std::shared_ptr<distributed::MeshBuffer> make_dram(
    distributed::MeshDevice* dev, std::size_t bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = PAGE_BYTES, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

static std::shared_ptr<distributed::MeshBuffer> make_dram_paged(
    distributed::MeshDevice* dev, std::size_t bytes, std::size_t page_bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{
        .page_size = page_bytes, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

// Dual-mover sort tail (task #35): the radix (sort_radix_tile) and the
// materialize (sort_subchunk_materialize) run on BRISC as well as NCRISC.
// GSPLAT_TT_SORT_RADIX_MOVERS=1 / GSPLAT_TT_SORT_MAT_MOVERS=1 keep each on
// NCRISC alone (A/B baseline in the same build). Read once.
static uint32_t env_movers(const char* name) {
    const char* e = std::getenv(name);
    return (e != nullptr && std::atoi(e) == 1) ? 1u : 2u;
}
static uint32_t sort_radix_movers() {
    static const uint32_t v = env_movers("GSPLAT_TT_SORT_RADIX_MOVERS");
    return v;
}
static uint32_t sort_mat_movers() {
    static const uint32_t v = env_movers("GSPLAT_TT_SORT_MAT_MOVERS");
    return v;
}
// Task #106 (lever 1): GSPLAT_TT_SORT_ONELAUNCH replaces count + hist D2H +
// host layout + emit + radix + publish with one launch (sort_bin_onelaunch.cpp)
// and lets the materialize sort the tile buckets. Task #121: default on (v2,
// 29.55 -> 24.60 ms/view on yyzo-bh-07); GSPLAT_TT_SORT_ONELAUNCH=0 is the kill
// switch back to the legacy multi-launch sort. Read once.
static bool sort_onelaunch_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_SORT_ONELAUNCH");
        return e == nullptr || e[0] != '0';
    }();
    return v;
}
// Records per tile bucket (== the legacy MAX_TILE_ENTRIES limit). The emit's
// L1 page window is env_config::ol_win_pages() (define OL_WIN_PAGES).
constexpr uint32_t kOneLaunchTileCap = sort_onelaunch::kTileCap;
// Tiles with an emit record ring (== OL_RING_TILES); more tiles: rings off.
constexpr uint32_t kOneLaunchRingTiles = 1024;
// Coefficient/mask tiles in flight per mover in the fused mat+cull program
// (== the depth of each CB_COEFF / CB_KEEP; 4 KB fp32 tiles, 4 CBs).
// GSPLAT_TT_MATCULL_DEPTH overrides it for tuning (1..8).
static uint32_t mat_cull_depth() {
    static const uint32_t v = [] {
        const char* e = std::getenv("GSPLAT_TT_MATCULL_DEPTH");
        // 4 CBs x depth x 4 KB: depth 8 is 128 KB of the materialize program's
        // L1, the most it can spare next to the slab; out-of-range values warn
        // and use the default instead of silently overflowing L1.
        constexpr int kMaxDepth = 8;
        if (e == nullptr) return 2u;
        const int d = std::atoi(e);
        if (d >= 1 && d <= kMaxDepth) return static_cast<uint32_t>(d);
        std::cerr << "[gsplat_tt::sort] GSPLAT_TT_MATCULL_DEPTH=" << e
                  << " outside 1.." << kMaxDepth << "; using 2\n";
        return 2u;
    }();
    return v;
}
static void build_program(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto big_cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    big_cb(0, SCRATCH_BYTES);  // CB_KIN
    big_cb(1, SCRATCH_BYTES);  // CB_IIN
    big_cb(2, SCRATCH_BYTES);  // CB_KOUT
    big_cb(3, SCRATCH_BYTES);  // CB_IOUT
    big_cb(4, PAGE_BYTES);     // CB_TIDS
    big_cb(5, PAGE_BYTES);     // CB_META
    // Dual mover: BRISC sorts the other part of each core's tile slice on its
    // own full-size copies (id + 16, created after NCRISC's so those keep their
    // single-mover L1 addresses). 2 x 4 x 128 KB fits the 1.5 MB L1.
    big_cb(16, SCRATCH_BYTES);
    big_cb(17, SCRATCH_BYTES);
    big_cb(18, SCRATCH_BYTES);
    big_cb(19, SCRATCH_BYTES);
    big_cb(20, PAGE_BYTES);
    big_cb(21, PAGE_BYTES);

    std::vector<uint32_t> ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_radix_tile.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    ctx.kernel_m0 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_radix_tile.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
            .defines = {{"RADIX_CB_BASE", "16"}},
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.workload.add_program(device_range, std::move(program));
}

struct LptAssignment {
    std::vector<uint32_t> flat_tile_ids;
    std::vector<uint32_t> per_core_offset;
    std::vector<uint32_t> per_core_count;
};

struct SubchunkLayout {
    uint32_t total_subchunks = 0;
    uint64_t total_payload_pages = 0;
    uint32_t tiles_split = 0;
    uint32_t max_subchunks_per_tile = 0;
    std::vector<uint32_t> tile_meta;   // num_tiles*2: [dir_base, num_subchunks]
    std::vector<uint32_t> prefix;      // num_tiles: payload page offset
    std::vector<uint32_t> dir;         // total_subchunks*4: page,L,flags,0
};

static SubchunkLayout build_subchunk_layout(
    const std::vector<int64_t>& counts, uint32_t num_tiles, uint32_t bucket_fit) {
    SubchunkLayout layout;
    layout.tile_meta.assign(static_cast<std::size_t>(num_tiles) * 2u, 0u);
    layout.prefix.assign(num_tiles, 0u);
    uint32_t dir_cursor = 0;
    uint64_t page_cursor = 0;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t count = static_cast<uint32_t>(counts[t]);
        const uint32_t num_sc =
            count == 0u ? 1u : (count + bucket_fit - 1u) / bucket_fit;
        layout.tile_meta[static_cast<std::size_t>(t) * 2u + 0u] = dir_cursor;
        layout.tile_meta[static_cast<std::size_t>(t) * 2u + 1u] = num_sc;
        layout.prefix[t] = static_cast<uint32_t>(page_cursor);
        layout.total_subchunks += num_sc;
        if (num_sc > 1u) {
            layout.tiles_split += 1u;
        }
        layout.max_subchunks_per_tile =
            std::max(layout.max_subchunks_per_tile, num_sc);
        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * bucket_fit;
            const uint32_t l_sub = (sc_off >= count) ? 0u
                : ((count - sc_off > bucket_fit) ? bucket_fit : (count - sc_off));
            const uint32_t flags =
                ((sc > 0u) ? 2u : 0u) | ((sc + 1u == num_sc) ? 1u : 0u);
            layout.dir.push_back(static_cast<uint32_t>(page_cursor));
            layout.dir.push_back(l_sub);
            layout.dir.push_back(flags);
            layout.dir.push_back(0u);
            if (l_sub > 0u) {
                page_cursor += (static_cast<uint64_t>(l_sub) + SLAB_RECS_PER_PAGE - 1u) /
                               SLAB_RECS_PER_PAGE;
            }
            dir_cursor += 1u;
        }
    }
    layout.total_payload_pages = page_cursor;
    return layout;
}

static void log_subchunk_layout_stats(const SubchunkLayout& layout) {
    std::fprintf(
        stderr,
        "[SUBCHUNK] tiles_split=%u total_subchunks=%u max_subchunks_per_tile=%u "
        "payload_pages=%llu\n",
        layout.tiles_split,
        layout.total_subchunks,
        layout.max_subchunks_per_tile,
        static_cast<unsigned long long>(layout.total_payload_pages));
}

static void build_program_subchunk(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    const uint32_t bucket_fit = render_config::kBucketFit;
    auto page_cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    // iter-138 (Stage-2b overflow pre-pack): the in-cap overflow path reads the
    // WHOLE overflow tile (up to kOverflowL1Cap records) into CB_BUCKET and L1-
    // radix-sorts it in place. CB_BUCKET already holds bucket_fit*64 B = cap*32 B
    // (cap = 2*bucket_fit) with ZERO growth — the in-budget path only used half of
    // it (bucket_fit/2 PACK2 pages). CB_BSORT (the radix idx/cnt scratch) grows to
    // (2*cap+256) u32 to index cap records. CB_SLAB stays bucket_fit-sized: the
    // sorted slab is streamed to DRAM ONE subchunk (<=bucket_fit recs) at a time.
    const uint32_t ov_cap = render_config::kOverflowL1Cap;
    page_cb(0, PAGE_BYTES);
    page_cb(1, PAGE_BYTES);
    page_cb(2, 32u * PAGE_BYTES);  // REC_BATCH=32 blendrec gather ring (iter 76)
    page_cb(3, 32u);
    page_cb(4, std::max(bucket_fit * 64u, ov_cap * 32u));  // CB_BUCKET (holds cap recs)
    page_cb(5, (2u * ov_cap + 256u) * 4u);                 // CB_BSORT (radix over cap recs)
    // iter 113 (sort Stage 1): CB_SLAB — contiguous L1 scratch the in-budget
    // depth permutation lands in (bucket_fit * 32B records) so the slab is
    // emitted in coalesced SLAB_PAGE_BYTES writes, not per-record DRAM scatter.
    page_cb(6, bucket_fit * 32u);
    // Dual mover: BRISC's copies (id + 16) sized for kMatMover0Cap records —
    // NCRISC's ~900 KB set does not fit twice in L1. build_mat_worklist gives
    // BRISC only whole-tile items of <= kMatMover0Cap records and gather items.
    const uint32_t m0_cap = kMatMover0Cap;
    page_cb(16, PAGE_BYTES);
    page_cb(17, PAGE_BYTES);
    page_cb(18, 32u * PAGE_BYTES);
    page_cb(19, 32u);
    page_cb(20, m0_cap * 32u);           // CB_BUCKET (PACK2, m0_cap recs)
    page_cb(21, (2u * m0_cap + 256u) * 4u);  // CB_BSORT
    page_cb(22, m0_cap * 32u);           // CB_SLAB

    // Task #86: GSPLAT_TT_MATCULL_PROF=1 compiles fine per-item Tracy zones
    // into the materialize kernel (attribution). Default OFF.
    const char* mc_prof = std::getenv("GSPLAT_TT_MATCULL_PROF");
    std::map<std::string, std::string> mat_defines;
    if (mc_prof != nullptr && mc_prof[0] == '1') mat_defines["MATCULL_PROF"] = "1";
    // Task #90: fused SFPU cull. Per mover a coefficient and a mask CB of
    // mat_cull_depth() fp32 tiles at id 8 / 9 (+16 on BRISC), served by one
    // compute kernel (same config as the tile_l1_cull compute).
    const bool fuse_cull = sort_matcull_fused();
    if (fuse_cull) {
        auto tile_cb = [&](uint32_t id) {
            CircularBufferConfig c(mat_cull_depth() * 4096u, {{id, DataFormat::Float32}});
            c.set_page_size(id, 4096u);
            CreateCircularBuffer(program, cores, c);
        };
        tile_cb(8);
        tile_cb(9);
        tile_cb(24);
        tile_cb(25);
        mat_defines["FUSE_CULL"] = "1";
        mat_defines["FUSE_CULL_DEPTH"] = std::to_string(mat_cull_depth()) + "u";
        // GSPLAT_TT_MATCULL_FOLD=1: fill the coefficient tiles inside the permute
        // (permute_cull) instead of a second pass over the slab (cull_slab).
        const char* fold = std::getenv("GSPLAT_TT_MATCULL_FOLD");
        mat_defines["MATCULL_FOLD"] = (fold != nullptr && fold[0] == '1') ? "1" : "0";
        std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
        u2d[8] = UnpackToDestMode::UnpackToDestFp32;
        u2d[24] = UnpackToDestMode::UnpackToDestFp32;
        ctx.kmatcull = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "kernels/compute/mat_cull_compute.cpp",
            cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi3,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = true,
                .unpack_to_dest_mode = u2d,
                .math_approx_mode = false,
            });
    }
    // Task #106: the one-launch bucket branch (args 16, 17).
    if (sort_onelaunch_enabled()) {
        mat_defines["SORT_ONELAUNCH"] = "1";
        // Task #124: big-tile items sort only their own ranks' depth bins.
        if (gsplat_tt::env_config::ol_mat_select()) {
            mat_defines["OL_MAT_SELECT"] = "1";
            mat_defines["OL_MAT_PART"] = std::to_string(gsplat_tt::sort_split::kOlMatPartRecs) + "u";
        }
    }
    std::map<std::string, std::string> mat_defines_m0 = mat_defines;
    mat_defines_m0["MAT_CB_BASE"] = "16";
    // A gather part is staged whole in the mover's slab (BRISC: kMatMover0Cap).
    static_assert(kGatherPartRecs <= kMatMover0Cap);
    mat_defines["GATHER_PART_RECS_HOST"] = std::to_string(kGatherPartRecs) + "u";
    mat_defines["MAT_M0_CAP"] = std::to_string(kMatMover0Cap) + "u";
    mat_defines_m0["GATHER_PART_RECS_HOST"] = mat_defines["GATHER_PART_RECS_HOST"];
    mat_defines_m0["MAT_M0_CAP"] = mat_defines["MAT_M0_CAP"];
    std::vector<uint32_t> ct;
    // 9 base accessors + iter-138 {overflow region, per-tile overflow base}.
    for (int i = 0; i < 11; i++) {
        TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    }
    ctx.ksubchunk = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_subchunk_materialize.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
            .defines = mat_defines,
        });
    ctx.ksubchunk_m0 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_subchunk_materialize.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
            .defines = mat_defines_m0,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_subchunk.add_program(device_range, std::move(program));
}

// Host layout sizing + DRAM alloc; upload_subchunk_directory fills meta/dir/prefix.
static bool prepare_subchunk_buffers(
    SortDeviceContext* ctx,
    const SubchunkLayout& layout,
    uint32_t num_tiles) {
    const std::size_t payload_bytes =
        std::max<std::size_t>(SLAB_PAGE_BYTES, layout.total_payload_pages * SLAB_PAGE_BYTES);
    if (!ctx->buf_subchunk_payload ||
        ctx->cap_subchunk_payload_bytes < payload_bytes) {
        ctx->buf_subchunk_payload =
            make_dram_paged(ctx->mesh_device.get(), payload_bytes, SLAB_PAGE_BYTES);
        ctx->cap_subchunk_payload_bytes = payload_bytes;
        device_state::register_buffer("sort_subchunk_payload", ctx->buf_subchunk_payload);
    }
    const std::size_t dir_bytes = round_up(
        std::max<std::size_t>(PAGE_BYTES, layout.dir.size() * 4u), PAGE_BYTES);
    if (!ctx->buf_subchunk_dir || ctx->cap_subchunk_dir_bytes < dir_bytes) {
        ctx->buf_subchunk_dir = make_dram(ctx->mesh_device.get(), dir_bytes);
        ctx->cap_subchunk_dir_bytes = dir_bytes;
        device_state::register_buffer("sort_subchunk_dir", ctx->buf_subchunk_dir);
    }
    const std::size_t prefix_bytes = round_up(
        static_cast<std::size_t>(round_up(num_tiles, ELEMS_PER_PAGE)) * 4u, PAGE_BYTES);
    if (!ctx->buf_subchunk_prefix ||
        ctx->cap_subchunk_prefix_bytes < prefix_bytes) {
        ctx->buf_subchunk_prefix = make_dram(ctx->mesh_device.get(), prefix_bytes);
        ctx->cap_subchunk_prefix_bytes = prefix_bytes;
        device_state::register_buffer("sort_subchunk_prefix", ctx->buf_subchunk_prefix);
    }
    const std::size_t blend_meta_bytes = round_up(
        static_cast<std::size_t>(round_up(num_tiles, ELEMS_PER_PAGE)) * 8u, PAGE_BYTES);
    if (!ctx->buf_blend_subchunk_meta ||
        ctx->cap_blend_subchunk_meta_bytes < blend_meta_bytes) {
        ctx->buf_blend_subchunk_meta =
            make_dram(ctx->mesh_device.get(), blend_meta_bytes);
        ctx->cap_blend_subchunk_meta_bytes = blend_meta_bytes;
        device_state::register_buffer(
            "blend_subchunk_meta", ctx->buf_blend_subchunk_meta);
    }
    return true;
}

// Upload the per-tile blend meta [dir_base, num_subchunks], the payload page
// prefix and the per-subchunk dir entries [page, L_sub, flags, 0]. These are
// exactly build_subchunk_layout's tile_meta / prefix / dir, which the host has
// already computed for sizing; the former sort_subchunk_directory kernel rebuilt
// them serially on ONE core with ~4 dependent DRAM read-modify-write round trips
// per tile (~2.4 ms/view of device time between publish and materialize). ~30 KB
// of H2D, ordered on the CQ ahead of materialize/cull/blend, their only readers.
static void upload_subchunk_directory(
    SortDeviceContext* ctx, const SubchunkLayout& layout) {
    auto upload = [&](std::shared_ptr<distributed::MeshBuffer>& buf,
                      std::size_t cap_bytes, const std::vector<uint32_t>& src) {
        // EnqueueWriteMeshBuffer writes the WHOLE (grow-only) buffer.
        std::vector<uint32_t> v(cap_bytes / 4, 0u);
        std::copy(src.begin(), src.begin() + std::min(src.size(), v.size()), v.begin());
        distributed::EnqueueWriteMeshBuffer(*ctx->cq, buf, v, false);
    };
    upload(ctx->buf_blend_subchunk_meta, ctx->cap_blend_subchunk_meta_bytes,
           layout.tile_meta);
    upload(ctx->buf_subchunk_prefix, ctx->cap_subchunk_prefix_bytes, layout.prefix);
    upload(ctx->buf_subchunk_dir, ctx->cap_subchunk_dir_bytes, layout.dir);
}

// Device post-radix PACK2 materialize (enqueue only; caller Finish()).
// Kernel: in-budget sc==0 uses buf_l1_recs bulk; overflow sc==0/sc>=1 use sorted_ids gather.
static bool launch_subchunk_materialize(
    SortDeviceContext* ctx,
    const MatWorkAssignment& work,
    uint32_t num_cores,
    uint32_t tiles_x,
    uint32_t bucket_fit,
    const SortBlendContinuation* cont = nullptr) {
    if (work.flat.empty()) {
        return true;
    }
    auto bbrec = device_state::get_buffer("proj_m_blendrec");
    auto bl1 = device_state::get_buffer("sort_l1_recs");
    auto bsids = device_state::get_buffer("sort_sorted_ids");
    auto brng = device_state::get_buffer("sort_tile_ranges");
    if (!bbrec || !bl1 || !bsids || !brng || !ctx->buf_blend_subchunk_meta ||
        !ctx->buf_subchunk_dir) {
        return false;
    }
    // Grow-only work buffer; EnqueueWriteMeshBuffer writes the WHOLE buffer, so
    // the host vector is padded to capacity (mirrors buf_sorted_ids).
    const std::size_t work_bytes =
        round_up(std::max<std::size_t>(work.flat.size() * 4u, PAGE_BYTES), PAGE_BYTES);
    if (!ctx->buf_mat_work || ctx->cap_mat_work_bytes < work_bytes) {
        ctx->buf_mat_work = make_dram(ctx->mesh_device.get(), work_bytes);
        ctx->cap_mat_work_bytes = work_bytes;
    }
    const uint32_t cap_elems = static_cast<uint32_t>(ctx->cap_mat_work_bytes / 4);
    std::vector<uint32_t> wbuf(cap_elems, 0);
    for (std::size_t i = 0; i < work.flat.size(); ++i) wbuf[i] = work.flat[i];
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_mat_work, wbuf, false);

    // iter-138: overflow region + per-tile overflow base for the coalesced path.
    // Both 0 ⇒ no in-cap overflow tiles this view (kernel keeps gather/in-budget).
    const uint32_t ov_addr = ctx->buf_l1_ov
        ? static_cast<uint32_t>(ctx->buf_l1_ov->address()) : 0u;
    const uint32_t ov_base_addr = ctx->buf_tile_ov_base
        ? static_cast<uint32_t>(ctx->buf_tile_ov_base->address()) : 0u;
    // Task #90 fused cull: the blend's contrib floor and cull switch. Without a
    // continuation (non-piped launch) use the same device_state parameters the
    // blend reads, so the fused cull never runs with floor 0 while blend skips
    // its own cull.
    float floor_f = 1.0f / 16384.0f;
    bool cull_off = false;
    if (cont != nullptr) {
        floor_f = cont->mb_contrib_floor;
        cull_off = cont->cull_disabled;
    } else {
        device_state::get_bucket_cull_params(&floor_f, &cull_off);
    }
    uint32_t floor_bits = 0;
    std::memcpy(&floor_bits, &floor_f, 4);
    const uint32_t cull_disabled = cull_off ? 1u : 0u;
    // Task #106: a one-launch frame's records are in the tile buckets.
    const uint32_t l1_addr = ctx->ol_frame ? static_cast<uint32_t>(ctx->buf_ol_bucket->address())
                                           : static_cast<uint32_t>(bl1->address());
    Program& prog = ctx->wl_subchunk.get_programs().begin()->second;
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
        // Slot 2c = NCRISC, 2c+1 = BRISC; single mover: slot c, BRISC idle.
        for (uint32_t m = 0; m < 2; ++m) {
            const bool ncrisc = (m == 0);
            const uint32_t slot = (work.movers == 2) ? 2u * c + m : c;
            const bool idle = !ncrisc && work.movers != 2;
            std::vector<uint32_t> args = {
                static_cast<uint32_t>(bsids->address()),
                static_cast<uint32_t>(brng->address()),
                static_cast<uint32_t>(bbrec->address()),
                l1_addr,
                static_cast<uint32_t>(ctx->buf_subchunk_payload->address()),
                static_cast<uint32_t>(ctx->buf_blend_subchunk_meta->address()),
                static_cast<uint32_t>(ctx->buf_subchunk_dir->address()),
                static_cast<uint32_t>(ctx->buf_mat_work->address()),
                idle ? 0u : work.per_core_offset[slot],
                idle ? 0u : work.per_core_count[slot],
                tiles_x,
                bucket_fit,
                ov_addr,
                ov_base_addr,
                render_config::kOverflowL1Cap,
                ncrisc ? bucket_fit : kMatMover0Cap,
            };
            if (sort_onelaunch_enabled()) {
                // Built with SORT_ONELAUNCH: bucket capacity (0 = legacy frame)
                // and this mover's whole-tile sort capacity.
                args.push_back(ctx->ol_frame ? kOneLaunchTileCap : 0u);
                args.push_back(ncrisc ? render_config::kOverflowL1Cap : kMatMover0Cap);
            }
            SetRuntimeArgs(prog, ncrisc ? ctx->ksubchunk : ctx->ksubchunk_m0, core, args);
        }
        if (sort_matcull_fused()) {
            // Both movers always end their stream (idle ones at once): live = 3.
            SetRuntimeArgs(prog, ctx->kmatcull, core, {floor_bits, cull_disabled, 3u});
        }
    }
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_subchunk, false);
    return true;
}

// Task #106: one-launch sort program (sort_bin_onelaunch.cpp). The same kernel
// on BRISC (mover 0) and NCRISC (mover 1) of every core; private CBs at id
// (NCRISC) / id + 16 (BRISC), the count/base row (10) and the prefix staging
// (11) shared. Semaphores: the mover handshake and two all-core barriers.
static void build_program_sort_onelaunch(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    const uint32_t num_cores = ctx.grid.x * ctx.grid.y;
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    // Task #124 v2 emit: OL_PB pair pages per batch (3 staging buffers for
    // pages past the window, 2 blendrec rings), OL_RING records per per-tile
    // run (cb 13: kOneLaunchRingTiles x R x 32 B + the 4 B run starts).
    const uint32_t pb = gsplat_tt::env_config::ol_pair_batch();
    const uint32_t ring = gsplat_tt::env_config::ol_ring();
    const uint32_t win = gsplat_tt::env_config::ol_win_pages();
    const uint32_t ring_bytes = kOneLaunchRingTiles * (ring * 32u + 4u);
    uint32_t mover_bytes = 0;
    for (const uint32_t off : {0u, 16u}) {
        mover_bytes = 0;
        auto mcb = [&](uint32_t id, uint32_t bytes) {
            cb(id + off, bytes);
            mover_bytes += bytes;
        };
        mcb(0, 3u * pb * PAGE_BYTES);         // gid pages (past the window)
        mcb(1, 3u * pb * PAGE_BYTES);         // tid pages
        mcb(2, 3u * pb * PAGE_BYTES);         // keep pages
        mcb(3, PAGE_BYTES);                   // depth page (EMIT_PUBOC=0 only)
        mcb(4, BIN_ROW_BYTES);                // per-tile count of this mover
        mcb(5, BIN_ROW_BYTES);                // per-tile cursor
        mcb(6, 2u * 32u * PAGE_BYTES);        // count-pass read batch (tid, keep)
        mcb(7, 2u * pb * 16u * PAGE_BYTES);   // blendrec rings: 16 pages per pair page
        mcb(8, 16u * 32u);                    // 32 B record staging (OL_RING=0)
        mcb(12, win * 3u * PAGE_BYTES);       // gid/tid/keep window
        if (ring != 0u) mcb(13, ring_bytes);  // per-tile record runs
    }
    cb(10, BIN_ROW_BYTES);                       // the core's count row, then base row
    cb(11, (2u * num_cores + 2u) * PAGE_BYTES);  // prefix pass staging
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::fprintf(stderr,
                     "[SORT] ONELAUNCH v2 OL_PB=%u OL_RING=%u OL_WIN_PAGES=%u OL_MAT_SELECT=%u "
                     "cb_bytes/mover=%u shared=%u\n",
                     pb, ring, win, gsplat_tt::env_config::ol_mat_select() ? 1u : 0u, mover_bytes,
                     BIN_ROW_BYTES + (2u * num_cores + 2u) * PAGE_BYTES);
    }
    for (uint32_t& sem : ctx.ol_sem) sem = CreateSemaphore(program, cores, 0);
    std::vector<uint32_t> ct;
    for (int i = 0; i < 9; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    // Task #100 PUBOC: copy the gather-published op/color/depth words.
    std::map<std::string, std::string> defines;
    defines["EMIT_PUBOC"] = gsplat_tt::env_config::emit_puboc() ? "1u" : "0u";
    defines["OL_PB"] = std::to_string(pb) + "u";
    defines["OL_RING"] = std::to_string(ring) + "u";
    defines["OL_RING_TILES"] = std::to_string(kOneLaunchRingTiles) + "u";
    defines["OL_WIN_PAGES"] = std::to_string(win) + "u";
    // Task #154: GSPLAT_TT_OL_EMIT_PROF=1 (profiling only) records the emit's
    // per-part cycle totals as Tracy "ep_*" markers. Unset: no define, the same
    // kernel binary as before.
    if (const char* e = std::getenv("GSPLAT_TT_OL_EMIT_PROF"); e != nullptr && std::atoi(e) != 0) {
        defines["OL_EMIT_PROF"] = "1";
    }
    // Task #160: GSPLAT_TT_OL_EMIT_FAST=0 is the kill switch of the fast emit
    // loop (default on; same output either way).
    if (const char* e = std::getenv("GSPLAT_TT_OL_EMIT_FAST"); e != nullptr && std::atoi(e) == 0) {
        defines["OL_EMIT_FAST"] = "0";
    }
    ctx.kol = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin_onelaunch.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
            .defines = defines,
        });
    ctx.kol0 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin_onelaunch.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
            .defines = defines,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_onelaunch.add_program(device_range, std::move(program));
    ctx.ol_built = true;
}

static void build_program_publish(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    auto page_cb = [&](uint32_t id) {
        CircularBufferConfig c(PAGE_BYTES, {{id, DataFormat::UInt32}});
        c.set_page_size(id, PAGE_BYTES);
        CreateCircularBuffer(program, cores, c);
    };
    page_cb(0);  // CB_TIDS
    page_cb(1);  // CB_META
    page_cb(2);  // CB_SCRATCH

    std::vector<uint32_t> ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kpublish = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_publish.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_publish.add_program(device_range, std::move(program));
}

static bool fused_tile_enabled() { return false; }  // FUSED_TILE=0

static bool tile_bucket_enabled() { return true; }  // TILE_BUCKET=1

// R4/R5 binning program: one data-movement kernel (count + scatter modes).
// T-C dual-data-mover emit. GSPLAT_TT_SORT_EMIT_MOVERS=1 keeps the scatter on
// NCRISC alone and builds the pre-T-C program (no BRISC kernel, semaphores or
// h0 buffer), an exact A/B in the same build. GSPLAT_TT_SORT_EMIT_SPLIT=<n>
// gives mover 0 (BRISC) n/1000 of each core's pair pages (default 500; an
// unparsable or out-of-range value warns and uses the default). Read once.
static uint32_t sort_emit_movers() {
    static const uint32_t v = [] {
        const char* e = std::getenv("GSPLAT_TT_SORT_EMIT_MOVERS");
        return (e != nullptr && std::atoi(e) == 1) ? 1u : 2u;
    }();
    return v;
}
static uint32_t sort_emit_split_permille() {
    static const uint32_t v = [] {
        constexpr uint32_t kDefault = 500u;
        const char* e = std::getenv("GSPLAT_TT_SORT_EMIT_SPLIT");
        if (e == nullptr || *e == '\0') return kDefault;
        char* end = nullptr;
        errno = 0;
        const long x = std::strtol(e, &end, 10);
        if (errno != 0 || end == e || *end != '\0' || x < 0 || x > 1000) {
            std::cerr << "[gsplat_tt::sort] GSPLAT_TT_SORT_EMIT_SPLIT=\"" << e
                      << "\" is not an integer in [0, 1000]; using " << kDefault << "\n";
            return kDefault;
        }
        return static_cast<uint32_t>(x);
    }();
    return v;
}

// Task #166: GSPLAT_TT_OL_SPLIT_ROWS="<row0>/<row1>/..." sets BRISC's share
// (permille) of a one-launch core's pair pages per logical core row (rows past
// the list: GSPLAT_TT_SORT_EMIT_SPLIT). "" = no per-row split. Same output for
// any split. Read once. Default from the t166 sweep (yyzo-bh-07 p100a): BRISC
// on NOC0 rows y=2,3 stalls on NoC issue, so those rows give NCRISC more pages.
// Task #174: the default applies only with GSPLAT_TT_PRECULL=2 (it costs
// PRECULL=1 +0.18 ms/view) and only when ol_mover_speed() is off.
static const std::vector<uint32_t>& ol_split_rows() {
    static const std::vector<uint32_t> v = [] {
        const char* dflt = (gsplat_tt::precull_mode() == 2) ? "400/440/490" : "";
        const char* e = std::getenv("GSPLAT_TT_OL_SPLIT_ROWS");
        std::vector<uint32_t> rows;
        if (!gsplat_tt::sort_split::parse_row_permille(e != nullptr ? e : dflt, &rows)) {
            std::cerr << "[gsplat_tt::sort] GSPLAT_TT_OL_SPLIT_ROWS=\"" << e
                      << "\" is not a '/'-separated list of integers in [0, 1000]; using \""
                      << dflt << "\"\n";
            gsplat_tt::sort_split::parse_row_permille(dflt, &rows);
        }
        return rows;
    }();
    return v;
}

// Task #174: GSPLAT_TT_OL_MOVER_SPEED=1 gives every one-launch emit mover pair
// pages in proportion to its measured speed (sort_mover_speed.h, by physical
// NoC core) instead of an even split per core: NOC0's top rows (BRISC) and
// NOC1's right columns (NCRISC) stall on NoC issue, and the slowest mover sets
// the emit time. The ranges stay contiguous in (core, BRISC, NCRISC) order, so
// the output is the same for any speeds. Default 1 with GSPLAT_TT_PRECULL=2,
// else 0. Overrides GSPLAT_TT_OL_SPLIT_ROWS / GSPLAT_TT_SORT_EMIT_SPLIT.
static bool ol_mover_speed() { return gsplat_tt::sort_split::ol_mover_speed_enabled(); }

static void build_program_bin(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    // T-C: every staging CB except the shared counting-sort regions (7, 8) and
    // the unused recrow (10) gets a private copy for mover 0 (BRISC) at id +
    // kMover0CbOffset, created after the originals so NCRISC's CBs keep their
    // single-mover L1 addresses.
    constexpr uint32_t kMover0CbOffset = 16;  // == sort_bin.cpp MOVER0_CB_OFFSET
    std::vector<std::pair<uint32_t, uint32_t>> mover0_cbs;
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
        if (id != 7 && id != 8 && id != 10) mover0_cbs.emplace_back(id + kMover0CbOffset, bytes);
    };
    // Task #100 emit knobs (see sort_bin.cpp sub-pass 2): EMIT_PB pair pages
    // per batch (3 staging buffers and 2 blendrec rings when > 1), EMIT_RING
    // records per per-tile staging run (cb 15), EMIT_PUBOC op/color/depth taken
    // from the gather's blendrec words 10..12.
    const uint32_t emit_pb = gsplat_tt::env_config::emit_pair_batch();
    const uint32_t emit_npbuf = emit_pb > 1u ? 3u : 1u;
    const uint32_t emit_nring = emit_pb > 1u ? 2u : 1u;
    cb(0, emit_npbuf * emit_pb * PAGE_BYTES);  // gid_in
    cb(1, emit_npbuf * emit_pb * PAGE_BYTES);  // tid_in
    cb(2, emit_npbuf * emit_pb * PAGE_BYTES);  // keep_in
    cb(3, PAGE_BYTES);        // depth
    cb(4, BIN_ROW_BYTES);     // row (hist out / base in)
    cb(5, BIN_ROW_BYTES);     // cur (local per-tile cursor)
    cb(6, BIN_ROW_BYTES);     // off (local per-tile L1 offset)
    cb(7, BIN_LOCAL_BYTES);   // ksort (L1 counting-sort keys)
    cb(8, BIN_LOCAL_BYTES);   // isort (L1 counting-sort ids)

    const bool tile_bucket = tile_bucket_enabled();
    const bool l1_record = gsplat_tt::env_config::l1_record_enabled();
    if (tile_bucket) {
        cb(9, emit_nring * emit_pb * 16u * PAGE_BYTES);  // blendrec ring(s): 16 pages per pair page
        cb(10, BIN_ROW_BYTES);    // recrow (per-(core,tile) DENSE record base)
    }
    if (l1_record) {
        // cb(11): per-(core,tile) L1 slot base (same shape as recrow/BIN_ROW_BYTES)
        cb(11, BIN_ROW_BYTES);
        // cb(12): REC_BATCH × 32B staging area for packing L1 records before write
        // (512B), plus REC_BATCH × 4B for the optional M1 gid stash (L1_SORT_VERIFY).
        cb(12, 16u * 32u + 16u * 4u);  // 512B staging + 64B gid scratch
        // iter-138: cb(14) per-(core,tile) overflow-region slot base row (sentinel
        // = tile is not a pre-packed overflow tile). Same shape as cb(11).
        cb(14, BIN_ROW_BYTES);
    }
    // iter 132: cb(13) — PACKOC_BATCH × 16B ring staging the per-gaussian blendrec
    // chunk [words 8,9,10,11] (orig cb/depth + packed op/color) written back
    // 16B-aligned (offset 32) to blendrec for materialize. 16B is the natural DRAM
    // write granule (sub-16B / unaligned writes to blendrec do not land on this BH).
    // Sized to the kernel's PACKOC_BATCH=16 (256B). Allocated unconditionally (used
    // whenever the blendrec record is read, i.e. tile_bucket).
    cb(13, 16u * 16u);
    // Task #100: cb(15) per-tile record staging runs (EMIT_RING_TILES x R x 32B)
    // plus each tile's first local cursor (MAX_BIN_TILES x 4B).
    constexpr uint32_t kEmitRingTiles = 1024;  // == sort_bin.cpp EMIT_RING_TILES
    const uint32_t emit_ring = l1_record ? gsplat_tt::env_config::emit_ring() : 0u;
    if (emit_ring != 0u) cb(15, kEmitRingTiles * emit_ring * 32u + MAX_BIN_TILES * 4u);
    // GSPLAT_TT_SORT_EMIT_MOVERS=1: none of the mover-0 resources exist, so the
    // program is the pre-T-C single-mover one.
    const bool dual_movers = sort_emit_movers() == 2;
    if (dual_movers) {
        for (const auto& [id, bytes] : mover0_cbs) {
            CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
            c.set_page_size(id, bytes);
            CreateCircularBuffer(program, cores, c);
        }
        ctx.bin_sem[0] = CreateSemaphore(program, cores, 0);
        ctx.bin_sem[1] = CreateSemaphore(program, cores, 0);
    }

    std::vector<uint32_t> ct;
    // Accessors: 7 base + 3 tile_bucket + 2 l1_record + 2 l1_overflow (iter-138)
    int bin_accessors = 7;
    if (tile_bucket) bin_accessors += 3;  // blendrec, tile_recs, recbase
    if (l1_record)   bin_accessors += 2;  // l1_recs, l1_rec_base
    if (l1_record)   bin_accessors += 2;  // iter-138: l1_overflow, l1_overflow_base
    for (int i = 0; i < bin_accessors; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    // Single-path bin kernel: BIN_EMIT_REC / L1_BUCKET_REC are inlined in the
    // kernel source; the debug/verify defines (BIN_NO_DEPTH, BIN_DUMP,
    // L1_SORT_VERIFY) were removed with their kernel branches.
    std::map<std::string, std::string> defines;
    // Task #98: GSPLAT_TT_EMIT_ABLATE=<mask> (profiling only, output is wrong)
    // removes parts of the emit; see EMIT_ABLATE in sort_bin.cpp.
    if (const char* e = std::getenv("GSPLAT_TT_EMIT_ABLATE"); e != nullptr && std::atoi(e) > 0) {
        defines["EMIT_ABLATE"] = std::to_string(std::atoi(e)) + "u";
        std::cerr << "[gsplat_tt::sort] GSPLAT_TT_EMIT_ABLATE=" << e
                  << ": profiling ablation, the rendered output is WRONG\n";
    }
    defines["EMIT_PB"] = std::to_string(emit_pb) + "u";
    defines["EMIT_RING"] = std::to_string(emit_ring) + "u";
    defines["EMIT_RING_TILES"] = std::to_string(kEmitRingTiles) + "u";
    defines["EMIT_PUBOC"] = gsplat_tt::env_config::emit_puboc() ? "1u" : "0u";
    ctx.kbin = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
            .defines = defines,
        });
    if (dual_movers) {
        ctx.kbin0 = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin.cpp",
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = ct,
                .defines = defines,
            });
    }
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_bin.add_program(device_range, std::move(program));
}

// S5.4 (iter-125): number of cores the parallel Pass-2 base emit is spread over.
// The coordinator publishes one prefix checkpoint per worker (W × 2 × row_span u32
// of serial DRAM writes), so W trades emit parallelism against checkpoint-publish
// cost — ~16 is near the makespan minimum (the single serial Pass-1 count read
// dominates the coordinator either way). Clamped to num_cores at enqueue time.
constexpr uint32_t kLayoutEmitWorkers = 16u;

static void build_program_bin_layout(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreCoord core0{0, 0};
    const CoreRangeSet cores(core0);
    // Bulk-row layout kernel L1 budget (single core). MAX_TILES/MAX_CORES mirror
    // the kernel's compile-time bounds.
    constexpr uint32_t KMAX_TILES = 2048;
    constexpr uint32_t KMAX_CORES = 128;
    constexpr uint32_t scratch_bytes = (5u * KMAX_TILES + 5u * KMAX_CORES) * 4u;
    constexpr uint32_t row_bytes = KMAX_TILES * 4u;       // one core row (hist/bin2d/l1base)
    constexpr uint32_t out_bytes = 2u * KMAX_TILES * 4u;  // per-tile staging (tmeta/ranges/...)
    // S5.2 histogram cache: hold the whole per-core histogram in L1 so the layout
    // kernel streams DRAM once (Pass 1) and the base-emit pass reads from L1. Cap
    // = 128*1024 u32 = 512 KB (fits hero's 110*1024 ≈ 450 KB); the kernel falls
    // back to a 2nd DRAM read pass if num_cores*row_span exceeds this. MUST match
    // the kernel's HCACHE_CAP constant in sort_bin_layout.cpp.
    constexpr uint32_t KHCACHE_CAP = 128u * 1024u;
    constexpr uint32_t hcache_bytes = KHCACHE_CAP * 4u;
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    cb(0, PAGE_BYTES);      // CB_CTRL
    cb(1, scratch_bytes);   // CB_SCRATCH
    cb(2, row_bytes);       // CB_ROW (fallback row buffer)
    cb(3, row_bytes);       // CB_BIN
    cb(4, row_bytes);       // CB_L1B
    cb(5, out_bytes);       // CB_OUT
    cb(6, hcache_bytes);    // CB_HCACHE (full per-core histogram cache)

    std::vector<uint32_t> ct;
    for (int i = 0; i < 12; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kbin_layout = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin_layout.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_bin_layout.add_program(device_range, std::move(program));
}

// S5.4 (iter-125): parallel Pass-2 base emit. Each worker core emits the bin2d
// base + l1_rec_base rows for a contiguous range of source-cores, seeded from the
// coordinator's per-worker prefix checkpoint. Created on the full grid; cores
// without an assigned range get core_count=0 and no-op.
static void build_program_bin_layout_emit(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;
    constexpr uint32_t KMAX_TILES = 2048;
    constexpr uint32_t row_bytes = KMAX_TILES * 4u;  // one row (page_acc/rec_acc/hist/bin/l1b)
    auto cb = [&](uint32_t id, uint32_t bytes) {
        CircularBufferConfig c(bytes, {{id, DataFormat::UInt32}});
        c.set_page_size(id, bytes);
        CreateCircularBuffer(program, cores, c);
    };
    cb(0, PAGE_BYTES);  // CB_CTRL (status)
    cb(1, row_bytes);   // CB_PAGE (page_acc)
    cb(2, row_bytes);   // CB_REC  (rec_acc)
    cb(3, row_bytes);   // CB_HIST (source-core hist row)
    cb(4, row_bytes);   // CB_BIN  (bin2d base out)
    cb(5, row_bytes);   // CB_L1B  (l1_rec_base out)

    std::vector<uint32_t> ct;
    // 4 DRAM-interleaved accessors: bin2d, l1_base, ckpt, ctrl.
    for (int i = 0; i < 4; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    ctx.kbin_layout_emit = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/sort_bin_emit.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = ct,
        });
    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_bin_layout_emit.add_program(device_range, std::move(program));
}

// ROUTE C: 3-kernel bucket-cull program. reader_bucket_cull streams each LPT
// tile's dense records into the SAME microblock_cull_compute SFPU kernel the
// blend-side cull used; writer_bucket_cull packs the 32-bit keep mask and RMWs
// it into record word 10. Because it is dispatched in the SORT stage, the mask
// reads back spin-free in the downstream L1 blend (no cull_masks DRAM, no spin).
static void build_program_bucket_cull(SortDeviceContext& ctx) {
    Program program = CreateProgram();
    const CoreRangeSet& cores = ctx.all_cores;

    auto cb_cfg = [&](uint32_t id, uint32_t page_bytes, uint32_t depth, DataFormat fmt) {
        CircularBufferConfig c(depth * page_bytes, {{id, fmt}});
        c.set_page_size(id, page_bytes);
        CreateCircularBuffer(program, cores, c);
    };
    // CB ids must match the three kernels exactly:
    //   reader: 0,1 (box ramps) 2 (coeff) 3 (counts) 4 (record/ids/meta scratch)
    //   compute: 0,1,2,3,16
    //   writer: 6 (RMW scratch) 16 (keep)
    cb_cfg(0, RAMP_TILE_BYTES, 1, DataFormat::Float32);   // CB_BOX_OX
    cb_cfg(1, RAMP_TILE_BYTES, 1, DataFormat::Float32);   // CB_BOX_OY
    cb_cfg(2, PAGE_BYTES, 32, DataFormat::Float32);       // CB_CULL_COEFF (7 used)
    cb_cfg(3, PAGE_BYTES, 2, DataFormat::UInt32);         // CB_CULL_COUNTS
    cb_cfg(4, 16u * PAGE_BYTES, 1, DataFormat::UInt32);   // reader scratch (16 records)
    cb_cfg(6, 32u * PAGE_BYTES, 1, DataFormat::UInt32);   // writer RMW scratch (32 records)
    cb_cfg(16, RAMP_TILE_BYTES, 4, DataFormat::Float32);  // CB_KEEP

    // Reader: 5 DRAM-interleaved accessors (tile_recs, bucket_meta, box_ox,
    // box_oy, tile_ids).
    std::vector<uint32_t> reader_ct;
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(reader_ct);
    ctx.kc_reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/reader_bucket_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct,
        });

    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[0] = UnpackToDestMode::UnpackToDestFp32;
    u2d[1] = UnpackToDestMode::UnpackToDestFp32;
    ctx.kc_compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/compute/microblock_cull_compute.cpp",
        cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = u2d,
            .math_approx_mode = false,
        });

    // Writer: 3 DRAM-interleaved accessors (tile_recs, bucket_meta, tile_ids).
    std::vector<uint32_t> writer_ct;
    for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(writer_ct);
    ctx.kc_writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "kernels/dataflow/writer_bucket_cull.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct,
        });

    distributed::MeshCoordinateRange device_range(ctx.mesh_device->shape());
    ctx.wl_cull.add_program(device_range, std::move(program));
    ctx.cull_built = true;
}

static SortDeviceContext init_context() {
    SortDeviceContext ctx;
    ctx.mesh_device = device_state::get_device();
    ctx.cq = device_state::command_queue();
    ctx.grid = ctx.mesh_device->compute_with_storage_grid_size();
    ctx.all_cores =
        CoreRangeSet(CoreRange({0, 0}, {ctx.grid.x - 1, ctx.grid.y - 1}));
    build_program(ctx);
    build_program_bin(ctx);
    build_program_bin_layout(ctx);
    build_program_bin_layout_emit(ctx);
    build_program_publish(ctx);
    build_program_subchunk(ctx);
    if (bucket_mask_enabled()) {
        build_program_bucket_cull(ctx);
    }
    return ctx;
}

static std::unique_ptr<SortDeviceContext>& context_slot() {
    static std::unique_ptr<SortDeviceContext> ctx;
    return ctx;
}

static SortDeviceContext* ensure_context() {
    auto& slot = context_slot();
    if (!slot) {
        try {
            slot = std::make_unique<SortDeviceContext>(init_context());
        } catch (const std::exception& e) {
            std::cerr << "[gsplat_tt::sort] device init failed: " << e.what() << "\n";
            slot.reset();
        }
    }
    return slot.get();
}

// LPT (longest-processing-time) tile->core assignment over non-empty tiles.
// Mirrors blend_device.cpp compute_lpt_assignment: heaviest tiles first, each
// onto the currently-least-loaded core. Empty tiles never reach the kernel.

// Contiguous page-range split of num_pages over num_cores (matches the
// gather/tile_assign convention so count + scatter use identical ranges).
struct PageSplit {
    std::vector<uint32_t> start;
    std::vector<uint32_t> count;
};
static PageSplit split_pages(uint32_t num_pages, uint32_t num_cores) {
    PageSplit ws;
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

static LptAssignment build_lpt(
    const std::vector<int64_t>& counts, uint32_t num_tiles, uint32_t num_cores) {
    std::vector<std::pair<uint32_t, uint32_t>> cost_id;
    cost_id.reserve(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t c = static_cast<uint32_t>(counts[t]);
        if (c > 0) cost_id.emplace_back(c, t);
    }
    std::sort(cost_id.begin(), cost_id.end(), std::greater<>());

    // Least-loaded core first, ties -> lowest core index: a (load, core) min-heap
    // makes exactly the choices std::min_element (first minimum) made, in
    // O(tiles log cores) instead of O(tiles * cores) on the critical path.
    std::vector<std::vector<uint32_t>> per_core(num_cores);
    using Slot = std::pair<uint64_t, uint32_t>;
    std::vector<Slot> heap_store;
    heap_store.reserve(num_cores);
    for (uint32_t c = 0; c < num_cores; c++) heap_store.emplace_back(0u, c);
    std::priority_queue<Slot, std::vector<Slot>, std::greater<Slot>> heap(
        std::greater<Slot>{}, std::move(heap_store));
    for (const auto& [cost, id] : cost_id) {
        const Slot top = heap.top();
        heap.pop();
        per_core[top.second].push_back(id);
        heap.emplace(top.first + cost, top.second);
    }

    LptAssignment a;
    a.per_core_offset.assign(num_cores, 0);
    a.per_core_count.assign(num_cores, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        a.per_core_offset[c] = static_cast<uint32_t>(a.flat_tile_ids.size());
        a.per_core_count[c] = static_cast<uint32_t>(per_core[c].size());
        a.flat_tile_ids.insert(
            a.flat_tile_ids.end(), per_core[c].begin(), per_core[c].end());
    }

    return a;
}

// Task #188: blend lists in record-count-descending round-robin order
// (blend_claim_order.h).
static LptAssignment build_desc_rr(
    const std::vector<int64_t>& counts, uint32_t num_tiles, uint32_t num_cores) {
    LptAssignment a;
    blend_order::desc_round_robin(counts, num_tiles, num_cores, a.flat_tile_ids,
                                  a.per_core_offset, a.per_core_count);
    return a;
}

// Blend per-core lists: record-count-descending deal (default) or LPT on the
// padded cost (GSPLAT_TT_BLEND_CLAIM_DESC=0).
static LptAssignment build_blend_lists(
    const std::vector<int64_t>& rec_counts, const std::vector<int64_t>& pad_counts,
    uint32_t num_tiles, uint32_t num_cores) {
    return gsplat_tt::env_config::blend_claim_desc()
        ? build_desc_rr(rec_counts, num_tiles, num_cores)
        : build_lpt(pad_counts, num_tiles, num_cores);
}

struct BinLayoutResult {
    std::vector<uint32_t> hist;  // page-aligned bases in bin2d layout
    std::vector<int64_t> counts;
    std::vector<int64_t> starts;
    std::vector<uint32_t> pstart_page;
    std::vector<uint32_t> pstart_elem;
    std::vector<uint32_t> tile_pad;
    std::vector<uint32_t> histrec;
    std::vector<uint32_t> bucket_meta;
    // M0: per-(core,tile) 32B record slot base in buf_l1_recs.
    // l1_base[c*stride+t] = t*bucket_fit + sum_{c'<c} count[c'][t].
    // Populated when l1_record is true.
    std::vector<uint32_t> histrec_l1;
    // iter-138: per-(core,tile) start slot in the compact overflow region
    // (0xFFFFFFFF for tiles that are NOT pre-packed overflow tiles), and the
    // per-TILE start slot (even-aligned for PACK2; 0xFFFFFFFF otherwise). Both
    // populated when l1_record is true. ov_total_slots = region size in 32B slots.
    std::vector<uint32_t> histrec_overflow;
    std::vector<uint32_t> tile_ov_base;
    uint64_t ov_total_slots = 0;
    uint32_t ov_tiles = 0;       // # tiles routed to the pre-pack path
    uint64_t ov_records = 0;     // # records in those tiles (the coalesced win)
    LptAssignment lpt;
    uint32_t P_kept = 0;
    uint32_t P_aligned = 0;
    uint32_t max_pad_n = 0;
    uint32_t status = 0;  // 0=ok, 1=BIN_LOCAL_MAX, 2=MAX_TILE_ENTRIES
};

// Host page layout from the per-(core,tile) histogram. Fills `r` IN PLACE so a
// caller-owned result keeps the capacity of its num_cores x stride tables across
// views. Three row-major passes over the histogram (sequential in memory; the
// per-tile running state stays in cache) replace the former tile-outer /
// core-inner walk that touched a new 4 KB row per element. Every output is
// bit-identical to that walk:
//   hist[c][t]       = 16 * (pstart_page[t] + sum_{c'<c} ceil16(h[c'][t]))
//   histrec_l1[c][t] = t*bucket_fit + sum_{c'<c} h[c'][t]
//   histrec[c][t]    = starts[t]    + sum_{c'<c} h[c'][t]   (only if dense_recbase)
// dense_recbase=false skips histrec: the emit kernel ignores recbase (it only fed
// the retired tile_recs scatter), so production neither builds nor uploads it.
static void host_bin_layout_into(
    const std::vector<uint32_t>& hist_in,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool tile_bucket,
    bool l1_record,
    uint32_t bucket_fit,
    bool dense_recbase,
    BinLayoutResult& r) {
    constexpr uint32_t SENT = 0xFFFFFFFFu;
    const std::size_t n2d = static_cast<std::size_t>(num_cores) * stride;
    r.status = 0;
    r.ov_total_slots = 0;
    r.ov_tiles = 0;
    r.ov_records = 0;
    r.P_kept = 0;
    r.P_aligned = 0;
    r.max_pad_n = 0;

    // Pass 1: per-tile totals + padded page counts, per-core padded footprint.
    std::vector<uint64_t> cnt(num_tiles, 0);
    std::vector<uint32_t> tpages(num_tiles, 0);
    uint64_t max_core_padded = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        const uint32_t* row = hist_in.data() + static_cast<std::size_t>(c) * stride;
        uint64_t s = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t h = row[t];
            const uint32_t pg = (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
            cnt[t] += h;
            tpages[t] += pg;
            s += static_cast<uint64_t>(pg) * ELEMS_PER_PAGE;
        }
        if (s > max_core_padded) max_core_padded = s;
    }
    r.counts.resize(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) r.counts[t] = static_cast<int64_t>(cnt[t]);
    if (max_core_padded > BIN_LOCAL_MAX) {
        r.status = 1;
        return;
    }

    // Pass 2: per-tile prefixes (starts, page starts, overflow region bases).
    r.starts.resize(num_tiles);
    r.pstart_page.resize(num_tiles);
    r.pstart_elem.resize(num_tiles);
    r.tile_pad.resize(num_tiles);
    if (tile_bucket) r.bucket_meta.assign(static_cast<std::size_t>(num_tiles) * 2u, 0u);
    if (l1_record) {
        // iter-138: prefix-allocate the COMPACT overflow region over in-cap
        // overflow tiles only (kBucketFit < count <= kOverflowL1Cap). Each such
        // tile's base is page-aligned (task #86: kRecsPerPage records per page) so
        // the materialize reader reads record g at page base/64 + g/64, slot g%64.
        r.tile_ov_base.assign(num_tiles, SENT);
        const uint32_t ov_cap = render_config::kOverflowL1Cap;
        uint64_t ov_cursor = 0;  // in 32B slots; page-aligned per tile
        constexpr uint64_t kPg = render_config::kRecsPerPage;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const uint64_t c = static_cast<uint64_t>(r.counts[t]);
            if (c > bucket_fit && c <= ov_cap) {
                r.tile_ov_base[t] = static_cast<uint32_t>(ov_cursor);
                ov_cursor += (c + kPg - 1u) / kPg * kPg;  // round up to a page
                r.ov_tiles += 1u;
                r.ov_records += c;
            }
        }
        r.ov_total_slots = ov_cursor;
    }
    int64_t cstart = 0;
    uint32_t apage = 0;
    uint32_t max_pad_n = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        r.starts[t] = cstart;
        cstart += r.counts[t];
        r.pstart_page[t] = apage;
        r.pstart_elem[t] = apage * ELEMS_PER_PAGE;
        apage += tpages[t];
        r.tile_pad[t] = tpages[t] * ELEMS_PER_PAGE;
        if (tile_bucket) {
            r.bucket_meta[static_cast<std::size_t>(t) * 2 + 0] =
                static_cast<uint32_t>(r.starts[t]);
            r.bucket_meta[static_cast<std::size_t>(t) * 2 + 1] =
                static_cast<uint32_t>(r.counts[t]);
        }
        if (r.tile_pad[t] > max_pad_n) max_pad_n = r.tile_pad[t];
    }
    if (max_pad_n > MAX_TILE_ENTRIES) {
        r.status = 2;
        return;
    }
    r.P_kept = static_cast<uint32_t>(cstart);
    r.max_pad_n = max_pad_n;
    r.P_aligned = std::max<uint32_t>(apage, 1u) * ELEMS_PER_PAGE;

    // Pass 3: per-(core,tile) bases, core-major with per-tile running cursors.
    const bool want_rec = tile_bucket && dense_recbase;
    r.hist.resize(n2d);
    if (want_rec) r.histrec.resize(n2d);
    if (l1_record) {
        r.histrec_l1.resize(n2d);
        r.histrec_overflow.resize(n2d);
    }
    std::vector<uint32_t> rec_run(num_tiles, 0u);
    std::vector<uint32_t> page_cur(r.pstart_page);
    for (uint32_t c = 0; c < num_cores; c++) {
        const std::size_t row0 = static_cast<std::size_t>(c) * stride;
        const uint32_t* in = hist_in.data() + row0;
        uint32_t* base = r.hist.data() + row0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t h = in[t];
            const uint32_t run = rec_run[t];
            if (want_rec) r.histrec[row0 + t] = static_cast<uint32_t>(r.starts[t]) + run;
            if (l1_record) {
                // M0: pre-sized bucket; tile t starts at slot t*bucket_fit.
                r.histrec_l1[row0 + t] = t * bucket_fit + run;
                // iter-138: pre-packed overflow tiles also place the core's block
                // in the compact region; sentinel => non-overflow tile.
                const uint32_t ovb = r.tile_ov_base[t];
                r.histrec_overflow[row0 + t] = (ovb != SENT) ? ovb + run : SENT;
            }
            base[t] = page_cur[t] * ELEMS_PER_PAGE;
            rec_run[t] = run + h;
            page_cur[t] += (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
        }
        // Stride padding columns: same values the old full-table init left there.
        for (uint32_t t = num_tiles; t < stride; t++) {
            base[t] = in[t];
            if (want_rec) r.histrec[row0 + t] = 0u;
            if (l1_record) {
                r.histrec_l1[row0 + t] = 0u;
                r.histrec_overflow[row0 + t] = SENT;
            }
        }
    }

    if (l1_record) {
        // iter-138 feasibility diagnostic: how the GATHERED records (all records of
        // tiles with count > bucket_fit) split across cap buckets. The pre-pack path
        // captures the (bucket_fit, kOverflowL1Cap] band; the rest still gathers.
        uint64_t gathered_total = 0, in_cap = 0, over_cap = 0;
        uint32_t over_cap_tiles = 0, max_tile = 0;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const uint64_t c = static_cast<uint64_t>(r.counts[t]);
            if (c > max_tile) max_tile = static_cast<uint32_t>(c);
            if (c > bucket_fit) {
                gathered_total += c;
                if (c <= render_config::kOverflowL1Cap) in_cap += c;
                else { over_cap += c; over_cap_tiles += 1u; }
            }
        }
        std::fprintf(stderr,
            "[OVERFLOW-DIST] bucket_fit=%u cap=%u num_tiles=%u P_kept=%u max_tile=%u "
            "| overflow_tiles=%u(prepack)+%u(gather) gathered_recs=%llu = in_cap=%llu "
            "(%.1f%% prepacked) + over_cap=%llu | ov_region_slots=%llu (~%.1f MB DRAM)\n",
            bucket_fit, render_config::kOverflowL1Cap, num_tiles, r.P_kept, max_tile,
            r.ov_tiles, over_cap_tiles,
            static_cast<unsigned long long>(gathered_total),
            static_cast<unsigned long long>(in_cap),
            gathered_total ? 100.0 * static_cast<double>(in_cap) /
                             static_cast<double>(gathered_total) : 0.0,
            static_cast<unsigned long long>(over_cap),
            static_cast<unsigned long long>(r.ov_total_slots),
            static_cast<double>(r.ov_total_slots) * 32.0 / (1024.0 * 1024.0));
    }
    std::vector<int64_t> pad_counts(num_tiles, 0);
    for (uint32_t t = 0; t < num_tiles; t++)
        pad_counts[t] = static_cast<int64_t>(r.tile_pad[t]);
    r.lpt = build_lpt(pad_counts, num_tiles, num_cores);
}

static BinLayoutResult host_bin_layout_from_hist(
    const std::vector<uint32_t>& hist_in,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool tile_bucket,
    bool l1_record = false,
    uint32_t bucket_fit = 8192u) {
    BinLayoutResult r;
    host_bin_layout_into(hist_in, num_cores, num_tiles, stride, tile_bucket, l1_record,
                         bucket_fit, /*dense_recbase=*/true, r);
    return r;
}

// S5.4 (iter-125): enqueue the parallel Pass-2 base emit. Splits the num_cores
// source rows into W contiguous ranges (must match the coordinator's checkpoint
// split exactly), one per worker core; grid cores beyond W get core_count=0.
static void enqueue_bin_layout_emit_kernel(
    SortDeviceContext* ctx,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool l1_record,
    uint32_t bucket_fit,
    uint32_t ckpt_addr,
    uint32_t W) {
    Program& prog = ctx->wl_bin_layout_emit.get_programs().begin()->second;
    const uint32_t grid_cores = ctx->grid.x * ctx->grid.y;
    const uint32_t base_c = num_cores / W;
    const uint32_t rem_c = num_cores % W;
    const uint32_t l1_base_addr = (l1_record && ctx->buf_l1_rec_base)
        ? static_cast<uint32_t>(ctx->buf_l1_rec_base->address())
        : 0u;
    uint32_t cc = 0;
    for (uint32_t g = 0; g < grid_cores; g++) {
        CoreCoord core{g % ctx->grid.x, g / ctx->grid.x};
        uint32_t cstart = 0, ccount = 0, widx = 0;
        if (g < W) {
            const uint32_t cnt = base_c + (g < rem_c ? 1u : 0u);
            cstart = cc;
            ccount = cnt;
            widx = g;
            cc += cnt;
        }
        SetRuntimeArgs(prog, ctx->kbin_layout_emit, core, {
            static_cast<uint32_t>(ctx->buf_bin2d->address()),
            l1_base_addr,
            ckpt_addr,
            static_cast<uint32_t>(ctx->buf_bin_ctrl->address()),
            num_tiles,
            stride,
            bucket_fit,
            cstart,
            ccount,
            widx,
        });
    }
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin_layout_emit, false);
}

static void enqueue_bin_layout_kernel(
    SortDeviceContext* ctx,
    uint32_t num_cores,
    uint32_t num_tiles,
    uint32_t stride,
    bool tile_bucket,
    uint32_t tile_ranges_addr,
    bool l1_record,
    uint32_t bucket_fit) {
    if (!ctx->buf_bin_ctrl) {
        ctx->buf_bin_ctrl = make_dram(ctx->mesh_device.get(), PAGE_BYTES);
    }
    // Checkpoint buffer: W slots × {page_acc row, rec_acc row}, each row_pages
    // 64B pages. W is the emit worker count (clamped to num_cores).
    const uint32_t W = std::min(num_cores, kLayoutEmitWorkers);
    const uint32_t row_pages = (num_tiles + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
    const uint32_t row_span = row_pages * ELEMS_PER_PAGE;
    const std::size_t ckpt_bytes =
        static_cast<std::size_t>(W) * 2u * row_span * 4u;
    if (!ctx->buf_layout_ckpt || ctx->cap_layout_ckpt_bytes < ckpt_bytes) {
        ctx->buf_layout_ckpt = make_dram(ctx->mesh_device.get(), ckpt_bytes);
        ctx->cap_layout_ckpt_bytes = ckpt_bytes;
    }
    const uint32_t ckpt_addr = static_cast<uint32_t>(ctx->buf_layout_ckpt->address());

    Program& prog = ctx->wl_bin_layout.get_programs().begin()->second;
    CoreCoord core0{0, 0};
    SetRuntimeArgs(prog, ctx->kbin_layout, core0, {
        static_cast<uint32_t>(ctx->buf_bin2d->address()),
        static_cast<uint32_t>(ctx->buf_tmeta->address()),
        static_cast<uint32_t>(ctx->buf_tile_ids->address()),
        static_cast<uint32_t>(ctx->buf_lpt_meta->address()),
        static_cast<uint32_t>(ctx->buf_tile_counts->address()),
        static_cast<uint32_t>(ctx->buf_bin_ctrl->address()),
        num_cores,
        num_tiles,
        stride,
        tile_bucket && ctx->buf_bin2d_rec
            ? static_cast<uint32_t>(ctx->buf_bin2d_rec->address())
            : 0u,
        tile_bucket && ctx->buf_bucket_meta
            ? static_cast<uint32_t>(ctx->buf_bucket_meta->address())
            : 0u,
        tile_ranges_addr,
        bucket_fit,
        l1_record && ctx->buf_l1_rec_base
            ? static_cast<uint32_t>(ctx->buf_l1_rec_base->address())
            : 0u,
        W,
        ckpt_addr,
    });
    distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin_layout, false);
    // Parallel Pass-2 base emit (CQ-ordered after the coordinator: it sees the
    // coordinator's checkpoint writes and the still-intact histogram in bin2d).
    enqueue_bin_layout_emit_kernel(
        ctx, num_cores, num_tiles, stride, l1_record, bucket_fit, ckpt_addr, W);
}

static bool read_bin_layout_ctrl(
    SortDeviceContext* ctx,
    uint32_t& P_kept,
    uint32_t& P_aligned,
    uint32_t& max_pad_n,
    uint32_t& status) {
    std::vector<uint32_t> ctrl(ELEMS_PER_PAGE, 0);
    distributed::EnqueueReadMeshBuffer(*ctx->cq, ctrl, ctx->buf_bin_ctrl, true);
    P_kept = ctrl[0];
    P_aligned = ctrl[1];
    max_pad_n = ctrl[2];
    status = ctrl[3];
    return status == 0;
}

// Publish the contiguous (sorted_ids, tile_ranges) into device_state as uint32
// DRAM buffers so downstream device stages can read them resident.
static bool resident_blend_chain_enabled() { return true; }  // RESIDENT_BLEND=1

static bool sort_device_publish_enabled();

// Drop the sort-publish Finish() and blend's blocking sort_P_kept D2H so the
// publish kernel chains into FUSED_TILE cull+blend with one CQ drain.
static bool sort_blend_pipe_enabled() {
    return gsplat_tt::env_config::sort_blend_pipe_enabled();
}

// Chain upload+scatter+radix(+publish) into the blend/cull CQ drain — drops the
// per-stage Finish locks between sort kernels when publish pipes to blend.
static bool sort_stage_defer_finish() {
    return sort_blend_pipe_enabled();
}

static void finish_sort_cq_if_needed(SortDeviceContext* ctx) {
    if (device_state::sort_publish_pending()) {
        GSPLAT_HOST_ZONE("host_finish_sort");
        distributed::Finish(*ctx->cq);
        device_state::clear_sort_publish_pending();
    }
}

// Publish LPT tile-id list + per-core (offset,count) and per-tile kept counts
// so blend/cull never scan host tile_ranges or rebuild LPT from host vectors.
static void publish_sort_downstream_metadata(
    SortDeviceContext* ctx,
    const LptAssignment& lpt,
    const std::vector<int64_t>& counts,
    uint32_t num_tiles,
    uint32_t num_cores) {
    device_state::register_buffer("sort_lpt_tile_ids", ctx->buf_tile_ids);

    const uint32_t meta_elems = num_cores * 2;
    const uint32_t meta_pad = round_up(std::max(meta_elems, 1u), ELEMS_PER_PAGE);
    const std::size_t meta_bytes = static_cast<std::size_t>(meta_pad) * 4;
    if (!ctx->buf_lpt_meta || ctx->cap_lpt_meta_bytes < meta_bytes) {
        ctx->buf_lpt_meta = make_dram(ctx->mesh_device.get(), meta_bytes);
        ctx->cap_lpt_meta_bytes = meta_bytes;
        device_state::register_buffer("sort_lpt_meta", ctx->buf_lpt_meta);
    }
    const uint32_t cap_meta = static_cast<uint32_t>(ctx->cap_lpt_meta_bytes / 4);
    std::vector<uint32_t> meta(cap_meta, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        meta[c * 2 + 0] = lpt.per_core_offset[c];
        meta[c * 2 + 1] = lpt.per_core_count[c];
    }
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_lpt_meta, meta, false);

    const uint32_t counts_pad = round_up(std::max(num_tiles, 1u), ELEMS_PER_PAGE);
    const std::size_t counts_bytes = static_cast<std::size_t>(counts_pad) * 4;
    if (!ctx->buf_tile_counts || ctx->cap_tile_counts_bytes < counts_bytes) {
        ctx->buf_tile_counts = make_dram(ctx->mesh_device.get(), counts_bytes);
        ctx->cap_tile_counts_bytes = counts_bytes;
        device_state::register_buffer("sort_tile_counts", ctx->buf_tile_counts);
    }
    const uint32_t cap_counts = static_cast<uint32_t>(ctx->cap_tile_counts_bytes / 4);
    std::vector<uint32_t> cu32(cap_counts, 0);
    for (uint32_t t = 0; t < num_tiles; t++) {
        cu32[t] = static_cast<uint32_t>(counts[t]);
    }
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_counts, cu32, false);

    // Scalar P_kept + padded cull-mask footprint for downstream SFPU cull (blend
    // must not D2H sort_tile_counts to build cull_mask_base).
    if (!ctx->buf_P_kept) {
        ctx->buf_P_kept = make_dram(ctx->mesh_device.get(), PAGE_BYTES);
        device_state::register_buffer("sort_P_kept", ctx->buf_P_kept);
    }
    std::vector<uint32_t> pkept_buf(ELEMS_PER_PAGE, 0);
    uint64_t acc = 0;
    uint64_t mask_elems = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        acc += static_cast<uint64_t>(counts[t]);
        mask_elems += (static_cast<uint64_t>(counts[t]) + 15u) & ~static_cast<uint64_t>(15u);
    }
    pkept_buf[0] = static_cast<uint32_t>(acc);
    pkept_buf[1] = static_cast<uint32_t>(mask_elems);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_P_kept, pkept_buf, false);
    if (sort_blend_pipe_enabled()) {
        device_state::set_sort_blend_pipe_scalars(
            static_cast<uint32_t>(acc), static_cast<uint32_t>(mask_elems));
    }

    if (resident_blend_chain_enabled()) {
        std::vector<uint32_t> mask_base(counts_pad, 0u);
        uint64_t off = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            mask_base[t] = static_cast<uint32_t>(off);
            off += (static_cast<uint64_t>(counts[t]) + 15u) & ~static_cast<uint64_t>(15u);
        }
        const std::size_t base_bytes = static_cast<std::size_t>(counts_pad) * 4;
        if (!ctx->buf_cull_mask_base || ctx->cap_cull_mask_base_bytes < base_bytes) {
            ctx->buf_cull_mask_base = make_dram(ctx->mesh_device.get(), base_bytes);
            ctx->cap_cull_mask_base_bytes = base_bytes;
            device_state::register_buffer("cull_mask_base", ctx->buf_cull_mask_base);
        }
        distributed::EnqueueWriteMeshBuffer(
            *ctx->cq, ctx->buf_cull_mask_base, mask_base, false);
    }
}

static void publish_resident(
    SortDeviceContext* ctx,
    const std::vector<int64_t>& sorted_ids,
    const std::vector<int64_t>& tile_ranges,
    double* publish_ms) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t P = static_cast<uint32_t>(sorted_ids.size());
    const uint32_t P_pad = round_up(std::max<uint32_t>(P, 1), ELEMS_PER_PAGE);
    const std::size_t sorted_bytes = static_cast<std::size_t>(P_pad) * 4;
    if (!ctx->buf_sorted_ids || ctx->cap_sorted_bytes < sorted_bytes) {
        ctx->buf_sorted_ids = make_dram(ctx->mesh_device.get(), sorted_bytes);
        ctx->cap_sorted_bytes = sorted_bytes;
        device_state::register_buffer("sort_sorted_ids", ctx->buf_sorted_ids);
    }
    // buf_sorted_ids is grow-only: a smaller-P frame keeps a larger (hero)
    // capacity. EnqueueWriteMeshBuffer writes the WHOLE buffer, so the host
    // vector must be sized to capacity (not the current P_pad) or tt-metal
    // asserts "source vector too small" and crashes the 30-view sweep.
    const uint32_t cap_sorted_elems = static_cast<uint32_t>(ctx->cap_sorted_bytes / 4);
    std::vector<uint32_t> sids(cap_sorted_elems, 0);
    for (uint32_t i = 0; i < P; i++)
        sids[i] = static_cast<uint32_t>(sorted_ids[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_sorted_ids, sids, false);

    const uint32_t R = static_cast<uint32_t>(tile_ranges.size());
    const uint32_t R_pad = round_up(std::max<uint32_t>(R, 1), ELEMS_PER_PAGE);
    const std::size_t ranges_bytes = static_cast<std::size_t>(R_pad) * 4;
    if (!ctx->buf_tile_ranges || ctx->cap_ranges_bytes < ranges_bytes) {
        ctx->buf_tile_ranges = make_dram(ctx->mesh_device.get(), ranges_bytes);
        ctx->cap_ranges_bytes = ranges_bytes;
        device_state::register_buffer("sort_tile_ranges", ctx->buf_tile_ranges);
    }
    const uint32_t cap_ranges_elems = static_cast<uint32_t>(ctx->cap_ranges_bytes / 4);
    std::vector<uint32_t> ranges(cap_ranges_elems, 0);
    for (uint32_t i = 0; i < R; i++)
        ranges[i] = static_cast<uint32_t>(tile_ranges[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ranges, ranges, false);
    distributed::Finish(*ctx->cq);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (publish_ms) *publish_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Grow-only resident sort_sorted_ids buffer (no host id upload).
static void ensure_resident_sorted_buffer(SortDeviceContext* ctx, uint32_t P_kept) {
    const uint32_t P_pad = round_up(std::max<uint32_t>(P_kept, 1), ELEMS_PER_PAGE);
    const std::size_t sorted_bytes = static_cast<std::size_t>(P_pad) * 4;
    if (!ctx->buf_sorted_ids || ctx->cap_sorted_bytes < sorted_bytes) {
        ctx->buf_sorted_ids = make_dram(ctx->mesh_device.get(), sorted_bytes);
        ctx->cap_sorted_bytes = sorted_bytes;
        device_state::register_buffer("sort_sorted_ids", ctx->buf_sorted_ids);
    }
}

// Upload tile_ranges only (grow-on-demand whole-buffer write).
static void upload_resident_tile_ranges(
    SortDeviceContext* ctx, const std::vector<int64_t>& tile_ranges) {
    const uint32_t R = static_cast<uint32_t>(tile_ranges.size());
    const uint32_t R_pad = round_up(std::max<uint32_t>(R, 1), ELEMS_PER_PAGE);
    const std::size_t ranges_bytes = static_cast<std::size_t>(R_pad) * 4;
    if (!ctx->buf_tile_ranges || ctx->cap_ranges_bytes < ranges_bytes) {
        ctx->buf_tile_ranges = make_dram(ctx->mesh_device.get(), ranges_bytes);
        ctx->cap_ranges_bytes = ranges_bytes;
        device_state::register_buffer("sort_tile_ranges", ctx->buf_tile_ranges);
    }
    const uint32_t cap_ranges_elems = static_cast<uint32_t>(ctx->cap_ranges_bytes / 4);
    std::vector<uint32_t> ranges(cap_ranges_elems, 0);
    for (uint32_t i = 0; i < R; i++)
        ranges[i] = static_cast<uint32_t>(tile_ranges[i]);
    distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ranges, ranges, false);
}

static bool sort_device_publish_enabled() { return true; }  // SORT_DEVICE_PUBLISH=1

static void maybe_run_sort_blend_continuation(
    SortBlendContinuation* cont, int tiles_x, uint32_t num_tiles) {
    if (cont == nullptr || cont->image_out == nullptr) {
        return;
    }
    if (!sort_blend_pipe_enabled() || !resident_blend_chain_enabled()) {
        return;
    }
    bool blend_ok = false;
    double cull_ms = 0.0, blend_ms = 0.0;
    (void)blend_mb_devcull_resident(
        cont->mb_contrib_floor,
        cont->cull_disabled,
        static_cast<int>(num_tiles),
        tiles_x,
        cont->image_height,
        cont->image_width,
        cont->image_out,
        &blend_ok,
        &cull_ms,
        &blend_ms,
        cont->transmittance_threshold);
    cont->cull_ms = cull_ms;
    cont->blend_ms = blend_ms;
    cont->invoked = true;
    if (cont->blend_ok != nullptr) {
        *cont->blend_ok = blend_ok;
    }
}

// ── R4/R5 resident-pairs device binning ─────────────────────────────────
// Bins the resident full-P (gid,tid) pairs + keep mask into the page-aligned
// per-tile (key,id) layout on-device, runs the radix kernel, compacts, and
// publishes. Returns a SortResult identical in shape to the host path.
static gsplat_cpu::SortResult sort_resident_pairs(
    SortDeviceContext* ctx,
    uint32_t num_tiles,
    int tiles_x,
    int tiles_y,
    std::size_t M,
    bool need_host_sorted_ids,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    SortCallTimings& T,
    SortBlendContinuation* sort_blend) {
    using clk = std::chrono::high_resolution_clock;
    const auto t_total0_rp = clk::now();
    (void)pool;  // resident binning needs no host thread pool; kept for ABI
    auto fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::SortResult{};
    };
    gsplat_cpu::SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);

    if (num_tiles > MAX_BIN_TILES) {
        std::cerr << "[gsplat_tt::sort] num_tiles=" << num_tiles
                  << " > MAX_BIN_TILES=" << MAX_BIN_TILES
                  << "; unsupported (render_clean is single-path TT, no host "
                     "fallback) — hard fail\n";
        return fail();
    }

    auto bgid = device_state::get_buffer("ta_pairs_gid");
    auto btid = device_state::get_buffer("ta_pairs_tid");
    auto bkeep = device_state::get_buffer("ta_pairs_keep");
    auto bP = device_state::get_buffer("ta_pairs_P");
    auto bdep = device_state::get_buffer("proj_m_depth");
    if (!bgid || !btid || !bkeep || !bP || !bdep) {
        std::cerr << "[gsplat_tt::sort] resident pairs / proj_m_depth missing; "
                     "the upstream resident stages did not run — hard fail "
                     "(render_clean is single-path TT, no host fallback)\n";
        return fail();
    }

    try {
        // Read full P + P_pad published by tile_assign.
        std::vector<uint32_t> pbuf(ELEMS_PER_PAGE);
        distributed::EnqueueReadMeshBuffer(*ctx->cq, pbuf, bP, true);
        T.pread_ms =
            std::chrono::duration<double, std::milli>(clk::now() - t_total0_rp).count();
        const uint32_t P_full = pbuf[0];
        const uint32_t P_pad = pbuf[1];
        // S5.3 host-free overflow guard: tile_assign's scan_bases CLAMPS the
        // published P (pbuf[0]) to the static pair ceiling so neither it nor K2
        // ever indexes past the p_max-sized pair buffers (no silent memory
        // corruption). pbuf[2]=overflow / pbuf[3]=P_true surface a too-small
        // ceiling here — the one place that would read OOB — as a hard fail
        // (render_clean is single-path TT, no fallback). This reuses the P read
        // already on this path, so it adds no new mid-frame drain.
        const uint32_t p_overflow = pbuf[2];
        const uint32_t P_true = pbuf[3];
        if (p_overflow != 0) {
            std::cerr << "[gsplat_tt::sort] PAIR-CEILING OVERFLOW: pre-cull P_true="
                      << P_true << " exceeds the static pair_ceiling()=" << P_full
                      << " — pairs were clamped (output would be corrupt). Raise "
                         "env_config::pair_ceiling() above " << P_true
                      << " and rebuild. Hard fail (single-path TT, no fallback).\n";
            return fail();
        }
        if (P_full == 0) {
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);
            if (device_ok) *device_ok = true;
            return result;
        }

        const uint32_t num_cores = ctx->grid.x * ctx->grid.y;
        const uint32_t dump_tile = 0;
        const uint32_t stride = round_up(num_tiles, ELEMS_PER_PAGE);
        const uint32_t total_p_pages = P_pad / ELEMS_PER_PAGE;
        const PageSplit ws = split_pages(total_p_pages, num_cores);

        // ── bin2d (per-core hist / base rows) ───────────────────────────
        const std::size_t bin2d_bytes =
            static_cast<std::size_t>(num_cores) * stride * 4;
        if (!ctx->buf_bin2d || ctx->cap_bin2d_bytes < bin2d_bytes) {
            ctx->buf_bin2d = make_dram(ctx->mesh_device.get(), bin2d_bytes);
            ctx->cap_bin2d_bytes = bin2d_bytes;
        }
        if (!ctx->buf_bin_hist || ctx->cap_bin_hist_bytes < bin2d_bytes) {
            ctx->buf_bin_hist = make_dram(ctx->mesh_device.get(), bin2d_bytes);
            ctx->cap_bin_hist_bytes = bin2d_bytes;
        }
        if (sort_emit_movers() == 2 &&
            (!ctx->buf_bin_h0 || ctx->cap_bin_h0_bytes < bin2d_bytes)) {
            ctx->buf_bin_h0 = make_dram(ctx->mesh_device.get(), bin2d_bytes);
            ctx->cap_bin_h0_bytes = bin2d_bytes;
        }

        // T1 (GSPLAT_TT_TILE_BUCKET): scatter full records into per-tile buckets.
        const bool tile_bucket = tile_bucket_enabled();
        auto bbrec = tile_bucket ? device_state::get_buffer("proj_m_blendrec") : nullptr;
        if (tile_bucket && !bbrec) {
            std::cerr << "[gsplat_tt::sort] proj_m_blendrec missing; the gather "
                         "stage did not run — hard fail (render_clean is "
                         "single-path TT, no host fallback)\n";
            return fail();
        }
        const uint32_t blendrec_addr = bbrec ? static_cast<uint32_t>(bbrec->address()) : 0u;
        uint32_t tile_recs_addr = 0u;  // real address set after P_aligned is known

        // Metadata buffers (layout kernel writes these resident on device).
        const uint32_t tmeta_pad = round_up(std::max(num_tiles * 2u, 1u), ELEMS_PER_PAGE);
        const std::size_t tmeta_bytes = static_cast<std::size_t>(tmeta_pad) * 4;
        if (!ctx->buf_tmeta || ctx->cap_tmeta_bytes < tmeta_bytes) {
            ctx->buf_tmeta = make_dram(ctx->mesh_device.get(), tmeta_bytes);
            ctx->cap_tmeta_bytes = tmeta_bytes;
        }
        const uint32_t tile_ids_pad = round_up(std::max(num_tiles, 1u), ELEMS_PER_PAGE);
        const std::size_t tile_ids_bytes = static_cast<std::size_t>(tile_ids_pad) * 4;
        if (!ctx->buf_tile_ids || ctx->cap_tile_ids_bytes < tile_ids_bytes) {
            ctx->buf_tile_ids = make_dram(ctx->mesh_device.get(), tile_ids_bytes);
            ctx->cap_tile_ids_bytes = tile_ids_bytes;
        }
        device_state::register_buffer("sort_lpt_tile_ids", ctx->buf_tile_ids);
        const uint32_t meta_elems = num_cores * 2;
        const uint32_t meta_pad = round_up(std::max(meta_elems, 1u), ELEMS_PER_PAGE);
        const std::size_t meta_bytes = static_cast<std::size_t>(meta_pad) * 4;
        if (!ctx->buf_lpt_meta || ctx->cap_lpt_meta_bytes < meta_bytes) {
            ctx->buf_lpt_meta = make_dram(ctx->mesh_device.get(), meta_bytes);
            ctx->cap_lpt_meta_bytes = meta_bytes;
        }
        device_state::register_buffer("sort_lpt_meta", ctx->buf_lpt_meta);
        const uint32_t counts_pad = round_up(std::max(num_tiles, 1u), ELEMS_PER_PAGE);
        const std::size_t counts_bytes = static_cast<std::size_t>(counts_pad) * 4;
        if (!ctx->buf_tile_counts || ctx->cap_tile_counts_bytes < counts_bytes) {
            ctx->buf_tile_counts = make_dram(ctx->mesh_device.get(), counts_bytes);
            ctx->cap_tile_counts_bytes = counts_bytes;
        }
        device_state::register_buffer("sort_tile_counts", ctx->buf_tile_counts);
        const std::size_t ranges_bytes =
            static_cast<std::size_t>(round_up(num_tiles * 2u, ELEMS_PER_PAGE)) * 4;
        if (!ctx->buf_tile_ranges || ctx->cap_ranges_bytes < ranges_bytes) {
            ctx->buf_tile_ranges = make_dram(ctx->mesh_device.get(), ranges_bytes);
            ctx->cap_ranges_bytes = ranges_bytes;
        }
        device_state::register_buffer("sort_tile_ranges", ctx->buf_tile_ranges);
        if (tile_bucket) {
            const std::size_t rec_base_bytes =
                static_cast<std::size_t>(num_cores) * stride * 4;
            if (!ctx->buf_bin2d_rec || ctx->cap_bin2d_rec_bytes < rec_base_bytes) {
                ctx->buf_bin2d_rec = make_dram(ctx->mesh_device.get(), rec_base_bytes);
                ctx->cap_bin2d_rec_bytes = rec_base_bytes;
            }
            const uint32_t bm_pad = round_up(num_tiles * 2u, ELEMS_PER_PAGE);
            const std::size_t bm_bytes = static_cast<std::size_t>(bm_pad) * 4;
            if (!ctx->buf_bucket_meta || ctx->cap_bucket_meta_bytes < bm_bytes) {
                ctx->buf_bucket_meta = make_dram(ctx->mesh_device.get(), bm_bytes);
                ctx->cap_bucket_meta_bytes = bm_bytes;
                device_state::register_buffer("sort_bucket_meta", ctx->buf_bucket_meta);
            }
        }

        // ── Task #106 (lever 1): one-launch device sort ─────────────────────
        // GSPLAT_TT_SORT_ONELAUNCH (default on): sort_bin_onelaunch.cpp counts, lays out
        // (device prefix sum over cores between two semaphore barriers) and
        // emits every record into its tile's fixed-capacity bucket in ONE
        // launch; the materialize sorts each bucket (canonical order, stable
        // depth radix), byte-identical to the prefix-sum layout's sort. The
        // host reads the totals rows (8 KB) instead of the histogram, uploads
        // no layout and launches no radix or publish.
        ctx->ol_frame = false;
        // Task #170 fold: the segment K2's per-mover count rows (taken every
        // frame so a row set never outlives its pairs).
        device_state::K2CountRows k2rows;
        const bool have_k2rows = device_state::take_k2_count_rows(&k2rows);
        if (sort_onelaunch_enabled() && tile_bucket && !need_host_sorted_ids &&
            resident_blend_chain_enabled() && sort_device_publish_enabled()) {
            using ms_t = std::chrono::duration<double, std::milli>;
            if (!ctx->ol_built) build_program_sort_onelaunch(*ctx);
            const uint32_t cap = kOneLaunchTileCap;
            const uint32_t row_pages = stride / ELEMS_PER_PAGE;
            auto* dev = ctx->mesh_device.get();
            const std::size_t bucket_bytes = static_cast<std::size_t>(num_tiles) * cap * 32u;
            if (!ctx->buf_ol_bucket || ctx->cap_ol_bucket_bytes < bucket_bytes) {
                ctx->buf_ol_bucket = make_dram_paged(dev, bucket_bytes, render_config::kRecPageBytes);
                ctx->cap_ol_bucket_bytes = bucket_bytes;
            }
            // The blend's argument lists name sort_l1_recs and sort_tile_recs; a
            // run with only one-launch frames never creates them (not read here).
            if (!device_state::get_buffer("sort_l1_recs")) {
                device_state::register_buffer("sort_l1_recs", ctx->buf_ol_bucket);
            }
            if (!ctx->buf_tile_recs) {
                ctx->buf_tile_recs = make_dram(dev, PAGE_BYTES);
                ctx->cap_tile_recs_bytes = PAGE_BYTES;
                device_state::register_buffer("sort_tile_recs", ctx->buf_tile_recs);
            }
            const std::size_t rows_bytes = static_cast<std::size_t>(num_cores) * stride * 4u;
            if (!ctx->buf_ol_counts || ctx->cap_ol_rows_bytes < rows_bytes) {
                ctx->buf_ol_counts = make_dram(dev, rows_bytes);
                ctx->buf_ol_bases = make_dram(dev, rows_bytes);
                ctx->cap_ol_rows_bytes = rows_bytes;
            }
            const std::size_t totals_bytes = static_cast<std::size_t>(stride) * 2u * 4u;
            if (!ctx->buf_ol_totals || ctx->cap_ol_totals_bytes < totals_bytes) {
                ctx->buf_ol_totals = make_dram(dev, totals_bytes);
                ctx->cap_ol_totals_bytes = totals_bytes;
            }

            // This launch's mover page ranges: core c's BRISC [lo, mid), NCRISC
            // [mid, hi); task #174 speed-proportional when ol_mover_speed().
            std::vector<uint32_t> noc_xy(num_cores, 0u);
            for (uint32_t c = 0; c < num_cores; c++) {
                const CoreCoord v = ctx->mesh_device->worker_core_from_logical_core(
                    CoreCoord{c % ctx->grid.x, c / ctx->grid.x});
                noc_xy[c] = static_cast<uint32_t>(v.x) | (static_cast<uint32_t>(v.y) << 16);
            }
            std::vector<uint32_t> sb;  // task #174: speed-proportional mover ranges
            if (ol_mover_speed()) {
                sb = gsplat_tt::sort_split::speed_bounds(
                    total_p_pages, gsplat_tt::sort_split::mover_speeds(noc_xy));
            }
            std::vector<uint32_t> r_lo(num_cores), r_mid(num_cores), r_hi(num_cores);
            for (uint32_t c = 0; c < num_cores; c++) {
                r_lo[c] = ws.start[c];
                r_hi[c] = ws.start[c] + ws.count[c];
                r_mid[c] = r_lo[c] + gsplat_tt::sort_split::split_pages(
                    ws.count[c], gsplat_tt::sort_split::row_permille(ol_split_rows(), c / ctx->grid.x,
                                                          sort_emit_split_permille()));
                if (!sb.empty()) {
                    r_lo[c] = sb[2u * c];
                    r_mid[c] = sb[2u * c + 1u];
                    r_hi[c] = sb[2u * c + 2u];
                }
            }
            // Fold only if the K2 counted exactly these ranges. why: 0 folded,
            // 1 rows absent, 2 cores, 3 row pages, 4 tiles, 5 P, 7 a core's
            // page range (logged when it changes).
            int why = !have_k2rows                                    ? 1
                      : k2rows.num_cores != num_cores                 ? 2
                      : k2rows.row_pages != row_pages                 ? 3
                      : k2rows.num_tiles != num_tiles                 ? 4
                      : k2rows.P_pub != P_full                        ? 5
                      : k2rows.bounds.size() != 2u * num_cores + 1u   ? 7
                                                                      : 0;
            for (uint32_t c = 0; why == 0 && c < num_cores; c++) {
                if (k2rows.bounds[2u * c] != r_lo[c] || k2rows.bounds[2u * c + 1u] != r_mid[c] ||
                    k2rows.bounds[2u * c + 2u] != r_hi[c])
                    why = 7;
            }
            const bool fold = why == 0;
            {
                static int logged = -1;
                if (logged != why) {
                    logged = why;
                    std::fprintf(stderr,
                                 "[SORT] ONELAUNCH k2_fold=%d why=%d (K2 rows %s: cores %u/%u "
                                 "row_pages %u/%u tiles %u/%u P %u/%u speed_split %d)\n",
                                 static_cast<int>(fold), why, have_k2rows ? "published" : "absent",
                                 k2rows.num_cores, num_cores, k2rows.row_pages, row_pages,
                                 k2rows.num_tiles, num_tiles, k2rows.P_pub, P_full,
                                 static_cast<int>(!sb.empty()));
                }
            }
            const uint32_t cnt_rows_addr = fold
                ? static_cast<uint32_t>(k2rows.buf->address())
                : static_cast<uint32_t>(ctx->buf_ol_counts->address());

            const auto t_e0 = clk::now();
            Program& oprog = ctx->wl_onelaunch.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                const uint32_t lo = r_lo[c], mid = r_mid[c], hi = r_hi[c];
                std::vector<uint32_t> a = {
                    static_cast<uint32_t>(bgid->address()),
                    static_cast<uint32_t>(btid->address()),
                    static_cast<uint32_t>(bkeep->address()),
                    static_cast<uint32_t>(bdep->address()),
                    blendrec_addr,
                    static_cast<uint32_t>(ctx->buf_ol_bucket->address()),
                    cnt_rows_addr,
                    static_cast<uint32_t>(ctx->buf_ol_bases->address()),
                    static_cast<uint32_t>(ctx->buf_ol_totals->address()),
                    mid, hi, P_full, num_tiles, row_pages, c, num_cores, cap,
                    static_cast<uint32_t>(tiles_x), 1u,
                    ctx->ol_sem[0], ctx->ol_sem[1], ctx->ol_sem[2],
                    ctx->ol_sem[3], ctx->ol_sem[4], ctx->ol_sem[5],
                    noc_xy[0] & 0xFFFFu, noc_xy[0] >> 16, fold ? 1u : 0u,
                };
                SetRuntimeArgs(oprog, ctx->kol, core, a);
                a[9] = lo;
                a[10] = mid;
                a[18] = 0u;
                if (c == 0) a.insert(a.end(), noc_xy.begin(), noc_xy.end());  // coordinator
                SetRuntimeArgs(oprog, ctx->kol0, core, a);
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_onelaunch, false);
            std::vector<uint32_t> tot(ctx->cap_ol_totals_bytes / 4, 0u);
            {
                GSPLAT_HOST_ZONE("host_finish_sort_onelaunch");
                distributed::EnqueueReadMeshBuffer(*ctx->cq, tot, ctx->buf_ol_totals, true);
            }
            const auto t_e1 = clk::now();
            T.bin_emit_ms = ms_t(t_e1 - t_e0).count();
            T.bin_ms = T.bin_emit_ms;
            // GSPLAT_TT_SORT_ONELAUNCH_CHECK=1 (debug): read the count and base
            // rows back and check the device prefix against the host's.
            static const bool ol_check = [] {
                const char* e = std::getenv("GSPLAT_TT_SORT_ONELAUNCH_CHECK");
                return e != nullptr && e[0] == '1';
            }();
            if (ol_check) {
                std::vector<uint32_t> crow(ctx->cap_ol_rows_bytes / 4, 0u);
                std::vector<uint32_t> brow(ctx->cap_ol_rows_bytes / 4, 0u);
                if (fold) {
                    // Per-core rows = the sum of the K2's two mover rows.
                    std::vector<uint32_t> krow(k2rows.bytes / 4, 0u);
                    distributed::EnqueueReadMeshBuffer(*ctx->cq, krow, k2rows.buf, true);
                    for (uint32_t c = 0; c < num_cores; c++) {
                        for (uint32_t t = 0; t < stride; t++) {
                            crow[c * stride + t] =
                                krow[(2u * c) * stride + t] + krow[(2u * c + 1u) * stride + t];
                        }
                    }
                } else {
                    distributed::EnqueueReadMeshBuffer(*ctx->cq, crow, ctx->buf_ol_counts, true);
                }
                distributed::EnqueueReadMeshBuffer(*ctx->cq, brow, ctx->buf_ol_bases, true);
                const uint32_t bad = sort_onelaunch::check_prefix(crow, brow, tot, num_cores,
                                                                  stride, num_tiles);
                std::fprintf(stderr, "[SORT] ONELAUNCH_CHECK bad_tiles=%u\n", bad);
            }

            // Per-tile layout from the totals alone.
            std::vector<int64_t> counts(num_tiles, 0);
            std::vector<int64_t> pad_counts(num_tiles, 0);
            std::vector<uint32_t> bmeta(ctx->cap_bucket_meta_bytes / 4, 0u);
            uint32_t P_kept = 0;
            uint32_t max_n = 0;
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t n = tot[t];
                if (n > cap) {
                    std::cerr << "[gsplat_tt::sort] tile " << t << " holds " << n
                              << " records > bucket capacity " << cap
                              << " (records past it were dropped) — hard fail "
                                 "(render_clean is single-path TT, no host fallback)\n";
                    return fail();
                }
                if (tot[stride + t] > MAX_TILE_ENTRIES) {  // the legacy layout's status 2
                    std::cerr << "[gsplat_tt::sort] padded tile exceeds MAX_TILE_ENTRIES; "
                                 "hard fail (render_clean is single-path TT, no host fallback)\n";
                    return fail();
                }
                counts[t] = n;
                pad_counts[t] = tot[stride + t];  // == the legacy LPT cost (per-core pages)
                bmeta[static_cast<std::size_t>(t) * 2 + 0] = P_kept;
                bmeta[static_cast<std::size_t>(t) * 2 + 1] = n;
                if (n > 0) {
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = P_kept;
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] = P_kept + n;
                }
                P_kept += n;
                max_n = std::max(max_n, n);
            }
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bucket_meta, bmeta, false);
            const LptAssignment lpt =
                build_blend_lists(counts, pad_counts, num_tiles, num_cores);
            std::vector<uint32_t> tile_ids_flat(ctx->cap_tile_ids_bytes / 4, 0u);
            std::copy(lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
            // Padded [start, start + count) per tile, the publish path's layout
            // (the materialize and blend read counts from it).
            std::vector<int64_t> padded_ranges(result.tile_ranges.size(), 0);
            uint32_t padded_cursor = 0;
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t n = static_cast<uint32_t>(counts[t]);
                padded_ranges[2 * t] = static_cast<int64_t>(padded_cursor);
                padded_ranges[2 * t + 1] = static_cast<int64_t>(padded_cursor + n);
                padded_cursor += round_up(n, ELEMS_PER_PAGE);
            }
            // sort_sorted_ids has no reader here; keep one registered for the
            // blend's argument lists.
            ensure_resident_sorted_buffer(ctx, ELEMS_PER_PAGE);
            upload_resident_tile_ranges(ctx, padded_ranges);
            const auto t_l1 = clk::now();
            T.bin_layout_ms = ms_t(t_l1 - t_e1).count();

            const SubchunkLayout sc_layout =
                build_subchunk_layout(counts, num_tiles, render_config::kBucketFit);
            log_subchunk_layout_stats(sc_layout);
            const MatWorkAssignment ol_work =
                build_mat_worklist(counts, num_tiles, num_cores, render_config::kBucketFit,
                                   sort_mat_movers(), kMatMover0Cap, /*onelaunch=*/true,
                                   gsplat_tt::env_config::ol_mat_select());
            if (ol_work.max_items_per_core > 1024u) {
                std::cerr << "[gsplat_tt::sort] materialize work items/core "
                          << ol_work.max_items_per_core << " > MAX_WORK=1024\n";
                return fail();
            }
            if (!prepare_subchunk_buffers(ctx, sc_layout, num_tiles)) {
                std::cerr << "[gsplat_tt::sort] subchunk buffer setup failed\n";
                return fail();
            }
            upload_subchunk_directory(ctx, sc_layout);
            T.publish_host_ms = ms_t(clk::now() - t_l1).count();
            T.publish_ms = T.publish_host_ms;
            T.total_ms = ms_t(clk::now() - t_total0_rp).count();
            std::fprintf(stderr,
                "[SORT] stage=ONELAUNCH P=%u P_kept=%u num_tiles=%u max_tile_n=%u "
                "onelaunch=%.2f layout=%.2f pub_host=%.2f total=%.2fms\n",
                P_full, P_kept, num_tiles, max_n, T.bin_emit_ms, T.bin_layout_ms,
                T.publish_host_ms, T.total_ms);
            if (device_ok) *device_ok = true;
            // Materialize as the legacy path does (piped: before the blend, no
            // drain unless GSPLAT_TT_SPLIT_BLEND=1).
            ctx->ol_frame = true;
            bool mat_ok = true;
            if (sort_blend_pipe_enabled()) {
                device_state::mark_sort_publish_pending();
                const bool split = stagetimers::split_blend();
                stagetimers::Span mat_span(split ? stagetimers::acc().mat : T.materialize_ms);
                mat_ok = launch_subchunk_materialize(ctx, ol_work, num_cores,
                                                     static_cast<uint32_t>(tiles_x),
                                                     render_config::kBucketFit, sort_blend);
                if (mat_ok && split) {
                    GSPLAT_HOST_ZONE("host_finish_mat_split");
                    distributed::Finish(*ctx->cq);
                }
                mat_span.stop();
            } else {
                const auto t_mat0 = clk::now();
                mat_ok = launch_subchunk_materialize(ctx, ol_work, num_cores,
                                                     static_cast<uint32_t>(tiles_x),
                                                     render_config::kBucketFit);
                if (mat_ok) {
                    GSPLAT_HOST_ZONE("host_finish_sort_materialize");
                    distributed::Finish(*ctx->cq);
                }
                T.materialize_ms = ms_t(clk::now() - t_mat0).count();
            }
            ctx->ol_frame = false;
            if (!mat_ok) {
                std::cerr << "[gsplat_tt::sort] subchunk materialize launch failed\n";
                return fail();
            }
            maybe_run_sort_blend_continuation(sort_blend, tiles_x, num_tiles);
            return result;
        }

        // M0: l1_record buffers.
        const bool l1_record_early = gsplat_tt::env_config::l1_record_enabled();
        const uint32_t bucket_fit = render_config::kBucketFit;
        uint32_t l1_recs_addr = 0u;
        uint32_t l1_base_addr = 0u;
        // iter-138: overflow pre-pack region + per-(core,tile) base row. Sized and
        // filled AFTER the host layout pass (count-dependent), so they stay 0 for
        // the count pass (launch_bin(0)) and the kernel's overflow path is disabled
        // there; set to real addresses before the scatter pass (launch_bin(1)).
        uint32_t l1_ov_addr = 0u;
        uint32_t l1_ov_base_addr = 0u;
        if (l1_record_early) {
            // M0/iter50: two 32B splats per 64B DRAM page (PACK2). Sub-64B paging
            // is unreliable; 64B pages hold low/high splat at +0/+32. kBucketFit
            // logical slots => bucket_fit/2 pages per tile.
            const std::size_t l1_rec_bytes =
                static_cast<std::size_t>(num_tiles) * bucket_fit * 32u;
            if (!ctx->buf_l1_recs || ctx->cap_l1_recs_bytes < l1_rec_bytes) {
                ctx->buf_l1_recs = make_dram_paged(
                    ctx->mesh_device.get(), l1_rec_bytes, render_config::kRecPageBytes);
                ctx->cap_l1_recs_bytes = l1_rec_bytes;
                device_state::register_buffer("sort_l1_recs", ctx->buf_l1_recs);
            }
            l1_recs_addr = static_cast<uint32_t>(ctx->buf_l1_recs->address());
            const std::size_t l1_base_bytes =
                static_cast<std::size_t>(num_cores) * stride * 4u;
            if (!ctx->buf_l1_rec_base || ctx->cap_l1_rec_base_bytes < l1_base_bytes) {
                ctx->buf_l1_rec_base = make_dram(ctx->mesh_device.get(), l1_base_bytes);
                ctx->cap_l1_rec_base_bytes = l1_base_bytes;
                device_state::register_buffer("sort_l1_rec_base", ctx->buf_l1_rec_base);
            }
            l1_base_addr = static_cast<uint32_t>(ctx->buf_l1_rec_base->address());
        }

        const bool device_layout = gsplat_tt::env_config::sort_device_layout_enabled();
        const bool layout_verify = false;
        // The (gated-off) device layout kernels transform the histogram in bin2d
        // in place, so only the host bridge keeps it in buf_bin_hist for the emit.
        const bool hist_rows = !device_layout && !layout_verify;

        // T-C: split each core's emit over BRISC (pages [lo, mid)) and NCRISC
        // ([mid, hi)). Needs the count pass's rows (the emit's whole-core counts
        // and the h0 snapshot at mid) and the full tile_bucket/l1_record arg list.
        // Ordering invariant (=> byte-identical output): lo <= mid <= hi and the
        // count pass snapshots at the SAME mid the emit splits at, so mover 1's
        // cursors start exactly where the single-mover pass would have them.
        const bool dual_emit =
            sort_emit_movers() == 2 && hist_rows && tile_bucket && l1_record_early;
        std::vector<uint32_t> mover_mid(num_cores);
        for (uint32_t c = 0; c < num_cores; c++) {
            mover_mid[c] = ws.start[c] + static_cast<uint32_t>(
                static_cast<uint64_t>(ws.count[c]) * sort_emit_split_permille() / 1000u);
            if (mover_mid[c] < ws.start[c] || mover_mid[c] > ws.start[c] + ws.count[c]) {
                std::cerr << "[gsplat_tt::sort] emit mover split outside core " << c
                          << "'s page range — hard fail\n";
                return fail();
            }
        }

        // ── Pass A: per-core histogram (count) ──────────────────────────
        const auto t_bin0 = clk::now();
        auto launch_bin = [&](uint32_t mode, bool finish_cq) {
            Program& prog = ctx->wl_bin.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                // mode 0 writes the histogram (buf_bin_hist on the host bridge);
                // mode 1 reads the page bases the host wrote into bin2d.
                const auto& row_buf =
                    (mode == 0 && hist_rows) ? ctx->buf_bin_hist : ctx->buf_bin2d;
                std::vector<uint32_t> args = {
                    static_cast<uint32_t>(bgid->address()),
                    static_cast<uint32_t>(btid->address()),
                    static_cast<uint32_t>(bkeep->address()),
                    static_cast<uint32_t>(bdep->address()),
                    static_cast<uint32_t>(row_buf->address()),
                    ctx->buf_keys ? static_cast<uint32_t>(ctx->buf_keys->address()) : 0u,
                    ctx->buf_ids ? static_cast<uint32_t>(ctx->buf_ids->address()) : 0u,
                    ws.start[c], ws.count[c], P_full, num_tiles, stride, c, mode,
                    dump_tile,
                };
                if (tile_bucket) {
                    args.push_back(blendrec_addr);
                    args.push_back(tile_recs_addr);
                    // Arg 17: count-pass histogram rows for the emit (0 => recount).
                    args.push_back((mode == 1 && hist_rows)
                                       ? static_cast<uint32_t>(ctx->buf_bin_hist->address())
                                       : 0u);
                }
                if (l1_record_early) {
                    args.push_back(l1_recs_addr);
                    args.push_back(l1_base_addr);
                    args.push_back(bucket_fit);  // per-tile bucket slot count (clamp)
                    args.push_back(static_cast<uint32_t>(tiles_x));  // tile-local mean
                    args.push_back(l1_ov_addr);       // iter-138: overflow region (0=off)
                    args.push_back(l1_ov_base_addr);  // iter-138: per-(core,tile) ov base
                }
                // T-C args 24..29: mover, dual, h0 rows, mid, own / peer semaphore.
                args.resize(24, 0u);
                const uint32_t lo = ws.start[c];
                const uint32_t hi = ws.start[c] + ws.count[c];
                const uint32_t mid = mover_mid[c];
                const uint32_t h0_addr =
                    ctx->buf_bin_h0 ? static_cast<uint32_t>(ctx->buf_bin_h0->address()) : 0u;
                std::vector<uint32_t> args0 = args;  // BRISC (mover 0)
                args.insert(args.end(), {1u, dual_emit ? 1u : 0u, h0_addr, mid,
                                         ctx->bin_sem[1], ctx->bin_sem[0]});
                args0.insert(args0.end(), {0u, dual_emit ? 1u : 0u, h0_addr, mid,
                                           ctx->bin_sem[0], ctx->bin_sem[1]});
                if (mode == 1 && dual_emit) {
                    args0[7] = lo;
                    args0[8] = mid - lo;
                    args[7] = mid;
                    args[8] = hi - mid;
                }
                SetRuntimeArgs(prog, ctx->kbin, core, args);
                if (sort_emit_movers() == 2) SetRuntimeArgs(prog, ctx->kbin0, core, args0);
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_bin, false);
            if (finish_cq) {
                if (mode == 0) {
                    GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
                } else {
                    GSPLAT_HOST_ZONE("host_finish_sort_bin_scat");
                }
                distributed::Finish(*ctx->cq);
            }
        };
        launch_bin(0, !device_layout);
        const auto t_cnt = clk::now();
        T.bin_count_ms = std::chrono::duration<double, std::milli>(t_cnt - t_bin0).count();

        std::vector<int64_t> counts(num_tiles, 0);
        std::vector<int64_t> starts(num_tiles, 0);
        std::vector<uint32_t> pstart_page(num_tiles, 0);
        std::vector<uint32_t> pstart_elem(num_tiles, 0);
        std::vector<uint32_t> tile_pad(num_tiles, 0);
        LptAssignment lpt;
        uint32_t P_kept = 0;
        uint32_t P_aligned = 0;
        uint32_t max_n = 0;
        clk::time_point t_d2h = t_cnt;
        clk::time_point t_bin1 = t_cnt;

        std::optional<BinLayoutResult> layout_verify_ref;
        if (layout_verify) {
            if (!device_layout) {
                GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
                distributed::Finish(*ctx->cq);
            }
            std::vector<uint32_t> hist_ref(static_cast<std::size_t>(num_cores) * stride);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, hist_ref, ctx->buf_bin2d, true);
            t_d2h = clk::now();
            layout_verify_ref =
                host_bin_layout_from_hist(hist_ref, num_cores, num_tiles, stride, tile_bucket);
            if (layout_verify_ref->status != 0) {
                std::cerr << "[gsplat_tt::sort] layout verify: host ref status="
                          << layout_verify_ref->status << "\n";
            }
            enqueue_bin_layout_kernel(
                ctx,
                num_cores,
                num_tiles,
                stride,
                tile_bucket,
                static_cast<uint32_t>(ctx->buf_tile_ranges->address()),
                l1_record_early,
                bucket_fit);
            GSPLAT_HOST_ZONE("host_finish_sort_bin_layout");
            distributed::Finish(*ctx->cq);
            if (layout_verify_ref->status == 0) {
                const BinLayoutResult& href = *layout_verify_ref;
                uint32_t dev_P_kept = 0, dev_P_aligned = 0, dev_max_n = 0, dev_status = 0;
                read_bin_layout_ctrl(ctx, dev_P_kept, dev_P_aligned, dev_max_n, dev_status);
                std::vector<uint32_t> hist_dev(hist_ref.size());
                std::vector<uint32_t> tmeta_d(tmeta_pad, 0);
                std::vector<uint32_t> tids_d(tile_ids_pad, 0);
                std::vector<uint32_t> lptm(meta_pad, 0);
                std::vector<uint32_t> cnts(counts_pad, 0);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, hist_dev, ctx->buf_bin2d, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, tmeta_d, ctx->buf_tmeta, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, tids_d, ctx->buf_tile_ids, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, lptm, ctx->buf_lpt_meta, true);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, cnts, ctx->buf_tile_counts, true);
                std::size_t mism_hist = 0, mism_cnt = 0, mism_tids = 0, mism_lpt = 0, mism_tmeta = 0;
                for (std::size_t i = 0; i < hist_dev.size(); i++) {
                    if (hist_dev[i] != href.hist[i]) {
                        mism_hist++;
                        if (mism_hist == 1) {
                            std::fprintf(stderr,
                                "[SORT_LAYOUT_VERIFY] first hist mism i=%zu dev=%u ref=%u\n",
                                i, hist_dev[i], href.hist[i]);
                        }
                    }
                }
                for (uint32_t t = 0; t < num_tiles; t++) {
                    if (cnts[t] != static_cast<uint32_t>(href.counts[t])) mism_cnt++;
                }
                const uint32_t nflat =
                    static_cast<uint32_t>(href.lpt.flat_tile_ids.size());
                for (uint32_t i = 0; i < nflat; i++) {
                    if (tids_d[i] != href.lpt.flat_tile_ids[i]) mism_tids++;
                }
                for (uint32_t c = 0; c < num_cores; c++) {
                    if (lptm[c * 2 + 0] != href.lpt.per_core_offset[c] ||
                        lptm[c * 2 + 1] != href.lpt.per_core_count[c])
                        mism_lpt++;
                }
                for (uint32_t t = 0; t < num_tiles; t++) {
                    if (tmeta_d[t * 2 + 0] != href.pstart_page[t] ||
                        tmeta_d[t * 2 + 1] != href.tile_pad[t])
                        mism_tmeta++;
                }
                const std::size_t mism =
                    mism_hist + mism_cnt + mism_tids + mism_lpt + mism_tmeta;
                std::fprintf(stderr,
                    "[SORT_LAYOUT_VERIFY] P_kept dev=%u ref=%u P_aligned dev=%u ref=%u "
                    "status=%u hist=%zu cnt=%zu tids=%zu lpt=%zu tmeta=%zu %s\n",
                    dev_P_kept, href.P_kept, dev_P_aligned, href.P_aligned, dev_status,
                    mism_hist, mism_cnt, mism_tids, mism_lpt, mism_tmeta,
                    mism == 0 ? "IDENTICAL" : "FAIL");
                if (mism != 0 || dev_P_kept != href.P_kept || dev_P_aligned != href.P_aligned) {
                    return fail();
                }
            }
            t_bin1 = clk::now();
        }

        if (device_layout) {
            if (!layout_verify) {
                enqueue_bin_layout_kernel(
                    ctx,
                    num_cores,
                    num_tiles,
                    stride,
                    tile_bucket,
                    static_cast<uint32_t>(ctx->buf_tile_ranges->address()),
                    l1_record_early,
                    bucket_fit);
                GSPLAT_HOST_ZONE("host_finish_sort_bin_cnt");
                distributed::Finish(*ctx->cq);
                t_bin1 = clk::now();
            }
            uint32_t layout_status = 0;
            if (!read_bin_layout_ctrl(ctx, P_kept, P_aligned, max_n, layout_status)) {
                std::cerr << "[gsplat_tt::sort] device layout status=" << layout_status
                          << "; tile exceeds device sort capacity — hard fail "
                             "(render_clean is single-path TT, no host fallback)\n";
                return fail();
            }
            counts.assign(num_tiles, 0);
            std::vector<uint32_t> cnts_u(counts_pad, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, cnts_u, ctx->buf_tile_counts, true);
            for (uint32_t t = 0; t < num_tiles; t++)
                counts[t] = static_cast<int64_t>(cnts_u[t]);
            std::vector<uint32_t> ranges_u(
                static_cast<std::size_t>(ctx->cap_ranges_bytes / 4), 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, ranges_u, ctx->buf_tile_ranges, true);
            for (uint32_t t = 0; t < num_tiles; t++) {
                starts[t] = static_cast<int64_t>(ranges_u[t * 2 + 0]);
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                    static_cast<int64_t>(ranges_u[t * 2 + 1]);
            }
            std::vector<uint32_t> tmeta_d(tmeta_pad, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tmeta_d, ctx->buf_tmeta, true);
            for (uint32_t t = 0; t < num_tiles; t++) {
                pstart_page[t] = tmeta_d[t * 2 + 0];
                tile_pad[t] = tmeta_d[t * 2 + 1];
                pstart_elem[t] = pstart_page[t] * ELEMS_PER_PAGE;
            }
            const uint32_t cap_meta = static_cast<uint32_t>(ctx->cap_lpt_meta_bytes / 4);
            std::vector<uint32_t> lptm(cap_meta, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, lptm, ctx->buf_lpt_meta, true);
            lpt.per_core_offset.assign(num_cores, 0);
            lpt.per_core_count.assign(num_cores, 0);
            for (uint32_t c = 0; c < num_cores; c++) {
                lpt.per_core_offset[c] = lptm[c * 2 + 0];
                lpt.per_core_count[c] = lptm[c * 2 + 1];
            }
            const uint32_t cap_tids = static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
            std::vector<uint32_t> tids_d(cap_tids, 0);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, tids_d, ctx->buf_tile_ids, true);
            uint32_t flat_n = 0;
            for (uint32_t c = 0; c < num_cores; c++) flat_n += lpt.per_core_count[c];
            lpt.flat_tile_ids.assign(tids_d.begin(), tids_d.begin() + flat_n);
            device_state::register_buffer("sort_lpt_tile_ids", ctx->buf_tile_ids);
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            T.upload_ms = 0.0;
        } else if (layout_verify_ref) {
            BinLayoutResult bl = std::move(*layout_verify_ref);
            if (bl.status == 1 || bl.status == 2) {
                return fail();
            }
            counts = std::move(bl.counts);
            starts = std::move(bl.starts);
            pstart_page = std::move(bl.pstart_page);
            pstart_elem = std::move(bl.pstart_elem);
            tile_pad = std::move(bl.tile_pad);
            lpt = std::move(bl.lpt);
            P_kept = bl.P_kept;
            P_aligned = bl.P_aligned;
            max_n = bl.max_pad_n;
            for (uint32_t t = 0; t < num_tiles; t++) {
                if (counts[t] > 0) {
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                        starts[t] + counts[t];
                }
            }
            t_bin1 = clk::now();
            if (tile_bucket) {
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bin2d_rec, bl.histrec, false);
                bl.bucket_meta.resize(
                    static_cast<std::size_t>(ctx->cap_bucket_meta_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bucket_meta, bl.bucket_meta, false);
            }
            if (l1_record_early && !bl.histrec_l1.empty()) {
                bl.histrec_l1.resize(
                    static_cast<std::size_t>(ctx->cap_l1_rec_base_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_l1_rec_base, bl.histrec_l1, false);
                l1_base_addr = static_cast<uint32_t>(ctx->buf_l1_rec_base->address());
            }
            std::vector<uint32_t> tmeta(tmeta_pad, 0);
            for (uint32_t t = 0; t < num_tiles; t++) {
                tmeta[t * 2 + 0] = pstart_page[t];
                tmeta[t * 2 + 1] = tile_pad[t];
            }
            const uint32_t cap_tile_ids_elems =
                static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
            std::vector<uint32_t> tile_ids_flat(cap_tile_ids_elems, 0);
            std::copy(
                lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            const auto t_up0 = clk::now();
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d, bl.hist, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
            distributed::EnqueueWriteMeshBuffer(
                *ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
            if (!sort_stage_defer_finish()) {
                GSPLAT_HOST_ZONE("host_finish_sort_upload");
                distributed::Finish(*ctx->cq);
            }
            T.upload_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_up0).count();
        } else {
            // Host bridge: D2H histogram + page layout + LPT + H2D metadata. This
            // whole block runs with the device idle between the count and emit
            // kernels, so the histogram and the layout tables are kept across
            // views (no per-view ~2 MB of allocation + page faults).
            static std::vector<uint32_t> hist;
            static BinLayoutResult bl;
            hist.resize(static_cast<std::size_t>(num_cores) * stride);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, hist, ctx->buf_bin_hist, true);
            t_d2h = clk::now();
            T.bin_hist_d2h_ms =
                std::chrono::duration<double, std::milli>(t_d2h - t_cnt).count();

            host_bin_layout_into(hist, num_cores, num_tiles, stride, tile_bucket,
                                 l1_record_early, bucket_fit, /*dense_recbase=*/false, bl);
            if (bl.status == 1) {
                std::cerr << "[gsplat_tt::sort] per-core padded run > BIN_LOCAL_MAX; "
                             "exceeds device sort capacity — hard fail "
                             "(render_clean is single-path TT, no host fallback)\n";
                return fail();
            }
            if (bl.status == 2) {
                std::cerr << "[gsplat_tt::sort] padded tile exceeds MAX_TILE_ENTRIES; "
                             "exceeds device sort capacity — hard fail "
                             "(render_clean is single-path TT, no host fallback)\n";
                return fail();
            }
            counts = bl.counts;
            starts = bl.starts;
            pstart_page = bl.pstart_page;
            pstart_elem = bl.pstart_elem;
            tile_pad = bl.tile_pad;
            lpt = bl.lpt;
            P_kept = bl.P_kept;
            P_aligned = bl.P_aligned;
            max_n = bl.max_pad_n;
            for (uint32_t t = 0; t < num_tiles; t++) {
                if (counts[t] > 0) {
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[t];
                    result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                        starts[t] + counts[t];
                }
            }
            t_bin1 = clk::now();
            T.bin_layout_ms =
                std::chrono::duration<double, std::milli>(t_bin1 - t_d2h).count();

            if (tile_bucket) {
                // buf_bin2d_rec (the dense per-(core,tile) record base) only fed the
                // retired tile_recs scatter; its per-view 450 KB build + H2D are
                // skipped (dense_recbase=false above).
                bl.bucket_meta.resize(
                    static_cast<std::size_t>(ctx->cap_bucket_meta_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_bucket_meta, bl.bucket_meta, false);
            }
            if (l1_record_early && !bl.histrec_l1.empty()) {
                bl.histrec_l1.resize(
                    static_cast<std::size_t>(ctx->cap_l1_rec_base_bytes / 4), 0u);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_l1_rec_base, bl.histrec_l1, false);
                l1_base_addr = static_cast<uint32_t>(ctx->buf_l1_rec_base->address());
            }
            // iter-138: allocate + upload the compact overflow region and its base
            // rows. Region is sized to the actual in-cap overflow record count
            // (even-padded per tile); base rows are sentinel-filled for tiles that
            // are not pre-packed. Always allocated (>=1 page) so the kernel reads
            // valid addresses even when this view has no in-cap overflow tile.
            if (l1_record_early && !bl.histrec_overflow.empty()) {
                const std::size_t ov_region_bytes = std::max<std::size_t>(
                    render_config::kRecPageBytes,
                    static_cast<std::size_t>(bl.ov_total_slots) * 32u);
                if (!ctx->buf_l1_ov || ctx->cap_l1_ov_bytes < ov_region_bytes) {
                    ctx->buf_l1_ov =
                        make_dram_paged(ctx->mesh_device.get(), ov_region_bytes,
                                        render_config::kRecPageBytes);
                    ctx->cap_l1_ov_bytes = ov_region_bytes;
                    device_state::register_buffer("sort_l1_overflow", ctx->buf_l1_ov);
                }
                l1_ov_addr = static_cast<uint32_t>(ctx->buf_l1_ov->address());

                const std::size_t ov_base_bytes =
                    static_cast<std::size_t>(num_cores) * stride * 4u;
                if (!ctx->buf_l1_ov_base || ctx->cap_l1_ov_base_bytes < ov_base_bytes) {
                    ctx->buf_l1_ov_base =
                        make_dram(ctx->mesh_device.get(), ov_base_bytes);
                    ctx->cap_l1_ov_base_bytes = ov_base_bytes;
                }
                bl.histrec_overflow.resize(
                    static_cast<std::size_t>(ctx->cap_l1_ov_base_bytes / 4), 0xFFFFFFFFu);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_l1_ov_base, bl.histrec_overflow, false);
                l1_ov_base_addr = static_cast<uint32_t>(ctx->buf_l1_ov_base->address());

                const uint32_t tov_pad = round_up(num_tiles, ELEMS_PER_PAGE);
                const std::size_t tov_bytes = static_cast<std::size_t>(tov_pad) * 4u;
                if (!ctx->buf_tile_ov_base ||
                    ctx->cap_tile_ov_base_bytes < tov_bytes) {
                    ctx->buf_tile_ov_base =
                        make_dram(ctx->mesh_device.get(), tov_bytes);
                    ctx->cap_tile_ov_base_bytes = tov_bytes;
                    device_state::register_buffer(
                        "sort_tile_ov_base", ctx->buf_tile_ov_base);
                }
                bl.tile_ov_base.resize(
                    static_cast<std::size_t>(ctx->cap_tile_ov_base_bytes / 4), 0xFFFFFFFFu);
                distributed::EnqueueWriteMeshBuffer(
                    *ctx->cq, ctx->buf_tile_ov_base, bl.tile_ov_base, false);
            }
            std::vector<uint32_t> tmeta(tmeta_pad, 0);
            for (uint32_t t = 0; t < num_tiles; t++) {
                tmeta[t * 2 + 0] = pstart_page[t];
                tmeta[t * 2 + 1] = tile_pad[t];
            }
            const uint32_t cap_tile_ids_elems =
                static_cast<uint32_t>(ctx->cap_tile_ids_bytes / 4);
            std::vector<uint32_t> tile_ids_flat(cap_tile_ids_elems, 0);
            std::copy(
                lpt.flat_tile_ids.begin(), lpt.flat_tile_ids.end(), tile_ids_flat.begin());
            publish_sort_downstream_metadata(ctx, lpt, counts, num_tiles, num_cores);
            const auto t_up0 = clk::now();
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_bin2d, bl.hist, false);
            distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_tmeta, tmeta, false);
            distributed::EnqueueWriteMeshBuffer(
                *ctx->cq, ctx->buf_tile_ids, tile_ids_flat, false);
            if (!sort_stage_defer_finish()) {
                GSPLAT_HOST_ZONE("host_finish_sort_upload");
                distributed::Finish(*ctx->cq);
            }
            T.upload_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_up0).count();
        }
        T.bin_ms = std::chrono::duration<double, std::milli>(t_bin1 - t_bin0).count();

        // ── Allocate aligned keys/ids/out ───────────────────────────────
        const std::size_t aligned_bytes = static_cast<std::size_t>(P_aligned) * 4;
        if (!ctx->buf_keys || ctx->cap_aligned_bytes < aligned_bytes) {
            ctx->buf_keys = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_ids  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->buf_out  = make_dram(ctx->mesh_device.get(), aligned_bytes);
            ctx->cap_aligned_bytes = aligned_bytes;
        }
        if (tile_bucket) {
            const std::size_t recs_bytes =
                static_cast<std::size_t>(std::max<uint32_t>(P_kept, 1)) * PAGE_BYTES;
            if (!ctx->buf_tile_recs || ctx->cap_tile_recs_bytes < recs_bytes) {
                ctx->buf_tile_recs = make_dram(ctx->mesh_device.get(), recs_bytes);
                ctx->cap_tile_recs_bytes = recs_bytes;
                device_state::register_buffer("sort_tile_recs", ctx->buf_tile_recs);
            }
            tile_recs_addr = static_cast<uint32_t>(ctx->buf_tile_recs->address());
        }

        // ── Pass B: device scatter into aligned (key,id) layout ─────────
        const auto t_sc0 = clk::now();
        launch_bin(1, true);
        const auto t_sc1 = clk::now();
        T.bin_emit_ms = std::chrono::duration<double, std::milli>(t_sc1 - t_sc0).count();
        T.bin_ms += T.bin_emit_ms;
        T.upload_ms = std::chrono::duration<double, std::milli>(t_sc0 - t_bin1).count();

        // ── ROUTE C: SFPU microblock cull over the dense bucket ─────────
        // Records are now scattered (launch_bin(1) Finished) and bucket_meta /
        // tile_ids are uploaded. Run the 3-kernel cull program over the SAME
        // LPT tile assignment, baking each candidate's 32-bit keep mask into
        // record word 10. This is a SORT-STAGE write -> the downstream L1 blend
        // reads it back spin-free (GSPLAT_TT_BUCKET_MASK path). Fully overlaps
        // no DRAM round-trip beyond the in-place record RMW.
        if (tile_bucket && bucket_mask_enabled() && ctx->cull_built && P_kept > 0) {
            const auto t_cl0 = clk::now();
            if (!ctx->buf_box_ox) {
                ctx->buf_box_ox = make_dram_paged(ctx->mesh_device.get(), RAMP_TILE_BYTES, RAMP_TILE_BYTES);
                ctx->buf_box_oy = make_dram_paged(ctx->mesh_device.get(), RAMP_TILE_BYTES, RAMP_TILE_BYTES);
                ctx->box_ramp_uploaded = false;
            }
            if (!ctx->box_ramp_uploaded) {
                std::vector<uint32_t> bx = make_box_ramp(/*is_x=*/true);
                std::vector<uint32_t> by = make_box_ramp(/*is_x=*/false);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_box_ox, bx, false);
                distributed::EnqueueWriteMeshBuffer(*ctx->cq, ctx->buf_box_oy, by, false);
                ctx->box_ramp_uploaded = true;
            }
            float floor = 1.0f / 16384.0f;
            bool cull_disabled = false;
            device_state::get_bucket_cull_params(&floor, &cull_disabled);
            uint32_t floor_bits;
            std::memcpy(&floor_bits, &floor, 4);
            const uint32_t recs_addr = tile_recs_addr;
            const uint32_t meta_addr = static_cast<uint32_t>(ctx->buf_bucket_meta->address());
            const uint32_t box_ox_addr = static_cast<uint32_t>(ctx->buf_box_ox->address());
            const uint32_t box_oy_addr = static_cast<uint32_t>(ctx->buf_box_oy->address());
            const uint32_t tids_addr = static_cast<uint32_t>(ctx->buf_tile_ids->address());
            Program& cprog = ctx->wl_cull.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                const uint32_t start = lpt.per_core_offset[c];
                const uint32_t count = lpt.per_core_count[c];
                SetRuntimeArgs(cprog, ctx->kc_reader, core, {
                    recs_addr, meta_addr, box_ox_addr, box_oy_addr,
                    tids_addr, start, count, static_cast<uint32_t>(tiles_x), floor_bits,
                });
                SetRuntimeArgs(cprog, ctx->kc_compute, core, {
                    count, floor_bits, cull_disabled ? 1u : 0u,
                });
                SetRuntimeArgs(cprog, ctx->kc_writer, core, {
                    recs_addr, meta_addr, tids_addr, start, count,
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_cull, false);
            distributed::Finish(*ctx->cq);
        }

        // ── Device radix kernel (per-tile stable depth sort) ────────────
        const auto t_k0 = clk::now();
        Program& prog = ctx->workload.get_programs().begin()->second;
        for (uint32_t c = 0; c < num_cores; c++) {
            CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
            const uint32_t start = lpt.per_core_offset[c];
            const uint32_t count = lpt.per_core_count[c];
            // NCRISC sorts the first k tiles of the core's slice, BRISC the
            // rest; tiles are independent (exclusive pages), so the output is
            // byte-identical for any k.
            const uint32_t k = (sort_radix_movers() == 2)
                ? radix_split_point(lpt.flat_tile_ids, start, count, counts)
                : count;
            auto args = [&](uint32_t s, uint32_t n) {
                return std::vector<uint32_t>{
                    static_cast<uint32_t>(ctx->buf_keys->address()),
                    static_cast<uint32_t>(ctx->buf_ids->address()),
                    static_cast<uint32_t>(ctx->buf_out->address()),
                    static_cast<uint32_t>(ctx->buf_tile_ids->address()),
                    static_cast<uint32_t>(ctx->buf_tmeta->address()),
                    s,
                    n,
                };
            };
            SetRuntimeArgs(prog, ctx->kernel, core, args(start, k));
            SetRuntimeArgs(prog, ctx->kernel_m0, core, args(start + k, count - k));
        }
        distributed::EnqueueMeshWorkload(*ctx->cq, ctx->workload, false);
        if (!sort_stage_defer_finish()) {
            GSPLAT_HOST_ZONE("host_finish_sort_radix");
            distributed::Finish(*ctx->cq);
        }
        const auto t_k1 = clk::now();
        T.kernel_ms = std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

        const bool dev_publish = sort_device_publish_enabled();
        std::vector<uint32_t> out_aligned;  // only populated for host-id readback / BIN_DEBUG
        SubchunkLayout sc_layout{};
        MatWorkAssignment mat_work;  // iter 130: subchunk-balanced materialize work
        bool subchunk_materialize = false;

        if (dev_publish) {
            // ── Device compact+publish (skip D2H buf_out + host Pass4) ────
            // sort_publish.cpp copies WHOLE 16-elem pages from the page-aligned
            // radix output into sort_sorted_ids at dst_base = range_start/16, which
            // is only bit-correct when each tile's dst slice is page-aligned. So we
            // publish a PADDED layout: every tile starts on a 16-elem boundary. The
            // resident sort_tile_ranges carries the padded [start, start+count) so
            // all downstream resident readers (cull/blend) index the right slice.
            const auto t_pub0 = clk::now();
            std::vector<int64_t> padded_ranges(result.tile_ranges.size(), 0);
            uint32_t padded_cursor = 0;
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t s = static_cast<uint32_t>(result.tile_ranges[2 * t]);
                const uint32_t e = static_cast<uint32_t>(result.tile_ranges[2 * t + 1]);
                const uint32_t cnt = (e > s) ? (e - s) : 0u;
                padded_ranges[2 * t] = static_cast<int64_t>(padded_cursor);
                padded_ranges[2 * t + 1] = static_cast<int64_t>(padded_cursor + cnt);
                padded_cursor += round_up(cnt, ELEMS_PER_PAGE);
            }
            ensure_resident_sorted_buffer(ctx, padded_cursor);
            upload_resident_tile_ranges(ctx, padded_ranges);
            Program& pub_prog = ctx->wl_publish.get_programs().begin()->second;
            for (uint32_t c = 0; c < num_cores; c++) {
                CoreCoord core{c % ctx->grid.x, c / ctx->grid.x};
                SetRuntimeArgs(pub_prog, ctx->kpublish, core, {
                    static_cast<uint32_t>(ctx->buf_out->address()),
                    static_cast<uint32_t>(ctx->buf_sorted_ids->address()),
                    static_cast<uint32_t>(ctx->buf_tile_ranges->address()),
                    static_cast<uint32_t>(ctx->buf_tile_ids->address()),
                    static_cast<uint32_t>(ctx->buf_tmeta->address()),
                    lpt.per_core_offset[c],
                    lpt.per_core_count[c],
                });
            }
            distributed::EnqueueMeshWorkload(*ctx->cq, ctx->wl_publish, false);
            sc_layout = build_subchunk_layout(counts, num_tiles, bucket_fit);
            log_subchunk_layout_stats(sc_layout);
            mat_work = build_mat_worklist(counts, num_tiles, num_cores, bucket_fit,
                                          sort_mat_movers(), kMatMover0Cap);
            if (!prepare_subchunk_buffers(ctx, sc_layout, num_tiles)) {
                std::cerr << "[gsplat_tt::sort] subchunk buffer setup failed\n";
                return fail();
            }
            upload_subchunk_directory(ctx, sc_layout);
            const auto t_pubw0 = clk::now();
            T.publish_host_ms =
                std::chrono::duration<double, std::milli>(t_pubw0 - t_pub0).count();
            if (sort_blend_pipe_enabled()) {
                // C1: drain radix + publish before enqueueing mat on the piped CQ
                // (not between mat and blend).
                GSPLAT_HOST_ZONE("host_finish_sort_publish");
                distributed::Finish(*ctx->cq);
            }
            T.publish_wait_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_pubw0).count();
            if (mat_work.max_items_per_core > 1024u) {
                std::cerr << "[gsplat_tt::sort] materialize work items/core "
                          << mat_work.max_items_per_core << " > MAX_WORK=1024\n";
                return fail();
            }
            subchunk_materialize = true;
            T.publish_ms =
                std::chrono::duration<double, std::milli>(clk::now() - t_pub0).count();
            if (sort_blend_pipe_enabled()) {
                device_state::mark_sort_publish_pending();
            } else {
                const auto t_mat0 = clk::now();
                if (!launch_subchunk_materialize(
                        ctx, mat_work, num_cores,
                        static_cast<uint32_t>(tiles_x), bucket_fit)) {
                    std::cerr << "[gsplat_tt::sort] subchunk materialize launch failed\n";
                    return fail();
                }
                GSPLAT_HOST_ZONE("host_finish_sort_materialize");
                distributed::Finish(*ctx->cq);
                T.materialize_ms =
                    std::chrono::duration<double, std::milli>(clk::now() - t_mat0).count();
            }
            // Resident blend reads sort_sorted_ids over NoC — skip the large ids
            // D2H + host Pass4 unless the caller needs the dense host vector.
            const bool need_host_ids = need_host_sorted_ids ||
                                       !resident_blend_chain_enabled();
            if (need_host_ids) {
                finish_sort_cq_if_needed(ctx);
                const uint32_t cap_sorted_elems =
                    static_cast<uint32_t>(ctx->cap_sorted_bytes / 4);
                const auto t_d0 = clk::now();
                std::vector<uint32_t> sids(cap_sorted_elems);
                distributed::EnqueueReadMeshBuffer(*ctx->cq, sids, ctx->buf_sorted_ids, true);
                T.d2h_ms = std::chrono::duration<double, std::milli>(clk::now() - t_d0).count();
                result.sorted_gaussian_ids.assign(P_kept, 0);
                for (uint32_t t = 0; t < num_tiles; t++) {
                    const uint32_t ds = static_cast<uint32_t>(result.tile_ranges[2 * t]);
                    const uint32_t de = static_cast<uint32_t>(result.tile_ranges[2 * t + 1]);
                    const uint32_t ps = static_cast<uint32_t>(padded_ranges[2 * t]);
                    for (uint32_t k = 0; k + ds < de && (ds + k) < P_kept; k++) {
                        result.sorted_gaussian_ids[ds + k] =
                            static_cast<int64_t>(sids[ps + k]);
                    }
                }
            } else {
                T.d2h_ms = 0.0;
                // tile_ranges kept for stats/timing only; blend uses resident DRAM.
            }
        } else {
            // ── D2H aligned sorted ids + Pass4 compact -> contiguous ────
            const uint32_t cap_aligned_elems =
                static_cast<uint32_t>(ctx->cap_aligned_bytes / 4);
            const auto t_d0 = clk::now();
            out_aligned.resize(cap_aligned_elems);
            distributed::EnqueueReadMeshBuffer(*ctx->cq, out_aligned, ctx->buf_out, true);
            const auto t_d1 = clk::now();
            T.d2h_ms = std::chrono::duration<double, std::milli>(t_d1 - t_d0).count();

            const auto t_c0 = clk::now();
            result.sorted_gaussian_ids.resize(P_kept);
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t n = static_cast<uint32_t>(counts[t]);
                if (n == 0) continue;
                const uint32_t src = pstart_elem[t];
                const std::size_t dst = static_cast<std::size_t>(starts[t]);
                for (uint32_t k = 0; k < n; k++)
                    result.sorted_gaussian_ids[dst + k] =
                        static_cast<int64_t>(out_aligned[src + k]);
            }
            const auto t_c1 = clk::now();
            T.compact_ms = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
        }

        if (!dev_publish)
            publish_resident(ctx, result.sorted_gaussian_ids, result.tile_ranges, &T.publish_ms);

        T.total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total0_rp).count();
        std::fprintf(stderr,
            "[SORT] stage=RP P=%u P_kept=%u num_tiles=%u max_tile_n=%u bin=%.2f "
            "up=%.2f kernel=%.2f d2h=%.2f compact=%.2f publish=%.2f mat=%.2f total=%.2fms"
            " | pread=%.2f count=%.2f hist_d2h=%.2f layout=%.2f emit=%.2f"
            " pub_host=%.2f pub_wait=%.2f\n",
            P_full, P_kept, num_tiles, max_n, T.bin_ms, T.upload_ms, T.kernel_ms,
            T.d2h_ms, T.compact_ms, T.publish_ms, T.materialize_ms, T.total_ms,
            T.pread_ms, T.bin_count_ms, T.bin_hist_d2h_ms, T.bin_layout_ms,
            T.bin_emit_ms, T.publish_host_ms, T.publish_wait_ms);
        if (device_ok) *device_ok = true;
        // Step C1: materialize before blend on the piped CQ (no Finish here —
        // sort_publish_pending: one drain at blend readback; iter-58/83).
        if (subchunk_materialize && sort_blend_pipe_enabled()) {
            // GSPLAT_TT_SPLIT_BLEND=1: drain mat here so stage `mat` is its device
            // window (stage_timers.h); sort_mat then stays 0.
            const bool split = stagetimers::split_blend();
            stagetimers::Span mat_span(split ? stagetimers::acc().mat : T.materialize_ms);
            if (!launch_subchunk_materialize(
                    ctx, mat_work, num_cores,
                    static_cast<uint32_t>(tiles_x), bucket_fit, sort_blend)) {
                std::cerr << "[gsplat_tt::sort] subchunk materialize launch failed\n";
                return fail();
            }
            if (split) {
                GSPLAT_HOST_ZONE("host_finish_mat_split");
                distributed::Finish(*ctx->cq);
            }
            mat_span.stop();
            std::fprintf(
                stderr, "[SUBCHUNK] materialize_ms=%.2f (piped pre-blend)\n",
                T.materialize_ms);
        }
        maybe_run_sort_blend_continuation(sort_blend, tiles_x, num_tiles);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[gsplat_tt::sort] resident-pairs path failed: " << e.what() << "\n";
        finish_sort_cq_if_needed(ctx);
        return fail();
    }
}

}  // namespace

bool sort_device_ready() { return ensure_context() != nullptr; }

bool sort_matcull_fused() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_FUSE_MATCULL");
        return !(e != nullptr && e[0] == '0');
    }();
    return v;
}

void sort_device_shutdown() {
    auto& slot = context_slot();
    if (slot) {
        // Intentionally leak the context: MeshWorkload::~MeshWorkload can
        // SIGSEGV in tt_metal if ProgramImpl runs after MeshDevice::close().
        (void)slot.release();
    }
}

gsplat_cpu::SortResult sort_and_bin_tt(
    const int64_t* gaussian_ids,
    const int64_t* tile_ids,
    const float* depths,
    const std::size_t P,
    const std::size_t M,
    const int tiles_x,
    const int tiles_y,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    SortCallTimings* timings,
    const bool need_host_sorted_ids,
    SortBlendContinuation* sort_blend) {
    auto set_fail = [&]() {
        if (device_ok) *device_ok = false;
        return gsplat_cpu::SortResult{};
    };

    // render_clean is single-path: the resident-pairs device binning stage reads
    // the resident full-P (gid,tid) pairs + keep mask that tile_assign left in
    // device_state, plus the resident proj_m_depth, and bins them on-device into
    // the page-aligned per-tile (key,id) layout the radix kernel consumes. The
    // host depth/id/tile_id arguments are unused (kept for ABI). The legacy
    // host-sort (S0) and host-binning (S1) fallbacks were removed; unsupported
    // input hard-fails via set_fail() (render.cpp turns that into a throw).
    (void)gaussian_ids;
    (void)tile_ids;
    (void)depths;
    (void)P;

    auto* ctx = ensure_context();
    if (ctx == nullptr) return set_fail();

    SortCallTimings tlocal;
    auto& T = (timings ? *timings : tlocal);
    T.stage = 1;

    const uint32_t num_tiles =
        static_cast<uint32_t>(tiles_x) * static_cast<uint32_t>(tiles_y);

    return sort_resident_pairs(
        ctx, num_tiles, tiles_x, tiles_y, M, need_host_sorted_ids, pool,
        device_ok, T, sort_blend);
}

}  // namespace gsplat_tt
