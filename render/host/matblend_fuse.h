// Task #280/#285: fused mat+blend program (GSPLAT_TT_MATBLEND_FUSE, default on
// since #293; =0 turns it off). Design: docs/matblend-ready-t273/README.md step 2.
//
// Only for one-launch frames with the sort->blend continuation. Instead of
// enqueueing the materialize program, launch_subchunk_materialize (sort_device)
// parks its per-core runtime args here; the resident blend (blend_device) then
// enqueues ONE program whose kernels run the mat body and then the blend body
// on each RISC (kernels/dataflow/matblend_{ncrisc,brisc}.cpp,
// kernels/compute/matblend_compute.cpp). The blend reader waits per
// (tile, subchunk) on a DRAM ready flag the mat movers write after the
// subchunk's payload, so a core starts blending once its own mat items are
// done instead of at the program barrier.
//
// Layout in the fused program: mat keeps its CB ids, runtime args and
// compile-time args; blend CB ids are +kBlendCbBase, blend runtime args start at
// kDmRtaBase (DM) / kCpRtaBase (compute), blend compile-time args follow mat's.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>

namespace gsplat_tt::matblend_fuse {

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_MATBLEND_FUSE");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

constexpr uint32_t kBlendCbBase = 32;  // blend CB ids (BH has 64)
constexpr uint32_t kDmRtaBase = 32;    // blend reader/writer args start (mat uses <= 20)
constexpr uint32_t kCpRtaBase = 4;     // blend compute args start (mat cull uses 3)
// Ready-flag pages per tile: the blend reader polls page tile*8+sc
// (reader_alpha_blend_mb_devcull.cpp), so a tile has at most 8 subchunks.
constexpr uint32_t kReadyPagesPerTile = 8;

// Task #289: blend CBs (local ids, add kBlendCbBase) whose storage add_mat_part shares
// with a mat CB; build_program_and_workload_mb skips them when fused.
constexpr uint32_t kAliasMbCounts = 3;   // on mat CB 5 (CB_BSORT)
constexpr uint32_t kAliasScrAttr = 5;    // on mat CB 2 (REC_BATCH ring)
constexpr uint32_t kAliasImgU8 = 8;      // on mat CB 20 (BRISC CB_BUCKET)
constexpr uint32_t kAliasOut = 16;       // on mat CB 6 (CB_SLAB)
constexpr uint32_t kOutPageBytes = 32u * 32u * 2u;  // bf16 tile
constexpr uint32_t kImgU8Bytes = 32u * 32u * 3u;    // 32 rows x 96 B RGB

// Mat half of the fused program: CBs (created by add_mat_part), kernel defines
// for mover 1 (NCRISC) / mover 0 (BRISC) and the mat compile-time args (the
// blend accessors are appended after them).
struct MatPart {
    std::map<std::string, std::string> defines_n;
    std::map<std::string, std::string> defines_m0;
    std::vector<uint32_t> ct;
};

// sort_device.cpp: creates the mat CBs in `program` on `cores` (CB 4 doubles as
// blend CB_BUCKET_BULK at 4 + kBlendCbBase + 8) plus the 64 B flag-source CBs.
MatPart add_mat_part(tt::tt_metal::Program& program, const tt::tt_metal::CoreRangeSet& cores);

struct CoreArgs {
    std::vector<uint32_t> n;    // mat mover 1 (NCRISC), args 0-17
    std::vector<uint32_t> m0;   // mat mover 0 (BRISC), args 0-17
    std::vector<uint32_t> cp;   // mat cull compute, args 0-2
};

inline uint32_t core_key(uint32_t x, uint32_t y) { return (x << 16) | y; }

// Mat launch parked for the fused program. `fallback` enqueues the plain mat
// program with the parked args; run_fallback() runs it if nobody consumed the
// launch (blend not reached / a core missing), so mat always runs before blend.
struct Pending {
    using Cores = std::map<uint32_t, CoreArgs>;
    bool active = false;
    Cores cores;  // by core_key(logical x, y)
    uint32_t flags_addr = 0;
    uint32_t epoch = 0;
    std::function<void(const Cores&)> fallback;

    void clear() {
        active = false;
        cores.clear();
        fallback = nullptr;
    }
    void run_fallback() {
        if (!active) return;
        auto f = std::move(fallback);
        Cores c = std::move(cores);
        clear();
        if (f) f(c);
    }
};

inline Pending& pending() {
    static Pending p;
    return p;
}

}  // namespace gsplat_tt::matblend_fuse
