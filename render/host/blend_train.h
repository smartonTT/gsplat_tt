// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Task #708: forward training flag (host env GSPLAT_TT_BLEND_TRAIN=1, default
// off). The blend then runs the training forward (BLEND_TRAIN in
// alpha_blend_compute_mb.cpp: fp32-accurate exp, gsplat's exact alpha/T
// rules, no T early-out, SCHED 0 bodies, unfused program) and also writes the
// fp32 R, G, B, T accumulators per screen tile to a train buffer that stays
// alive until the next forward, for T1 (device blend backward). Layout: one
// 16 KB page per screen tile, page ty * image_tiles_x + tx, 4 fp32 32x32 tiles
// (R, G, B, T) in the blend's microblock layout (img_pack_u8.h permutation).
// T < 0 marks a pixel whose blend stopped on the 1e-4 rule; |T| is gsplat's
// final T. GSPLAT_TT_BLEND_TRAIN_DUMP=<path> also reads it back after each
// forward and writes it to <path> (raw pages, no header).

#include <cstdint>
#include <cstdlib>

namespace gsplat_tt::blend_train {

inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_BLEND_TRAIN");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

inline const char* dump_path() {
    const char* e = std::getenv("GSPLAT_TT_BLEND_TRAIN_DUMP");
    return (e != nullptr && e[0] != '\0') ? e : nullptr;
}

constexpr uint32_t kCb = 17;                          // CB_TRAIN_OUT (fp32, 4 tiles)
constexpr uint32_t kTileBytes = 32u * 32u * 4u;
constexpr uint32_t kPageBytes = 4u * kTileBytes;      // R, G, B, T

// Task #717: lossless op/colour in train mode. The per-scene packs (#122) then
// carry no UNORM16 colour: p01 = the cull's opacity code only (rounded UP, so
// the microblock cull never drops what the fp32 opacity would keep), p23 = the
// scene index. The blend reader follows rec[7] (scene index) to the fp32
// [op, r, g, b] entry of the "scene_train_opc" buffer (16 B per gaussian, 64 B
// DRAM pages) and hands those words to the compute bit for bit.
constexpr uint32_t kOpcBytes = 16u;                   // fp32 op, r, g, b
constexpr uint32_t kOpcPageBytes = 64u;               // 4 entries per DRAM page
constexpr uint32_t kOpcReaderArg = 28u;               // reader runtime arg index

// Task #730: GSPLAT_TT_TRAIN_OPC_WRITER=1 moves that gather from the blend
// reader (NCRISC, NoC1, on the slab critical path) to the blend writer (BRISC,
// NoC0, idle between tiles). Same DRAM bytes, same CB_TRAIN_OPC layout. The
// reader queues {slab L1 address, record count} on CB kOpcReqCb; the writer
// gets the scene_train_opc address as runtime arg kOpcWriterArg.
inline bool opc_on_writer() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_TRAIN_OPC_WRITER");
        return e != nullptr && e[0] == '1';
    }();
    return enabled() && on;
}
constexpr uint32_t kOpcReqCb = 14;
constexpr uint32_t kOpcWriterArg = 11u;

// Task #741: GSPLAT_TT_TRAIN_OPC_SIDE=1 (needs GSPLAT_TT_PFWC_REC32=0, whose
// 64 B blendrec carries fp32 op, r, g, b in words 5..8). The TOWN emit writes
// those 16 B per record into a side bucket ("train_opc_side": tile t, slot c at
// entry t * tile_cap + c, 1 KB pages == the bucket's 2 KB pages) and stamps c + 1
// in rec[6] bits 16..31 (the cull reads bits 0..15 only; train compute ignores
// rec[6]). Materialize copies records verbatim, so the blend reader reads the
// tile's side entries contiguously and permutes them by stamp. A record without
// a stamp (another emit path, c + 1 > 0xFFFF) sends the subchunk to the gather.
inline bool opc_side() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_TRAIN_OPC_SIDE");
        return e != nullptr && e[0] == '1';
    }();
    return enabled() && on && !opc_on_writer();
}
constexpr uint32_t kSideBytes = 16u;
constexpr uint32_t kSidePageBytes = 1024u;  // 64 entries == one 2 KB bucket page
constexpr uint32_t kSideReaderArg = 29u;    // 29: side address (0 = gather), 30: tile cap
// Tile cap of this frame's side bucket; 0 = no side data this frame (set by the
// sort for each frame, read by the blend).
inline uint32_t& side_tile_cap() {
    static uint32_t cap = 0u;
    return cap;
}

// Task #755: GSPLAT_TT_TRAIN_OPC_RING3=1 lets the blend reader run two subchunks
// ahead of the compute instead of one: CB_BUCKET_BULK and CB_TRAIN_OPC get 3
// slots. CB_TRAIN_OPC pages shrink to 32 B (a slot = 4096 x 32 B = 8192 x 16 B
// entries) and the gather staging moves to its own CB kOpcStageCb (one batch).
// Reader gather only (not with the writer gather or the side bucket).
inline bool opc_ring3() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_TRAIN_OPC_RING3");
        return e != nullptr && e[0] == '1';
    }();
    return enabled() && on && !opc_on_writer() && !opc_side();
}
constexpr uint32_t kOpcStageCb = 15;

// Task #724: decode-ahead ring slot in train mode (alpha_blend_compute_mb.cpp
// DA_SLOT_BYTES with BLEND_TRAIN): 18 SFPLOADI words for 9 fp32 values, mask
// and marker kind in the last two words. Flag off keeps 64 B slots.
inline uint32_t da_slot_bytes() { return enabled() ? 128u : 64u; }

// Smallest UNORM16 code whose decode is >= v (v not NaN): the round-to-nearest
// code (nearest) or the next one, saturating at 65535 (decode 1.0).
inline uint32_t cull_op16_from_nearest(uint32_t nearest) {
    return nearest >= 65535u ? 65535u : nearest + 1u;
}

}  // namespace gsplat_tt::blend_train
