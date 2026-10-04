// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #165 (host knob GSPLAT_TT_OL_EMIT_TPACK): the one-launch emit's record
// pack runs on the 3 TRISCs (sort_ol_tpack_compute.cpp); the movers
// (sort_bin_onelaunch.cpp) scan the pair planes, assign the slots, issue the
// blendrec reads and write the runs. Shared layout of the per-mover mailbox CB.
//
// A mover publishes one segment at a time: n items (one word per record:
// ring entry index << 8 | blendrec page slot), the blendrec page base, and the
// chunk bounds of TRISCs 1 and 2. It writes MB_SEQ last; each TRISC packs its
// chunk and writes MB_SEQ's value to its MB_DONE word. n == TPK_FINAL ends the
// launch; the mover then clears MB_MAGIC, MB_SEQ and the done words, so the next
// launch starts from zeros.
#pragma once

#include <cstdint>

namespace sort_ol_tpack {

constexpr uint32_t CB_TPK = 14;  // + 16 for mover 0 (BRISC), like the other private CBs
constexpr uint32_t MAGIC = 0x7E5A1650u;
constexpr uint32_t TPK_FINAL = 0xFFFFFFFFu;
// Header words.
constexpr uint32_t MB_MAGIC = 0, MB_SEQ = 1, MB_N = 2, MB_ITEMS = 3, MB_B1 = 4, MB_B2 = 5, MB_BREC = 6,
                   MB_RING = 7, MB_MSK = 8, MB_SH = 9;
constexpr uint32_t MB_DONE = 16;  // + 4 * trisc
constexpr uint32_t HDR_BYTES = 128;
constexpr uint32_t SEG_MAX = 128;  // items per segment (<= one batch of pair elements)
constexpr uint32_t ITEMS_OFF = HDR_BYTES;                      // 2 x SEG_MAX item words
constexpr uint32_t RUNS_OFF = ITEMS_OFF + 2u * SEG_MAX * 4u;  // 2 x SEG_MAX run words
constexpr uint32_t TPK_BYTES = RUNS_OFF + 2u * SEG_MAX * 4u;
// Mover cursors carry the segment tag of the tile's last completed run above
// the 20-bit cursor (a reused ring entry in the same segment ends the segment).
constexpr uint32_t CUR_BITS = 20, CUR_MASK = (1u << CUR_BITS) - 1u;

}  // namespace sort_ol_tpack
