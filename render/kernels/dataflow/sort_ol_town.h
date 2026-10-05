// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #202 (host knob GSPLAT_TT_OL_EMIT_TOWN=1, docs/emit-trisc-own-t200): the
// one-launch emit's per-record loop runs on the 3 TRISCs, tile-owned. TRISC i
// owns the tiles with t % 3 == i in both movers' streams: it alone does their
// cursor read-modify-write and packs their 32 B records into the mover's ring,
// in stream order, so every record gets the same slot as the movers' own loop.
//
// Per mover (stream) one mailbox CB (CB_TWN, + 16 for mover 0 like the other
// private CBs):
//   header   GO words, constants, the published batch count (mover), per TRISC
//            its finished batch count, run words pushed (TRISC) and consumed
//            (mover).
//   slots    SLOTS batch slots (batch k in slot k % SLOTS, its blendrec pages
//            in blendrec slot k % SLOTS): per TRISC the count and a list of
//            entries t << 16 | byte offset of the record's blendrec page.
//   queues   per TRISC a ring of QCAP run words t | c << 10 (c = the run's
//            last cursor); the mover writes the run, flushes, then sets
//            fl[t] = c + 1 and consumes the word.
//   fl       per tile the cursor up to which the tile's records left L1. A
//            TRISC starts a run (c % R == 0) only once fl[t] == c.
// The mover writes GO last; at its end, after every TRISC finished all its
// batches, it clears GO, so the next launch's TRISCs wait for the new one.
#pragma once

#include <cstdint>

namespace sort_ol_town {

constexpr uint32_t CB_TWN = 14;  // + 16 for mover 0 (BRISC)
constexpr uint32_t NT = 3;       // owners: TRISC 0 (UNPACK), 1 (MATH), 2 (PACK)
constexpr uint32_t SLOTS = 4;    // batch / blendrec slots per stream
constexpr uint32_t LIST_MAX = 128;  // entries per list (>= OL_PB * 16 records per batch)
constexpr uint32_t QCAP = 256;      // run words per queue (power of two)
constexpr uint32_t TILES = 1024;    // fl entries (== OL_RING_TILES)
constexpr uint32_t MAGIC = 0x70E7A202u, MAGIC2 = 0x8F185DFDu;

// Header words. The mover writes everything before GO2, then GO; a TRISC
// writes FIN (= MAGIC) once it is done with the stream, and the mover clears
// GO, GO2 and FIN at its end only after all three FIN words are set.
constexpr uint32_t H_GO = 0, H_GO2 = 1, H_NB = 2, H_READY = 3, H_CAP = 4, H_MSK = 5, H_SH = 6, H_RING = 7,
                   H_CUR = 8;
constexpr uint32_t H_FIN = 12;   // + trisc
constexpr uint32_t H_DONE = 16;  // + 4 * trisc: batches finished
constexpr uint32_t H_QWP = 32;   // + 4 * trisc: run words pushed
constexpr uint32_t H_QRD = 48;   // + 4 * trisc: run words consumed
constexpr uint32_t HDR_BYTES = 256;
// Byte offsets in the mailbox.
constexpr uint32_t DESC_OFF = HDR_BYTES;  // per slot 4 words: n[0..2], blendrec slot address
constexpr uint32_t LIST_OFF = DESC_OFF + SLOTS * 16u;
constexpr uint32_t Q_OFF = LIST_OFF + SLOTS * NT * LIST_MAX * 4u;
constexpr uint32_t FL_OFF = Q_OFF + NT * QCAP * 4u;
constexpr uint32_t BYTES = FL_OFF + TILES * 4u;

constexpr uint32_t desc_off(uint32_t s) { return DESC_OFF + s * 16u; }
constexpr uint32_t list_off(uint32_t s, uint32_t i) { return LIST_OFF + (s * NT + i) * LIST_MAX * 4u; }
constexpr uint32_t q_off(uint32_t i) { return Q_OFF + i * QCAP * 4u; }

// t % 3 for t < 2^16.
inline uint32_t owner(uint32_t t) { return t - 3u * ((t * 0xAAABu) >> 17); }
constexpr uint32_t entry(uint32_t t, uint32_t page_off) { return (t << 16) | page_off; }
constexpr uint32_t run_word(uint32_t t, uint32_t c) { return t | (c << 10); }
constexpr uint32_t run_tile(uint32_t w) { return w & 1023u; }
constexpr uint32_t run_last(uint32_t w) { return w >> 10; }

static_assert((QCAP & (QCAP - 1u)) == 0u, "QCAP: power of two");
static_assert(TILES <= 1024u, "run words hold t in 10 bits");

}  // namespace sort_ol_town
