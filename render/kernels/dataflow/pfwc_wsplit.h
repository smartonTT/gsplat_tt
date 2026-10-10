// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #207 (GSPLAT_TT_PFWC_WRITER_SPLIT=1): the lever B fused pfwc writer
// split over BRISC (even chunks) and NCRISC (odd chunks), writer_pfwc_split.cpp.
// Device-free, so tests/unit/test_pfwc_wsplit.cpp runs this page and record
// logic on the host.
//
// Chunk k of a core (tile chunk_start + k * stride) holds the core's visible
// gaussians [m_k, m_{k+1}) (segment storage index seg_base + j) and their
// segment-local pair offsets from pr_k on. The writer of chunk k gets from the
// writer of chunk k - 1 (the other RISC), through a mailbox slot in L1:
//   PREFIX: (m_k, pr_k), sent as soon as chunk k - 1 is classified;
//   OPEN:   the dep / offs / aabb words [0, m_k % 16) of the 16-word page that
//           chunk k starts in, sent when chunk k - 1 is done (only if m_k % 16).
// Chunk k stages that page apart (the head) so its record loop never waits on
// OPEN; finish() merges the words in at the end of the chunk. Every page is
// written once, by the writer that completes it (or by the core's last chunk,
// padded like writer_pfwc_fuse.cpp's tail), so the DRAM bytes match the single
// writer's.
#pragma once

#include <cstdint>

#include "pfwc_fuse.h"

namespace pfwc_wsplit {

constexpr uint32_t PW = pfwc_fuse::PAGE_WORDS;
constexpr uint32_t RL = 16, NB_MAX = 8;  // record staging, as writer_pfwc_fuse.cpp
constexpr uint32_t TILE_BYTES = 4096;    // fp32 32x32 tile

// Mailbox: R slots per direction (to BRISC, to NCRISC) of MSG_WORDS words, one
// semaphore per slot: EMPTY -> PREFIX -> (OPEN) -> EMPTY, the receiver empties.
constexpr uint32_t R = 2;
constexpr uint32_t MSG_WORDS = 64;
constexpr uint32_t MSG_M = 0, MSG_PR = 1, MSG_DEP = 16, MSG_OFFS = 32, MSG_AABB = 48;
constexpr uint32_t MBX_BYTES = 2 * R * MSG_WORDS * 4;
constexpr uint32_t F_EMPTY = 0, F_PREFIX = 1, F_OPEN = 2;
constexpr uint32_t NUM_SEMS = 2 * R;

// Per-RISC staging CB: opacity, col r / g / b, puboc01 / 23 tiles, the record
// staging, the dep / offs / aabb / counts pages, the head dep / offs / aabb
// pages and the 128 B mask, after 64 B alignment.
constexpr uint32_t STG_BYTES = 64 + 6 * TILE_BYTES + NB_MAX * RL * PW * 4 + 7 * PW * 4 + 128;

// Odd chunks' compute output CB of even-set CB id cb (9..16 -> 41..48,
// 35 / 36 -> 49 / 50); project_pfwc_compute.cpp's OCB.
constexpr uint32_t odd_cb(uint32_t cb) { return cb < 32u ? cb + 32u : cb + 14u; }
constexpr uint32_t CB_STG_ODD = 51, CB_MBX = 52;

// Writer of chunk k: 0 = BRISC, 1 = NCRISC.
inline uint32_t owner(uint32_t k) { return k & 1u; }
// Mailbox slot (0 .. 2R - 1) of chunk j's PREFIX / OPEN, j >= 1.
inline uint32_t slot_index(uint32_t j) { return owner(j) * R + ((j - 1u) / 2u) % R; }

// One chunk's dep / offs / aabb page staging.
struct ChunkPages {
    volatile uint32_t* sd;  // staging page
    volatile uint32_t* so;
    volatile uint32_t* sa;
    volatile uint32_t* hd;  // head page
    volatile uint32_t* ho;
    volatile uint32_t* ha;
    volatile uint32_t* cd;  // the page being filled (head or staging)
    volatile uint32_t* co;
    volatile uint32_t* ca;
    uint32_t slot, page, head_page, head_s;
    bool in_head, head_full;

    // Chunk starting at segment index m0; page_base = seg_base / PW.
    void begin(uint32_t m0, uint32_t page_base) {
        slot = m0 % PW;
        page = page_base + m0 / PW;
        head_page = page;
        head_s = slot;
        in_head = slot != 0;
        head_full = false;
        cd = in_head ? hd : sd;
        co = in_head ? ho : so;
        ca = in_head ? ha : sa;
    }
    // flush(d, o, a, page) writes the three staged pages.
    template <class Flush>
    inline void put(uint32_t dep, uint32_t offs, uint32_t aabb, Flush&& flush) {
        cd[slot] = dep;
        co[slot] = offs;
        ca[slot] = aabb;
        if (++slot == PW) {
            if (in_head) {
                in_head = false;
                head_full = true;
                cd = sd;
                co = so;
                ca = sa;
            } else {
                flush(sd, so, sa, page);
            }
            slot = 0;
            page++;
        }
    }
    // End of the chunk, when head_s != 0: msg is chunk's OPEN message. After
    // this the open page is cd / co / ca words [0, slot) of page `page`.
    template <class Flush>
    void finish(const volatile uint32_t* msg, Flush&& flush) {
        for (uint32_t i = 0; i < head_s; i++) {
            hd[i] = msg[MSG_DEP + i];
            ho[i] = msg[MSG_OFFS + i];
            ha[i] = msg[MSG_AABB + i];
        }
        if (head_full) flush(hd, ho, ha, head_page);
    }
    // The open page to the next chunk's OPEN message (slot != 0).
    void export_open(volatile uint32_t* msg) const {
        for (uint32_t i = 0; i < slot; i++) {
            msg[MSG_DEP + i] = cd[i];
            msg[MSG_OFFS + i] = co[i];
            msg[MSG_AABB + i] = ca[i];
        }
    }
    // The core's last chunk: the partial last page, depth / aabb 0 and offs =
    // the segment's pairs past the last gaussian (writer_pfwc_fuse.cpp's tail).
    template <class Flush>
    void close(uint32_t pr_end, Flush&& flush) {
        if (slot == 0) return;
        for (uint32_t s = slot; s < PW; s++) {
            cd[s] = 0;
            co[s] = pr_end;
            ca[s] = 0;
        }
        flush(cd, co, ca, page);
    }
};

// One chunk's blend-record staging: writer_pfwc_fuse.cpp's bank-major groups
// of RL * nb pages, restarted at the chunk's first record and flushed at its
// end, so each writer writes exactly its own records.
struct RecStage {
    uint32_t nb, GS, G0, gs, rb, rl;
    void init(uint32_t nbanks) {
        nb = nbanks;
        GS = RL * nb;
    }
    void begin(uint32_t g) {
        G0 = g / GS * GS;
        gs = g;
        rb = (g - G0) % nb;
        rl = (g - G0) / nb;
    }
    uint32_t slot() const { return rb * RL + rl; }
    // flush_rec(G0, gs, ge) writes staged records [gs, ge) of group G0.
    template <class F>
    inline void next(F&& flush_rec) {
        if (++rb == nb) {
            rb = 0;
            if (++rl == RL) {
                flush_rec(G0, gs, G0 + GS);
                G0 += GS;
                gs = G0;
                rl = 0;
            }
        }
    }
    template <class F>
    void end(F&& flush_rec) {
        const uint32_t ge = G0 + rl * nb + rb;
        if (ge > gs) flush_rec(G0, gs, ge);
    }
    // flush_rec's writes of staged records [gs, ge) of group G0, as write(s, g,
    // n): n staged slots from slot s to pages g, g + nb, ... (one DRAM bank, nb
    // = the bank count), one call per bank; per_page (more banks than NB_MAX):
    // one call per page, n = 1.
    template <class W>
    void writes(uint32_t G0_, uint32_t gs_, uint32_t ge_, bool per_page, W&& write) const {
        const uint32_t d = gs_ - G0_, e = ge_ - G0_;
        for (uint32_t b = 0; b < nb; ++b) {
            uint32_t lo = 0, hi = RL;
            if (d != 0 || e != GS) {
                lo = d > b ? (d - b + nb - 1) / nb : 0;
                hi = e > b ? (e - b + nb - 1) / nb : 0;
            }
            if (hi <= lo) continue;
            if (per_page) {
                for (uint32_t l = lo; l < hi; ++l) write(b * RL + l, G0_ + b + l * nb, 1u);
            } else {
                write(b * RL + lo, G0_ + b + lo * nb, hi - lo);
            }
        }
    }
};

// Task #467 (PFWC_REC32, docs/pfwc-rec32-t467): 32 B records, RW words
// [a, b, c, mx, my, u01, u23, dep], two per 64 B page (gaussian g: page g / 2,
// words (g % 2) * RW). RecStage then stages pages. A chunk starting at an odd g
// shares its first page with chunk k - 1: chunk k - 1 sends its last record in
// the OPEN message (words MSG_REC ..; OPEN is sent whenever m_k % 16, so always
// when m_k is odd) and chunk k writes the whole page from its head page. A chunk
// ending at an odd g sends that half page on, or (the core's last chunk) writes
// it with the upper half zero. Every page is written once, by one writer.
constexpr uint32_t RW = 8, MSG_REC = 2;
static_assert(MSG_REC + RW <= MSG_DEP, "OPEN record words overlap the dep words");
constexpr uint32_t STG_BYTES_REC32 = STG_BYTES + PW * 4;  // + the head record page
struct Rec32Stage {
    RecStage rs;
    uint32_t head_page;
    bool head;  // the chunk starts at an odd g: its first record is the head page's upper half
    bool head_used, half;
    void init(uint32_t nbanks) { rs.init(nbanks); }
    void begin(uint32_t g) {
        head = (g & 1u) != 0u;
        head_used = false;
        half = false;
        head_page = g >> 1;
        rs.begin((g + 1u) >> 1);
    }
    bool in_head() const { return head && !head_used; }
    // Word offset of the next record in the staging (when !in_head()).
    uint32_t stage_word() const { return rs.slot() * PW + (half ? RW : 0u); }
    template <class F>
    inline void next(F&& flush_rec) {
        if (in_head()) {
            head_used = true;
        } else if (half) {
            half = false;
            rs.next(flush_rec);
        } else {
            half = true;
        }
    }
    // Flushes the complete staged pages; an open half page (half) stays staged.
    template <class F>
    void end(F&& flush_rec) {
        rs.end(flush_rec);
    }
    template <class W>
    void writes(uint32_t G0_, uint32_t gs_, uint32_t ge_, bool per_page, W&& write) const {
        rs.writes(G0_, gs_, ge_, per_page, write);
    }
    // The page of the open staged half page (half) and its staging slot.
    uint32_t open_page() const { return rs.G0 + rs.rl * rs.nb + rs.rb; }
    uint32_t open_word() const { return rs.slot() * PW; }
};

}  // namespace pfwc_wsplit
