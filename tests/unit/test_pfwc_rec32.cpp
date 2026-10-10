// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #467 (GSPLAT_TT_PFWC_REC32=1): host model of the split writer's 32 B
// record staging (pfwc_wsplit::Rec32Stage, writer_pfwc_split.cpp's PFWC_REC32
// steps) and of the emit's page addressing (sort_bin_onelaunch.cpp brec_pg /
// brec_ho, bulk runs). Checks, over random chunk splits (empty chunks, odd
// starts, one-record chunks) and bank counts: every page of the core's records
// is written exactly once, by one writer, holding record g at words (g % 2) * 8
// (the core's odd last page: upper half 0); and the emit's bulk-run slot and
// per-g offsets find record g.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "render/kernels/dataflow/pfwc_wsplit.h"

using namespace pfwc_wsplit;

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            if (fails++ < 10) std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
        }                                                                \
    } while (0)

// Record word w of gaussian g (never 0).
static uint32_t word(uint32_t g, uint32_t w) { return (g + 1u) * 16u + w + 1u; }

struct Writer {
    std::vector<uint32_t> stg = std::vector<uint32_t>(NB_MAX * RL * PW, 0xDEADu);
    uint32_t hrec[PW] = {};
    Rec32Stage rs{};
};

static void run_core(uint32_t nb, uint32_t seg_base, const std::vector<uint32_t>& counts) {
    uint32_t M = 0;
    for (uint32_t c : counts) M += c;
    const uint32_t p_lo = seg_base / 2, p_hi = (seg_base + M + 1) / 2;
    std::vector<uint32_t> dram((p_hi + 4) * PW, 0xBADu), nwr(p_hi + 4, 0);
    Writer wr[2];
    for (auto& w : wr) w.rs.init(nb);
    uint32_t msg[2][MSG_WORDS] = {};  // msg[k % 2]: chunk k's OPEN (the model runs chunks in order)
    auto put_page = [&](const uint32_t* src, uint32_t page) {
        for (uint32_t i = 0; i < PW; i++) dram[page * PW + i] = src[i];
        nwr[page]++;
    };
    uint32_t m0 = 0;
    const uint32_t K = static_cast<uint32_t>(counts.size());
    for (uint32_t k = 0; k < K; k++) {
        Writer& w = wr[owner(k)];
        Rec32Stage& rs = w.rs;
        auto flush_rec = [&](uint32_t G0, uint32_t gs, uint32_t ge) {
            rs.rs.writes(G0, gs, ge, false, [&](uint32_t s, uint32_t g, uint32_t n) {
                for (uint32_t i = 0; i < n; i++) put_page(&w.stg[(s + i) * PW], g + i * nb);
            });
        };
        rs.begin(seg_base + m0);
        for (uint32_t j = 0; j < counts[k]; j++) {
            const uint32_t g = seg_base + m0 + j;
            uint32_t* r = rs.in_head() ? w.hrec + RW : &w.stg[rs.stage_word()];
            for (uint32_t i = 0; i < RW; i++) r[i] = word(g, i);
            rs.next(flush_rec);
        }
        rs.end(flush_rec);
        const bool open_in = (m0 % PW) != 0;  // OPEN arrives (pg.head_s != 0)
        CHECK(!rs.head || open_in);
        if (open_in && rs.head) {
            for (uint32_t i = 0; i < RW; i++) w.hrec[i] = msg[k % 2][MSG_REC + i];
            if (rs.head_used) put_page(w.hrec, rs.head_page);
        }
        const uint32_t m1 = m0 + counts[k];
        if (k + 1 < K) {
            if (m1 % PW != 0) {  // pg.slot != 0
                const uint32_t* src = rs.half ? &w.stg[rs.open_word()] : w.hrec;
                if (rs.half || rs.in_head())
                    for (uint32_t i = 0; i < RW; i++) msg[(k + 1) % 2][MSG_REC + i] = src[i];
            }
            CHECK((m1 & 1u) == ((rs.half || rs.in_head()) ? 1u : 0u));
        } else if (rs.half || rs.in_head()) {
            uint32_t* p = rs.half ? &w.stg[rs.open_word()] : w.hrec;
            for (uint32_t i = RW; i < PW; i++) p[i] = 0;
            put_page(p, rs.half ? rs.open_page() : rs.head_page);
        }
        m0 = m1;
    }
    for (uint32_t p = 0; p < p_hi + 4; p++) {
        const bool mine = p >= p_lo && p < p_hi;
        CHECK(nwr[p] == (mine ? 1u : 0u));
        if (!mine) continue;
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t g = 2 * p + h;
            for (uint32_t i = 0; i < RW; i++)
                CHECK(dram[p * PW + h * RW + i] == (g < seg_base + M ? word(g, i) : 0u));
        }
    }
}

// The emit's bulk run (issue_brec / build_lists, REC32 page addressing): pages
// pa + r of a run of n pages land at slot (r % nb) * sf + r / nb.
static void check_emit(uint32_t nb, uint32_t half_pages, uint32_t ga, uint32_t gb) {
    auto pg = [](uint32_t g) { return g >> 1; };
    auto ho = [](uint32_t g) { return (g & 1u) * 32u; };
    const uint32_t sf = half_pages / nb, cap = sf * nb, PBY = PW * 4;
    const uint32_t n = pg(gb) - pg(ga) + 1;
    if (n > cap) return;  // per-g pages instead
    std::vector<uint32_t> l1(half_pages * PW, 0);
    const uint32_t q = n / nb, rem = n - q * nb, nr = n < nb ? n : nb;
    for (uint32_t r = 0; r < nr; r++) {
        const uint32_t len = q + (r < rem ? 1u : 0u);
        for (uint32_t i = 0; i < len; i++) {  // bank run: pages pa + r + i * nb
            const uint32_t page = pg(ga) + r + i * nb;
            for (uint32_t h = 0; h < 2; h++)
                for (uint32_t w = 0; w < RW; w++) l1[(r * sf + i) * PW + h * RW + w] = word(2 * page + h, w);
        }
    }
    const uint32_t run = sf * PBY, wrap = cap * PBY - PBY;
    uint32_t jl = 0, jr = 0, off = 0;
    for (uint32_t g = ga; g <= gb; g++) {
        const uint32_t jg = pg(g) - pg(ga);
        CHECK(jg < n);
        for (; jl != jg; jl++) {
            off += run;
            if (++jr == nb) {
                jr = 0;
                off -= wrap;
            }
        }
        const uint32_t e = off + ho(g);
        CHECK(e < 65536u);
        for (uint32_t w = 0; w < RW; w++) CHECK(l1[e / 4 + w] == word(g, w));
    }
}

int main() {
    std::mt19937 rng(467);
    for (uint32_t nb : {1u, 2u, 3u, 7u, 8u}) {
        for (int it = 0; it < 300; it++) {
            const uint32_t K = 1 + rng() % 9;
            std::vector<uint32_t> counts(K);
            for (auto& c : counts) {
                const uint32_t r = rng() % 6;
                c = r == 0 ? 0u : r == 1 ? 1u : r == 2 ? rng() % 40 : rng() % (2 * RL * nb * 2 + 7);
            }
            run_core(nb, 1024u * (rng() % 4), counts);
        }
        for (int it = 0; it < 200; it++) {
            const uint32_t ga = rng() % 5000, gb = ga + rng() % 520;
            check_emit(nb, 256u, ga, gb);
        }
    }
    if (fails) {
        std::printf("test_pfwc_rec32: %d failures\n", fails);
        return 1;
    }
    std::printf("test_pfwc_rec32: ok\n");
    return 0;
}
