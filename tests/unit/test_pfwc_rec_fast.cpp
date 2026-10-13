// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #599: pfwc_wsplit::rec32_chunk (the register-cursor REC32 record loop)
// against the per-survivor ChunkPages::put + Rec32Stage::next loop of
// writer_pfwc_split.cpp (rec32_chunk_ref). Over random masks, chunk starts
// (odd g, mid-page m0), bank counts and chunk runs of one writer: the same
// flushes in the same order with the same staged bytes, the same staging
// memory, cursor state, pr and poll calls.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "render/kernels/dataflow/pfwc_wsplit.h"

using namespace pfwc_wsplit;

static int fails = 0;
#define CHECK(c)                                                                      \
    do {                                                                              \
        if (!(c)) {                                                                   \
            if (fails++ < 10) std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
        }                                                                             \
    } while (0)

constexpr uint32_t MW = 32, N = MW * 32, PAYLOAD = 0x00FFFFFFu;

struct L1 {
    uint32_t pgw[7 * PW];
    uint32_t stg[NB_MAX * RL * PW];
    uint32_t hrec[PW];
    ChunkPages pg;
    Rec32Stage rs;
    std::vector<uint32_t> log;  // flushes: tag, args, staged words
    uint32_t polls = 0;
    void init(uint32_t nb) {
        std::memset(pgw, 0xA5, sizeof pgw);
        std::memset(stg, 0x5A, sizeof stg);
        std::memset(hrec, 0x3C, sizeof hrec);
        pg.sd = pgw;
        pg.so = pgw + PW;
        pg.sa = pgw + 2 * PW;
        pg.hd = pgw + 4 * PW;
        pg.ho = pgw + 5 * PW;
        pg.ha = pgw + 6 * PW;
        rs.init(nb);
    }
};

struct In {
    uint32_t mw[MW], dep[N], aabb[N], tpg[N], a[N], b[N], c[N], mx[N], my[N], u01[N], u23[N];
};

template <bool FAST>
static uint32_t run_chunk(L1& m, const In& in, uint32_t m0, uint32_t seg_base, uint32_t pr) {
    auto flush_pg = [&](volatile uint32_t* d, volatile uint32_t* o, volatile uint32_t* a, uint32_t page) {
        m.log.push_back(1);
        m.log.push_back(page);
        for (uint32_t i = 0; i < PW; i++) m.log.push_back(uint32_t(d[i])), m.log.push_back(uint32_t(o[i])), m.log.push_back(uint32_t(a[i]));
    };
    auto flush_rec = [&](uint32_t G0, uint32_t gs, uint32_t ge) {
        m.log.push_back(2);
        m.log.push_back(G0), m.log.push_back(gs), m.log.push_back(ge);
        m.rs.writes(G0, gs, ge, false, [&](uint32_t s, uint32_t g, uint32_t n) {
            m.log.push_back(s), m.log.push_back(g), m.log.push_back(n);
            for (uint32_t i = 0; i < n * PW; i++) m.log.push_back(m.stg[s * PW + i]);
        });
    };
    m.pg.begin(m0, seg_base / PW);
    m.rs.begin(seg_base + m0);
    volatile uint32_t* w_hrec = m.hrec;
    volatile uint32_t* w_rec = m.stg;
    if (FAST) {
        const RecSrc s{in.dep, in.aabb, in.tpg, in.a, in.b, in.c, in.mx, in.my, in.u01, in.u23};
        pr = rec32_chunk(in.mw, MW, s, PAYLOAD, pr, m.pg, m.rs, w_hrec, w_rec, flush_pg, flush_rec,
                         [&] { m.polls++; });
    } else {  // writer_pfwc_split.cpp's PFWC_REC32 loop
        for (uint32_t w = 0; w < MW; w++) {
            if ((w & 7u) == 7u) m.polls++;
            for (uint32_t bits = in.mw[w]; bits != 0; bits &= bits - 1) {
                const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                const uint32_t dep = in.dep[il], aabb = in.aabb[il], tpg = in.tpg[il];
                m.pg.put(dep, pr, aabb & PAYLOAD, flush_pg);
                pr += tpg & PAYLOAD;
                volatile uint32_t* r = m.rs.in_head() ? w_hrec + RW : w_rec + m.rs.stage_word();
                r[0] = in.a[il], r[1] = in.b[il], r[2] = in.c[il], r[3] = in.mx[il], r[4] = in.my[il];
                r[5] = in.u01[il], r[6] = in.u23[il], r[7] = dep;
                m.rs.next(flush_rec);
            }
        }
    }
    m.rs.end(flush_rec);
    return pr;
}

static bool same(const L1& x, const L1& y) {
    const ChunkPages &p = x.pg, &q = y.pg;
    const Rec32Stage &r = x.rs, &s = y.rs;
    return !std::memcmp(x.pgw, y.pgw, sizeof x.pgw) && !std::memcmp(x.stg, y.stg, sizeof x.stg) &&
           !std::memcmp(x.hrec, y.hrec, sizeof x.hrec) && x.log == y.log && x.polls == y.polls &&
           p.slot == q.slot && p.page == q.page && p.in_head == q.in_head && p.head_full == q.head_full &&
           (p.cd - x.pgw) == (q.cd - y.pgw) && (p.co - x.pgw) == (q.co - y.pgw) && (p.ca - x.pgw) == (q.ca - y.pgw) &&
           r.head == s.head && r.head_used == s.head_used && r.half == s.half && r.head_page == s.head_page &&
           r.rs.rb == s.rs.rb && r.rs.rl == s.rs.rl && r.rs.G0 == s.rs.G0 && r.rs.gs == s.rs.gs;
}

int main() {
    std::mt19937 rng(599);
    In in;
    long surv = 0;
    for (uint32_t nb : {1u, 2u, 3u, 5u, 8u}) {
        for (int it = 0; it < 150; it++) {
            L1 ref, fast;
            ref.init(nb);
            fast.init(nb);
            const uint32_t seg_base = 1024u * (rng() % 4);
            uint32_t m0 = rng() % 40, pr = rng();
            for (int k = 0; k < 6; k++) {
                const uint32_t dens = rng() % 5;  // empty, sparse, ~28 %, dense, full
                for (uint32_t w = 0; w < MW; w++) {
                    uint32_t v = 0;
                    for (uint32_t bit = 0; bit < 32; bit++) {
                        const uint32_t r = rng() % 100;
                        if (dens == 4 || (dens == 1 && r < 3) || (dens == 2 && r < 28) || (dens == 3 && r < 80))
                            v |= 1u << bit;
                    }
                    in.mw[w] = v;
                }
                for (uint32_t i = 0; i < N; i++) {
                    in.dep[i] = rng(), in.aabb[i] = rng(), in.tpg[i] = rng() & 0xFFFF;
                    in.a[i] = rng(), in.b[i] = rng(), in.c[i] = rng(), in.mx[i] = rng(), in.my[i] = rng();
                    in.u01[i] = rng(), in.u23[i] = rng();
                }
                const uint32_t pr_ref = run_chunk<false>(ref, in, m0, seg_base, pr);
                const uint32_t pr_fast = run_chunk<true>(fast, in, m0, seg_base, pr);
                CHECK(pr_ref == pr_fast);
                CHECK(same(ref, fast));
                uint32_t cnt = 0;
                for (uint32_t w = 0; w < MW; w++) cnt += __builtin_popcount(in.mw[w]);
                surv += cnt;
                m0 += cnt + rng() % 3;  // the other writer's chunks between
                pr = pr_ref;
            }
        }
    }
    if (fails) {
        std::printf("test_pfwc_rec_fast: %d failures\n", fails);
        return 1;
    }
    std::printf("test_pfwc_rec_fast: ok (%ld survivors)\n", surv);
    return 0;
}
