// Task #599 p11: the pfwc split writer's PFWC_REC32 record loop
// (writer_pfwc_split.cpp) on one mover, cycles per survivor.
//   MB_V=0: the production loop (ChunkPages::put + Rec32Stage::next per survivor)
//   MB_V=1: pfwc_wsplit::rec32_chunk (register cursors, loads before stores)
// n chunks of 1024 gaussians from synthetic 4 KB input tiles in scratch; the
// survivor mask (~28 %, bicycle's density) is an LCG per chunk, made outside
// the timed part. Flushes are real NoC writes to the 64 B page buffer (page
// index masked), timed as aux A. total = the record loop incl. flushes, ops =
// survivors, so tpo - a_med = cycles per survivor without flushes (the t484
// "rec - fl"). chk hashes every flushed word: equal across MB_V.
// mode: bits 0-3 nb (banks, 1..8); bit 4 calls the (empty) reader poll.
#include "mb_common.h"
#include "../../kernels/dataflow/pfwc_wsplit.h"

using namespace pfwc_wsplit;

void kernel_main() {
    const mb::Args a = mb::args();
    MB_DATA_ACC(a);
    constexpr uint32_t TB = 4096, PB = PW * 4, MW = 32, PAYLOAD = 0x00FFFFFFu;
    const uint32_t nb = (a.mode & 15u) ? (a.mode & 15u) : 8u;
    const bool rd_here = (a.mode & 16u) != 0;
    // Scratch: 10 input tiles, record staging, 7 page-staging pages, mask, head record.
    const uint32_t l1_in = a.scratch, l1_rec = l1_in + 10 * TB, l1_pg = l1_rec + NB_MAX * RL * PB;
    const uint32_t l1_mask = l1_pg + 7 * PB, l1_hrec = l1_mask + 128;
    auto tile = [&](uint32_t i) { return reinterpret_cast<volatile uint32_t*>(l1_in + i * TB); };
    auto p_dep = tile(0), p_aabb = tile(1), p_tpg = tile(2), p_a = tile(3), p_b = tile(4), p_c = tile(5);
    auto p_m2x = tile(6), p_m2y = tile(7), q01 = tile(8), q23 = tile(9);
    for (uint32_t i = 0; i < 10 * 1024; i++) tile(0)[i] = i * 2654435761u;
    for (uint32_t i = 0; i < 1024; i++) p_tpg[i] = i & 63u;
    auto w_rec = reinterpret_cast<volatile uint32_t*>(l1_rec);
    auto w_hrec = reinterpret_cast<volatile uint32_t*>(l1_hrec);
    auto mw = reinterpret_cast<volatile uint32_t*>(l1_mask);
    auto pgw = [&](uint32_t i) { return reinterpret_cast<volatile uint32_t*>(l1_pg + i * PB); };
    ChunkPages pg{};
    pg.sd = pgw(0), pg.so = pgw(1), pg.sa = pgw(2), pg.hd = pgw(4), pg.ho = pgw(5), pg.ha = pgw(6);
    Rec32Stage rs{};
    rs.init(nb);

    const uint32_t pmask = a.data_npages - 1;
    uint32_t chk = 0;
    uint64_t fl = 0, tot = 0;
    auto hash = [&](volatile uint32_t* p, uint32_t nw) {
        for (uint32_t i = 0; i < nw; i++) chk = (chk ^ p[i]) * 16777619u;
    };
    auto flush_rec = [&](uint32_t G0, uint32_t gs, uint32_t ge) {
        const uint64_t t0 = mb::now();
        rs.writes(G0, gs, ge, true, [&](uint32_t s, uint32_t g, uint32_t n) {
            noc_async_write(l1_rec + s * PB, get_noc_addr((g + a.salt) & pmask, data_acc), n * PB);
            hash(w_rec + s * PW, n * PW);
            chk += g;
        });
        noc_async_writes_flushed();
        fl += mb::now() - t0;
    };
    auto l1a = [](volatile uint32_t* p) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p)); };
    auto flush_pg = [&](volatile uint32_t* d, volatile uint32_t* o, volatile uint32_t* aa, uint32_t page) {
        const uint64_t t0 = mb::now();
        const uint32_t p = (3 * page + a.salt) & pmask;
        noc_async_write(l1a(d), get_noc_addr(p, data_acc), PB);
        noc_async_write(l1a(o), get_noc_addr((p + 1) & pmask, data_acc), PB);
        noc_async_write(l1a(aa), get_noc_addr((p + 2) & pmask, data_acc), PB);
        hash(d, PW), hash(o, PW), hash(aa, PW);
        chk += page;
        noc_async_writes_flushed();
        fl += mb::now() - t0;
    };
    auto rd_poll = [&] { asm volatile("" ::: "memory"); };

    uint32_t m0 = 0, pr = 0, ops = 0, lcg = 599u + a.salt;
    {
        DeviceZoneScopedN("p11_pfwc_rec");
        for (uint32_t k = 0; k < a.n; k++) {
            uint32_t cnt = 0;
            for (uint32_t w = 0; w < MW; w++) {  // ~28 % survivors
                uint32_t v = 0;
                for (uint32_t bit = 0; bit < 32; bit++) {
                    lcg = lcg * 1664525u + 1013904223u;
                    v |= ((lcg >> 24) < 72u ? 1u : 0u) << bit;
                }
                mw[w] = v;
                cnt += __builtin_popcount(v);
            }
            const uint64_t t0 = mb::now();
            pg.begin(m0, 0);
            rs.begin(m0);
#if MB_V == 0
            for (uint32_t w = 0; w < MW; w++) {
                if (rd_here && (w & 7u) == 7u) rd_poll();
                for (uint32_t bits = mw[w]; bits != 0; bits &= bits - 1) {
                    const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                    const uint32_t dep = p_dep[il], aabb = p_aabb[il], tpg = p_tpg[il];
                    pg.put(dep, pr, aabb & PAYLOAD, flush_pg);
                    pr += tpg & PAYLOAD;
                    volatile uint32_t* r = rs.in_head() ? w_hrec + RW : w_rec + rs.stage_word();
                    {
                        const uint32_t a0 = p_a[il], b = p_b[il], cc = p_c[il];
                        const uint32_t mx = p_m2x[il], my = p_m2y[il];
                        r[0] = a0;
                        r[1] = b;
                        r[2] = cc;
                        r[3] = mx;
                        r[4] = my;
                    }
                    {
                        const uint32_t u01 = q01[il], u23 = q23[il];
                        r[5] = u01;
                        r[6] = u23;
                        r[7] = dep;
                    }
                    rs.next(flush_rec);
                }
            }
#else
            const RecSrc src{p_dep, p_aabb, p_tpg, p_a, p_b, p_c, p_m2x, p_m2y, q01, q23};
            pr = rec32_chunk(mw, MW, src, PAYLOAD, pr, pg, rs, w_hrec, w_rec, flush_pg, flush_rec, [&] {
                if (rd_here) rd_poll();
            });
#endif
            rs.end(flush_rec);
            tot += mb::now() - t0;
            hash(w_hrec, PW);
            ops += cnt;
            m0 += cnt + (k & 1u);  // a gap of one g (the other writer) every other chunk
        }
    }
    noc_async_write_barrier();
    mb::publish(a, ops, tot, fl, 0, chk + pr);
}
