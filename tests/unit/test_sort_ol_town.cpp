// Checks the task #202 tile-owned TRISC emit (GSPLAT_TT_OL_EMIT_TOWN):
// render/kernels/dataflow/sort_ol_town.h and a model of its protocol, written
// like sort_bin_onelaunch.cpp (movers) and sort_ol_town_compute.cpp (TRISCs).
// Standalone:
//
//   c++ -O2 -std=c++17 -Irender/kernels/dataflow tests/unit/test_sort_ol_town.cpp
//     -o /tmp/t_town && /tmp/t_town
//
//  - owner(t) == t % 3, the entry / run word encodings round-trip, the mailbox
//    regions do not overlap and fit BYTES;
//  - the list builder's bulk page offsets equal the bank-major slot of issue_brec
//    for every bank count 1..16, also when g steps back inside the run;
//  - two movers and three TRISCs, interleaved at random (each step one record
//    of one TRISC or one action of one mover), never deadlock and write the same
//    bucket image as the sequential emit: bulk and per-g blendrec batches, the
//    per-g fallback for pairs that are not g-sorted, tiles past tile_cap, a
//    queue of 4 words (TRISCs wait for space), 2..4 slots; no ring entry
//    changes between a run's write issue and its flush.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "sort_ol_town.h"
#include "sort_onelaunch_algo.h"

namespace tw = sort_ol_town;

static int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            if (++g_fail > 20) std::exit(1);                             \
        }                                                                \
    } while (0)

constexpr uint32_t PAGE = 64, BE = 128, R = 8, HALF = 256;

// Page j of a bulk run at slot (j % nb) * sf + j / nb (issue_brec).
static uint32_t bulk_slot_page(uint32_t j, uint32_t nb, uint32_t sf) { return (j % nb) * sf + j / nb; }

struct Sim {
    // inputs: one pair stream per mover
    uint32_t ntiles, cap, slots, qcap, bk_nb;
    std::array<std::vector<uint32_t>, 2> g, t;
    // bucket image written by both movers (each tile: mover 0 cursors, then mover 1)
    std::vector<uint64_t> bucket;
    std::mt19937 rng;

    struct Write { uint32_t t, s0, last; std::vector<uint64_t> snap; };
    struct Stream {
        uint32_t nb = 0;
        std::vector<uint32_t> cur, startp, fl;
        std::vector<std::array<uint64_t, R>> ring;
        std::vector<std::vector<uint32_t>> slot_mem;  // page index in slot -> g
        std::vector<std::array<std::vector<uint32_t>, 3>> lists;
        std::vector<uint32_t> bkn;
        std::array<std::vector<uint32_t>, 3> q;
        std::array<uint32_t, 3> qwp{}, qrd{}, done{};
        uint32_t ready = 0;
        // mover: phase 0 loop top, 1 slot wait, 2 tail wait, 3 flush pending, 4 finished
        uint32_t mk = 0, phase = 0, after_flush = 0;
        std::array<uint32_t, 3> q_rd{}, wp{};
        std::vector<Write> inflight;
        // TRISC side, per TRISC
        std::array<uint32_t, 3> tk{}, tj{}, tqwp{};
    };
    std::array<Stream, 2> st;

    uint64_t rec(uint32_t gg, uint32_t tt) const { return (uint64_t(gg) << 16) | tt; }

    void issue(Stream& s, uint32_t m, uint32_t k) {
        const uint32_t sl = k % slots;
        const uint32_t p0 = k * BE, n = std::min<uint32_t>(BE, uint32_t(g[m].size()) - p0);
        auto& mem = s.slot_mem[sl];
        std::fill(mem.begin(), mem.end(), 0xFFFFFFFFu);
        // issue_brec: bulk when the batch's g run fits the slot
        uint32_t ba = 0, bn = 0;
        const uint32_t sf = bk_nb ? HALF / bk_nb : 0, bcap = sf * bk_nb;
        if (bk_nb) {
            const int32_t ga = int32_t(g[m][p0]), gb = int32_t(g[m][p0 + n - 1]);
            const uint32_t nn = uint32_t(gb - ga) + 1u;
            if (gb >= ga && nn <= bcap) {
                for (uint32_t j = 0; j < nn; j++) mem[bulk_slot_page(j, bk_nb, sf)] = uint32_t(ga) + j;
                ba = uint32_t(ga);
                bn = nn;
            }
        }
        auto per_g = [&]() {
            std::fill(mem.begin(), mem.end(), 0xFFFFFFFFu);
            int32_t sg = -1;
            uint32_t npf = 0;
            for (uint32_t j = 0; j < n; j++) {
                if (int32_t(g[m][p0 + j]) != sg) {
                    sg = int32_t(g[m][p0 + j]);
                    mem[npf++] = uint32_t(sg);
                }
            }
            bn = 0;
        };
        if (!bn) per_g();
        // build_lists (sort_bin_onelaunch.cpp)
        auto build = [&]() -> bool {
            for (auto& l : s.lists[sl]) l.clear();
            const uint32_t bk_run = sf * PAGE, bk_wrap = bcap * PAGE - PAGE;
            uint32_t jl = 0, jr = 0, off = 0, npg = 0;
            int32_t g_c = -1;
            for (uint32_t j = 0; j < n; j++) {
                const int32_t gg = int32_t(g[m][p0 + j]);
                const uint32_t tt = t[m][p0 + j];
                if (gg != g_c) {
                    g_c = gg;
                    if (bn != 0u) {
                        const uint32_t jg = uint32_t(gg) - ba;
                        if (jg >= bn) return false;
                        if (jg < jl) { jl = 0; jr = 0; off = 0; }
                        for (; jl != jg; jl++) {
                            off += bk_run;
                            if (++jr == bk_nb) { jr = 0; off -= bk_wrap; }
                        }
                    } else {
                        off = npg * PAGE;
                        npg++;
                    }
                }
                s.lists[sl][tw::owner(tt)].push_back(tw::entry(tt, off));
            }
            return true;
        };
        if (!build()) {
            per_g();
            CHECK(build());
        }
    }

    // One mover action. Returns false once finished.
    bool mover_step(uint32_t m) {
        Stream& s = st[m];
        auto min_done = [&]() { return std::min(s.done[0], std::min(s.done[1], s.done[2])); };
        auto service_issue = [&](uint32_t next_phase) -> bool {
            bool any = false;
            for (uint32_t i = 0; i < 3; i++) {
                s.wp[i] = s.qwp[i];
                for (uint32_t j = s.q_rd[i]; j != s.wp[i]; j++) {
                    const uint32_t w = s.q[i][j % qcap];
                    const uint32_t tt = tw::run_tile(w), last = tw::run_last(w);
                    const uint32_t s0 = sort_ol::ring_run_start(s.startp[tt], last, R);
                    Write wr{tt, s0, last, {}};
                    for (uint32_t c = s0; c <= last; c++) wr.snap.push_back(s.ring[tt][c % R]);
                    s.inflight.push_back(wr);
                    any = true;
                }
            }
            if (any) {
                s.after_flush = next_phase;
                s.phase = 3;
            }
            return any;
        };
        switch (s.phase) {
            case 0: {  // loop top of batch mk: publish, then refill the next slot
                if (s.mk >= s.nb) { s.phase = 2; return true; }
                s.ready = s.mk + 1;
                s.phase = (s.mk + 1 < s.nb) ? 1 : 5;
                return true;
            }
            case 1:  // slot wait (service while waiting)
                if (min_done() + slots < s.mk + 2) {
                    service_issue(1);
                    return true;
                }
                issue(s, m, s.mk + 1);
                s.phase = 5;
                return true;
            case 5:  // one service after the batch, then the next batch
                s.mk++;
                if (!service_issue(0)) s.phase = 0;
                return true;
            case 2:  // tail: wait for the TRISCs, then one last service and the drain
                if (min_done() < s.nb) {
                    service_issue(2);
                    return true;
                }
                if (service_issue(2)) return true;
                for (uint32_t tt = 0; tt < ntiles; tt++) {
                    uint32_t last;
                    if (sort_ol::ring_drain(s.startp[tt], s.cur[tt], cap, R, &last)) {
                        const uint32_t s0 = sort_ol::ring_run_start(s.startp[tt], last, R);
                        for (uint32_t c = s0; c <= last; c++) bucket[tt * cap + c] = s.ring[tt][c % R];
                    }
                }
                s.phase = 4;
                return true;
            case 3: {  // writes flushed: check, land, free the rings, consume the words
                for (const Write& wr : s.inflight) {
                    for (uint32_t c = wr.s0; c <= wr.last; c++) {
                        CHECK(s.ring[wr.t][c % R] == wr.snap[c - wr.s0]);  // not overwritten in flight
                        bucket[wr.t * cap + c] = wr.snap[c - wr.s0];
                    }
                }
                for (uint32_t i = 0; i < 3; i++) {
                    for (uint32_t j = s.q_rd[i]; j != s.wp[i]; j++) {
                        const uint32_t w = s.q[i][j % qcap];
                        s.fl[tw::run_tile(w)] = tw::run_last(w) + 1u;
                    }
                    s.q_rd[i] = s.wp[i];
                    s.qrd[i] = s.wp[i];
                }
                s.inflight.clear();
                s.phase = s.after_flush;
                return true;
            }
            default:
                return false;
        }
    }

    // One TRISC step of TRISC i: at most one record. Returns false when idle.
    bool trisc_step(uint32_t i, uint32_t& rr) {
        for (uint32_t a = 0; a < 2; a++) {
            const uint32_t m = (rr + a) & 1u;
            Stream& s = st[m];
            if (s.tk[i] >= s.nb || s.ready <= s.tk[i]) continue;
            const uint32_t sl = s.tk[i] % slots;
            const auto& l = s.lists[sl][i];
            if (s.tj[i] < l.size()) {
                const uint32_t e = l[s.tj[i]];
                const uint32_t tt = e >> 16, off = e & 0xFFFFu;
                CHECK(off % PAGE == 0 && off / PAGE < HALF);
                const uint32_t gg = s.slot_mem[sl][off / PAGE];
                const uint32_t c = s.cur[tt];
                if (c < cap) {
                    const uint32_t ri = c & (R - 1u);
                    if (ri == 0u && s.fl[tt] != c) return false;  // previous run still in flight
                    if (ri == R - 1u && s.tqwp[i] - s.qrd[i] >= qcap) return false;  // queue full
                    s.ring[tt][ri] = rec(gg, tt);
                    if (ri == R - 1u) {
                        s.q[i][s.tqwp[i] % qcap] = tw::run_word(tt, c);
                        s.qwp[i] = ++s.tqwp[i];
                    }
                }
                s.cur[tt] = c + 1u;
                s.tj[i]++;
                return true;
            }
            s.tj[i] = 0;
            s.done[i] = ++s.tk[i];
            rr = m ^ 1u;
            return true;
        }
        return false;
    }

    bool run(std::vector<uint64_t>& ref) {
        // sequential emit (sort_bin_onelaunch.cpp's fast loop), mover 0 then mover 1 cursors
        std::vector<uint32_t> base(ntiles, 0), cnt0(ntiles, 0);
        for (uint32_t tt : t[0]) cnt0[tt]++;
        ref.assign(size_t(ntiles) * cap, 0);
        for (uint32_t m = 0; m < 2; m++) {
            std::vector<uint32_t> cur(ntiles);
            for (uint32_t tt = 0; tt < ntiles; tt++) cur[tt] = m ? cnt0[tt] : 0u;
            for (size_t j = 0; j < g[m].size(); j++) {
                const uint32_t c = cur[t[m][j]]++;
                if (c < cap) ref[t[m][j] * cap + c] = rec(g[m][j], t[m][j]);
            }
        }
        bucket.assign(size_t(ntiles) * cap, 0);
        for (uint32_t m = 0; m < 2; m++) {
            Stream& s = st[m];
            s = Stream{};
            s.nb = uint32_t((g[m].size() + BE - 1) / BE);
            s.cur.resize(ntiles);
            for (uint32_t tt = 0; tt < ntiles; tt++) s.cur[tt] = m ? cnt0[tt] : 0u;
            s.startp = s.cur;
            s.fl = s.cur;
            s.ring.assign(ntiles, {});
            s.slot_mem.assign(slots, std::vector<uint32_t>(HALF, 0xFFFFFFFFu));
            s.lists.assign(slots, {});
            for (auto& qq : s.q) qq.assign(qcap, 0);
            if (s.nb) issue(s, m, 0);
        }
        std::array<uint32_t, 3> rr{};
        uint64_t steps = 0, idle = 0;
        for (;;) {
            const bool fin = st[0].phase == 4 && st[1].phase == 4;
            if (fin) break;
            const uint32_t who = rng() % 5u;
            bool prog;
            if (who < 2) prog = mover_step(who);
            else prog = trisc_step(who - 2, rr[who - 2]);
            idle = prog ? 0 : idle + 1;
            if (++steps > 50000000ull || idle > 100000) {
                std::printf("deadlock / no progress after %llu steps\n", (unsigned long long)steps);
                return false;
            }
        }
        return true;
    }
};

int main() {
    // encodings and layout
    for (uint32_t tt = 0; tt < 65536u; tt++) CHECK(tw::owner(tt) == tt % 3u);
    for (uint32_t tt : {0u, 1u, 513u, 1023u}) {
        for (uint32_t c : {0u, 7u, 4095u, (1u << 22) - 1u}) {
            CHECK(tw::run_tile(tw::run_word(tt, c)) == tt && tw::run_last(tw::run_word(tt, c)) == c);
        }
        CHECK((tw::entry(tt, 16320u) >> 16) == tt && (tw::entry(tt, 16320u) & 0xFFFFu) == 16320u);
    }
    CHECK(tw::desc_off(tw::SLOTS - 1u) + 16u <= tw::LIST_OFF);
    CHECK(tw::list_off(tw::SLOTS - 1u, tw::NT - 1u) + tw::LIST_MAX * 4u <= tw::Q_OFF);
    CHECK(tw::q_off(tw::NT - 1u) + tw::QCAP * 4u <= tw::FL_OFF);
    CHECK(tw::FL_OFF + tw::TILES * 4u == tw::BYTES && tw::BYTES % 32u == 0u);
    CHECK(tw::H_QRD + 4u * (tw::NT - 1u) < tw::HDR_BYTES / 4u && tw::H_FIN + tw::NT <= tw::H_DONE);
    // bulk page stepping == bank-major slot, forward and after a step back
    for (uint32_t nb = 1; nb <= 16; nb++) {
        const uint32_t sf = HALF / nb, bcap = sf * nb, bk_run = sf * PAGE, bk_wrap = bcap * PAGE - PAGE;
        std::mt19937 r(nb);
        uint32_t jl = 0, jr = 0, off = 0;
        for (int it = 0; it < 2000; it++) {
            const uint32_t jg = r() % bcap;
            if (jg < jl) { jl = 0; jr = 0; off = 0; }
            for (; jl != jg; jl++) {
                off += bk_run;
                if (++jr == nb) { jr = 0; off -= bk_wrap; }
            }
            CHECK(off == bulk_slot_page(jg, nb, sf) * PAGE);
        }
    }
    // protocol
    int runs = 0;
    for (uint32_t seed = 1; seed <= 300; seed++) {
        Sim sim;
        sim.rng.seed(seed);
        std::mt19937 r(seed * 7919u);
        sim.ntiles = 24 + r() % 80;
        sim.cap = 64u * (1 + r() % 2);
        sim.slots = 2 + r() % 3;
        sim.qcap = (seed % 3 == 0) ? 4u : 256u;
        sim.bk_nb = (seed % 5 == 0) ? 0u : 1 + r() % 12;
        const bool unsorted = seed % 7 == 0;
        for (uint32_t m = 0; m < 2; m++) {
            const uint32_t ng = r() % 700;
            uint32_t gg = r() % 50;
            for (uint32_t i = 0; i < ng; i++) {
                gg += 1 + (r() % 4 == 0 ? r() % 6 : 0);  // near-dense g
                const uint32_t k = 1 + r() % 6, c0 = r() % sim.ntiles;
                for (uint32_t a = 0; a < k; a++) {
                    sim.g[m].push_back(gg);
                    sim.t[m].push_back((c0 + a * (1 + r() % 3)) % sim.ntiles);
                }
            }
            if (unsorted && sim.g[m].size() > 10) std::swap(sim.g[m][3], sim.g[m][sim.g[m].size() - 2]);
        }
        std::vector<uint64_t> ref;
        if (!sim.run(ref)) {
            std::printf("seed %u: no progress\n", seed);
            g_fail++;
            continue;
        }
        CHECK(sim.bucket == ref);
        runs++;
    }
    std::printf("%s: %d protocol runs\n", g_fail ? "FAILED" : "OK", runs);
    return g_fail ? 1 : 0;
}
