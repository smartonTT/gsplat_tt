// Host check of the split pfwc writer (task #207, writer_pfwc_split.cpp and
// pfwc_wsplit.h). Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_pfwc_wsplit.cpp
//
// Runs one core's chunks through two writer state machines (BRISC even
// chunks, NCRISC odd chunks; one of them runs the reader) that follow the kernel's steps:
// output wait, classify, PREFIX receive / send, record loop, OPEN receive,
// OPEN send or the last chunk's close and counts, pop. Compute and the reader
// are modelled as CB counts (inputs depth 3, two output sets of depth 2); a
// reader only advances where its RISC polls it. Task #232: NCRISC reads all
// input tiles, BRISC reads all, or each reads a subset (two readers with their
// own input CBs; compute needs both); all three are run. A random scheduler interleaves
// the agents. For random chunk sizes (incl. 0, 1, 15..17 and full tiles),
// chunk counts 0..13, segment bases and DRAM bank counts 1..12 it checks
// against writer_pfwc_fuse.cpp's single writer: the dep / offs / aabb pages,
// the records and the counts page are identical and each written exactly
// once, nothing else is written, no mailbox message is overwritten before it
// is consumed, all semaphores end EMPTY and no interleaving deadlocks.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "render/kernels/dataflow/pfwc_wsplit.h"

namespace {

using namespace pfwc_wsplit;

int failures = 0;
uint64_t trial_id = 0;
void fail(const char* what, uint64_t x = 0) {
    if (failures++ < 20)
        std::fprintf(stderr, "FAIL trial %llu: %s (%llu)\n", (unsigned long long)trial_id, what,
                     (unsigned long long)x);
}

constexpr uint32_t PAYLOAD = 0x00FFFFFFu;  // stand-in for vis_tile::PAYLOAD

struct Core {
    uint32_t n = 0, seg_base = 0, nb_real = 1;
    std::vector<uint32_t> vc;                    // visible gaussians per chunk
    std::vector<uint32_t> dep, aabb, tpg, rec;  // per visible gaussian j
    uint32_t m_total = 0;
};

struct Dram {
    uint32_t pages = 0;  // dep / offs / aabb pages, page index
    std::vector<uint32_t> dep, offs, aabb, pg_hits;
    std::vector<uint32_t> rec, rec_hits;  // record page index
    uint32_t cnt_m = 0, cnt_p = 0, cnt_hits = 0;
    void init(uint32_t npages, uint32_t nrec) {
        pages = npages;
        dep.assign(npages * PW, ~0u);
        offs.assign(npages * PW, ~0u);
        aabb.assign(npages * PW, ~0u);
        pg_hits.assign(npages, 0);
        rec.assign(nrec, ~0u);
        rec_hits.assign(nrec, 0);
    }
    void put_page(const volatile uint32_t* d, const volatile uint32_t* o, const volatile uint32_t* a,
                  uint32_t page) {
        if (page >= pages) return fail("page out of range", page);
        for (uint32_t i = 0; i < PW; i++) {
            dep[page * PW + i] = d[i];
            offs[page * PW + i] = o[i];
            aabb[page * PW + i] = a[i];
        }
        pg_hits[page]++;
    }
};

// writer_pfwc_fuse.cpp: one writer over all chunks.
void reference(const Core& c, Dram& out) {
    uint32_t page = c.seg_base / PW, slot = 0, pr = 0;
    uint32_t d[PW], o[PW], a[PW];
    for (uint32_t j = 0; j < c.m_total; j++) {
        d[slot] = c.dep[j];
        o[slot] = pr;
        a[slot] = c.aabb[j] & PAYLOAD;
        pr += c.tpg[j] & PAYLOAD;
        out.rec[c.seg_base + j] = c.rec[j];
        out.rec_hits[c.seg_base + j]++;
        if (++slot == PW) {
            out.put_page(d, o, a, page);
            slot = 0;
            page++;
        }
    }
    if (slot != 0) {
        for (uint32_t s = slot; s < PW; s++) {
            d[s] = 0;
            o[s] = pr;
            a[s] = 0;
        }
        out.put_page(d, o, a, page);
    }
    out.cnt_m = c.m_total;
    out.cnt_p = pr;
    out.cnt_hits++;
}

// The core's L1 shared by the two writers, compute and the reader.
struct Shared {
    uint32_t flag[NUM_SEMS] = {};
    uint32_t msg[NUM_SEMS][MSG_WORDS] = {};
    int32_t msg_id[NUM_SEMS];  // chunk the slot's message is for (-1 none)
    uint32_t out_q[2] = {0, 0}, comp_k = 0;  // output CB pages, compute's next chunk
    // Per RISC's reader: pages in its input CBs, next chunk, read in flight, any tiles at all.
    uint32_t in_q[2] = {0, 0}, rd_k[2] = {0, 0};
    bool rd_pend[2] = {false, false}, rd_on[2] = {false, true};
    Shared() {
        for (auto& i : msg_id) i = -1;
    }
};

constexpr uint32_t IN_DEPTH = 3, OUT_DEPTH = 2;

// Compute: chunk comp_k needs each reader's input page and room in its output set.
bool compute_step(const Core& c, Shared& sh) {
    if (sh.comp_k >= c.n || sh.out_q[sh.comp_k & 1u] >= OUT_DEPTH) return false;
    for (uint32_t r = 0; r < 2; r++)
        if (sh.rd_on[r] && sh.in_q[r] == 0) return false;
    for (uint32_t r = 0; r < 2; r++)
        if (sh.rd_on[r]) sh.in_q[r]--;
    sh.out_q[sh.comp_k & 1u]++;
    sh.comp_k++;
    return true;
}

// writer_pfwc_split.cpp rd_step on RISC r: push the landed chunk, issue the next one.
bool rd_step(const Core& c, Shared& sh, uint32_t r) {
    bool moved = false;
    if (sh.rd_pend[r]) {
        sh.in_q[r]++;  // reads land by the next poll
        if (sh.in_q[r] > IN_DEPTH) fail("input CB overflow", sh.in_q[r]);
        sh.rd_pend[r] = false;
        sh.rd_k[r]++;
        moved = true;
    }
    if (sh.rd_k[r] >= c.n || sh.in_q[r] >= IN_DEPTH) return moved;
    sh.rd_pend[r] = true;
    return true;
}

enum Pt { START, WAITOUT, CLASSIFY, PFX_RECV, PFX_SEND, REC, OPEN_RECV, TAIL, DRAIN, DONE };

struct Writer {
    uint32_t role = 0;
    uint32_t k = 0;
    Pt pt = START;
    // L1 staging: 7 pages (dep, offs, aabb, counts, head dep, offs, aabb) and
    // the record slots (one word per record here).
    uint32_t pgs[7][PW] = {};
    std::vector<uint32_t> stage;
    ChunkPages pg{};
    RecStage rs{};
    bool per_page = false;
    uint32_t nb = 1;
    uint32_t m0 = 0, pr = 0, vc = 0, pc = 0, jbase = 0, i = 0;

    void init(const Core& c, uint32_t r) {
        role = r;
        k = r;
        per_page = c.nb_real > NB_MAX;
        nb = per_page ? NB_MAX : c.nb_real;
        rs.init(nb);
        stage.assign(NB_MAX * RL, ~0u);
        pg.sd = pgs[0];
        pg.so = pgs[1];
        pg.sa = pgs[2];
        pg.hd = pgs[4];
        pg.ho = pgs[5];
        pg.ha = pgs[6];
    }

    // One step; false = blocked with nothing changed.
    bool step(const Core& c, Shared& sh, Dram& out, std::mt19937& rng) {
        auto poll = [&]() { return sh.rd_on[role] ? rd_step(c, sh, role) : false; };
        auto slot_of = [&](uint32_t j) { return slot_index(j); };
        auto flush_pg = [&](volatile uint32_t* d, volatile uint32_t* o, volatile uint32_t* a, uint32_t page) {
            out.put_page(d, o, a, page);
        };
        auto flush_rec = [&](uint32_t G0, uint32_t gs, uint32_t ge) {
            rs.writes(G0, gs, ge, per_page, [&](uint32_t s, uint32_t g, uint32_t n) {
                for (uint32_t x = 0; x < n; x++) {
                    const uint32_t p = per_page ? g : g + x * c.nb_real;  // one bank, consecutive rows
                    if (s + x >= stage.size() || p >= out.rec.size()) return fail("record write range", p);
                    if (!per_page && n > 1 && c.nb_real != nb) fail("bulk write with wrong bank count");
                    out.rec[p] = stage[s + x];
                    out.rec_hits[p]++;
                }
            });
        };
        switch (pt) {
            case START:
                if (k >= c.n) {
                    if (role == 0 && c.n == 0) {
                        out.cnt_m = 0;
                        out.cnt_p = 0;
                        out.cnt_hits++;
                    }
                    pt = DRAIN;
                    return true;
                }
                pt = WAITOUT;
                return true;
            case WAITOUT:
                // The set's front is chunk k: each set is popped in chunk order.
                if (sh.out_q[k & 1u] == 0) return poll();
                pt = CLASSIFY;
                return true;
            case CLASSIFY: {
                jbase = 0;
                for (uint32_t q = 0; q < k; q++) jbase += c.vc[q];
                vc = c.vc[k];
                pc = 0;
                for (uint32_t j = jbase; j < jbase + vc; j++) pc += c.tpg[j] & PAYLOAD;
                poll();
                pt = PFX_RECV;
                return true;
            }
            case PFX_RECV:
                m0 = 0;
                pr = 0;
                if (k != 0) {
                    const uint32_t s = slot_of(k);
                    if (sh.flag[s] == F_EMPTY) return poll();
                    // OPEN may already be there; it only follows when m % PW != 0.
                    if (sh.flag[s] != F_PREFIX && !(sh.flag[s] == F_OPEN && sh.msg[s][MSG_M] % PW != 0))
                        fail("PREFIX flag", sh.flag[s]);
                    if (sh.msg_id[s] != static_cast<int32_t>(k)) fail("PREFIX for another chunk", k);
                    m0 = sh.msg[s][MSG_M];
                    pr = sh.msg[s][MSG_PR];
                    if (m0 != jbase) fail("PREFIX m", m0);
                    if (m0 % PW == 0) {
                        sh.flag[s] = F_EMPTY;
                        sh.msg_id[s] = -1;
                    }
                }
                pt = PFX_SEND;
                return true;
            case PFX_SEND:
                if (k + 1 < c.n) {
                    const uint32_t s = slot_of(k + 1);
                    if (sh.flag[s] != F_EMPTY) return poll();
                    if (sh.msg_id[s] != -1) fail("PREFIX overwrites a live message", k + 1);
                    sh.msg_id[s] = static_cast<int32_t>(k + 1);
                    for (uint32_t w = 0; w < MSG_WORDS; w++) sh.msg[s][w] = 0xDEAD0000u + w;  // stale words
                    sh.msg[s][MSG_M] = m0 + vc;
                    sh.msg[s][MSG_PR] = pr + pc;
                    sh.flag[s] = F_PREFIX;
                }
                pg.begin(m0, c.seg_base / PW);
                rs.begin(c.seg_base + m0);
                i = 0;
                pt = REC;
                return true;
            case REC: {
                // A random run of records, then (the reader's RISC) a poll, as every 8 mask words.
                const uint32_t run = 1 + rng() % 48;
                for (uint32_t x = 0; x < run && i < vc; x++, i++) {
                    const uint32_t j = jbase + i;
                    pg.put(c.dep[j], pr, c.aabb[j] & PAYLOAD, flush_pg);
                    pr += c.tpg[j] & PAYLOAD;
                    stage[rs.slot()] = c.rec[j];
                    rs.next(flush_rec);
                }
                poll();
                if (i == vc) {
                    rs.end(flush_rec);
                    pt = OPEN_RECV;
                }
                return true;
            }
            case OPEN_RECV:
                if (pg.head_s != 0) {
                    const uint32_t s = slot_of(k);
                    if (sh.flag[s] != F_OPEN) return poll();
                    if (sh.msg_id[s] != static_cast<int32_t>(k)) fail("OPEN for another chunk", k);
                    pg.finish(sh.msg[s], flush_pg);
                    sh.flag[s] = F_EMPTY;
                    sh.msg_id[s] = -1;
                }
                pt = TAIL;
                return true;
            case TAIL:
                if (k + 1 < c.n) {
                    if (pg.slot != 0) {
                        const uint32_t s = slot_of(k + 1);
                        if (sh.flag[s] != F_PREFIX || sh.msg_id[s] != static_cast<int32_t>(k + 1))
                            fail("OPEN without its PREFIX", k + 1);
                        pg.export_open(sh.msg[s]);
                        sh.flag[s] = F_OPEN;
                    }
                } else {
                    pg.close(pr, flush_pg);
                    if (m0 + vc != c.m_total) fail("last chunk m", m0 + vc);
                    out.cnt_m = m0 + vc;
                    out.cnt_p = pr;
                    out.cnt_hits++;
                }
                if (sh.out_q[k & 1u] == 0) fail("pop of an empty output set", k);
                sh.out_q[k & 1u]--;
                k += 2;
                pt = START;
                return true;
            case DRAIN:
                if (sh.rd_on[role] && sh.rd_k[role] < c.n) return poll();
                pt = DONE;
                return true;
            case DONE:
                return false;
        }
        return false;
    }
};

uint32_t pick_vc(std::mt19937& rng) {
    switch (rng() % 8) {
        case 0: return 0;
        case 1: return 1 + rng() % 3;
        case 2: return 15 + rng() % 3;
        case 3: return 1024;
        case 4: return PW * (1 + rng() % 8);
        default: return rng() % 1025;
    }
}

// rd_mode: 0 NCRISC reads, 1 BRISC reads, 2 both read a subset.
void run_trial(const Core& c, std::mt19937& rng, uint32_t rd_mode) {
    const uint32_t npages = c.seg_base / PW + (c.m_total + PW - 1) / PW + 4;
    const uint32_t nrec = c.seg_base + c.m_total + 4 * NB_MAX * RL;
    Dram ref, got;
    ref.init(npages, nrec);
    got.init(npages, nrec);
    reference(c, ref);

    Shared sh;
    sh.rd_on[0] = rd_mode != 0;
    sh.rd_on[1] = rd_mode != 1;
    Writer w[2];
    w[0].init(c, 0);
    w[1].init(c, 1);
    // Agents: 0 = BRISC, 1 = NCRISC, 2 = compute. A full round without any
    // change while not finished is a deadlock.
    uint64_t steps = 0;
    for (;;) {
        bool any = false;
        const uint32_t order = rng() % 6;
        static const uint8_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        for (uint32_t a : perms[order]) {
            // Uneven speeds: an agent may take several steps in a row.
            const uint32_t reps = 1 + rng() % 4;
            for (uint32_t r = 0; r < reps; r++) {
                const bool moved = a == 2 ? compute_step(c, sh) : w[a].step(c, sh, got, rng);
                any |= moved;
                if (!moved) break;
            }
        }
        if (w[0].pt == DONE && w[1].pt == DONE && sh.comp_k == c.n) break;
        if (!any) return fail("deadlock", (uint64_t(w[0].pt) << 32) | w[1].pt);
        if (++steps > 10000000ull) return fail("no termination");
    }

    for (uint32_t s = 0; s < NUM_SEMS; s++)
        if (sh.flag[s] != F_EMPTY || sh.msg_id[s] != -1) fail("semaphore left set", s);
    if (sh.out_q[0] != 0 || sh.out_q[1] != 0) fail("output CB pages left");
    for (uint32_t r = 0; r < 2; r++)
        if (sh.in_q[r] != 0 || sh.rd_pend[r] || sh.rd_k[r] != (sh.rd_on[r] ? c.n : 0u))
            fail("input CB pages left", r);
    for (uint32_t p = 0; p < npages; p++) {
        if (got.pg_hits[p] != ref.pg_hits[p]) fail("page write count", p);
        for (uint32_t x = 0; x < PW; x++)
            if (got.dep[p * PW + x] != ref.dep[p * PW + x] || got.offs[p * PW + x] != ref.offs[p * PW + x] ||
                got.aabb[p * PW + x] != ref.aabb[p * PW + x])
                return fail("page bytes", p);
    }
    for (uint32_t g = 0; g < nrec; g++)
        if (got.rec_hits[g] != ref.rec_hits[g] || got.rec[g] != ref.rec[g]) return fail("record", g);
    if (got.cnt_hits != 1 || got.cnt_m != ref.cnt_m || got.cnt_p != ref.cnt_p) fail("counts", got.cnt_hits);
}

Core make_core(std::mt19937& rng, uint32_t n) {
    Core c;
    c.n = n;
    static const uint32_t bases[] = {0, 1024, 5 * 1024, 37 * 1024};
    c.seg_base = bases[rng() % 4];
    c.nb_real = 1 + rng() % 12;
    for (uint32_t k = 0; k < n; k++) c.vc.push_back(pick_vc(rng));
    for (uint32_t v : c.vc) c.m_total += v;
    for (uint32_t j = 0; j < c.m_total; j++) {
        c.dep.push_back(rng());
        c.aabb.push_back(rng());
        c.tpg.push_back((rng() % 4 == 0 ? 0u : rng() % 40) | (rng() % 4) << 24);  // payload + tag bits
        c.rec.push_back(0x5EC00000u + j);
    }
    return c;
}

}  // namespace

int main() {
    std::mt19937 rng(207);
    // Mailbox slots: the R in-flight messages to each RISC never share a slot.
    for (uint32_t j = 1; j < 64; j++) {
        if (slot_index(j) / R != owner(j)) fail("slot direction", j);
        for (uint32_t i = j + 2; i < j + 2 * R; i += 2)
            if (slot_index(i) == slot_index(j)) fail("slot reused too early", j);
    }
    // Small CB / staging layout guards.
    static_assert(odd_cb(9) == 41 && odd_cb(16) == 48 && odd_cb(35) == 49 && odd_cb(36) == 50, "odd CBs");
    static_assert(CB_STG_ODD == 51 && CB_MBX == 52, "staging / mailbox CBs");
    static_assert(MSG_AABB + PW <= MSG_WORDS, "OPEN message fits");

    uint64_t trials = 0;
    for (uint32_t n = 0; n <= 13; n++)
        for (uint32_t t = 0; t < 400; t++, trials++) {
            trial_id = trials;
            run_trial(make_core(rng, n), rng, t % 3);
        }
    // Larger cores, as on bicycle (up to ~50 chunks per core).
    for (uint32_t t = 0; t < 120; t++, trials++) {
        trial_id = trials;
        run_trial(make_core(rng, 30 + rng() % 30), rng, t % 3);
    }
    std::printf("%llu trials, %d failures\n", (unsigned long long)trials, failures);
    return failures == 0 ? 0 : 1;
}
