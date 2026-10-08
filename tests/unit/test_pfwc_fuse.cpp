// Host checks for lever B (task #125, GSPLAT_TT_PFWC_FUSE=1). Standalone:
//
//   CXXFLAGS="-Irender/kernels/dataflow -ffp-contract=off" \
//       tests/unit/run_cpp.sh tests/unit/test_pfwc_fuse.cpp
//
// Random visibility / pair counts / tile rectangles per gaussian. The legacy
// chain (strided-deal compaction in SeqMap order, global exclusive offsets,
// gaussian-major K2 over contiguous page ranges) is the reference. The fused
// chain models writer_pfwc_fuse.cpp (per-core segments at seg_base, local
// offsets, counts page) and runs pfwc_fuse.h's seg_table / k2_range /
// emit_pairs per (K2 core, mover) exactly as tile_assign_scatter_seg.cpp does.
// Checks: segments never overlap; M, P, P_pub, overflow; every pair page is
// written once; the pair list equals the legacy one after mapping the storage
// index back to the dense compact index (gid), tid identical, padding (0, 0);
// depth by gid equals the legacy depth by dense gid.
//
// Task #384 (host model, no kernel): the pfwc writers emit the (gid, tid) pairs
// themselves into per-core fixed-capacity segments (core c at c * cap + lofs),
// plus per-unit per-tile count rows, and the sort skips K2. check_writer_pairs:
// a unit is a contiguous run of a core's chunks (chunk-order split into U parts;
// chunk k is written by writer k & 1, both writers count into the unit's row);
// the sort prefix runs over units in global (core, unit) order and the movers
// process whole units in any order. Checks: the final (tile, depth, gid) order
// equals the legacy K2 + sort order exactly, ties included; per-writer-parity
// units change it exactly where same-tile equal-depth pairs from different-parity
// chunks of one core meet; overflow sets a flag, grow + rerun gives the same
// result. Prints the per-writer pair / page / count work.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "pfwc_fuse.h"
#include "render/host/sort_mover_split.h"

namespace {

int failures = 0;
void fail(const char* what, uint64_t a, uint64_t b) {
    if (failures++ < 20)
        std::fprintf(stderr, "FAIL %s: %llu %llu\n", what, static_cast<unsigned long long>(a),
                     static_cast<unsigned long long>(b));
}

struct Scene {
    uint32_t N = 0, num_tiles = 0, tiles_x = 0;
    std::vector<uint8_t> vis;
    std::vector<uint32_t> pairs, box, depth;
};

Scene make_scene(std::mt19937& rng, uint32_t num_tiles, double p_vis) {
    Scene s;
    s.num_tiles = num_tiles;
    s.N = num_tiles * vis_tile::TILE_ELEMS - (rng() % vis_tile::TILE_ELEMS);
    s.tiles_x = 20 + rng() % 60;
    const uint32_t tiles_y = 10 + rng() % 40;
    const uint32_t n = num_tiles * vis_tile::TILE_ELEMS;
    s.vis.assign(n, 0);
    s.pairs.assign(n, 0);
    s.box.assign(n, 0);
    s.depth.assign(n, 0);
    std::uniform_real_distribution<double> U(0, 1);
    // Some tiles fully empty, some fully visible.
    std::vector<double> tile_p(num_tiles);
    for (auto& tp : tile_p) {
        const uint32_t r = rng() % 8;
        tp = r == 0 ? 0.0 : (r == 1 ? 1.0 : p_vis);
    }
    for (uint32_t i = 0; i < s.N; i++) {
        if (U(rng) >= tile_p[i / vis_tile::TILE_ELEMS]) continue;
        s.vis[i] = 1;
        const uint32_t w = 1 + rng() % 4, h = rng() % 4;  // h == 0: no pairs
        const uint32_t mx = rng() % (s.tiles_x - w + 1);
        const uint32_t my = rng() % (tiles_y - 3);
        s.box[i] = vis_tile::aabb_pack(mx, my, w);
        s.pairs[i] = w * h;
        s.depth[i] = rng();
    }
    return s;
}

struct Pairs {
    uint32_t M = 0, P = 0;
    std::vector<uint32_t> src;  // dense gid -> scene index
    std::vector<uint32_t> gid, tid;
};

// Legacy compaction + K2 (tile_assign_scatter.cpp, TA_K2_AABB).
Pairs legacy(const Scene& s, uint32_t C) {
    Pairs r;
    vis_tile::SeqMap sm;
    sm.init(s.num_tiles, C);
    for (uint32_t c = 0; c < C; c++)
        for (uint32_t k = 0; k < sm.count(c); k++) {
            const uint32_t t = c + k * C;
            for (uint32_t il = 0; il < vis_tile::TILE_ELEMS; il++)
                if (s.vis[t * vis_tile::TILE_ELEMS + il]) r.src.push_back(t * vis_tile::TILE_ELEMS + il);
        }
    r.M = static_cast<uint32_t>(r.src.size());
    for (uint32_t g = 0; g < r.M; g++) {
        const uint32_t i = r.src[g], b = s.box[i];
        const uint32_t w = vis_tile::aabb_w(b);
        for (uint32_t l = 0; l < s.pairs[i]; l++) {
            r.gid.push_back(g);
            r.tid.push_back((vis_tile::aabb_min_y(b) + l / w) * s.tiles_x + vis_tile::aabb_min_x(b) +
                            l % w);
        }
    }
    r.P = static_cast<uint32_t>(r.gid.size());
    return r;
}

struct Fused {
    std::vector<uint32_t> lofs, box, depth, cnt;  // storage-indexed / counts pages
    std::vector<uint32_t> s2src;                  // storage index -> scene index
};

// writer_pfwc_fuse.cpp: core c, tiles c + k * C, lanes in mask bit order.
Fused fused_writer(const Scene& s, uint32_t C) {
    Fused f;
    const uint32_t n = s.num_tiles * vis_tile::TILE_ELEMS;
    f.lofs.assign(n, 0xDEADBEEFu);
    f.box.assign(n, 0xDEADBEEFu);
    f.depth.assign(n, 0xDEADBEEFu);
    f.s2src.assign(n, 0xFFFFFFFFu);
    f.cnt.assign(static_cast<size_t>(C) * pfwc_fuse::PAGE_WORDS, 0);
    vis_tile::SeqMap sm;
    sm.init(s.num_tiles, C);
    for (uint32_t c = 0; c < C; c++) {
        const uint32_t sb = pfwc_fuse::seg_base(s.num_tiles, C, c);
        uint32_t m = 0, pr = 0;
        for (uint32_t k = 0; k < sm.count(c); k++) {
            const uint32_t t = c + k * C;
            for (uint32_t il = 0; il < vis_tile::TILE_ELEMS; il++) {
                const uint32_t i = t * vis_tile::TILE_ELEMS + il;
                if (!s.vis[i]) continue;
                const uint32_t st = sb + m;
                if (f.s2src[st] != 0xFFFFFFFFu) fail("storage written twice", st, c);
                f.s2src[st] = i;
                f.lofs[st] = pr;
                f.box[st] = s.box[i];
                f.depth[st] = s.depth[i];
                pr += s.pairs[i];
                m++;
            }
        }
        if (c + 1 < C && sb + m > pfwc_fuse::seg_base(s.num_tiles, C, c + 1))
            fail("segment overlaps the next", c, m);
        if (sb % vis_tile::TILE_ELEMS) fail("segment not page aligned", c, sb);
        f.cnt[c * pfwc_fuse::PAGE_WORDS + pfwc_fuse::T_M] = m;
        f.cnt[c * pfwc_fuse::PAGE_WORDS + pfwc_fuse::T_P] = pr;
    }
    return f;
}

// Task #170: emit_pairs_diet's Io as the device behaves at its worst. An issued
// read poisons its ring slot and lands only at wait_reads(); a write copies its
// staging page out only at writes_flushed() (or the kernel's final barrier), so
// reading a slot before its wait or restaging a page before its flush changes
// the output.
struct ModelIo {
    const std::vector<uint32_t>* lofs = nullptr;
    const std::vector<uint32_t>* box = nullptr;
    std::vector<uint32_t>* gid = nullptr;
    std::vector<uint32_t>* tid = nullptr;
    std::vector<uint32_t>* written = nullptr;  // per pair page
    uint32_t ra_l[pfwc_fuse::RA_SLOTS * 16], ra_b[pfwc_fuse::RA_SLOTS * 16];
    uint32_t st_g[pfwc_fuse::OUT_SLOTS * 16], st_t[pfwc_fuse::OUT_SLOTS * 16];
    struct Rd {
        uint32_t page, slot;
        bool is_box;
    };
    struct Wr {
        uint32_t page, o;
    };
    std::vector<Rd> rd;
    std::vector<Wr> wr;
    uint32_t reads = 0, waits = 0, first_issue_wait = 0;
    bool streaming = false;

    void issue_lofs(uint32_t page, uint32_t slot) { push(page, slot, false); }
    void issue(uint32_t page, uint32_t slot) {
        if (!streaming) first_issue_wait = waits;  // search rounds before the stream
        streaming = true;
        push(page, slot, false);
        push(page, slot, true);
    }
    void push(uint32_t page, uint32_t slot, bool is_box) {
        if (slot >= pfwc_fuse::RA_SLOTS) fail("ring slot", slot, pfwc_fuse::RA_SLOTS);
        if ((page + 1) * 16 > lofs->size()) fail("read past the stream", page, lofs->size() / 16);
        uint32_t* d = (is_box ? ra_b : ra_l) + (slot % pfwc_fuse::RA_SLOTS) * 16;
        for (uint32_t w = 0; w < 16; w++) d[w] = 0xA5A5A5A5u;
        rd.push_back({page, slot % pfwc_fuse::RA_SLOTS, is_box});
        reads++;
    }
    void wait_reads() {
        for (const Rd& r : rd) {
            const std::vector<uint32_t>& src = r.is_box ? *box : *lofs;
            uint32_t* d = (r.is_box ? ra_b : ra_l) + r.slot * 16;
            for (uint32_t w = 0; w < 16; w++) d[w] = src[r.page * 16 + w];
        }
        rd.clear();
        waits++;
    }
    const uint32_t* lofs_slot(uint32_t s) const { return ra_l + s * 16; }
    const uint32_t* box_slot(uint32_t s) const { return ra_b + s * 16; }
    uint32_t* gid_slot(uint32_t o) { return st_g + o * 16; }
    uint32_t* tid_slot(uint32_t o) { return st_t + o * 16; }
    void write_page(uint32_t page, uint32_t o) {
        if (o >= pfwc_fuse::OUT_SLOTS) fail("staging slot", o, pfwc_fuse::OUT_SLOTS);
        for (const Wr& x : wr)
            if (x.o == o) fail("staging page rewritten before its flush", o, page);
        wr.push_back({page, o % pfwc_fuse::OUT_SLOTS});
    }
    void writes_flushed() {
        for (const Wr& x : wr) {
            if (x.page >= written->size()) {
                fail("write past the pair pages", x.page, written->size());
                continue;
            }
            (*written)[x.page]++;
            for (uint32_t w = 0; w < 16; w++) {
                (*gid)[x.page * 16 + w] = st_g[x.o * 16 + w];
                (*tid)[x.page * 16 + w] = st_t[x.o * 16 + w];
            }
        }
        wr.clear();
    }
};

// emit_pairs_diet per (K2 core, mover) must give emit_pairs' pages (gid, tid)
// and, with COUNT, the per-tile pair counts of its range.
uint32_t max_search_waits = 0;
// speed (2 K entries, task #170 fold): ranges from k2_range_speed, which must
// equal the one-launch sort's sort_split::speed_bounds.
template <bool COUNT>
void check_diet(const Scene& s, const Fused& f, const std::vector<uint32_t>& tab, uint32_t C,
                uint32_t K, uint32_t dual, uint32_t permille, uint32_t P_pub,
                const std::vector<uint32_t>& gid, const std::vector<uint32_t>& tid,
                const std::vector<uint32_t>* speed = nullptr) {
    std::vector<uint32_t> acc, sb;
    if (speed != nullptr) {
        acc.assign(speed->size() + 1u, 0u);
        for (std::size_t k = 0; k < speed->size(); k++) acc[k + 1] = acc[k] + (*speed)[k];
        sb = gsplat_tt::sort_split::speed_bounds((P_pub + 15) / 16, *speed);
    }
    const uint32_t pages = (P_pub + 15) / 16;
    std::vector<uint32_t> dg(pages * 16, 0xFFFFFFFFu), dt(pages * 16, 0xFFFFFFFFu);
    std::vector<uint32_t> written(pages, 0);
    std::vector<uint32_t> cnt(1u << 20), ref_cnt(1u << 20);
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t mv = 0; mv < 2; mv++) {
            uint32_t pg0 = 0, npg = 0;
            if (speed == nullptr) {
                pfwc_fuse::k2_range(P_pub, K, k, mv, dual, permille, &pg0, &npg);
            } else {
                pfwc_fuse::k2_range_speed(P_pub, acc[2 * k + mv], acc[2 * k + mv + 1], acc.back(),
                                          &pg0, &npg);
                if (pg0 != sb[2 * k + mv] || pg0 + npg != sb[2 * k + mv + 1])
                    fail("speed range != speed_bounds", 2 * k + mv, pg0);
            }
            ModelIo io;
            io.lofs = &f.lofs;
            io.box = &f.box;
            io.gid = &dg;
            io.tid = &dt;
            io.written = &written;
            std::fill(cnt.begin(), cnt.end(), 0u);
            pfwc_fuse::emit_pairs_diet<COUNT>(tab.data(), C, P_pub, s.tiles_x, pg0, npg, io,
                                              COUNT ? cnt.data() : nullptr);
            if (!io.rd.empty()) fail("reads in flight at the end", k, io.rd.size());
            io.writes_flushed();  // the kernel's final write barrier
            if (io.streaming && io.first_issue_wait > max_search_waits)
                max_search_waits = io.first_issue_wait;
            if (!COUNT) continue;
            std::fill(ref_cnt.begin(), ref_cnt.end(), 0u);
            for (uint32_t p = pg0 * 16; p < (pg0 + npg) * 16 && p < P_pub; p++) ref_cnt[tid[p]]++;
            for (uint32_t t = 0; t < cnt.size(); t++)
                if (cnt[t] != ref_cnt[t]) {
                    fail("diet tile count", t, cnt[t]);
                    break;
                }
        }
    for (uint32_t pg = 0; pg < pages; pg++)
        if (written[pg] != 1) fail("diet pair page written", pg, written[pg]);
    for (uint32_t p = 0; p < pages * 16; p++) {
        if (dg[p] != gid[p]) fail("diet gid", p, dg[p]);
        if (dt[p] != tid[p]) fail("diet tid", p, dt[p]);
    }
}

// tile_assign_scatter_seg.cpp over all (K2 core, mover) slots.
void check_k2(const Scene& s, const Pairs& ref, const Fused& f, uint32_t C, uint32_t K,
              uint32_t dual, uint32_t permille, uint32_t p_cap) {
    std::vector<uint32_t> tab = f.cnt;
    uint32_t M = 0, P = 0;
    pfwc_fuse::seg_table(tab.data(), C, s.num_tiles, &M, &P);
    if (M != ref.M) fail("M", M, ref.M);
    if (P != ref.P) fail("P", P, ref.P);
    const uint32_t P_pub = P < p_cap ? P : p_cap;
    const uint32_t pages = (P_pub + 15) / 16;
    std::vector<uint32_t> gid(pages * 16, 0xFFFFFFFFu), tid(pages * 16, 0xFFFFFFFFu);
    std::vector<uint32_t> written(pages, 0);
    uint32_t cover = 0;
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t mv = 0; mv < 2; mv++) {
            uint32_t pg0 = 0, npg = 0;
            pfwc_fuse::k2_range(P_pub, K, k, mv, dual, permille, &pg0, &npg);
            if (npg == 0) continue;
            if (pg0 != cover) fail("k2 ranges not contiguous", pg0, cover);
            cover = pg0 + npg;
            for (uint32_t pg = pg0; pg < pg0 + npg; pg++) written[pg]++;
            pfwc_fuse::emit_pairs(
                tab.data(), C, P_pub, s.tiles_x, pg0 * 16, (pg0 + npg) * 16,
                [&](uint32_t st) { return f.lofs[st]; }, [&](uint32_t st) { return f.box[st]; },
                [&](uint32_t p, uint32_t g, uint32_t t) {
                    gid[p] = g;
                    tid[p] = t;
                });
        }
    if (cover != pages) fail("k2 ranges do not cover the pages", cover, pages);
    for (uint32_t pg = 0; pg < pages; pg++)
        if (written[pg] != 1) fail("pair page written", pg, written[pg]);
    // Dense gid of a storage index: position of its scene index in ref.src.
    std::vector<uint32_t> src2g(s.num_tiles * vis_tile::TILE_ELEMS, 0xFFFFFFFFu);
    for (uint32_t g = 0; g < ref.M; g++) src2g[ref.src[g]] = g;
    for (uint32_t p = 0; p < pages * 16; p++) {
        if (p >= P_pub) {
            if (gid[p] != 0 || tid[p] != 0) fail("pad pair", p, gid[p]);
            continue;
        }
        const uint32_t st = gid[p];
        if (st >= f.s2src.size() || f.s2src[st] == 0xFFFFFFFFu) {
            fail("gid is not a written storage index", p, st);
            continue;
        }
        if (src2g[f.s2src[st]] != ref.gid[p]) fail("pair gid", p, ref.gid[p]);
        if (tid[p] != ref.tid[p]) fail("pair tid", p, ref.tid[p]);
        if (f.depth[st] != s.depth[ref.src[ref.gid[p]]]) fail("depth by gid", p, st);
    }
    check_diet<true>(s, f, tab, C, K, dual, permille, P_pub, gid, tid);
    check_diet<false>(s, f, tab, C, K, dual, permille, P_pub, gid, tid);
    // Speed-proportional ranges (p150 table span 700..1100; some cores slow).
    std::vector<uint32_t> speed(2u * K);
    for (uint32_t k = 0; k < 2u * K; k++) speed[k] = 650u + (k * 2654435761u >> 7) % 500u;
    check_diet<true>(s, f, tab, C, K, dual, permille, P_pub, gid, tid, &speed);
}


// ---- task #384: writers emit pairs (host model) --------------------------------
struct WriterPairs {
    uint32_t cap = 0, nunits = 0, T = 0;
    std::vector<uint32_t> gid, tid;            // C * cap slots (storage gid, tile)
    std::vector<uint32_t> cnt;                 // nunits * T, both writers summed
    std::vector<std::vector<uint32_t>> upos;   // unit -> pair slots in storage order
    std::vector<uint8_t> ovf;                  // per core
    std::vector<uint32_t> pc;                  // per core pairs
    uint64_t w_pairs[2] = {0, 0}, w_pages[2] = {0, 0}, max_core_writer_pairs = 0;
};

// parity_units: unit = (core, writer k & 1) instead of a chunk-order run.
WriterPairs writer_pairs(const Scene& s, uint32_t C, uint32_t U, uint32_t cap, uint32_t T,
                         bool parity_units) {
    WriterPairs w;
    w.cap = cap;
    w.T = T;
    w.nunits = C * (parity_units ? 2u : U);
    const uint32_t upc = parity_units ? 2u : U;
    w.gid.assign(static_cast<size_t>(C) * cap, 0xFFFFFFFFu);
    w.tid.assign(static_cast<size_t>(C) * cap, 0xFFFFFFFFu);
    w.cnt.assign(static_cast<size_t>(w.nunits) * T, 0);
    w.upos.assign(w.nunits, {});
    w.ovf.assign(C, 0);
    w.pc.assign(C, 0);
    vis_tile::SeqMap sm;
    sm.init(s.num_tiles, C);
    for (uint32_t c = 0; c < C; c++) {
        const uint32_t sb = pfwc_fuse::seg_base(s.num_tiles, C, c);
        uint32_t m = 0, pr = 0;
        uint64_t wp[2] = {0, 0};
        for (uint32_t k = 0; k < sm.count(c); k++) {
            const uint32_t wr = k & 1u;  // pfwc_wsplit owner(k)
            const uint32_t u = c * upc + (parity_units ? wr : (k * U) / sm.count(c));
            const uint32_t t = c + k * C;
            for (uint32_t il = 0; il < vis_tile::TILE_ELEMS; il++) {
                const uint32_t i = t * vis_tile::TILE_ELEMS + il;
                if (!s.vis[i]) continue;
                const uint32_t st = sb + m++;
                const uint32_t b = s.box[i], bw = vis_tile::aabb_w(b);
                // AABB walk: row-major over the rectangle, as K2 does.
                for (uint32_t l = 0; l < s.pairs[i]; l++, pr++) {
                    const uint32_t tt =
                        (vis_tile::aabb_min_y(b) + l / bw) * s.tiles_x + vis_tile::aabb_min_x(b) + l % bw;
                    if (tt >= T) fail("tile out of range", tt, T);
                    wp[wr]++;
                    if (pr >= cap) {  // segment full: flag, keep counting for the host
                        w.ovf[c] = 1;
                        continue;
                    }
                    const size_t slot = static_cast<size_t>(c) * cap + pr;
                    w.gid[slot] = st;
                    w.tid[slot] = tt;
                    w.cnt[static_cast<size_t>(u) * T + tt]++;
                    w.upos[u].push_back(static_cast<uint32_t>(slot));
                }
            }
        }
        w.pc[c] = pr;
        for (uint32_t r = 0; r < 2; r++) {
            w.w_pairs[r] += wp[r];
            w.w_pages[r] += 2 * ((wp[r] + pfwc_fuse::PAGE_WORDS - 1) / pfwc_fuse::PAGE_WORDS);
            w.max_core_writer_pairs = std::max<uint64_t>(w.max_core_writer_pairs, wp[r]);
        }
    }
    return w;
}

struct Sorted {
    std::vector<uint32_t> tid, depth, g;  // g: dense gid
};

// One-launch sort over writer units: prefix in (core, unit) order, movers take
// whole units in `order`, bucket fill, then a stable depth sort per tile bucket.
Sorted sort_units(const WriterPairs& w, const Fused& f, const std::vector<uint32_t>& src2g,
                  const std::vector<uint32_t>& order) {
    const uint32_t T = w.T;
    std::vector<uint64_t> base(static_cast<size_t>(w.nunits) * T);
    uint64_t run = 0;
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t u = 0; u < w.nunits; u++) {
            base[static_cast<size_t>(u) * T + t] = run;
            run += w.cnt[static_cast<size_t>(u) * T + t];
        }
    std::vector<uint32_t> bstart(T + 1, 0);
    for (uint32_t t = 0; t < T; t++)
        bstart[t] = static_cast<uint32_t>(base[t]);  // unit 0, tile t
    bstart[T] = static_cast<uint32_t>(run);
    std::vector<uint32_t> out_slot(run, 0xFFFFFFFFu);
    for (uint32_t u : order) {
        std::vector<uint64_t> cur(base.begin() + static_cast<size_t>(u) * T,
                                  base.begin() + static_cast<size_t>(u + 1) * T);
        for (uint32_t slot : w.upos[u]) {
            const uint64_t pos = cur[w.tid[slot]]++;
            if (out_slot[pos] != 0xFFFFFFFFu) fail("sort slot written twice", pos, u);
            out_slot[pos] = slot;
        }
    }
    Sorted r;
    for (uint32_t t = 0; t < T; t++) {
        std::vector<uint32_t> b(out_slot.begin() + bstart[t], out_slot.begin() + bstart[t + 1]);
        std::stable_sort(b.begin(), b.end(), [&](uint32_t a, uint32_t c) {
            return f.depth[w.gid[a]] < f.depth[w.gid[c]];
        });
        for (uint32_t slot : b) {
            r.tid.push_back(w.tid[slot]);
            r.depth.push_back(f.depth[w.gid[slot]]);
            r.g.push_back(src2g[f.s2src[w.gid[slot]]]);
        }
    }
    return r;
}

uint64_t wp_cases = 0, wp_parity_diff_cases = 0, wp_parity_diff_pairs = 0, wp_grow = 0;
uint64_t wp_pairs[2] = {0, 0}, wp_pages[2] = {0, 0};

void check_writer_pairs(const Scene& s, const Pairs& ref, const Fused& f, uint32_t C,
                        std::mt19937& rng) {
    uint32_t T = 1;
    for (uint32_t t : ref.tid) T = std::max(T, t + 1);
    // Reference: legacy K2 order, then the sort (tile buckets in pair order,
    // stable by depth).
    std::vector<uint32_t> idx(ref.P);
    std::iota(idx.begin(), idx.end(), 0u);
    std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        if (ref.tid[a] != ref.tid[b]) return ref.tid[a] < ref.tid[b];
        return s.depth[ref.src[ref.gid[a]]] < s.depth[ref.src[ref.gid[b]]];
    });
    std::vector<uint32_t> src2g(s.num_tiles * vis_tile::TILE_ELEMS, 0xFFFFFFFFu);
    for (uint32_t g = 0; g < ref.M; g++) src2g[ref.src[g]] = g;
    auto same = [&](const Sorted& r, uint64_t* ndiff) {
        if (r.g.size() != ref.P) {
            if (!ndiff) fail("sorted size", r.g.size(), ref.P);
            return false;
        }
        uint64_t d = 0;
        for (uint32_t i = 0; i < ref.P; i++) {
            const uint32_t p = idx[i];
            if (r.tid[i] != ref.tid[p] || r.g[i] != ref.gid[p]) {
                if (r.tid[i] != ref.tid[p] || r.depth[i] != s.depth[ref.src[ref.gid[p]]])
                    fail("order differs outside a (tile, depth) tie", i, r.g[i]);
                d++;
            }
        }
        if (ndiff) *ndiff = d;
        else if (d) fail("sorted order differs (ties)", d, ref.P);
        return d == 0;
    };
    uint32_t pmax = 0, psum = 0;
    for (uint32_t c = 0; c < C; c++) psum += f.cnt[c * pfwc_fuse::PAGE_WORDS + pfwc_fuse::T_P];
    for (uint32_t c = 0; c < C; c++)
        pmax = std::max(pmax, f.cnt[c * pfwc_fuse::PAGE_WORDS + pfwc_fuse::T_P]);
    for (uint32_t U : {1u, 2u, 4u, 8u}) {
        // Capacity: start at the mean per-core pairs (page rounded); on overflow the
        // host grows it to the reported max and reruns (pair-overflow-t213 style).
        uint32_t cap = ((psum / C + 15) / 16) * 16;
        WriterPairs w = writer_pairs(s, C, U, cap, T, false);
        bool any = false;
        uint32_t need = 0;
        for (uint32_t c = 0; c < C; c++) {
            if (w.ovf[c] != (w.pc[c] > cap)) fail("overflow flag", c, w.pc[c]);
            any |= w.ovf[c] != 0;
            need = std::max(need, w.pc[c]);
        }
        if (need != pmax) fail("reported per-core max pairs", need, pmax);
        if (any) {
            wp_grow++;
            cap = ((need + 15) / 16) * 16;
            w = writer_pairs(s, C, U, cap, T, false);
            for (uint32_t c = 0; c < C; c++)
                if (w.ovf[c]) fail("overflow after grow", c, w.pc[c]);
        }
        // Movers take whole units in any order: canonical, reversed, shuffled.
        std::vector<uint32_t> order(w.nunits);
        std::iota(order.begin(), order.end(), 0u);
        same(sort_units(w, f, src2g, order), nullptr);
        std::shuffle(order.begin(), order.end(), rng);
        same(sort_units(w, f, src2g, order), nullptr);
        if (U == 2) {
            for (uint32_t r = 0; r < 2; r++) {
                wp_pairs[r] += w.w_pairs[r];
                wp_pages[r] += w.w_pages[r];
            }
        }
        wp_cases++;
    }
    // Per-writer-parity units: rows (core, BRISC) then (core, NCRISC).
    const WriterPairs wpar = writer_pairs(s, C, 2, ((pmax + 15) / 16) * 16 + 16, T, true);
    std::vector<uint32_t> order(wpar.nunits);
    std::iota(order.begin(), order.end(), 0u);
    uint64_t d = 0;
    same(sort_units(wpar, f, src2g, order), &d);
    if (d) {
        wp_parity_diff_cases++;
        wp_parity_diff_pairs += d;
    }
}
}  // namespace

int main() {
    std::mt19937 rng(125);
    const uint32_t cores[] = {1, 3, 13, 110};
    int cases = 0;
    for (int trial = 0; trial < 40; trial++) {
        const uint32_t num_tiles = 1 + rng() % 300;  // includes num_tiles < cores
        const double p_vis = (trial % 4 == 0) ? 0.002 : 0.05 + 0.5 * (rng() % 100) / 100.0;
        const Scene s = make_scene(rng, num_tiles, p_vis);
        for (uint32_t C : cores) {
            const Pairs ref = legacy(s, C);
            const Fused f = fused_writer(s, C);
            const uint32_t K = (trial & 1) ? C : 1 + rng() % 110;
            const uint32_t permille = 300 + rng() % 500;
            // Full capacity, and a clamp below P (overflow) on some trials.
            const uint32_t p_cap = (trial % 5 == 3 && ref.P > 1) ? ref.P / 2 + 1 : ref.P + 64;
            check_k2(s, ref, f, C, K, trial % 3 != 0, permille, p_cap);
            cases++;
            if (trial % 2 == 0 && num_tiles <= 120) {
                // Forced depth ties (t384): few distinct depth keys.
                Scene st = s;
                if (trial % 4 == 0)
                    for (auto& d : st.depth) d &= 3u;
                check_writer_pairs(st, ref, fused_writer(st, C), C, rng);
            }
        }
    }
    // Empty scene: M = P = 0, no pages.
    {
        std::mt19937 r2(1);
        const Scene s = make_scene(r2, 5, 0.0);
        Scene e = s;
        std::fill(e.vis.begin(), e.vis.end(), 0);
        check_k2(e, legacy(e, 110), fused_writer(e, 110), 110, 110, 1, 500, 1000);
        cases++;
    }
    if (failures) {
        std::fprintf(stderr, "test_pfwc_fuse: %d failures in %d cases\n", failures, cases);
        return 1;
    }
    std::printf("test_pfwc_fuse: %d cases OK (diet start search: <= %u read rounds)\n", cases,
                max_search_waits);
    std::printf("t384 writer pairs: %llu unit cases md5-identical (U = 1, 2, 4, 8; canonical + shuffled mover order),"
                " %llu overflow grow+rerun; parity units reorder ties in %llu scenes (%llu pairs);"
                " U=2 writer pairs B %llu N %llu, pair pages B %llu N %llu\n",
                (unsigned long long)wp_cases, (unsigned long long)wp_grow,
                (unsigned long long)wp_parity_diff_cases, (unsigned long long)wp_parity_diff_pairs,
                (unsigned long long)wp_pairs[0], (unsigned long long)wp_pairs[1],
                (unsigned long long)wp_pages[0], (unsigned long long)wp_pages[1]);
    return 0;
}
