// Model check for the dual-data-mover split of the sort_bin scatter pass
// (render/kernels/dataflow/sort_bin.cpp, mode 1; host side in
// render/host/sort_device.cpp launch_bin). Standalone:
//
//   c++ -O2 -std=c++17 tests/unit/test_sort_bin_dual_mover.cpp -o /tmp/t_dm && /tmp/t_dm
//
// Replays the kernel's placement rules for one core's pair-page range on random
// gaussian-major inputs: once on a single mover over [lo, hi), and once split at
// every page boundary mid, with h0 from a replay of the kernel's batched count
// pass (mover 0 = [lo, mid) with cursors from 0, mover 1 =
// [mid, hi) with cursors from the count pass's snapshot h0). The split must
// leave the same counting-sort bytes in the core's shared L1 region, the same
// record / overflow slots, and the same written-out pages. The movers run
// concurrently on the device, so the model runs the worst order for the shared
// region: mover 0 pre-fills and fills completely before mover 1 pre-fills.
// Returns non-zero on any mismatch (e.g. mover 1 cursors starting at 0, or a
// mover pre-filling tiles it does not write out).
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t EPP = 16;          // pairs per 64 B page
constexpr uint32_t SENT = 0xFFFFFFFFu;
constexpr uint32_t GARBAGE = 0xDEADBEEFu;

struct Input {
    uint32_t P = 0, num_tiles = 0, fit = 0, lo = 0, hi = 0;
    std::vector<uint32_t> gid, tid, keep;  // per pair (page-padded)
    std::vector<uint32_t> key;             // per gaussian
    std::vector<uint32_t> l1base, ovbase;  // per tile (host layout for this core)
};

struct Output {
    std::vector<uint32_t> ks, is;                     // shared counting-sort region
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> recs, ov;  // slot -> (g, t)
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> pages;  // (t, page) -> 32 words
    bool conflict = false;
};

std::vector<uint32_t> count(const Input& in, uint32_t a, uint32_t b) {
    std::vector<uint32_t> h(in.num_tiles, 0);
    for (uint32_t pg = a; pg < b; pg++)
        for (uint32_t j = 0; j < EPP; j++) {
            const uint32_t p = pg * EPP + j;
            if (p >= in.P) break;
            if (in.keep[p]) h[in.tid[p]]++;
        }
    return h;
}

// The kernel's count pass (mode 0): CNT_BATCH-page batches over [lo, hi) that
// stop at the split, snapshotting the running histogram when pg0 reaches it
// (or before the loop when split == lo). Returns the full histogram in n and
// the snapshot in h0; h0 stays all-zero if the snapshot never fires.
void count_pass(const Input& in, bool dual, uint32_t mid, std::vector<uint32_t>& n,
                std::vector<uint32_t>& h0) {
    constexpr uint32_t CNT_BATCH = 32u;
    n.assign(in.num_tiles, 0);
    h0.assign(in.num_tiles, 0);
    const uint32_t split = dual ? mid : in.hi + 1u;
    if (split == in.lo) h0 = n;
    for (uint32_t pg0 = in.lo; pg0 < in.hi;) {
        const uint32_t lim = (pg0 < split && split < in.hi) ? split : in.hi;
        const uint32_t nb = (lim - pg0 < CNT_BATCH) ? (lim - pg0) : CNT_BATCH;
        for (uint32_t b = 0; b < nb; b++)
            for (uint32_t j = 0; j < EPP; j++) {
                const uint32_t p = (pg0 + b) * EPP + j;
                if (p >= in.P) break;
                if (in.keep[p]) n[in.tid[p]]++;
            }
        pg0 += nb;
        if (pg0 == split) h0 = n;
    }
}

// One mover of the scatter: the kernel's prefix, t_split, tail pre-fill,
// cursor init and fill loop. Write-out is separate (after both fills).
struct Mover {
    uint32_t wt_lo = 0, wt_hi = 0;
};

Mover run_mover(const Input& in, bool dual, uint32_t mover, uint32_t pg_lo, uint32_t pg_hi,
                const std::vector<uint32_t>& n, const std::vector<uint32_t>& h0,
                std::vector<uint32_t>& offp, uint32_t& run, Output& o,
                std::vector<int>& owner) {
    const uint32_t T = in.num_tiles;
    offp.assign(T, 0);
    run = 0;
    for (uint32_t t = 0; t < T; t++) {
        offp[t] = run;
        run += ((n[t] + EPP - 1) / EPP) * EPP;
    }
    uint32_t t_split = T;
    if (dual) {
        const uint32_t half = run / 2u;
        t_split = 0;
        while (t_split < T && offp[t_split] < half) t_split++;
    }
    Mover m;
    m.wt_lo = (dual && mover == 1) ? t_split : 0u;
    m.wt_hi = (dual && mover == 0) ? t_split : T;
    auto put = [&](uint32_t i, uint32_t k, uint32_t g) {
        if (owner[i] != -1 && owner[i] != static_cast<int>(mover)) o.conflict = true;
        owner[i] = static_cast<int>(mover);
        o.ks[i] = k;
        o.is[i] = g;
    };
    for (uint32_t t = m.wt_lo; t < m.wt_hi; t++) {
        const uint32_t end = (t + 1u < T) ? offp[t + 1u] : run;
        for (uint32_t i = offp[t] + n[t]; i < end; i++) put(i, 0xffffffffu, 0u);
    }
    std::vector<uint32_t> cur = (dual && mover == 1) ? h0 : std::vector<uint32_t>(T, 0u);
    for (uint32_t pg = pg_lo; pg < pg_hi; pg++)
        for (uint32_t j = 0; j < EPP; j++) {
            const uint32_t p = pg * EPP + j;
            if (p >= in.P) break;
            if (!in.keep[p]) continue;
            const uint32_t g = in.gid[p], t = in.tid[p];
            const uint32_t ovb = in.ovbase[t];
            const auto rec = std::make_pair(g, t);
            if (ovb != SENT) {
                if (!o.ov.emplace(ovb + cur[t], rec).second) o.conflict = true;
            } else {
                const uint32_t s = in.l1base[t] + cur[t];
                if (s < (t + 1u) * in.fit && !o.recs.emplace(s, rec).second) o.conflict = true;
            }
            put(offp[t] + cur[t], in.key[g], g);
            cur[t]++;
        }
    return m;
}

void write_out(const Input& in, const Mover& m, const std::vector<uint32_t>& offp,
               uint32_t run, Output& o) {
    for (uint32_t t = m.wt_lo; t < m.wt_hi; t++) {
        const uint32_t end = (t + 1u < in.num_tiles) ? offp[t + 1u] : run;
        for (uint32_t pp = 0; pp < (end - offp[t]) / EPP; pp++) {
            std::vector<uint32_t> w;
            for (uint32_t e = 0; e < EPP; e++) w.push_back(o.ks[offp[t] + pp * EPP + e]);
            for (uint32_t e = 0; e < EPP; e++) w.push_back(o.is[offp[t] + pp * EPP + e]);
            if (!o.pages.emplace(std::make_pair(t, pp), w).second) o.conflict = true;
        }
    }
}

Output emit(const Input& in, bool dual, uint32_t mid) {
    std::vector<uint32_t> n, h0;
    count_pass(in, dual, mid, n, h0);
    // Sanity: the batched pass must agree with a direct count of each range.
    if (n != count(in, in.lo, in.hi) || (dual && h0 != count(in, in.lo, mid))) {
        Output bad;
        bad.conflict = true;
        return bad;
    }
    uint32_t cap = 0;
    for (uint32_t t = 0; t < in.num_tiles; t++) cap += ((n[t] + EPP - 1) / EPP) * EPP;
    Output o;
    o.ks.assign(cap, GARBAGE);
    o.is.assign(cap, GARBAGE);
    std::vector<int> owner(cap, -1);
    std::vector<uint32_t> off0, off1;
    uint32_t run0 = 0, run1 = 0;
    if (!dual) {
        const Mover m = run_mover(in, false, 1, in.lo, in.hi, n, h0, off1, run1, o, owner);
        write_out(in, m, off1, run1, o);
        return o;
    }
    const Mover m0 = run_mover(in, true, 0, in.lo, mid, n, h0, off0, run0, o, owner);
    const Mover m1 = run_mover(in, true, 1, mid, in.hi, n, h0, off1, run1, o, owner);
    if (off0 != off1 || run0 != run1 || m0.wt_hi != m1.wt_lo) o.conflict = true;
    write_out(in, m0, off0, run0, o);
    write_out(in, m1, off1, run1, o);
    return o;
}

Input make_input(std::mt19937& rng) {
    Input in;
    in.num_tiles = 8 + rng() % 40;
    in.fit = 1 + rng() % 12;
    const uint32_t G = 1 + rng() % 600;
    for (uint32_t g = 0; g < G; g++) {
        in.key.push_back(rng());
        const uint32_t k = 1 + rng() % 4;  // tiles per gaussian (gaussian-major)
        for (uint32_t i = 0; i < k; i++) {
            in.gid.push_back(g);
            in.tid.push_back(rng() % in.num_tiles);
            in.keep.push_back(rng() % 5 != 0);
        }
    }
    in.P = static_cast<uint32_t>(in.gid.size());
    const uint32_t pages = (in.P + EPP - 1) / EPP;
    in.gid.resize(pages * EPP, 0);
    in.tid.resize(pages * EPP, 0);
    in.keep.resize(pages * EPP, 0);
    in.lo = rng() % pages;  // this core's page range
    in.hi = in.lo + rng() % (pages - in.lo + 1);
    uint32_t ov_cursor = 0;
    for (uint32_t t = 0; t < in.num_tiles; t++) {
        in.l1base.push_back(t * in.fit + rng() % (in.fit + 1));  // + other cores' prefix
        in.ovbase.push_back(rng() % 4 == 0 ? (ov_cursor += 4096) : SENT);
    }
    return in;
}

}  // namespace

int main() {
    std::mt19937 rng(12345);
    uint32_t cases = 0, fails = 0;
    for (int it = 0; it < 3000; it++) {
        const Input in = make_input(rng);
        const Output ref = emit(in, false, in.lo);
        if (ref.conflict) {
            std::printf("single-mover model conflict (case %d)\n", it);
            return 2;
        }
        for (uint32_t mid = in.lo; mid <= in.hi; mid++) {
            const Output d = emit(in, true, mid);
            cases++;
            if (d.conflict || d.ks != ref.ks || d.is != ref.is || d.recs != ref.recs ||
                d.ov != ref.ov || d.pages != ref.pages) {
                if (fails++ < 5)
                    std::printf("MISMATCH case %d lo=%u mid=%u hi=%u conflict=%d\n", it, in.lo,
                                mid, in.hi, d.conflict ? 1 : 0);
            }
        }
    }
    std::printf("dual-mover split: %u (input, mid) cases, %u mismatches\n", cases, fails);
    return fails == 0 ? 0 : 1;
}
