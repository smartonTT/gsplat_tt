// Host check of writer_pfwc_fuse.cpp's bank-major record staging and
// flush_rec index math (task #122). Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_pfwc_flush_rec.cpp
//
// Mirrors the kernel: page g lives on DRAM bank g % nb_real at offset
// g / nb_real; records are staged at slot (rb * RL + rl) in groups of
// RL * nb pages and flushed as one write per bank (or per page when the bank
// count exceeds NB_MAX). For nb_real 1..12, several seg_base values and record
// counts (incl. non-multiples of 16) checks: every record page is written
// exactly once with its own record, nothing outside [seg_base, seg_base + n)
// is written, each write stays in one bank and is contiguous there, and the
// bank-major path issues at most one write per bank per flush.
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;
void fail(const char* what, uint32_t nb, uint32_t sb, uint32_t n, uint32_t x) {
    if (failures++ < 20) std::fprintf(stderr, "FAIL %s nb=%u seg_base=%u n=%u at %u\n", what, nb, sb, n, x);
}

constexpr uint32_t RL = 16, NB_MAX = 8;

void run(uint32_t nb_real, uint32_t seg_base, uint32_t n) {
    const bool per_page = nb_real > NB_MAX;
    const uint32_t nb = per_page ? NB_MAX : nb_real;
    const uint32_t GS = RL * nb;
    std::vector<uint32_t> stage(NB_MAX * RL, ~0u);
    const uint32_t lim = seg_base + n + 2 * NB_MAX * RL;
    std::vector<uint32_t> dram(lim, ~0u), hits(lim, 0);

    uint32_t G0 = seg_base / GS * GS, gs = seg_base;
    uint32_t rb = (seg_base - G0) % nb, rl = (seg_base - G0) / nb;

    // One NoC write of `len` staged pages starting at slot s to page g; the
    // destination pages are g, g + nb_real, ... (contiguous in bank g % nb_real).
    auto write = [&](uint32_t s, uint32_t g, uint32_t len) {
        for (uint32_t i = 0; i < len; ++i) {
            const uint32_t p = g + i * nb_real;
            if (p >= lim || p < seg_base || p >= seg_base + n) { fail("write outside", nb_real, seg_base, n, p); continue; }
            dram[p] = stage[s + i];
            hits[p]++;
        }
    };
    auto flush_rec = [&](uint32_t ge) {
        const uint32_t d = gs - G0, e = ge - G0;
        std::vector<uint32_t> per_bank(nb, 0);
        for (uint32_t b = 0; b < nb; ++b) {
            uint32_t lo = 0, hi = RL;
            if (d != 0 || e != GS) {
                lo = d > b ? (d - b + nb - 1) / nb : 0;
                hi = e > b ? (e - b + nb - 1) / nb : 0;
            }
            if (hi <= lo) continue;
            if (per_page) {
                for (uint32_t l = lo; l < hi; ++l) write(b * RL + l, G0 + b + l * nb, 1);
            } else {
                write(b * RL + lo, G0 + b + lo * nb, hi - lo);
                if (++per_bank[b] > 1) fail("two writes one bank", nb_real, seg_base, n, b);
            }
        }
    };
    for (uint32_t g = seg_base; g < seg_base + n; ++g) {
        stage[rb * RL + rl] = g;  // record content = its page index
        if (++rb == nb) {
            rb = 0;
            if (++rl == RL) {
                flush_rec(G0 + GS);
                G0 += GS;
                gs = G0;
                rl = 0;
            }
        }
    }
    if (G0 + rl * nb + rb > gs) flush_rec(G0 + rl * nb + rb);

    for (uint32_t g = seg_base; g < seg_base + n; ++g) {
        if (hits[g] != 1) fail("hits != 1", nb_real, seg_base, n, g);
        else if (dram[g] != g) fail("wrong record", nb_real, seg_base, n, g);
    }
}

}  // namespace

int main() {
    const uint32_t bases[] = {0, 1024, 3072, 5 * 1024, 1000, 7};
    const uint32_t counts[] = {0, 1, 5, 15, 16, 17, 31, 47, 100, 127, 128, 129, 255, 1000, 1023, 1500};
    uint32_t cases = 0;
    for (uint32_t nb = 1; nb <= 12; ++nb)
        for (uint32_t sb : bases)
            for (uint32_t n : counts) { run(nb, sb, n); cases++; }
    std::printf("%u cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
