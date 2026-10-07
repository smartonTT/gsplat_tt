// Host checks for task #198 (GSPLAT_TT_SORT_OL_EARLY / GSPLAT_TT_MAT_CQ1). Standalone:
//
//   CXXFLAGS="-Irender/kernels/dataflow" tests/unit/run_cpp.sh tests/unit/test_two_cq_t198.cpp
//
// - totals_from_k2_rows (the host's bucket totals from the fold K2's per-mover
//   count rows) equals device_prefix's totals of the per-core rows, the rows
//   the one-launch sort's prefix phase sums, on random heavy-tailed counts;
// - the early launch's mover ranges (pfwc_fuse::k2_range_speed on the kernel's
//   uint32 args: P from ta_pairs_P, the running speed sums) equal the K2's
//   published bounds and the sort's speed_bounds, and tile every page once.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "pfwc_fuse.h"
#include "render/host/sort_mover_speed.h"
#include "render/host/sort_mover_split.h"
#include "render/host/sort_onelaunch_layout.h"

namespace {

int failures = 0;
void fail(const char* what, uint64_t a, uint64_t b) {
    if (failures++ < 20)
        std::fprintf(stderr, "FAIL %s: %llu %llu\n", what, static_cast<unsigned long long>(a),
                     static_cast<unsigned long long>(b));
}

void check_totals(std::mt19937& rng, uint32_t num_cores, uint32_t num_tiles) {
    const uint32_t stride = (num_tiles + 15u) & ~15u;
    std::vector<uint32_t> krow(2u * num_cores * stride, 0u);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (uint32_t r = 0; r < 2u * num_cores; r++) {
        for (uint32_t t = 0; t < num_tiles; t++) {  // padding words stay 0, as the K2 writes them
            const double x = u(rng);
            krow[r * stride + t] = x < 0.3 ? 0u : static_cast<uint32_t>(x * x * x * 400.0);
        }
    }
    std::vector<uint32_t> h(num_cores * stride, 0u);
    for (uint32_t c = 0; c < num_cores; c++)
        for (uint32_t t = 0; t < stride; t++)
            h[c * stride + t] = krow[(2u * c) * stride + t] + krow[(2u * c + 1u) * stride + t];
    const auto ref = gsplat_tt::sort_onelaunch::device_prefix(h, num_cores, stride);
    std::vector<uint32_t> tot(2u * stride + 37u, 0xDEADu);  // a grown buffer: extra words
    gsplat_tt::sort_onelaunch::totals_from_k2_rows(krow, num_cores, stride, tot);
    for (uint32_t i = 0; i < 2u * stride; i++)
        if (tot[i] != ref.totals[i]) fail("totals_from_k2_rows != device_prefix", i, tot[i]);
}

void check_early_ranges(std::mt19937& rng, uint32_t num_cores,
                        gsplat_tt::sort_split::MoverBoard board) {
    std::vector<uint32_t> noc_xy(num_cores);
    for (uint32_t c = 0; c < num_cores; c++) {
        const uint32_t x = 1u + (c % 10u) + (c % 10u >= 4u ? 1u : 0u);
        const uint32_t y = 2u + c / 10u + (c / 10u >= 5u ? 1u : 0u);
        noc_xy[c] = x | (y << 16);
    }
    const std::vector<uint32_t> speed = gsplat_tt::sort_split::mover_speeds(noc_xy, board);
    std::vector<uint64_t> acc(speed.size() + 1u, 0u);
    for (std::size_t k = 0; k < speed.size(); k++) acc[k + 1] = acc[k] + speed[k];
    if (acc.size() != 2u * num_cores + 1u) fail("acc size", acc.size(), 2u * num_cores + 1u);
    std::uniform_int_distribution<uint32_t> pd(0u, 4718592u);
    for (int it = 0; it < 200; it++) {
        const uint32_t P_pub = it == 0 ? 0u : (it == 1 ? 1u : pd(rng));
        const uint32_t pages = (P_pub + 15u) / 16u;
        const std::vector<uint32_t> sb = gsplat_tt::sort_split::speed_bounds(pages, speed);
        uint32_t next = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            for (uint32_t m = 0; m < 2u; m++) {
                // Kernel: arg 10 = acc[2c+m], 28 = acc[2c+m+1], 29 = acc.back() (uint32).
                const uint32_t a10 = static_cast<uint32_t>(acc[2u * c + m]);
                const uint32_t a28 = static_cast<uint32_t>(acc[2u * c + m + 1u]);
                const uint32_t a29 = static_cast<uint32_t>(acc.back());
                uint32_t s = 0, n = 0;
                pfwc_fuse::k2_range_speed(P_pub, a10, a28, a29, &s, &n);
                if (s != sb[2u * c + m] || s + n != sb[2u * c + m + 1u])
                    fail("early range != speed_bounds", 2u * c + m, s);
                if (s != next) fail("early ranges not contiguous", 2u * c + m, s);
                next = s + n;
            }
        }
        if (next != pages) fail("early ranges do not cover the pages", next, pages);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(198u);
    check_totals(rng, 110u, 1014u);
    check_totals(rng, 110u, 4056u);
    check_totals(rng, 7u, 16u);
    check_totals(rng, 1u, 1u);
    for (const auto b : {gsplat_tt::sort_split::MoverBoard::P100a,
                         gsplat_tt::sort_split::MoverBoard::P150}) {  // task #358
        check_early_ranges(rng, 110u, b);
        check_early_ranges(rng, 64u, b);
    }
    std::printf("%s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
