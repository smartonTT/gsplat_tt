// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Weighted pfwc tile deal (task #439, GSPLAT_TT_PFWC_DEAL=lpt, default off).
// Header-only so tests/unit/test_pfwc_deal.cpp can check it without tt-metal.
//
// The strided deal (tile t -> core t mod C) gives some cores up to 24% more
// visible gaussians than the mean at ETH 12x10, and the fused writer is bound by
// record emission per visible gaussian (docs/pfwc-imbalance-t437/README.md). This
// is a static equal-count LPT deal from per-tile weights: tiles sorted by weight
// (heaviest first) are dealt in rounds of C, each to the least-loaded core not
// yet served that round; the last, partial round goes to the cores c < rem that
// get the extra tile in the strided deal. Every core keeps SeqMap.count(c) tiles,
// so pfwc_fuse::seg_base and the segment table hold. Within a core the tiles stay
// in round order, so chunk k is a round-k tile (parity = split writer role).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace pfwc_deal {

// Per-core tile lists, list[c] in chunk order; list[c].size() == base + (c < rem).
inline std::vector<std::vector<uint32_t>> lpt(const std::vector<double>& w, uint32_t num_cores) {
    const uint32_t T = static_cast<uint32_t>(w.size()), C = num_cores;
    const uint32_t base = T / C, rem = T % C;
    std::vector<uint32_t> order(T);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return w[a] > w[b]; });
    std::vector<std::vector<uint32_t>> list(C);
    std::vector<double> load(C, 0.0);
    std::vector<uint32_t> cores(C);
    std::size_t i = 0;
    for (uint32_t r = 0; r <= base; ++r) {
        const uint32_t n = r < base ? C : rem;  // last round: only the c < rem cores
        cores.resize(n);
        std::iota(cores.begin(), cores.end(), 0u);
        // Least-loaded first (ties by core id); the round's tiles go heaviest first.
        std::stable_sort(cores.begin(), cores.end(), [&](uint32_t a, uint32_t b) { return load[a] < load[b]; });
        for (uint32_t j = 0; j < n; ++j, ++i) {
            const uint32_t c = cores[j], t = order[i];
            list[c].push_back(t);
            load[c] += w[t];
        }
    }
    return list;
}

// One page of `words` uint32 per core, core c's ids at c * words (chunk_cull::deal layout).
inline void pack(const std::vector<std::vector<uint32_t>>& list, uint32_t words, std::vector<uint32_t>& host) {
    host.assign(std::size_t{words} * list.size(), 0u);
    for (std::size_t c = 0; c < list.size(); ++c)
        std::copy(list[c].begin(), list[c].end(), host.begin() + static_cast<std::ptrdiff_t>(c * words));
}

// Weights file: whitespace-separated numbers, one per 1024-gaussian tile; '#' starts a comment line.
inline bool load_weights(const std::string& path, std::vector<double>& w) {
    std::ifstream f(path);
    if (!f) return false;
    w.clear();
    std::string tok;
    while (f >> tok) {
        if (tok[0] == '#') {
            std::getline(f, tok);
            continue;
        }
        w.push_back(std::stod(tok));
    }
    return true;
}

}  // namespace pfwc_deal
