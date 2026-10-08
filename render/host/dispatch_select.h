// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Task #383: where tt-metal puts its dispatch cores (GSPLAT_TT_DISPATCH=worker|eth|auto).
// Blackhole's default (worker) takes one Tensix column, so the p150 (2 harvested columns,
// 12 live) renders on 11x10 = 110 cores. Ethernet dispatch frees that column: 12x10 = 120
// (tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml, 2xharvested, 1 or 2 CQs;
// llrt/core_descriptor.cpp picks it when the dispatch core type is ETH and skips ETH cores
// with an active link). The p100a has no Ethernet cores and must stay on worker.
// Header-only and free of tt-metal types so tests/unit/test_dispatch_select.cpp can check it.
#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>

namespace gsplat_tt::dispatch {

enum class Mode { kWorker, kEth, kAuto };
enum class Kind { kWorker, kEth };

struct Choice {
    Kind kind;
    const char* why;
};

inline std::string lower_trim(const std::string& s) {
    std::string out;
    for (char ch : s) {
        if (!std::isspace(static_cast<unsigned char>(ch)))
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

// GSPLAT_TT_DISPATCH (nullptr if unset). Unset or empty = worker; *bad is set for an
// unknown value, which also falls back to worker.
inline Mode mode_from_env(const char* s, bool* bad) {
    *bad = false;
    if (s == nullptr) return Mode::kWorker;
    const std::string v = lower_trim(s);
    if (v.empty() || v == "worker") return Mode::kWorker;
    if (v == "eth") return Mode::kEth;
    if (v == "auto") return Mode::kAuto;
    *bad = true;
    return Mode::kWorker;
}

// Blackhole cards with Ethernet cores: p150 (a/b/c) and p300. The p100a has none.
inline bool card_has_eth(const std::string& card_type) {
    const std::string c = lower_trim(card_type);
    return c.rfind("p150", 0) == 0 || c.rfind("p300", 0) == 0;
}

// num_cqs: the descriptor has 1- and 2-CQ entries only.
inline Choice resolve(Mode mode, const std::string& card_type, uint32_t num_cqs) {
    switch (mode) {
        case Mode::kWorker: return {Kind::kWorker, "worker"};
        case Mode::kEth: return {Kind::kEth, "eth (forced)"};
        case Mode::kAuto: break;
    }
    if (num_cqs > 2) return {Kind::kWorker, "auto: ETH descriptor has 1-2 CQs only"};
    if (card_type.empty()) return {Kind::kWorker, "auto: card type unknown"};
    if (!card_has_eth(card_type)) return {Kind::kWorker, "auto: card has no ETH cores"};
    return {Kind::kEth, "auto: card has ETH cores"};
}

// tt-kmd's card type attribute ("p150a", "p100a", ...) of /dev/tenstorrent/<id>.
inline std::string card_type_path(int device_id) {
    return "/sys/class/tenstorrent/tenstorrent!" + std::to_string(device_id) + "/tt_card_type";
}

// First line of `path`, trimmed and lower-cased; "" if it cannot be read.
inline std::string read_card_type_file(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (!f || !std::getline(f, line)) return "";
    return lower_trim(line);
}

}  // namespace gsplat_tt::dispatch
