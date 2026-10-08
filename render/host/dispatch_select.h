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

// GSPLAT_TT_DISPATCH (nullptr if unset). Unset or empty = auto (task #409: ETH on a p150
// whose TT_METAL_RUNTIME_ROOT is an opt/eth overlay, else worker); *bad is set for an
// unknown value, which falls back to worker.
inline Mode mode_from_env(const char* s, bool* bad) {
    *bad = false;
    if (s == nullptr) return Mode::kAuto;
    const std::string v = lower_trim(s);
    if (v.empty() || v == "auto") return Mode::kAuto;
    if (v == "worker") return Mode::kWorker;
    if (v == "eth") return Mode::kEth;
    *bad = true;
    return Mode::kWorker;
}

// Blackhole cards with Ethernet cores: p150 (a/b/c) and p300. The p100a has none.
inline bool card_has_eth(const std::string& card_type) {
    const std::string c = lower_trim(card_type);
    return c.rfind("p150", 0) == 0 || c.rfind("p300", 0) == 0;
}

// Marker opt/eth/make_overlay.sh writes into the overlay root. Stock tt-metal cannot open
// ETH dispatch on a p150 (14-core yaml lists, 24 KB idle-ERISC .ld), so auto needs it.
inline const char* kOverlayMarker = ".gsplat-eth-overlay";

// True if `runtime_root` (TT_METAL_RUNTIME_ROOT, nullptr if unset) holds the marker.
inline bool overlay_active(const char* runtime_root) {
    if (runtime_root == nullptr || *runtime_root == '\0') return false;
    std::ifstream f(std::string(runtime_root) + "/" + kOverlayMarker);
    return static_cast<bool>(f);
}

// num_cqs: the descriptor has 1- and 2-CQ entries only. overlay: overlay_active().
inline Choice resolve(Mode mode, const std::string& card_type, uint32_t num_cqs, bool overlay) {
    switch (mode) {
        case Mode::kWorker: return {Kind::kWorker, "worker"};
        case Mode::kEth: return {Kind::kEth, "eth (forced)"};
        case Mode::kAuto: break;
    }
    if (num_cqs > 2) return {Kind::kWorker, "auto: ETH descriptor has 1-2 CQs only"};
    if (card_type.empty()) return {Kind::kWorker, "auto: card type unknown"};
    if (!card_has_eth(card_type)) return {Kind::kWorker, "auto: card has no ETH cores"};
    if (!overlay) return {Kind::kWorker, "auto: TT_METAL_RUNTIME_ROOT is not an opt/eth overlay, falling back"};
    return {Kind::kEth, "auto: card has ETH cores, overlay active"};
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
