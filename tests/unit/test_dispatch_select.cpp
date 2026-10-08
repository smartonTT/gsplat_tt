// Host check of the dispatch core choice (task #383, render/host/dispatch_select.h).
// Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_dispatch_select.cpp
//
// GSPLAT_TT_DISPATCH "worker" keeps tt-metal's default (Tensix column). "eth" forces
// Ethernet dispatch. Unset or "auto" (the default since task #409) takes it only on a card
// with Ethernet cores (p150, p300; never the p100a), at most 2 command queues
// (blackhole_140_arch_eth_dispatch.yaml) and the opt/eth overlay as TT_METAL_RUNTIME_ROOT.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "render/host/dispatch_select.h"

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    using namespace gsplat_tt::dispatch;
    bool bad = true;
    // Parsing: unset / empty = auto (task #409 p150 default).
    check(mode_from_env(nullptr, &bad) == Mode::kAuto && !bad, "unset -> auto");
    check(mode_from_env("", &bad) == Mode::kAuto && !bad, "empty -> auto");
    check(mode_from_env("worker", &bad) == Mode::kWorker && !bad, "worker");
    check(mode_from_env("eth", &bad) == Mode::kEth && !bad, "eth");
    check(mode_from_env("ETH", &bad) == Mode::kEth && !bad, "ETH (case)");
    check(mode_from_env("auto", &bad) == Mode::kAuto && !bad, "auto");
    check(mode_from_env("ethernet", &bad) == Mode::kWorker && bad, "unknown -> worker, flagged");

    // Card types.
    check(card_has_eth("p150a"), "p150a has ETH");
    check(card_has_eth("p150b"), "p150b has ETH");
    check(card_has_eth("p300c"), "p300c has ETH");
    check(card_has_eth("P150A\n"), "case / newline");
    check(!card_has_eth("p100a"), "p100a has no ETH");
    check(!card_has_eth("p100"), "p100 has no ETH");
    check(!card_has_eth(""), "unknown card");
    check(!card_has_eth("n300"), "Wormhole card: not handled");

    // Resolution.
    check(resolve(Mode::kWorker, "p150a", 1, true).kind == Kind::kWorker, "worker override on p150");
    check(resolve(Mode::kEth, "p100a", 1, false).kind == Kind::kEth, "eth is forced as asked");
    check(resolve(Mode::kAuto, "p150a", 1, true).kind == Kind::kEth, "auto p150 1 CQ -> eth");
    check(resolve(Mode::kAuto, "p150a", 2, true).kind == Kind::kEth, "auto p150 2 CQs -> eth");
    check(resolve(Mode::kAuto, "p150a", 1, false).kind == Kind::kWorker, "auto p150, no overlay -> worker");
    check(std::strstr(resolve(Mode::kAuto, "p150a", 1, false).why, "overlay") != nullptr,
          "no-overlay fallback names the overlay");
    check(resolve(Mode::kAuto, "p150a", 3, true).kind == Kind::kWorker, "auto 3 CQs: no descriptor");
    check(resolve(Mode::kAuto, "p100a", 1, true).kind == Kind::kWorker, "auto p100a -> worker");
    check(resolve(Mode::kAuto, "", 1, true).kind == Kind::kWorker, "auto unknown card -> worker");
    check(std::strlen(resolve(Mode::kAuto, "p100a", 1, false).why) > 0, "reason logged");

    // Overlay marker in TT_METAL_RUNTIME_ROOT.
    check(!overlay_active(nullptr), "runtime root unset");
    check(!overlay_active(""), "runtime root empty");
    check(!overlay_active("/nonexistent-overlay"), "no marker");
    {
        const char* ov = "/tmp/test_dispatch_select_overlay";
        const std::string mark = std::string(ov) + "/" + kOverlayMarker;
        std::remove(mark.c_str());
        std::string mk = std::string("mkdir -p ") + ov;
        check(std::system(mk.c_str()) == 0, "mkdir overlay");
        check(!overlay_active(ov), "dir without marker");
        { std::ofstream f(mark); f << "src=/x\nn_eth=12\n"; }
        check(overlay_active(ov), "marker present");
        std::remove(mark.c_str());
    }

    // Card type file: first line, trimmed; missing file -> "".
    check(read_card_type_file("/nonexistent/tt_card_type").empty(), "missing file");
    const char* tmp = "/tmp/test_dispatch_select_card_type";
    {
        std::ofstream f(tmp);
        f << "p150a\n";
    }
    check(read_card_type_file(tmp) == "p150a", "file read + trim");
    std::remove(tmp);
    check(card_type_path(0) == "/sys/class/tenstorrent/tenstorrent!0/tt_card_type", "sysfs path");

    if (failures) return 1;
    std::printf("test_dispatch_select: OK\n");
    return 0;
}
