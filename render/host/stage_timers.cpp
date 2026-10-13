// SPDX-License-Identifier: Apache-2.0
//
// Out-of-line definition of the per-stage accumulator. See stage_timers.h.

#include "stage_timers.h"

#include <cstdlib>

namespace gsplat_tt::stagetimers {

Acc& acc() {
    static Acc a;
    return a;
}

void reset() { acc() = Acc{}; }

bool split_blend() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_SPLIT_BLEND");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

const char* bucket_name(const double* sink) {
    static const char* const kNames[] = {
        "head",
        "project",
        "tile_assign",
        "sort",
        "blend_setup",
        "mat",
        "cull",
        "blend",
        "d2h",
        "assemble",
        "tail",
        "xview",
        "view_total",
        "sort_pread",
        "sort_bin_count",
        "sort_bin_hist_d2h",
        "sort_bin_layout",
        "sort_upload",
        "sort_bin_emit",
        "sort_kernel",
        "sort_d2h",
        "sort_compact",
        "sort_publish_host",
        "sort_publish_wait",
        "sort_mat",
        "sort_pre",
        "sort_log",
        "sort_cont_prep",
        "sort_cont_rtargs",
        "sort_cont_other",
        "project_cov3d",
        "project_pfwc_setup",
        "project_pfwc_rtargs",
        "project_pfwc_chunkcull",
        "project_pfwc_enqueue",
        "project_pfwc_finish",
        "project_gather_setup",
        "project_gather_rtargs",
        "project_gather_enqueue",
        "project_gather_wait",
        "project_gather_result",
        "tile_assign_setup",
        "tile_assign_rtargs",
        "tile_assign_enqueue",
        "tile_assign_scan_finish",
        "tile_assign_p_d2h",
        "tile_assign_k2_finish",
        "tile_assign_publish"};
    constexpr std::size_t n = sizeof(kNames) / sizeof(kNames[0]);
    const double* base = &acc().head;
    if (sink < base || sink >= base + n) return "span";
    return kNames[sink - base];
}

}  // namespace gsplat_tt::stagetimers

#include <cstddef>
static_assert(offsetof(gsplat_tt::stagetimers::Acc, views) == 48 * sizeof(double),
              "bucket_name: update kNames with the Acc fields");
