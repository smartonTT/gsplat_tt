#include "gsplat_tt/jit_warmup.h"

#include <chrono>
#include <cstdio>
#include <mutex>

#include "gsplat_tt/blend.h"
#include "gsplat_tt/device_state.h"
#include "gsplat_tt/env_config.h"
#include "gsplat_tt/gather_visible.h"
#include "gsplat_tt/pfwc.h"
#include "gsplat_tt/project.h"
#include "gsplat_tt/sort.h"
#include "gsplat_tt/tile_assign.h"

namespace gsplat_tt {

void jit_warmup_ideal_path() {
    if (!env_config::flag_is_one("GSPLAT_TT_JIT_WARMUP")) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, []() {
        const auto t0 = std::chrono::steady_clock::now();
        (void)device_state::get_device();
        (void)project_device_ready();
        (void)pfwc_device_ready();
        (void)gather_visible_device_ready();
        (void)tile_assign_device_ready();
        (void)sort_device_ready();
        blend_warmup_resident_contexts();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        std::fprintf(stderr, "[JIT_WARMUP] ideal-path contexts compiled in %.1f ms\n", ms);
    });
}

}  // namespace gsplat_tt
