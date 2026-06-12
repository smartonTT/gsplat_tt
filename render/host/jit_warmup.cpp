#include "jit_warmup.h"

#include <chrono>
#include <cstdio>
#include <mutex>

#include "blend.h"
#include "device_state.h"
#include "env_config.h"
#include "gather_visible.h"
#include "pfwc.h"
#include "project.h"
#include "sort.h"
#include "tile_assign.h"

namespace gsplat_tt {

void jit_warmup_ideal_path() {
    if (!env_config::jit_warmup_enabled()) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, []() {
        const auto t0 = std::chrono::steady_clock::now();
        (void)device_state::get_device();
        // iter-133: the standalone project_means_cam program is fused into pfwc;
        // only the fused pfwc context is warmed (project_device.cpp is no longer
        // on the render path).
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
