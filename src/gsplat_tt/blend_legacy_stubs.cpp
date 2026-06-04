// Legacy upload-payload blend entry points still declared in blend.h and
// referenced from render_blend.cpp for non-resident fallbacks. Stage2
// production uses resident blend only; these exist so _gsplat_cpu links.
#include "gsplat_tt/blend.h"

#include <cstdio>
#include <cstdlib>

namespace gsplat_tt {

namespace {

[[noreturn]] void legacy_removed(const char* fn) {
    std::fprintf(stderr,
                 "[gsplat_tt] FATAL: %s is not built in the stage2 tree; "
                 "set GSPLAT_TT_RESIDENT_BLEND=1 (production default).\n",
                 fn);
    std::abort();
}

}  // namespace

double blend_from_payload(
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<uint32_t>&,
    const std::vector<uint32_t>&,
    int,
    int,
    std::vector<float>&) {
    legacy_removed("blend_from_payload");
}

double blend_mb_from_payload(
    const std::vector<uint32_t>&,
    const std::vector<uint32_t>&,
    const std::vector<float>&,
    int,
    int,
    int,
    int,
    std::vector<float>&) {
    legacy_removed("blend_mb_from_payload");
}

}  // namespace gsplat_tt
