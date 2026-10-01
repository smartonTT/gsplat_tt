// SPDX-License-Identifier: Apache-2.0
//
// Lever 2 (task #99): the visibility predicate and the tile_assign rectangle
// run on the SFPU inside pfwc, so proj_count, ta_gauss_aabb and the three TA
// scan programs drop out of the frame. Selected once per process:
//
//   GSPLAT_TT_SFPU_VIS=0           legacy path, unchanged (kill switch).
//   GSPLAT_TT_SFPU_VIS=1 (default) pfwc (PFWC_VIS) emits the tpg / aabb word
//       tiles, the visibility mask and per-tile counts; gather_vis_scan +
//       gather_vis_scatter compact with a balanced cut and write proj_m_depth,
//       the blend record, proj_m_offs and proj_m_aabb; tile_assign runs K2 only
//       (TA_K2_AABB). Same compact order, offs and pairs as the legacy path.
//   GSPLAT_TT_SFPU_VIS=2           as 1, plus a per-view cross-check against the
//       legacy passes (proj_count mask/M, K1 + scans offs, host K1 rectangle);
//       prints [VIS-CHECK] lines to stderr. Slow; for validation only.
//
// GSPLAT_TT_VIS_BALANCE=0 keeps the legacy per-core halves for the scatter
// (A/B of the balance alone); GSPLAT_TT_VIS_TILE_WEIGHT / _EMPTY_WEIGHT set the
// cut's cost model (defaults 24 / 1, in units of one visible gaussian).
// GSPLAT_TT_VIS_EDGE_TAU (default 2^-12, 0 = off): lanes whose right / bottom
// tile edge q = fl(m + r) / tile_size is within tau of an integer are resolved
// by the pfwc writer's exact soft-float path (insurance against an SFPMAD
// rounding that is not nearest-even).
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gsplat_tt {

inline int sfpu_vis_mode() {
    static const int v = [] {
        const char* e = std::getenv("GSPLAT_TT_SFPU_VIS");
        if (e == nullptr || *e == '\0') return 1;  // adopted in task #102
        const int x = std::atoi(e);
        if (x < 0 || x > 2)
            throw std::invalid_argument(std::string("GSPLAT_TT_SFPU_VIS must be 0, 1 or 2, got '") +
                                        e + "'");
        return x;
    }();
    return v;
}

inline unsigned vis_env_u32(const char* name, unsigned dflt) {
    const char* e = std::getenv(name);
    return (e != nullptr && *e != '\0') ? static_cast<unsigned>(std::atoi(e)) : dflt;
}

}  // namespace gsplat_tt
