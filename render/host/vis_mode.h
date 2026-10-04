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

// Lever B (task #125, GSPLAT_TT_PFWC_FUSE): 1 = the pfwc writer also compacts
// the visible gaussians into per-core segments (writer_pfwc_fuse.cpp) and the
// tile_assign segment K2 builds the pairs from them; the gather scan / scatter
// and the TA vis K2 do not run. 0 = kill switch, lever 2 path. Needs
// GSPLAT_TT_SFPU_VIS != 0. Default 1 since task #122 (bicycle 30 views on
// yyzo-bh-07 p100a: 26.2-26.3 against 29.6-29.7 ms/view, md5-identical), but
// only with GSPLAT_TT_SFPU_VIS=1: the =2 cross-check keeps the lever 2 path.
inline int pfwc_fuse_mode() {
    static const int v = [] {
        const char* e = std::getenv("GSPLAT_TT_PFWC_FUSE");
        if (e == nullptr || *e == '\0') return sfpu_vis_mode() == 1 ? 1 : 0;
        const int x = std::atoi(e);
        if (x < 0 || x > 1)
            throw std::invalid_argument(std::string("GSPLAT_TT_PFWC_FUSE must be 0 or 1, got '") +
                                        e + "'");
        return x;
    }();
    return v;
}

// Task #122, GSPLAT_TT_PUBOC_PRE: with EMIT_PUBOC, record words 10 / 11 (the
// UNORM16 opacity / color packs) depend only on the scene, so the host builds
// them once per scene (scene_puboc01 / scene_puboc23, the integer
// sort_bin_fp32::unorm16) and the fused writer and the gather scatter copy
// them. 1 (default) = on; 0 = kill switch, packs computed per gaussian on the
// device. A scene with a NaN opacity / color keeps the device path.
inline int puboc_pre_mode() {
    static const int v = [] {
        const char* e = std::getenv("GSPLAT_TT_PUBOC_PRE");
        if (e == nullptr || *e == '\0') return 1;
        const int x = std::atoi(e);
        if (x < 0 || x > 1)
            throw std::invalid_argument(std::string("GSPLAT_TT_PUBOC_PRE must be 0 or 1, got '") +
                                        e + "'");
        return x;
    }();
    return v;
}

// Lever C (task #140, GSPLAT_TT_PRECULL): 1 = pfwc (PFWC_PRECULL) shrinks the
// tile rectangle to the opacity-aware extent the microblock band cull can keep
// (project_pfwc_compute.cpp step 11.6), so dead (gaussian, tile) records are
// never made. Bit-identical with BLEND_T_PERIOD=0; at the default period the
// T-saturation checkpoints shift, so <=1 LSB on ~0.3-0.5% of pixels (#142,
// 74.9-78.2 dB). Default 1 since task #156 (reviewed; 1 LSB accepted);
// 0 = kill switch (the pre-#156 output). Only with GSPLAT_TT_SFPU_VIS=1 (mode
// 2 cross-checks the rectangle against the legacy path) and with the band
// cull on.
inline int precull_mode() {
    static const int v = [] {
        const char* e = std::getenv("GSPLAT_TT_PRECULL");
        if (e == nullptr || *e == '\0') return 1;
        const int x = std::atoi(e);
        if (x < 0 || x > 1)
            throw std::invalid_argument(std::string("GSPLAT_TT_PRECULL must be 0 or 1, got '") +
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
