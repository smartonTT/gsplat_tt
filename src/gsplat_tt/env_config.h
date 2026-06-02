#pragma once

#include <cstdlib>

namespace gsplat_tt::env_config {

inline bool flag_is_one(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}

inline bool flag_is_zero(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}

// Production gather path (FUSED_TILE). Default OFF — the loop/verify ideal path is
// TILE_BUCKET + SFPU cull_masks, not per-candidate gather.
inline bool fused_tile_enabled() {
    return flag_is_one("GSPLAT_TT_FUSED_TILE");
}

// L1-resident full-record bucket scatter. Default ON when SFPU+resident blend are on
// and FUSED_TILE is not (matches ttw.toml verify_cmd).
inline bool tile_bucket_enabled() {
    const char* v = std::getenv("GSPLAT_TT_TILE_BUCKET");
    if (v != nullptr) {
        return v[0] == '1';
    }
    if (fused_tile_enabled()) {
        return false;
    }
    return flag_is_one("GSPLAT_TT_SFPU_CULL") && flag_is_one("GSPLAT_TT_RESIDENT_BLEND");
}

inline bool proj_device_scan_enabled() {
    return !flag_is_zero("GSPLAT_TT_PROJ_DEVICE_SCAN");
}

// L1 bucket CB producer/consumer fence (default ON when TILE_BUCKET active).
inline bool bucket_cb_fence_enabled() {
    return !flag_is_zero("GSPLAT_TT_BUCKET_CB_FENCE");
}

inline bool resident_blend_enabled() {
    return flag_is_one("GSPLAT_TT_RESIDENT_BLEND");
}

inline bool sfpu_cull_enabled() {
    return flag_is_one("GSPLAT_TT_SFPU_CULL");
}

// Overlap blend host setup with the SFPU cull device window (same in-order CQ).
// Default ON for the ideal resident SFPU-cull path; set =0 to force a cull Finish.
inline bool cull_pipeline_enabled() {
    if (flag_is_zero("GSPLAT_TT_CULL_PIPELINE")) {
        return false;
    }
    if (flag_is_one("GSPLAT_TT_CULL_PIPELINE")) {
        return true;
    }
    return resident_blend_enabled() && sfpu_cull_enabled() && !fused_tile_enabled();
}

// Chain sort publish -> cull -> blend on one CQ drain (skip sort publish Finish +
// sort_P_kept D2H). Default ON for resident publish + (FUSED_TILE or ideal bucket).
inline bool sort_blend_pipe_enabled() {
    if (flag_is_zero("GSPLAT_TT_SORT_BLEND_PIPE")) {
        return false;
    }
    if (flag_is_one("GSPLAT_TT_SORT_BLEND_PIPE")) {
        return true;
    }
    if (!resident_blend_enabled()) {
        return false;
    }
    if (!flag_is_one("GSPLAT_TT_SORT_DEVICE_PUBLISH")) {
        return false;
    }
    if (fused_tile_enabled()) {
        return true;
    }
    return tile_bucket_enabled() && sfpu_cull_enabled();
}

// Blend writer fully overwrites res_out each frame — skip the ~6MB zero H2D.
inline bool blend_skip_zero_out_enabled() {
    if (flag_is_zero("GSPLAT_TT_BLEND_SKIP_ZERO_OUT")) {
        return false;
    }
    if (flag_is_one("GSPLAT_TT_BLEND_SKIP_ZERO_OUT")) {
        return true;
    }
    return resident_blend_enabled();
}

}  // namespace gsplat_tt::env_config
