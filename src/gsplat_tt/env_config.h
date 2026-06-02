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

}  // namespace gsplat_tt::env_config
