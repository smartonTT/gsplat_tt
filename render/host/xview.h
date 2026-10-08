// Task #379 (GSPLAT_TT_XVIEW_OVERLAP): cross-view overlap bookkeeping.
//
// With a next-view hint, render_view enqueues view N+1's fused pfwc right
// behind view N's blend (blend_set_after_enqueue_hook), before the host waits
// for and reads view N's image. The device then runs pfwc N+1 while the host
// reads/copies N, returns to Python and starts N+1, instead of idling.
//
// Prefetch records the run_project inputs that prefetch used. The next
// run_project consumes it: on an exact key match it skips pfwc_tt (its outputs
// and the pfwc globals are already those of this view); on any mismatch (a
// different pose, scene, size, floor, or an overflow retry of view N) it runs
// pfwc_tt again, so a wrong hint only costs one wasted pfwc, never a wrong
// image. A prefetch is consumed once; any later run_project runs pfwc.
//
// Device-free so tests/unit/test_xview_t379.cpp can check it on the host.
#pragma once

#include <cstddef>
#include <cstring>

namespace gsplat_tt::xview {

// Every run_project input the pfwc enqueue depends on. The env modes
// (SFPU_VIS / PFWC_FUSE / PRECULL / VIS_EDGE_TAU) are process constants and
// the scene uploads are cached by pointer and N, as in pfwc_tt itself.
struct Key {
    const float* means = nullptr;
    const float* cov3d = nullptr;
    const float* colors = nullptr;
    const float* opacities = nullptr;
    std::size_t n = 0;
    float extr[16] = {};
    float intr[9] = {};
    float min_opacity = 0.0f;
    int image_height = 0;
    int image_width = 0;
    int max_radius = 0;
    int tile_size = 0;
    float mb_contrib_floor = 0.0f;
    bool cull_disabled = false;
};

inline Key make_key(const float* means, const float* cov3d, const float* extrinsics,
                    const float* intrinsics, const float* colors, const float* opacities,
                    float min_opacity, std::size_t n, int image_height, int image_width,
                    int max_radius, int tile_size, float mb_contrib_floor, bool cull_disabled) {
    Key k;
    k.means = means;
    k.cov3d = cov3d;
    k.colors = colors;
    k.opacities = opacities;
    k.n = n;
    std::memcpy(k.extr, extrinsics, sizeof(k.extr));
    std::memcpy(k.intr, intrinsics, sizeof(k.intr));
    k.min_opacity = min_opacity;
    k.image_height = image_height;
    k.image_width = image_width;
    k.max_radius = max_radius;
    k.tile_size = tile_size;
    k.mb_contrib_floor = mb_contrib_floor;
    k.cull_disabled = cull_disabled;
    return k;
}

// Bitwise on the floats: the prefetch must have used exactly these values.
inline bool same(const Key& a, const Key& b) {
    return a.means == b.means && a.cov3d == b.cov3d && a.colors == b.colors &&
           a.opacities == b.opacities && a.n == b.n &&
           std::memcmp(a.extr, b.extr, sizeof(a.extr)) == 0 &&
           std::memcmp(a.intr, b.intr, sizeof(a.intr)) == 0 &&
           std::memcmp(&a.min_opacity, &b.min_opacity, sizeof(float)) == 0 &&
           a.image_height == b.image_height && a.image_width == b.image_width &&
           a.max_radius == b.max_radius && a.tile_size == b.tile_size &&
           std::memcmp(&a.mb_contrib_floor, &b.mb_contrib_floor, sizeof(float)) == 0 &&
           a.cull_disabled == b.cull_disabled;
}

class Prefetch {
public:
    // A pfwc for `k` is on the queue (the last pfwc enqueued).
    void set(const Key& k) {
        key_ = k;
        pending_ = true;
    }
    // Called by every run_project: true = skip pfwc_tt (the queued one is this
    // view's). Clears the prefetch either way.
    bool consume(const Key& k) {
        const bool was = pending_;
        const bool hit = was && same(key_, k);
        pending_ = false;
        if (hit) ++hits_;
        else if (was) ++misses_;
        return hit;
    }
    // A pfwc ran for some other reason (not via run_project), so the queued
    // outputs are no longer the prefetched view's.
    void clear() { pending_ = false; }
    bool pending() const { return pending_; }
    std::size_t hits() const { return hits_; }
    std::size_t misses() const { return misses_; }

private:
    Key key_{};
    bool pending_ = false;
    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
};

}  // namespace gsplat_tt::xview
