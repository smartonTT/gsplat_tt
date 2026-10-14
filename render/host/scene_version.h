// Task #671: explicit scene version for the device-resident scene uploads.
//
// pfwc (means, cov3d), project (means) and gather_visible (colours,
// opacities) keep the scene on the device and skip the pack + upload when the
// inputs are the ones already uploaded. Keyed by host pointer and N alone, an
// in-place parameter update (training: same tensor, new values) hit the cache
// and rendered the old scene. The caller now bumps the scene version after
// every in-place update (render_clean.bump_scene_version()); a cache hits only
// when pointer, N and version all match. The pointer and N stay in the key so
// that a different tensor still misses without a bump.
//
// The xview prefetch key (xview.h) carries the version too, so a pfwc queued
// before a bump is never consumed after it.
//
// Device-free so tests/unit/test_scene_version_t671.cpp can check it on the host.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace gsplat_tt::scene_version {

namespace detail {
inline std::atomic<uint64_t>& counter() {
    static std::atomic<uint64_t> v{1};
    return v;
}
}  // namespace detail

// The current scene version (starts at 1).
inline uint64_t current() { return detail::counter().load(std::memory_order_acquire); }

// Call after changing scene parameters in place. Returns the new version.
inline uint64_t bump() { return detail::counter().fetch_add(1, std::memory_order_acq_rel) + 1; }

// What one cached upload holds: the host pointer, N and the scene version it
// was uploaded at. version 0 = nothing uploaded (current() is never 0).
struct UploadKey {
    const void* ptr = nullptr;
    std::size_t n = 0;
    uint64_t version = 0;

    bool hit(const void* p, std::size_t count, uint64_t v) const {
        return version != 0 && ptr == p && n == count && version == v;
    }
    void record(const void* p, std::size_t count, uint64_t v) {
        ptr = p;
        n = count;
        version = v;
    }
    // The device buffers were (re)allocated: nothing on them is valid.
    void invalidate() { *this = UploadKey{}; }
};

}  // namespace gsplat_tt::scene_version
