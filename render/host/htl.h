#pragma once
//
// htl.h — task #614 host timeline (profiling only). GSPLAT_TT_HTL=1 records a
// steady_clock stamp at every stage-timer Span begin/end (stage_timers.h) and at
// the named points below (program enqueue begin/end, blocking waits), into a
// fixed in-memory ring that is written once at exit to $GSPLAT_TT_HTL_OUT
// (default htl.txt): one "<ns> <B|E|P> <tag>" line per mark. Unset: one cached
// getenv read per mark, nothing recorded, no pixel change.
//
// Two slack probes for the b2b floor (also profiling only, default 0):
//   GSPLAT_TT_HTL_FOLD_SPIN_US  host busy-wait between the next view's pfwc
//                               enqueue and this view's mat+blend enqueue
//   GSPLAT_TT_PFWC_SPIN_US      device busy-wait at the end of the pfwc writer
//                               (pfwc_device.cpp, define PFWC_SPIN_CYC)

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace gsplat_tt::htl {

inline bool on() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_HTL");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

struct Mark {
    const char* tag;
    char kind;
    int64_t ns;
};

constexpr std::size_t kCap = 1u << 21;

inline Mark* ring() {
    static Mark* r = static_cast<Mark*>(std::calloc(kCap, sizeof(Mark)));
    return r;
}

inline std::atomic<std::size_t>& count() {
    static std::atomic<std::size_t> n{0};
    return n;
}

inline void dump() {
    const char* p = std::getenv("GSPLAT_TT_HTL_OUT");
    FILE* f = std::fopen(p != nullptr && p[0] != '\0' ? p : "htl.txt", "w");
    if (f == nullptr) return;
    const std::size_t n = std::min(count().load(), kCap);
    for (std::size_t i = 0; i < n; i++)
        std::fprintf(f, "%lld %c %s\n", static_cast<long long>(ring()[i].ns), ring()[i].kind,
                     ring()[i].tag);
    std::fclose(f);
    std::fprintf(stderr, "HTL wrote %zu marks\n", n);
}

inline void mark_at(const char* tag, char kind, std::chrono::steady_clock::time_point t) {
    if (!on()) return;
    static const bool reg = [] { return std::atexit(dump) == 0; }();
    (void)reg;
    const std::size_t i = count().fetch_add(1);
    if (i >= kCap) return;
    ring()[i] = Mark{tag, kind,
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count()};
}

inline void mark(const char* tag, char kind = 'P') {
    if (on()) mark_at(tag, kind, std::chrono::steady_clock::now());
}

// Busy-wait `us` microseconds (slack probes).
inline void spin_us(double us) {
    if (us <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {
    }
}

inline double env_us(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr ? std::atof(e) : 0.0;
}

// Scoped B/E pair (host zones without Tracy, host_tracy.hpp).
struct Zone {
    const char* tag;
    explicit Zone(const char* t) : tag(t) { mark(tag, 'B'); }
    ~Zone() { mark(tag, 'E'); }
    Zone(const Zone&) = delete;
    Zone& operator=(const Zone&) = delete;
};

}  // namespace gsplat_tt::htl

#define GSPLAT_HTL_CAT2(a, b) a##b
#define GSPLAT_HTL_CAT(a, b) GSPLAT_HTL_CAT2(a, b)
#define GSPLAT_HTL_ZONE(name) ::gsplat_tt::htl::Zone GSPLAT_HTL_CAT(htl_zone_, __LINE__)(name)
