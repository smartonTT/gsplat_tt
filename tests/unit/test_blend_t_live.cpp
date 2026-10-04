// Task #146: blend_t_live_fast == blend_t_live_ref (the original T-saturation
// reduce of alpha_blend_compute_mb.cpp) on random and edge bf16 T tiles.
//   tests/unit/run_cpp.sh tests/unit/test_blend_t_live.cpp
#include <cstdint>
#include <cstdio>
#include <random>

#include "render/kernels/compute/blend_t_live.h"

int main() {
    std::mt19937 rng(146);
    // Specials: +-0, +-inf, NaNs, smallest/largest, values around eps = 1/256.
    const uint16_t sp[] = {0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFC1, 0x0001,
                           0x7F7F, 0x3B80, 0x3B7F, 0x3B81, 0x3F80, 0x3F7F, 0xBB80, 0x0080};
    const uint32_t eps_list[] = {0x3B800000u, 0x3B800001u, 0x3B7FFFFFu, 0x3B80FFFFu, 0x3B810000u,
                                 0x00000001u, 0x00010000u, 0x7F800000u, 0x7F7FFFFFu, 0x3F800000u,
                                 0x00000000u, 0x80000000u, 0xBB800000u, 0x7FC00000u, 0x0000FFFFu};
    uint64_t checks = 0, bad = 0;
    uint32_t w[512];
    for (int it = 0; it < 20000; ++it) {
        const int mode = it % 4;
        for (uint32_t i = 0; i < 512; ++i) {
            uint32_t h[2];
            for (uint32_t j = 0; j < 2; ++j) {
                const uint32_t r = rng();
                if (mode == 0) h[j] = r & 0xffffu;                       // any bf16
                else if (mode == 1) h[j] = sp[r % 16];                   // specials
                else if (mode == 2) h[j] = 0x3B00u + (r & 0x1ffu);       // around eps
                else h[j] = (r & 7u) ? (0x3B80u - 1u - (r % 0x3B00u)) : (0x3B7Eu + (r >> 29));  // mostly dead
            }
            w[i] = h[0] | (h[1] << 16);
        }
        for (uint32_t eps : eps_list) {
            ++checks;
            const uint32_t a = blend_t_live_ref(w, eps), b = blend_t_live_fast(w, eps);
            if (a != b && bad++ < 10) std::printf("mismatch it=%d eps=%08x ref=%08x fast=%08x\n", it, eps, a, b);
        }
        const uint32_t reps = rng();
        ++checks;
        if (blend_t_live_ref(w, reps) != blend_t_live_fast(w, reps) && bad++ < 10)
            std::printf("mismatch it=%d eps=%08x\n", it, reps);
    }
    std::printf("%llu checks, %llu bad\n", (unsigned long long)checks, (unsigned long long)bad);
    return bad != 0;
}
