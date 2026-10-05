// Check for the gather's visibility mask (task #75): the lazy-load predicate
// gather_pred::visible_at (count pass, fast sub_lt_pos path) equals the old
// visible_bits on every element, and walking the 1024-bit tile mask with
// ctz visits the visible elements in the same increasing order as the old
// linear scan. Standalone:
//
//   c++ -O2 -ffp-contract=off -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_gather_visible_mask.cpp -o /tmp/t_gvm && /tmp/t_gvm
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gather_visible_pred.h"

static uint32_t u(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

int main() {
    const uint32_t k_near = u(0.2f), min_op = u(1.0f / 255.0f), W = u(1024.0f), H = u(1024.0f),
                   maxr = u(512.0f);
    const uint32_t special[] = {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 1u,
                                u(1024.0f), u(-1024.0f), u(512.0f), u(0.2f), u(1.0f / 255.0f)};
    uint64_t x = 0x243F6A8885A308D3ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    uint64_t bad = 0, vis = 0, total = 0;
    std::vector<uint32_t> dep(1024), op(1024), mx(1024), my(1024), rx(1024), ry(1024);
    for (int tile = 0; tile < 20000; tile++) {
        for (int il = 0; il < 1024; il++) {
            const uint64_t r = rnd();
            auto pick = [&](float lo, float hi, int sh) -> uint32_t {
                const uint64_t q = rnd();
                if (((q >> 60) & 15) == 0) return special[(q >> 32) % 11];
                if (((q >> 60) & 15) == 1) return static_cast<uint32_t>(q);  // raw bits
                return u(lo + (hi - lo) * static_cast<float>((q >> sh) & 0xFFFFF) / 1048576.0f);
            };
            dep[il] = pick(-5.0f, 50.0f, 3);
            op[il] = pick(0.0f, 0.05f, 5);
            mx[il] = pick(-1500.0f, 2500.0f, 7);
            my[il] = pick(-1500.0f, 2500.0f, 9);
            rx[il] = (r & 7) == 0 ? mx[il] : pick(-2.0f, 700.0f, 11);  // incl. boundary ties
            ry[il] = pick(-2.0f, 700.0f, 13);
        }
        // Count pass: mask words.
        uint32_t mask[32] = {};
        for (uint32_t il = 0; il < 1024; il++)
            if (gather_pred::visible_at(dep.data(), op.data(), mx.data(), my.data(), rx.data(),
                                        ry.data(), il, k_near, min_op, W, H, maxr))
                mask[il / 32] |= 1u << (il % 32);
        // Old scatter order vs mask walk.
        std::vector<uint32_t> ref, got;
        for (uint32_t il = 0; il < 1024; il++)
            if (gather_pred::visible_bits(dep[il], op[il], mx[il], my[il], rx[il], ry[il], k_near,
                                          min_op, W, H, maxr))
                ref.push_back(il);
        for (uint32_t w = 0; w < 32; w++)
            for (uint32_t b = mask[w]; b != 0; b &= b - 1)
                got.push_back(w * 32 + static_cast<uint32_t>(__builtin_ctz(b)));
        if (got != ref && bad++ < 10) std::printf("tile %d: mask walk != linear scan\n", tile);
        vis += ref.size();
        total += 1024;
    }
    std::printf("elements=%llu visible=%llu mismatched_tiles=%llu\n%s\n",
                (unsigned long long)total, (unsigned long long)vis, (unsigned long long)bad,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
