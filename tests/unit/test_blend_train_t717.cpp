// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Task #717: train-mode op/colour packs. The cull's opacity code must decode
// to >= the fp32 opacity (the cull stays conservative), and the fp32 side
// entry carries the scene values bit for bit.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>

#include "render/host/blend_train.h"
#include "render/kernels/dataflow/dm_fp32.h"
#include "render/kernels/dataflow/sort_bin_fp32.h"

static uint32_t bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static float flt(uint32_t b) { float v; std::memcpy(&v, &b, 4); return v; }

int main() {
    namespace bt = gsplat_tt::blend_train;
    int bad = 0;
    uint64_t n = 0;
    // Every fp32 in [0, 1] with a coarse mantissa stride, plus the edges.
    auto check = [&](float v) {
        uint32_t q;
        if (!sort_bin_fp32::unorm16(bits(v), &q)) { ++bad; return; }
        const uint32_t c = bt::cull_op16_from_nearest(q);
        const float d = flt(dm_fp32::unorm16_to_f(c));
        ++n;
        if (!(d >= v) || c > 65535u) {
            if (bad < 5) std::printf("FAIL op %.9g -> code %u decodes %.9g\n", v, c, d);
            ++bad;
        }
    };
    for (uint32_t b = 0; b <= bits(1.0f); b += 97u) check(flt(b));
    for (uint32_t q = 0; q <= 65535u; ++q) {
        const float k = static_cast<float>(q) / 65535.0f;
        check(k);
        check(std::nextafter(k, 2.0f) <= 1.0f ? std::nextafter(k, 2.0f) : 1.0f);
        check(k + 0.4999f / 65535.0f <= 1.0f ? k + 0.4999f / 65535.0f : 1.0f);
    }
    check(0.0f);
    check(1.0f);
    if (bt::cull_op16_from_nearest(65535u) != 65535u) ++bad;
    if (bt::kOpcBytes * 4u != bt::kOpcPageBytes) ++bad;
    std::printf("%s t717 cull-op conservative over %llu values\n", bad ? "FAIL" : "PASS",
                static_cast<unsigned long long>(n));
    return bad ? 1 : 0;
}
