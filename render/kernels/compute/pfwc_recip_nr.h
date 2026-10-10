// Task #266 (GSPLAT_TT_PFWC_RECIP_NEWTON=1): accurate 1/tz for pfwc step 2.
// recip_tile() takes the legacy_compat path (_reciprocal_compat_<3>: a linear seed
// on the mantissa in [0.5, 1) plus two Newton steps). Its seed error reaches 0.196 at
// mantissa -> 1, so 1/z is off by up to (0.196)^4 = 1.5e-3 relative for z just below a
// power of two (task #260: means p99 0.41 px off-centre, conic p99 1.7e-3). Here the
// SFPARECIP seed (approx_recip) gets two Newton steps, the same form as the conic
// reciprocal (pfwc_conic_one): ~1 ulp. tz <= k_near rows are culled; tz == 0 gives NaN.
#pragma once

#include <cstdint>

// One Newton step y <- y (2 - x y); each step squares the relative error.
template <typename V>
inline V pfwc_recip_nr_step(V x, V y) {
    return y * (V(2.0f) - x * y);
}

#ifdef TRISC_MATH
// In place on DEST tile TILE (32 vectors). Call between SFPU start(0) and done.
template <uint32_t TILE = 0>
inline void pfwc_inv_tz_nr() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        vFloat x = dst_reg[TILE * 32];
        vFloat y = approx_recip(x);
        y = pfwc_recip_nr_step(x, y);
        y = pfwc_recip_nr_step(x, y);
        dst_reg[TILE * 32] = y;
        dst_reg++;
    }
}
#endif
