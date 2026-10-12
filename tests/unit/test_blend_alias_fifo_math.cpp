// Task #544: the device wrapper blend_fifo_shrink (render/kernels/blend_alias_fifo.h)
// must not touch the CB interface on the math TRISC, which has no cb_interface
// symbol (the #544 bh-33 run failed to link trisc1 of matblend_compute on it).
//
//   tests/unit/run_cpp.sh tests/unit/test_blend_alias_fifo_math.cpp
//
// Builds the header as the math TRISC sees it: BLEND_ALIAS_CUT and TRISC_MATH set and
// no get_local_cb_interface / cb_addr_shift declared. Any reference to them fails to
// compile, the host stand-in for the device link failure.
#include <cstdint>
#include <cstdio>

#define FORCE_INLINE inline __attribute__((always_inline))
#define BLEND_ALIAS_CUT 1
#define TRISC_MATH 1
#include "render/kernels/blend_alias_fifo.h"

int main() {
    blend_fifo_shrink(19, 6 * 4096, 4096);
    std::printf("test_blend_alias_fifo_math: OK\n");
    return 0;
}
