#pragma once
#include <cstdint>
void _llk_math_eltwise_unary_sfpu_start_(unsigned); void _llk_math_eltwise_unary_sfpu_done_();
// Raw SFPU instructions (task #206). TTI_ operands must be constants (".ttinsn" "n").
namespace ckernel {
struct p_sfpu {
    static constexpr uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3, LREG4 = 4, LREG5 = 5,
                              LREG6 = 6, LREG7 = 7, LCONST_0 = 9, LCONST_1 = 10;
};
constexpr uint8_t ADDR_MOD_7 = 7;
extern volatile uint32_t instrn_buffer[];
template <uint32_t... Ops> inline void tti() {}
}  // namespace ckernel
#define TTI_SFPLOAD(l, m, a, d) ckernel::tti<(l), (m), (a), (d)>()
#define TTI_SFPSTORE(l, m, a, d) ckernel::tti<(l), (m), (a), (d)>()
#define TTI_SFPMUL(a, b, c, d, m) ckernel::tti<(a), (b), (c), (d), (m)>()
#define TTI_SFPADD(a, b, c, d, m) ckernel::tti<(a), (b), (c), (d), (m)>()
#define TT_SFPLOAD(l, m, a, d) (ckernel::instrn_buffer[0] = (l) + (m) + (a) + (d))
#define TT_SFPSTORE(l, m, a, d) (ckernel::instrn_buffer[0] = (l) + (m) + (a) + (d))
#define TT_SFPLOADI(l, m, i) (ckernel::instrn_buffer[0] = (l) + (m) + (i))
