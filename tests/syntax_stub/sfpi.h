// Syntax-only sfpi stub (task #99 local checks).
#pragma once
#include <cstdint>
namespace sfpi {
struct vFloat; struct vInt; struct vUInt;
struct vCond { vCond operator&&(const vCond&) const; vCond operator||(const vCond&) const; };
struct vFloat {
    vFloat(); vFloat(float);
    vFloat operator+(const vFloat&) const; vFloat operator-(const vFloat&) const;
    vFloat operator*(const vFloat&) const; vFloat operator-() const;
    vCond operator<(const vFloat&) const; vCond operator<=(const vFloat&) const;
    vCond operator>(const vFloat&) const; vCond operator>=(const vFloat&) const;
    vCond operator==(const vFloat&) const; vCond operator!=(const vFloat&) const;
};
struct vInt {
    vInt(); vInt(int);
    vInt operator+(const vInt&) const; vInt operator-(const vInt&) const;
    vInt operator|(const vInt&) const; vInt operator&(const vInt&) const;
    vInt operator<<(int) const;
    vCond operator<(const vInt&) const; vCond operator>=(const vInt&) const;
    vCond operator==(const vInt&) const; vCond operator!=(const vInt&) const;
};
struct vUInt {
    vUInt(); vUInt(unsigned);
    vUInt operator<<(int) const; vUInt operator>>(int) const;
    vUInt operator|(const vUInt&) const; vUInt operator&(const vUInt&) const;
};
struct DstProxy { operator vFloat() const; DstProxy& operator=(const vFloat&); };
struct DstReg { DstProxy operator[](int) const; void operator+=(int) const; void operator++(int) const; };
extern DstReg dst_reg;
template <class T, class U> T reinterpret(const U&);
void vec_min_max(vFloat&, vFloat&);
vFloat approx_recip(const vFloat&);
vFloat abs(const vFloat&);
vInt exexp(const vFloat&);
enum class RoundMode { NearestEven };
vFloat int32_to_float(const vInt&, RoundMode);
}  // namespace sfpi
#define sfpi_inline inline
#define v_if(c) { (void)(c);
#define v_endif }
