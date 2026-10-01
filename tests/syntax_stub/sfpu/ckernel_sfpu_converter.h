#pragma once
#include <cstdint>
namespace ckernel::sfpu { struct Converter { static float as_float(uint32_t v) { return __builtin_bit_cast(float, v); } }; }
