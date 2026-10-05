#pragma once
#define DeviceZoneScopedN(n) (void)n
#include <cstdint>
void DeviceTimestampedData(const char*, uint64_t);  // task #202 (EMIT_PROF variants)
#ifndef tt_reg_ptr
#define tt_reg_ptr
#endif
#ifndef RISCV_DEBUG_REG_WALL_CLOCK_L
#define RISCV_DEBUG_REG_WALL_CLOCK_L 0xFFB121F0u
#endif
