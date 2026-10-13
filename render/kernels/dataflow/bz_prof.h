// SPDX-License-Identifier: Apache-2.0
// Task #630: profiler-only phase counters for the blend part of the fused mat+blend
// program (host env GSPLAT_TT_BZ_PROF=1 -> BZ_PROF, needs a device-profiler build).
// Each RISC sums wall-clock cycles per phase and reports them once at kernel end as
// "bz_*" timestamped-data markers (docs/profile-t630/bz630.py). On the TRISCs only
// MATH counts. Untraced builds compile none of it.
#pragma once
#include <cstdint>
#if defined(BZ_PROF) && BZ_PROF && defined(PROFILE_KERNEL) && !defined(TRISC_UNPACK) && !defined(TRISC_PACK)
#define BZ_ON 1
inline uint32_t bz_now() { return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
struct BzAcc {
    uint32_t& a;
    const uint32_t t0;
    explicit BzAcc(uint32_t& x) : a(x), t0(bz_now()) {}
    ~BzAcc() { a += bz_now() - t0; }
};
#define BZ_CAT2(a, b) a##b
#define BZ_CAT(a, b) BZ_CAT2(a, b)
#define BZ_ACC(var) BzAcc BZ_CAT(bz_acc_, __LINE__)(var)
#define BZ_DO(...) __VA_ARGS__
#else
#define BZ_ON 0
#define BZ_ACC(var) ((void)0)
#define BZ_DO(...)
#endif
