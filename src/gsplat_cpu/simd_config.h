#pragma once

// Compile-time SIMD backend selection for the blend microblock kernels.
//
// Priority (first match wins):
//   GSPLAT_SCALAR_ONLY  — force scalar everywhere (Mac reference / CI)
//   GSPLAT_SIMD_AVX2    — x86 AVX2 path (bh-30 default when available)
//   __ARM_NEON          — Apple Silicon NEON path (Mac default)
//   otherwise           — scalar fallback

#if defined(GSPLAT_SCALAR_ONLY)
#define GSPLAT_HAS_NEON 0
#define GSPLAT_HAS_AVX2 0
#elif defined(GSPLAT_SIMD_AVX2) && defined(__AVX2__)
#define GSPLAT_HAS_NEON 0
#define GSPLAT_HAS_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON)
#define GSPLAT_HAS_NEON 1
#define GSPLAT_HAS_AVX2 0
#else
#define GSPLAT_HAS_NEON 0
#define GSPLAT_HAS_AVX2 0
#endif
