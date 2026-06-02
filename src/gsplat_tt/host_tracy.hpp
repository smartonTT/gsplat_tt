#pragma once

// Host-side Tracy zones for render orchestration gaps (Finish / LPT / H2D).
// Active when tt-metal was built with Tracy (libtracy present) and this TU
// is compiled with TRACY_ENABLE (see cmake/TtMetalInTree.cmake).

#if defined(TRACY_ENABLE)
#include <tracy/Tracy.hpp>
#define GSPLAT_HOST_ZONE(name) ZoneScopedN(name)
#define GSPLAT_HOST_ZONE_C(name) ZoneScopedNC(name, tracy::Color::SteelBlue)
#define GSPLAT_HOST_FRAME_MARK(name) FrameMarkNamed(name)
#else
#define GSPLAT_HOST_ZONE(name) ((void)0)
#define GSPLAT_HOST_ZONE_C(name) ((void)0)
#define GSPLAT_HOST_FRAME_MARK(name) ((void)0)
#endif
