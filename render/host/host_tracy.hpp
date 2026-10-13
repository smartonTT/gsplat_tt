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
// Task #614: without Tracy the host zones feed the GSPLAT_TT_HTL timeline (htl.h).
#include "htl.h"
#define GSPLAT_HOST_ZONE(name) GSPLAT_HTL_ZONE(name)
#define GSPLAT_HOST_ZONE_C(name) GSPLAT_HTL_ZONE(name)
#define GSPLAT_HOST_FRAME_MARK(name) ::gsplat_tt::htl::mark(name)
#endif
