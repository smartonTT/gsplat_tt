// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Extra Tensix kernel config buffer (KB above the default 69 KB) the device opens with.
// Header-only so tests/unit/test_kcfg_size.cpp can check it without tt-metal.
#pragma once

#include <cstdlib>

namespace gsplat_tt {

// TT_METAL_DEVICE_PROFILER set and not "0".
inline bool kcfg_profiler_on(const char* prof_env) {
    return prof_env != nullptr && prof_env[0] != '\0' && !(prof_env[0] == '0' && prof_env[1] == '\0');
}

// extra_env: GSPLAT_TT_KCFG_EXTRA_KB (nullptr if unset); an explicit value wins.
// Task #207/#221: the split pfwc writer adds the writer code to the NCRISC kernel
// (92496 B program, too large at +8 and +16 KB); SFPU cov_cam alone needs ~4 KB.
// split_fused: the split is on and builds (it only builds with the fused writer).
// Task #258: the device profiler grows every kernel binary; the default program then
// overflows +24 KB (TT_FATAL state.offset <= max_size in pfwc) and fits at +32 KB.
inline long kcfg_extra_kb(const char* extra_env, bool split_fused, bool covcam_sfpu, bool profiler) {
    if (extra_env != nullptr) return std::atol(extra_env);
    long kb = split_fused ? 24 : (covcam_sfpu ? 8 : 0);
    if (profiler) kb += 8;
    return kb;
}

}  // namespace gsplat_tt
