// Task #393: the cross-view overlap and pinned output are the default render
// path; GSPLAT_TT_XVIEW_OVERLAP=0 / GSPLAT_TT_OUT_PINNED=0 opt out.
//   tests/unit/run_cpp.sh tests/unit/test_xvpin_default_t393.cpp
#include "render/host/env_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    using namespace gsplat_tt::env_config;
    if (argc > 1 && std::strcmp(argv[1], "optout") == 0) {
        // Child: both flags were set to 0 before the first (cached) read.
        if (xview_overlap() || out_pinned()) { std::puts("FAIL: =0 did not opt out"); return 1; }
        return 0;
    }
    unsetenv("GSPLAT_TT_XVIEW_OVERLAP");
    unsetenv("GSPLAT_TT_OUT_PINNED");
    unsetenv("GSPLAT_TT_OUT_ZEROCOPY");
    if (!xview_overlap()) { std::puts("FAIL: xview_overlap() not on by default"); return 1; }
    if (!out_pinned()) { std::puts("FAIL: out_pinned() not on by default"); return 1; }
    if (out_zerocopy()) { std::puts("FAIL: out_zerocopy() on by default"); return 1; }
    setenv("GSPLAT_TT_XVIEW_OVERLAP", "0", 1);
    setenv("GSPLAT_TT_OUT_PINNED", "0", 1);
    const std::string child = std::string("'") + argv[0] + "' optout";
    if (std::system(child.c_str()) != 0) return 1;
    return 0;
}
