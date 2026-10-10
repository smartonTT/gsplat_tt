// Task #471: the pfwc 32 B blend records (GSPLAT_TT_PFWC_REC32) are the default;
// =0 opts out, and every kill switch of the chain they need turns them off
// instead of tripping the host guards.
//   tests/unit/run_cpp.sh tests/unit/test_pfwc_rec32_default_t471.cpp
#include "render/host/env_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const char* const kKnobs[] = {
    "GSPLAT_TT_PFWC_REC32",   "GSPLAT_TT_PFWC_WRITER_SPLIT", "GSPLAT_TT_EMIT_PUBOC",
    "GSPLAT_TT_OL_EMIT_TOWN", "GSPLAT_TT_OL_RING",           "GSPLAT_TT_SORT_ONELAUNCH",
    "GSPLAT_TT_OL_BREC_BULK", "GSPLAT_TT_PFWC_FUSE",         "GSPLAT_TT_SFPU_VIS",
    "GSPLAT_TT_OL_PB",
};

int main(int argc, char** argv) {
    using namespace gsplat_tt::env_config;
    if (argc > 1 && std::strcmp(argv[1], "off") == 0) {
        // Child: one knob was set before the first (cached) read.
        if (pfwc_rec32()) { std::printf("FAIL: rec32 still on with %s\n", argv[2]); return 1; }
        return 0;
    }
    for (const char* k : kKnobs) unsetenv(k);
    if (!pfwc_rec32()) { std::puts("FAIL: pfwc_rec32() not on by default"); return 1; }
    const std::string self = std::string("'") + argv[0] + "'";
    struct { const char* knob; const char* val; } offs[] = {
        {"GSPLAT_TT_PFWC_REC32", "0"},     {"GSPLAT_TT_PFWC_WRITER_SPLIT", "0"},
        {"GSPLAT_TT_EMIT_PUBOC", "0"},     {"GSPLAT_TT_OL_EMIT_TOWN", "0"},
        {"GSPLAT_TT_OL_RING", "0"},        {"GSPLAT_TT_SORT_ONELAUNCH", "0"},
        {"GSPLAT_TT_OL_BREC_BULK", "0"},   {"GSPLAT_TT_PFWC_FUSE", "0"},
        {"GSPLAT_TT_SFPU_VIS", "2"},       {"GSPLAT_TT_OL_PB", "16"},
    };
    for (const auto& o : offs) {
        setenv(o.knob, o.val, 1);
        const std::string cmd = self + " off " + o.knob + "=" + o.val;
        if (std::system(cmd.c_str()) != 0) return 1;
        unsetenv(o.knob);
    }
    std::puts("test_pfwc_rec32_default_t471: ok");
    return 0;
}
