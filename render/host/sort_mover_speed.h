// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort_mover_speed.h — task #174: measured one-launch emit speed of every sort
// mover (BRISC on NOC0, NCRISC on NOC1), keyed by the translated NoC core (x, y)
// the host gets from worker_core_from_logical_core. Relative page share (mean
// mover = 1000). Task #177 refit from the GSPLAT_TT_OL_EMIT_PROF Tracy captures
// t174-p2w and t177-v1 (yyzo-bh-07 p100a, bicycle 30 views, PRECULL=2) with
// docs/mover-speed-t177/fit_table.py (fixed + per-record cost where a mover's
// record count moved, else proportional). Tracy prints a different core x:
// on yyzo-bh-07 its x 11..15 is translated x 7, 10..13 (the #174 table used
// Tracy x, so its rows for x 11..15 hit the wrong cores or none). The slow
// movers are NOC0's top rows (y = 2, 3: brec read issue stalls) and NOC1's
// right columns (translated x = 12, 13). Cores missing from the table count
// as 1000. Harvesting differs per board, so x 7..13 is yyzo-bh-07 specific.

#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

#include "vis_mode.h"

namespace gsplat_tt::sort_split {

struct MoverSpeed {
    uint8_t x, y;
    uint16_t brisc, ncrisc;
};

inline constexpr MoverSpeed kMoverSpeedP150[] = {
    {1, 2, 600, 1049},
    {2, 2, 537, 1049},
    {3, 2, 484, 1075},
    {4, 2, 541, 1066},
    {5, 2, 656, 1035},
    {6, 2, 742, 1033},
    {7, 2, 834, 1104},
    {10, 2, 804, 1086},
    {11, 2, 784, 1018},
    {12, 2, 797, 964},
    {13, 2, 809, 934},
    {1, 3, 898, 1044},
    {2, 3, 816, 1057},
    {3, 3, 764, 1078},
    {4, 3, 767, 1074},
    {5, 3, 800, 1049},
    {6, 3, 840, 1036},
    {7, 3, 931, 1053},
    {10, 3, 920, 1046},
    {11, 3, 912, 1020},
    {12, 3, 922, 977},
    {13, 3, 929, 973},
    {1, 4, 1016, 1060},
    {2, 4, 1004, 1046},
    {3, 4, 975, 1040},
    {4, 4, 966, 1022},
    {5, 4, 964, 1046},
    {6, 4, 987, 1011},
    {7, 4, 983, 1033},
    {10, 4, 982, 1024},
    {11, 4, 983, 1002},
    {12, 4, 992, 931},
    {13, 4, 989, 913},
    {1, 5, 1027, 1049},
    {2, 5, 1022, 1056},
    {3, 5, 1013, 1043},
    {4, 5, 1014, 1037},
    {5, 5, 1014, 1026},
    {6, 5, 1011, 1013},
    {7, 5, 1020, 1033},
    {10, 5, 1027, 1018},
    {11, 5, 1019, 990},
    {12, 5, 1020, 893},
    {13, 5, 1021, 881},
    {1, 6, 1037, 1042},
    {2, 6, 1032, 1043},
    {3, 6, 1031, 1044},
    {4, 6, 1039, 1041},
    {5, 6, 1039, 1036},
    {6, 6, 1043, 1027},
    {7, 6, 1045, 1050},
    {10, 6, 1041, 1041},
    {11, 6, 1047, 1002},
    {12, 6, 1045, 902},
    {13, 6, 1046, 886},
    {1, 7, 1051, 1041},
    {2, 7, 1046, 1040},
    {3, 7, 1047, 1037},
    {4, 7, 1046, 1034},
    {5, 7, 1043, 1022},
    {6, 7, 1042, 1014},
    {7, 7, 1045, 1039},
    {10, 7, 1042, 1025},
    {11, 7, 1042, 994},
    {12, 7, 1044, 909},
    {13, 7, 1046, 894},
    {1, 8, 1047, 1046},
    {2, 8, 1052, 1049},
    {3, 8, 1055, 1040},
    {4, 8, 1053, 1032},
    {5, 8, 1053, 1023},
    {6, 8, 1054, 1014},
    {7, 8, 1053, 1035},
    {10, 8, 1046, 1019},
    {11, 8, 1042, 993},
    {12, 8, 1046, 918},
    {13, 8, 1045, 898},
    {1, 9, 1050, 1039},
    {2, 9, 1046, 1037},
    {3, 9, 1051, 1039},
    {4, 9, 1052, 1036},
    {5, 9, 1054, 1028},
    {6, 9, 1057, 1018},
    {7, 9, 1058, 1035},
    {10, 9, 1060, 1026},
    {11, 9, 1055, 999},
    {12, 9, 1060, 939},
    {13, 9, 1047, 922},
    {1, 10, 1050, 1031},
    {2, 10, 1046, 1030},
    {3, 10, 1049, 1025},
    {4, 10, 1050, 1023},
    {5, 10, 1048, 1020},
    {6, 10, 1049, 1009},
    {7, 10, 1047, 1038},
    {10, 10, 1056, 1019},
    {11, 10, 1058, 996},
    {12, 10, 1061, 958},
    {13, 10, 1054, 937},
    {1, 11, 1051, 1040},
    {2, 11, 1051, 1048},
    {3, 11, 1049, 1048},
    {4, 11, 1042, 1042},
    {5, 11, 1045, 1038},
    {6, 11, 1044, 1023},
    {7, 11, 1046, 1044},
    {10, 11, 1051, 1024},
    {11, 11, 1053, 1010},
    {12, 11, 1053, 971},
    {13, 11, 1051, 979},
};

// GSPLAT_TT_OL_MOVER_SPEED: the one-launch sort takes speed-proportional
// mover ranges (default 1 with GSPLAT_TT_PRECULL=2, else 0). Read once; the
// segment K2 reads it too, to count the same ranges (task #170 fold).
inline bool ol_mover_speed_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("GSPLAT_TT_OL_MOVER_SPEED");
        if (e == nullptr || *e == '\0') return gsplat_tt::precull_mode() == 2;
        return std::atoi(e) != 0;
    }();
    return v;
}

// Task #177: parses a mover speed table from text (GSPLAT_TT_OL_MOVER_SPEED_FILE)
// so a refit can be measured without a rebuild. Every line holding exactly
// four unsigned integers is one entry {x, y, brisc, ncrisc} (the initializer
// lines above parse as is); other lines are ignored. False, with *out empty,
// when there is no entry or one is out of range (x, y < 256, speeds 1..65535).
inline bool parse_mover_speed(const std::string& text, std::vector<MoverSpeed>* out) {
    out->clear();
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t e = text.find('\n', i);
        if (e == std::string::npos) e = text.size();
        std::vector<unsigned long> v;
        for (std::size_t k = i; k < e;) {
            if (!std::isdigit(static_cast<unsigned char>(text[k]))) {
                k++;
                continue;
            }
            char* end = nullptr;
            v.push_back(std::strtoul(text.c_str() + k, &end, 10));
            k = static_cast<std::size_t>(end - text.c_str());
        }
        if (v.size() == 4u) {
            if (v[0] > 255u || v[1] > 255u || v[2] == 0u || v[2] > 65535u || v[3] == 0u || v[3] > 65535u) {
                out->clear();
                return false;
            }
            out->push_back({static_cast<uint8_t>(v[0]), static_cast<uint8_t>(v[1]),
                            static_cast<uint16_t>(v[2]), static_cast<uint16_t>(v[3])});
        }
        i = e + 1u;
    }
    return !out->empty();
}

// The mover speed table in use: kMoverSpeedP150, or the table in the file
// GSPLAT_TT_OL_MOVER_SPEED_FILE (task #177, A/B a refit without a rebuild).
// Read once.
inline const std::vector<MoverSpeed>& mover_speed_table() {
    static const std::vector<MoverSpeed> v = [] {
        std::vector<MoverSpeed> t(std::begin(kMoverSpeedP150), std::end(kMoverSpeedP150));
        const char* path = std::getenv("GSPLAT_TT_OL_MOVER_SPEED_FILE");
        if (path == nullptr || *path == '\0') return t;
        std::string text;
        if (FILE* f = std::fopen(path, "rb")) {
            char buf[4096];
            std::size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
            std::fclose(f);
        }
        std::vector<MoverSpeed> parsed;
        if (!parse_mover_speed(text, &parsed)) {
            std::fprintf(stderr,
                         "[gsplat_tt::sort] GSPLAT_TT_OL_MOVER_SPEED_FILE=\"%s\" is missing or not a mover "
                         "speed table; using the built-in table\n",
                         path);
            return t;
        }
        std::fprintf(stderr, "[gsplat_tt::sort] mover speed table: %zu cores from %s\n", parsed.size(), path);
        return parsed;
    }();
    return v;
}

// Mover speeds (BRISC, NCRISC per core, core order) of the cores at NoC
// x | y << 16; cores missing from the table count as 1000. Both the sort and
// the segment K2 read mover_speed_table(), so a file override reaches both.
inline std::vector<uint32_t> mover_speeds(const std::vector<uint32_t>& noc_xy) {
    std::vector<uint32_t> speed(2u * noc_xy.size(), 1000u);
    for (std::size_t c = 0; c < noc_xy.size(); c++) {
        for (const auto& m : mover_speed_table()) {
            if (m.x == (noc_xy[c] & 0xFFFFu) && m.y == (noc_xy[c] >> 16)) {
                speed[2u * c] = m.brisc;
                speed[2u * c + 1u] = m.ncrisc;
            }
        }
    }
    return speed;
}

}  // namespace gsplat_tt::sort_split
