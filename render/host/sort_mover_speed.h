// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort_mover_speed.h — task #174: measured one-launch emit speed of every sort
// mover (BRISC on NOC0, NCRISC on NOC1), by physical NoC core (x, y). Records
// per cycle relative to the mean mover (= 1000), from the GSPLAT_TT_OL_EMIT_PROF
// Tracy capture t166-p2f (yyzo-bh-07 p100a, bicycle 30 views, PRECULL=2, even
// split): `opt/profiler/emit_cores.py dev30.csv 30 --weights`. The slow movers
// are NOC0's top rows (y = 2, 3: brec read issue stalls) and NOC1's right
// columns (x = 14, 15). The host gives each mover pages in proportion to this.
// Cores missing from the table count as 1000.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "vis_mode.h"

namespace gsplat_tt::sort_split {

struct MoverSpeed {
    uint8_t x, y;
    uint16_t brisc, ncrisc;
};

inline constexpr MoverSpeed kMoverSpeedP150[] = {
    {1, 2, 731, 1053},
    {2, 2, 706, 1040},
    {3, 2, 701, 1037},
    {4, 2, 736, 1035},
    {5, 2, 774, 1027},
    {6, 2, 789, 1057},
    {11, 2, 813, 1090},
    {12, 2, 791, 1049},
    {13, 2, 799, 1033},
    {14, 2, 810, 975},
    {15, 2, 838, 964},
    {1, 3, 897, 1035},
    {2, 3, 849, 1041},
    {3, 3, 828, 1057},
    {4, 3, 827, 1064},
    {5, 3, 826, 1065},
    {6, 3, 840, 1056},
    {11, 3, 897, 1069},
    {12, 3, 895, 1077},
    {13, 3, 913, 1026},
    {14, 3, 928, 957},
    {15, 3, 932, 962},
    {1, 4, 1001, 1040},
    {2, 4, 989, 1031},
    {3, 4, 975, 1035},
    {4, 4, 992, 1040},
    {5, 4, 977, 1015},
    {6, 4, 978, 1008},
    {11, 4, 994, 1024},
    {12, 4, 997, 1020},
    {13, 4, 987, 996},
    {14, 4, 983, 932},
    {15, 4, 980, 922},
    {1, 5, 1018, 1037},
    {2, 5, 1011, 1035},
    {3, 5, 1009, 1032},
    {4, 5, 1006, 1027},
    {5, 5, 1005, 1019},
    {6, 5, 1007, 1015},
    {11, 5, 1014, 1014},
    {12, 5, 1011, 1006},
    {13, 5, 1012, 985},
    {14, 5, 1013, 901},
    {15, 5, 1012, 887},
    {1, 6, 1023, 1035},
    {2, 6, 1022, 1043},
    {3, 6, 1023, 1041},
    {4, 6, 1025, 1038},
    {5, 6, 1027, 1029},
    {6, 6, 1032, 1020},
    {11, 6, 1040, 1025},
    {12, 6, 1035, 1010},
    {13, 6, 1037, 992},
    {14, 6, 1037, 907},
    {15, 6, 1029, 889},
    {1, 7, 1035, 1039},
    {2, 7, 1032, 1038},
    {3, 7, 1031, 1037},
    {4, 7, 1027, 1029},
    {5, 7, 1028, 1023},
    {6, 7, 1032, 1017},
    {11, 7, 1030, 1017},
    {12, 7, 1030, 1009},
    {13, 7, 1034, 996},
    {14, 7, 1037, 920},
    {15, 7, 1035, 895},
    {1, 8, 1049, 1045},
    {2, 8, 1050, 1038},
    {3, 8, 1047, 1034},
    {4, 8, 1045, 1031},
    {5, 8, 1046, 1024},
    {6, 8, 1048, 1015},
    {11, 8, 1041, 1015},
    {12, 8, 1034, 1006},
    {13, 8, 1038, 996},
    {14, 8, 1038, 925},
    {15, 8, 1040, 902},
    {1, 9, 1040, 1036},
    {2, 9, 1041, 1043},
    {3, 9, 1041, 1039},
    {4, 9, 1045, 1036},
    {5, 9, 1047, 1032},
    {6, 9, 1047, 1026},
    {11, 9, 1044, 1026},
    {12, 9, 1044, 1012},
    {13, 9, 1043, 1006},
    {14, 9, 1037, 945},
    {15, 9, 1036, 925},
    {1, 10, 1037, 1032},
    {2, 10, 1037, 1035},
    {3, 10, 1037, 1030},
    {4, 10, 1041, 1026},
    {5, 10, 1046, 1019},
    {6, 10, 1049, 1010},
    {11, 10, 1050, 1020},
    {12, 10, 1047, 1010},
    {13, 10, 1047, 1000},
    {14, 10, 1047, 967},
    {15, 10, 1041, 942},
    {1, 11, 1040, 1045},
    {2, 11, 1040, 1050},
    {3, 11, 1039, 1045},
    {4, 11, 1036, 1040},
    {5, 11, 1041, 1035},
    {6, 11, 1039, 1019},
    {11, 11, 1045, 1032},
    {12, 11, 1047, 1019},
    {13, 11, 1050, 1004},
    {14, 11, 1048, 977},
    {15, 11, 1046, 984},
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

// Mover speeds (BRISC, NCRISC per core, core order) of the cores at NoC
// x | y << 16; cores missing from the table count as 1000.
inline std::vector<uint32_t> mover_speeds(const std::vector<uint32_t>& noc_xy) {
    std::vector<uint32_t> speed(2u * noc_xy.size(), 1000u);
    for (std::size_t c = 0; c < noc_xy.size(); c++) {
        for (const auto& m : kMoverSpeedP150) {
            if (m.x == (noc_xy[c] & 0xFFFFu) && m.y == (noc_xy[c] >> 16)) {
                speed[2u * c] = m.brisc;
                speed[2u * c + 1u] = m.ncrisc;
            }
        }
    }
    return speed;
}

}  // namespace gsplat_tt::sort_split
