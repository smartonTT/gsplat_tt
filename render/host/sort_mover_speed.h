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
//
// Task #358: that table was measured on a p100a (it was misnamed P150) and
// mis-weights a real p150 (emit window 2.92 ms vs 1.34 on the p100a, rows
// y = 2-4 slow; docs/p150-gap.md). kMoverSpeedP150 is re-derived on bh-30
// (p150b) from the t356 EMIT capture by docs/p150-gap-t356/reweight.py
// (v = share / time, out/reweight-E.txt). The table is picked by the detected
// board (tt::tt_metal::GetClusterType(), see mover_board());
// GSPLAT_TT_MOVER_TABLE=p100a|p150 overrides it.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

#include "vis_mode.h"

namespace gsplat_tt::sort_split {

struct MoverSpeed {
    uint8_t x, y;
    uint16_t brisc, ncrisc;
};

inline constexpr MoverSpeed kMoverSpeedP100a[] = {
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

// Task #358: re-derived on p150b bh-30 (t356 EMIT capture, iter 207).
inline constexpr MoverSpeed kMoverSpeedP150[] = {
    {1, 2, 499, 1038},
    {2, 2, 473, 963},
    {3, 2, 483, 989},
    {4, 2, 523, 862},
    {5, 2, 579, 784},
    {6, 2, 629, 786},
    {11, 2, 555, 786},
    {12, 2, 600, 785},
    {13, 2, 736, 854},
    {14, 2, 822, 809},
    {15, 2, 938, 799},
    {1, 3, 588, 707},
    {2, 3, 555, 707},
    {3, 3, 543, 715},
    {4, 3, 544, 730},
    {5, 3, 557, 741},
    {6, 3, 575, 734},
    {11, 3, 579, 702},
    {12, 3, 592, 750},
    {13, 3, 716, 738},
    {14, 3, 773, 726},
    {15, 3, 839, 745},
    {1, 4, 776, 893},
    {2, 4, 748, 887},
    {3, 4, 741, 909},
    {4, 4, 756, 883},
    {5, 4, 773, 895},
    {6, 4, 798, 871},
    {11, 4, 765, 836},
    {12, 4, 802, 842},
    {13, 4, 813, 809},
    {14, 4, 804, 736},
    {15, 4, 846, 737},
    {1, 5, 927, 1081},
    {2, 5, 907, 1102},
    {3, 5, 905, 1116},
    {4, 5, 910, 1123},
    {5, 5, 924, 1157},
    {6, 5, 948, 1072},
    {11, 5, 961, 1057},
    {12, 5, 1005, 958},
    {13, 5, 978, 839},
    {14, 5, 958, 713},
    {15, 5, 1046, 695},
    {1, 6, 993, 1144},
    {2, 6, 978, 1108},
    {3, 6, 977, 1126},
    {4, 6, 988, 1135},
    {5, 6, 1008, 1113},
    {6, 6, 1040, 1052},
    {11, 6, 1062, 1094},
    {12, 6, 1104, 944},
    {13, 6, 1003, 826},
    {14, 6, 976, 704},
    {15, 6, 1031, 686},
    {1, 7, 1097, 1179},
    {2, 7, 1079, 1186},
    {3, 7, 1079, 1191},
    {4, 7, 1086, 1186},
    {5, 7, 1104, 1156},
    {6, 7, 1127, 1045},
    {11, 7, 1143, 1059},
    {12, 7, 1170, 927},
    {13, 7, 1064, 823},
    {14, 7, 1044, 707},
    {15, 7, 1125, 685},
    {1, 8, 1190, 1409},
    {2, 8, 1180, 1369},
    {3, 8, 1178, 1323},
    {4, 8, 1187, 1243},
    {5, 8, 1204, 1134},
    {6, 8, 1234, 1017},
    {11, 8, 1275, 1038},
    {12, 8, 1283, 911},
    {13, 8, 1098, 811},
    {14, 8, 969, 706},
    {15, 8, 996, 689},
    {1, 9, 1271, 1436},
    {2, 9, 1257, 1399},
    {3, 9, 1264, 1338},
    {4, 9, 1280, 1254},
    {5, 9, 1306, 1147},
    {6, 9, 1290, 1019},
    {11, 9, 1407, 1052},
    {12, 9, 1421, 927},
    {13, 9, 1029, 818},
    {14, 9, 925, 720},
    {15, 9, 930, 707},
    {1, 10, 1313, 1437},
    {2, 10, 1301, 1408},
    {3, 10, 1310, 1358},
    {4, 10, 1324, 1290},
    {5, 10, 1346, 1172},
    {6, 10, 1358, 1041},
    {11, 10, 1436, 1081},
    {12, 10, 1443, 946},
    {13, 10, 1127, 828},
    {14, 10, 1029, 744},
    {15, 10, 1007, 732},
    {1, 11, 1385, 1445},
    {2, 11, 1368, 1439},
    {3, 11, 1374, 1421},
    {4, 11, 1375, 1376},
    {5, 11, 1378, 1308},
    {6, 11, 1403, 1221},
    {11, 11, 1471, 1257},
    {12, 11, 1487, 1177},
    {13, 11, 1342, 982},
    {14, 11, 1269, 860},
    {15, 11, 1330, 854},
};

enum class MoverBoard { P100a, P150 };

// Table for a tt::tt_metal::ClusterType value (P100 = 6, P150 = 7, P150_X2 = 8,
// P150_X4 = 9, P150_X8 = 13): the p150 table for p150 cards, else the p100a
// table (the pre-#358 behaviour for every board).
inline MoverBoard mover_board_for_cluster(int cluster_type) {
    switch (cluster_type) {
        case 7: case 8: case 9: case 13: return MoverBoard::P150;
        default: return MoverBoard::P100a;
    }
}

// GSPLAT_TT_MOVER_TABLE: "p100a" or "p150" forces a table; unset, empty or
// "auto" gives `detected`. Anything else is an error (nullptr ok = false).
inline MoverBoard mover_board_from_env(const char* e, MoverBoard detected, bool* ok) {
    *ok = true;
    if (e == nullptr || *e == '\0' || std::strcmp(e, "auto") == 0) return detected;
    if (std::strcmp(e, "p100a") == 0) return MoverBoard::P100a;
    if (std::strcmp(e, "p150") == 0) return MoverBoard::P150;
    *ok = false;
    return detected;
}

inline const char* mover_board_name(MoverBoard b) {
    return b == MoverBoard::P150 ? "p150" : "p100a";
}

// The table of this process's device: defined in sort_device.cpp (needs the
// tt-metal cluster), read once, so the sort and the K2 fold use the same one.
MoverBoard mover_board();

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
// x | y << 16 from `board`'s table; cores missing from it count as 1000.
inline std::vector<uint32_t> mover_speeds(const std::vector<uint32_t>& noc_xy, MoverBoard board) {
    std::vector<uint32_t> speed(2u * noc_xy.size(), 1000u);
    const MoverSpeed* tab = board == MoverBoard::P150 ? kMoverSpeedP150 : kMoverSpeedP100a;
    const std::size_t n = board == MoverBoard::P150 ? std::size(kMoverSpeedP150)
                                                    : std::size(kMoverSpeedP100a);
    for (std::size_t c = 0; c < noc_xy.size(); c++) {
        for (std::size_t i = 0; i < n; i++) {
            const MoverSpeed& m = tab[i];
            if (m.x == (noc_xy[c] & 0xFFFFu) && m.y == (noc_xy[c] >> 16)) {
                speed[2u * c] = m.brisc;
                speed[2u * c + 1u] = m.ncrisc;
            }
        }
    }
    return speed;
}

}  // namespace gsplat_tt::sort_split
