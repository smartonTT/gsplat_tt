# #767: T1 v2 blend-backward pair setup (BWD_PRESTAGE)

p150 bh-49, ABBA, 5 hero rounds x 8 reps per arm, own TT_METAL_CACHE per arm (trisc0/1/2 and
reader ELF md5s differ per arm, see out-ps*/remote/jit-elf-md5.txt). Base = #747 path (BWD_PRESTAGE=0).

| arm | hero warm median ms (5 rounds) | core 107 setup cyc/pair | body | fin | setup ms |
|---|---|---|---|---|---|
| base (#747) | 82.636-82.667, med 82.66 | 172.9 | 2383.0 | 113.8 | 5.30 |
| mode 1 (out-ps1) | med 83.33 | 198.2 | 2378.2 | 114.8 | - |
| mode 2 (out-ps2) | 82.388-82.420, med 82.39 | 172.3 | 2372.4 | 112.2 | 5.28 |

Mode 2: -0.27 ms (-0.3%), no overlap between arms. Pair setup itself barely moved (-0.6 cyc/pair);
most of the gain is body (-10.6) and fin (-1.6) cycles/pair.

Exactness: the 9 output floats per slot are bit-identical to base for sh0, sh3, s1, s2 and all 5
hero rounds, both modes. Device T0 oracle, both arms: sh0 3.79e-5, sh3 1.28e-4, s1 1.04e-3,
s2 1.64e-3, all pass. Hero: outputs bit-identical to #747, whose hero oracle passed (3.28e-3).

Default is now BWD_PRESTAGE=2. The +0.95 ms goal is not reached: setup stays ~172 cyc/pair.
