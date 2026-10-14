# t772: pair-setup sub-step profile (blend backward, BWD_PRESTAGE=2)

Run 2026-10-14 12:44-12:47 PDT on p150 bh-49 (measurement reservation, `ttp lock p100`), commit 0cf2d921,
hero training step, busiest core 107 (41398 pairs, 11.89 views/pair). Driver `docs/t772/profile.sh`,
summary `python3 docs/t772/stats.py docs/t772/out-prof/remote/st*/stats.json`.
Outputs bit-identical in both modes (slots9 md5 055233b6471859499ccfb19dc70e9e9b).

| cycles/pair, core 107      | st1 (issue side) | st2 (Tensix idle before each lap) | st1 minus counter overhead (~12.5/lap) |
|----------------------------|-----:|-----:|-----:|
| L1 loads, 25 words         | 118.6 | 94.9 | ~106 |
| 9 pair scalars (27 ib stores) | 46.2 | 69.2 | ~34 |
| P0 Lx Ly (9 ib stores)     | 27.0 | 38.5 | ~15 |
| UNPACK mailbox             | 13.2 | 21.1 | ~1 |
| term-sum zeroing (9 SFPSTORE) | 20.3 | 35.5 | ~8 |
| rest                       | 9.3 | 9.1 | ~9 |
| **setup total**            | 234.7 | 268.3 | **~172** (172.3 without sub-counters, #767) |
| body                       | 2377.8 | 2391.8 | |
| finish                     | 114.7 | 185.3 | |

Hero with counters: 84.43 ms (st1), 88.13 ms (st2); 82.39 ms without (#767).

Findings
- The 25 RISC-V L1 loads are ~62% of setup (~4.2 cyc/load). With the Tensix idle first (st2) they
  take ~82 cycles (3.3/load): ~24 cycles of st1's load time is the previous pair's finish still
  executing (finish shows ~70 cycles of SFPU backlog: 114.7 issue vs 185.3 executed). That part is
  SFPU work the pair needs anyway, not a setup cost that can be removed.
- Body issue and execution match (2378 vs 2392 incl. sync cost): the MATH RISC-V is the limit
  through the whole pair; the SFPU waits on setup.
- Staging is ~1.3 cyc per instruction-buffer store (near the 1/store floor); mailbox ~1; zeroing ~8.
- Fewer loads does not help on this core: mode 0 (13 loads + building both SFPLOADI words with
  ALU ops) and mode 2 (25 loads, no ALU) measured the same (172.9 vs 172.3), so one L1 load costs
  about as much as the ~4 ALU ops it replaces; mode 1 (loads interleaved with ib stores) was slower.

Decision: no clear lossless change on the MATH RISC-V, so no A/B was run. The remaining lever is
taking the scalar staging off the MATH RISC-V (see the hand-off follow-up: unpack the pair scalars
to Dst with the UNPACK thread, double-buffered), worth up to ~150 of the 172 cyc/pair (~4.6 ms of
82.39 ms if fully hidden).
