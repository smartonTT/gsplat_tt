# t571: sort in blend crash, split sort body vs PACK town

## Arms 3 and 4 (out-sibdbg34/, 442cbb57, bh-33, 2026-10-12 18:54-19:03 PDT)

- Arm 3 (sort body, PACK town off): hangs at view 2 (rc 124).
- Arm 4 (PACK town handshake only, NB = 0, no sort body): hangs at view 2 (rc 124).
- Arm 1 (barrier only, t568) passes. What arms 2, 3 and 4 share and arm 1 lacks: BRISC writes the
  town mailbox header (arm 3 too: H_NB, H_READY, H_CAP, ... are written even with town = false).
- Static placement (host sib_arena, defaults OL_PB 8, WIN 1024, R 4): the mailbox (id 30, 13632 B
  host-sized) sits in region CB5 (NCRISC mat CB_BSORT) at byte 87936 of the CB; every buffer is
  inside its region (WIN in CB6, RING in CB20, the rest in CB5 / CB6). No out-of-range offsets.

## Next run (out-sibdbg54/, 67678223, TRIAGE=1)

- Arm 5: arm 4's BRISC mailbox writes (NB, READY, FIN, GO2, GO, then cleared) with the PACK town
  off (no handshake, no FIN wait). Passes: the writes are harmless and arm 4 is a handshake
  deadlock. Hangs: the mailbox address itself is bad.
- Arm 4 again with tt-triage callstacks: BRISC in the FIN loop with PACK in town_drain = PACK never
  sees GO (address or visibility mismatch); wild PCs = corruption.

## Arms 5 and 4 (out-sibdbg54/, 67678223, bh-33, 2026-10-12 19:09-19:17 PDT)

- Arm 5 (mailbox header writes only, PACK town off, no handshake): hangs at view 2, tt-triage shows
  wild BRISC PCs (0x13ab94, 0x1de4ec, "not in range of any ELF") on several cores. So the crash
  needs no sort body and no PACK town: ~20 BRISC word writes at the mailbox address are enough.
- Arm 4 hangs the same way.
- Host check: every arena region constant matches the CB sizes (CB5 = (2*16384+256)*4 = 132096 =
  8192 cut + 123904; CB6, CB20, CB21, CB22 too), and the host's mailbox size (13632, NT = 3) is
  larger than the kernel's (NT = 1). So either the computed address is not what the host placed,
  or CB5's tail is live during blend(N).

## Next run (out-sibdbg671/, 4d581ac6)

- Arm 6: the same writes at CB21's base (BRISC's dead mat CB_BSORT).
- Arm 7: the same writes at CB5's base + 8192 (start of the CB5 region past the alias cut).
- Arm 1 again at this head (control: barrier only should still pass).
- Reading: 6 passes and 7 hangs = CB5's tail is live during blend (move the mailbox and CB5
  buffers out of CB5). Both hang = any BRISC L1 write after the barrier breaks it (look at the
  address computation / CB interface on BRISC). 1 hangs = the control moved; recheck t571 changes.

## Result: arms 6, 7, 1 (out-sibdbg671, bh-33, 2026-10-12 19:26-19:32 PDT, commit 4d581ac6)

| arm | what BRISC does in blend(N) | result |
|---|---|---|
| 1 | barrier only (control) | 30/30 views ran, rc=0 |
| 5 | ~20 mailbox header writes at CB5 +87936, PACK town off | hang/crash at view 2 |
| 6 | same writes at CB21 base | hang at view 2 (rc=124) |
| 7 | same writes at CB5 base + 8192 | hang at view 2 (rc=124) |

All hangs happen right after `[B4A] fold view 2: prepared=1 cores_with_k2_args=110/110 ran=1`,
i.e. the first view where blend(N) carries the folded K2(N+1) and the sort-in-blend arm is live.

Conclusion: the write address does not matter. Three different target regions (CB5 tail,
CB5 interior, CB21 base) all hang, while the barrier-only arm passes. So the cause is not an
out-of-arena sort write. It is something every arm >=2 shares and arm 1 lacks: the extra BRISC
code itself (kernel binary size / text placement overlapping a CB or the K2 scratch kB4aCb 62),
the KCFG/runtime-arg reads that feed the arm, or the arm's handshake with the K2 fold path on
view 2. Next checks, in order: (a) compare BRISC kernel text size and end address for arm 1 vs
arm 5 against the lowest CB base on that core; (b) arm 8 = arm 5 with the writes compiled in
but skipped at runtime (`if (rt_arg == magic)`), which separates code size from the stores;
(c) arm 5 with GSPLAT_TT_B4A_FOLD=0, which separates the fold from sort-in-blend.
