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
