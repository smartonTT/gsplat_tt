# t568: sort in blend at KCFG 48 KB: L1 fit and the view-2 crash

1. L1 fit (4f0c385d): with OL_SOLO the standalone sort makes only BRISC's private CBs, which
   frees about 0.5 MB L1. The KCFG +48 KB fused program now builds and loads, and the warmup and
   view 1 render.
2. Crash (out-triage/, 0e42737f): the fuse check hangs in the first view whose blend runs
   sort(N+1). tt-triage callstacks while it hung show BRISC wild PCs on many cores and 34 cores
   with every RISC at PC 0. This is memory or code corruption, not a deadlock.
   Ruled out:
   - BRISC stack overflow: the olk frame is 1040 B, against the 4528 B K2 path that works.
   - A stateful NoC command buffer clobbered: the sort uses no set_state/with_state.
   - Arena or alias size mismatches: host and kernel constants agree.
   Still open: the CB21 arena region coincides with the K2 scratch (kB4aCb 62). K2 finishes
   before the sort, so this is probably harmless.
3. Next experiment (GSPLAT_TT_SIB_DBG, remote_triage.sh arms):
   - 1 = barrier only, no sort and no PACK town.
   - 2 = full sort, with the result discarded so the standalone sort reruns.
   If 1 passes and 2 crashes, the cause is in the sort body or the town. If 1 crashes, the
   cause is the barrier/semaphores or the SIB program config itself.

## SIB_DBG result (out-sibdbg/, f6546891, bh-33, 2026-10-12 18:37-18:43 PDT)

- Arm 1 (barrier only): passed, all 30 views rendered (rc 0). The KCFG 48 KB program, the SIB
  arena, the barrier and the semaphores are not the cause.
- Arm 2 (full sort in blend, result discarded): hangs at view 2, right after
  `[B4A] fold view 2 ... ran=1` (rc 124, same as the plain fuse check).
- So the corruption comes from the sort body or the PACK town running inside blend. Next split:
  run the sort body without the PACK town (TRISC side) and vice versa, and check whether the sort
  writes past CB21's arena (e.g. bucket/output pointers sized for the standalone layout landing
  in K2 scratch kB4aCb 62 or in kernel text).
