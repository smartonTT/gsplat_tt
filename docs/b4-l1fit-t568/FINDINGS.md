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
