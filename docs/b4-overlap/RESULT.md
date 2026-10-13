# B4 cross-frame persistent overlap (task #591): result

**Shelved at sizing, no device run.** See `PLAN.md` in this folder.

- Base best-iter-227: 8.454 ms/view b2b (bh-33 p100, md5 golden 077caaba). No candidate was built,
  so there are no A/B medians, no md5 or PSNR and no screenshot, and no iteration was added to
  opt/ttw/iters.jsonl.
- What is left of B4 after B4a (landed), SIB (shelved) and L4 (shelved) is the program boundaries:
  0.383 ms/view ceiling (t224 end spreads + gaps).
- The only design that fits kcfg, L1 and the BRISC stack (pfwc on its own sub-device core group,
  pipelined with sort+mat+blend) models at **+0.07 ms/view at best** (`model.py`), under the
  0.1 gate. A barrierless per-core hand-off is <= 0.05 and needs a tt-metal dispatcher change;
  resident per-frame state gives 0 on the bench.
