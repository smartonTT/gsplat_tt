# t629: sizing the four unsized ideas from t614 (draft, device run pending)

Base: best-iter-227 code, FOLD=1, bh-33 p100, b2b about 8.45 ms/view (t614).
Sources: docs/b2b-floor-t614/RESULT.md (slack probes), its HTL capture out-c/htl-r6-htl.txt.gz,
docs/sort-host-t483/README.md, docs/b4-overlap/RESULT.md, docs/pfwc-noc1-balance-t232,
docs/pfwc-mover-t598, docs/pfwc-bundle-t606.

Device order per view (FOLD=1): mat+blend(N) with K2(N+1) -> device idle at the view boundary
(host: blend done, tail, python G, head, early sort(N+1) rtargs + enqueue) -> sort onelaunch(N+1)
about 1.44 ms -> layout/pub_host H2D -> pfwc(N+2) about 1.45 ms -> mat+blend(N+1).

| idea | sized bound (b2b ms/view) | basis |
|---|---|---|
| (a) pfwc mover pattern (both NoCs, bigger bursts, fewer barriers) | <= ~0.09-0.13 ceiling, realistic 0.05-0.07; knee probe pending | both NoCs already used (t232), record bursts batched (t598), flush=0 <= 0.064-0.077, RECFAST 0.07; pfwc window 1.527 vs TRISC pure ~1.39 + tail 0.045 |
| (b) sort device time on the path | whole sort onelaunch (~1.44 ms) is 1:1 on the path; the 0.13-0.23 "queued" part is not a separate cost | ps150/ps400 hidden; in-order CQ0 |
| (c) fold gather into pfwc | ~0: already folded under FOLD=1 (pfwc_ran_fused, no gather program) | "project_gather_wait" 0.24 ms is the early sort(N+1) rtargs/enqueue + cq1 projM read |
| (d) overlap TA/sort host prep with the blend wait | ~0: already overlapped (ps/hs probes hidden; t483 found host sort ~0 b2b) | only the view boundary is on the path: <= 0.30 ms (g3/g7/g15 slope 1.00) |

Probes in this branch (default off):
- HTL zones `sort_early_rtargs`, `sort_early_enqueue` (sort_device.cpp) and `cq1_proj_m`
  (tile_assign_device.cpp): split the 0.24 ms at the view boundary.
- `GSPLAT_TT_PFWC_TSPIN_US=N`: MATH TRISC busy-waits N us per core per pfwc launch. The b2b
  slope vs N stays 0 until TRISC becomes the bottleneck; the knee is the most any mover-only
  lever can save.

Run: `ttp detach t629 -- docs/size-unsized-t629/drive.sh 51111ee6` (arms htl, base, ts50..ts400).
