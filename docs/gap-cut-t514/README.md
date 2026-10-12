# t514: cut inter-program idle gaps in the b2b frame — shelved

Lever: `GSPLAT_TT_GAP_CUT=1` (default 0, commit 94cec795) builds the subchunk layout and the mat worklist on a
helper thread under LPT/publish, and drops the per-view `[SORT]`/`[SUBCHUNK]` logs, so the host enqueues the
next program sooner.

## Gap table before (t506, best-iter-223, p150 b2b Tracy)

| gap | ms/view |
|---|---|
| pfwc -> k2 | 0.008 |
| k2 -> sort | 0.014 |
| sort -> mat+blend | 0.045 |
| blend -> pfwc N+1 | 0.015 |
| total | 0.083 |

## A/B (bh-33, p100a, aiclk 1350, 2026-10-12 09:20 PDT, head c8b8514c)

`render/run.py --back-to-back`, 30 bicycle views, 20 passes per run, 5 alternating rounds (median of passes):

| round | off | gap-cut | delta |
|---|---|---|---|
| r1 | 9.368 | 9.376 | +0.008 |
| r2 | 9.369 | 9.355 | -0.014 |
| r3 | 9.371 | 9.355 | -0.016 |
| r4 | 9.367 | 9.356 | -0.011 |
| r5 | 9.382 | 9.372 | -0.010 |
| median | 9.369 | 9.356 | **-0.013** (paired median -0.011) |

Image: md5 906e0435 30/30 golden in both arms; the 30 per-view md5s are identical between arms.
Hero PSNR vs benchmarks/reference_v2/hero.png = 42.51 dB (both). out/hero-gc.png, out/diff-gc.png (|diff| x8):
no tile seams or artifacts seen.

## Verdict

Shelved: ~0.011-0.013 ms/view, below the 0.04 stack-candidate line. The helper-thread host work only removes the
host part of the sort -> mat+blend gap; the rest of the 0.083 ms bound is device-side dispatch latency
(program launch + runtime-arg writes), which a host-side change cannot reach. A remaining route is metal trace
capture/replay of the per-frame program sequence; it would combine with the CQ1 pfwc overlap task, which hides the
blend -> pfwc N+1 gap anyway. No per-gap "after" timeline was taken: the total gain is already bounded by the A/B.
