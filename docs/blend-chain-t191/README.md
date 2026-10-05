# Tail-chained blend walk: md5 + paired A/B on p100 (task #191)

**Result: md5-identical, but below the 0.3 ms/view gate. Knob 1 saves 0.163 ms/view of blend
(-0.25 ms/view end-to-end, of which ~0.09 is d2h noise). `GSPLAT_TT_BLEND_CHAIN_WALK` stays
default 0 and is not landed. Decision recorded in `docs/blend-dispatch-t189`.**

## Setup
- Build: t190 chain walk rebased on tip 859706a, commit 2c2976d, one build on yyzo-bh-07 p100a
  (`/localdev/smarton/gstt2-t191`), every device step under `ttp lock p100`.
- `drive.sh`: sync, then 3 rounds of 30 untraced bicycle views for knobs 0/1/2 on the same
  build, arm order rotated per round (r1: 0,1,2; r2: 1,2,0; r3: 2,0,1). `remote_time.sh` runs
  one arm and writes the md5 list. Logs and md5 lists: `out/`.

## md5
All 9 runs (3 rounds x 3 knobs) give the same 30-line md5 list, identical to
`md5-r82new.txt` (list md5 46a725ab). `MD5_R1_OK`; every run also reports
`ALL_VIEWS_IDENTICAL` and hero 100 dB vs golden.

## Paired A/B (ms/view, STAGES line, 30 views each)
| | r1 | r2 | r3 | mean |
|---|---:|---:|---:|---:|
| knob 1 - knob 0, blend | -0.165 | -0.167 | -0.156 | **-0.163** |
| knob 1 - knob 0, view_total | -0.250 | -0.341 | -0.164 | -0.252 |
| knob 2 - knob 0, blend | -0.102 | -0.104 | -0.087 | -0.098 |
| knob 2 - knob 0, view_total | -0.065 | -0.210 | -0.038 | -0.104 |

Means: knob 0 16.307 ms/view (blend 8.548), knob 1 16.056 (8.385), knob 2 16.203 (8.450).

Only blend's MATH code changes, so the blend delta is the clean signal (spread 0.011 ms).
The extra view_total gap comes from d2h: knob 0 rounds 1 and 2 had d2h 0.28/0.29 ms against
0.19-0.21 in the other runs, which is noise unrelated to the knob. Without d2h, knob 1 is
-0.176 / -0.240 / -0.161 (mean -0.19).

## Decision
Below the >= 0.3 ms/view gate on both measures, so the knob stays off (t189 rule). The t189
model put the lever at 0.14 / 0.31 / 0.52 (low / mid / high); the measurement sits at the low
end: removing two taken jumps and hiding the table `lw` per body call recovers ~3.7 cycles per
call, not the ~9 of the middle estimate. The quad / vertical-body idea is closed with it (same
per-call cost; see `docs/blend-dispatch-t189`).
