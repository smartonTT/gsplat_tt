# t542 B4a step 0: shadow K2 on the fused mat+blend BRISC (bh-33, 2026-10-12)

> **Correction (t548, review #547):** these numbers measured a no-op. The fused BRISC K2 args
> were gated on ctx->fused_ready, which is cleared every view before blend, so the shadow K2 never
> ran. Re-measured with the fix and on-device proof in docs/b4a-t548/RESULT.md: off 8.989 vs on
> 8.992 ms/view, shadow verified on 110/110 cores. The verdict below is superseded there.


Commit ae794619 (base best-iter-225 b612d4ed + t540 77d8fc3c). Flag GSPLAT_TT_B4A_SHADOW=1 (default off).
Step 0 only measures the cost of hosting K2: BRISC reruns this view's K2 mover parts (diet path)
after its mat body, into scratch DRAM outputs, yielding to the blend writer once per pair page.
It does not yet take K2 off the critical path (no N+1 schedule, no TRISC2 part, no double buffers).

## A/B (render/run.py --back-to-back, 30 bicycle views, 20 passes each, median ms/view)

| round | off   | on    |
|-------|-------|-------|
| warm  | 8.990 | 8.985 |
| r1    | 8.989 | 8.987 |
| r2    | 8.989 | 8.984 |
| r3    | 8.989 | 8.989 |
| median of r1-r3 | 8.989 | 8.987 |

Blend stage (B2B_STAGES): off 5.736 / 5.828 / 5.693, on 5.697 / 5.834 / 5.713. Latency mode: 8.9 ms both arms.

raw md5 5a438f5a on every b2b run (both arms, identical across passes). On-arm hero: 30-view sweep
md5 906e0435, golden match (max LSB 0), PSNR 42.51 dB vs benchmarks/reference_v2/hero.png,
opt/metal-screenshots/ttw-t542/hero.png + hero_diff10.png; checked by eye, no tile seams.

## Verdict

Not kept (no gain claimed; step 0 cannot give one). Hosting K2 on BRISC inside mat+blend costs
nothing measurable (-0.002 ms/view, noise), so the fold has room. Caveat: there is no device-side
proof that the shadow K2 actually ran (it is skipped when tile_assign_b4a_args finds no fused K2
args; md5 cannot show it because the outputs go to scratch). Step 1 must first read back the
b4a scratch outputs and compare them to the real K2 outputs before trusting the zero cost.

Tries 1-3 failed before device numbers (build ambiguity, L1 CB overflow, CB page divisibility);
logs in out-try1..3.
