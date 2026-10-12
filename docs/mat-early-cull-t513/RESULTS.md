# t519: MAT_EARLY_CULL A/B on bh-33 (P100), commit 6340cc45, 2026-10-12

Back-to-back, 30 bicycle views, 20 passes per run, median ms/view per run.
Three alternating rounds in one session (bh-33 carried CPU load ~13-15 from t510 throughout).

| round | A (unset) | B (GSPLAT_TT_MAT_EARLY_CULL=1) | B-A |
|---|---|---|---|
| r1 (A,B) | 9.365 | 9.141 | -0.224 |
| r2 (B,A) | 9.370 | 9.130 | -0.240 |
| r3 (A,B) | 9.364 | 9.142 | -0.222 |
| median | **9.365** | **9.141** | **-0.224** (-2.4%) |

Stage medians (passes 1+): blend 5.90-5.96 (A) vs 5.53-5.64 (B); sort 0.99-1.08 (A) vs 1.06-1.17 (B).
The materialize work moves off the blend critical path; sort grows slightly.

Image: raw_md5 5a438f5a in both arms and identical across passes; shot B sweep md5 906e0435 on 30/30 views,
hero PSNR 42.51 dB vs benchmarks/reference_v2/hero.png, golden match (max 0 LSB).
Screenshot: opt/metal-screenshots/t519-B/hero.png, diff opt/metal-screenshots/t519-B/hero_diff10.png;
checked by eye: no seams or tile artifacts, diff is the usual edge/spoke pattern.

Tracy MATCULL captures (A and B, views 0:10) did not run: the bh-33 venv
(/localdev/smarton/gstt2/.venv) has no `loguru`, which `python3 -m tracy` imports. The per-core
finish-spread analysis is therefore not done; see the follow-up.

Verdict: -0.224 ms/view >= 0.1 gate with an unchanged image: kept-iteration candidate.
Logs: out/run-r*-{A,B}.log, out/shotB.out, out/T-M{A,B}-0.log.
