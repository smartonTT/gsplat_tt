# t267 run-2 checklist (written by run 1)

Chain: `ttp detach t267-chain -- bash docs/profile-iter199-t267/drive.sh HEAD` (run dir 723).
Outputs: docs/profile-iter199-t267/out/ (run-rv1-{def,hp}.log, md5-*, t267-def-{zones,gaps,roofline,capture}.txt/log),
tmp/t267/t267-def-dev30.csv (not committed).

To do:
1. Tracy verify: t267-def-capture.log has `[capture_tracy] DONE rc=0` and no TT_FATAL -> t258 kcfg auto-size PASS.
2. `python3 opt/profiler/interview_gaps.py tmp/t267/t267-def-dev30.csv --skip-first` (device idle between views).
3. Host bridge from run-rv1-hp.log (HP*/STAGES lines), per #155 rule (>=0.3 ms before targeting a gap).
4. Compare with t258 capture (pre-t232, docs/tracy-kcfg-t258/out): pfwc 2.332, K2 0.985, sort_ol 1.438,
   mat+blend 6.910 window (mat_cull_mask TRISC max 3.00, blend sfpu 3.81/3.759 makespan, tile_blend_load NCRISC 3.73),
   span 11.665 traced; and #230: pfwc 2.33, K2 0.98, sort_ol 1.44, mat+blend 7.56, blend sfpu 4.42.
5. Bench = per-view render() latency, views not back to back (#171): cross-view pipelining gains 0 on this metric.
6. Write docs/profile-iter199-t267.md, commit, push, result.json.
