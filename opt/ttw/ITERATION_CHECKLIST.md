# Iteration checklist

Every row in `opt/ttw/iters.jsonl` (keep, reject, diagnostic, rebaseline) must
carry these before the iteration is done. `python3 opt/build_report.py` and
`python3 opt/validate_report.py` fail when the device screenshot is missing.

1. Measure on the p150/p100 device with the bicycle reference (ms/view, md5).
   **Headline = back-to-back ms/view (user 2026-10-10, task #465).** Run
   `render/run.py --back-to-back --b2b-passes 3` and record in `metrics`:
   `ms_view_b2b` (median of all measured passes), `ms_view_b2b_passes` (every
   pass value, at least 3), `ms_view_latency` (secondary: run.py without
   `--back-to-back` and without `--dump-views`), plus `board`, `commit`, `build`.
   The validator fails any iteration above 216 without them. Rows up to 216
   keep their latency number, shown as "latency (legacy)"; never invent b2b
   numbers for them. Template: `docs/b2b-rebaseline-t465/bench465.sh`.
   **Keep gate:** a lever is kept if it saves at least 0.1 ms/view b2b, taken
   as the median of at least 3 alternating A/B passes in one session on one
   board, with the same sweep md5 (or hero PSNR at least 42.4 dB vs
   `benchmarks/reference_v2/hero.png`) and no tile seams. Latency with
   `--dump-views` hides about 1.7 ms of pfwc device time (#464) and never
   keeps or shelves a lever.
2. **Device screenshot (required, user 2026-10-05).** Render the bicycle hero
   view on the device at the iteration's commit and config (never the CPU
   reference) and save it as `opt/metal-screenshots/ttw-<NNN>/hero.png`.
   Save the diff vs `benchmarks/reference_v2/hero.png` next to it as
   `hero_diff10.png` (10x amplified; `opt/ttw/screenshot.sh` does both).
   The PSNR uses the SAME reference as the diff. The golden fixture
   (`tests/fixtures/hero/hero_golden_8bit.png`) is only a match badge, never the PSNR.
3. **Visual check (required).** Open hero.png and the diff and look for tile
   artifacts: seams on 16x16/32x32 tile edges, blocky or empty tiles, stripes,
   color shifts. md5 and PSNR alone are not enough. Write what you saw.
4. Add the `device_screenshot` object to the row:

   ```json
   "device_screenshot": {
     "hero": "opt/metal-screenshots/ttw-198/hero.png",
     "diff": "opt/metal-screenshots/ttw-198/hero_diff10.png",
     "ref": "benchmarks/reference_v2/hero.png",
     "psnr_vs_ref": 41.16,
     "diff_ref": "benchmarks/reference_v2/hero.png",
     "golden": "tests/fixtures/hero/hero_golden_8bit.png",
     "golden_match": true,
     "md5": "46a725ab",
     "commit": "abc1234",
     "config": "defaults (GSPLAT_TT_BLEND_SCHED=2 ...)",
     "device": "yyzo-bh-07 p100a",
     "visual_check": "no tile seams or blocky tiles; diff black except ..."
   }
   ```

   All fields except `device`, `golden` and `golden_match` are required.
   `psnr_vs_ref` is PSNR(hero, `ref`) in dB (or `"inf"`); `diff_ref` must be the
   same file as `ref`. The validator recomputes the PSNR, the diff and the golden
   match from the images and fails on any mismatch. Paths are repo-relative PNGs
   that exist. The report card names the reference; clicking the hero or the diff
   opens the compare viewer (ref / candidate / diff / split slider).
5. Capture Tracy for kept iterations (`tracy` field or `tracy_waiver`).
6. Rebuild and validate: `python3 opt/build_report.py && python3 opt/validate_report.py`.
   Commit `iters.jsonl`, the screenshots and both REPORT.html files.

Iterations up to 197 without a device screenshot predate the rule and are shown
as they are, with no placeholder (the backfill was cancelled by the user,
2026-10-05). Iterations after 197 must have one.
Tests: `python3 opt/test_screenshot_requirement.py`.
