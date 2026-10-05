# Iteration checklist

Every row in `opt/ttw/iters.jsonl` (keep, reject, diagnostic, rebaseline) must
carry these before the iteration is done. `python3 opt/build_report.py` and
`python3 opt/validate_report.py` fail when the device screenshot is missing.

1. Measure on the p150/p100 device with the bicycle reference (ms/view, md5).
2. **Device screenshot (required, user 2026-10-05).** Render the bicycle hero
   view on the device at the iteration's commit and config (never the CPU
   reference) and save it as `opt/metal-screenshots/ttw-<NNN>/hero.png`.
   Save the diff vs `benchmarks/reference_v2/hero.png` next to it as
   `hero_diff10.png` (10x amplified; `build_report.py` writes it if missing).
3. **Visual check (required).** Open hero.png and the diff and look for tile
   artifacts: seams on 16x16/32x32 tile edges, blocky or empty tiles, stripes,
   color shifts. md5 and PSNR alone are not enough. Write what you saw.
4. Add the `device_screenshot` object to the row:

   ```json
   "device_screenshot": {
     "hero": "opt/metal-screenshots/ttw-198/hero.png",
     "diff": "opt/metal-screenshots/ttw-198/hero_diff10.png",
     "psnr_vs_ref": 100.0,
     "md5": "46a725ab",
     "commit": "abc1234",
     "config": "defaults (GSPLAT_TT_BLEND_SCHED=2 ...)",
     "device": "yyzo-bh-07 p100a",
     "visual_check": "no tile seams or blocky tiles; diff black except ..."
   }
   ```

   All fields except `device` are required. `psnr_vs_ref` is a number in dB or
   `"inf"`. Paths are repo-relative and must be PNG files that exist.
5. Capture Tracy for kept iterations (`tracy` field or `tracy_waiver`).
6. Rebuild and validate: `python3 opt/build_report.py && python3 opt/validate_report.py`.
   Commit `iters.jsonl`, the screenshots and both REPORT.html files.

Legacy iterations 160-197 that were never screenshotted carry
`"screenshot_backfill_pending": true`; the report shows them as a red
"screenshot missing - backfill pending" box. When a backfill adds the
`device_screenshot` object, remove the flag. Iterations after 197 cannot use
the flag. Tests: `python3 opt/test_screenshot_requirement.py`.
