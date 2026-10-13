# t613: one script for the per-landing viewer deploy + page-FPS check

`opt/viewer/deploy_and_measure.sh <N>` (usage in its header and in `opt/ird/README.md`).

Live run 2026-10-13 ~09:40 PDT, bh-30 (viewer reservation IRD 138278, extended to 06:32 UTC 10-14;
nothing else reserved or released). The viewer ran 2346ba31 (best-iter-227 plus the #597/#602 viewer
commits), a descendant of best-iter-227, so the script kept it (no redeploy) and only extended.

`opt/viewer/deploy_and_measure.sh 227` (`live.json`):
page_fps 110.89 (runs 105.38 / 110.89 / 121.32), device_ms 8.04, rc 0.
`--no-deploy` rerun (`live2.json`): page_fps 119.02 (runs 119.02 / 64.81 / 121.50), device_ms 8.05.

Device ms was steady at 8.0; the low runs had link 3.8 and 5.2 MB/s against ~12 MB/s in the
others, so they were limited by the VPN link, not the viewer. Clean runs give 121.3-121.5, the same as
#586/#597 (121). Bench b2b for best-iter-227: 8.454 ms/view (118.3 FPS, bh-33 p100).
