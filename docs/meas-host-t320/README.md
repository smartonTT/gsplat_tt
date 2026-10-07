# t320: measurement box moved to yyzo-bh-04

- Reservation: IRD job 244892, cluster tt_yyz, ssh port 46483, start 2026-10-06 20:01:48,
  end 2026-10-07 10:01:48 (squeue on yyz-ird, EDT). yyzo-bh-07's reservation expired.
- ssh: `~/.ssh/config` (`Host yyzo-bh-*`, ProxyJump yyz-ird, Port 46483) and
  `~/.devsync/ssh-hosts` already cover yyzo-bh-04; nothing to change there.
- Box state before the sync: tt-metal 437bc366 built with Tracy at /localdev/smarton/tt-metal,
  venv /localdev/smarton/gstt2/.venv and scenes/bicycle.ply already made by
  opt/viewer/setup_box.sh (setup.rc 0). yyzo-bh-07 ran tt-metal e77780fe (local commit).
- Repo default changed: opt/ttw/screenshot.sh H defaults to yyzo-bh-04. Task drivers
  hardcode `H=` per task; new drivers must use yyzo-bh-04.
- drive.sh: one `ttp lock p100` around opt/sync_remote.sh to /localdev/smarton/gstt2 (the
  box's base dir) and remote_smoke.sh (board info, bicycle 30 views at defaults, md5 vs
  golden 906e0435). Results land in out/.
- Absolute ms/view on this box are not comparable with yyzo-bh-07: A/B both arms here.

## Smoke result (2026-10-06 20:28 EDT, drive2, under `ttp lock p100`)

- bicycle.ply upload resumed (rsync --append-verify), md5 3745b7a6... verified (PLY_MD5_OK).
- Sync + build at cab7f9d6 (contains smarton/tt-project-opt b9ea1cb6 and best-iter-206), so_md5 764f1de3.
- Board: p100a, serial 0000043100000000, fw 19.12.0.0 (out/tt-smi.json).
- bicycle 30 views 1024x1024 at defaults: avg 11.27 ms/view (p50 11.4, min 9.5, max 13.1),
  stages project 3.04, sort 0.89, blend 6.90, d2h 0.39. Single untraced run, not an A/B.
- SWEEP_MD5 906e0435 on 30/30 views, all identical to golden (out/md5.txt, out/run.log).
- Caveat: the ttw BUILD ID banner still shows the old stamp (#123, 3f4759e, iter-140); the
  sync line's sha/so_md5 are the real build. tt-metal here is 437bc366, not bh-07's e77780fe.
