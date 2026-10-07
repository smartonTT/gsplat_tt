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
