# t676: device runs without the devsync/mutagen mirror

`opt/sync_remote.sh` copies one commit of this worktree to a project-owned dir on the box with a
one-way rsync and an explicit include list (git-tracked files minus logs, traces, fixtures, docs
data; 1031 files, 63 MB instead of 644 MB). No .git: the remote gets `SHA` and `SYNC_STAMP`
(sha, source worktree, time, files, bytes). Scenes are copied only when the remote md5 differs.

    opt/sync_remote.sh [--dry-run|--list|--no-build] <host> [remote_dir] [rev]
    opt/sync_remote.sh --pull <remote_out> <local_out> <host> [remote_dir]

Default remote dir `/localdev/$USER/gstt2-sync`. The source must be under
/Users/smarton/dev/gsplat_tt or ~/dev/gstt2 (worktrees included); $HOME, ~/dev and / are refused.

Verification on bh-33 (IRD 138447), 2026-10-13 21:18-21:23 PDT, one `ttp lock p100`
(`docs/sync-t676/drive.sh best-iter-229`, log `verify-bh33.log`):
- dry run listed 1033 files (1031 + 2 stamps, `dryrun-bh33.log`); sync+build 2 min;
  stamp sha 3702fbb5 = best-iter-229 (MATCH)
- b2b 3 runs: 8.062 / 8.064 / 8.060 ms/view (best-iter-229: 8.063), raw_md5 1d372b94 all three
- resync: 1 file sent (the stamp); remote dir 128 MB, JIT cache 112 MB

Outside-repo dependencies taken from the box, never copied: tt-metal (/localdev/smarton/tt-metal,
GitHub clone at 437bc366, built there), the venv and scenes in /localdev/smarton/gstt2 (box-local
on bh-33, not a mirror). Viewer on bh-30: /localdev/smarton/viewer (own tt-metal and venv).
