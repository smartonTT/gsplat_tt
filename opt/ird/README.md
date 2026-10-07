# opt/ird: scripted IRD renewals

`opt/ird/renew.sh <viewer|measure> [--dry-run]` renews one role's IRD reservation through the
jump host `yyz-ird` and prints one JSON line on stdout (log lines go to stderr). It replaces the
manual steps the renewal tasks (#263, #316, #318, #320, #321, #325) worked out again on every run.

## What it does

1. ssh-preflight of the jump host, then read-only `ird list` (running reservations only),
   `ird list-slurm-reservations` (squeue per cluster: our queued requests too) and
   `ird list-machines`.
2. Decide:
   - **viewer** (resource key `viewer`): keep bh-30. If its job is gone, reserve bh-30 if it is
     free, else any free p150, else a free p100. The measurement box is never picked.
   - **measure**: if a p150 other than bh-30 and the viewer box is free and the current box is a p100,
     reserve it and move there (`--stay` skips the move). Otherwise extend the current p100.
     bh-30 is never picked, nor any machine we already hold.
   - Extend = `ird change-timeout <sel> 14:00:00` (IRD caps silicon at 14 h). If IRD does not
     extend and more than 1 h is left, it exits 3. With 1 h or less left, or when the job is
     gone, it re-reserves (`ird reserve --cluster C --ssh-port 46483 --timeout 14:00:00
     --no-shell blackhole --machine M`, bounded by `IRD_RESERVE_WAIT`, default 240 s).
   - viewer: if IRD refuses to extend bh-30 in its last hour, the run exits 4 (`wait_expiry`)
     instead of moving: our own job still holds bh-30. Rerun after the expiry in the note to
     re-reserve bh-30.
   - Queued request: `ird reserve` blocks while salloc waits, and the request stays queued in
     slurm after the timeout ends the ssh. The run finds it as a new job of ours in
     `ird list-slurm-reservations` (salloc's `Pending job allocation N` line is only a fallback;
     an ird traceback is never taken as queued) and stores `pending_machine`, `pending_job` and
     `pending_cluster`. Every later run checks it first:
     - running (in `ird list`): adopt it. For measure with the p100 still held this is the move
       (`action: moved`, `release_job` = the p100 job), then setup, sync and release as below.
     - still queued: no new `ird reserve` for any machine while it waits; the current job is
       extended, or the run exits 4 when it is gone.
     - gone (crashed, cancelled or expired): forget it and continue (re-reserve if the job is gone).
3. Host key: strict `ssh -o StrictHostKeyChecking=yes` to the box. If the key changed or is
   unknown, the key the box offers (ssh-keyscan via the jump host) is compared with a trusted
   fingerprint: `--fingerprint SHA256:...`, or the box's own
   `/etc/ssh/ssh_host_ed25519_key.pub` read over IRD's channel (`ird connect-to <sel>`). Only on
   a match is the old line removed (`ssh-keygen -R "[host]:46483"`) and the new key added. Any
   mismatch or failure exits 10 with a `HIGH SEVERITY ALERT` note. Then it runs
   `tt-project/harness/bin/ssh-preflight <host>`.
4. Box setup: if the box lacks tt-metal, the venv or `scenes/bicycle.ply`, it uploads and
   starts `opt/viewer/setup_box.sh` detached (log and `setup.rc` in `/localdev/$USER/viewer/`)
   and exits 4. Rerun once `setup.rc` exists. A non-zero `setup.rc` is reported (exit 6,
   `setup_failed`, with the `setup.log` tail), not restarted; delete it after fixing the box.
5. Viewer: redeploys the newest `best-iter-<N>` tag with
   `ttp lock viewer -- env VIEWER_HOST=<host> opt/viewer/viewer.sh deploy <tag>` into
   `/localdev/$USER/viewer`. If the deployed build already contains the tag, it only starts the
   viewer when it is not running.
6. Measure after a move: `ttp lock p100 -- opt/sync_remote.sh <host> /localdev/$USER/gstt2 <tag>`,
   then releases the old job under `ttp lock p100`. The old box stays reserved until the new one
   has built, and every run until then extends it too (`ird change-timeout`).

`--dry-run` makes only read-only queries (ird list and list-machines, strict ssh and preflight to a
box we already hold, `viewer.sh status`). It prints every changing command as
`[dry-run] would run: ...` and writes no state. It never opens an ssh to a box we do not hold.

## Output

One JSON line, for example:

```
{"role":"viewer","host":"bh-30","job_id":"135630","port":46483,"expiry_utc":"2026-10-07T22:52Z",
 "box_type":"p150","box_changed":false,"prev_host":"bh-30","cluster":"tt_aus","action":"extend",
 "status":"ok","deploy":"...","note":"","dry_run":true,"exit":0}
```

Optional fields: `pending_machine`/`pending_job`/`pending_cluster` (a queued request), and
`release_job`/`release_host` (the old box a move still has to release). The last JSON line per
role is kept in `<git common dir>/ird-state/<role>.json` (for the task worktrees `~/dev/gsplat_tt/.git/ird-state/`,
shared by every worktree; override with `IRD_STATE_DIR`). It is the script's memory of the
current host, a pending request and an unfinished move. `--dry-run` never writes it.

Exit codes: 0 ok; 2 usage or role conflict; 3 extend refused while the job is alive;
4 pending (queued request, setup running, or viewer waiting for bh-30 to expire: rerun later);
5 no free box; 6 setup, deploy, sync or release failed; 10-13 ssh-preflight classes (10 host key changed and not verified,
11 auth failed, 12 unreachable, 13 other).

Other options: `--host H` overrides the role's current host, and `--no-deploy` skips step 5/6.
Environment overrides: `IRD_JUMP`, `IRD_USER`, `IRD_RESERVE_WAIT`, `IRD_STATE_DIR`, `IRD_SCENE`,
`SSH_PREFLIGHT`.

## Tests

`bash opt/ird/test_renew.sh` runs the decision logic against stubbed `ssh`, `scp`, `ttp` and
ssh-preflight with fake `ird list` / `list-slurm-reservations` / `list-machines` text: queued,
adopted, setup failed, stale pending, ird crash, legacy state, viewer last hour and --dry-run.
No network, no IRD.

## Restrictions (binding, from the charter)

- No host scanning or sweeping, no exploratory logins. NEVER disable SSH host-key checking (no
  `StrictHostKeyChecking=no`, no `UserKnownHostsFile=/dev/null`). Refreshing a key is allowed only
  after its fingerprint is verified (user, 2026-10-04).
- Use only charter resources: IRD p150/p100 via the jump host `yyz-ird`.
- Only the single designated measurement reserver (the renewal task) may run `measure` without
  `--dry-run`. Only the viewer task may run `viewer` without `--dry-run`. All other tasks use
  `--dry-run` or read-only queries only.
- The viewer task never touches the measurement reservation or the `p100` lock. Measurement tasks
  never use the viewer box (bh-30 is the user's always-on viewer machine).
- Device sync and build for measurement run under `ttp lock p100 -- ...`.
- Viewer deploys go into `/localdev/$USER/viewer` with its own venv, never `/localdev/$USER/gstt2`
  on bh-30 (a Mac sync overwrites it). Delete nothing in bh-30's `/home` (the quota is full).
- The user's Mac reaches the viewer on local port 8091 (forward 8091 -> viewer box:8080). Never
  suggest port 8081. If the viewer box changes, the JSON note says which forward to change.
- The `ird reserve` output contains a container password. The script filters it out; never log it.

Caveat: `~/.ssh/config` sets `StrictHostKeyChecking no` for `bh-* yyzo-bh-*`. The script,
`opt/sync_remote.sh` and `tt-project/harness/bin/ssh-preflight` override it with
`-o StrictHostKeyChecking=yes` on every ssh, so a changed or unknown key is never accepted.

## Spec template for light-tier renewal tasks

```
Run `ttp detach renew-<role> -- opt/ird/renew.sh <role>` in your worktree (the role's designated reserver only) and report its JSON line; exit 4 -> hand off waiting with retry_when `ttp detach --check <rc path>` then rerun, 10 -> blocked "HIGH SEVERITY ALERT: ssh host key changed", other non-zero -> failed with the note.
If box_changed is true, record the new host/job/expiry in memory (viewer: tell the user to change the Mac forward to 8091 -> <host>:8080; measure: A/B both arms on the new box).
```
