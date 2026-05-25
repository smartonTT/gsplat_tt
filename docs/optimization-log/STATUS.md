# Supervisor Status

## ESCALATIONS (read these first)

### 2026-05-25 — bh-14 ssh auth blocked (Task 19)

iter-0 baseline render started successfully on yyzo-bh-14 (exit 0 from
`render_fixed.py --cycles --backend tt --scene stitch --size 1024`), then the
ssh connection to bh-14 broke — `Permission denied (publickey,password)` from
that point onward, despite:

- agent has the right key loaded (`ssh-add -L` matches what worked moments earlier)
- jump host `yyz-ird` still accepts the key (agent forwarding works)
- bh-14 host is reachable (TCP-level)
- key is unchanged on Mac

So the box's authorized_keys on yyzo-bh-14 stopped trusting this key
mid-session. Probable cause: box rebooted or had its runtime auth refreshed
(common pattern on shared lab boxes). The artifacts from the iter-0 render
are stuck at `/tmp/iter-000-baseline/{hero,side,top}.png` + `timing.jsonl` on
the box itself — they may still be there, but I can't `scp` them down.

**To unblock:** refresh ssh access to yyzo-bh-14 (re-run whatever IRD/lab
auth setup the team uses to provision keys), then resume by re-running
`scripts/run_iter.sh 0 baseline baseline` or manually `scp`ing the
`/tmp/iter-000-baseline/` artifacts down.

**Independent fix that already landed (does not require bh-14):**
`benchmarks/cameras.json` was missing from Task 2's infra carry-over and
`/benchmarks/` was a blanket gitignore. Commit `ef8fc55` restored both.

## Current State

- Loop: not yet started (blocked at Task 19 — iter-0 baseline)
- Branch: smarton/opt-v2
- iter-0 baseline: render attempted, artifacts unreachable until ssh restored
- Current best kernel ms: n/a
- Target: 1.0 ms
