#!/usr/bin/env bash
# Decision-logic tests for opt/ird/renew.sh with stubbed ssh/scp/ttp/ssh-preflight: no network,
# no IRD, no box. Fake `ird list`, `ird list-slurm-reservations` and `ird list-machines` text comes
# from per-case fixture files. Run: bash opt/ird/test_renew.sh   (exit 0 = all passed)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RENEW=${RENEW:-$HERE/renew.sh}
T=$(mktemp -d "${TMPDIR:-/tmp}/renew-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"

# Fake ssh. Jump host commands read/write $F/{list,slurm,machines}; box commands read $F/box_setup.
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do case $1 in -o|-p|-i|-l) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=$1; shift; cmd="$*"
echo "ssh $host $cmd" >> "$F/calls"
if [ "$host" != yyz-ird ]; then
  case $cmd in
    *"test -f"*) cat "$F/box_setup" 2>/dev/null ;;
    *tail*) echo "setup log tail" ;;
  esac
  exit 0
fi
case $cmd in
  "ird list")
    echo "SELECTION ID USER CLUSTER PARTITION MACHINE TIME LEFT CPU_IDS MEMORY BOARDS JOB ID SSH PORT"
    awk '{print $1, "smarton", $2, "sw_interactive", $3, $4, "-", "-", "-", $5, 46483, 51483, "x"}' "$F/list" ;;
  "ird list-slurm-reservations")
    for c in tt_yyz tt_aus tt_bgd; do
      echo "CLUSTER: $c"; echo "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)"
      awk -v c="$c" '$1 == c {print $2, "sw_intera", "interact", "smarton", $3, "0:01", 1, $4}' "$F/slurm"
    done ;;
  "ird list-machines --noheader") cat "$F/machines" ;;
  "ird change-timeout "*)
    read -r _ _ sel _ <<<"$cmd"
    [ "$(cat "$F/extend" 2>/dev/null)" = refuse ] && exit 0
    awk -v s="$sel" '$1 == s {$4 = "14:00:00"} {print}' "$F/list" > "$F/list.n" && mv "$F/list.n" "$F/list" ;;
  "ird reserve "*)
    m=${cmd##*--machine }; c=${cmd#*--cluster }; c=${c%% *}
    mode=$(cat "$F/reserve_$m" 2>/dev/null || echo crash)
    case $mode in
      ok:*) j=${mode#ok:}
        echo "9 $c $m 13:59:59 $j" >> "$F/list"; echo "$c $j R $m" >> "$F/slurm"
        echo "INFO - Allocating resources..."; echo "INFO - $j on $m: Reservation complete..." ;;
      queued:*) j=${mode#queued:}
        echo "$c $j PD (Resources)" >> "$F/slurm"; echo "INFO - Allocating resources..."; exec sleep 30 ;;
      crash)
        echo 'Traceback (most recent call last):'
        echo '  File "utils.py", line 244, in get_nodes_with_gres: pending resources queued'
        echo 'ValueError: not enough values to unpack'; exit 1 ;;
    esac ;;
  *"ird release "*) ;;
esac
EOF
# shellcheck disable=SC2016  # expands in the stub, not here
printf '#!/bin/sh\necho "scp $*" >> "$F/calls"\n' > "$T/bin/scp"
# Fake ttp: `ttp lock R -- cmd`: log; run ssh (release), skip sync_remote.sh (logged).
cat > "$T/bin/ttp" <<'EOF'
#!/usr/bin/env bash
echo "ttp $*" >> "$F/calls"
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done; shift
case $1 in opt/sync_remote.sh) exit 0 ;; esac
exec "$@"
EOF
printf '#!/bin/sh\necho SSH_PREFLIGHT=ok\n' > "$T/bin/preflight"
chmod +x "$T/bin/"*

MACHINES='bh-30 tt_aus sw_interactive blackhole mixed unharvested 4 p150(0|4) - -
bh-50 tt_bgd bgd_interactive blackhole mixed unharvested 8 p150(1|8) - 2
yyzo-bh-04 tt_yyz sw_interactive blackhole mixed unharvested 1 p100(0|1) - -'

FAILS=0; N=0
# new_case <name>: fresh fixture dir; measure on yyzo-bh-04 (job 244892), viewer on bh-30 (135630).
new_case() {
  CASE=$1; export F=$T/$1; mkdir -p "$F/state"; : > "$F/calls"
  printf '%s\n' "$MACHINES" > "$F/machines"
  printf '1 tt_aus bh-30 13:00:00 135630\n2 tt_yyz yyzo-bh-04 01:30:00 244892\n' > "$F/list"
  printf 'tt_aus 135630 R bh-30\ntt_yyz 244892 R yyzo-bh-04\n' > "$F/slurm"
  echo '{"host":"yyzo-bh-04"}' > "$F/state/measure.json"
  echo '{"host":"bh-30"}' > "$F/state/viewer.json"
  echo ready > "$F/box_setup"
}
run() {
  OUT=$(PATH="$T/bin:$PATH" IRD_STATE_DIR="$F/state" SSH_PREFLIGHT="$T/bin/preflight" IRD_RESERVE_WAIT=1 \
    IRD_SCENE="$HERE/README.md" bash "$RENEW" "$@" 2>"$F/stderr" | tail -1); RC=$?
  RC=$(jq -r .exit <<<"$OUT" 2>/dev/null || echo "$RC")
}
check() {  # check <description> <command...>
  N=$((N + 1))
  if "${@:2}"; then echo "ok   $CASE: $1"
  else echo "FAIL $CASE: $1"; echo "     json: $OUT"; sed 's/^/     calls: /' "$F/calls"; FAILS=$((FAILS + 1)); fi
}
j() { jq -r "$1 // empty" <<<"$OUT"; }
sj() { jq -r "$1 // empty" "$F/state/$2.json"; }
calls() { grep -c -- "$1" "$F/calls"; }

# 1. A p150 request that outlasts IRD_RESERVE_WAIT is remembered by job id; the p100 is extended.
new_case queued; echo queued:777 > "$F/reserve_bh-50"
run measure
check "exit 0, p100 extended" [ "$RC" = 0 ] && [ "$(j .action)" = extended ] && [ "$(j .host)" = yyzo-bh-04 ]
check "pending bh-50 job 777 on tt_bgd in state" [ "$(sj .pending_machine measure)/$(sj .pending_job measure)/$(sj .pending_cluster measure)" = bh-50/777/tt_bgd ]
check "one ird reserve" [ "$(calls 'ird reserve')" = 1 ]
# 1b. Next run while still queued: no second request, keep extending.
sed -i.b 's/^2 tt_yyz yyzo-bh-04 [^ ]*/2 tt_yyz yyzo-bh-04 13:30:00/' "$F/list"   # time passes
: > "$F/calls"; run measure
check "still queued: no ird reserve" [ "$(calls 'ird reserve')" = 0 ]
check "still queued: p100 extended, pending kept" [ "$RC" = 0 ] && [ "$(j .action)" = extended ] && [ "$(sj .pending_job measure)" = 777 ]

# 2. The queued request now runs while the p100 is still held: adopt it as the move target.
new_case adopted
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
echo '3 tt_bgd bh-50 13:50:00 777' >> "$F/list"; echo 'tt_bgd 777 R bh-50' >> "$F/slurm"
run measure
check "moved to bh-50, exit 0" [ "$RC" = 0 ] && [ "$(j .action)" = moved ] && [ "$(j .host)" = bh-50 ] && [ "$(j .job_id)" = 777 ]
check "no ird reserve" [ "$(calls 'ird reserve')" = 0 ]
check "old job extended, then synced, then released under ttp lock p100" \
  [ "$(calls 'ird change-timeout 2 ')" = 1 ] && [ "$(calls 'ttp lock p100 -- opt/sync_remote.sh bh-50')" = 1 ] \
  && [ "$(calls 'ird release 2')" = 1 ]
check "state: no pending, no release left" [ -z "$(sj .pending_machine measure)" ] && [ -z "$(sj .release_job measure)" ]
check "measure never touches bh-30 or the viewer lock" [ "$(calls 'bh-30')" = 0 ] && [ "$(calls 'lock viewer')" = 0 ]

# 2b. Adopted, but setup_box.sh failed on the new box: report (exit 6), keep and extend the old box.
new_case setup_failed
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
echo '3 tt_bgd bh-50 13:50:00 777' >> "$F/list"; echo 'tt_bgd 777 R bh-50' >> "$F/slurm"; echo 1 > "$F/box_setup"
run measure
check "exit 6 setup_failed with log tail" [ "$RC" = 6 ] && [ "$(j .status)" = setup_failed ] && [[ $(j .note) == *"setup log tail"* ]]
check "setup not restarted, nothing synced or released" [ "$(calls setsid)" = 0 ] && [ "$(calls sync_remote)" = 0 ] && [ "$(calls 'ird release')" = 0 ]
check "old job extended and kept in state" [ "$(calls 'ird change-timeout 2 ')" = 1 ] && [ "$(sj .release_job measure)" = 244892 ]
# 2c. The next run (host bh-50, release pending) still reports the failure and extends both jobs.
: > "$F/calls"; run measure
check "rerun: exit 6 again, both jobs extended" [ "$RC" = 6 ] && [ "$(calls 'ird change-timeout 2 ')" = 1 ] && [ "$(calls 'ird change-timeout 3 ')" = 1 ]

# 3. Stale pending (request gone) and the p100 job gone: forget it and re-reserve.
new_case stale
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
printf '1 tt_aus bh-30 13:00:00 135630\n' > "$F/list"; printf 'tt_aus 135630 R bh-30\n' > "$F/slurm"
echo ok:888 > "$F/reserve_bh-50"
run measure
check "reserved bh-50, exit 0" [ "$RC" = 0 ] && [ "$(j .action)" = reserved ] && [ "$(j .host)" = bh-50 ] && [ "$(j .job_id)" = 888 ]
check "pending cleared" [ -z "$(sj .pending_machine measure)" ] && [[ $(j .note) == *"cleared stale request"* ]]

# 4. ird crashes (utils.py:244 traceback) on the p150: not 'queued'; fall back to the p100.
new_case crash
printf '1 tt_aus bh-30 13:00:00 135630\n' > "$F/list"; printf 'tt_aus 135630 R bh-30\n' > "$F/slurm"
sed -i.b 's/p100(0|1)/p100(1|1)/; s/1 p100(1|1) - -/1 p100(1|1) - 0/' "$F/machines"
echo crash > "$F/reserve_bh-50"; echo ok:999 > "$F/reserve_yyzo-bh-04"
run measure
check "crash is no request; reserved yyzo-bh-04" [ "$RC" = 0 ] && [ "$(j .host)" = yyzo-bh-04 ] && [ "$(j .job_id)" = 999 ]
check "no pending in state" [ -z "$(sj .pending_machine measure)" ]
# 4b. Crash on the move attempt while the p100 is alive: stay, extend, nothing pending.
new_case crash_move; echo crash > "$F/reserve_bh-50"
run measure
check "stay on p100, extended, no pending" [ "$RC" = 0 ] && [ "$(j .action)" = extended ] && [ -z "$(sj .pending_machine measure)" ]
# 4c. Legacy state (no job id), nothing of ours queued: cleared, never exit 4 forever.
new_case legacy_stale
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50"}' > "$F/state/measure.json"; echo crash > "$F/reserve_bh-50"
run measure
check "legacy pending cleared, p100 extended" [ "$RC" = 0 ] && [ -z "$(sj .pending_machine measure)" ] && [ "$(j .action)" = extended ]

# 5. Current job gone while our request is queued: wait (exit 4), no second request anywhere.
new_case queued_job_gone
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
printf '1 tt_aus bh-30 13:00:00 135630\n' > "$F/list"; printf 'tt_aus 135630 R bh-30\ntt_bgd 777 PD (Resources)\n' > "$F/slurm"
sed -i.b 's/1 p100(0|1) - -/1 p100(1|1) - 0/' "$F/machines"; echo ok:999 > "$F/reserve_yyzo-bh-04"
run measure
check "exit 4 pending, no ird reserve" [ "$RC" = 4 ] && [ "$(j .status)" = pending ] && [ "$(calls 'ird reserve')" = 0 ]

# 6. Viewer: IRD refuses to extend bh-30 in its last hour: wait for expiry, do not leave bh-30.
new_case viewer_last_hour
printf '1 tt_aus bh-30 00:40:00 135630\n2 tt_yyz yyzo-bh-04 13:00:00 244892\n' > "$F/list"; echo refuse > "$F/extend"
echo ok:555 > "$F/reserve_bh-50"
run viewer --no-deploy
check "exit 4 wait_expiry on bh-30, no ird reserve" [ "$RC" = 4 ] && [ "$(j .status)" = wait_expiry ] && [ "$(j .host)" = bh-30 ] && [ "$(calls 'ird reserve')" = 0 ]

# 7. --dry-run: read-only (no state written, no reserve/extend/release/sync run).
new_case dry; echo queued:777 > "$F/reserve_bh-50"; cp "$F/state/measure.json" "$F/before.json"
run measure --dry-run
check "dry-run: state untouched, no mutating call" cmp -s "$F/state/measure.json" "$F/before.json"
check "dry-run: only listings over ssh" [ "$(grep -vE 'ird list|ssh bh-50 |ssh yyzo-bh-04 ' "$F/calls" | grep -c .)" = 0 ]
check "dry-run: plans the move to bh-50" [ "$(j .host)" = bh-50 ] && [ "$(j .dry_run)" = true ]

echo "$((N - FAILS))/$N passed"
[ "$FAILS" = 0 ]
