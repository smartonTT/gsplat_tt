#!/usr/bin/env bash
# Decision-logic tests for opt/ird/renew.sh with stubbed ssh/scp/ttp/ssh-preflight: no network,
# no IRD, no box. Fake `ird list`, `ird list-slurm-reservations` and `ird list-machines` text comes
# from per-case fixture files. Run: bash opt/ird/test_renew.sh   (exit 0 = all passed)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RENEW=${RENEW:-$HERE/renew.sh}
export T; T=$(mktemp -d "${TMPDIR:-/tmp}/renew-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"

# Fake ssh. Jump host commands read/write $F/{list,slurm,machines}; box commands read $F/box_setup.
# Every call's -o options, -tt and whether it runs under `timeout` go to $T/opts (all cases).
# With $F/box_key set, the box presents that key: a strict `true` fails unless known_hosts has it.
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
opts= tt=0
while [ $# -gt 0 ]; do case $1 in -o) opts="$opts $2"; shift 2 ;; -p|-i|-l) shift 2 ;; -tt) tt=1; shift ;; -*) shift ;; *) break ;; esac; done
host=$1; shift; cmd="$*"
echo "ssh $host $cmd" >> "$F/calls"
echo "ssh $host T=${UNDER_TIMEOUT:-0} tt=$tt$opts" >> "$T/opts"
if [ "$host" != yyz-ird ]; then
  if [ "$cmd" = true ] && [ -s "$F/box_key" ] && ! grep -qF "$(awk '{print $2}' "$F/box_key")" "$IRD_KNOWN_HOSTS" 2>/dev/null; then
    echo '@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @' >&2
    echo 'The fingerprint for the ED25519 key sent by the remote host is' >&2
    echo "$(ssh-keygen -lf "$F/box_key" | awk '{print $2}')." >&2
    echo 'Host key verification failed.' >&2; exit 255
  fi
  case $cmd in
    *"test -f"*) cat "$F/box_setup" 2>/dev/null ;;
    *tail*) echo "setup log tail" ;;
  esac
  exit 0
fi
# With $F/jump_key_bad, the jump host presents a changed key (a stale mux let ssh-preflight pass).
if [ -e "$F/jump_key_bad" ]; then
  echo '@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @' >&2
  echo 'Host key for yyz-ird has changed and you have requested strict checking.' >&2
  echo 'Host key verification failed.' >&2; exit 255
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
  "ssh-keyscan "*) printf '[bh-30]:46483 %s\n' "$(awk '{print $1, $2}' "$F/box_key")" ;;
  "ird connect-to "*)
    # Like ird: no terminal, no shell in the container.
    [ "$tt" = 1 ] || { echo 'There was a problem connecting to/logging out of your container'; exit 1; }
    cat > /dev/null; printf '$ cat /etc/ssh/ssh_host_ed25519_key.pub; exit\r\n%s\r\nlogout\r\n' "$(cat "$F/ird_key")" ;;
esac
EOF
# Fake timeout: marks its child as bounded, then runs the real one.
REAL_TIMEOUT=$(command -v timeout || command -v gtimeout)
printf '#!/bin/sh\nUNDER_TIMEOUT=1 exec %s "$@"\n' "$REAL_TIMEOUT" > "$T/bin/timeout"
# shellcheck disable=SC2016  # expands in the stub, not here
printf '#!/bin/sh\necho "scp $*" >> "$F/calls"; echo "scp T=${UNDER_TIMEOUT:-0} $*" >> "$T/opts"\n' > "$T/bin/scp"
# Fake ttp: `ttp lock R -- cmd`: log; run ssh (release), skip sync_remote.sh (logged).
cat > "$T/bin/ttp" <<'EOF'
#!/usr/bin/env bash
echo "ttp $*" >> "$F/calls"
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done; shift
case $1 in opt/sync_remote.sh) exit 0 ;; esac
case " $* " in *" opt/viewer/viewer.sh "*) exit 0 ;; esac   # viewer start/deploy: logged only
exec "$@"
EOF
# Fake git fetch (no network); every other git command runs the real git.
printf '#!/bin/sh\n[ "$1" = fetch ] && exit 0\nexec %q "$@"\n' "$(command -v git)" > "$T/bin/git"
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
  : > "$F/known_hosts"
}
run() {
  # HOME and IRD_KNOWN_HOSTS point into $T: no test ever touches the real ~/.ssh.
  OUT=$(PATH="$T/bin:$PATH" HOME="$T/home" IRD_KNOWN_HOSTS="$F/known_hosts" IRD_PTY_DELAY=0 \
    IRD_STATE_DIR="$F/state" SSH_PREFLIGHT="$T/bin/preflight" IRD_RESERVE_WAIT=1 \
    IRD_SCENE="$HERE/README.md" bash "$RENEW" "$@" 2>"$F/stderr" | tail -1); RC=$?
  RC=$(jq -r .exit <<<"$OUT" 2>/dev/null || echo "$RC")
}
# A compound condition goes in a function (c) so that check sees all of it, not only its first [ ].
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
c() { [ "$RC" = 0 ] && [ "$(j .action)" = extended ] && [ "$(j .host)" = yyzo-bh-04 ]; }; check "exit 0, p100 extended" c
check "pending bh-50 job 777 on tt_bgd in state" [ "$(sj .pending_machine measure)/$(sj .pending_job measure)/$(sj .pending_cluster measure)" = bh-50/777/tt_bgd ]
check "one ird reserve" [ "$(calls 'ird reserve')" = 1 ]
# 1b. Next run while still queued: no second request, keep extending.
sed -i.b 's/^2 tt_yyz yyzo-bh-04 [^ ]*/2 tt_yyz yyzo-bh-04 13:30:00/' "$F/list"   # time passes
: > "$F/calls"; run measure
check "still queued: no ird reserve" [ "$(calls 'ird reserve')" = 0 ]
c() { [ "$RC" = 0 ] && [ "$(j .action)" = extended ] && [ "$(sj .pending_job measure)" = 777 ]; }; check "still queued: p100 extended, pending kept" c

# 2. The queued request now runs while the p100 is still held: adopt it as the move target.
new_case adopted
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
echo '3 tt_bgd bh-50 13:50:00 777' >> "$F/list"; echo 'tt_bgd 777 R bh-50' >> "$F/slurm"
run measure
c() { [ "$RC" = 0 ] && [ "$(j .action)" = moved ] && [ "$(j .host)" = bh-50 ] && [ "$(j .job_id)" = 777 ]; }; check "moved to bh-50, exit 0" c
check "no ird reserve" [ "$(calls 'ird reserve')" = 0 ]
c() { \
  [ "$(calls 'ird change-timeout 2 ')" = 1 ] && [ "$(calls 'ttp lock p100 -- opt/sync_remote.sh bh-50')" = 1 ] \
  && [ "$(calls '^ssh yyz-ird echo y | ird release 2')" = 1 ] && [ "$(calls '^ttp lock p100 -- .*ird release 2')" = 1 ]; }; check "old job extended, then synced, then released under ttp lock p100" c
c() { [ -z "$(sj .pending_machine measure)" ] && [ -z "$(sj .release_job measure)" ]; }; check "state: no pending, no release left" c
c() { [ "$(calls 'bh-30')" = 0 ] && [ "$(calls 'lock viewer')" = 0 ]; }; check "measure never touches bh-30 or the viewer lock" c

# 2b. Adopted, but setup_box.sh failed on the new box: report (exit 6), keep and extend the old box.
new_case setup_failed
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
echo '3 tt_bgd bh-50 13:50:00 777' >> "$F/list"; echo 'tt_bgd 777 R bh-50' >> "$F/slurm"; echo 1 > "$F/box_setup"
run measure
c() { [ "$RC" = 6 ] && [ "$(j .status)" = setup_failed ] && [[ $(j .note) == *"setup log tail"* ]]; }; check "exit 6 setup_failed with log tail" c
c() { [ "$(calls setsid)" = 0 ] && [ "$(calls sync_remote)" = 0 ] && [ "$(calls 'ird release')" = 0 ]; }; check "setup not restarted, nothing synced or released" c
c() { [ "$(calls 'ird change-timeout 2 ')" = 1 ] && [ "$(sj .release_job measure)" = 244892 ]; }; check "old job extended and kept in state" c
# 2c. The next run (host bh-50, release pending) still reports the failure and extends both jobs.
: > "$F/calls"; run measure
c() { [ "$RC" = 6 ] && [ "$(calls 'ird change-timeout 2 ')" = 1 ] && [ "$(calls 'ird change-timeout 3 ')" = 1 ]; }; check "rerun: exit 6 again, both jobs extended" c

# 3. Stale pending (request gone) and the p100 job gone: forget it and re-reserve.
new_case stale
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
printf '1 tt_aus bh-30 13:00:00 135630\n' > "$F/list"; printf 'tt_aus 135630 R bh-30\n' > "$F/slurm"
echo ok:888 > "$F/reserve_bh-50"
run measure
c() { [ "$RC" = 0 ] && [ "$(j .action)" = reserved ] && [ "$(j .host)" = bh-50 ] && [ "$(j .job_id)" = 888 ]; }; check "reserved bh-50, exit 0" c
c() { [ -z "$(sj .pending_machine measure)" ] && [[ $(j .note) == *"cleared stale request"* ]]; }; check "pending cleared" c

# 4. ird crashes (utils.py:244 traceback) on the p150: not 'queued'; fall back to the p100.
new_case crash
printf '1 tt_aus bh-30 13:00:00 135630\n' > "$F/list"; printf 'tt_aus 135630 R bh-30\n' > "$F/slurm"
sed -i.b 's/p100(0|1)/p100(1|1)/; s/1 p100(1|1) - -/1 p100(1|1) - 0/' "$F/machines"
echo crash > "$F/reserve_bh-50"; echo ok:999 > "$F/reserve_yyzo-bh-04"
run measure
c() { [ "$RC" = 0 ] && [ "$(j .host)" = yyzo-bh-04 ] && [ "$(j .job_id)" = 999 ]; }; check "crash is no request; reserved yyzo-bh-04" c
check "no pending in state" [ -z "$(sj .pending_machine measure)" ]
# 4b. Crash on the move attempt while the p100 is alive: stay, extend, nothing pending.
new_case crash_move; echo crash > "$F/reserve_bh-50"
run measure
c() { [ "$RC" = 0 ] && [ "$(j .action)" = extended ] && [ -z "$(sj .pending_machine measure)" ]; }; check "stay on p100, extended, no pending" c
# 4c. Legacy state (no job id), nothing of ours queued: cleared, never exit 4 forever.
new_case legacy_stale
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50"}' > "$F/state/measure.json"; echo crash > "$F/reserve_bh-50"
run measure
c() { [ "$RC" = 0 ] && [ -z "$(sj .pending_machine measure)" ] && [ "$(j .action)" = extended ]; }; check "legacy pending cleared, p100 extended" c

# 5. Current job gone while our request is queued: wait (exit 4), no second request anywhere.
new_case queued_job_gone
echo '{"host":"yyzo-bh-04","pending_machine":"bh-50","pending_job":"777","pending_cluster":"tt_bgd"}' > "$F/state/measure.json"
printf '1 tt_aus bh-30 13:00:00 135630\n' > "$F/list"; printf 'tt_aus 135630 R bh-30\ntt_bgd 777 PD (Resources)\n' > "$F/slurm"
sed -i.b 's/1 p100(0|1) - -/1 p100(1|1) - 0/' "$F/machines"; echo ok:999 > "$F/reserve_yyzo-bh-04"
run measure
c() { [ "$RC" = 4 ] && [ "$(j .status)" = pending ] && [ "$(calls 'ird reserve')" = 0 ]; }; check "exit 4 pending, no ird reserve" c

# 6. Viewer: IRD refuses to extend bh-30 in its last hour: wait for expiry, do not leave bh-30.
new_case viewer_last_hour
printf '1 tt_aus bh-30 00:40:00 135630\n2 tt_yyz yyzo-bh-04 13:00:00 244892\n' > "$F/list"; echo refuse > "$F/extend"
echo ok:555 > "$F/reserve_bh-50"
run viewer --no-deploy
c() { [ "$RC" = 4 ] && [ "$(j .status)" = wait_expiry ] && [ "$(j .host)" = bh-30 ] && [ "$(calls 'ird reserve')" = 0 ]; }; check "exit 4 wait_expiry on bh-30, no ird reserve" c

# 7. --dry-run: read-only (no state written, no reserve/extend/release/sync run).
new_case dry; echo queued:777 > "$F/reserve_bh-50"; cp "$F/state/measure.json" "$F/before.json"
run measure --dry-run
check "dry-run: state untouched, no mutating call" cmp -s "$F/state/measure.json" "$F/before.json"
check "dry-run: only listings over ssh" [ "$(grep -vE 'ird list|ssh bh-50 |ssh yyzo-bh-04 ' "$F/calls" | grep -c .)" = 0 ]
c() { [ "$(j .host)" = bh-50 ] && [ "$(j .dry_run)" = true ]; }; check "dry-run: plans the move to bh-50" c

# 8. bh-30 presents a new host key that the key read through ird does not confirm (#496): exit 10
# with both fingerprints, after the extend and before any deploy; known_hosts untouched.
mkdir -p "$T/home"
ssh-keygen -q -t ed25519 -N '' -C old -f "$T/old"; ssh-keygen -q -t ed25519 -N '' -C new -f "$T/new"
fp() { ssh-keygen -lf "$1" | awk '{print $2}'; }
key_case() {  # key_case <name> <key ird reports>: known_hosts has the old key, the box presents the new
  new_case "$1"; printf '[bh-30]:46483 %s\n' "$(awk '{print $1, $2}' "$T/old.pub")" > "$F/known_hosts"
  cp "$F/known_hosts" "$F/kh.before"; cp "$T/new.pub" "$F/box_key"; cp "$2" "$F/ird_key"
}
key_case key_changed "$T/old.pub"
run viewer
c() { [ "$RC" = 10 ] && [ "$(j .status)" = host_key_changed ] \
  && [ "$(j .presented)" = "$(fp "$T/new.pub")" ] && [ "$(j .stored)" = "$(fp "$T/old.pub")" ] && [ "$(j .ird_fp)" = "$(fp "$T/old.pub")" ]; }; check "exit 10 host_key_changed with presented/stored fingerprints" c
c() { [ "$(calls 'ird change-timeout 1 ')" = 1 ] && [ "$(j .action)" = extended ]; }; check "extend ran before the stop" c
c() { \
  [ "$(calls 'lock viewer')" = 0 ] && [ "$(grep '^ssh bh-30 ' "$F/calls" | grep -vc '^ssh bh-30 true$')" = 0 ]; }; check "nothing deployed: no viewer lock, no box command but the key check" c
c() { [ "$(grep -c '^ssh yyz-ird T=1 tt=1 ' "$T/opts")" -ge 1 ] && [ "$(calls 'ird connect-to 1')" = 1 ]; }; check "key read through ird connect-to on a pty" c
check "known_hosts untouched" cmp -s "$F/known_hosts" "$F/kh.before"
# 8b. The same in --dry-run: exit 10, nothing changed.
key_case key_changed_dry "$T/old.pub"; run viewer --dry-run
c() { [ "$RC" = 10 ] && [ "$(j .status)" = host_key_changed ] \
  && [ "$(j .presented)" = "$(fp "$T/new.pub")" ] && cmp -s "$F/known_hosts" "$F/kh.before"; }; check "dry-run: exit 10 host_key_changed, known_hosts untouched" c
# 9. ird confirms the new key: refresh known_hosts (stored key replaced) and go on.
key_case key_verified "$T/new.pub"; run viewer --no-deploy
c() { [ "$RC" = 0 ] && [ "$(j .status)" = ok ] \
  && grep -qF "$(awk '{print $2}' "$T/new.pub")" "$F/known_hosts" && ! grep -qF "$(awk '{print $2}' "$T/old.pub")" "$F/known_hosts"; }; check "verified: exit 0, known_hosts holds only the new key" c
c() { [ "$(j .stored)" = "$(fp "$T/old.pub")" ] && [[ $(j .note) == *refreshed* ]]; }; check "verified: fingerprints and note in the JSON" c
# 9b. --dry-run with a verified key: plans the refresh, changes nothing, plans nothing on the box.
key_case key_verified_dry "$T/new.pub"; run viewer --dry-run
c() { [ "$RC" = 0 ] \
  && [ "$(j .status)" = host_key_verified ] && cmp -s "$F/known_hosts" "$F/kh.before" && [ "$(calls 'ssh bh-30 test')" = 0 ]; }; check "dry-run verified: exit 0 host_key_verified, known_hosts untouched, no box commands" c

# 9c. The jump host's key changed but ssh-preflight passed (it rode a live ControlMaster mux, #499):
# the first strict jump call fails on the key. Exit 10 HIGH SEVERITY naming yyz-ird, not 13.
for role in measure viewer; do
  new_case "jump_key_changed_$role"; : > "$F/jump_key_bad"; run "$role"
  c() { [ "$RC" = 10 ] && [ "$(j .status)" = jump_host_key_changed ] \
    && [[ $(j .note) == "HIGH SEVERITY ALERT: ssh host key changed for yyz-ird"* ]]; }; check "exit 10 jump_host_key_changed, HIGH SEVERITY note names yyz-ird" c
  c() { [ "$(grep -c '^ssh yyz-ird ' "$F/calls")" = 1 ] && [ "$(grep -vc '^ssh yyz-ird ird list$' "$F/calls")" = 0 ]; }; check "stops at the first jump call: nothing extended, reserved or deployed" c
done

# 9d. ssh-preflight itself opens its own strictly checked connection, never a ControlMaster mux.
CASE=ssh_preflight; OUT=; PF=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight
if [ -x "$PF" ]; then
  F=$T/pf; mkdir -p "$F"; : > "$F/calls"; cp "$T/opts" "$T/opts.keep"   # case 10 checks renew.sh's calls only
  PATH="$T/bin:$PATH" "$PF" yyz-ird > "$F/out" 2>&1; prc=$?
  pfo=$(tail -1 "$T/opts")
  c() { [ "$prc" = 0 ] && [[ $pfo == *ControlMaster=no* && $pfo == *ControlPath=none* && $pfo == *StrictHostKeyChecking=yes* ]]; }; check "ssh-preflight: ControlMaster=no ControlPath=none, strict" c
  : > "$F/jump_key_bad"; PATH="$T/bin:$PATH" "$PF" yyz-ird > "$F/out" 2>&1; prc=$?
  check "ssh-preflight: changed key is exit 10" [ "$prc" = 10 ]
  mv "$T/opts.keep" "$T/opts"
else echo "skip $CASE: $PF not found"; fi

# 10. Every ssh and scp in all cases above: own connection (no ~/.ssh/config mux), strict host-key
# check, batch mode, a connect timeout and an outer timeout.
CASE=all_calls; OUT=; : > "$F/calls"
bad=$(grep -vE 'ControlMaster=no' "$T/opts"; grep -vE 'ControlPath=none' "$T/opts"; grep -vE 'StrictHostKeyChecking=yes' "$T/opts"
  grep -vE 'BatchMode=yes' "$T/opts"; grep -vE 'ConnectTimeout=' "$T/opts"; grep -vE '^[a-z]+ ([^ ]+ )?T=1 ' "$T/opts")
c() { [ -s "$T/opts" ] && [ -z "$bad" ]; }; check "all $(grep -c . "$T/opts") ssh/scp calls are mux-free, strict and bounded" c
[ -z "$bad" ] || sort -u <<<"$bad" | head -5 | sed 's/^/     bad: /'

# 11. Viewer deploy passes VIEWER_B4A_FOLD through (empty by default: viewer.sh's default, fold
# on since #597); VIEWER_B4A_FOLD in the caller's env overrides. viewer.sh status runs for real
# here; its ssh must run under renew.sh's outer `timeout 60` (#564).
: > "$T/opts"
new_case viewer_fold_off; run viewer --rev HEAD
c() { [ "$RC" = 0 ] && [ "$(calls 'lock viewer -- env VIEWER_HOST=bh-30 VIEWER_B4A_FOLD= opt/viewer/viewer.sh deploy ')" = 1 ]; }; check "deploy runs viewer.sh with the viewer's fold default" c
new_case viewer_fold_env; VIEWER_B4A_FOLD=0 run viewer --rev HEAD
c() { [ "$RC" = 0 ] && [ "$(calls 'lock viewer -- env VIEWER_HOST=bh-30 VIEWER_B4A_FOLD=0 opt/viewer/viewer.sh deploy ')" = 1 ]; }; check "VIEWER_B4A_FOLD=0 in the env wins" c
# 12. Every bh-30 ssh in case 11, viewer.sh status included, runs under an outer timeout (#564).
CASE=viewer_status_timeout; OUT=
st=$(grep -c '^ssh bh-30 ' "$T/opts"); unb=$(grep '^ssh bh-30 ' "$T/opts" | grep -vc ' T=1 ')
c() { [ "$st" -ge 2 ] && [ "$unb" = 0 ]; }; check "viewer.sh status ssh runs under timeout ($st calls, $unb unbounded)" c

# 13. viewer.sh turns the B4a fold on unless VIEWER_B4A_FOLD=0 (#597: on beats off in page FPS on bh-30).
vstart() { PATH="$T/bin:$PATH" HOME="$T/home" timeout 30 bash "$HERE/../viewer/viewer.sh" start >/dev/null 2>&1; }
new_case viewer_sh_fold_default; VIEWER_B4A_FOLD= vstart
c() { [ "$(calls '^ssh bh-30 .*GSPLAT_TT_B4A_FOLD=1 bash -s')" = 1 ]; }; check "viewer.sh start: fold on when VIEWER_B4A_FOLD is empty" c
new_case viewer_sh_fold_unset; (unset VIEWER_B4A_FOLD; vstart)
c() { [ "$(calls '^ssh bh-30 .*GSPLAT_TT_B4A_FOLD=1 bash -s')" = 1 ]; }; check "viewer.sh start: fold on when VIEWER_B4A_FOLD is unset" c
new_case viewer_sh_fold_off; VIEWER_B4A_FOLD=0 vstart
c() { [ "$(calls '^ssh bh-30 .*GSPLAT_TT_B4A_FOLD=0 bash -s')" = 1 ]; }; check "viewer.sh start: VIEWER_B4A_FOLD=0 turns it off" c

echo "$((N - FAILS))/$N passed"
[ "$FAILS" = 0 ]
