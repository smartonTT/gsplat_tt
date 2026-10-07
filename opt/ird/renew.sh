#!/usr/bin/env bash
# Renew one role's IRD reservation through the jump host and print one JSON line.
#   opt/ird/renew.sh <viewer|measure> [--dry-run] [--host H] [--fingerprint SHA256:...] [--stay] [--no-deploy]
# viewer : resource key 'viewer'. Keeps bh-30 (else a free p150, else a free p100), then
#          redeploys the newest best-iter-<N> tag into /localdev/$USER/viewer (opt/viewer/viewer.sh).
# measure: moves to a free p150 that is not the viewer box (never bh-30) if one is free, else
#          extends the p100. Sync, build and release run under `ttp lock p100`.
# Only the role's designated reserver runs this without --dry-run. See opt/ird/README.md.
# Exit: 0 ok; 2 usage; 3 extend refused while the job is alive; 4 pending (queued request, box
# setup still running, or viewer waiting for bh-30 to expire: rerun later); 5 no free box;
# 6 setup/deploy/sync/release failed;
# 10-13 ssh-preflight classes (10 = host key changed or unknown and not verified).
set -uo pipefail

JUMP=${IRD_JUMP:-yyz-ird}
PORT=46483
TIMEOUT=14:00:00
IRD_USER=${IRD_USER:-$USER}
RROOT=/localdev/$IRD_USER
VIEWER_BOX=bh-30
RESERVE_WAIT=${IRD_RESERVE_WAIT:-240}
PREFLIGHT=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
# bicycle.ply (md5 3745b7a6) is not in git: worktrees take the main checkout's copy.
SCENE=${IRD_SCENE:-scenes/bicycle.ply}; [ -s "$SCENE" ] || SCENE=$HOME/dev/gstt2/scenes/bicycle.ply
# Host-key checking stays on everywhere. ~/.ssh/config sets StrictHostKeyChecking=no for bh-*;
# a command-line -o wins over it, so every ssh to a box goes through BOX_SSH.
BOX_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)

ROLE=; DRY=0; HOST_ARG=; FP_ARG=; STAY=0; DEPLOY=1
while [ $# -gt 0 ]; do
  case $1 in
    viewer|measure) ROLE=$1 ;;
    --dry-run) DRY=1 ;;
    --host) HOST_ARG=${2:?--host needs a value}; shift ;;
    --fingerprint) FP_ARG=${2:?--fingerprint needs a value}; shift ;;
    --stay) STAY=1 ;;
    --no-deploy) DEPLOY=0 ;;
    *) sed -n 2,12p "$0" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$ROLE" ] || { sed -n 2,12p "$0" >&2; exit 2; }
OTHER_ROLE=measure; [ "$ROLE" = measure ] && OTHER_ROLE=viewer
cd "$(git rev-parse --show-toplevel)" || exit 2
# Per-role state lives in the repository's common git dir, which every worktree shares.
STATE_DIR=${IRD_STATE_DIR:-$(cd "$(git rev-parse --git-common-dir)" && pwd)/ird-state}

log() { echo "[renew $ROLE] $*" >&2; }
# Mutating step: printed under --dry-run, run otherwise.
act() {
  if [ "$DRY" = 1 ]; then printf '[dry-run] would run:' >&2; printf ' %q' "$@" >&2; echo >&2; return 0; fi
  log "run: $*"; "$@"
}
jump() { ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 "$JUMP" "$@" 2>&1 | sed '/Pseudo-terminal/d'; }
box() { local h=$1; shift; "${BOX_SSH[@]}" "$h" "$@"; }

# "13:49:03", "49:03" or "1-02:03:04" -> seconds
secs_of() {
  local t=$1 d=0 a b c
  case $t in *-*) d=${t%%-*}; t=${t#*-} ;; esac
  IFS=: read -r a b c <<<"$t"
  if [ -n "${c:-}" ]; then echo $((10#$d*86400 + 10#$a*3600 + 10#$b*60 + 10#$c))
  elif [ -n "${b:-}" ]; then echo $((10#$d*86400 + 10#$a*60 + 10#$b))
  else echo $((10#$d*86400 + 10#${a:-0})); fi
}
utc_in() {
  local e=$(( $(date +%s) + $1 ))
  date -u -r "$e" +%Y-%m-%dT%H:%MZ 2>/dev/null || date -u -d "@$e" +%Y-%m-%dT%H:%MZ
}

# State: last JSON line per role (current host, pending request, old job to release).
st() { jq -r --arg k "$2" '.[$k] // empty' "$STATE_DIR/$1.json" 2>/dev/null; }
DEFAULT_HOST=$VIEWER_BOX; [ "$ROLE" = measure ] && DEFAULT_HOST=yyzo-bh-04
OTHER_DEFAULT=yyzo-bh-04; [ "$ROLE" = measure ] && OTHER_DEFAULT=$VIEWER_BOX
CUR=${HOST_ARG:-$(st "$ROLE" host)}; CUR=${CUR:-$DEFAULT_HOST}
OTHER=$(st "$OTHER_ROLE" host); OTHER=${OTHER:-$OTHER_DEFAULT}
PENDING=$(st "$ROLE" pending_machine); PENDING_JOB=$(st "$ROLE" pending_job); PENDING_CL=$(st "$ROLE" pending_cluster)
OTHER_PENDING_JOB=$(st "$OTHER_ROLE" pending_job)
OLD_JOB=$(st "$ROLE" release_job); OLD_HOST=$(st "$ROLE" release_host)

# Result fields
HOST=$CUR; JOB=; CLUSTER=; LEFT_S=0; ACTION=none; STATUS=ok; DEPLOY_MSG=; NOTE=; BOX_TYPE=
emit() {  # emit <exit code>; writes state unless --dry-run, prints the one JSON line
  local rc=$1 changed=false exp=
  [ "$HOST" != "$CUR" ] && changed=true
  [ "$LEFT_S" -gt 0 ] && exp=$(utc_in "$LEFT_S")
  local j
  j=$(jq -cn --arg role "$ROLE" --arg host "$HOST" --arg job "$JOB" --arg port "$PORT" \
    --arg exp "$exp" --arg type "$BOX_TYPE" --argjson changed "$changed" --arg prev "$CUR" \
    --arg action "$ACTION" --arg status "$STATUS" --arg cluster "$CLUSTER" --arg deploy "$DEPLOY_MSG" \
    --arg note "$NOTE" --argjson dry "$([ "$DRY" = 1 ] && echo true || echo false)" \
    --arg pm "${PENDING:-}" --arg pj "${PENDING_JOB:-}" --arg pc "${PENDING_CL:-}" --arg rj "${OLD_JOB:-}" --arg rh "${OLD_HOST:-}" --argjson rc "$rc" \
    '{role:$role, host:$host, job_id:$job, port:($port|tonumber), expiry_utc:$exp, box_type:$type,
      box_changed:$changed, prev_host:$prev, cluster:$cluster, action:$action, status:$status,
      deploy:$deploy, note:($note|sub("; *$";"")), dry_run:$dry, exit:$rc}
     + (if $pm != "" then {pending_machine:$pm, pending_job:$pj, pending_cluster:$pc} else {} end)
     + (if $rj != "" then {release_job:$rj, release_host:$rh} else {} end)')
  # HOST changes only once a reservation on it exists, so the next run finds the job it holds.
  [ "$DRY" = 0 ] && mkdir -p "$STATE_DIR" && echo "$j" > "$STATE_DIR/$ROLE.json"
  echo "$j"
  exit "$rc"
}

# 1. Jump host
pf=$("$PREFLIGHT" "$JUMP" 2>&1); prc=$?
if [ $prc -ne 0 ]; then
  STATUS="jump_${pf##*SSH_PREFLIGHT=}"; NOTE="ssh-preflight $JUMP failed: $(tail -3 <<<"$pf" | tr '\n' ' ')"
  [ $prc = 10 ] && NOTE="HIGH SEVERITY ALERT: ssh host key changed for $JUMP. $NOTE"
  emit $prc
fi
[ "$CUR" = "$OTHER" ] && { STATUS=role_conflict; NOTE="$ROLE host $CUR is the $OTHER_ROLE box; refusing"; emit 2; }

# 2. Read-only IRD state: our reservations, our slurm jobs and the machine list
# LIST: "sel cluster machine left job port" per running reservation (ird list shows only running jobs).
# SLURM: "cluster job state node|(reason)" per ird job of ours in any state, queued ones included
# (ird list-slurm-reservations prints squeue per responsive cluster); SLURM_CL: the clusters it covered.
list_ours() { LIST=$(jump "ird list" | awk -v u="$IRD_USER" '$1 ~ /^[0-9]+$/ && $2 == u {print $1, $3, $5, $6, $10, $11}'); }
refresh() {
  list_ours
  local raw; raw=$(jump "ird list-slurm-reservations")
  SLURM=$(awk -v u="${IRD_USER:0:8}" '/^CLUSTER:/ {c = $2; next}
    $1 ~ /^[0-9]+$/ && $4 == u && $3 ~ /^interact/ {print c, $1, $5, $NF}' <<<"$raw")
  SLURM_CL=" $(awk '/^CLUSTER:/ {printf "%s ", $2}' <<<"$raw")"
  # Requests of ours that are not running yet (queued, or allocated but not set up): not ours to
  # double. The other role's known pending request is its own business.
  QUEUED=$(awk -v o="$OTHER_PENDING_JOB" 'NR == FNR {run[$5]; next} !($2 in run) && $2 != o {print $1, $2, $3, $4}' \
    <(printf '%s\n' "$LIST") <(printf '%s\n' "$SLURM"))
}
refresh
MACH=$(jump "ird list-machines --noheader" | awk 'NF >= 10 && $4 == "blackhole" {
  t = $8; f = $8; sub(/\(.*/, "", t); sub(/^[^(]*\(/, "", f); sub(/\|.*/, "", f); print $1, $2, $5, t, f + 0, $10 }')
[ -n "$MACH" ] || { STATUS=ird_unavailable; NOTE="ird list-machines returned nothing"; emit 13; }
log "our reservations: $(tr '\n' ';' <<<"$LIST")"
row_m() { awk -v m="$1" '$3 == m' <<<"$LIST" | head -1; }
row_j() { awk -v j="$1" '$5 == j' <<<"$LIST" | head -1; }
type_of() { awk -v m="$1" '$1 == m {print $4}' <<<"$MACH"; }
# Free machines of a type (glob), minus the excluded ones; one "machine cluster" per line.
free_of() {
  local ty=$1 ex m c s t f ids; shift
  ex=" $* $(awk '{printf "%s ", $3}' <<<"$LIST")"   # nor a machine we already hold
  while read -r m c s t f ids; do
    # shellcheck disable=SC2053  # $ty is a glob on purpose
    [[ $t == $ty ]] || continue
    [[ $ex == *" $m "* ]] && continue
    case $s in *drain*|*down*|*reserved*|*maint*|*fail*) continue ;; esac
    [ "$f" -ge 1 ] && [ "$ids" != "-" ] || continue
    case $m in bh-*|yyzo-bh-*) echo "$m $c" ;; esac   # only names ~/.ssh/config routes via the jump host
  done <<<"$MACH"
}

ROW=$(row_m "$CUR")
OLD_ROW=
# A request an earlier run left queued: adopt it once it runs, wait while it is queued, forget it
# once it is gone (crashed, cancelled or expired).
if [ -n "$PENDING" ]; then
  [ -n "$PENDING_CL" ] || PENDING_CL=$(awk -v m="$PENDING" '$1 == m {print $2}' <<<"$MACH")
  arow=
  if [ -n "$PENDING_JOB" ]; then arow=$(row_j "$PENDING_JOB")
  else arow=$(row_m "$PENDING"); [ -n "$arow" ] && [ "$arow" = "$ROW" ] && arow=; fi
  if [ -n "$arow" ]; then
    ahost=$(awk '{print $3}' <<<"$arow")
    log "queued request on $ahost (job $(awk '{print $5}' <<<"$arow")) is now running: adopting it"
    if [ "$ROLE" = measure ] && [ -n "$ROW" ] && [ "$ahost" != "$CUR" ]; then
      OLD_ROW=$ROW; OLD_JOB=$(awk '{print $5}' <<<"$ROW"); OLD_HOST=$CUR; ACTION=moved
    else ACTION=adopted; fi
    HOST=$ahost; ROW=$arow; PENDING=; PENDING_JOB=; PENDING_CL=
  elif [ -n "$PENDING_JOB" ] && awk -v j="$PENDING_JOB" '$2 == j {f = 1} END {exit !f}' <<<"$QUEUED"; then
    log "request $PENDING_JOB for $PENDING is still queued"
  elif [ -z "$PENDING_JOB" ] && awk -v c="$PENDING_CL" '$1 == c {f = 1} END {exit !f}' <<<"$QUEUED"; then
    log "a request of ours on $PENDING_CL (for $PENDING) is still queued"
  elif [ -n "$PENDING_CL" ] && [[ $SLURM_CL != *" $PENDING_CL "* ]]; then
    log "cluster $PENDING_CL did not answer: keeping the request for $PENDING as queued"
    QUEUED="$PENDING_CL ${PENDING_JOB:-?} unknown -"
  else
    log "request for $PENDING (job ${PENDING_JOB:-?}) no longer exists: forgetting it"
    NOTE="${NOTE}cleared stale request for $PENDING; "; PENDING=; PENDING_JOB=; PENDING_CL=
  fi
fi
# Never file a second request while one of ours is queued.
[ -n "$QUEUED" ] && log "queued request(s) of ours: $(tr '\n' ';' <<<"$QUEUED"); no new ird reserve this run"

# 3. Candidates for a new reservation
cands() {
  if [ "$ROLE" = viewer ]; then
    free_of 'p150*' "$OTHER" | awk -v v="$VIEWER_BOX" '$1 == v'
    free_of 'p150*' "$OTHER" "$VIEWER_BOX"
    free_of 'p100*' "$OTHER" "$VIEWER_BOX"
  else
    free_of 'p150*' "$OTHER" "$VIEWER_BOX"
    [ "${1:-}" = p150only ] && return
    free_of 'p100*' "$OTHER" "$VIEWER_BOX" | awk -v c="$CUR" '$1 == c'
    free_of 'p100*' "$OTHER" "$VIEWER_BOX" "$CUR"
  fi
}

# Reserve machine $1 on cluster $2. Sets ROW/HOST on success; returns 4 when queued.
reserve() {
  local m=$1 c=$2 out job before
  local cmd="ird reserve --cluster $c --ssh-port $PORT --timeout $TIMEOUT --no-shell blackhole --machine $m"
  if [ "$DRY" = 1 ]; then act ssh "$JUMP" "$cmd"; HOST=$m; CLUSTER=$c; JOB=planned; ROW=; LEFT_S=$(secs_of "$TIMEOUT"); return 0; fi
  log "reserving $m ($c)"
  before=" $(awk '{printf "%s ", $2}' <<<"$SLURM")"
  # ird reserve blocks while the request is queued; the output carries a container password: never log it.
  out=$(timeout "$RESERVE_WAIT" ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$JUMP" "$cmd" 2>&1 | grep -vi password)
  tail -4 <<<"$out" | sed 's/^/[ird] /' >&2
  job=$(grep -oE '[0-9]+ on [^ :]+: Reservation complete' <<<"$out" | awk '{print $1}' | tail -1)
  refresh
  ROW=$(if [ -n "$job" ]; then row_j "$job"; else row_m "$m"; fi)
  if [ -n "$ROW" ]; then HOST=$m; PENDING=; PENDING_JOB=; PENDING_CL=; return 0; fi
  # Queued: salloc (inside ird, output captured) still waits when the timeout ends the ssh, and its
  # request stays in slurm. A job of ours on $c that was not listed before is that request; salloc's
  # own lines are the fallback. Anything else (an ird traceback, an salloc error) filed nothing.
  job=$(awk -v c="$c" -v b="$before" '$1 == c && index(b, " " $2 " ") == 0 {print $2}' <<<"$SLURM" | tail -1)
  [ -n "$job" ] || job=$(grep -oE 'salloc: (Pending job allocation [0-9]+|job [0-9]+ queued and waiting for resources)' <<<"$out" |
    grep -oE '[0-9]+' | tail -1)
  if [ -n "$job" ]; then PENDING=$m; PENDING_JOB=$job; PENDING_CL=$c; QUEUED="$c $job PD -"; return 4; fi
  return 1
}
try_reserve() {  # try candidates in order; 0 reserved, 4 queued, 5 none worked
  local m c rc
  while read -r m c; do
    [ -n "$m" ] || continue
    reserve "$m" "$c"; rc=$?
    [ $rc = 0 ] && { ACTION=reserved; return 0; }
    [ $rc = 4 ] && { ACTION=reserve_queued; return 4; }
    log "reserve $m failed; trying the next free box"
  done <<<"$(cands "${1:-}")"
  return 5
}

# 4. Decide: move (measure -> p150), extend, or re-reserve
if [ "$ROLE" = measure ] && [ "$STAY" = 0 ] && [ "$ACTION" = none ] && [ -z "$QUEUED" ] && [ -n "$ROW" ] \
   && [[ $(type_of "$CUR") != p150* ]] && [ -n "$(cands p150only)" ]; then
  log "a non-viewer p150 is free: moving the measurement box off $CUR ($(type_of "$CUR"))"
  OLD_ROW=$ROW; try_reserve p150only; rc=$?
  if [ $rc = 0 ]; then
    OLD_JOB=$(awk '{print $5}' <<<"$OLD_ROW"); OLD_HOST=$CUR; ACTION=moved
  else
    log "no p150 reservation came up (rc $rc): staying on $CUR"; ROW=$OLD_ROW; HOST=$CUR; OLD_ROW=
    [ $rc = 4 ] && NOTE="${NOTE}p150 request on $PENDING (job $PENDING_JOB) is queued; "
  fi
fi
[ -n "$PENDING" ] && [ "$ACTION" != reserve_queued ] && NOTE="${NOTE}request for $PENDING (job ${PENDING_JOB:-?}) is still queued; "
if [ -n "$ROW" ] && [ "$HOST" = "$CUR" ] && [ "$ACTION" != moved ] && [ "$ACTION" != adopted ]; then
  read -r sel _ _ left _ _ <<<"$ROW"
  before=$(secs_of "$left")
  if [ "$DRY" = 1 ]; then act ssh "$JUMP" "ird change-timeout $sel $TIMEOUT"; ACTION=extend; LEFT_S=$(secs_of "$TIMEOUT")
  else
    log "run: ird change-timeout $sel $TIMEOUT (job on $CUR, $left left)"
    jump "ird change-timeout $sel $TIMEOUT" | grep -vE 'Responsive clusters|Using clusters' | tail -3 >&2
    list_ours
    ROW=$(row_m "$CUR")
    after=$( [ -n "$ROW" ] && secs_of "$(awk '{print $4}' <<<"$ROW")" || echo 0)
    if [ "$after" -gt "$before" ]; then ACTION=extended
    elif [ "$after" -gt 3600 ] && [ -n "$OLD_JOB" ]; then
      log "ird change-timeout did not extend $CUR ($after s left): finishing the move first"; ACTION=extend_refused
    elif [ "$after" -gt 3600 ]; then
      ACTION=extend_refused; STATUS=extend_refused; LEFT_S=$after
      NOTE="${NOTE}ird change-timeout did not extend job on $CUR ($(awk '{print $4}' <<<"$ROW") left); rerun within the last hour or after expiry to re-reserve; "
      JOB=$(awk '{print $5}' <<<"$ROW"); CLUSTER=$(awk '{print $2}' <<<"$ROW"); BOX_TYPE=$(type_of "$CUR"); emit 3
    elif [ "$ROLE" = viewer ] && [ "$after" -gt 0 ]; then
      # Our own job still holds bh-30, so a re-reserve now would move the viewer off it for good.
      ACTION=wait_expiry; STATUS=wait_expiry; LEFT_S=$after
      NOTE="${NOTE}ird change-timeout did not extend job on $CUR ($(awk '{print $4}' <<<"$ROW") left): rerun after it expires (~$(utc_in "$after")) to re-reserve $CUR; "
      JOB=$(awk '{print $5}' <<<"$ROW"); CLUSTER=$(awk '{print $2}' <<<"$ROW"); BOX_TYPE=$(type_of "$CUR"); emit 4
    else log "extension failed and the job is ending: re-reserving"; ROW=; fi
  fi
fi
if [ -z "$ROW" ] && [ -n "$QUEUED" ]; then
  STATUS=pending; ACTION=wait_queued; NOTE="${NOTE}the $ROLE job on $CUR is gone and a request of ours is queued ($(tr '\n' ';' <<<"$QUEUED")): no second request; rerun later"; emit 4
fi
if [ -z "$ROW" ] && [ "$ACTION" != moved ] && [ "$ACTION" != reserved ]; then
  try_reserve; rc=$?
  if [ $rc = 4 ]; then STATUS=pending; NOTE="${NOTE}request for $PENDING (job $PENDING_JOB) is queued; rerun later"; emit 4; fi
  if [ $rc = 5 ]; then STATUS=no_free_box; ACTION=none; NOTE="${NOTE}no free box for role $ROLE (and the $ROLE job on $CUR is gone)"; emit 5; fi
fi
if [ -n "$ROW" ]; then
  read -r _ CLUSTER _ left JOB _ <<<"$ROW"
  [ "$LEFT_S" -gt 0 ] || LEFT_S=$(secs_of "$left")
fi
BOX_TYPE=$(type_of "$HOST")
[ "$ROLE" = viewer ] && [ "$HOST" != "$VIEWER_BOX" ] && NOTE="${NOTE}viewer is on $HOST, not bh-30: tell the user to change the Mac forward to 8091 -> $HOST:8080; "
[ "$HOST" = "$VIEWER_BOX" ] && [ "$ROLE" = measure ] && { STATUS=role_conflict; NOTE="measure picked $VIEWER_BOX"; emit 2; }

# An unfinished move keeps the old box until the new one is built: extend it too.
if [ -n "$OLD_JOB" ]; then
  osel=$(awk '{print $1}' <<<"$(row_j "$OLD_JOB")")
  if [ -z "$osel" ]; then log "old job $OLD_JOB on $OLD_HOST is gone already"
  elif [ "$DRY" = 1 ]; then act ssh "$JUMP" "ird change-timeout $osel $TIMEOUT"
  else
    log "run: ird change-timeout $osel $TIMEOUT (old job $OLD_JOB on $OLD_HOST, kept until $HOST is built)"
    jump "ird change-timeout $osel $TIMEOUT" | grep -vE 'Responsive clusters|Using clusters' | tail -3 >&2
  fi
fi

# A box that --dry-run only plans to reserve is not ours yet: no ssh to it at all.
PLANNED=0; [ "$DRY" = 1 ] && [ "$HOST" != "$CUR" ] && PLANNED=1

# 5. Host key: a strict check first. A plain ssh (as ssh-preflight runs) would, under the
# StrictHostKeyChecking=no in ~/.ssh/config, store an unknown key unverified.
verify_key() {  # refresh known_hosts only if the box's own key (via IRD) matches what the network offers
  local h=$1 sel offered trusted ofp tfp
  offered=$(jump "ssh-keyscan -T 10 -p $PORT -t ed25519 $h 2>/dev/null" | grep -E '^[^#].* ssh-ed25519 ' | head -1)
  ofp=$(ssh-keygen -lf /dev/stdin <<<"$offered" 2>/dev/null | awk '{print $2}')
  if [ -n "$FP_ARG" ]; then tfp=$FP_ARG
  else
    sel=$(awk '{print $1}' <<<"$(row_m "$h")")
    [ -n "$sel" ] || { log "no ird selection for $h"; return 1; }
    # ird connect-to opens a shell in our own container over IRD's channel, not the box's sshd.
    trusted=$(printf 'cat /etc/ssh/ssh_host_ed25519_key.pub; exit\n' |
      timeout 120 ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$JUMP" "ird connect-to $sel" 2>/dev/null |
      grep -oE 'ssh-ed25519 [A-Za-z0-9+/=]+' | head -1)
    tfp=$(ssh-keygen -lf /dev/stdin <<<"$trusted" 2>/dev/null | awk '{print $2}')
  fi
  log "offered ed25519 key $h:$PORT ${ofp:-none}; trusted ${tfp:-none}"
  [ -n "$ofp" ] && [ "$ofp" = "$tfp" ] || return 1
  act ssh-keygen -R "[$h]:$PORT" >/dev/null 2>&1
  if [ "$DRY" = 0 ]; then printf '[%s]:%s %s\n' "$h" "$PORT" "$(awk '{print $2, $3}' <<<"$offered")" >> "$HOME/.ssh/known_hosts"
  else echo "[dry-run] would append the verified key for [$h]:$PORT to ~/.ssh/known_hosts" >&2; fi
}
krc=0; kerr=
if [ $PLANNED = 1 ]; then echo "[dry-run] would check the host key of $HOST strictly, then run ssh-preflight $HOST" >&2
else kerr=$(box "$HOST" true 2>&1 >/dev/null); krc=$?; fi
if [ $krc -ne 0 ]; then
  case $kerr in
    *"IDENTIFICATION HAS CHANGED"*|*"Host key verification failed"*|*"host key is known"*)
      log "host key of $HOST changed or unknown: verifying the fingerprint through IRD"
      if [ "$DRY" = 1 ]; then
        echo "[dry-run] would compare ssh-keyscan via $JUMP with the key read through 'ird connect-to' (or --fingerprint), then refresh [$HOST]:$PORT in ~/.ssh/known_hosts only on a match" >&2
        NOTE="${NOTE}host key of $HOST needs verification; "
      elif ! verify_key "$HOST" || ! box "$HOST" true; then
        STATUS=host_key_changed; NOTE="HIGH SEVERITY ALERT: ssh host key changed for $HOST and it could not be verified: $(tail -2 <<<"$kerr" | tr '\n' ' ')"
        emit 10
      fi ;;
    *) pf=$("$PREFLIGHT" "$HOST" 2>&1); prc=$?
       [ $prc -ne 0 ] && { STATUS="box_${pf##*SSH_PREFLIGHT=}"; NOTE="${NOTE}ssh to $HOST failed: $(tail -2 <<<"$kerr" | tr '\n' ' ')"; emit $prc; } ;;
  esac
fi
if [ $PLANNED = 0 ] && { [ "$DRY" = 0 ] || [ $krc = 0 ]; }; then
  pf=$("$PREFLIGHT" "$HOST" 2>&1); prc=$?
  [ $prc -ne 0 ] && { STATUS="box_${pf##*SSH_PREFLIGHT=}"; NOTE="${NOTE}ssh-preflight $HOST: $(tail -2 <<<"$pf" | tr '\n' ' ')"; emit $prc; }
fi

# 6. Box setup (tt-metal, venv, scenes) on a box that lacks it: started detached, rerun when done.
ensure_setup() {  # ensure_setup <tt dir> <venv> <scenes dir> [lock cmd...]
  local tt=$1 venv=$2 scenes=$3; shift 3
  local s=
  [ $PLANNED = 1 ] || s=$(box "$HOST" "test -f $tt/build/tt_metal/libtt_metal.so && test -x $venv/bin/python && test -s $scenes/bicycle.ply && echo ready;
    pgrep -u \$USER -f setup_box.sh >/dev/null && echo running; cat $RROOT/viewer/setup.rc 2>/dev/null" 2>/dev/null)
  grep -qx ready <<<"$s" && return 0
  local src; src=$(grep -xE '[0-9]+' <<<"$s" | tail -1)
  if [ -n "$src" ] && [ "$src" != 0 ] && ! grep -qx running <<<"$s"; then
    # A failed setup is reported, not restarted: someone has to look at the log first.
    STATUS=setup_failed; DEPLOY_MSG="setup_box.sh failed on $HOST (setup.rc $src)"
    NOTE="${NOTE}setup_box.sh failed on $HOST (rc $src); fix it, then delete $RROOT/viewer/setup.rc and rerun; log tail: $(box "$HOST" "tail -5 $RROOT/viewer/setup.log" 2>/dev/null | tr '\n' ' '); "
    return 6
  fi
  if grep -qx running <<<"$s"; then STATUS=setup_running; NOTE="${NOTE}setup_box.sh is running on $HOST ($RROOT/viewer/setup.log); rerun when $RROOT/viewer/setup.rc exists; "; return 4; fi
  log "box $HOST lacks tt-metal/venv/scenes (last setup.rc: ${s:-none}): starting setup"
  act "$@" "${BOX_SSH[@]}" "$HOST" "mkdir -p $RROOT/viewer $scenes && cat > $RROOT/viewer/setup_box.sh" < opt/viewer/setup_box.sh
  if [ $PLANNED = 1 ] || ! box "$HOST" "test -s $scenes/bicycle.ply" 2>/dev/null; then
    act "$@" scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$SCENE" "$HOST:$scenes/bicycle.ply"
  fi
  act "$@" "${BOX_SSH[@]}" "$HOST" "(TT=$tt VENV=$venv setsid nohup bash $RROOT/viewer/setup_box.sh > $RROOT/viewer/setup.log 2>&1 < /dev/null &)"
  STATUS=setup_running; NOTE="${NOTE}started setup_box.sh on $HOST; rerun when $RROOT/viewer/setup.rc exists (0 = ok); "
  return 4
}

if [ "$DEPLOY" = 1 ] && [ "$ROLE" = viewer ]; then
  VDIR=$RROOT/viewer
  # Never /localdev/$USER/gstt2 (a devsync mirror of the Mac) and nothing in /home (quota full).
  ensure_setup "$VDIR/tt-metal" "$VDIR/venv" "$VDIR/scenes" ttp lock viewer --; rc=$?
  [ $rc = 4 ] || [ $rc = 6 ] && emit $rc
  timeout 60 git fetch -q --tags origin 2>/dev/null || log "git fetch failed: using local tags"
  TAG=$(git tag -l 'best-iter-*' --sort=-v:refname | head -1)
  TSHA=$(git rev-parse "$TAG^{commit}")
  vs=; [ $PLANNED = 1 ] || vs=$(env VIEWER_HOST="$HOST" opt/viewer/viewer.sh status 2>&1 | head -1)
  DSHA=$(sed -nE 's/.*sha=([0-9a-f]+).*/\1/p' <<<"$vs")
  if [ -n "$DSHA" ] && git merge-base --is-ancestor "$TSHA" "$DSHA" 2>/dev/null; then
    if [[ $vs == *running* ]]; then DEPLOY_MSG="current: $DSHA contains $TAG, running"
    else
      DEPLOY_MSG="started: $DSHA contains $TAG"
      act ttp lock viewer -- env VIEWER_HOST="$HOST" opt/viewer/viewer.sh start >&2 || { DEPLOY_MSG="start failed"; STATUS=deploy_failed; emit 6; }
    fi
  else
    DEPLOY_MSG="deployed $TAG ($TSHA) over ${DSHA:-none}"
    act ttp lock viewer -- env VIEWER_HOST="$HOST" opt/viewer/viewer.sh deploy "$TAG" >&2 || { DEPLOY_MSG="deploy $TAG failed"; STATUS=deploy_failed; emit 6; }
  fi
  [ "$DRY" = 1 ] && DEPLOY_MSG="planned: $DEPLOY_MSG"
fi

if [ "$DEPLOY" = 1 ] && [ "$ROLE" = measure ] && { [ "$HOST" != "$CUR" ] || [ -n "$OLD_JOB" ]; }; then
  # New measurement box: base tree /localdev/$USER/gstt2 with its own venv and scenes, built at the newest tag.
  ensure_setup "$RROOT/tt-metal" "$RROOT/gstt2/.venv" "$RROOT/gstt2/scenes" ttp lock p100 --; rc=$?
  if [ $rc = 4 ] || [ $rc = 6 ]; then
    [ -n "$OLD_JOB" ] && NOTE="${NOTE}old box $OLD_HOST (job $OLD_JOB) stays reserved until the new one builds; "
    emit $rc
  fi
  TAG=$(git tag -l 'best-iter-*' --sort=-v:refname | head -1)
  act ttp lock p100 -- opt/sync_remote.sh "$HOST" "$RROOT/gstt2" "$TAG" >&2 || { STATUS=sync_failed; DEPLOY_MSG="sync+build of $TAG on $HOST failed"; emit 6; }
  DEPLOY_MSG="synced+built $TAG into $HOST:$RROOT/gstt2"
  if [ -n "$OLD_JOB" ]; then
    osel=$(awk '{print $1}' <<<"$(row_j "$OLD_JOB")")
    if [ -n "$osel" ]; then
      act ttp lock p100 -- ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$JUMP" "echo y | ird release $osel" >&2 ||
        { STATUS=release_failed; NOTE="${NOTE}ird release of job $OLD_JOB on $OLD_HOST failed"; emit 6; }
      NOTE="${NOTE}released old job $OLD_JOB on $OLD_HOST; "
    fi
    [ "$DRY" = 0 ] && { OLD_JOB=; OLD_HOST=; }
  fi
  [ "$DRY" = 1 ] && DEPLOY_MSG="planned: $DEPLOY_MSG"
fi
[ "$HOST" != "$CUR" ] && NOTE="${NOTE}box changed $CUR -> $HOST: update the host in memory and drivers; absolute ms/view are box-specific; "
emit 0
