#!/usr/bin/env bash
# Sync a committed tree to a remote build dir, build, and flag a stale host .so.
#   opt/sync_remote.sh [--dry-run|--list|--no-build] <host> [remote_dir] [rev=HEAD]
#   opt/sync_remote.sh --pull <remote_out> <local_out> <host> [remote_dir]
# One-way, repo-scoped rsync (user msg #369): device runs never rely on the user's
# mutagen/devsync mirrors of ~/dev. The source is always the calling worktree's own
# repo, which must resolve under /Users/smarton/dev/gsplat_tt or ~/dev/gstt2 (their
# worktrees included); anything else is refused. The tracked files of <rev> that a
# device run reads (sources, kernels, CMake inputs, python, docs/* scripts; see
# list_files) are checked out into tmp/sync-stage/<sha> and rsynced with --checksum:
# only changed files are sent and get fresh mtimes, so ninja rebuilds what changed
# (git archive mtimes hid changes from ninja, task #47). Nothing is deleted on the
# remote. The remote gets SHA and SYNC_STAMP (sha, source worktree, time); no .git.
# Scenes (SYNC_SCENES, default bicycle.ply) are copied into the remote scenes dir
# only when missing or their md5 differs.
#   remote_dir  default $SYNC_REMOTE_DIR or /localdev/$USER/gstt2-sync (project-owned)
#   --dry-run   rsync -n against the remote: list what would be sent; no build, no copy
#   --list      no ssh: print the file list summary (SYNC_LIST_OUT=<file> keeps the list)
#   --no-build  sync tree and scenes only
#   --pull R L  copy <remote_dir>/R (relative, no ..) back into L inside this worktree
# Wrap device use in `ttp lock p100 -- ...`: the sync, build and run of one device
# run go inside ONE lock call.
# Exit codes: 0 ok, 1 configure/build failed (last log lines printed), 2 refused,
# 3 stale .so, 4 rsync/ssh transfer failed. Callers MUST check the exit code: don't
# pipe the output (e.g. `| tail`) without `set -o pipefail`, or a failed build looks
# like success.
# Remote env defaults: TT_METAL_HOME=/localdev/$USER/tt-metal,
# TT_METAL_ARCH_NAME=blackhole; set them on the remote side to override.
# A fresh remote dir is made self-sufficient: missing .venv and scenes are
# symlinked to $BASE (/localdev/$USER/gstt2, the box's own venv and scenes made by
# opt/viewer/setup_box.sh through opt/ird/renew.sh, not a mirror), and a missing
# _gsplat_cpu .so is built (build-avx2-tt) or, if that fails, copied from $BASE.
# GSTT2_BASE, GSTT2_VENV and GSTT2_SCENES (Mac side) override $BASE, the venv and
# the scenes dir; set links that point elsewhere are replaced. The viewer deploy
# uses them for its own $VIEWER_DIR (task #324).
# SYNC_REMOTE_JOBS sets the build -j (default 16). On a viewer host the build runs
# under nice -n 19 ionice -c3 with half the remote cores, so it does not starve the
# live viewer (user, 2026-10-07). SYNC_REMOTE_DRY_RUN=1 prints the build settings
# and exits without any ssh.
set -euo pipefail
MODE=sync PULL_R= PULL_L=
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) MODE=dry; shift ;;
    --list) MODE=list; shift ;;
    --no-build) MODE=nobuild; shift ;;
    --pull) MODE=pull; PULL_R=${2:?--pull <remote_out> <local_out>}; PULL_L=${3:?--pull <remote_out> <local_out>}; shift 3 ;;
    --) shift; break ;;
    -*) echo "[sync_remote] unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
HOST=${1:?host}; DIR=${2:-${SYNC_REMOTE_DIR:-/localdev/$USER/gstt2-sync}}; REV=${3:-HEAD}
die() { echo "[sync_remote] $1" >&2; exit "${2:-2}"; }
case $DIR in /*) ;; *) die "remote_dir must be absolute: $DIR" ;; esac
case "/$DIR/" in */../*|*/./*) die "remote_dir must not contain . or .. parts: $DIR" ;; esac
# The viewer box is the user's always-on viewer: only opt/viewer/viewer.sh
# (SYNC_REMOTE_VIEWER=1) deploys there, never a measurement task (task #324).
case " ${GSTT2_VIEWER_HOSTS:-bh-30} " in
  *" $HOST "*) [ "${SYNC_REMOTE_VIEWER:-0}" = 1 ] || [ "$MODE" = list ] || {
    echo "[sync_remote] refusing $HOST: it is the viewer box; deploy there with opt/viewer/viewer.sh" >&2; exit 2; }
    VIEWER_BUILD=1 ;;
esac
VIEWER_BUILD=${VIEWER_BUILD:-0} JOBS=${SYNC_REMOTE_JOBS:-16}
if [ "${SYNC_REMOTE_DRY_RUN:-0}" = 1 ]; then
  if [ "$VIEWER_BUILD" = 1 ]; then echo "[sync_remote] dry run: $HOST build: nice -n 19 ionice -c3, -j \$(( \$(nproc)/2 ))"
  else echo "[sync_remote] dry run: $HOST build: -j $JOBS"; fi
  exit 0
fi

# Source guard: only the two project repos (and their worktrees) are ever copied.
real() { (cd "$1" 2>/dev/null && pwd -P); }
TOP=$(real "$(git rev-parse --show-toplevel)") || die "not inside a git worktree"
in_roots() {  # path -> 0 if it resolves strictly inside an allowed repo root
  local p=$1 r
  for r in "${SYNC_ALLOWED_ROOTS[@]}"; do
    [ -n "$r" ] || continue
    case "$p/" in "$r"/*) return 0 ;; esac
  done
  return 1
}
if [ -n "${SYNC_REMOTE_ALLOWED_ROOTS:-}" ]; then  # tests only
  read -r -a SYNC_ALLOWED_ROOTS <<< "$SYNC_REMOTE_ALLOWED_ROOTS"
else
  SYNC_ALLOWED_ROOTS=("$(real /Users/smarton/dev/gsplat_tt || true)" "$(real "$HOME/dev/gstt2" || true)")
fi
for bad in "$(real "$HOME")" "$(real "$HOME/dev" || true)" /; do
  [ "$TOP" != "$bad" ] || die "refusing source $TOP: never sync \$HOME, ~/dev or /"
done
in_roots "$TOP" || die "refusing source $TOP: not under /Users/smarton/dev/gsplat_tt or ~/dev/gstt2"
cd "$TOP"

# Own connection with a strict key check: never the ~/.ssh/config mux, which checks no key (#496).
SSH_OPTS=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
RSH="ssh ${SSH_OPTS[*]}"
RSYNC=(rsync -e "$RSH")

if [ "$MODE" = pull ]; then
  case "/$PULL_R/" in /|//|*/../*) die "--pull remote_out must be a relative path inside $DIR: $PULL_R" ;; esac
  case $PULL_R in /*) die "--pull remote_out must be relative: $PULL_R" ;; esac
  mkdir -p "$PULL_L"
  L=$(real "$PULL_L") || die "cannot resolve $PULL_L"
  case "$L/" in "$TOP"/*) ;; *) die "--pull local_out must be inside this worktree ($TOP): $L" ;; esac
  out=$("${RSYNC[@]}" -rlt --stats "$HOST:$DIR/${PULL_R%/}/" "$L/") || { echo "$out" >&2; die "pull of $HOST:$DIR/$PULL_R failed" 4; }
  n=$(sed -nE 's/^Number of regular files transferred: ([0-9,]+).*/\1/p' <<<"$out" | tr -d ,)
  b=$(sed -nE 's/^Total transferred file size: ([0-9,]+).*/\1/p' <<<"$out" | tr -d ,)
  echo "[sync_remote] pulled $HOST:$DIR/$PULL_R -> $L: ${n:-0} files, ${b:-0} bytes"
  exit 0
fi

SHA=$(git rev-parse --verify "$REV^{commit}") || die "cannot resolve rev $REV"
# The files a device run reads. Skipped: the LFS hero fixtures (*.npz, test inputs
# only), committed profiler captures (ttw-*/, *.tracy, profile_log_device*.csv) and
# screenshots, .pyc, and in docs/ everything but scripts and small inputs (results,
# logs and images there are ~1 GB). SYNC_ALL=1 lists every tracked file.
list_files() {  # sha -> "size<TAB>path" lines of the files to send
  git -c core.quotePath=false ls-tree -r -l "$1" | awk -v all="${SYNC_ALL:-0}" '
  { t=index($0, "\t"); split(substr($0, 1, t-1), m, " "); p=substr($0, t+1); s=m[4]
    if (m[2] != "blob") next
    if (p ~ /^"/) { print "[sync_remote] skipping quoted path " p > "/dev/stderr"; next }
    if (all != 1) {
      if (p ~ /^tests\/fixtures\/hero\/.*\.npz$/ || p ~ /^opt\/profiler\/ttw-/ || p ~ /^opt\/metal-screenshots\// ||
          p ~ /\.tracy$/ || p ~ /(^|\/)profile_log_device[^\/]*\.csv$/ || p ~ /(^|\/)__pycache__\// ||
          p ~ /\.pyc$/ || p == ".venv" || p ~ /^scenes\//) next
      if (p ~ /^docs\// && (p !~ /\.(sh|py|json|toml|cfg|patch|diff|cpp|hpp|h|yaml|yml|mjs)$/ || s+0 > 1048576)) next
    }
    printf "%s\t%s\n", s, p }'
}
[ -n "${SYNC_LIST_OUT:-}" ] && case $SYNC_LIST_OUT in /*) ;; *) SYNC_LIST_OUT=$OLDPWD/$SYNC_LIST_OUT ;; esac
STAGE_ROOT=$TOP/tmp/sync-stage STAGE=$TOP/tmp/sync-stage/$SHA
LIST=$(mktemp "${TMPDIR:-/tmp}/sync_remote.XXXXXX"); trap 'rm -f "$LIST" "$LIST".*' EXIT
list_files "$SHA" > "$LIST.sz"
cut -f2- < "$LIST.sz" > "$LIST"
NF=$(wc -l < "$LIST" | tr -d ' ')
NB=$(awk -F'\t' '{b+=$1} END{printf "%d", b}' "$LIST.sz")
echo "[sync_remote] $SHA from $TOP: $NF files, $NB bytes listed"
if [ "$MODE" = list ]; then
  awk -F'\t' '{split($2,a,"/"); k=(a[2]==""?a[1]:a[1]"/"); n[k]++; b[k]+=$1}
    END{for (k in n) printf "%12d %6d %s\n", b[k], n[k], k}' "$LIST.sz" | sort -rn
  [ -n "${SYNC_LIST_OUT:-}" ] && mkdir -p "$(dirname "$SYNC_LIST_OUT")" && cp "$LIST" "$SYNC_LIST_OUT" && echo "[sync_remote] list in $SYNC_LIST_OUT"
  exit 0
fi

# Stage the listed files at $SHA (own index, so the worktree and its index are untouched).
if [ ! -f "$STAGE/.complete" ]; then
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  GIT_INDEX_FILE=$LIST.idx git read-tree "$SHA"
  GIT_INDEX_FILE=$LIST.idx git checkout-index --stdin --prefix="$STAGE/" < "$LIST"
  : > "$STAGE/.complete"
fi
for o in "$STAGE_ROOT"/*; do [ "$o" = "$STAGE" ] || rm -rf "$o"; done   # keep one stage only
echo "$SHA" > "$STAGE/SHA"
printf 'sha=%s\nsource=%s\nsynced_utc=%s\nfiles=%s\nbytes=%s\n' "$SHA" "$TOP" "$(date -u +%FT%TZ)" "$NF" "$NB" > "$STAGE/SYNC_STAMP"
printf 'SHA\nSYNC_STAMP\n' >> "$LIST"

RS=(-rlEc --files-from="$LIST" --mkpath --stats)
if [ "$MODE" = dry ]; then
  out=$("${RSYNC[@]}" "${RS[@]}" -n --out-format='%n' "$STAGE/" "$HOST:$DIR/") || { echo "$out" >&2; die "rsync --dry-run to $HOST:$DIR failed" 4; }
  sed -n '/^Number of files:/q;p' <<<"$out" | grep -v -E '/$|^$' | sed 's/^/would send: /' || true
  n=$(sed -nE 's/^Number of regular files transferred: ([0-9,]+).*/\1/p' <<<"$out" | tr -d ,)
  b=$(sed -nE 's/^Total transferred file size: ([0-9,]+).*/\1/p' <<<"$out" | tr -d ,)
  echo "[sync_remote] dry run $HOST:$DIR: would send ${n:-0} of $NF files, ${b:-0} bytes"
  exit 0
fi
out=$("${RSYNC[@]}" "${RS[@]}" "$STAGE/" "$HOST:$DIR/") || { echo "$out" >&2; die "rsync to $HOST:$DIR failed" 4; }
n=$(sed -nE 's/^Number of regular files transferred: ([0-9,]+).*/\1/p' <<<"$out" | tr -d ,)
b=$(sed -nE 's/^Total transferred file size: ([0-9,]+).*/\1/p' <<<"$out" | tr -d ,)
echo "[sync_remote] synced $HOST:$DIR: sent ${n:-0} of $NF files, ${b:-0} bytes"

# Scenes the bench needs (after the build step, which links $DIR/scenes): copied only when missing or their md5 differs. A local
# scene is taken from this worktree, else the main checkout (not in git).
sync_scenes() {
  local sc f src lm rm_
  for sc in ${SYNC_SCENES-bicycle.ply}; do
    src=; for f in "$TOP/scenes/$sc" "$HOME/dev/gstt2/scenes/$sc"; do [ -s "$f" ] && { src=$f; break; }; done
    [ -n "$src" ] || die "no local scene $sc (scenes/ in $TOP or ~/dev/gstt2)" 4
    src=$(real "$(dirname "$src")")/$sc; in_roots "$src" || die "refusing scene $src: outside the project repos"
    lm=$(md5 -q "$src" 2>/dev/null || md5sum "$src" | awk '{print $1}')
    rm_=$(ssh "${SSH_OPTS[@]}" "$HOST" "md5sum '$DIR/scenes/$sc' 2>/dev/null | cut -d' ' -f1" || true)
    if [ "$rm_" = "$lm" ]; then echo "[sync_remote] scene $sc ok ($lm)"; continue; fi
    echo "[sync_remote] scene $sc ${rm_:+md5 $rm_ != }${rm_:-missing}: copying $(wc -c < "$src" | tr -d ' ') bytes"
    "${RSYNC[@]}" -t --mkpath "$src" "$HOST:$DIR/scenes/$sc" || die "scene copy of $sc failed" 4
  done
}
[ "$MODE" = nobuild ] && exit 0   # scenes need the links the build step makes
# REMOTE_TT_METAL_HOME (Mac side) overrides the remote TT_METAL_HOME default below.
ssh "${SSH_OPTS[@]}" "$HOST" "DIR='$DIR' VIEWER_BUILD=$VIEWER_BUILD JOBS='$JOBS' ${REMOTE_TT_METAL_HOME:+TT_METAL_HOME='$REMOTE_TT_METAL_HOME'}\
${GSTT2_BASE:+ GSTT2_BASE='$GSTT2_BASE'}${GSTT2_VENV:+ GSTT2_VENV='$GSTT2_VENV'}${GSTT2_SCENES:+ GSTT2_SCENES='$GSTT2_SCENES'} bash -s" <<'REMOTE'
set -eu
cd "$DIR"
# non-interactive ssh has no TT env; cmake needs TT_METAL_HOME (GSPLAT_WITH_TT)
U=${USER:-$(id -un)}
export TT_METAL_HOME=${TT_METAL_HOME:-/localdev/$U/tt-metal}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME} TT_METAL_ARCH_NAME=${TT_METAL_ARCH_NAME:-blackhole}
BASE=${GSTT2_BASE:-/localdev/$U/gstt2}
VENV=${GSTT2_VENV:-$BASE/.venv} SCENES=${GSTT2_SCENES:-$BASE/scenes}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
[ -L .venv ] && [ ! -e .venv ] && rm -f .venv   # tracked .venv symlink breaks remote CMake
if [ "$(cd "$BASE" 2>/dev/null && pwd -P)" != "$(pwd -P)" ]; then
  for d in .venv scenes; do
    t=$VENV; [ "$d" = scenes ] && t=$SCENES
    # A set override also replaces a link to elsewhere (e.g. a deploy dir once linked into gstt2).
    if [ ! -e "$d" ] || { [ -n "${GSTT2_BASE:-}${GSTT2_VENV:-}${GSTT2_SCENES:-}" ] && [ -L "$d" ] && [ "$(readlink "$d")" != "$t" ]; }; then
      ln -sfn "$t" "$d"; echo "[sync_remote] linked $d -> $t"
    fi
  done
fi
so_md5() { md5sum render/render_clean*.so 2>/dev/null | awk '{print $1}' | sort | tr '\n' ' '; }
src_md5() { find render -path render/build-tt -prune -o -path '*/kernels' -prune -o \
  -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' -o -name CMakeLists.txt \) \
  -print | LC_ALL=C sort | xargs md5sum | md5sum | awk '{print $1}'; }
SRC_NOW=$(src_md5); SRC_OLD=$(cat .host_src.md5 2>/dev/null || true); SO_OLD=$(so_md5)
mkdir -p tmp
NICE=()
if [ "$VIEWER_BUILD" = 1 ]; then   # viewer box: low priority, half the cores
  NICE=(nice -n 19 ionice -c3); JOBS=$(( $(nproc)/2 )); [ "$JOBS" -ge 1 ] || JOBS=1
fi
(cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg.log 2>&1 &&
 ${NICE[@]+"${NICE[@]}"} cmake --build render/build-tt -j "$JOBS" > tmp/build.log 2>&1) || { tail -n 30 tmp/cfg.log tmp/build.log 2>/dev/null; exit 1; }
if ! ls backends/cpu_cpp/_gsplat_cpu*.so >/dev/null 2>&1; then
  if (cmake -G Ninja -S src -B build-avx2-tt -DCMAKE_BUILD_TYPE=Release > tmp/cpu_cfg.log 2>&1 &&
      ${NICE[@]+"${NICE[@]}"} cmake --build build-avx2-tt --target _gsplat_cpu -j "$JOBS" > tmp/cpu_build.log 2>&1) &&
     ls backends/cpu_cpp/_gsplat_cpu*.so >/dev/null 2>&1; then
    echo "[sync_remote] built _gsplat_cpu"
  elif cp "$BASE"/backends/cpu_cpp/_gsplat_cpu*.so backends/cpu_cpp/ 2>/dev/null; then
    echo "[sync_remote] _gsplat_cpu build failed (tmp/cpu_*.log); copied from $BASE"
  else
    echo "[sync_remote] no _gsplat_cpu .so (build failed, none in $BASE)" >&2; exit 1
  fi
fi
SO_NEW=$(so_md5); echo "$SRC_NOW" > .host_src.md5
echo "[sync_remote] sha=$(cat SHA) so_md5=$SO_NEW"
if [ -n "$SRC_OLD" ] && [ "$SRC_OLD" != "$SRC_NOW" ] && [ "$SO_OLD" = "$SO_NEW" ]; then
  echo "[sync_remote] WARNING: host sources changed but render_clean .so md5 did not -- stale build? rm render/render_clean*.so and rebuild" >&2
  exit 3
fi
REMOTE
sync_scenes
