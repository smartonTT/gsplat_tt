#!/usr/bin/env bash
# Blend PHASING ablation (Track 2): measure whether iter-112 "Track 1" phasing
# (phase the per-gaussian power->exp->alpha->RGBT chain across all K covered
# microblocks, exposing ILP) actually helps the SFPU vs the pre-Track-1 UNPHASED
# structure (full per-microblock chain back-to-back, stalls on op latency).
#
# Both phased and unphased run the SAME instructions, the SAME DR_SCR spill, and
# the SAME per-microblock math+inputs -> BIT-IDENTICAL pixels (microblocks are
# independent accumulators; only the SFPU ISSUE ORDER differs). So this isolates
# Track 1's ILP effect cleanly, holding the DEST-spill overhead constant.
#
# Method mirrors the original prof_ablation.sh: edit kernel -> capture device
# zones (30-view render_clean Tracy) -> save CSV -> restore from pristine backup.
# Kernels are JIT-compiled keyed on source hash, so an edit auto-recompiles on
# the next capture (no cmake build). NEVER `git checkout` the device tree — it is
# an uncommitted rsync of Mac HEAD; restore from the .profbak made here.
#
# RUN UNDER devrun.sh (holds the device flock, sets TTW_DEVRUN=1). Pass
# --no-verify because we DELIBERATELY edit kernels mid-run (the build-ID check
# would otherwise abort on the kernel-source hash change):
#   devrun.sh --no-verify --timeout 2700 --tag ablation-phasing -- \
#     bash /localdev/smarton/prof_ablation_phasing.sh
set -uo pipefail
REPO=/localdev/smarton/gstt2
cd "$REPO" || exit 1

COMPUTE=render/kernels/compute/alpha_blend_compute_mb.cpp
READER=render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp
OUT=opt/profiler/ablation_phasing
HERO=tmp/rc-tracy/hero_clean.png
mkdir -p "$OUT"

edit() { # file old new
  python3 - "$1" "$2" "$3" <<'PY'
import sys
f,old,new=sys.argv[1],sys.argv[2],sys.argv[3]
s=open(f).read()
n=s.count(old)
assert n==1, f"FATAL expected 1 occurrence in {f}, found {n}"
open(f,'w').write(s.replace(old,new,1))
print(f"[edit] {f}: OK")
PY
}

cp -f "$COMPUTE" "$COMPUTE.profbak" || exit 1
cp -f "$READER"  "$READER.profbak"  || exit 1
restore(){ cp -f "$COMPUTE.profbak" "$COMPUTE"; cp -f "$READER.profbak" "$READER"; }

capture(){ # tag
  local tag="$1"
  echo "############### CAPTURE: $tag ###############"
  rm -f "$HERO"
  bash render/profiler/capture_tracy_clean.sh 2>&1 \
    | grep -E "capture_clean|render-clean-inner|run.py rc|device profiler CSV|FAIL|FATAL|SUMMARY"
  local src="opt/profiler/wrap_out_render-clean/.logs/profile_log_device.csv"
  if [[ -s "$src" ]]; then
    cp -f "$src" "$OUT/$tag.csv"
    echo "[saved] $OUT/$tag.csv rows=$(($(wc -l < "$OUT/$tag.csv")-1))"
  else
    echo "[ERR] no CSV for $tag"
  fi
  if [[ -f "$HERO" ]]; then
    echo "[hero_md5] $tag $(md5sum "$HERO" | awk '{print $1}')"
  else
    echo "[hero_md5] $tag MISSING"
  fi
}

restore  # clean slate

# ---- 1) phased = current HEAD kernel (Track 1 in place). NO edit. ----
capture phased
restore

# ---- 2) unphased = pre-Track-1 structure: full per-microblock chain
#         back-to-back (serial SFPU dependency chain, no cross-mb ILP). ----
edit "$COMPUTE" '// Dispatch one gaussian (coeffs in GPRs) to every microblock its mask selects,
// phased so each SFPU op runs across all covered microblocks before the next
// dependent op (ILP; the phasing also fixes the DEST read-after-write hazard).
inline void dispatch_blend_guarded(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t fc, uint32_t op, uint32_t cr, uint32_t cg, uint32_t cbv) {
    (void)fc;
    blend_phase_power<0>(mask, a, b, c, d, e);
    blend_phase_exp<0>(mask);
    blend_phase_alpha<0>(mask, op);
    blend_phase_rgbt<0>(mask, cr, cg, cbv);
}' '// UNPHASED variant (Track-2 ablation): the pre-Track-1 structure. Run the FULL
// power->exp->alpha->RGBT chain back-to-back for ONE microblock before moving to
// the next, so each gaussian'\''s per-microblock SFPU dependency chain
// (power->DR_SCR->exp->DR_SCR->alpha->DR_SCR->rgbt) issues serially with NO
// cross-microblock ILP. Same instructions, same DR_SCR spill, same per-microblock
// math+inputs as the phased path -> BIT-IDENTICAL pixels (independent accumulators;
// only the SFPU issue ORDER differs). Isolates Track 1'\''s ILP effect.
template <uint32_t M>
inline void blend_chain_unphased(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t op, uint32_t cr, uint32_t cg, uint32_t cbv) {
    if constexpr (M < NUM_MB) {
        if (mask & (1u << M)) {
            MATH((blend_power_math<M>(a, b, c, d, e)));
            MATH((blend_exp_math<M>()));
            MATH((blend_alpha_math<M>(op)));
            MATH((blend_rgbt_math<M>(cr, cg, cbv)));
        }
        blend_chain_unphased<M + 1>(mask, a, b, c, d, e, op, cr, cg, cbv);
    }
}
inline void dispatch_blend_guarded(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t fc, uint32_t op, uint32_t cr, uint32_t cg, uint32_t cbv) {
    (void)fc;
    blend_chain_unphased<0>(mask, a, b, c, d, e, op, cr, cg, cbv);
}' || { echo "[FATAL] unphased edit failed"; restore; }
capture unphased
restore

# ---- 3) noexp_phased = phased kernel with the SFPU exp replaced by identity
#         (weight = power). delta vs phased = exp cost on the phased path. ----
edit "$COMPUTE" 'dst_reg[DR_SCR + IX] = cs::_sfpu_exp_21f_bf16_</*is_fp32_dest_acc_en=*/true>(power);' 'dst_reg[DR_SCR + IX] = power;' \
  || { echo "[FATAL] noexp edit failed"; restore; }
capture noexp_phased
restore

# ---- 4) nomath = neutralize the blend SFPU dispatch entirely -> reader floor. ----
edit "$COMPUTE" 'if (mask != 0u) {' 'if (false) {' \
  || { echo "[FATAL] nomath edit failed"; restore; }
capture nomath
restore

# ---- 5) noread = skip the NoC payload reads (keep CB handshake) -> SFPU floor
#         (stale masks => lower bound, same as the original ablation). ----
edit "$READER" 'while (pp < rec_pages) {' 'while (false) {' \
  || { echo "[FATAL] noread edit failed"; restore; }
capture noread
restore

echo "############### ANALYSIS ###############"
python3 - "$OUT" <<'PY'
import csv,sys,os
from collections import defaultdict
FREQ=1350e6
OUT=sys.argv[1]
ZONES=['tile_blend_sfpu','tile_blend_load','rd_l1_bulk']
configs=['phased','unphased','noexp_phased','nomath','noread']
def makespans(path):
    iv=defaultdict(list); st=defaultdict(list)
    with open(path) as f:
        r=csv.reader(f); next(r); next(r)
        for row in r:
            if len(row)<12: continue
            try: t=int(row[5])
            except: continue
            z=row[10].strip(); typ=row[11].strip(); k=(row[1],row[2],row[3].strip(),z)
            if typ=='ZONE_START': st[k].append(t)
            elif typ=='ZONE_END' and st[k]: iv[z].append((st[k].pop(),t))
    GAP=int(1.0e-3*FREQ); res={}
    for z in ZONES:
        if z not in iv: continue
        ivs=sorted(iv[z]); cl=[]; cur=[ivs[0]]
        for s,e in ivs[1:]:
            if s-cur[-1][0]>GAP: cl.append(cur); cur=[(s,e)]
            else: cur.append((s,e))
        cl.append(cur)
        ms=[(max(e for _,e in c)-min(s for s,_ in c))/FREQ*1000 for c in cl]
        body=ms[1:] if len(ms)>1 else ms   # drop the warmup view (first cluster)
        res[z]=sum(body)/len(body)
    return res
data={}
for c in configs:
    p=os.path.join(OUT,f'{c}.csv')
    if os.path.exists(p):
        try: data[c]=makespans(p)
        except Exception as ex: print(f'[warn] {c}: {ex}')
hdr='%-18s'%'zone (ms/view)'+''.join('%14s'%c for c in configs)
print(hdr); print('-'*len(hdr))
for z in ZONES:
    row='%-18s'%z+''.join('%14s'%(f'{data[c][z]:.2f}' if c in data and z in data[c] else '-') for c in configs)
    print(row)
print()
print('INTERPRETATION:')
b=data.get('phased',{}); up=data.get('unphased',{}); ne=data.get('noexp_phased',{})
nm=data.get('nomath',{}); nr=data.get('noread',{})
if 'tile_blend_sfpu' in b:
    base=b['tile_blend_sfpu']
    print(f'  phased   tile_blend_sfpu = {base:.2f} ms/view')
    if 'tile_blend_sfpu' in up:
        u=up['tile_blend_sfpu']
        d=u-base; pct=100.0*d/u if u else 0.0
        print(f'  unphased tile_blend_sfpu = {u:.2f} ms/view  (phased is {d:.2f} ms / {pct:.1f}% faster than unphased)')
    if 'tile_blend_sfpu' in ne:
        print(f'  noexp(phased)            = {ne["tile_blend_sfpu"]:.2f} ms/view  -> exp cost on phased = {base-ne["tile_blend_sfpu"]:.2f} ms')
    if 'tile_blend_sfpu' in nm:
        print(f'  nomath (reader floor)    = {nm["tile_blend_sfpu"]:.2f} ms/view')
    if 'tile_blend_sfpu' in nr:
        print(f'  noread (sfpu floor)      = {nr["tile_blend_sfpu"]:.2f} ms/view')
PY
echo "############### DONE ###############"
restore
rm -f "$COMPUTE.profbak" "$READER.profbak"
echo "[restore] working-tree kernels restored from pristine backup; verify clean (md5 vs Mac HEAD):"
md5sum "$COMPUTE" "$READER"
git diff --stat "$COMPUTE" "$READER"
