"""Summarize the sort_bucket_emit accumulating sub-zones (task #27).

Capture with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SUM=1 (e.g.
opt/profiler/capture_tracy.sh <dir> 0:10), then:
    python3 opt/profiler/emit_sumzones.py <profile_log_device.csv>
Each RISC writes one ZONE_TOTAL row per launch for emit_pack_invariants and
emit_pack_rec; launch 0 (warmup) is dropped. Cycles are 1.35 GHz ticks.
"""
import csv, sys, collections, statistics as st
f = sys.argv[1]
rows = list(csv.reader(open(f)))
hdr = [h.strip() for h in rows[1]]
ix = {h: i for i, h in enumerate(hdr)}
tot = collections.defaultdict(list)     # (core,risc,zone) -> [cycles per launch]
emit = collections.defaultdict(list)    # (core,risc) -> [emit zone cycles per launch]
opn = {}
for r in rows[2:]:
    z = r[ix['zone name']].strip(); t = r[ix['type']].strip()
    key = (r[1].strip(), r[2].strip(), r[ix['RISC processor type']].strip())
    if t == 'ZONE_TOTAL':
        tot[key + (z,)].append(int(r[ix['data']]))
    elif z == 'sort_bucket_emit':
        ts = int(r[ix['time[cycles since reset]']])
        if t == 'ZONE_START': opn[key] = ts
        elif t == 'ZONE_END' and key in opn: emit[key].append(ts - opn.pop(key))
def show(name):
    seqs = [v for k, v in tot.items() if k[3] == name]
    print(name, 'series(core0):', seqs[0][:11])
    # drop launch 0 (warmup)
    per = [x for v in seqs for x in v[1:]]
    print('  mean cyc/launch/risc', round(st.mean(per)), 'ms', round(st.mean(per)/1350e3, 3), 'max ms', round(max(per)/1350e3, 3))
    return per
a = show('emit_pack_invariants'); b = show('emit_pack_rec')
e = [x for v in emit.values() for x in v[1:]]
print('emit zone per launch/risc: n', len(e), 'mean ms', round(st.mean(e)/1350e3, 3), 'max ms', round(max(e)/1350e3, 3))
print('share of emit (mean): inv %.1f%%  rec %.1f%%' % (100*st.mean(a)/st.mean(e), 100*st.mean(b)/st.mean(e)))
