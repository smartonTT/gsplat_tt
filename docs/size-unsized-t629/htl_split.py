# t629: per-zone median/p90 and median start offset (us from view_total begin) from HTL captures.
# Usage: python3 htl_split.py out/htl-r2-htl.txt.gz
import gzip,sys,statistics as st,collections as C
for f in sys.argv[1:]:
    open_={}; d=C.defaultdict(list); vstart=None; off=C.defaultdict(list)
    for line in gzip.open(f,'rt'):
        p=line.split()
        if len(p)<3: continue
        t,k,z=int(p[0]),p[1],p[2]
        if k=='B':
            open_[z]=t
            if z=='view_total': vstart=t
        elif k=='E' and z in open_:
            d[z].append((t-open_[z])/1e3)
            if vstart: off[z].append((open_[z]-vstart)/1e3)
            del open_[z]
    print('==',f)
    for z in sorted(d,key=lambda z: st.median(off[z]) if off[z] else 0):
        if z in('span','d2h'): continue
        v=sorted(d[z]); print(f"{z:26s} n={len(v):5d} med={st.median(v):8.1f} p90={v[int(len(v)*.9)]:8.1f} start@{st.median(off[z]) if off[z] else 0:8.1f} us")
