import numpy as np, gzip, collections
# Input: NCRISC rows of the t297 raw Tracy CSV (worktrees/t297/tmp/t297/t297-ns-dev30.csv), e.g.
#   grep ',NCRISC,' t297-ns-dev30.csv | grep -E 'rd_l1_bulk|NCRISC-KERNEL' > /tmp/t303_nc.csv
# Output: per-tile blend spacing bins (us) and lane load-time fit used by model.py.
# zones per (view k, core): list of (start, dur) us
st={}; per=collections.defaultdict(lambda: collections.defaultdict(list)); kidx=collections.Counter()
for ln in open('/tmp/t303_nc.csv'):
    f=ln.split(','); core=(f[1],f[2]); t=int(f[5]); z=f[10]; ty=f[11]
    if z=='NCRISC-KERNEL':
        if ty=='ZONE_START': kidx[core]+=1
        continue
    if ty=='ZONE_START': st[core]=t
    else: per[kidx[core]][core].append((st[core]/1350., (t-st[core])/1350.))
ks=sorted(per)
lines=gzip.open("docs/mat-split-sort-model/out/dump.txt.gz","rt").read().splitlines()[1:]
N=[];T=[];FIRST=[]
for v,k in enumerate(ks):
    cnt=[int(p.split(":")[1]) for p in lines[v].split()[3:]]
    items=[]
    for c in sorted(cnt,reverse=True):
        if c==0: continue
        nsc=(c+8191)//8192
        items+=[(c, nsc)]*nsc   # one zone per subchunk; tag with tile count
    allz=sorted((s,d,core) for core,zs in per[k].items() for s,d in zs)
    if len(allz)!=len(items): print('mismatch',v,len(allz),len(items)); continue
    lab={}
    for (s,d,core),it in zip(allz,items): lab[(core,s)]=it
    for core,zs in per[k].items():
        zs=sorted(zs)
        for j in range(len(zs)-2):
            n,nsc=lab[(core,zs[j][0])]
            if nsc>1: continue
            T.append(zs[j+2][0]-zs[j+1][0]); N.append(n)
N=np.array(N,float);T=np.array(T)
print('pairs',len(N))
for lo,hi in [(0,128),(128,256),(256,512),(512,1024),(1024,2048),(2048,3072),(3072,4096),(4096,6144),(6144,8193)]:
    s=(N>=lo)&(N<hi)
    if s.sum()<5: print(lo,hi,"n",s.sum()); continue
    print(f'{lo:5d}-{hi:5d} n={s.sum():5d} med_us={np.median(T[s]):7.1f} p25={np.percentile(T[s],25):7.1f} p75={np.percentile(T[s],75):7.1f}')
s=(T>0)&(T<3000)
c=np.polyfit(N[s],T[s],1); print('lsq today: blend_us = %.1f + %.4f n'%(c[1],c[0]))
xm=[];ym=[]
for b in range(0,8192,256):
    q=(N>=b)&(N<b+256)
    if q.sum()>30: xm.append(np.median(N[q])); ym.append(np.median(T[q]))
c2=np.polyfit(xm,ym,1); print('median-bin today: %.1f + %.4f n'%(c2[1],c2[0]))
print('t147 scaled (x0.7?) for compare: 87+0.1841n')
