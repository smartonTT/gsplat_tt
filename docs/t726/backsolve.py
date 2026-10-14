import json, sys, numpy as np
from pathlib import Path
from gsplat import t2_dev as td, t2_param_backward as t2
f32=np.float32
ind, outd = Path(sys.argv[1]), Path(sys.argv[2])
meta=json.loads((ind/"meta.json").read_text()); N, ng = meta["n"], meta["n_groups"]
r=np.load(ind/"ref_in.npz"); p={k:r[f"p_{k}"] for k in t2.PARAM_NAMES}
H,W=int(r["H"]),int(r["W"])
gr=td.from_cols(td.from_pages(np.fromfile(outd/"grads.f32",f32).reshape(ng,1024),N,td.STREAMS))
def flat(g): return np.concatenate([np.asarray(g[k],np.float64).reshape(N,-1) for k in t2.PARAM_NAMES],1)
gref=t2.segment_reduce(r["slots"],r["P"],r["counts"]).astype(np.float64)
# exact fp64 segsum
sl=r["slots"].astype(np.float64); P=r["P"]; c=r["counts"]
ex=np.stack([sl[P[i]:P[i]+c[i]].sum(0) if c[i] else np.zeros(sl.shape[1]) for i in range(N)])
print("ref fp32 vs fp64 segsum maxrel", np.abs(gref-ex).max()/np.abs(ex).max())
cols=[]
for j in range(9):
    e=np.zeros((N,9),f32); e[:,j]=1
    cols.append(flat(t2.projection_backward_sh0(p,e,r["w2c"],r["K"],H,W)[0]))
J=np.stack(cols,2)  # N x 14ish x 9
gd=flat(gr); gs=np.zeros((N,9))
for i in range(N): gs[i]=np.linalg.lstsq(J[i],gd[i],rcond=None)[0]
res=np.abs(np.einsum('nij,nj->ni',J,gs)-gd).max()
print("lstsq residual",res, "grad scale", np.abs(gd).max())
d=gs-ex; rel=np.abs(d)/np.maximum(np.abs(ex),1e-12)
np.set_printoptions(precision=3,linewidth=200)
print("per-col max rel err",rel.max(0))
print("per-col max abs err / col scale",np.abs(d).max(0)/np.abs(ex).max(0))
# bitwise: compare fp32 bits
a=gs.astype(f32).view(np.uint32); b=ex.astype(f32).view(np.uint32)
x=a^b; print("xor bit histogram (low 23):",[int(((x>>k)&1).sum()) for k in range(23)])
for name,mod in (("bf16 seg",t2.bf16_round(ex.astype(f32))),):
    print(name, np.abs(mod-ex).max(0)/np.abs(ex).max(0))
# worst gaussians
w=np.argsort(-rel.max(1))[:5]
for i in w: print(i,c[i],ex[i],gs[i])
print("---- hypotheses")
sa=np.stack([np.abs(sl[P[i]:P[i]+c[i]]).sum(0) if c[i] else np.zeros(sl.shape[1]) for i in range(N)])
m=sa>0
print("obs |d|/sum|slot| max per col", np.where(m,np.abs(d)/np.where(m,sa,1),0).max(0))
def trunc(x,bits):  # keep `bits` mantissa bits, truncation
    u=x.astype(f32).view(np.uint32); return (u & np.uint32((0xFFFFFFFF<<(23-bits))&0xFFFFFFFF)).view(f32)
def rne(x,bits):
    u=x.astype(f32).view(np.uint32).astype(np.uint64); s=23-bits
    u=(u+(1<<(s-1))-1+((u>>s)&1))>>s<<s; return u.astype(np.uint32).view(f32)
s32=r["slots"].astype(f32)
hi=t2.bf16_round(s32); mid=t2.bf16_round(s32-hi); lo=t2.bf16_round(s32-hi-mid)
models={"hi":hi,"hi+mid":hi+mid,"tf32rne":rne(s32,10),"tf32trunc":trunc(s32,10),"fp16ish 10b":rne(s32,10),
        "hi+mid(+lo trunc tf32)":hi+mid+trunc(lo,10)}
for nm,sm in models.items():
    smd=sm.astype(np.float64)
    e=np.stack([smd[P[i]:P[i]+c[i]].sum(0) if c[i] else np.zeros(9) for i in range(N)])
    dm=e-ex
    cor=np.corrcoef(dm.ravel(),d.ravel())[0,1] if np.abs(dm).max()>0 else 0
    print(f"{nm:24s} maxabs/colscale {np.abs(dm).max()/np.abs(ex).max():.2e} corr {cor:.3f} resid {np.abs(dm-d).max()/np.abs(d).max():.3f}")
print("---- bits")
q=np.where(m,np.abs(d)/np.where(m,sa,1),0); print("obs |d|/sum|slot| max per col", ["%.2e"%v for v in q.max(0)])
idx=np.argsort(-np.abs(d).ravel())[:15]
for t in idx:
    i,j=divmod(t,9); a_=np.float32(ex[i,j]).view(np.uint32); b_=np.float32(gs[i,j]).view(np.uint32)
    print(i,j,c[i],f"{a_:08x} {b_:08x} xor {a_^b_:08x} d/ulp={(gs[i,j]-ex[i,j])/np.spacing(np.float32(ex[i,j])):.1f}")
print("---- cases")
for (i,j) in ((27,8),(36,6),(9,1),(11,2)):
    print("g",i,"col",j,"P",P[i],"c",c[i], "page", P[i]//td.SLOTS_PER_PAGE, (P[i]+c[i]-1)//td.SLOTS_PER_PAGE)
    for k in range(c[i]):
        v=s32[P[i]+k,j]; print("  slot %08x hi %08x mid %08x lo %08x"%(np.float32(v).view(np.uint32),hi[P[i]+k,j].view(np.uint32),mid[P[i]+k,j].view(np.uint32),lo[P[i]+k,j].view(np.uint32)), v)
    print("  exact %08x dev %08x"%(np.float32(ex[i,j]).view(np.uint32),np.float32(gs[i,j]).view(np.uint32)))
print("SLOTS_PER_PAGE",td.SLOTS_PER_PAGE)
print("---- matmul rounding models")
def rnd(x,n,mode):
    x=np.asarray(x,np.float64); e=np.floor(np.log2(np.where(x==0,1,np.abs(x)))); q=2.0**(e-n)
    if mode=="rne": y=np.round(x/q)*q
    elif mode=="trunc": y=np.trunc(x/q)*q
    else: y=np.floor(x/q)*q
    return np.where(x==0,0,y)
def seg(a): return np.stack([a[P[i]:P[i]+c[i]].sum(0) if c[i] else np.zeros(9) for i in range(N)])
pieces=[seg(z.astype(np.float64)) for z in (hi,mid,lo)]
best=[]
for n in range(8,24):
    for mode in ("rne","trunc","floor"):
        for order in ("hml","lmh"):
            ps=pieces if order=="hml" else pieces[::-1]
            acc=np.zeros_like(ex)
            for pc in ps: acc=(acc+rnd(pc,n,mode)).astype(f32).astype(np.float64)  # fp32 dst, per-matmul result n bits
            dm=acc-ex; best.append((np.abs(dm-d).max()/np.abs(d).max(),n,mode,order,np.corrcoef(dm.ravel(),d.ravel())[0,1]))
            # model 2: dst itself n bits
            acc=np.zeros_like(ex)
            for pc in ps: acc=rnd(acc+pc,n,mode)
            dm=acc-ex; best.append((np.abs(dm-d).max()/np.abs(d).max(),n,mode,order+"-dstN",np.corrcoef(dm.ravel(),d.ravel())[0,1]))
for b in sorted(best)[:8]: print(b)
print("---- by count")
ulp=np.spacing(np.abs(ex).astype(f32)).astype(np.float64)
for cc in sorted(set(c.tolist())):
    s=c==cc
    if s.any(): print("count",cc,"n",s.sum(),"max |d|/ulp",np.abs(d[s]/np.where(ulp[s]>0,ulp[s],1)).max().round(1), "max|d|", np.abs(d[s]).max())
print("---- count1")
s1=np.where(c==1)[0]
rows=[]
for i in s1:
    for j in range(9):
        if ulp[i,j]>0: rows.append((abs(d[i,j]/ulp[i,j]),i,j))
for _,i,j in sorted(rows,reverse=True)[:10]:
    k=P[i]; f=lambda x: "%08x"%np.float32(x).view(np.uint32)
    print(i,j,"slot",f(s32[k,j]),"hi",f(hi[k,j]),"mid",f(mid[k,j]),"lo",f(lo[k,j]),"dev",f(gs[i,j]),"hi+mid",f(np.float32(hi[k,j])+np.float32(mid[k,j])), "d/ulp %.1f"%(d[i,j]/ulp[i,j]))
print("---- per-term alignment models")
def ex2(x): return np.floor(np.log2(np.where(x==0,1e-300,np.abs(x))))
res=[]
big=np.abs(d)>16*ulp
for n in range(9,20):
  for mode in ("floor","trunc","rne"):
    for dstal in (False,True):
      acc=np.zeros_like(ex)
      for z in (hi,mid,lo):
        z=z.astype(np.float64); out=np.zeros_like(ex)
        for i in range(N):
            if not c[i]: continue
            t=z[P[i]:P[i]+c[i]]                       # terms (c,9)
            em=ex2(t).max(0); q=2.0**(em-n)
            tt={"floor":np.floor,"trunc":np.trunc,"rne":np.round}[mode](t/q)*q
            out[i]=np.where((t!=0).any(0),tt.sum(0),0)
        if dstal:
            em=np.maximum(ex2(acc),ex2(out)); q=2.0**(em-n)
            f={"floor":np.floor,"trunc":np.trunc,"rne":np.round}[mode]
            acc=(f(acc/q)*q+f(out/q)*q)
        else: acc=(acc+out)
        acc=acc.astype(f32).astype(np.float64)
      dm=acc-ex
      res.append((np.abs(dm-d)[big].max()/np.abs(d).max(), (np.abs(dm-d)<=16*ulp+1e-12)[big].mean(), n,mode,dstal))
for r_ in sorted(res)[:8]: print(r_)
print("n big",big.sum())
