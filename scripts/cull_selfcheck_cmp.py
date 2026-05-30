import math, re, sys

floor = 6.10352e-05

coeffs = {}   # (t,local) -> dict
m2dev = {}    # (t,local) -> {m: m2}

cre = re.compile(r"CULLCOEF t=(\d+) local=(\d+) a=(\S+) b=(\S+) c=(\S+) mx=(\S+) my=(\S+) op=(\S+) tx=(\d+) ty=(\d+)")
mre = re.compile(r"M2DMP t=(\d+) local=(\d+) m=(\d+) m2=(\S+)")

for line in open("/tmp/self_extract.txt"):
    m = cre.search(line)
    if m:
        t, lo = int(m.group(1)), int(m.group(2))
        coeffs[(t, lo)] = dict(a=float(m.group(3)), b=float(m.group(4)), c=float(m.group(5)),
                               mx=float(m.group(6)), my=float(m.group(7)), op=float(m.group(8)),
                               tx=float(m.group(9)), ty=float(m.group(10)))
        continue
    m = mre.search(line)
    if m:
        t, lo, mm = int(m.group(1)), int(m.group(2)), int(m.group(3))
        m2dev.setdefault((t, lo), {})[mm] = float(m.group(4))

def scalar_m2(d):
    a, b, c = d['a'], d['b'], d['c']
    mean_x, mean_y = d['mx'], d['my']
    tx_tile, ty_tile = d['tx'], d['ty']
    det = a*c - b*b
    if det < 1e-6: det = 1e-6
    ci_a = c/det; ci_b = -b/det; ci_c = a/det
    ci_a_safe = ci_a if ci_a > 1e-12 else 1e-12
    ci_c_safe = ci_c if ci_c > 1e-12 else 1e-12
    tcb = 2*ci_b; kInf = float('inf')
    out = {}
    for my in range(8):
        voy = ty_tile + my*4; vlo = voy - mean_y; vhi = vlo + 4
        yin = (vlo <= 0 <= vhi); vfix = vlo if vlo > 0 else vhi
        rus = 0.0; rt3 = 0.0
        if not yin:
            rus = -ci_b*vfix/ci_a_safe; rt3 = ci_c*vfix*vfix
        for mx in range(4):
            m = (my << 2) | mx
            ox = tx_tile + mx*8; ulo = ox - mean_x; uhi = ulo + 8
            xin = (ulo <= 0 <= uhi); ufix = ulo if ulo > 0 else uhi
            if xin and yin:
                m2 = 0.0
            else:
                m2v = kInf
                if not xin:
                    vs = -ci_b*ufix/ci_c_safe; vs = max(vlo, min(vhi, vs))
                    m2v = ci_a*ufix*ufix + tcb*ufix*vs + ci_c*vs*vs
                m2h = kInf
                if not yin:
                    us = rus; us = max(ulo, min(uhi, us))
                    m2h = ci_a*us*us + tcb*us*vfix + rt3
                m2 = min(m2v, m2h)
            out[m] = m2
    return out, -2.0*math.log(floor/d['op']) if d['op'] > floor else -1.0

keys = sorted(set(coeffs) & set(m2dev))
print(f"{len(keys)} (tile,local) pairs with both coeffs and device m2\n")
nbad = 0
for k in keys:
    d = coeffs[k]
    sm, thresh = scalar_m2(d)
    dev = m2dev[k]
    # max abs diff over microblocks present in both
    maxd = 0.0; worstm = -1
    refmask = 0; devmask = 0
    for m in range(32):
        s = sm.get(m, float('inf'))
        dv = dev.get(m, None)
        if dv is None: continue
        # device m2 for microblocks outside bbox isn't computed by scalar (inf);
        # compare only finite scalar values for the diff metric.
        if s < 1e29 and dv < 1e29:
            dd = abs(dv - s)
            if dd > maxd: maxd = dd; worstm = m
        if s <= thresh: refmask |= 1 << m
        if dv <= thresh: devmask |= 1 << m
    flag = "" if refmask == devmask else "  <<< MASK MISMATCH"
    if maxd > 0.1 or refmask != devmask:
        nbad += 1
        print(f"t={k[0]} local={k[1]} a={d['a']:.3g} c={d['c']:.3g} op={d['op']:.4g} "
              f"thresh={thresh:.3f} maxdiff={maxd:.4f}@m{worstm} "
              f"ref=0x{refmask:08X} dev=0x{devmask:08X}{flag}")
print(f"\n{nbad} divergent gaussians of {len(keys)}")
