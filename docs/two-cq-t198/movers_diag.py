# t198: per-mover sort_ol_emit start/duration/end per arm from Tracy dev30.csv files.
#   python3 movers_diag.py name=path/dev30.csv ...   (cycles at 1350 MHz -> us)
import collections, statistics as st, sys
F = 1350.0
def load(p):
    z, op = collections.defaultdict(list), {}
    for line in open(p, errors="replace"):
        if "sort_ol_emit" not in line: continue
        r = line.split(",")
        k = (int(r[1]), int(r[2]), r[3].strip()); t = int(r[5])
        if r[11].strip() == "ZONE_START": op[k] = t
        elif k in op: z[k].append((op.pop(k), t))
    return z
arms = [a.split("=", 1) for a in sys.argv[1:]]
stats = {}
for name, p in arms:
    z = load(p); nv = min(len(v) for v in z.values())
    s = {k: [[], [], []] for k in z}; last = collections.Counter(); mk = []
    for v in range(1, nv):  # view 0 = warm-up
        s0 = min(z[k][v][0] for k in z); e1 = max(z[k][v][1] for k in z); mk.append((e1 - s0) / F)
        last[max(z, key=lambda k: z[k][v][1])] += 1
        for k in z:
            a, b = z[k][v]; s[k][0].append((a - s0) / F); s[k][1].append((b - a) / F); s[k][2].append((b - s0) / F)
    m = {k: tuple(st.mean(x) for x in s[k]) for k in s}
    stats[name] = (m, last, mk)
    durs = sorted(v[1] for v in m.values())
    print(f"{name}: views {nv-1} makespan mean {st.mean(mk):.1f} us; mover dur med {durs[len(durs)//2]:.1f} "
          f"p90 {durs[int(.9*len(durs))]:.1f} max {durs[-1]:.1f}; start off max {max(v[0] for v in m.values()):.1f}")
ref = arms[0][0]; m0 = stats[ref][0]
top = sorted(m0, key=lambda k: -max(stats[n][0][k][2] for n, _ in arms))[:16]
print("mover (x,y,risc)      " + "  ".join(f"{n:>24s}" for n, _ in arms) + "   [start/dur/end us, #views last]")
for k in top:
    print(f"{str(k):22s}" + "  ".join(f"{stats[n][0][k][0]:6.1f}/{stats[n][0][k][1]:6.1f}/{stats[n][0][k][2]:6.1f} {stats[n][1][k]:2d}" for n, _ in arms))
