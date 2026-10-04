import sys, collections
for tag in ("t170-on", "t170-nf"):
    f = f"/localdev/smarton/gstt2-t170/opt/profiler/{tag}/dev30.csv"
    starts = collections.defaultdict(list); durs = collections.defaultdict(list)
    open_ = {}
    for line in open(f):
        if "sort_ol_prefix" not in line: continue
        c = line.split(",")
        key = (c[1], c[2])
        t = int(c[5])
        if c[11] == "ZONE_START": open_[key] = t
        elif key in open_: durs[key].append((open_.pop(key), t - open_.get(key, 0) if False else t))
    # per view index i: durations per core
    nv = min(len(v) for v in durs.values())
    per_view_mk = []; per_core = collections.defaultdict(list)
    for i in range(1, nv):
        s = [durs[k][i][0] for k in durs]; e = [durs[k][i][1] for k in durs]
        per_view_mk.append((max(e) - min(s)) / 1350.0)
        for k in durs: per_core[k].append((durs[k][i][1] - durs[k][i][0]) / 1350.0)
    avg = sorted((sum(v)/len(v), k) for k, v in per_core.items())
    print(tag, "cores", len(durs), "views", nv, "makespan us %.1f" % (sum(per_view_mk)/len(per_view_mk)))
    print("  per-core avg dur us: min %.1f med %.1f max %.1f" % (avg[0][0], avg[len(avg)//2][0], avg[-1][0]))
    print("  slowest:", [("%s,%s" % k, round(d, 1)) for d, k in avg[-6:]])
    print("  start skew us (median view):", round(sorted(((max(durs[k][i][0] for k in durs) - min(durs[k][i][0] for k in durs))/1350.0) for i in range(1, nv))[nv//2], 1))
