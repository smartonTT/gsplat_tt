#!/usr/bin/env python3
"""t303: no-device model of lever A (a blend lane on the TRISCs during the mat phase of the
fused mat+blend program), docs/next-levers-after-205.md section A step 1.

Built on the t273 ready-flag model (docs/matblend-ready-t273/model.py), which predicted the
fused program within ~0.1 ms of the #293 Tracy capture.

Inputs (measured unless named as a fit):
  - per-tile record counts, 30 bicycle views: docs/mat-split-sort-model/out/dump.txt.gz
  - blend TRISC per-tile cost: t147 fit 87.0 + 0.1841 n - 0.1161 max(n-8192,0) us
    (docs/fuse-matblend-t147/out/t147-tc.dprint), scaled so the mean per-core blend span is
    the t297 measured 3.655 ms (blend_end mean 6.168 - mat_end mean 2.513)
  - mat schedule: replay of build_mat_worklist with the t144/t147 mover fits (as t273)
  - per-core mat timeline: docs/profile-postl1-t297/out/tracy-t297-ns-percore.csv
    (mat_end, mov_br, batches per core; t297 Tracy, iter 206)
  - lane load cost on the mover (fit here from the t297 rd_l1_bulk zones, rank-matched to
    the dump's subchunk sizes): payload 0.5 + 0.0008 n us, plus ~2.4 us per tile for the
    claim atomic and the tid / range / meta / dir / ready reads (5th pct of the gap between
    bulk zones). --load_mult scales both (mover DRAM contention during mat).

Model per view:
  1. Mat items replayed per slot (2 per core), core mat end = max of its two slots.
     Calibration "csv": cores ranked by modeled end get the measured per-core mat_end (sorted)
     of the CSV, ready times on that core scale with it; BRISC slack = mat_end - mov_br and the
     cull batch rate come from the same CSV row. Calibration "t273": t273's compression to
     mean 2.513 / max 2.989 (per-view spread), BRISC slack = CSV mean 0.137 for every core.
  2. Lane: while a core is in its mat phase its TRISCs take ready tiles of <= cap records
     (cap = ring/32 B) from the small end of the claim list (ascending count). Per lane tile:
       TRISC busy = blend(n) * (1 + rho) + switch, rho = preempt_us * batch_rate + cull duty
       mover cost = load(n); on NCRISC it extends that core's mat end 1:1 (NCRISC is busy
       ~100% of mat); on BRISC it uses the core's slack first, then extends mat end.
       Writer (BRISC, 3 KB u8 out tile) 1 us per lane tile, against BRISC slack.
     A lane tile is only started before the core's (extended) mat end; the core enters the
     main blend at max(mat end, lane end).
  3. Main blend: unchanged dynamic claim over the remaining tiles in descending count
     order, gated on ready (t273 'flags' sim). Program end = max core end.
  Saving = baseline (no lane, same calibration) - lane end, per view, ms.
L1 is not charged here (see the doc): the result is an upper bound for a given ring size.
"""
import argparse, csv, gzip, heapq, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="docs/mat-split-sort-model/out/dump.txt.gz")
ap.add_argument("--tc", default="docs/fuse-matblend-t147/out/t147-tc.dprint")
ap.add_argument("--percore", default="docs/profile-postl1-t297/out/tracy-t297-ns-percore.csv")
ap.add_argument("--cores", type=int, default=110)
ap.add_argument("--blend_len", type=float, default=3.655)  # t297 blend_end mean - mat_end mean
ap.add_argument("--mat_max", type=float, default=2.989)    # t297 per-view max mat end
ap.add_argument("--mat_mean", type=float, default=2.513)   # t297 mean mat end
ap.add_argument("--lag", type=float, default=0.02)
ap.add_argument("--cal", default="csv", choices=["csv", "t273"])
ap.add_argument("--ring_kb", type=int, default=64)
ap.add_argument("--switch_us", type=float, default=5.0)    # per lane tile
ap.add_argument("--preempt_us", type=float, default=0.0)   # per cull batch during a lane tile
ap.add_argument("--mover", default="ncrisc", choices=["ncrisc", "brisc", "none"])
ap.add_argument("--load_mult", type=float, default=1.0)
ap.add_argument("--writer_us", type=float, default=1.0)
ap.add_argument("--matorder", default="desc", choices=["desc", "smallfirst"])
ap.add_argument("--fit", default="t147", choices=["t147", "today"])  # today = t297 per-bin medians
ap.add_argument("--views", type=int, default=0)  # 0 = all
ap.add_argument("--l1_cost", action="store_true",
                help="carve ring_kb + lane_cb_kb from CB4: lower bucket_fit/ov_cap for the lane run")
ap.add_argument("--lane_cb_kb", type=int, default=25)  # non-aliased MB_COUNTS 8 + OUT 12 + IMG_U8 3 + SCR 2
ap.add_argument("-v", action="store_true")
a = ap.parse_args()

BF, OVCAP, M0CAP = 8192, 16384, 6144

rec, cyc = [], []
for ln in open(a.tc, errors="replace"):
    m = re.search(r"TC\s+(\d+)\s+(\d+)\s+(\d+)\s*$", ln)
    if m:
        rec.append(int(m.group(1))); cyc.append(int(m.group(3)) / 1350.0)
rec, us = np.array(rec, float), np.array(cyc, float)
A = np.vstack([np.ones_like(rec), rec, np.maximum(rec - 8192, 0)]).T
coef, *_ = np.linalg.lstsq(A, us, rcond=None)
# t297 per-tile blend spacing medians (fit_today.py), bin centres -> us
TX = [0, 192, 384, 768, 1536, 2560, 3584, 5120, 7168]
TY = [50.0, 74.2, 99.7, 164.3, 276.8, 429.0, 571.1, 716.2, 871.5]
def blend_us(n):
    if a.fit == "today":
        return float(np.interp(n, TX, TY)) if n <= 7168 else 871.5 + 0.0776 * (n - 7168)
    return coef[0] + coef[1] * n + coef[2] * max(n - 8192, 0)

pc = list(csv.DictReader(open(a.percore)))
pc.sort(key=lambda r: float(r["mat_end"]))
CSV_END = np.array([float(r["mat_end"]) for r in pc])
CSV_SLACK = np.array([max(float(r["mat_end"]) - float(r["mov_br"]), 0.0) for r in pc])
CSV_RATE = np.array([float(r["batches"]) / float(r["mat_end"]) / 1000.0 for r in pc])  # per us
CSV_DUTY = np.array([float(r["t1_busy_mat"]) / float(r["mat_end"]) for r in pc])

def load_us(n):  # lane tile on a mover: claim + 5 small reads + payload
    return a.load_mult * (2.4 + 0.5 + 0.0008 * n)

def mat_items(counts, BF=BF, OVCAP=OVCAP):
    items = []
    for t, c in counts.items():
        if c <= OVCAP:
            items.append((t, c, c > M0CAP, None))
            continue
        nsc = (c + BF - 1) // BF
        for sc in range(nsc):
            l = min(BF, c - sc * BF)
            items.append((t, c + l, True, (c, l)))
    items.sort(key=lambda x: -x[1])
    return items

def item_us(it, mover):
    t, cost, big, ov = it
    if ov is not None:
        n, l = ov
        return 0.1083 * n + 0.1267 * l
    n = cost
    if n > M0CAP:
        return -282 + 0.2226 * n
    return 5.5 + 0.197 * n if mover == 0 else 16.1 + 0.187 * n

def schedule(counts, cap, bf=BF, ovcap=OVCAP):
    C = a.cores; slots = 2 * C
    load = [0] * slots; per = [[] for _ in range(slots)]
    for it in mat_items(counts, bf, ovcap):
        step = 2 if it[2] else 1
        c = 0
        for k in range(step, slots, step):
            if load[k] < load[c]: c = k
        per[c].append(it); load[c] += it[1]
    if a.matorder == "smallfirst":  # same slot totals, small (lane-eligible) items first
        for s in range(slots):
            per[s].sort(key=lambda it: (it[1] > cap, -it[1]) if it[3] is None else (True, -it[1]))
    raw = []; slot_end = np.zeros(slots)
    for s in range(slots):
        tt = 0.0
        for it in per[s]:
            tt += item_us(it, s & 1)
            raw.append((it[0], s // 2, tt))
        slot_end[s] = tt
    core_raw = np.maximum(slot_end[0::2], slot_end[1::2])
    return raw, core_raw

def calibrate(raw, core_raw):
    C = a.cores
    if a.cal == "csv":
        order = np.argsort(core_raw, kind="stable")
        end = np.zeros(C); slack = np.zeros(C); rate = np.zeros(C); duty = np.zeros(C)
        idx = np.linspace(0, len(CSV_END) - 1, C).round().astype(int)
        for rank, c in enumerate(order):
            j = idx[rank]
            end[c], slack[c], rate[c], duty[c] = CSV_END[j], CSV_SLACK[j], CSV_RATE[j], CSV_DUTY[j]
        k = end / np.maximum(core_raw, 1e-9)
        ready = {}
        for t, c, tt in raw:
            ready[t] = max(ready.get(t, 0.0), tt * k[c] + a.lag)  # k is ms per model-us
        return end + a.lag, ready, slack, rate, duty
    sc = a.mat_max * 1000.0 / core_raw.max()
    end = core_raw * sc / 1000.0 + a.lag
    mx = end.max(); k = (mx - a.mat_mean) / (mx - end.mean())
    end = mx - (mx - end) * k
    ready = {}
    for t, c, tt in raw:
        r = tt * sc / 1000.0 + a.lag
        ready[t] = max(ready.get(t, 0.0), mx - max(mx - r, 0.0) * k)
    C_ = len(end)
    return (end, ready, np.full(C_, CSV_SLACK.mean()), np.full(C_, CSV_RATE.mean()),
            np.full(C_, CSV_DUTY.mean()))

def main_blend(tiles, b, ready, free):
    free = list(free); end = 0.0; last_claim = 0.0
    for t, d in zip(tiles, b):
        c = int(np.argmin(free))
        s = max(free[c], ready[t])
        last_claim = max(last_claim, free[c])
        free[c] = s + d; end = max(end, free[c])
    return end, np.array(free), last_claim

def model(counts, cap):
    C = a.cores
    raw, core_raw = schedule(counts, cap)
    mat_end, ready, slack, rate, duty = calibrate(raw, core_raw)
    tiles = sorted(counts, key=lambda t: (-counts[t], -t))
    bmap = {t: blend_us(counts[t]) for t in tiles}
    scale = a.blend_len * C * 1000.0 / sum(bmap.values())
    bmap = {t: v * scale / 1000.0 for t, v in bmap.items()}  # ms
    base_end, base_free, _ = main_blend(tiles, [bmap[t] for t in tiles], ready, mat_end)
    if a.mover == "none" and cap == 0:
        return dict(base=base_end)
    # ---- lane ----
    if a.l1_cost:  # smaller CB4: more tiles take the subchunk path in mat and blend
        cb4 = 512 - a.ring_kb - a.lane_cb_kb
        ovc = cb4 * 1024 // 32; bf2 = min(BF, ovc // 2)
        raw2, core_raw2 = schedule(counts, cap, bf2, ovc)
        kk = (mat_end - a.lag) / np.maximum(core_raw, 1e-9)  # keep the base per-core calibration
        mat_end = core_raw2 * kk + a.lag
        ready = {}
        for t_, c_, tt in raw2:
            ready[t_] = max(ready.get(t_, 0.0), tt * kk[c_] + a.lag)
        sub = lambda n, b_: (n + b_ - 1) // b_ if n > b_ else 1
        bmap = {t: v + 0.087 * scale / 1000.0 * (sub(counts[t], bf2) - sub(counts[t], BF))
                for t, v in bmap.items()}
    elig = sorted([t for t in tiles if 0 < counts[t] <= cap], key=lambda t: (counts[t], t))
    taken = set()
    end = mat_end.copy(); br_used = np.zeros(C); lane_end = np.zeros(C)
    lane_n = np.zeros(C); lane_work = np.zeros(C); nc_ext = np.zeros(C)
    lane_first = np.full(C, 1e18)
    ev = [(0.0, c) for c in range(C)]; heapq.heapify(ev)
    while ev:
        t, c = heapq.heappop(ev)
        if t >= end[c]:
            continue
        pick = None; nxt = None
        for u in elig:
            if u in taken: continue
            if ready[u] <= t:
                pick = u; break
            nxt = ready[u] if nxt is None else min(nxt, ready[u])
        if pick is None:
            if nxt is not None and nxt < end[c]:
                heapq.heappush(ev, (nxt, c))
            continue
        taken.add(pick)
        n = counts[pick]
        ld = load_us(n) / 1000.0; wr = a.writer_us / 1000.0
        if a.mover == "ncrisc":
            nc_ext[c] += ld
            br_used[c] += wr
        else:
            br_used[c] += ld + wr
        # NCRISC has ~0 slack in mat (t297), BRISC has slack[c]; the core's mat ends when
        # the later mover ends
        end[c] = max(mat_end[c] + nc_ext[c], mat_end[c] - slack[c] + br_used[c])
        rho = a.preempt_us * rate[c] + duty[c]
        d = bmap[pick] * (1.0 + rho) + a.switch_us / 1000.0
        start = t + ld  # the tile's data must be in L1 first (no prefetch)
        lane_first[c] = min(lane_first[c], start)
        lane_end[c] = start + d
        lane_n[c] += 1; lane_work[c] += bmap[pick]
        heapq.heappush(ev, (lane_end[c], c))
    free0 = np.maximum(end, lane_end)
    rest = [t for t in tiles if t not in taken]
    lane_endt, free, last_claim = main_blend(rest, [bmap[t] for t in rest], ready, free0)
    # sanity: the main claim must not reach a lane tile before the lane took it (lane is done
    # by max(free0)); last_claim is when the main blend claimed its last tile
    return dict(base=base_end, lane=lane_endt, saving=base_end - lane_endt,
                lane_tiles=lane_n.sum(), lane_work_ms_per_core=lane_work.mean(),
                mat_ext_mean=(end - mat_end).mean(), mat_ext_max=(end - mat_end).max(),
                entry_delay_mean=(free0 - mat_end).mean(),
                lane_past_mat_max=np.maximum(lane_end - end, 0).max(),
                lane_first_start_mean=np.mean(lane_first[lane_first < 1e9]) if (lane_first < 1e9).any() else 0.0,
                lane_tiles_core_max=lane_n.max(), lane_tiles_core_zero=(lane_n == 0).sum(),
                ready_small_by_mat_end=sum(1 for t in elig if ready[t] <= mat_end.max()),
                eligible=len(elig), safe=float(last_claim >= free0.max()))

lines = gzip.open(a.dump, "rt").read().splitlines()[1:]
views = [{int(k): int(c) for k, c in (p.split(":") for p in ln.split()[3:])} for ln in lines]

if a.views: views = views[:a.views]

def run(cap):
    rows = [model(v, cap) for v in views]
    return {k: (np.mean([r[k] for r in rows]), min(r[k] for r in rows), max(r[k] for r in rows))
            for k in rows[0]}, rows

cap = a.ring_kb * 1024 // 32
res, rows = run(cap)
print(f"blend fit us = {coef[0]:.1f} + {coef[1]:.4f} n + {coef[2]:.4f} max(n-8192,0); "
      f"cal={a.cal} ring={a.ring_kb}KB cap={cap} mover={a.mover} switch={a.switch_us}us "
      f"preempt={a.preempt_us}us load_mult={a.load_mult} matorder={a.matorder} fit={a.fit}")
if a.v:
    for v, r in enumerate(rows):
        print(v, " ".join(f"{k}={x:.3f}" for k, x in r.items()))
for k, (m, lo, hi) in res.items():
    print(f"  {k:24s} {m:8.3f}  (min {lo:.3f} max {hi:.3f})")
