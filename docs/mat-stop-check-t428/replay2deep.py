#!/usr/bin/env python3
"""t428: replay the mat phase of a GSPLAT_TT_MATCULL_PROF capture, per core, and ask what a
2-deep mover job pipeline would save.

Today (task #306, sort_subchunk_materialize.cpp flush_pending/defer_emit) each mover has ONE
slab and ONE cull job in flight: it posts job j, reads + sorts item j+1, then waits for job j
(mat_cull_wait), emits it (mat_ol_wr) and only then permutes item j+1 into the slab. So the perm
of j+1 sits between the end of cull j and the post of j+1, and TRISC1 idles for it.

2-deep: when slab j and slab j+1 fit in CB_SLAB together (NCRISC kBucketFit 8192 records,
BRISC kMatMover0Cap 6144; 32 B each), the mover permutes j+1 behind j, posts it, and waits for
job j only where it waited for job j+1 before (one block later). No L1 growth.

Replay: each mover = its zones in time order (the inter-zone time is kept as work), with every
mat_cull_wait turned into "wait for my oldest job"; a post follows every mat_ol_perm and
mat_ol_gather zone (checked: posts == TRISC1 mj_job count). TRISC1 runs the posted jobs in
arrival order with their measured mj_job lengths plus a fixed per-job overhead. Records per job
= mj_job us x RPU (11.3 rec/us, #419 model calibration). Output: per core, replayed mat length
today (calibration vs the measured TRISC1 mat_cull_mask) and with 2-deep; the mean-core change
is the per-view gain (blend claims tiles dynamically, so cores end within ~0.1 ms: #419, and
the mat+blend core length spread in ana428).

Usage: replay2deep.py STITCHED.csv [--cache PKL]
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "xvpin-tracy"))
import ana407 as A  # noqa: E402

CYC_US = 1350.0
RPU = 11.3
CAP = {"NCRISC": int(os.environ.get("T428_CAP_N", 8192)), "BRISC": int(os.environ.get("T428_CAP_B", 6144))}
JOB_OVH_US = 1.0
POST_Z = ("mat_ol_perm", "mat_ol_gather")
SKIP_Z = ("mat_cull_mask", "sort_subchunk_mat")


def load(path, cache):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    iv, td = A.load_dev(path)
    _, Z, D, NV, cores = A.assign(iv, td)
    out = {}
    for v in range(NV):
        per = defaultdict(lambda: {"mat": None, "jobs": [], "NCRISC": [], "BRISC": []})
        for (vv, p, z), L in Z.items():
            if vv != v or p != 3:
                continue
            for c, r, s, e in L:
                if z == "mat_cull_mask" and r == "TRISC_1":
                    per[c]["mat"] = (s, e)
                elif z == "mj_job" and r == "TRISC_1":
                    per[c]["jobs"].append((s, e))
                elif r in ("NCRISC", "BRISC") and z.startswith("mat_") and z not in SKIP_Z:
                    per[c][r].append((s, e, z))
        out[v] = {c: d for c, d in per.items() if d["mat"] and d["jobs"]}
    if cache:
        pickle.dump(out, open(cache, "wb"))
    return out


def tokens(zs, t0, t_end):
    """mover zones -> [('w', dur) | ('P',) | ('W',)] from mat start, durations in us."""
    zs = sorted(zs)
    tk, t = [], t0
    for s, e, z in zs:
        if s > t:
            tk.append(("w", (s - t) / CYC_US))
        if z == "mat_cull_wait":
            tk.append(("W",))
        else:
            tk.append(("w", (e - max(s, t)) / CYC_US if e > t else 0.0))
            if z in POST_Z:
                tk.append(("P",))
        t = max(t, e)
    return tk


def deepen(tk, jl, cap):
    """Move each wait block [W, w(wr)...] for job j to where the block for job j+1 was, when
    the records of jobs j and j+1 fit the slab together. jl = this mover's job lengths (us)."""
    # split into segments ending at each W; a block = W + the work up to the next perm/post
    blocks, i, out_pre = [], 0, []
    # find W positions and the job each waits for (FIFO: k-th W waits for the k-th posted job
    # still outstanding; with 1-deep it is the job posted just before)
    posted, waits = 0, []
    for k, x in enumerate(tk):
        if x[0] == "P":
            posted += 1
        elif x[0] == "W":
            waits.append((k, posted - 1))
    if not waits:
        return tk
    # block k: tk[W_k] plus following 'w' tokens up to (not incl.) the next zone after the
    # wr zone. Keep it simple: W + the single next 'w' (the mat_ol_wr zone).
    res = list(tk)
    moved = {}
    for n, (k, j) in enumerate(waits):
        if n + 1 < len(waits) and j + 1 < len(jl):
            recs = (jl[j] + jl[j + 1]) * RPU
            if recs <= cap:
                moved[k] = waits[n + 1][0]  # re-insert before the next wait block
    if not moved:
        return tk
    # rebuild: drop moved blocks from their place, insert them before their target
    blk = {}
    for k in moved:
        blk[k] = [res[k]] + ([res[k + 1]] if k + 1 < len(res) and res[k + 1][0] == "w" else [])
    drop = set()
    for k in moved:
        drop.add(k)
        if len(blk[k]) == 2:
            drop.add(k + 1)
    ins = defaultdict(list)
    for k, tgt in moved.items():
        ins[tgt].append(k)
    final = []
    for k, x in enumerate(res):
        for src in sorted(ins.get(k, [])):
            final.extend(blk[src])
        if k not in drop:
            final.append(x)
    return final


def simulate(mov_tk, job_len):
    """mov_tk: {mover: tokens}; job_len: {mover: [us per job in post order]} -> (end, t1 busy)."""
    t = {m: 0.0 for m in mov_tk}
    pc = {m: 0 for m in mov_tk}
    nposted = {m: 0 for m in mov_tk}
    outstanding = {m: deque() for m in mov_tk}
    done = {}  # (m, j) -> finish time
    queue = []  # (post time, m, j)
    trisc_t = 0.0
    trisc_end = 0.0
    while True:
        progressed = False
        for m in mov_tk:
            tk = mov_tk[m]
            while pc[m] < len(tk):
                x = tk[pc[m]]
                if x[0] == "w":
                    t[m] += x[1]
                elif x[0] == "P":
                    j = nposted[m]
                    nposted[m] += 1
                    outstanding[m].append(j)
                    queue.append((t[m], m, j))
                else:  # W
                    if not outstanding[m]:
                        pc[m] += 1
                        progressed = True
                        continue
                    j = outstanding[m][0]
                    if (m, j) not in done:
                        break
                    outstanding[m].popleft()
                    t[m] = max(t[m], done[(m, j)])
                pc[m] += 1
                progressed = True
        # TRISC: run the earliest-posted queued job whose post time is known
        if queue:
            queue.sort()
            # only safe to run a job if no mover could still post an earlier one: movers that
            # are blocked or finished have t >= their next post time
            pt, m, j = queue[0]
            blocked_ok = all(pc[mm] >= len(mov_tk[mm]) or t[mm] >= pt or
                             mov_tk[mm][pc[mm]][0] == "W" for mm in mov_tk)
            if blocked_ok:
                queue.pop(0)
                st = max(trisc_t, pt)
                ln = job_len[m][j] if j < len(job_len[m]) else 0.0
                trisc_t = st + ln + JOB_OVH_US
                done[(m, j)] = trisc_t
                trisc_end = trisc_t
                progressed = True
        if not progressed:
            if all(pc[m] >= len(mov_tk[m]) for m in mov_tk) and not queue:
                break
            if not queue:
                return None  # deadlock (mismatched tokens)
    return max([trisc_end] + list(t.values()))


def core_case(d):
    s0, e0 = d["mat"]
    tks = {m: tokens(d[m], s0, e0) for m in ("NCRISC", "BRISC") if d[m]}
    posts = sorted((tk_time, m) for m in tks for tk_time in post_times(d[m]))
    jobs = sorted(d["jobs"])
    if len(posts) != len(jobs):
        return None
    jl = {m: [] for m in tks}
    for (pt, m), (s, e) in zip(posts, jobs):
        jl[m].append((e - s) / CYC_US)
    return tks, jl, (e0 - s0) / CYC_US


def post_times(zs):
    return [e for s, e, z in sorted(zs) if z in POST_Z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cache")
    a = ap.parse_args()
    data = load(a.csv, a.cache)
    cal, gain, base_len, deep_len, meas, skipped, n = [], [], [], [], [], 0, 0
    worst_b, worst_d = [], []
    for v, cores in sorted(data.items()):
        vb, vd, vm = [], [], []
        for c, d in cores.items():
            cc = core_case(d)
            n += 1
            if cc is None:
                skipped += 1
                continue
            tks, jl, mlen = cc
            b = simulate(tks, jl)
            dp = simulate({m: deepen(tk, jl[m], CAP[m]) for m, tk in tks.items()}, jl)
            if b is None or dp is None:
                skipped += 1
                continue
            vb.append(b)
            vd.append(dp)
            vm.append(mlen)
        base_len.append(np.mean(vb))
        deep_len.append(np.mean(vd))
        meas.append(np.mean(vm))
        worst_b.append(max(vb))
        worst_d.append(max(vd))
    print(f"views {len(data)}, core-views {n}, skipped {skipped} (post/job count mismatch or replay stuck)")
    print(f"measured mat length (T1 mat_cull_mask), mean core: {np.mean(meas) / 1000:.3f} ms")
    print(f"replayed today (1-deep):                  mean core {np.mean(base_len) / 1000:.3f} ms, "
          f"worst core {np.mean(worst_b) / 1000:.3f} ms")
    print(f"replayed 2-deep when both slabs fit:      mean core {np.mean(deep_len) / 1000:.3f} ms, "
          f"worst core {np.mean(worst_d) / 1000:.3f} ms")
    print(f"mean-core mat change: {(np.mean(deep_len) - np.mean(base_len)) / 1000:+.3f} ms/view")


if __name__ == "__main__":
    main()
