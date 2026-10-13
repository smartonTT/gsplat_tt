#!/usr/bin/env python3
"""t614: summarize host timelines (htl-*.txt[.gz]) and B2B lines of the probe arms.

  analyze.py <out dir>
Per arm: b2b ms/frame (median of measured passes), and from the host timeline of the
measured passes (views after the first SKIP view_total spans): mean ms/view of every
B/E tag, and the fold window H = fold_hook begin -> fz_enqueue end (host work between
the next view's pfwc enqueue and this view's mat+blend enqueue)."""
import collections, glob, gzip, os, re, statistics, sys

d = sys.argv[1]
SKIP = int(os.environ.get("SKIP", "60"))


def b2b(log):
    m = re.search(r"^B2B scene.*?ms_frame=([\d.]+).*?ms_frame_median=([\d.]+).*?pass_ms_frame=([\d.,]+)",
                  open(log).read(), re.M)
    return (float(m.group(2)), m.group(3)) if m else (None, "")


def htl(path):
    op = gzip.open if path.endswith(".gz") else open
    marks = []
    with op(path, "rt") as f:
        for ln in f:
            ns, k, tag = ln.split(None, 2)
            marks.append((int(ns), k, tag.strip()))
    marks.sort(key=lambda x: x[0])
    views = [m[0] for m in marks if m[2] == "view_total" and m[1] == "B"]
    if len(views) <= SKIP + 2:
        return None
    lo, hi = views[SKIP], views[-1]
    n = len(views) - 1 - SKIP
    open_ = collections.defaultdict(list)
    tot = collections.Counter()
    cnt = collections.Counter()
    hook_b, H = None, []
    for ns, k, tag in marks:
        if k == "B":
            open_[tag].append(ns)
            if tag == "fold_hook" and lo <= ns < hi:
                hook_b = ns
        elif k == "E" and open_[tag]:
            b = open_[tag].pop()
            if lo <= b < hi:
                tot[tag] += ns - b
                cnt[tag] += 1
            if tag == "fz_enqueue" and hook_b is not None:
                H.append(ns - hook_b)
                hook_b = None
    period = (hi - lo) / n / 1e6
    per = {t: (tot[t] / n / 1e6, cnt[t] / n) for t in tot}
    return period, per, (statistics.median(H) / 1e6 if H else None), n


rows = []
for log in sorted(glob.glob(f"{d}/run-*.log")):
    tag = os.path.basename(log)[4:-4]
    med, passes = b2b(log)
    h = next(iter(glob.glob(f"{d}/htl-{tag}.txt*")), None)
    r = htl(h) if h else None
    rows.append((tag, med, passes, r))
    print(f"{tag:12s} b2b_median={med} passes={passes}"
          + (f" htl_period={r[0]:.3f} H_fold_med={r[2]} views={r[3]}" if r else ""))
if "-v" in sys.argv:
    for tag, med, passes, r in rows:
        if not r:
            continue
        print(f"\n== {tag}  (ms/view, calls/view)")
        for t, (ms, c) in sorted(r[1].items(), key=lambda x: -x[1][0]):
            if ms >= 0.005:
                print(f"  {t:36s} {ms:7.3f}  {c:5.2f}")
