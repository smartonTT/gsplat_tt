#!/usr/bin/env python3
"""t384 model: do the pfwc writers emitting the (gid, tid) pairs pay for dropping K2?

Host-only. Inputs are measured device captures already in the repo:
  T197  docs/pfwc-breakdown-t197/out/pc2-r9-dev.csv.gz   p100a, 5 views, single writer,
        pfwc_pw per core: n wall wait bar cls rec tail iss m pr (cycles / counts),
        k2_pairs and sort_ol_emit per mover.
  T366  docs/p150-blend-gap/out/t366-dev.csv.gz         bh-30 p150, iter-209, 31 frames:
        pfwc (TRISC), k2_pairs, sort_ol_emit per mover.
  render/host/sort_mover_speed.h                         kMoverSpeedP150 (page weights).
  t221 writer split (prof-both-pc_split.txt): max-core busy BRISC 1.67, NCRISC 1.58 ms.

Output: model-out.txt (run: python3 model.py > model-out.txt).
"""
import gzip, os, random, re, statistics as st
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
MHZ = 1350.0
CYC_MS = MHZ * 1e3


def speeds():
    txt = open(os.path.join(ROOT, 'render/host/sort_mover_speed.h')).read()
    body = txt.split('kMoverSpeedP150[] = {')[1].split('};')[0]
    tab = {}
    for x, y, b, n in re.findall(r'\{(\d+), (\d+), (\d+), (\d+)\}', body):
        tab[(int(x), int(y), 'BRISC')] = int(b)
        tab[(int(x), int(y), 'NCRISC')] = int(n)
    return tab


def zones(path, names):
    S, E = defaultdict(list), defaultdict(list)
    for line in gzip.open(path, 'rt'):
        p = line.split(',')
        if len(p) < 12 or p[10] not in names or p[11] not in ('ZONE_START', 'ZONE_END'):
            continue
        k = (p[10], int(p[1]), int(p[2]), p[3])
        (S if p[11] == 'ZONE_START' else E)[k].append(int(p[5]))
    out = defaultdict(dict)  # zone -> mover -> mean ms
    for k in S:
        n = min(len(S[k]), len(E[k]))
        out[k[0]][k[1:]] = st.mean((E[k][i] - S[k][i]) / CYC_MS for i in range(n))
    return out


def pfwc_pw(path):
    """per core: list over launches of dict(wall, wait, cls, rec, m, pr)."""
    names = 'n wall wait bar cls rec tail iss m pr'.split()
    raw = defaultdict(list)
    for line in gzip.open(path, 'rt'):
        p = line.split(',')
        if len(p) > 11 and p[10] == 'pfwc_pw':
            raw[(int(p[1]), int(p[2]))].append(int(p[6]))
    per = {}
    for core, vals in raw.items():
        recs, cur = [], {}
        for v in vals:
            idx, val = v >> 32, v & 0xFFFFFFFF
            cur[names[idx]] = val
            if idx == 9:
                recs.append(cur)
                cur = {}
        per[core] = recs
    return per


def z366_prefix(path):
    z = zones(path, {'sort_ol_prefix'})['sort_ol_prefix']
    return list(z.values())


def makespan_units(units, speed, lpt=True):
    """assign units (pairs) to movers (speed); greedy LPT on finish time."""
    if not lpt:  # one unit per mover, sizes sorted against speeds
        u = sorted(units, reverse=True)
        s = sorted(speed, reverse=True)
        return max(a / b for a, b in zip(u, s))
    load = [0.0] * len(speed)
    for w in sorted(units, reverse=True):
        # put w where it finishes earliest
        best = min(range(len(speed)), key=lambda i: (load[i] + w) / speed[i])
        load[best] += w
    return max(load[i] / speed[i] for i in range(len(speed)))


def main():
    tab = speeds()
    t197 = os.path.join(ROOT, 'docs/pfwc-breakdown-t197/out/pc2-r9-dev.csv.gz')
    t366 = os.path.join(ROOT, 'docs/p150-blend-gap/out/t366-dev.csv.gz')
    z197 = zones(t197, {'k2_pairs', 'sort_ol_emit'})
    z366 = zones(t366, {'k2_pairs', 'sort_ol_emit', 'pfwc'})
    pw = pfwc_pw(t197)
    cores = sorted(pw)
    L = min(len(v) for v in pw.values())
    P = [sum(pw[c][i]['pr'] for c in cores) for i in range(L)]
    M = [sum(pw[c][i]['m'] for c in cores) for i in range(L)]
    print(f'# t197 p100a: {len(cores)} cores, {L} views; P/view {st.mean(P):.0f} M/view {st.mean(M):.0f}'
          f' pairs/visible {st.mean(P) / st.mean(M):.3f}')

    # --- 1. K2 cost per pair (both captures) -------------------------------------------
    print('\n## 1. K2 cycles per pair (speed-split ranges, count rows on)')
    alpha = {}
    for tag, z, Pv in (('p100a t197', z197, st.mean(P)), ('bh-30 t366', z366, st.mean(P))):
        k2 = z['k2_pairs']
        ssum = sum(tab.get(m, 1000) for m in k2)
        cpp = [k2[m] * CYC_MS / (Pv * tab.get(m, 1000) / ssum) for m in k2]
        alpha[tag] = st.mean(cpp)
        print(f'{tag}: k2_pairs mover mean {st.mean(k2.values()):.3f} ms max {max(k2.values()):.3f};'
              f' cycles/pair mean {st.mean(cpp):.1f} (p10 {sorted(cpp)[len(cpp) // 10]:.1f},'
              f' p90 {sorted(cpp)[len(cpp) * 9 // 10]:.1f})  [bh-30 assumes the same P]')

    # --- 2. writer time after adding pair emission ---------------------------------------
    print('\n## 2. pfwc writers + pair emission (per core, split writer: each writer p_c/2 pairs)')
    busy = {c: st.mean((r['wall'] - r['wait']) / CYC_MS for r in pw[c]) for c in cores}
    pr = {c: st.mean(r['pr'] for r in pw[c]) for c in cores}
    scale = 1.67 / max(busy.values())  # t221 split: max-core writer busy 1.67 ms (BRISC)
    trisc366 = list(z366['pfwc'].values())
    floor = max(trisc366)
    print(f'single-writer busy (t197) mean {st.mean(busy.values()):.3f} max {max(busy.values()):.3f};'
          f' split scale {scale:.3f}; bh-30 pfwc TRISC mean {st.mean(trisc366):.3f} max {floor:.3f}')
    pmean = st.mean(pr.values())
    print(f'per-core pairs mean {pmean:.0f} max {max(pr.values()):.0f} (max/mean {max(pr.values()) / pmean:.3f})')
    a0 = alpha['p100a t197']
    rows = []
    for f in (0.6, 0.75, 0.9, 1.0):
        new = max(busy[c] * scale + f * a0 * pr[c] / 2 / CYC_MS for c in cores)
        grow = max(0.0, new - floor)
        rows.append((f, new, grow))
        print(f'  per-pair cost {f:.2f} x K2 ({f * a0:.0f} cyc): heaviest writer {new:.3f} ms -> pfwc +{grow:.3f} ms')

    # --- 3. sort emit with unit-locked count rows ---------------------------------------
    print('\n## 3. sort emit makespan, unit-locked vs dense speed split (bh-30 effective speeds)')
    em = z366['sort_ol_emit']
    eff = {m: tab.get(m, 1000) / em[m] for m in em}  # pages given / time taken
    movers = sorted(em)
    sp_table = [tab.get(m, 1000) for m in movers]
    sp_eff = [eff[m] for m in movers]
    emit_mean, emit_max = st.mean(em.values()), max(em.values())
    print(f'bh-30 emit per mover mean {emit_mean:.3f} max {emit_max:.3f} ms (table-speed split today)')
    # today's work per mover w_m = W * table_m / sum(table); time = w_m / eff_m (normalized)
    W = 1.0
    t_today = max(W * tab.get(m, 1000) / sum(sp_table) / (eff[m]) for m in movers)
    norm = emit_max / t_today
    t_cal = W / sum(sp_eff)  # dense split weighted by the bh-30 effective speeds
    print(f'  dense split, today\'s table : {t_today * norm:.3f} ms')
    print(f'  dense split, bh-30-calibrated : {t_cal * norm:.3f} ms  (separate lever, see README)')
    rng = random.Random(384)
    cl = list(pr.values())
    emit_units = {}
    for sd in (0.127,):
        for name, parts in (('halves', 2), ('quarters', 4), ('eighths', 8)):
            res_cal, res_tab = [], []
            for _ in range(20):
                units = []
                for p in cl:
                    fr = [max(0.05, rng.gauss(1.0, sd * (parts / 2) ** 0.5)) for _ in range(parts)]
                    s = sum(fr)
                    units += [p * x / s for x in fr]
                tot = sum(units)
                units = [u / tot for u in units]
                res_cal.append(makespan_units(units, sp_eff) / t_cal)
                res_tab.append(makespan_units(units, sp_eff) / t_today)
            emit_units[parts] = st.mean(res_cal) * t_cal * norm - t_cal * norm
            print(f'  unit-locked {name:8s} ({len(cl) * parts} units, LPT on bh-30 speeds):'
                  f' {st.mean(res_cal):.3f}x calibrated dense, {st.mean(res_tab):.3f}x today'
                  f' -> {st.mean(res_cal) * t_cal * norm:.3f} ms')

    # --- 4. net ------------------------------------------------------------------------
    print('\n## 4. net on bh-30 (ms/view; + = faster)')
    gross = 2.739 - 1.904  # sort_ol_prefix start - k2_pairs start (timeline-bh30-209)
    print(f'gross (sort starts where K2 starts today): {gross:.3f}')
    prefix = max(z366_prefix(t366))
    print(f'prefix today (220 rows): max mover {prefix:.3f} ms; scales with rows (units x 1)')
    print('baseline = calibrated dense emit (the speed-table lever is taken on its own either way)')
    print(f'{"pair cost":>10s} {"pfwc +":>7s} | ' + ' | '.join(
        f'{n}: emit + / prefix + / NET' for n in ('halves', 'quarters', 'eighths')))
    for f in (0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
        new = max(busy[c] * scale + f * a0 * pr[c] / 2 / CYC_MS for c in cores)
        grow = max(0.0, new - floor)
        cells = []
        for parts in (2, 4, 8):
            pre = prefix * (parts / 2 - 1)
            net = gross - grow - emit_units[parts] - pre
            cells.append(f'{emit_units[parts]:.3f} / {pre:.3f} / {net:+.3f}')
        print(f'{f:10.2f} {grow:7.3f} | ' + ' | '.join(cells))
    print('gate: NET >= 0.15 ms')


if __name__ == '__main__':
    main()
