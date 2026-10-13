"""Task #580: PSNR + modelled saving table from out/model_*/model_psnr.json.

Saving model (t205 units, docs/blend-loop-model-t205): work counts are hero-view totals over
all cores; ms/view = cycles / CORES / CLK. Per-unit cycle costs, low..high:
- microblock evaluation (one body on one 4x8 microblock): 55..79 SFPU cycles (A2 body ops
  per microblock 60 single / 54.5 per pair half; high adds walk + MAD stalls, t205 table)
- live record staging: 31..108 (31 Tensix pushes; 77 RISC cycles may overlap under A2)
- scanned record: 0..22 (record scan, RISC)
- exp degree-1: 1..3 SFPU cycles per microblock evaluation (drops the c2 MAD; high also folds
  MUL + ADDI into one MAD with the freed const reg and drops a MAD->use stall)
- T readback: no direct measurement. t111 measured period 512 vs period 0 as neutral (period 0
  0.06 ms/view faster), so readback cost at 512 ~= early-out work saved + 0.06 ms. Per-readback
  cost is fitted from that and applied to the change in readbacks.
- pairs outside blend (floor change only): sort 0.455 ms per million pairs (t224 model), scaled by
  the change in tile-list entries; P = 2.72 M pairs/view (30-view mean).
Run: python3 summarize.py > out/summary.txt
"""
import json, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
CORES, CLK = 110, 1.35e9
MEAS_BASE_PSNR = 42.51          # best-iter-226 device hero vs reference_v2 (md5 906e0435)
P_PAIRS = 2.72e6
C_MB, C_LIVE, C_SCAN, C_EXP = (55, 79), (31, 108), (0, 22), (1, 3)
T111_P0_FASTER = 0.06


def ms(cyc):
    return cyc / CORES / CLK * 1e3


def load(k):
    p = f'{D}/model_{k}/model_psnr.json'
    return json.load(open(p))['results'] if os.path.exists(p) else None


def work_ms(r, b, i):
    """Blend work saved (ms/view) of r vs base b at cost index i (0 low, 1 high)."""
    return ms((b['mb_evals'] - r['mb_evals']) * C_MB[i] + (b['live_recs'] - r['live_recs']) * C_LIVE[i]
              + (b['scanned'] - r['scanned']) * C_SCAN[i])


def saving(n, r, b, nochk):
    lo_hi = []
    for i in (0, 1):
        s = work_ms(r, b, i)
        if 'exp1' in n or n == 'all':
            s += ms(r['mb_evals'] * C_EXP[i])
        if nochk and b['readbacks']:
            rb_total = -work_ms(b, nochk, i) + T111_P0_FASTER   # readback cost at period 512
            s += rb_total / b['readbacks'] * (b['readbacks'] - r['readbacks'])
        if r['scanned'] != b['scanned']:
            s += 0.455 * P_PAIRS / 1e6 * (b['scanned'] - r['scanned']) / b['scanned']
        lo_hi.append(s)
    return min(lo_hi), max(lo_hi)


cpu, dev, cds = load('cpu'), load('dev'), load('cpu_devset')
print('Modelled PSNR (8-bit, vs benchmarks/reference_v2/hero.png) and saving, hero view.')
print('inputs cpu = cpu_cpp projection, all rows (~ device with the landed L6 Newton step).')
print(f'pred = {MEAS_BASE_PSNR} (measured device, best-iter-226) + (variant - base) on cpu inputs.\n')
b, nc = cpu['base'], cpu.get('base_nochk')
hdr = f"{'variant':26s} {'vs_ref':>7s} {'d_base':>7s} {'pred':>6s} {'vs_gold':>7s} {'vs_base':>7s} {'mb_evals':>9s} {'live':>8s} {'scanned':>8s} {'rb':>5s} {'save lo..hi ms':>15s} gate"
print(hdr)
for n, r in cpu.items():
    d = r['vs_ref'] - b['vs_ref']
    pred = MEAS_BASE_PSNR + d
    lo, hi = saving(n, r, b, nc) if n != 'base' else (0.0, 0.0)
    gate = 'GO' if (pred >= 42.4 and lo >= 0.1) else ('maybe' if pred >= 42.4 and hi >= 0.1 else 'no')
    if n in ('base', 'base_exact', 'base_nochk'):
        gate = '-'
    print(f"{n:26s} {r['vs_ref']:7.2f} {d:+7.2f} {pred:6.2f} {r['vs_golden']:7.2f} {r.get('vs_base', 99):7.2f} "
          f"{r['mb_evals']:9d} {r['live_recs']:8d} {r['scanned']:8d} {r['readbacks']:5d} {lo:+7.3f}..{hi:+6.3f} {gate}")

if dev and cds:
    print('\nL6 (Newton step on device 1/z + conic): device-dumped pfwc rows (pre-#266, iter 199) vs the')
    print('same rows with CPU projection values (= exact 1/z; #266 measured the device Newton at ~1 ulp).')
    print('Same-row sets lose ~5 dB vs ref to dump matching (t260), so only the deltas mean anything.\n')
    print(f"{'variant':26s} {'dev':>7s} {'dev+N':>7s} {'L6 gain':>8s} {'d_base dev':>10s} {'d_base dev+N':>12s}")
    for n in dev:
        if n not in cds:
            continue
        a, c = dev[n]['vs_ref'], cds[n]['vs_ref']
        print(f"{n:26s} {a:7.2f} {c:7.2f} {c - a:+8.2f} {a - dev['base']['vs_ref']:+10.2f} {c - cds['base']['vs_ref']:+12.2f}")
