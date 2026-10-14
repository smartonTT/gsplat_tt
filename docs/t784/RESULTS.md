# t784: T2 heavy-row split on device: hero A/B

Commit measured: 7fc59b3d (branch ttp/t784-t2-heavy-row-split-on-device-l1-kernel-h).
Box: bh-49 (p150a, measurement reservation), run 14:05-14:14 PDT 2026-10-14, under `ttp lock p100`.
Driver: docs/t784/drive.sh; raw outputs: docs/t784/out/remote/ (log: state/runs/2204/t784.log).
Knobs: `T2_SPLIT=1` turns the split on (default off), `T2_KCAP` heavy-row threshold (default 16).

## A/B (hero, 1014 tiles, 4,478,225 slots, 110 cores, 10 reps per arm, 4 alternating rounds)

| metric (warm median, ms) | base T2 | split kcap 16 | delta | split kcap 8 (1 round) |
|---|---|---|---|---|
| T2 | 98.42 | 9.29 | -89.13 (rounds -89.22 / -89.05 / -89.08 / -89.20) | 8.85 |
| phase A (heavy sums) | - | 2.06 | | 2.06 |
| phase B (light + heavy lanes) | - | 7.27 | | 6.83 |
| T1 (unchanged) | 81.39 | 81.37 | -0.02 (noise) | 81.36 |
| T1+T2 chain = trainer backward ms/view | 179.75 | 90.52 | **-89.23** (all 4 rounds -89.20..-89.26) | 90.16 |

Per-plane cost (busiest core):
- base: 98.42 ms / 30,772 planes = 3.20 us/plane
- phase A (heavy, sum only): 2.06 ms / 1,014 planes = 2.03 us/plane
- phase B kcap 16: 7.27 ms / 2,171 weight (planes + ovh 2 per group) = 3.35 us/unit
- phase B kcap 8: 6.83 ms / 1,937 = 3.53 us/unit

Plan stats kcap 16: 2,975 heavy rows in 186 groups (77,616 heavy planes), 181,694 light planes;
kcap 8: 10,193 heavy rows, 638 groups. Host plan 6.6-7.7 ms + upload 1.1 ms per view (CPU, not in the
T2 time; can be overlapped or cached per view). Phase A's bound (1,014) is one single row with 1,014
slots: a row is never split across lanes, so phase A cannot go below that row.

## Correctness

- Trainer outputs, split vs base on the hero (`check_g2d.py same`): params_out, m_out, v_out, pfwc, grads all **bit-equal**.
- g2d (pfwc) vs `t2_param_backward.segment_reduce` on the hero dump: max |diff| = 0.0 on all 16.3 M words;
  146,174 (kcap 16) / 160,978 (kcap 8) words differ only in the sign of zero (+0 vs -0), and the
  base device T2 has the same zero signs (outputs bit-equal above). The device plan matches the
  Python mirror (`plan_matches_mirror` true).
- Hero gradient check (`gsplat.train_step_dev check`, split arm): t1 / t2 pass, overall pass.
- T0 oracle crops with the split on at kcap 1 (every row heavy) and kcap 16: sh0 passes (t0 gated, pass).
  sh3: t1/t2 pass, t0 is ungated (`gated: false`) and reads false at kcap 16 too, where there are no heavy
  rows, so it is not from the split.
- JIT hashes (`opt/ab_jit_check.py`): JIT-HASH OK.

## Recommendation

Keep. T2 drops 10.6x and the trainer backward chain halves (-89.2 ms/view, far above the 0.1 ms
noise bar), losslessly. kcap 8 is 0.44 ms faster on T2 in one round; confirm over 4 rounds before
making it the default, then turn `T2_SPLIT` on by default.
