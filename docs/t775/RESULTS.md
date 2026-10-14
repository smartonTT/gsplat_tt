# #775 T1 load balance by body count (model only, no device)

Hero view (_DSC8679, 1237x822, 110 cores), body counts from `docs/t775/hero_bodies.py`
(1014 tiles, 4,478,225 pairs, 30,459,432 bodies). Model and plans: `gsplat/t1_balance.py`;
full numbers: `results/hero_balance.json`.

    PYTHONPATH=. python3 docs/t775/hero_bodies.py tmp/t775/hero_bodies.npz      # ~20 min on the Mac
    python3 -m gsplat.t1_balance tmp/t775/hero_bodies.npz \
        --stats docs/t747/results/stats-hero-cand.json --json docs/t775/results/hero_balance.json

Check against the device: per-core tiles, pairs and bodies of the current contiguous split equal
the #747 hero stats exactly; modelled per-core cycles are within 0.9% of the measured ones
(busiest core 82.6 ms measured and modelled).

| plan (modelled, lossless) | busiest/mean core bodies | busiest core ms | mean/busiest |
|---|---|---|---|
| contiguous by pairs (current bench) | 1.778 (492,267 / 276,904) | 82.61 | 0.602 |
| LPT by modelled cycles, whole tiles | 1.051 (290,991) | 51.39 | 0.968 |
| LPT + microblock split, k<=2 (81 tiles) | 1.023 (283,210) | 50.09 | 0.997 |
| LPT + split, k<=4 (75 tiles) | 1.025 | 50.05 | 0.998 |
| LPT + split, k<=8 (71 tiles) | 1.027 | 50.09 | 0.997 |

Ideal (total / 110) is 49.76 ms. LPT alone takes most of it (82.6 -> 51.4 ms modelled, outputs
bit-identical). The split adds about 1.3 ms more at the cost of ~80k partial slots (1.8% of pairs)
merged in T2 in fixed part order; k above 2 buys nothing. Heaviest tiles: 595/596 (~106k bodies
each, ~2.1x a core's mean share), so whole-tile LPT cannot go below ~1.05.

Not measured: these are model numbers. The device change (row permutation + per-core row
boundaries in blend_bwd_bench, and the T2 merge for split rows) is the next step.
