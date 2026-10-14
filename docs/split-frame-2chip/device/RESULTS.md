# Split-frame on 2 chips: device results (task #627)

Measured 2026-10-13 18:28-18:37 PDT on bh-50 (2x p150, IRD job 133915, released 19:04 PDT).
Build fe9cf434 (best-iter-229 + the #624 split build), one build and one JIT cache for all arms.
Grid capped at 11x10 on both cards. Bicycle, 30 views, `render/run.py --back-to-back`.
Warm-up round, then 3 rotated rounds per arm. Raw logs: `out/run3/logs.tar.gz`, summary
`out/run3/summary.txt`.

**Caveat: the two cards are not the same speed.** Card 0 (fw 18.12.1) runs one chip at 13.24 ms/view
b2b, card 1 (fw 19.7.0) at 7.92. Card 0 sets every 2-chip number below. The single-chip
headline (bh-33) is not affected by this run.

## Arms (median of 3 rounds, ms)

| arm | what | ms/view | per chip (card 0, card 1) | raw_md5 |
|---|---|---|---|---|
| A | 1 chip (card 0), b2b | 13.238 | - | 1d372b94 |
| A | 1 chip (card 0), latency | 13.137 | - | - |
| A1 | 1 chip (card 1), b2b | 7.917 (1 run) | - | 1d372b94 |
| B | 2 chips (split), latency, lockstep | 9.545 + 1.540 join = 11.085 | 9.65, 9.55 | 1d372b94 |
| C | 2 chips (split), b2b | 9.632 | 9.63, 5.87 | 1d372b94 |
| D | 2 chips (AFR), b2b | 6.643 | 13.29, 7.87 | 1d372b94 |

- Split at tile column 16 (x). `split_latency_ms` is max(t1) - min(t0) over both chips per view.
  `split_copy_ms` is the host join of the two window crops (a `.npy` stand-in for a shared pinned
  buffer). It is added on top for the latency figure.
- Latency cut vs arm A on the same card: (13.137 - 11.085) / 13.137 = **15.6%**. Without the join it
  would be 27%.
- In lockstep (B) both chips report about 9.6 ms per own view. Card 1's own time includes waiting
  for the coordinator to release the next view, so its 5.87 shows only in C.
- Half a frame takes 0.728 of the full frame on card 0 (9.632 / 13.238) and 0.741 on card 1
  (5.867 / 7.917). This matches the #616 model (central case 0.71).

## Correctness

- md5: the joined 30-view split md5 equals 1-chip on both cards (1d372b94, the golden).
- Hero: split hero vs 1-chip hero px_diff 0 (`out/run3/hero-split_vs1chip_x20.png`).
- PSNR vs benchmarks/reference_v2/hero.png: 42.85 dB for 1 chip, card 1 and split (bar 42.4).
- Visual check (I looked at `out/run3/hero-split.png` and `hero-split_diff10.png`): no seam at
  column 512 and no tile artifacts. The diff shows only the usual fine edges (spokes, foliage),
  the same as the 1-chip render.

## Verdict

Measured against arm A on the same host, the cut is 15.6%. That clears the 15% bar, but only
just. md5, PSNR and the visual check pass. **Kept as an opt-in mode** (GSPLAT_TT_SPLIT=1, default
off; the 1-chip path is unchanged). It does not change the single-chip headline.

The cut comes mostly from moving work off the slow card. A projection (not measured) for two cards
at card 1's speed: 5.87 + 1.54 join = 7.41 vs 7.92, a 6% cut, under the bar. With a join of 0.2 ms
or less (a shared pinned buffer, as the #624 spec planned) it would be about 23%. So the host join
decides this lever. Before any viewer use it needs a real shared-buffer output and a rerun on two
cards with the same firmware.

For throughput, AFR (6.64 ms/view, D) beats split b2b (9.63, C) on this host.
