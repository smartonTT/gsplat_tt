# t788 train-forward blend levers: A/B results

bh-49 (p150a), b2b ms/view, median of 20 passes per round, 4 rounds with the arm order rotated.
o = orbit (cameras_v2.json, 30 views incl. hero), n = first 30 dataset views (cameras_train_nn.json, 1237x822).

## Run 1 (build 531d4e58, 2026-10-14 15:00 PDT)

| arm | o r1..r4 | n r1..r4 | o delta | n delta |
|---|---|---|---|---|
| tr  (TRAIN=1 baseline) | 14.703 14.700 14.703 14.703 | 13.858 13.859 13.857 13.859 | - | - |
| tp  (exact T early-out, period 512, eps FLT_MIN) | 13.981 13.978 13.973 13.976 | 13.246 13.248 13.248 13.248 | -0.73 | -0.61 |
| mm  (SFPSWAP clamps) | 14.305 14.306 14.306 14.302 | 13.513 13.514 13.515 13.511 | -0.40 | -0.35 |
| fz / all (fused program) | failed: leftover `fz` refusal in blend_device.cpp | | | |

Image raw_md5 equal to the baseline in every tp and mm run (orbit 0c8d83a1, dataset 5361c6c6).
The per-forward train-buffer hash runs were refused by devrun (timeout 900 s over its 600 s ceiling).
Fixed in faea0299; run 2 repeats all arms with the fused ones and the hash identity runs.

## Run 2 (build faea0299, 2026-10-14 15:41-15:57 PDT, bh-49 p150a)

| arm | o r1..r4 | n r1..r4 | o delta | n delta |
|---|---|---|---|---|
| tr  (TRAIN=1 baseline) | 14.691 14.700 14.697 14.704 | 13.859 13.857 13.860 13.860 | - | - |
| tp  (exact T early-out) | 13.978 13.976 13.977 13.978 | 13.251 13.250 13.249 13.248 | -0.72 | -0.61 |
| mm  (SFPSWAP clamps) | 14.303 14.306 14.304 14.303 | 13.513 13.515 13.517 13.511 | -0.40 | -0.35 |
| fz / all (fused program) | failed at program compile (below) | | | |

Lossless check: per-forward fp32 train buffer (R, G, B, T per pixel; the backward's
input) hashed after every forward (GSPLAT_TT_BLEND_TRAIN_HASH=1), 31 forwards per arm
(warm-up + 30 views, hero included): tp and mm equal tr on every forward, orbit and
dataset views (out/hash-r0-*.txt). Image raw_md5 equal in every b2b run. The backward
kernels are unchanged (ab_jit_check: only alpha_blend_compute_mb differs, all other
kernels identical), so the gradients are identical to the TRAIN=1 baseline.

ab_jit_check (out/jit-check.txt): tr vs tp and tr vs mm OK (alpha_blend_compute_mb
recompiled with a new hash). tr vs fz / all: MISSING alpha_blend_compute_mb, expected:
the fused program runs matblend_compute instead (different kernel sets); with
--allow-kernel-swap the 8 shared kernels are identical.

Fused program in train mode: not feasible without a new L1 map. tt-metal refuses it
("Statically allocated circular buffers ... grow to 1591168 B, beyond max L1 size of
1572864 B", 18304 B over), and that is only the CB_TRAIN_OUT (16 KB) and the train
decode-ahead ring. The train reader also stages the fp32 op/colour of each subchunk in
CB_TRAIN_OPC (BULK_REC_SLOT 4096 x 64 B = 256 KB per slot), which the unfused program
gets from the in-budget CB_BUCKET; the fused program does not create that CB (its L1 is
already shared with the mat CBs). The fused code path was removed again.

Not built: SCHED keeps raw-TTI bodies that inline the fast exp_21f; fast exp and bf16
opacity/colour change the forward values, so they fail the lossless acceptance.
