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
