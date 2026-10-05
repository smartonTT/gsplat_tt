"""t260: per-arm ms/view (mean of per-view walls, 3 rounds) and hero 8-bit PSNR vs reference_v2."""
import glob, re, sys, numpy as np
from PIL import Image
O = sys.argv[1]
def rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
ref = rgb("benchmarks/reference_v2/hero.png"); gold = rgb("tests/fixtures/hero/hero_golden_8bit.png")
def psnr(a, b):
    m = np.mean((a - b) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)
for a in "ABCDE":
    rounds = []
    for r in (1, 2, 3):
        v = [float(x) for x in re.findall(r"view=\S+ ([0-9.]+)ms", open(f"{O}/logs/run-r{r}-{a}.log").read())]
        assert len(v) == 30, (a, r, len(v)); rounds.append(np.mean(v))
    h1, h3 = rgb(f"{O}/hero/hero-r1-{a}.png"), rgb(f"{O}/hero/hero-r3-{a}.png")
    d = h1 - ref
    print(f"{a} ms/view {np.mean(rounds):.3f} rounds {' '.join(f'{x:.3f}' for x in rounds)} "
          f"psnr_vs_ref {psnr(h1, ref):.2f} psnr_vs_golden {psnr(h1, gold):.2f} golden_match {np.array_equal(h1, gold)} "
          f"r1==r3 {np.array_equal(h1, h3)} mean_diff {d.mean():+.3f} max {np.abs(d).max():.0f} >=2lv {(np.abs(d).max(2) >= 2).mean()*100:.2f}%")
