"""t510 stage 0 summary: per-variant hero PSNR vs the reference, 30-view min/median PSNR vs the
fp32 model (`base`), max LSB diff, and a seam statistic (mean |diff| on 32-px tile-edge pixels
over the interior mean; ~1 means no seams). Writes amplified diff PNGs for the hero.
Usage: python3 summarize.py <out dir with *.png, views*.log, rows-hero-*.jsonl> [ref.png]"""
import glob, json, statistics, sys
from pathlib import Path
import numpy as np
from PIL import Image

out = Path(sys.argv[1])
ref = np.asarray(Image.open(sys.argv[2] if len(sys.argv) > 2 else "benchmarks/reference_v2/hero.png").convert("RGB")).astype(int)
rows = [json.loads(l) for f in sorted(glob.glob(str(out / "rows-hero-*.jsonl"))) + sorted(glob.glob(str(out / "views*.log")))
        for l in open(f) if l.startswith("{")]
seen = {}
for r in rows:
    seen[(r["view"], r["variant"])] = r  # last row wins (a resumed view overwrites its partial)
views = sorted({v for v, _ in seen})
variants = list(dict.fromkeys(vn for _, vn in seen))


def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)


edge = np.zeros(ref.shape[:2], bool)
edge[31::32, :] = edge[32::32, :] = edge[:, 31::32] = edge[:, 32::32] = True
base = np.asarray(Image.open(out / "hero_base.png")).astype(int)
print(f"views: {len(views)}")
print("variant | hero dB vs ref | views | min dB vs base | median dB vs base | max LSB | hero seam ratio")
for vn in variants:
    ps = [round(seen[(v, vn)]["psnr_vs_base"], 3) for v in views if (v, vn) in seen and "psnr_vs_base" in seen[(v, vn)]]
    md = max([seen[(v, vn)].get("maxdiff", 0) for v in views if (v, vn) in seen] or [0])
    h = seen.get(("hero", vn), {}).get("psnr_vs_ref")
    seam = "-"
    fp = out / f"hero_{vn}.png"
    if vn != "base" and fp.exists():
        d = np.abs(np.asarray(Image.open(fp)).astype(int) - base).sum(-1)
        ie = d[~edge].mean()
        seam = f"{d[edge].mean() / ie:.2f}" if ie > 0 else "n/a (identical)"
        Image.fromarray(np.clip(d * 16, 0, 255).astype(np.uint8)).save(out / f"hero_{vn}_diff_x16_vs_base.png")
        Image.fromarray(np.clip(np.abs(np.asarray(Image.open(fp)).astype(int) - ref).sum(-1) * 4, 0, 255).astype(np.uint8)).save(out / f"hero_{vn}_diff_x4_vs_ref.png")
    n = sum((v, vn) in seen for v in views)
    print(f"{vn} | {h} | {n} | {min(ps) if ps else '-'} | {round(statistics.median(ps), 3) if ps else '-'} | {md} | {seam}")
