"""t661: md5, PSNR vs benchmarks/reference_v2/hero.png and pixel diffs between the verify.sh heroes."""
import hashlib, itertools, pathlib, sys
import numpy as np
from PIL import Image

out = pathlib.Path(sys.argv[1])
ref = np.asarray(Image.open("benchmarks/reference_v2/hero.png").convert("RGB")).astype(np.float64)
ims = {}
for p in sorted(out.glob("*-hero_viewer*.png")):
    a = np.asarray(Image.open(p).convert("RGB"))
    ims[p.stem] = a
    mse = ((a.astype(np.float64) - ref) ** 2).mean()
    d = np.abs(a.astype(np.int16) - ref.astype(np.int16)).max(-1).astype(np.float64)
    Image.fromarray(np.clip(d * 10, 0, 255).astype(np.uint8)).save(out / f"diff10-{p.stem}.png")
    print(f"HERO {p.stem} md5={hashlib.md5(a.tobytes()).hexdigest()[:8]} psnr_vs_ref={10*np.log10(255**2/mse):.3f}")
for (na, a), (nb, b) in itertools.combinations(ims.items(), 2):
    m = (a != b).any(-1)
    if m.any():
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f"px-{na}_vs_{nb}.png")
    print(f"PAIR {na} {nb} px_diff={int(m.sum())} max={int(np.abs(a.astype(int)-b.astype(int)).max())}")
