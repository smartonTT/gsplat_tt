"""t260: seam check. Step of the error e = hero - ref across pixel pairs that straddle
a 32-px tile edge / an 8x4 microblock edge vs pairs that straddle no edge.
ratio ~1.0 = no seams; >1.2 = visible lines."""
import sys, numpy as np
from PIL import Image
ref = np.asarray(Image.open("benchmarks/reference_v2/hero.png").convert("RGB")).astype(float)
for p in sys.argv[1:]:
    e = np.asarray(Image.open(p).convert("RGB")).astype(float) - ref
    sx = np.abs(np.diff(e, axis=1)).mean(axis=(0, 2))  # step between col j and j+1
    sy = np.abs(np.diff(e, axis=0)).mean(axis=(1, 2))
    jx = np.arange(len(sx)); jy = np.arange(len(sy))
    def r(s, j, per):
        edge = (j % per) == per - 1; base = (j % 32 != 31) & ((j % 8 != 7) if per != 4 else (j % 4 != 3))
        return s[edge].mean() / s[(j % per) != per - 1].mean()
    print(f"{p.split('/')[-1]}: tile32 x {r(sx,jx,32):.3f} y {r(sy,jy,32):.3f} | mb x(8) {r(sx,jx,8):.3f} y(4) {r(sy,jy,4):.3f}")
