"""t270: PSNR + x10 diff image of a device render against a named reference."""
import sys
import numpy as np
from PIL import Image


def load(p):
    if p.endswith(".npy"):
        a = np.load(p)
        return np.clip(np.rint(np.clip(a, 0, 1) * 255), 0, 255).astype(np.uint8)
    return np.asarray(Image.open(p).convert("RGB"))


def main(cand, ref, out_diff):
    a, b = load(cand).astype(np.float64), load(ref).astype(np.float64)
    mse = ((a - b) ** 2).mean()
    psnr = 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    d = np.abs(a - b).max(axis=2)
    Image.fromarray(np.clip(d * 10, 0, 255).astype(np.uint8)).save(out_diff)
    print(f"{cand} vs {ref}: PSNR {psnr:.2f} dB, max abs diff {int(d.max())}, "
          f"px>8: {(d > 8).mean() * 100:.3f}%")


if __name__ == "__main__":
    main(*sys.argv[1:4])
