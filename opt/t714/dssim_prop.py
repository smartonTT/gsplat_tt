"""#714: how a relative error in the device's Gaussian filter outputs propagates to the D-SSIM
loss and dL/dimage (the T0 check: grad l2rel <= 1e-4, |loss err| < 1e-5).

Every filter output (mu_x, mu_y, E_xx, E_yy, E_xy and the three backward filters) is
multiplied by (1 + bias + std * N(0, 1)); bias/std as measured on the p150 (RESULTS-DSSIM.md).
Reference: the same algorithm with exact (fp32) filters, itself checked against fp64.
  python3 opt/t714/dssim_prop.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "opt", "t690"))
import gsplat.train_loss as tl  # noqa: E402
from loss_check import cases  # noqa: E402

EXACT = tl.filt_tiles
MODELS = [  # name, rel bias, rel std (per filter output element)
    ("gauss-split3 (measured 2.8e-4)", 2.6e-4, 1.5e-4),
    ("box accumulation (measured 1e-5)", 7.6e-6, 7.0e-6),
    ("3e-6", 2e-6, 2e-6),
    ("1e-6", 6e-7, 6e-7),
    ("fp32 rounding only", 0.0, 6e-8),
]


def run(x, y, bias, std, seed=0):
    rs = np.random.RandomState(seed)

    def noisy(z, scheme="fp32", rmode="rne"):
        f = EXACT(z, "fp32")
        return (f * (1.0 + bias + std * rs.standard_normal(f.shape))).astype(np.float32)

    tl.filt_tiles = noisy
    try:
        return tl.loss_and_grad_device(x, y, "fp32")
    finally:
        tl.filt_tiles = EXACT


def bike_cases():
    """Real bicycle photo (images_4/_DSC8679, never committed) as the target; the "render" is the
    photo blurred and noised to about 26 dB, like a mid-training render. Two 256x384 crops."""
    from PIL import Image, ImageFilter
    p = os.environ.get("BIKE_IMG", "/Users/smarton/dev/gsplat_tt/scenes/mipnerf360/bicycle/images_4/_DSC8679.JPG")
    if not os.path.exists(p):
        return
    im = Image.open(p).convert("RGB")
    y = np.asarray(im, np.float32) / 255.0
    xb = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0
    x = np.clip(xb + np.random.RandomState(714).normal(0, 0.02, y.shape), 0, 1).astype(np.float32)
    print(f"bike psnr={-10 * np.log10(np.mean((x - y) ** 2)):.1f} dB")
    for nm, r0, c0 in (("bike_top", 0, 400), ("bike_mid", 300, 400)):
        yield nm, x[r0:r0 + 256, c0:c0 + 384], y[r0:r0 + 256, c0:c0 + 384]


def main():
    import itertools
    for name, x, y in itertools.chain(cases(), bike_cases()):
        if name == "syn":
            x, y = x[:256, :384], y[:256, :384]  # keep the CPU run short
        rl, _, rs_, rg = tl.loss_and_grad_device(x, y, "fp32")
        print(f"case={name} {x.shape} loss={rl:.6f} ssim={rs_:.6f}")
        for m, b, s in MODELS:
            l, _, ss, g = run(x, y, b, s)
            gr = float(np.linalg.norm(g - rg) / np.linalg.norm(rg))
            ok = gr <= 1e-4 and abs(l - rl) < 1e-5
            print(f"  {m:34s} loss_err={l - rl:+.2e} ssim_err={ss - rs_:+.2e} grad_l2rel={gr:.2e} "
                  f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
