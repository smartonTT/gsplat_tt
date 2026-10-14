#!/usr/bin/env python3
"""Task #690: inputs for render/bench/loss_bench and the check of its outputs.

  loss_check.py prep <dir>            writes <dir>/{dump64,syn}/ {hw.txt,image.f32,target.f32}
  loss_check.py check <dir> [--mode l1]   compares <dir>/<case>/out/{grad.f32,loss.txt}
  loss_check.py check <dir> --mode filt   compares <dir>/<case>/out/filt.f32 (G * image, #706)

Cases: dump64 = the T0 oracle dump t0_sh0_64x64_n50 (image/target; the full-loss grad of the
oracle is checked by the model in tests/unit/test_train_loss.py); syn = deterministic 1237x822
pair (numpy only, no dataset image leaves the Mac) with exact-equal patches so sign(0) = 0 is
exercised. The L1 step is checked against gsplat/train_loss.loss_and_grad_l1 (fp32 elementwise,
expected bit-exact grad, loss to fp32 summation error).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def cases():
    d = np.load(os.path.join(ROOT, "benchmarks/train_oracle/t0_sh0_64x64_n50.npz"))
    yield "dump64", d["image"].astype(np.float32), d["target"].astype(np.float32)
    H, W = 822, 1237
    rs = np.random.RandomState(690)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = 0.5 + 0.4 * np.sin(xx[..., None] * np.float32(0.013) * np.arange(1, 4, dtype=np.float32)
                           + yy[..., None] * np.float32(0.007))
    t = t.astype(np.float32)
    x = np.clip(t + rs.normal(0, 0.05, t.shape).astype(np.float32), 0, 1).astype(np.float32)
    x[100:300, 200:600] = t[100:300, 200:600]  # exact matches: grad 0
    yield "syn", x, t


def prep(out):
    for name, x, y in cases():
        p = os.path.join(out, name)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "hw.txt"), "w") as f:
            f.write(f"{x.shape[0]} {x.shape[1]}\n")
        np.ascontiguousarray(x, np.float32).tofile(os.path.join(p, "image.f32"))
        np.ascontiguousarray(y, np.float32).tofile(os.path.join(p, "target.f32"))
        print(f"prep {name} {x.shape}")


def sep_filter(x, taps):
    """fp64 separable filter of (H, W, C) with 11 taps, zero padding (the device's G * x)."""
    r = len(taps) // 2
    H, W = x.shape[:2]
    xp = np.pad(x, ((r, r), (r, r), (0, 0)))
    rows = sum(taps[i] * xp[:, i:i + W] for i in range(len(taps)))
    return sum(taps[i] * rows[i:i + H] for i in range(len(taps)))


def check_filt(out):
    """#706: device G * x (split3 banded matmuls) against the fp64 filter and the split3 model."""
    import glob
    from gsplat.train_loss import filt_tiles, pad_planes
    ok = True
    for name, x, _ in cases():
        H, W = x.shape[:2]
        xp = pad_planes(x)
        ref = np.moveaxis(filt_tiles(xp, "fp32")[:, :H, :W], 0, -1)
        mod = np.moveaxis(filt_tiles(xp, "split3", "trunc")[:, :H, :W], 0, -1)
        outs = sorted(glob.glob(os.path.join(out, name, "out*")))
        if not outs:
            print(f"[CHECK] mode=filt case={name} no output dir FAIL")
            ok = False
        for o in outs:
            f = os.path.join(o, "filt.f32")
            if not os.path.exists(f):
                print(f"[CHECK] mode=filt case={name} out={os.path.basename(o)} no filt.f32 FAIL")
                ok = False
                continue
            g = np.fromfile(f, np.float32).reshape(x.shape)
            tag = os.path.basename(o)
            tf = os.path.join(o, "taps.txt")
            if os.path.exists(tf):  # LOSS_RDIAG=t<preset> (#714): the device's own taps, fp64 reference
                taps = np.loadtxt(tf, dtype=np.float64)
                dref = sep_filter(x.astype(np.float64), taps)
                rel = float(np.linalg.norm(g - dref) / np.linalg.norm(dref))
                nz = np.abs(dref) > 0
                bias = float(np.mean((g - dref)[nz] / dref[nz]))
                good = bool(np.isfinite(g).all() and rel < 2e-6)
                ok &= good
                print(f"[CHECK] mode=filt case={name} out={tag} taps=device vs_fp64_taps_l2rel={rel:.2e} "
                      f"mean_rel_bias={bias:+.2e} {'PASS' if good else 'FAIL'}")
                continue
            if "-r" in tag:  # LOSS_RDIAG=c run: R[0] = c I, so G * x = c^2 x (split3 error ~2^-18)
                c = float(np.float32(tag.split("-r", 1)[1]))
                dref = c * c * x.astype(np.float64)
                if c < 0:  # box: 11 taps of |c| each way, zero padding -> c^2 * 11x11 window sum
                    xp = np.pad(x.astype(np.float64), ((5, 5), (5, 5), (0, 0)))
                    cs = np.cumsum(np.cumsum(np.pad(xp, ((1, 0), (1, 0), (0, 0))), 0), 1)
                    H, W = x.shape[:2]
                    dref = c * c * (cs[11:11 + H, 11:11 + W] - cs[:H, 11:11 + W]
                                    - cs[11:11 + H, :W] + cs[:H, :W])
                rel = float(np.linalg.norm(g - dref) / np.linalg.norm(dref))
                bias = float(np.mean((g - dref)[dref != 0] / dref[dref != 0]))
                good = bool(np.isfinite(g).all() and rel < 2e-6)
                ok &= good
                print(f"[CHECK] mode=filt case={name} out={tag} rdiag={c!r} vs_c2x_l2rel={rel:.2e} "
                      f"mean_rel_bias={bias:+.2e} exact_frac={float(np.mean(g == dref)):.3f} "
                      f"{'PASS' if good else 'FAIL'}")
                continue
            rel = float(np.linalg.norm(g - ref) / np.linalg.norm(ref))
            mx = float(np.abs(g - ref).max())
            mrel = float(np.linalg.norm(g - mod) / np.linalg.norm(ref))
            good = bool(np.isfinite(g).all() and rel < 2e-6 and mx < 1e-5)
            ok &= good
            print(f"[CHECK] mode=filt case={name} out={os.path.basename(o)} vs_fp64_l2rel={rel:.2e} "
                  f"vs_fp64_maxabs={mx:.2e} vs_split3_model_l2rel={mrel:.2e} "
                  f"model_vs_fp64_l2rel={float(np.linalg.norm(mod - ref) / np.linalg.norm(ref)):.2e} "
                  f"{'PASS' if good else 'FAIL'}")
    return ok


def check(out, mode="l1"):
    """Checks every <out>/<case>/out*/ run dir (one per bench variant, e.g. out-u0, out-u1)."""
    import glob
    if mode == "filt":
        return check_filt(out)
    from gsplat.train_loss import loss_and_grad_l1, loss_and_grad_device
    ok = True
    for name, x, y in cases():
        if mode == "l1":
            rl, rg = loss_and_grad_l1(x, y)
        else:
            rl, _, _, rg = loss_and_grad_device(x, y, scheme="split3")
        outs = sorted(glob.glob(os.path.join(out, name, "out*")))
        if not outs:
            print(f"[CHECK] case={name} no output dir FAIL")
            ok = False
        for o in outs:
            tag = os.path.basename(o)
            if not os.path.exists(os.path.join(o, "grad.f32")):
                print(f"[CHECK] case={name} out={tag} no grad.f32 (run did not finish) FAIL")
                ok = False
                continue
            g = np.fromfile(os.path.join(o, "grad.f32"), np.float32).reshape(x.shape)
            loss = float(open(os.path.join(o, "loss.txt")).read().split()[0])
            err = np.abs(g - rg)
            rel = float(np.linalg.norm(g - rg) / max(np.linalg.norm(rg), 1e-30))
            lerr = abs(loss - rl) / max(abs(rl), 1e-30)
            tol_g = 0.0 if mode == "l1" else 1e-4
            good = rel <= tol_g + 1e-12 and lerr < 1e-5 and np.isfinite(g).all()
            ok &= good
            print(f"[CHECK] mode={mode} case={name} out={tag} shape={x.shape} loss={loss:.9g} ref={rl:.9g} "
                  f"loss_rel={lerr:.2e} grad_l2rel={rel:.2e} grad_maxabs={err.max():.3e} "
                  f"mismatch={int((err > 0).sum())} {'PASS' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "prep":
        prep(d)
    else:
        mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "l1"
        sys.exit(0 if check(d, mode) else 1)
