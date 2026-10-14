"""#720: SFPU fp32 11-tap separable Gaussian (render/bench/kernels/loss/sfilt_compute.cpp) checks.

  python3 opt/t720/sfilt_check.py prep <dir>          inputs for loss_bench LOSS_MODE=sfilt
  python3 opt/t720/sfilt_check.py check <dir>         device outputs vs fp64 / bit model, then
                                                      D-SSIM loss + dL/dimage vs the T0 oracle

Cases (planes.txt "P Hp Wp", planes.f32 (P, Hp, Wp)):
  dump64   T0 oracle dump image/target (64x64): 24 planes = the 5 forward maps' inputs
           (x, y, x^2, y^2, xy) and the 3 backward inputs (d_mu, d_xx, d_xy, from the exact
           fp32 forward), 3 channels each
  syncrop  loss_check.py syn, 256x384 crop, same 24 planes
  syn1     syn 1237x822, x only (3 planes): the FPU split3 filter's workload (0.209 ms/view)
  syn5     syn, 5 forward maps (15 planes)       syn6  per training step: x, x^2, xy + 3 backward (18)
The device's operation order (per output, both passes, fp32):
  acc = (z[-5] + z[5]) g0; acc = (z[-4] + z[4]) g1 + acc; ...; acc = z[0] g5 + acc
emulate() reproduces it (fused or unfused multiply-add); a bitwise match means the oracle check
of the emulated loss/grad holds for the device too.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "opt", "t690"))
sys.path.insert(0, os.path.join(ROOT, "opt", "t714"))
import gsplat.train_loss as tl  # noqa: E402

F32 = np.float32
TAPS = tl.gauss_1d().astype(F32)  # g[0..10]; the device's g0..g5 = TAPS[0..5]


def emulate(z, fused=True):
    """The kernel's filter on (..., Hp, Wp) fp32 planes: H pass then V pass, zero padding."""
    def one(a, axis):
        a = np.moveaxis(a, axis, -1)
        n = a.shape[-1]
        p = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(5, 5)])
        sh = lambda k: p[..., 5 + k:5 + k + n]  # noqa: E731
        acc = None
        for k in range(5, -1, -1):
            s = (sh(-k) + sh(k)).astype(F32) if k else sh(0)
            g = TAPS[5 - k]
            if acc is None:
                acc = (s * g).astype(F32)
            elif fused:
                acc = (s.astype(np.float64) * np.float64(g) + acc.astype(np.float64)).astype(F32)
            else:
                acc = ((s * g).astype(F32) + acc).astype(F32)
        return np.moveaxis(acc, -1, axis)
    return one(one(np.asarray(z, F32), -1), -2)


def ref64(z, taps=None):
    """fp64 separable filter of (..., Hp, Wp) planes (taps: default the fp32 taps)."""
    t = np.asarray(TAPS if taps is None else taps, np.float64)
    z = np.asarray(z, np.float64)
    n0, n1 = z.shape[-2], z.shape[-1]
    p = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 0), (5, 5)])
    r = sum(t[i] * p[..., i:i + n1] for i in range(11))
    p = np.pad(r, [(0, 0)] * (r.ndim - 2) + [(5, 5), (0, 0)])
    return sum(t[i] * p[..., i:i + n0, :] for i in range(11))


def bwd_inputs(x, y, filt):
    """d_mu, d_xx, d_xy planes (3, Hp, Wp) of gsplat/train_loss.py loss_and_grad_device."""
    f32 = F32
    H, W = x.shape[:2]
    X, Y = tl.pad_planes(x), tl.pad_planes(y)
    mask = np.zeros(X.shape[1:], F32)
    mask[:H, :W] = 1
    mu_y, e_yy, mu_x, e_xx, e_xy = (filt(v) for v in (Y, Y * Y, X, X * X, X * Y))
    mxy = mu_x * mu_y
    A = f32(2) * mxy + f32(tl.C1)
    B = f32(2) * (e_xy - mxy) + f32(tl.C2)
    C = mu_x * mu_x + mu_y * mu_y + f32(tl.C1)
    D = (e_xx - mu_x * mu_x) + (e_yy - mu_y * mu_y) + f32(tl.C2)
    rB, rD = f32(1) / B, f32(1) / D
    S = (A * B) / (C * D)
    Sw = S * (f32(-tl.LAMBDA) / f32(3 * H * W) * mask)
    d_mu = Sw * (f32(2) * mu_y / A - f32(2) * mu_y * rB - f32(2) * mu_x / C + f32(2) * mu_x * rD)
    return X, Y, d_mu, -Sw * rD, f32(2) * Sw * rB


def case_planes():
    from loss_check import cases
    for name, x, y in cases():
        if name == "dump64":
            subs = [("dump64", x, y)]
        else:
            subs = [("syncrop", x[:256, :384], y[:256, :384])]
        for nm, a, b in subs:
            X, Y, dm, dxx, dxy = bwd_inputs(a, b, lambda v: tl.filt_tiles(v, "fp32"))
            yield nm, np.concatenate([X, Y, X * X, Y * Y, X * Y, dm, dxx, dxy]).astype(F32)
        if name == "syn":
            X, Y, dm, dxx, dxy = bwd_inputs(x, y, lambda v: tl.filt_tiles(v, "fp32"))
            yield "syn1", X
            yield "syn5", np.concatenate([X, Y, X * X, Y * Y, X * Y]).astype(F32)
            yield "syn6", np.concatenate([X, X * X, X * Y, dm, dxx, dxy]).astype(F32)


def prep(out, only=None):
    for name, pl in case_planes():
        if only and name not in only:
            continue
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "planes.txt"), "w") as f:
            f.write("%d %d %d\n" % pl.shape)
        np.ascontiguousarray(pl, F32).tofile(os.path.join(d, "planes.f32"))
        print(f"prep {name} planes={pl.shape}")


def oracle_check(x, y, filt):
    """(loss_err, grad_l2rel) of the device algorithm with `filt` vs the T0 oracle (torch fp64)."""
    import torch
    from gsplat.train_oracle import loss_fn
    xi = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    lo = loss_fn(xi, torch.tensor(y, dtype=torch.float64))
    lo.backward()
    ref_g = xi.grad.numpy()
    ex = tl.filt_tiles
    tl.filt_tiles = lambda z, *a, **k: filt(z)
    try:
        loss, _, _, g = tl.loss_and_grad_device(x, y, "fp32")
    finally:
        tl.filt_tiles = ex
    return loss - lo.item(), float(np.linalg.norm(g - ref_g) / np.linalg.norm(ref_g))


def check(out):
    planes = dict(case_planes())
    ok_all = True
    fused_ok = {}
    for name in ("dump64", "syncrop", "syn1"):
        o = os.path.join(out, name, "out", "planes_out.f32")
        if not os.path.exists(o):
            print(f"[CHECK] case={name} missing {o}")
            ok_all = False
            continue
        z = planes[name]
        dev = np.fromfile(o, F32).reshape(z.shape)
        r = ref64(z)
        r_true = ref64(z, tl.gauss_1d())
        em = {v: emulate(z, v == "fused") for v in ("fused", "unfused")}
        err = dev.astype(np.float64) - r
        l2 = float(np.linalg.norm(err) / np.linalg.norm(r))
        big = np.abs(r) > 1e-3 * np.abs(r).max()
        rel = np.abs(err[big]) / np.abs(r[big])
        bias = float(np.mean(err[big] / np.abs(r[big])))
        l2t = float(np.linalg.norm(dev - r_true) / np.linalg.norm(r_true))
        eml2 = float(np.linalg.norm(em["fused"] - r) / np.linalg.norm(r))
        bits = {v: float(np.mean(em[v].view(np.uint32) == dev.view(np.uint32))) for v in em}
        ulp = np.abs(em["fused"].view(np.int32).astype(np.int64) - dev.view(np.int32).astype(np.int64))
        per = [float(np.linalg.norm(err[p]) / max(np.linalg.norm(r[p]), 1e-300)) for p in range(z.shape[0])]
        ok = l2 < 2e-6 and float(rel.max()) < 1e-5
        ok_all &= ok
        fused_ok[name] = bits["fused"]
        print(f"[CHECK] mode=sfilt case={name} planes={z.shape[0]} vs_fp64_fp32taps_l2rel={l2:.2e} "
              f"maxrel={float(rel.max()):.2e} bias={bias:+.2e} vs_fp64_truetaps_l2rel={l2t:.2e} "
              f"model_vs_fp64_l2rel={eml2:.2e} bitmatch_fused={bits['fused']:.6f} "
              f"bitmatch_unfused={bits['unfused']:.6f} max_ulp_vs_fused={int(ulp.max())} "
              f"worst_plane_l2rel={max(per):.2e} {'PASS' if ok else 'FAIL'}")
    # D-SSIM loss + dL/dimage vs the T0 oracle, with the device filter (its bit model).
    from loss_check import cases
    try:
        from dssim_prop import bike_cases
    except Exception:  # noqa: BLE001
        bike_cases = lambda: iter(())  # noqa: E731
    import itertools
    for name, x, y in itertools.chain(cases(), bike_cases()):
        if name == "syn":
            name, x, y = "syncrop", x[:256, :384], y[:256, :384]
        le, gr = oracle_check(x, y, lambda z: emulate(z, True))
        le0, gr0 = oracle_check(x, y, lambda z: ref64(z, tl.gauss_1d()).astype(F32))
        ok = gr <= 1e-4 and abs(le) < 1e-5
        ok_all &= ok
        print(f"[ORACLE] case={name} {x.shape} sfpu_model loss_err={le:+.2e} grad_l2rel={gr:.2e} "
              f"(exact-filter fp32 algorithm: loss_err={le0:+.2e} grad_l2rel={gr0:.2e}) "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"[CHECK] overall {'PASS' if ok_all else 'FAIL'} bitmatch_fused={fused_ok}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3:] or None)
    elif sys.argv[1] == "check":
        sys.exit(check(sys.argv[2]))
    elif sys.argv[1] == "model":  # oracle check of the bit model alone (no device)
        sys.exit(check(sys.argv[2] if len(sys.argv) > 2 else "/nonexistent"))
