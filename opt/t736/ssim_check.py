"""#736 (T7): fused L1 + D-SSIM loss and dL/dimage on device (LOSS_MODE=ssim, loss_bench.cpp
run_ssim, kernels/loss/ssim_*), on the SFPU fp32 filter of #720 (sfilt_compute.cpp).

  python3 opt/t736/ssim_check.py model                 T0 oracle of the sfilt algorithm (bit model,
                                                       Mac): hero + the 4 standard dumps
  python3 opt/t736/ssim_check.py prep <dir> [cases]    device inputs (image/target/consts)
  python3 opt/t736/ssim_check.py check <dir>           device loss + dL/dimage vs the T0 oracle

Cases: dump64 (T0 oracle dump), syncrop (loss_check syn, 256x384 crop), syn (1237x822, the timing
workload), bike_top / bike_mid (dssim_prop.py, need the bicycle JPG: Mac only), hero (T1 hero dump
#707: device-rule fp32 render of _DSC8679 vs the JPG, 1237x822: Mac only, HERO_NPZ).
Case dir: hw.txt "H W", image.f32 / target.f32 (H, W, 3), consts.f32 (2, 3, Hp, Wp): mu_y = G*y
and vy = G*(y y) - mu_y^2 (per-view constants, cached per GT image, here from the bit model).
The device writes grad.f32 (H, W, 3) and loss.txt (sum of its per-core partials + lambda).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "opt", "t690"), os.path.join(ROOT, "opt", "t714"),
          os.path.join(ROOT, "opt", "t720")):
    sys.path.insert(0, p)
import gsplat.train_loss as tl  # noqa: E402
from sfilt_check import F32, emulate, oracle_check  # noqa: E402

PROJ = os.environ.get("TTP_PROJECT_ROOT", "/Users/smarton/dev/gsplat_tt")
HERO_NPZ = os.environ.get("HERO_NPZ", os.path.join(PROJ, "tt-project/worktrees/t707/tmp/t707/hero.npz"))
JPG = os.environ.get("BIKE_IMG", os.path.join(PROJ, "scenes/mipnerf360/bicycle/images_4/_DSC8679.JPG"))
HERO_MD5 = "81c4a1983527e3a19eda3519d952db09"  # docs/t685/hero_meta.json image_md5_f32


def hero_case():
    import hashlib
    from PIL import Image
    if not (os.path.exists(HERO_NPZ) and os.path.exists(JPG)):
        print(f"[ORACLE] hero skipped: {HERO_NPZ} or {JPG} missing")
        return
    x = np.load(HERO_NPZ)["image"].astype(F32)
    assert hashlib.md5(x.tobytes()).hexdigest() == HERO_MD5, "hero image md5 differs from hero_meta.json"
    y = np.asarray(Image.open(JPG).convert("RGB"), F32) / F32(255)
    yield "hero", x, y


def all_cases(full_syn=False):
    from loss_check import cases
    from dssim_prop import bike_cases
    for name, x, y in cases():
        if name == "syn":
            yield "syncrop", x[:256, :384], y[:256, :384]
            if full_syn:
                yield "syn", x, y
        else:
            yield name, x, y
    yield from bike_cases()
    yield from hero_case()


def model():
    ok_all = True
    for name, x, y in all_cases():
        le, gr = oracle_check(x, y, lambda z: emulate(z, True))
        ok = gr <= 1e-4 and abs(le) < 1e-5
        ok_all &= ok
        print(f"[ORACLE] model=sfilt case={name} {x.shape} loss_err={le:+.2e} grad_l2rel={gr:.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[ORACLE] model overall {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def consts(y):
    Y = tl.pad_planes(y)
    mu_y, e_yy = emulate(Y, True), emulate(Y * Y, True)
    return np.stack([mu_y, (e_yy - mu_y * mu_y).astype(F32)]).astype(F32)


def prep(out, only=None):
    for name, x, y in all_cases(full_syn=True):
        if only and name not in only:
            continue
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "hw.txt"), "w") as f:
            f.write(f"{x.shape[0]} {x.shape[1]}\n")
        np.ascontiguousarray(x, F32).tofile(os.path.join(d, "image.f32"))
        np.ascontiguousarray(y, F32).tofile(os.path.join(d, "target.f32"))
        consts(y).tofile(os.path.join(d, "consts.f32"))
        print(f"prep {name} {x.shape}")


def device_check(d):
    """(loss_err, grad_l2rel, max_abs_err/max|g|) of the device outputs in case dir d vs the T0 oracle."""
    import torch
    from gsplat.train_oracle import loss_fn
    H, W = map(int, open(os.path.join(d, "hw.txt")).read().split())
    x = np.fromfile(os.path.join(d, "image.f32"), F32).reshape(H, W, 3)
    y = np.fromfile(os.path.join(d, "target.f32"), F32).reshape(H, W, 3)
    xi = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    lo = loss_fn(xi, torch.tensor(y, dtype=torch.float64))
    lo.backward()
    ref = xi.grad.numpy()
    g = np.fromfile(os.path.join(d, "out", "grad.f32"), F32).reshape(H, W, 3).astype(np.float64)
    loss = float(open(os.path.join(d, "out", "loss.txt")).read().split()[0])
    return loss - lo.item(), float(np.linalg.norm(g - ref) / np.linalg.norm(ref)), \
        float(np.abs(g - ref).max() / np.abs(ref).max()), bool(np.isfinite(g).all())


def check(out):
    ok_all, seen = True, 0
    for name in ("dump64", "syncrop", "bike_top", "bike_mid", "hero", "syn"):
        d = os.path.join(out, name)
        if not os.path.exists(os.path.join(d, "out", "grad.f32")):
            print(f"[ORACLE] device case={name} missing")
            if name in ("dump64", "syncrop", "syn"):
                ok_all = False
            continue
        seen += 1
        le, gr, mx, fin = device_check(d)
        ok = gr <= 1e-4 and abs(le) < 1e-5 and fin
        ok_all &= ok
        print(f"[ORACLE] device=ssim case={name} loss_err={le:+.2e} grad_l2rel={gr:.2e} "
              f"maxabs_rel={mx:.2e} finite={fin} {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[ORACLE] device overall {'PASS' if ok_all and seen else 'FAIL'} cases={seen}")
    return 0 if ok_all and seen else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "model":
        sys.exit(model())
    elif cmd == "prep":
        prep(sys.argv[2], sys.argv[3:] or None)
    elif cmd == "check":
        sys.exit(check(sys.argv[2]))
