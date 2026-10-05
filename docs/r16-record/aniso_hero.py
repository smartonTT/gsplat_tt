"""Precision analysis for a 16 B per-pair record (task #23).

Projects bicycle for the hero view (same math as src/gsplat_cpu/project.cpp),
then estimates the alpha error each candidate conic encoding introduces.
"""
import json, math, sys
import numpy as np

PLY = "scenes/bicycle.ply"
CAM = "benchmarks/cameras_v2.json"  # run from the repo root


def load_ply(path):
    with open(path, "rb") as f:
        props = []
        n = 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    g = lambda *ks: np.stack([data[:, col[k]] for k in ks], -1)
    means = g("x", "y", "z")
    scales = np.exp(g("scale_0", "scale_1", "scale_2"))
    q = g("rot_0", "rot_1", "rot_2", "rot_3")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    op = 1.0 / (1.0 + np.exp(-data[:, col["opacity"]]))
    sh = g("f_dc_0", "f_dc_1", "f_dc_2")
    color = np.clip(0.5 + 0.28209479177387814 * sh, 0.0, 1.0)
    return means, scales, q, op, color


def quat_to_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def main():
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    fov = cam["fov_deg"]
    view = sys.argv[1] if len(sys.argv) > 1 else "hero"
    c2w = np.array(cam["views"][view]["c2w"], np.float64)
    w2c = np.linalg.inv(c2w)
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(fov))
    means, scales, q, op, color = load_ply(PLY)
    N = means.shape[0]
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 0.01
    pc, z = pc[front], z[front]
    scales, q, op = scales[front], q[front], op[front]
    tanx = 0.5 * W / f
    lim = 1.3 * tanx
    tx = np.clip(pc[:, 0] / z, -lim, lim) * z
    ty = np.clip(pc[:, 1] / z, -lim, lim) * z
    j00 = f / z; j02 = -f * tx / (z * z); j11 = f / z; j12 = -f * ty / (z * z)
    R = quat_to_rot(q)
    M = R * scales[:, None, :]
    cov3 = M @ np.transpose(M, (0, 2, 1))
    Rc = w2c[:3, :3].astype(np.float32)
    covc = Rc[None] @ cov3 @ Rc.T[None]
    a = j00 * j00 * covc[:, 0, 0] + 2 * j00 * j02 * covc[:, 0, 2] + j02 * j02 * covc[:, 2, 2] + 0.3
    b = j00 * j11 * covc[:, 0, 1] + j00 * j12 * covc[:, 0, 2] + j02 * j11 * covc[:, 2, 1] + j02 * j12 * covc[:, 2, 2]
    c = j11 * j11 * covc[:, 1, 1] + 2 * j11 * j12 * covc[:, 1, 2] + j12 * j12 * covc[:, 2, 2] + 0.3
    mx = f * pc[:, 0] / z + 0.5 * W
    my = f * pc[:, 1] / z + 0.5 * H
    rx = np.ceil(3 * np.sqrt(a)); ry = np.ceil(3 * np.sqrt(c))
    valid = (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H) & (op > 1 / 255.0)
    a, b, c, mx, my, op, z = a[valid], b[valid], c[valid], mx[valid], my[valid], op[valid], z[valid]
    det = a * c - b * b
    ok = det > 0
    a, b, c, mx, my, op, det, z = a[ok], b[ok], c[ok], mx[ok], my[ok], op[ok], det[ok], z[ok]
    # device conic (power = A dx^2 + B dx dy + C dy^2), -0.5 folded
    A = (-0.5 * c / det).astype(np.float32)
    B = (b / det).astype(np.float32)
    C = (-0.5 * a / det).astype(np.float32)
    tr = a + c
    disc = np.sqrt(np.maximum((a - c) ** 2 / 4 + b * b, 0))
    lmax = tr / 2 + disc; lmin = tr / 2 - disc
    r = np.sqrt(lmax / lmin)
    # pairs per gaussian (32 px tiles, AABB) as weights
    tx0 = np.clip(np.floor((mx - rx[valid][ok]) / 32), 0, 31); tx1 = np.clip(np.floor((mx + rx[valid][ok]) / 32), 0, 31)
    ty0 = np.clip(np.floor((my - ry[valid][ok]) / 32), 0, 31); ty1 = np.clip(np.floor((my + ry[valid][ok]) / 32), 0, 31)
    npairs = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f"view={view} N={N} visible={A.size} pairs~{int(npairs.sum())} K={npairs.sum()/A.size:.2f}")
    for q_ in (1.5, 3, 5, 10, 20, 50, 100):
        sel = r > q_
        print(f"  aniso r>{q_:>5}: {sel.mean()*100:6.2f}% of gaussians, {npairs[sel].sum()/npairs.sum()*100:6.2f}% of pairs")
    print(f"  |A| range {np.abs(A).min():.3e}..{np.abs(A).max():.3e}  |B| max {np.abs(B).max():.3e}")
    print(f"  |mx| range {mx.min():.1f}..{mx.max():.1f}; |my| {my.min():.1f}..{my.max():.1f}")
    # --- error model: sample the ellipse at unit-Mahalanobis radii k in {0.5,1,1.5,2,2.5,3}
    # along 16 directions, evaluate alpha = op*exp(power) with fp32 vs quantized coeffs
    eig_major = np.stack([b, lmax - a], -1)  # eigenvector of cov for lmax
    nrm = np.linalg.norm(eig_major, axis=-1, keepdims=True)
    eig_major = np.where(nrm > 1e-12, eig_major / np.maximum(nrm, 1e-30), np.array([1.0, 0.0]))
    eig_minor = np.stack([-eig_major[:, 1], eig_major[:, 0]], -1)
    s1 = np.sqrt(lmax); s2 = np.sqrt(lmin)

    def fp16(x):
        return x.astype(np.float16).astype(np.float32)

    def enc_fp16(A, B, C):
        return fp16(A), fp16(B), fp16(C)

    def enc_chol16(A, B, C):
        # -power = |L^T d|^2 with M=[[-A,-B/2],[-B/2,-C]]
        l11 = np.sqrt(-A.astype(np.float64))
        l21 = (-B.astype(np.float64) / 2) / l11
        l22 = np.sqrt(np.maximum(-C.astype(np.float64) - l21 * l21, 0))
        l11, l21, l22 = fp16(l11.astype(np.float32)), fp16(l21.astype(np.float32)), fp16(l22.astype(np.float32))
        l11, l21, l22 = l11.astype(np.float64), l21.astype(np.float64), l22.astype(np.float64)
        return (-(l11 * l11)).astype(np.float32), (-(2 * l11 * l21)).astype(np.float32), (-(l21 * l21 + l22 * l22)).astype(np.float32)

    def enc_shexp(A, B, C, mbits=14):
        m = np.maximum(np.maximum(np.abs(A), np.abs(C)), np.abs(B) / 2)
        e = np.ceil(np.log2(m))
        s = 2.0 ** (e - mbits)
        return (np.round(A / s) * s).astype(np.float32), (np.round(B / s) * s).astype(np.float32), (np.round(C / s) * s).astype(np.float32)

    ks = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    angs = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    results = {}
    for name, enc in (("fp16_ABC", enc_fp16), ("chol_fp16", enc_chol16), ("shexp14", enc_shexp)):
        Aq, Bq, Cq = enc(A, B, C)
        worst = np.zeros(A.size, np.float64)
        for k in ks:
            for t in angs:
                u = k * np.cos(t) * s1; v = k * np.sin(t) * s2
                dx = u * eig_major[:, 0] + v * eig_minor[:, 0]
                dy = u * eig_major[:, 1] + v * eig_minor[:, 1]
                p0 = A * dx * dx + B * dx * dy + C * dy * dy
                p1 = Aq * dx * dx + Bq * dx * dy + Cq * dy * dy
                a0 = op * np.exp(np.minimum(p0, 0)); a1 = op * np.exp(np.minimum(p1, 0))
                worst = np.maximum(worst, np.abs(a1 - a0))
        # also check for loss of definiteness (hyperbola)
        detq = (Aq.astype(np.float64) * Cq - (Bq.astype(np.float64) ** 2) / 4)
        bad = (detq <= 0) | (Aq >= 0) | (Cq >= 0)
        w = npairs
        print(f"[{name}] non-PD: {bad.sum()} gaussians; alpha err (max over ellipse samples):")
        for thr in (1 / 1024, 1 / 510, 1 / 255, 4 / 255, 16 / 255):
            sel = worst > thr
            print(f"   > {thr*255:6.2f}/255: {sel.mean()*100:7.3f}% gaussians  {w[sel].sum()/w.sum()*100:7.3f}% pairs")
        print(f"   mean err (pair-weighted) {np.sum(worst*w)/w.sum()*255:.4f}/255, p99.9 {np.quantile(worst,0.999)*255:.3f}/255")


if __name__ == "__main__":
    main()
