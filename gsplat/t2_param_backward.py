"""T2 reference: fused segment reduce + projection/SH0 backward + sparse Adam + pfwc epilogue.

Task #689, docs/training-plan/PLAN-single-chip.md (T2). A float32 numpy model of the one
device pass, written lane-wise (every op is an elementwise op over Gaussians, as the SFPU
kernel runs it over 1024-Gaussian SoA tiles; no matrix library):

  1. segmented reduce of T1's Gaussian-major slots (docs/training-plan/T1-blend-backward.md
     §5): g2d[g] = sum_{j < counts[g]} slots[P[g] + j], j ascending (deterministic);
  2. projection forward recompute (pc, J, cov3d, cov2d, conic, colour clamp, visibility)
     from the fp32 params, then the analytic backward to the optimizer-space params
     (means, log_scales, quats, opacity_logits, sh band 0);
  3. Adam: sparse (visible rows only, gsplat SelectiveAdam / T0 `sparse_adam_step`) by
     default, dense with `dense=True`; fp32 params and math, moments stored bf16 (RNE)
     or fp32;
  4. epilogue: pfwc's input streams from the updated params, SoA padded to 1024
     (render/host/pfwc_device.cpp: means mx,my,mz + cov3d unique c00,c01,c02,c11,c12,c22;
     gather_visible: scene_opacities and scene_col_r/g/b).

Check against the T0 oracle (gsplat/train_oracle.py dumps):

    python3 -m gsplat.t2_param_backward benchmarks/train_oracle/t0_sh0_64x64_n50.npz

Bars: param grads rel 1e-2 (train_oracle.rel_error), Adam outputs rel 1e-3.
Lossless: no approximation beyond fp32 and the oracle's own bf16 moment storage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

SH_C0 = np.float32(0.28209479177387814)
PARAM_NAMES = ("means", "log_scales", "quats", "opacity_logits", "sh")
ELEMS_PER_TILE = 1024
PFWC_STREAMS = ("mx", "my", "mz", "c00", "c01", "c02", "c11", "c12", "c22",
                "opacity", "col_r", "col_g", "col_b")
f32 = np.float32


def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 (round to nearest even) -> fp32, as torch's .to(bfloat16)."""
    u = np.ascontiguousarray(x, f32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(f32)


def segment_reduce(slots: np.ndarray, P: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(N, 9) fp32 per-Gaussian sums of the slot rows, j ascending.

    Rank-major as on device: plane k holds slot P[g] + k of every Gaussian with
    counts[g] > k, added into the accumulator in k order (same fp32 sum order)."""
    N = P.shape[0]
    acc = np.zeros((N, slots.shape[1]), f32)
    cmax = int(counts.max()) if N else 0
    for k in range(cmax):
        sel = counts > k
        acc[sel] += slots[P[sel] + k]
    return acc


def _cam(w2c, K, ft=f32):
    R = np.asarray(w2c, ft)[:3, :3]
    t = np.asarray(w2c, ft)[:3, 3]
    Kf = np.asarray(K, ft)
    return R, t, Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]


def _quat_rot(q):
    """Unit quaternion (w, x, y, z) lanes -> 9 rotation-matrix lanes (row major)."""
    w, x, y, z = q
    ft = q[0].dtype.type
    two, one = ft(2), ft(1)
    return [one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y),
            two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x),
            two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y)]


def forward_lanes(params: dict, w2c, K, H: int, W: int, near: float = 0.2,
                  lowpass: float = 0.3, ft=f32, jclamp: float | None = 1.3) -> dict:
    """Per-Gaussian projection forward (fp32 lanes; ft=np.float64 for the conditioning
    check), the values the backward reuses."""
    R, t, fx, fy, cx, cy = _cam(w2c, K, ft)
    m = np.asarray(params["means"], ft)
    pc = [R[i, 0] * m[:, 0] + R[i, 1] * m[:, 1] + R[i, 2] * m[:, 2] + t[i] for i in range(3)]
    x, y, z = pc
    valid = z > ft(near)
    iz = ft(1) / np.where(valid, z, ft(1))
    mx, my = fx * x * iz + cx, fy * y * iz + cy
    q = np.asarray(params["quats"], ft)
    qinv = ft(1) / np.sqrt((q * q).sum(1, dtype=ft))
    qn = [q[:, i] * qinv for i in range(4)]
    Rq = _quat_rot(qn)
    s = np.exp(np.asarray(params["log_scales"], ft))
    M = [Rq[3 * i + j] * s[:, j] for i in range(3) for j in range(3)]       # row major
    C3 = {(i, k): M[3 * i] * M[3 * k] + M[3 * i + 1] * M[3 * k + 1] + M[3 * i + 2] * M[3 * k + 2]
          for i in range(3) for k in range(i, 3)}
    c3 = lambda i, k: C3[(min(i, k), max(i, k))]  # noqa: E731
    jx, jy = x, y
    if jclamp:
        # 3DGS/gsplat Jacobian clamp (#786, as train_oracle.RenderConfig.jclamp): J sees the
        # camera-space mean clamped to +-jclamp tan(half FOV) z per axis.
        lx, ly = ft(jclamp) * ft(W) / (ft(2) * fx) * z, ft(jclamp) * ft(H) / (ft(2) * fy) * z
        jx, jy = np.minimum(np.maximum(x, -lx), lx), np.minimum(np.maximum(y, -ly), ly)
    j00, j02 = fx * iz, -fx * jx * iz * iz
    j11, j12 = fy * iz, -fy * jy * iz * iz
    T0 = [j00 * R[0, k] + j02 * R[2, k] for k in range(3)]
    T1 = [j11 * R[1, k] + j12 * R[2, k] for k in range(3)]
    CT0 = [c3(i, 0) * T0[0] + c3(i, 1) * T0[1] + c3(i, 2) * T0[2] for i in range(3)]
    CT1 = [c3(i, 0) * T1[0] + c3(i, 1) * T1[1] + c3(i, 2) * T1[2] for i in range(3)]
    a = T0[0] * CT0[0] + T0[1] * CT0[1] + T0[2] * CT0[2] + ft(lowpass)
    b = T1[0] * CT0[0] + T1[1] * CT0[1] + T1[2] * CT0[2]
    c = T1[0] * CT1[0] + T1[1] * CT1[1] + T1[2] * CT1[2] + ft(lowpass)
    idet = ft(1) / (a * c - b * b)
    A, B, Cc = c * idet, -b * idet, a * idet
    rx = np.ceil(ft(3) * np.sqrt(np.maximum(a, ft(0))))
    ry = np.ceil(ft(3) * np.sqrt(np.maximum(c, ft(0))))
    visible = valid & (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H)
    ol = np.asarray(params["opacity_logits"], ft)
    op = ft(1) / (ft(1) + np.exp(-ol))
    raw = SH_C0 * np.asarray(params["sh"], ft)[:, 0] + ft(0.5)
    return dict(R=R, fx=fx, fy=fy, x=x, y=y, jx=jx, jy=jy, z=z, iz=iz, valid=valid, visible=visible,
                qn=qn, qinv=qinv, Rq=Rq, s=s, M=M, c3=c3, T0=T0, T1=T1, A=A, B=B, C=Cc,
                op=op, raw=raw, mx=mx, my=my, j00=j00, j02=j02, j11=j11, j12=j12)


def projection_backward_sh0(params: dict, g2d: np.ndarray, w2c, K, H: int, W: int,
                            near: float = 0.2, lowpass: float = 0.3,
                            ft=f32, jclamp: float | None = 1.3) -> tuple[dict, np.ndarray]:
    """g2d (N, 9) = (dmx, dmy, dA, dB, dC, dopacity, dr, dg, db) -> optimizer-space grads
    (fp32, same shapes as params; sh bands >= 1 get zero) and the visibility mask."""
    F = forward_lanes(params, w2c, K, H, W, near, lowpass, ft, jclamp)
    g = np.asarray(g2d, ft)
    gmx, gmy, gA, gB, gC, gop = (g[:, i] for i in range(6))
    A, B, C = F["A"], F["B"], F["C"]
    # conic = inv(cov2d): dSigma = -Q G Q, b appears twice in Sigma.
    ga = -(A * A * gA + A * B * gB + B * B * gC)
    gc = -(B * B * gA + B * C * gB + C * C * gC)
    gb = -(ft(2) * A * B * gA + (A * C + B * B) * gB + ft(2) * B * C * gC)
    T0, T1, c3, R = F["T0"], F["T1"], F["c3"], F["R"]
    hb = ft(0.5) * gb
    # dL/dcov3d = T^T Gs T (full symmetric), Gs = [[ga, gb/2], [gb/2, gc]].
    G3 = {(i, k): ga * T0[i] * T0[k] + hb * (T0[i] * T1[k] + T1[i] * T0[k]) + gc * T1[i] * T1[k]
          for i in range(3) for k in range(3)}
    # dL/dT = 2 Gs T C3.
    TC0 = [T0[0] * c3(0, k) + T0[1] * c3(1, k) + T0[2] * c3(2, k) for k in range(3)]
    TC1 = [T1[0] * c3(0, k) + T1[1] * c3(1, k) + T1[2] * c3(2, k) for k in range(3)]
    dT0 = [ft(2) * (ga * TC0[k] + hb * TC1[k]) for k in range(3)]
    dT1 = [ft(2) * (hb * TC0[k] + gc * TC1[k]) for k in range(3)]
    gj00 = dT0[0] * R[0, 0] + dT0[1] * R[0, 1] + dT0[2] * R[0, 2]
    gj02 = dT0[0] * R[2, 0] + dT0[1] * R[2, 1] + dT0[2] * R[2, 2]
    gj11 = dT1[0] * R[1, 0] + dT1[1] * R[1, 1] + dT1[2] * R[1, 2]
    gj12 = dT1[0] * R[2, 0] + dT1[1] * R[2, 1] + dT1[2] * R[2, 2]
    fx, fy, x, y, iz = F["fx"], F["fy"], F["x"], F["y"], F["iz"]
    iz2 = iz * iz
    # j02 = -fx jx / z^2: unclamped jx = x gives d/dx and 2 fx x / z^3 d/dz; clamped
    # jx = +-lim z gives no x term and fx jx / z^3 d/dz (gsplat persp_proj_vjp).
    jx, jy = F["jx"], F["jy"]
    ux, uy = jx == x, jy == y
    gx = gmx * fx * iz - np.where(ux, gj02 * fx * iz2, ft(0))
    gy = gmy * fy * iz - np.where(uy, gj12 * fy * iz2, ft(0))
    gz = (-(gmx * fx * x + gmy * fy * y + gj00 * fx + gj11 * fy) * iz2
          + (np.where(ux, ft(2), ft(1)) * gj02 * fx * jx
             + np.where(uy, ft(2), ft(1)) * gj12 * fy * jy) * iz2 * iz)
    v = F["valid"]
    gx, gy, gz = (np.where(v, a, ft(0)) for a in (gx, gy, gz))
    gmeans = np.stack([R[0, k] * gx + R[1, k] * gy + R[2, k] * gz for k in range(3)], 1)
    # cov3d = M M^T: dM = 2 G3 M; M = Rq diag(s).
    M, s, Rq = F["M"], F["s"], F["Rq"]
    dM = [ft(2) * (G3[(i, 0)] * M[j] + G3[(i, 1)] * M[3 + j] + G3[(i, 2)] * M[6 + j])
          for i in range(3) for j in range(3)]
    gls = np.stack([(dM[j] * Rq[j] + dM[3 + j] * Rq[3 + j] + dM[6 + j] * Rq[6 + j]) * s[:, j]
                    for j in range(3)], 1)
    gR = [dM[3 * i + j] * s[:, j] for i in range(3) for j in range(3)]
    w, qx, qy, qz = F["qn"]
    g00, g01, g02, g10, g11, g12, g20, g21, g22 = gR
    two = ft(2)
    gqn = [two * (-qz * g01 + qy * g02 + qz * g10 - qx * g12 - qy * g20 + qx * g21),
           two * (qy * g01 + qz * g02 + qy * g10 - two * qx * g11 - w * g12 + qz * g20
                  + w * g21 - two * qx * g22),
           two * (-two * qy * g00 + qx * g01 + w * g02 + qx * g10 + qz * g12 - w * g20
                  + qz * g21 - two * qy * g22),
           two * (-two * qz * g00 - w * g01 + qx * g02 + w * g10 - two * qz * g11 + qy * g12
                  + qx * g20 + qy * g21)]
    dot = w * gqn[0] + qx * gqn[1] + qy * gqn[2] + qz * gqn[3]
    gq = np.stack([(gqn[i] - F["qn"][i] * dot) * F["qinv"] for i in range(4)], 1)
    op = F["op"]
    gol = gop * op * (ft(1) - op)
    raw = F["raw"]
    # 3DGS/gsplat training colour: max(SH + 0.5, 0), no upper clamp (decision 2026-10-14, #763).
    keep = raw >= ft(0)
    gsh = np.zeros_like(np.asarray(params["sh"], ft))
    gsh[:, 0] = np.where(keep, SH_C0 * g[:, 6:9], ft(0))
    grads = {"means": gmeans, "log_scales": gls, "quats": gq, "opacity_logits": gol, "sh": gsh}
    # A row with no slots has exactly zero grads (the device writes 0); without this a degenerate
    # cov2d (det 0, idet inf; 4 rows of the #750 hero view) turns 0 * inf into NaN.
    zero = ~(g != 0).any(1)
    if zero.any():
        grads = {k: np.where(zero.reshape((-1,) + (1,) * (a.ndim - 1)), ft(0), a).astype(ft)
                 for k, a in grads.items()}
    return grads, F["visible"]


def adam_step(params: dict, grads: dict, m: dict, v: dict, step: int, rows: np.ndarray,
              lr: dict, betas=(0.9, 0.999), eps: float = 1e-15, bf16_moments: bool = True):
    """One Adam step over `rows` (visible mask for sparse, all-True for dense). fp32 math;
    moments stored bf16 (RNE) when bf16_moments. Same arithmetic as train_oracle's
    sparse_adam_step. Returns new (params, m, v)."""
    b1, b2 = f32(betas[0]), f32(betas[1])
    t = step + 1
    bc1, bc2 = 1.0 - betas[0] ** t, 1.0 - betas[1] ** t
    isq_bc2 = f32(1.0 / math.sqrt(bc2))
    out_p, out_m, out_v = {}, {}, {}
    for name in PARAM_NAMES:
        p = np.array(params[name], f32)
        mm = np.array(m[name], f32)
        vv = np.array(v[name], f32)
        gg = np.asarray(grads[name], f32)
        if name == "sh":
            ss = np.full((1, p.shape[1], 1), f32(lr["sh_rest"] / bc1), f32)
            ss[:, 0] = f32(lr["sh_dc"] / bc1)
            ss = np.broadcast_to(ss, p.shape)[rows]
        else:
            ss = f32(lr[name] / bc1)
        m1 = b1 * mm[rows] + (f32(1) - b1) * gg[rows]
        v1 = b2 * vv[rows] + (f32(1) - b2) * gg[rows] * gg[rows]
        den = np.sqrt(v1) * isq_bc2 + f32(eps)
        p[rows] = p[rows] - ss * (m1 / den)
        mm[rows] = bf16_round(m1) if bf16_moments else m1
        vv[rows] = bf16_round(v1) if bf16_moments else v1
        out_p[name], out_m[name], out_v[name] = p, mm, vv
    return out_p, out_m, out_v


def pfwc_inputs(params: dict) -> dict:
    """Epilogue: pfwc / gather_visible input streams (fp32 SoA, padded to 1024) from params."""
    F = forward_lanes(params, np.eye(4), np.eye(3), 1, 1)
    m = np.asarray(params["means"], f32)
    N = m.shape[0]
    c3 = F["c3"]
    col = np.maximum(F["raw"], f32(0))      # training colour: lower clamp only (3DGS/gsplat)
    vals = [m[:, 0], m[:, 1], m[:, 2], c3(0, 0), c3(0, 1), c3(0, 2), c3(1, 1), c3(1, 2),
            c3(2, 2), F["op"], col[:, 0], col[:, 1], col[:, 2]]
    pad = -(-N // ELEMS_PER_TILE) * ELEMS_PER_TILE
    out = {}
    for k, a in zip(PFWC_STREAMS, vals):
        buf = np.zeros(pad, f32)
        buf[:N] = a
        out[k] = buf
    return out


def t2_step(params: dict, slots: np.ndarray, P: np.ndarray, counts: np.ndarray, w2c, K,
            H: int, W: int, m: dict, v: dict, step: int, lr: dict, betas=(0.9, 0.999),
            eps: float = 1e-15, dense: bool = False, bf16_moments: bool = True) -> dict:
    """The whole T2 pass. Returns grads, visible, new params/m/v and the pfwc streams."""
    g2d = segment_reduce(slots, P, counts)
    grads, visible = projection_backward_sh0(params, g2d, w2c, K, H, W)
    rows = np.ones_like(visible) if dense else visible
    p1, m1, v1 = adam_step(params, grads, m, v, step, rows, lr, betas, eps, bf16_moments)
    return {"g2d": g2d, "grads": grads, "visible": visible, "params": p1, "m": m1, "v": v1,
            "pfwc": pfwc_inputs(p1)}


# ---------------------------------------------------------------- oracle check

def rel_error(ad, ref, floor_frac: float = 1e-4) -> float:
    a, f = np.asarray(ad, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    scale = np.abs(f).max() if f.size else 0.0
    if scale == 0.0:
        return float(np.abs(a).max()) if a.size else 0.0
    return float((np.abs(a - f) / np.maximum(np.abs(f), floor_frac * scale)).max())


def oracle_slots(d) -> dict:
    """T1's slots for a T0 dump (device-decomposition blend backward, frozen live masks)."""
    from gsplat import blend_backward_ref as bb
    cfgd = json.loads(str(d["cfg_json"]))
    cfg = bb.BlendCfg(cfgd["alpha_max"], cfgd["alpha_min"], cfgd["t_min"],
                      tuple(cfgd["background"]))
    live = np.asarray(d["live"], bool).reshape(-1, *np.shape(d["image"])[:2])
    return bb.blend_backward(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"],
                             d["valid"], d["image"], d["dL_dimage"], cfg, frozen_live=live)


def check(d, grad_bar: float = 1e-2, adam_bar: float = 1e-3) -> dict:
    """Run T2 on a T0 dump: grads vs grad_* (rel grad_bar), Adam vs adam_bf16_* / adam_fp32_*
    (rel adam_bar). Adam is checked twice: fed the oracle's grads (isolates the optimizer)
    and end to end from the slots."""
    if int(d["sh_degree"]) != 0:
        raise ValueError("T2 covers SH degree 0 (SH3 is T4)")
    H, W = int(d["H"]), int(d["W"])
    params = {k: np.asarray(d[f"param_{k}"], f32) for k in PARAM_NAMES}
    bw = oracle_slots(d)
    lr = json.loads(str(d["adam_lr_json"]))
    betas = tuple(float(b) for b in d["adam_betas"])
    eps, step = float(d["adam_eps"]), int(d["adam_step_in"])
    rep = {"n_slots": int(bw["slots"].shape[0]), "max_count": int(bw["counts"].max())}
    out = t2_step(params, bw["slots"], bw["P"], bw["counts"], d["w2c"], d["K"], H, W,
                  {k: d[f"adam_bf16_m_in_{k}"] for k in PARAM_NAMES},
                  {k: d[f"adam_bf16_v_in_{k}"] for k in PARAM_NAMES}, step, lr, betas, eps)
    rep["visible_match"] = bool((out["visible"] == np.asarray(d["visible"], bool)).all())
    rep["grads"] = {k: rel_error(out["grads"][k], d[f"grad_{k}"]) for k in PARAM_NAMES}
    vis = np.asarray(d["visible"], bool)
    ograds = {k: np.asarray(d[f"grad_{k}"], f32) for k in PARAM_NAMES}
    for tag, bf in (("bf16", True), ("fp32", False)):
        mi = {k: d[f"adam_{tag}_m_in_{k}" if bf else f"adam_m_in_{k}"] for k in PARAM_NAMES}
        vi = {k: d[f"adam_{tag}_v_in_{k}" if bf else f"adam_v_in_{k}"] for k in PARAM_NAMES}
        p1, m1, v1 = adam_step(params, ograds, mi, vi, step, vis, lr, betas, eps, bf)
        rep[f"adam_{tag}"] = {f"{w}_{k}": rel_error(a[k], d[f"adam_{tag}_{w}_{k}"])
                              for w, a in (("param", p1), ("m", m1), ("v", v1))
                              for k in PARAM_NAMES}
        # The update itself (param_out - param_in), not only the param it lands on.
        rep[f"adam_{tag}"].update({f"delta_{k}": rel_error(
            p1[k] - params[k], np.asarray(d[f"adam_{tag}_param_{k}"], f32) - params[k])
            for k in PARAM_NAMES})
    rep["e2e_bf16_param"] = {k: rel_error(out["params"][k], d[f"adam_bf16_param_{k}"])
                             for k in PARAM_NAMES}
    pf = out["pfwc"]
    rep["pfwc_opacity_rel"] = rel_error(pf["opacity"][:params["means"].shape[0]],
                                        1 / (1 + np.exp(-out["params"]["opacity_logits"])))
    rep["worst_grad"] = max(rep["grads"].values())
    rep["worst_adam"] = max(max(rep["adam_bf16"].values()), max(rep["adam_fp32"].values()))
    rep["worst_e2e_param"] = max(rep["e2e_bf16_param"].values())
    rep["pass"] = bool(rep["visible_match"] and rep["worst_grad"] <= grad_bar
                       and rep["worst_adam"] <= adam_bar and rep["worst_e2e_param"] <= adam_bar)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dumps", nargs="+")
    a = ap.parse_args(argv)
    ok = True
    for p in a.dumps:
        rep = check(np.load(p))
        print(p, json.dumps(rep, indent=1))
        ok &= rep["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
