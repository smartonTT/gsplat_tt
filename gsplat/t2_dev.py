"""T2 device pass (#697): host-side layout, inputs and the check against t2_step.

Layout (render/train/kernels/t2_*.cpp): Gaussians in groups of 64; one 32x32 tile page per
group holds 16 streams of 64 lanes, stream s at packed elements [64 s, 64 s + 64), Gaussian i
of the group at element 64 s + i (SFPU vectors 2 s, 2 s + 1). Block = 16 groups = 1024 rows
(the PFWC_VIS mask granularity).
  params  fp32 page/group: mx my mz ls0 ls1 ls2 qw qx qy qz ol sh_r sh_g sh_b (14, 15 zero)
  m, v    bf16 page/group, same streams
  meta    u32 page/group (512 B): P[g0 + i] (i < 64), then counts[g0 + i]
  slots   T1 v2 slot pages (16 floats, 9 used) packed 64 slots per 4 KB DRAM page
  vis     u32 page/block (128 B): bit i of word w = row 32 w + i visible (PFWC_VIS stand-in)
  pfwc    fp32 SoA, stream-major, 256 B per (stream, group): page = s * n_groups + g
  grads   fp32 page/group, param streams (debug output, T2_GRADS_OUT)

  python3 -m gsplat.t2_dev prep-dump <t0.npz> <dir>     T0 dump -> device inputs + reference
  python3 -m gsplat.t2_dev prep-synth <dir> --n N --mean-count C   timing case (random)
  python3 -m gsplat.t2_dev check <dir> <outdir>          bench outputs vs t2_step
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from gsplat import t2_param_backward as t2

f32 = np.float32
G = 64            # Gaussians per group (one tile page)
GPB = 16          # groups per block (1024 rows)
SLOTS_PER_PAGE = 64
STREAMS = 14
NCONST = 31


def param_cols(p: dict) -> np.ndarray:
    """(N, 14) fp32 param streams in kernel order."""
    return np.concatenate([np.asarray(p["means"], f32), np.asarray(p["log_scales"], f32),
                           np.asarray(p["quats"], f32),
                           np.asarray(p["opacity_logits"], f32)[:, None],
                           np.asarray(p["sh"], f32)[:, 0, :]], 1)


def from_cols(c: np.ndarray) -> dict:
    return {"means": c[:, 0:3], "log_scales": c[:, 3:6], "quats": c[:, 6:10],
            "opacity_logits": c[:, 10], "sh": c[:, 11:14][:, None, :]}


def to_pages(cols: np.ndarray, n_groups: int, pad_row=None) -> np.ndarray:
    """(N, S<=16) -> (n_groups, 1024): page[g, 64 s + i] = cols[64 g + i, s]."""
    N, S = cols.shape
    full = np.zeros((n_groups * G, 16), cols.dtype)
    if pad_row is not None:
        full[N:, :len(pad_row)] = pad_row
    full[:N, :S] = cols
    return full.reshape(n_groups, G, 16).transpose(0, 2, 1).reshape(n_groups, 1024).copy()


def from_pages(pages: np.ndarray, N: int, S: int) -> np.ndarray:
    ng = pages.shape[0]
    return pages.reshape(ng, 16, G).transpose(0, 2, 1).reshape(ng * G, 16)[:N, :S].copy()


def bf16_bits(x: np.ndarray) -> np.ndarray:
    return (t2.bf16_round(np.asarray(x, f32)).view(np.uint32) >> 16).astype(np.uint16)


def bits_bf16(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(f32)


def consts(w2c, K, H, W, lr, betas, eps, step, near=0.2, lowpass=0.3) -> list:
    """Compute-kernel runtime constants (fp32), same rounding as t2.adam_step."""
    R, t, fx, fy, cx, cy = t2._cam(w2c, K)
    t_ = step + 1
    bc1, bc2 = 1.0 - betas[0] ** t_, 1.0 - betas[1] ** t_
    b1, b2 = f32(betas[0]), f32(betas[1])
    ss = [f32(lr[k] / bc1) for k in ("means", "log_scales", "quats", "opacity_logits", "sh_dc")]
    c = list(R.ravel()) + list(t) + [fx, fy, cx, cy, f32(W), f32(H), f32(near), f32(lowpass)]
    c += ss + [b1, f32(1) - b1, b2, f32(1) - b2, f32(1.0 / math.sqrt(bc2)), f32(eps)]
    assert len(c) == NCONST
    return [float(f32(x)) for x in c]


def write_inputs(out: Path, params: dict, m: dict, v: dict, slots, P, counts, visible,
                 cmeta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    N = params["means"].shape[0]
    ng = -(-N // (G * GPB)) * GPB
    pad = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # unit quaternion in padded rows
    to_pages(param_cols(params), ng, pad).astype(f32).tofile(out / "params.f32")
    for nm, d in (("m", m), ("v", v)):
        to_pages(bf16_bits(param_cols(d)), ng).tofile(out / f"{nm}.u16")
    Pf = np.full(ng * G, int(P[-1] + counts[-1]) if N else 0, np.uint32)
    Pf[:N] = P
    cf = np.zeros(ng * G, np.uint32)
    cf[:N] = counts
    np.concatenate([Pf.reshape(ng, G), cf.reshape(ng, G)], 1).tofile(out / "meta.u32")
    ns = slots.shape[0]
    npages = max(1, -(-ns // SLOTS_PER_PAGE))
    sl = np.zeros((npages * SLOTS_PER_PAGE, 16), f32)
    sl[:ns, :slots.shape[1]] = slots
    sl.tofile(out / "slots.f32")
    vb = np.zeros(ng * G, bool)
    vb[:N] = visible
    words = np.packbits(vb.reshape(-1, 32)[:, ::-1], axis=1).view(">u4").astype(np.uint32)
    words.reshape(ng // GPB, 32).tofile(out / "vis.u32")
    meta = dict(cmeta, n=int(N), n_groups=int(ng), n_slot_pages=int(npages),
                n_slots=int(ns), max_count=int(counts.max()) if N else 0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def prep_dump(dump: str, out: Path) -> dict:
    d = np.load(dump)
    H, W = int(d["H"]), int(d["W"])
    params = {k: np.asarray(d[f"param_{k}"], f32) for k in t2.PARAM_NAMES}
    bw = t2.oracle_slots(d)
    lr = json.loads(str(d["adam_lr_json"]))
    betas = tuple(float(b) for b in d["adam_betas"])
    eps, step = float(d["adam_eps"]), int(d["adam_step_in"])
    m = {k: np.asarray(d[f"adam_bf16_m_in_{k}"], f32) for k in t2.PARAM_NAMES}
    v = {k: np.asarray(d[f"adam_bf16_v_in_{k}"], f32) for k in t2.PARAM_NAMES}
    vis = t2.forward_lanes(params, d["w2c"], d["K"], H, W)["visible"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "ref_in.npz", w2c=d["w2c"], K=d["K"], H=H, W=W, lr=json.dumps(lr),
             betas=betas, eps=eps, step=step, slots=bw["slots"], P=bw["P"],
             counts=bw["counts"], **{f"p_{k}": a for k, a in params.items()},
             **{f"m_{k}": a for k, a in m.items()}, **{f"v_{k}": a for k, a in v.items()})
    cm = {"consts": consts(d["w2c"], d["K"], H, W, lr, betas, eps, step), "source": str(dump)}
    write_inputs(out, params, m, v, bw["slots"], bw["P"], bw["counts"], vis, cm)
    return json.loads((out / "meta.json").read_text())


def synth(N: int, mean_count: float, seed: int = 0, H: int = 822, W: int = 1237) -> dict:
    """Random timing case: Gaussians in front of a 1237x822 camera, Poisson slot counts."""
    rng = np.random.default_rng(seed)
    f = f32(1100)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], f32)
    w2c = np.eye(4, dtype=f32)
    z = rng.uniform(1, 20, N).astype(f32)
    means = np.stack([rng.uniform(-0.7, 0.7, N) * z, rng.uniform(-0.5, 0.5, N) * z, z], 1)
    q = rng.normal(size=(N, 4)).astype(f32)
    params = {"means": means.astype(f32),
              "log_scales": rng.uniform(-5, -1, (N, 3)).astype(f32), "quats": q,
              "opacity_logits": rng.normal(size=N).astype(f32),
              "sh": rng.normal(0, 0.5, (N, 1, 3)).astype(f32)}
    counts = rng.poisson(mean_count, N).astype(np.uint32)
    P = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint32)
    slots = rng.normal(0, 1e-3, (int(counts.sum()), 9)).astype(f32)
    mv = lambda s: {k: (rng.normal(size=a.shape) * s).astype(f32) for k, a in params.items()}  # noqa: E731
    m, v = mv(1e-4), {k: np.abs(a) for k, a in mv(1e-6).items()}
    lr = {"means": 1.6e-4, "log_scales": 5e-3, "quats": 1e-3, "opacity_logits": 5e-2,
          "sh_dc": 2.5e-3, "sh_rest": 1.25e-4}
    return dict(params=params, m=m, v=v, slots=slots, P=P, counts=counts, w2c=w2c, K=K,
                H=H, W=W, lr=lr, betas=(0.9, 0.999), eps=1e-15, step=10)


def prep_synth(out: Path, N: int, mean_count: float, seed: int = 0, save_ref: bool = True):
    s = synth(N, mean_count, seed)
    out.mkdir(parents=True, exist_ok=True)
    vis = t2.forward_lanes(s["params"], s["w2c"], s["K"], s["H"], s["W"])["visible"]
    if save_ref:
        np.savez(out / "ref_in.npz", w2c=s["w2c"], K=s["K"], H=s["H"], W=s["W"],
                 lr=json.dumps(s["lr"]), betas=s["betas"], eps=s["eps"], step=s["step"],
                 slots=s["slots"], P=s["P"], counts=s["counts"],
                 **{f"p_{k}": a for k, a in s["params"].items()},
                 **{f"m_{k}": a for k, a in s["m"].items()},
                 **{f"v_{k}": a for k, a in s["v"].items()})
    cm = {"consts": consts(s["w2c"], s["K"], s["H"], s["W"], s["lr"], s["betas"], s["eps"],
                           s["step"]), "source": f"synth N={N} mean_count={mean_count}",
          "visible_frac": float(vis.mean())}
    write_inputs(out, s["params"], s["m"], s["v"], s["slots"], s["P"], s["counts"], vis, cm)
    return json.loads((out / "meta.json").read_text())


def reference(indir: Path) -> dict:
    r = np.load(indir / "ref_in.npz")
    g = lambda p: {k: r[f"{p}_{k}"] for k in t2.PARAM_NAMES}  # noqa: E731
    return t2.t2_step(g("p"), r["slots"], r["P"], r["counts"], r["w2c"], r["K"], int(r["H"]),
                      int(r["W"]), g("m"), g("v"), int(r["step"]), json.loads(str(r["lr"])),
                      tuple(r["betas"]), float(r["eps"]))


def check(indir: Path, outdir: Path, grad_bar=1e-2, adam_bar=1e-3) -> dict:
    meta = json.loads((indir / "meta.json").read_text())
    N, ng = meta["n"], meta["n_groups"]
    ref = reference(indir)
    # T2 owns SH0 only (higher SH bands are another pass): compare the band-0 slice.
    for w in ("grads", "params", "m", "v"):
        if w in ref:
            ref[w] = {**ref[w], "sh": np.asarray(ref[w]["sh"])[:, :1, :]}
    rd = lambda n, dt: np.fromfile(outdir / n, dt).reshape(ng, 1024)  # noqa: E731
    p = from_cols(from_pages(rd("params_out.f32", f32), N, STREAMS))
    m = from_cols(from_pages(bits_bf16(rd("m_out.u16", np.uint16)), N, STREAMS))
    v = from_cols(from_pages(bits_bf16(rd("v_out.u16", np.uint16)), N, STREAMS))
    rep = {"n": N, "n_slots": meta["n_slots"], "max_count": meta["max_count"]}
    if (outdir / "grads.f32").exists():
        gr = from_cols(from_pages(rd("grads.f32", f32), N, STREAMS))
        rep["grads"] = {k: t2.rel_error(gr[k], ref["grads"][k]) for k in t2.PARAM_NAMES}
    rep["adam"] = {f"{w}_{k}": t2.rel_error(a[k], ref[w][k])
                   for w, a in (("params", p), ("m", m), ("v", v)) for k in t2.PARAM_NAMES}
    pin = from_cols(param_cols({k: np.load(indir / "ref_in.npz")[f"p_{k}"]
                                for k in t2.PARAM_NAMES}))
    rep["adam"].update({f"delta_{k}": t2.rel_error(p[k] - pin[k], ref["params"][k] - pin[k])
                        for k in t2.PARAM_NAMES})
    pf = np.fromfile(outdir / "pfwc.f32", f32).reshape(13, ng * G)[:, :N]
    rep["pfwc"] = {s: t2.rel_error(pf[i], ref["pfwc"][s][:N])
                   for i, s in enumerate(t2.PFWC_STREAMS)}
    rep["m_bits_equal"] = bool(all((bf16_bits(m[k]) == bf16_bits(ref["m"][k])).all()
                                   for k in t2.PARAM_NAMES))
    rep["worst_grad"] = max(rep.get("grads", {"-": 0.0}).values())
    rep["worst_adam"] = max(rep["adam"].values())
    rep["worst_pfwc"] = max(rep["pfwc"].values())
    rep["pass"] = bool(rep["worst_grad"] <= grad_bar and rep["worst_adam"] <= adam_bar
                       and rep["worst_pfwc"] <= adam_bar)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("prep-dump")
    a1.add_argument("dump")
    a1.add_argument("out")
    a2 = sp.add_parser("prep-synth")
    a2.add_argument("out")
    a2.add_argument("--n", type=int, default=1 << 20)
    a2.add_argument("--mean-count", type=float, default=8.0)
    a2.add_argument("--seed", type=int, default=0)
    a2.add_argument("--no-ref", action="store_true")
    a3 = sp.add_parser("check")
    a3.add_argument("indir")
    a3.add_argument("outdir")
    a = ap.parse_args(argv)
    if a.cmd == "prep-dump":
        print(json.dumps(prep_dump(a.dump, Path(a.out))))
    elif a.cmd == "prep-synth":
        print(json.dumps(prep_synth(Path(a.out), a.n, a.mean_count, a.seed, not a.no_ref)))
    else:
        rep = check(Path(a.indir), Path(a.outdir))
        print(json.dumps(rep, indent=1))
        (Path(a.outdir) / "check.json").write_text(json.dumps(rep, indent=1))
        return 0 if rep["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
