"""Compare a backend stage's output against the numpy reference fixture.

Used by Phase 2 port iters (002 project, 003 tile_assign+sort, 004 blend) as
the Layer 2 gate: the C++ port of stage X must produce output close-to-equal
to tests/fixtures/hero/<stage>_outputs.npz before downstream stages can
even be exercised.

Usage:
  python3 scripts/verify_stage.py --backend cpu_cpp --stage project
  python3 scripts/verify_stage.py --backend cpu_cpp --stage tile_assign
  python3 scripts/verify_stage.py --backend cpu_cpp --stage sort
  python3 scripts/verify_stage.py --backend cpu_cpp --stage blend

Returns exit 0 on PASS, non-zero on FAIL. Prints per-output max-abs-error
and tolerance verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402


# Per-stage output tolerances (max absolute element-wise diff).
# project, tile_assign, sort are exact when seeds + types match; tiny epsilons
# only allow for fp32 operator-order differences. Blend uses a PSNR floor
# instead — see PSNR_FLOOR_DB and blend's special handling below.
TOLS = {
    "project": {"means_2d": 1e-4, "covs_2d": 1e-5, "depths": 1e-5,
                "radii": 1e-3, "valid_mask": 0},
    "tile_assign": {"gaussian_ids": 0, "tile_ids": 0, "tiles_per_gaussian": 0},
    "sort": {
        "sorted_gaussian_ids_depth_sequence": 0,
        "sorted_gaussian_ids_per_tile_set": 0,
        "tile_ranges": 0,
    },
    "blend": {"image": 1.0},   # max-abs check is informational; PSNR gates
    "microblock_cull": {"mb_header": 0, "mb_stream": 0},
}
# Layer 2 gate for the blend stage. Matches the north-star invariant (60 dB)
# from opt/plan.md — a port that's truly identical modulo fp accumulation
# order should be well above this.
PSNR_FLOOR_DB = 60.0


def _to_np(t):
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def _max_abs(a, b) -> float:
    a = _to_np(a).astype(np.float64)
    b = _to_np(b).astype(np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


def _exact_equal(a, b) -> bool:
    a = _to_np(a)
    b = _to_np(b)
    return a.shape == b.shape and bool((a == b).all())


def run_project(backend, fixtures_dir: Path):
    inp = np.load(fixtures_dir / "project_inputs.npz")
    out = np.load(fixtures_dir / "project_outputs.npz")
    means_2d, covs_2d, depths, radii, valid_mask = backend.project(
        torch.from_numpy(inp["means"]),
        torch.from_numpy(inp["scales"]),
        torch.from_numpy(inp["rotations"]),
        torch.from_numpy(inp["extrinsics"]),
        torch.from_numpy(inp["intrinsics"]),
        int(inp["H"]), int(inp["W"]),
        opacities=torch.from_numpy(inp["opacities"]),
    )
    return {
        "means_2d": (means_2d, out["means_2d"]),
        "covs_2d":  (covs_2d,  out["covs_2d"]),
        "depths":   (depths,   out["depths"]),
        "radii":    (radii,    out["radii"]),
        "valid_mask": (valid_mask, out["valid_mask"]),
    }


def run_tile_assign(backend, fixtures_dir: Path):
    inp = np.load(fixtures_dir / "tile_assign_inputs.npz")
    out = np.load(fixtures_dir / "tile_assign_outputs.npz")
    gids, tids, tpg = backend.tile_assign(
        torch.from_numpy(inp["means_2d"]),
        torch.from_numpy(inp["radii"]),
        int(inp["H"]), int(inp["W"]),
        tile_size=int(inp["tile_size"]),
        covs_2d=torch.from_numpy(inp["covs_2d"]),
        opacities=torch.from_numpy(inp["opacities"]),
    )
    return {
        "gaussian_ids":       (gids, out["gaussian_ids"]),
        "tile_ids":           (tids, out["tile_ids"]),
        "tiles_per_gaussian": (tpg,  out["tiles_per_gaussian"]),
    }


def run_sort(backend, fixtures_dir: Path):
    inp = np.load(fixtures_dir / "sort_inputs.npz")
    out = np.load(fixtures_dir / "sort_outputs.npz")
    sgids, tranges = backend.sort(
        torch.from_numpy(inp["gaussian_ids"]),
        torch.from_numpy(inp["tile_ids"]),
        torch.from_numpy(inp["depths"]),
        int(inp["tiles_x"]), int(inp["tiles_y"]),
    )
    # Semantic check for sorted_gaussian_ids: torch.argsort is unstable, so
    # tied depths within a tile can land in any order across implementations.
    # The downstream-meaningful invariant is:
    #   (1) the depths gathered through sorted_gaussian_ids are identical to
    #       the depths gathered through the reference, position-by-position.
    #   (2) per-tile, the set of gids is identical.
    # We materialize a "canonical" form for both arrays by re-sorting each
    # tile's gid slice in ascending gid order, then comparing — that wipes out
    # tie-break variation but preserves all real ordering errors. Also we
    # compare depths arrays directly, which catches any non-tie misordering.
    sgids_np = _to_np(sgids).astype(np.int64)
    ref_sgids_np = np.asarray(out["sorted_gaussian_ids"], np.int64)
    tranges_np = _to_np(tranges).astype(np.int64)
    ref_tranges = np.asarray(out["tile_ranges"], np.int64)
    depths_full = np.asarray(inp["depths"])

    def depths_in_order(sgids_arr):
        return depths_full[sgids_arr]

    cand_depth_seq = depths_in_order(sgids_np)
    ref_depth_seq = depths_in_order(ref_sgids_np)

    def canonical(sgids_arr, tranges_arr):
        out_arr = sgids_arr.copy()
        for t in range(tranges_arr.shape[0]):
            lo, hi = tranges_arr[t, 0], tranges_arr[t, 1]
            if lo == hi:
                continue
            out_arr[lo:hi] = np.sort(out_arr[lo:hi])
        return out_arr

    cand_canon = canonical(sgids_np, tranges_np)
    ref_canon = canonical(ref_sgids_np, ref_tranges)

    return {
        "sorted_gaussian_ids_depth_sequence": (cand_depth_seq, ref_depth_seq),
        "sorted_gaussian_ids_per_tile_set": (cand_canon, ref_canon),
        "tile_ranges": (tranges_np, ref_tranges),
    }


def run_blend(backend, fixtures_dir: Path):
    inp = np.load(fixtures_dir / "blend_inputs.npz")
    ref_out = np.load(fixtures_dir / "blend_output.npy")
    image, _sub = backend.blend(
        torch.from_numpy(inp["means_2d"]),
        torch.from_numpy(inp["covs_2d"]),
        torch.from_numpy(inp["colors"]),
        torch.from_numpy(inp["opacities"]),
        torch.from_numpy(inp["sorted_gaussian_ids"]),
        torch.from_numpy(inp["tile_ranges"]),
        int(inp["H"]), int(inp["W"]),
    )
    return {"image": (image, ref_out)}


def run_microblock_cull(backend, fixtures_dir: Path):
    inp = np.load(fixtures_dir / "microblock_cull_inputs.npz")
    out = np.load(fixtures_dir / "microblock_cull_outputs.npz")
    mod = backend._mod
    mb_header_flat, mb_stream, _stats = mod.microblock_cull(
        np.ascontiguousarray(inp["means_2d"], np.float32),
        np.ascontiguousarray(inp["covs_2d"].reshape(-1, 4), np.float32),
        np.ascontiguousarray(inp["opacities"], np.float32),
        np.ascontiguousarray(inp["sorted_gaussian_ids"], np.int64),
        np.ascontiguousarray(inp["tile_ranges"], np.int64),
        int(inp["tiles_x"]),
        int(inp["tiles_y"]),
        32,
        float(inp["mb_contrib_floor"]),
    )
    tiles_x = int(inp["tiles_x"])
    tiles_y = int(inp["tiles_y"])
    num_tiles = tiles_x * tiles_y
    return {
        "mb_header": (
            mb_header_flat.reshape(num_tiles, 32, 2),
            out["mb_header"],
        ),
        "mb_stream": (mb_stream, out["mb_stream"]),
    }


RUNNERS = {
    "project": run_project,
    "tile_assign": run_tile_assign,
    "sort": run_sort,
    "blend": run_blend,
    "microblock_cull": run_microblock_cull,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--stage", required=True, choices=list(RUNNERS))
    ap.add_argument("--fixtures-dir", default="tests/fixtures/hero", type=Path)
    args = ap.parse_args()

    if not (args.fixtures_dir / "meta.json").exists():
        raise SystemExit(
            f"missing {args.fixtures_dir}; run "
            f"`python3 scripts/capture_reference.py --scene stitch` first"
        )

    backend = get_backend(args.backend)
    pairs = RUNNERS[args.stage](backend, args.fixtures_dir)

    tols = TOLS[args.stage]
    rows = []
    all_pass = True
    psnr_dB = None
    for name, (cand, ref) in pairs.items():
        tol = tols.get(name, 0)
        if tol == 0:
            ok = _exact_equal(cand, ref)
            diff = 0.0 if ok else float("inf")
        else:
            diff = _max_abs(cand, ref)
            ok = diff <= tol
        # Special-case the blend image: PSNR gate is the actual pass criterion.
        if args.stage == "blend" and name == "image":
            cand_np = _to_np(cand).astype(np.float64)
            ref_np = _to_np(ref).astype(np.float64)
            mse = float(np.mean((cand_np - ref_np) ** 2))
            psnr_dB = float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)
            ok = psnr_dB >= PSNR_FLOOR_DB
        if not ok:
            all_pass = False
        row = {"name": name, "tol": tol, "max_abs_diff": diff, "pass": ok,
               "ref_shape": list(_to_np(ref).shape),
               "cand_shape": list(_to_np(cand).shape)}
        if psnr_dB is not None and name == "image":
            row["psnr_dB"] = psnr_dB
            row["psnr_floor_dB"] = PSNR_FLOOR_DB
        rows.append(row)

    print(json.dumps({"stage": args.stage, "backend": args.backend,
                      "pass": all_pass, "checks": rows}, indent=2))
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
