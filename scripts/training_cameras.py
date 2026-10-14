"""Task #667: training-view bench cameras for bicycle (Mip-NeRF360 dataset poses).

Writes camera files in the benchmarks/cameras_v2.json format at 1237x822 (the
dataset's images_4 size), with the dataset's own intrinsics scaled to that size
and stored as "intrinsics" {fx, fy, cx, cy} (render/run.py uses them over fov_deg):

  benchmarks/cameras_train_shuffled.json  train split (all but every 8th image), seeded order
  benchmarks/cameras_train_nn.json        same train split, greedy nearest-neighbour tour over
                                          camera position + view direction (max view coherence)
  benchmarks/cameras_test.json            test split (every 8th image, Mip-NeRF360 protocol)
  benchmarks/cameras_orbit_1237.json      the cameras_v2 orbit at 1237x822 (control, same fov)

Pose sources (one of):
  --colmap DIR        COLMAP sparse model (cameras.bin + images.bin), e.g. 360_v2/bicycle/sparse/0
  --gs-cameras FILE   3DGS training output cameras.json (id, img_name, width, height,
                      position, rotation, fx, fy)

Split: images sorted by name, index % 8 == 0 is test (the usual LLFF/Mip-NeRF360 holdout).
Check: the cameras_v2 hero is the dataset image _DSC8679; the script prints its c2w
difference against the dataset pose (must be ~0, else the frames do not match).

  python scripts/training_cameras.py --colmap /path/to/bicycle/sparse/0 [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
W_OUT, H_OUT = 1237, 822
# COLMAP model id -> (name, number of params)
_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
           3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8)}


def _qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap(d: Path):
    """-> list of (name, c2w 4x4, fx, fy, cx, cy, width, height)."""
    cams = {}
    with open(d / "cameras.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, n = _MODELS[mid]
            p = struct.unpack(f"<{n}d", f.read(8 * n))
            if name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[:4]
            cams[cid] = (fx, fy, cx, cy, w, h)
    out = []
    with open(d / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _iid, *rest = struct.unpack("<i7di", f.read(64))
            q, t, cid = rest[:4], rest[4:7], rest[7]
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)
            w2c = np.eye(4); w2c[:3, :3] = _qvec_to_R(q); w2c[:3, 3] = t
            out.append((name.decode(), np.linalg.inv(w2c), *cams[cid]))
    return out


def read_gs_cameras(p: Path):
    out = []
    for c in json.loads(p.read_text()):
        c2w = np.eye(4); c2w[:3, :3] = np.asarray(c["rotation"]); c2w[:3, 3] = c["position"]
        w, h = c["width"], c["height"]
        out.append((c["img_name"], c2w, c["fx"], c["fy"], w / 2, h / 2, w, h))
    return out


def nn_order(names, views, dir_weight):
    """Greedy nearest-neighbour tour from the first name: next = the unvisited camera
    minimising |dpos| / scene_scale + dir_weight * (1 - cos(view directions))."""
    pos = {n: np.asarray(views[n]["c2w"])[:3, 3] for n in names}
    fwd = {n: np.asarray(views[n]["c2w"])[:3, 2] for n in names}  # +z = view dir (OpenCV)
    P = np.stack(list(pos.values()))
    scale = float(np.linalg.norm(P - P.mean(0), axis=1).mean()) or 1.0
    tour, left = [names[0]], set(names[1:])
    while left:
        c = tour[-1]
        nxt = min(sorted(left), key=lambda n: np.linalg.norm(pos[n] - pos[c]) / scale
                  + dir_weight * (1.0 - float(fwd[n] @ fwd[c])))
        tour.append(nxt); left.remove(nxt)
    return tour


def _file(base, names, views, intr, fov):
    return {"bicycle": {
        "ply": base["ply"], "image_size": [W_OUT, H_OUT], "fov_deg": fov,
        "contrib_floor": base["contrib_floor"], "intrinsics": intr,
        "order": names, "views": views}}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--colmap", type=Path)
    g.add_argument("--gs-cameras", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn-dir-weight", type=float, default=1.0,
                    help="weight of (1 - cos) view-direction distance vs position/scene_scale")
    ap.add_argument("--out-dir", type=Path, default=REPO / "benchmarks")
    a = ap.parse_args()

    base = json.loads((REPO / "benchmarks/cameras_v2.json").read_text())["bicycle"]
    recs = sorted(read_colmap(a.colmap) if a.colmap else read_gs_cameras(a.gs_cameras))
    _, _, fx, fy, cx, cy, w, h = recs[0]
    assert all(r[2:] == recs[0][2:] for r in recs), "expected one shared camera model"
    sx, sy = W_OUT / w, H_OUT / h
    intr = {"fx": fx * sx, "fy": fy * sy, "cx": cx * sx, "cy": cy * sy}
    fov = float(np.degrees(2 * np.arctan(0.5 * W_OUT / intr["fx"])))
    print(f"{len(recs)} cameras, source {w}x{h} -> {W_OUT}x{H_OUT}, intr {intr}, fov(W) {fov:.3f}")

    key = lambda n: Path(n).stem  # noqa: E731
    views = {key(n): {"c2w": c2w.tolist(), "manual": False, "source": f"mipnerf360 {n}"}
             for n, c2w, *_ in recs}
    test = [key(r[0]) for i, r in enumerate(recs) if i % 8 == 0]
    train = [key(r[0]) for i, r in enumerate(recs) if i % 8 != 0]
    train_nn = nn_order(train, views, a.nn_dir_weight)  # tour starts at the first sorted name
    random.Random(a.seed).shuffle(train)
    if "_DSC8679" in views:
        d = np.abs(np.asarray(views["_DSC8679"]["c2w"]) - np.asarray(base["views"]["hero"]["c2w"]))
        print(f"hero check: max |c2w(_DSC8679) - cameras_v2 hero| = {d.max():.2e}")
    for fname, names in (("cameras_train_shuffled.json", train), ("cameras_train_nn.json", train_nn),
                         ("cameras_test.json", test)):
        p = a.out_dir / fname
        p.write_text(json.dumps(_file(base, names, {n: views[n] for n in names}, intr, fov), indent=1))
        print(f"wrote {p} ({len(names)} views, first {names[0]})")
    orbit = dict(base, image_size=[W_OUT, H_OUT])
    p = a.out_dir / "cameras_orbit_1237.json"
    p.write_text(json.dumps({"bicycle": orbit}, indent=1))
    print(f"wrote {p} (orbit control, {len(base['order'])} views, fov {base['fov_deg']} on W)")


if __name__ == "__main__":
    main()
