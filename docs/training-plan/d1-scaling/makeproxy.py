#!/usr/bin/env python3
"""Task #691 (D1): build the importance-pruned timing proxies and the camera files on the box.
Proxy N = the N Gaussians with the largest summed alpha*T over the train views (rank.npy from
d1prep.py), written in their original file order (a timing workload, not a lever). Camera
files: cameras_train_nn.json with image_size W//s x H//s and the intrinsics scaled by W'/W, H'/H.
  makeproxy.py --rank tmp/d1/rank.npy --ply scenes/bicycle.ply --cams benchmarks/cameras_train_nn.json
               --out tmp/d1 --ns 3000000,1500000,1000000,500000
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
for k in ("rank", "ply", "cams", "out", "ns"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
with open(a.ply, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        hdr += f.readline()
lines = hdr.decode().splitlines()
props = [l for l in lines if l.startswith("property")]
assert all(p.split()[1] == "float" for p in props), "non-float property"
n0 = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
data = np.fromfile(a.ply, "<f4", offset=len(hdr)).reshape(n0, len(props))
rank = np.load(a.rank)
assert len(rank) == n0
tags = {"full": (a.ply, n0)}
for n in (int(x) for x in a.ns.split(",")):
    keep = np.sort(rank[:n])
    tag = f"{n/1e6:g}M".replace(".", "p")
    path = os.path.join(a.out, f"bicycle_{tag}.ply")
    newhdr = hdr.decode().replace(f"element vertex {n0}\n", f"element vertex {n}\n").encode()
    with open(path, "wb") as f:
        f.write(newhdr)
        data[keep].tofile(f)
    tags[tag] = (path, n)
    print(f"proxy {tag} n={n} -> {path}", flush=True)
cam = json.load(open(a.cams))
c0 = cam["bicycle"]; W0, H0 = c0["image_size"]; K = c0["intrinsics"]
for tag, (path, n) in tags.items():
    for s in (1, 2, 4):
        W, H = W0 // s, H0 // s
        c = dict(c0, ply=path, image_size=[W, H],
                 intrinsics=dict(fx=K["fx"] * W / W0, fy=K["fy"] * H / H0, cx=K["cx"] * W / W0, cy=K["cy"] * H / H0))
        json.dump({"bicycle": c}, open(os.path.join(a.out, f"cams_{tag}_s{s}.json"), "w"))
json.dump({t: v[1] for t, v in tags.items()}, open(os.path.join(a.out, "proxies.json"), "w"))
print("cams written", flush=True)
