import json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backends import get_backend
from gsplat.loading_gaussians import load_ply
from gsplat.pipeline import Pipeline
from gsplat.utils import c2w_to_w2c
from scripts.a003_verify import build_intrinsics

cam = json.loads(Path("benchmarks/cameras_v2.json").read_text())["bicycle"]
fov_deg = float(cam["fov_deg"]); W, H = cam["image_size"]
contrib_floor = cam.get("contrib_floor", 1.0/255.0)
order = cam["order"]; views = cam["views"]
name = order[0]
c2w = np.asarray(views[name]["c2w"], dtype=np.float32)
extr = c2w_to_w2c(torch.from_numpy(c2w)); K = build_intrinsics(W, H, fov_deg)
gauss = load_ply("benchmarks/bicycle.ply") if Path("benchmarks/bicycle.ply").exists() else None

def render(bk):
    b = get_backend(bk); p = Pipeline(b, tile_size=32, contrib_floor=contrib_floor)
    g = load_ply(cam["ply"]) if "ply" in cam else gauss
    r = p.render(g, extr, K, H, W); img = r.image
    if hasattr(img, "numpy"): img = img.numpy()
    return np.clip(np.asarray(img, np.float32), 0, 1)

tt = render("tt"); ref = render("cpu_cpp_mb")
err = np.abs(tt - ref).mean(axis=2)  # HxW
mse = (err**2).mean(); print("PSNR", 10*math.log10(1.0/mse) if mse>0 else 99)
print("max err", err.max(), "mean err", err.mean(), "frac>0.02", (err>0.02).mean())
# Per-microblock-row (4px) error to see if error is structured by row band
TS=32
ty, tx = H//TS, W//TS
band_err = np.zeros(8)
for by in range(8):
    rows = []
    for tyy in range(ty):
        r0 = tyy*TS + by*4
        rows.append(err[r0:r0+4, :])
    band_err[by] = np.concatenate(rows,axis=0).mean()
print("err by microblock row-band (0..7):", np.round(band_err,4).tolist())
# error by column-group
colg_err=np.zeros(4)
for cg in range(4):
    cols=[]
    for txx in range(tx):
        c0=txx*TS+cg*8; cols.append(err[:, c0:c0+8])
    colg_err[cg]=np.concatenate(cols,axis=1).mean()
print("err by col-group (0..3):", np.round(colg_err,4).tolist())
from PIL import Image
Image.fromarray((np.clip(err*8,0,1)*255).astype(np.uint8)).save("opt/gm_err.png")
print("saved opt/gm_err.png")
