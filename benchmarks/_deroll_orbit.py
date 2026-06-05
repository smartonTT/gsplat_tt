"""De-roll the bicycle orbit views: keep each eye + the look-at center, but
rebuild every rotation with a look-at whose up is the SCENE GRAVITY (the real
hero camera's up), not world +Y. Fixes the tilted-horizon roll.

Convention (verified against stored hero c2w):
  fwd   = normalize(center - eye)
  right = normalize(cross(gravity_up, fwd))
  up    = cross(fwd, right)
  R     = [right | up | fwd]   (columns),  t = eye
"""
import json, sys
import numpy as np

PATH = "benchmarks/cameras_v2.json"
CENTER = np.array([-2.232, 1.318, -1.597])

doc = json.loads(open(PATH).read())
d = doc["bicycle"]
V = d["views"]

hero = np.array(V["hero"]["c2w"], dtype=float)
gravity_up = hero[:3, 1].copy()
gravity_up /= np.linalg.norm(gravity_up)

def look_at(eye, center, up):
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    upo = np.cross(fwd, right)
    R = np.column_stack([right, upo, fwd])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w

# --- self-check: rebuilding hero from (eye, center, gravity_up) must match stored ---
eye_h = hero[:3, 3]
rebuilt_h = look_at(eye_h, CENTER, gravity_up)
err = np.abs(rebuilt_h - hero).max()
print(f"hero rebuild max-abs-err = {err:.6f}  (must be ~0 to trust convention)")
if err > 1e-3:
    print("CONVENTION MISMATCH — aborting, not writing", file=sys.stderr)
    sys.exit(2)

def roll_metric(c2w):
    M = c2w[:3, :3]
    right = M[:, 0]
    fwd = M[:, 2]
    horiz = np.cross(gravity_up, fwd)
    horiz /= np.linalg.norm(horiz)
    return np.degrees(np.arccos(np.clip(np.dot(right, horiz), -1, 1)))

print(f"gravity_up = {np.round(gravity_up,4)}")
maxroll_before = 0.0
maxroll_after = 0.0
for name in d["order"]:
    c2w = np.array(V[name]["c2w"], dtype=float)
    eye = c2w[:3, 3]
    rb = roll_metric(c2w)
    new = look_at(eye, CENTER, gravity_up)
    ra = roll_metric(new)
    maxroll_before = max(maxroll_before, rb)
    maxroll_after = max(maxroll_after, ra)
    V[name]["c2w"] = new.tolist()

print(f"max roll-vs-gravity  before={maxroll_before:.3f} deg  after={maxroll_after:.3f} deg")
open(PATH, "w").write(json.dumps(doc, indent=2))
print("wrote", PATH)
