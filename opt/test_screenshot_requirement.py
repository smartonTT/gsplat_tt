"""python3 opt/test_screenshot_requirement.py — device screenshot gate (user, 2026-10-05)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_report as b

root = Path(tempfile.mkdtemp())
shots = root / "opt" / "metal-screenshots" / "ttw-198"
shots.mkdir(parents=True)
(shots / "hero.png").write_bytes(b.PNG_MAGIC + b"x")
(shots / "hero_diff10.png").write_bytes(b.PNG_MAGIC + b"x")
(shots / "notpng.png").write_bytes(b"GIF89a")
(root / "benchmarks" / "reference_v2").mkdir(parents=True)
(root / b.SHOT_REF).write_bytes(b.PNG_MAGIC + b"x")

good_shot = {
    "hero": "opt/metal-screenshots/ttw-198/hero.png",
    "diff": "opt/metal-screenshots/ttw-198/hero_diff10.png",
    "ref": b.SHOT_REF,
    "diff_ref": b.SHOT_REF,
    "psnr_vs_ref": 100.0,
    "md5": "46a725ab",
    "commit": "abc1234",
    "config": "defaults",
    "visual_check": "no tile seams or blocky artifacts; diff black",
}


def status(row):
    return b.screenshot_status(row, root)


assert status({"iter": 198, "device_screenshot": good_shot})[0] == "ok"
assert status({"iter": 198, "device_screenshot": dict(good_shot, psnr_vs_ref="inf")})[0] == "ok"
# New iteration without a screenshot fails; the cancelled backfill flag is not accepted.
assert status({"iter": 198})[0] == "missing"
assert status({"iter": 198, "screenshot_backfill_pending": True})[0] == "missing"
# Missing visual check, md5, commit, config, PSNR or either reference fails.
for k in ("visual_check", "md5", "commit", "config", "psnr_vs_ref", "ref", "diff_ref"):
    st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, **{k: ""})})
    assert st == "missing" and any(k in p for p in probs), (k, probs)
assert status({"iter": 198, "device_screenshot": dict(good_shot, visual_check="ok")})[0] == "missing"
# PSNR and diff must name the same reference (the iter-196..198 bug: PSNR vs the
# golden, diff vs reference_v2).
(root / b.SHOT_GOLDEN).parent.mkdir(parents=True)
(root / b.SHOT_GOLDEN).write_bytes(b.PNG_MAGIC + b"x")
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, ref=b.SHOT_GOLDEN)})
assert st == "missing" and any("same reference" in p for p in probs), probs
# Missing or non-PNG image files fail.
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, diff="opt/metal-screenshots/ttw-198/nope.png")})
assert st == "missing" and "does not exist" in probs[0], probs
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, hero="opt/metal-screenshots/ttw-198/notpng.png")})
assert st == "missing" and "not a PNG" in probs[0], probs
# The CPU reference is not a device screenshot.
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, hero="benchmarks/reference_v2/hero.png")})
assert st == "missing" and any("CPU/reference" in p for p in probs), probs
# Iters up to 197 without a screenshot predate the rule: no error, no placeholder.
assert status({"iter": 170, "screenshot_backfill_pending": True})[0] == "legacy"
assert status({"iter": 170})[0] == "legacy"
assert status({"iter": 159})[0] == "legacy"

errors = b.check_device_screenshots(
    [{"iter": 159}, {"iter": 170}, {"iter": 198, "device_screenshot": good_shot}, {"iter": 199}], root)
assert len(errors) == 1 and "ttw iter 199" in errors[0], errors

# Pixel check: PSNR, diff and golden badge are recomputed from the images.
import numpy as np
from PIL import Image
px = root / "px"
px.mkdir()
rng = np.random.default_rng(0)
ref_a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
hero_a = ref_a.copy()
hero_a[0, 0, 0] ^= 4
for name, arr in (("ref", ref_a), ("hero", hero_a), ("gold", hero_a),
                  ("diff", np.clip(np.abs(hero_a.astype(int) - ref_a) * 10, 0, 255).astype(np.uint8))):
    Image.fromarray(arr).save(px / f"{name}.png")
p_true = b._psnr(hero_a.astype(float), ref_a.astype(float))
px_shot = {"hero": "px/hero.png", "diff": "px/diff.png", "ref": "px/ref.png", "diff_ref": "px/ref.png",
           "psnr_vs_ref": round(p_true, 2), "golden": "px/gold.png", "golden_match": True}
assert b.screenshot_pixel_problems(px_shot, root) == [], b.screenshot_pixel_problems(px_shot, root)
probs = b.screenshot_pixel_problems(dict(px_shot, psnr_vs_ref="inf"), root)
assert probs and "psnr_vs_ref inf" in probs[0], probs
probs = b.screenshot_pixel_problems(dict(px_shot, diff="px/ref.png"), root)
assert probs and "is not |hero" in probs[0], probs
probs = b.screenshot_pixel_problems(dict(px_shot, golden_match=False), root)
assert probs and "golden_match" in probs[0], probs
probs = b.screenshot_pixel_problems(dict(px_shot, knob_on={"hero": "px/hero.png", "psnr_vs_ref": 1.0}), root)
assert probs and "knob_on.psnr_vs_ref" in probs[0], probs

# Report cards: no placeholder for pre-rule iters; viewer links + named ref + golden badge.
card = b._iter_card_html(b.normalize_ttw_row({"iter": 170, "idea": "x", "screenshot_backfill_pending": True}), "blackhole")
assert "backfill pending" not in card and "shot-missing" not in card and "MISSING" not in card, card
card = b._iter_card_html(b.normalize_ttw_row({"iter": 199, "idea": "x"}), "blackhole")
assert "device screenshot MISSING - required" in card
_orig_root = b.REPO_ROOT
b.REPO_ROOT = root
try:
    card = b._iter_card_html(b.normalize_ttw_row({"iter": 198, "idea": "x", "device_screenshot": dict(
        good_shot, golden=b.SHOT_GOLDEN, golden_match=True)}), "blackhole")
finally:
    b.REPO_ROOT = _orig_root
assert 'class="zoom"' in card and "metal-screenshots/ttw-198/hero.png" in card, card
assert "metal-screenshots/ttw-198/hero_diff10.png" in card and "PSNR 100.00 dB" in card, card
assert "psnr=100.00 dB vs benchmarks/reference_v2/hero.png" in card, card
assert 'data-img-ref="../benchmarks/reference_v2/hero.png"' in card, card
assert 'data-img-cand="metal-screenshots/ttw-198/hero.png"' in card, card
assert "golden match" in card and b.SHOT_GOLDEN in card, card
assert "no tile seams" in card and "shot-missing" not in card, card
# The opt/ttw/ mirror rebases the viewer paths like src/href.
mirror = b.rebase_for_ttw(card)
assert 'data-img-ref="../../benchmarks/reference_v2/hero.png"' in mirror, mirror
assert 'data-img-cand="../metal-screenshots/ttw-198/hero.png"' in mirror, mirror
assert 'data-viewer-label="benchmarks/reference_v2/hero.png"' in mirror, mirror

# Single-quoted links (Tracy, docs) are rebased too; absolute/external ones are not.
assert b.rebase_for_ttw("<a href='profiler/x.tracy'>") == "<a href='../profiler/x.tracy'>"
assert b.rebase_for_ttw("<a href='https://arxiv.org/x'><a href=\"#top\">") == "<a href='https://arxiv.org/x'><a href=\"#top\">"

# build_report main exits non-zero when a new iteration lacks a screenshot.
_patches = {"load_iters": lambda: [], "build_html": lambda rows: "", "write_reports": lambda h: None,
            "load_ttw_iters": lambda: [{"iter": 199, "idea": "x"}]}
_saved = {k: getattr(b, k) for k in _patches}
for k, v in _patches.items():
    setattr(b, k, v)
try:
    b.main()
    raise AssertionError("main() did not fail on a missing screenshot")
except SystemExit as e:
    assert e.code == 2, e.code
finally:
    for k, v in _saved.items():
        setattr(b, k, v)
print("ok")
