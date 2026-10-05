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

good_shot = {
    "hero": "opt/metal-screenshots/ttw-198/hero.png",
    "diff": "opt/metal-screenshots/ttw-198/hero_diff10.png",
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
# New iteration without a screenshot fails; the backfill flag is not accepted for it.
assert status({"iter": 198})[0] == "missing"
st, probs = status({"iter": 198, "screenshot_backfill_pending": True})
assert st == "missing" and "only allowed for legacy" in probs[0], probs
# Missing visual check, md5, commit or config fails.
for k in ("visual_check", "md5", "commit", "config", "psnr_vs_ref"):
    st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, **{k: ""})})
    assert st == "missing" and any(k in p for p in probs), (k, probs)
assert status({"iter": 198, "device_screenshot": dict(good_shot, visual_check="ok")})[0] == "missing"
# Missing or non-PNG image files fail.
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, diff="opt/metal-screenshots/ttw-198/nope.png")})
assert st == "missing" and "does not exist" in probs[0], probs
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, hero="opt/metal-screenshots/ttw-198/notpng.png")})
assert st == "missing" and "not a PNG" in probs[0], probs
# The CPU reference is not a device screenshot.
st, probs = status({"iter": 198, "device_screenshot": dict(good_shot, hero="benchmarks/reference_v2/hero.png")})
assert st == "missing" and any("CPU/reference" in p for p in probs), probs
# Legacy 160-197: backfill flag allowed, nothing at all is not.
assert status({"iter": 170, "screenshot_backfill_pending": True})[0] == "pending"
assert status({"iter": 170})[0] == "missing"
assert status({"iter": 159})[0] == "legacy"

errors, pending = b.check_device_screenshots(
    [{"iter": 159}, {"iter": 170, "screenshot_backfill_pending": True},
     {"iter": 198, "device_screenshot": good_shot}, {"iter": 199}], root)
assert pending == [170] and len(errors) == 1 and "ttw iter 199" in errors[0], (errors, pending)

# Report cards: red placeholder for pending, zoomable hero + diff + PSNR when present.
card = b._iter_card_html(b.normalize_ttw_row({"iter": 170, "idea": "x", "screenshot_backfill_pending": True}), "blackhole")
assert "screenshot missing - backfill pending" in card and "shot-missing" in card
card = b._iter_card_html(b.normalize_ttw_row({"iter": 199, "idea": "x"}), "blackhole")
assert "device screenshot MISSING - required" in card
_orig_root = b.REPO_ROOT
b.REPO_ROOT = root
try:
    card = b._iter_card_html(b.normalize_ttw_row({"iter": 198, "idea": "x", "device_screenshot": good_shot}), "blackhole")
finally:
    b.REPO_ROOT = _orig_root
assert 'class="zoom"' in card and "metal-screenshots/ttw-198/hero.png" in card, card
assert "metal-screenshots/ttw-198/hero_diff10.png" in card and "PSNR 100.00 dB" in card, card
assert "no tile seams" in card and "shot-missing" not in card, card

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
