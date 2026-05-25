"""Tests for scripts/compute_metrics.py."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compute_metrics.py"
FIXTURES = REPO / "tests" / "fixtures"
REF_DIR = FIXTURES / "reference"
ITER_DIR = FIXTURES / "iter-test-good"


def run_compute(iter_dir: Path, ref_dir: Path, class_tag: str = "kernel-algebra", prev_best: float = 20.0):
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--iter-dir", str(iter_dir),
         "--reference-dir", str(ref_dir),
         "--class", class_tag,
         "--prev-best-ms", str(prev_best)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    return json.loads((iter_dir / "metrics.json").read_text())


def test_metrics_schema(tmp_path):
    import shutil
    iter_copy = tmp_path / "iter-test"
    shutil.copytree(ITER_DIR, iter_copy)
    metrics = run_compute(iter_copy, REF_DIR)

    assert "kernel_ms_median" in metrics
    assert "kernel_ms_p99" in metrics
    assert "per_view_median" in metrics
    assert set(metrics["per_view_median"].keys()) == {"hero", "side", "top"}
    assert "psnr_per_view" in metrics
    assert set(metrics["psnr_per_view"].keys()) == {"hero", "side", "top"}
    assert metrics["class"] == "kernel-algebra"
    assert metrics["prev_best_kernel_ms"] == 20.0


def test_psnr_high_for_near_identical(tmp_path):
    import shutil
    iter_copy = tmp_path / "iter-test"
    shutil.copytree(ITER_DIR, iter_copy)
    metrics = run_compute(iter_copy, REF_DIR)
    for view, psnr in metrics["psnr_per_view"].items():
        assert psnr > 80.0, f"{view} PSNR = {psnr}, expected > 80"


def test_diff10_images_written(tmp_path):
    import shutil
    iter_copy = tmp_path / "iter-test"
    shutil.copytree(ITER_DIR, iter_copy)
    run_compute(iter_copy, REF_DIR)
    for view in ["hero", "side", "top"]:
        assert (iter_copy / f"{view}_diff10.png").exists(), f"{view}_diff10.png missing"
