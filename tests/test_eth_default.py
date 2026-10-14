"""Task #409: render/eth_default.setup picks ETH dispatch on a p150 and falls back to worker."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "render"))
import eth_default as e  # noqa: E402


def _fake_make_overlay(rc=0):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if rc == 0:
            ov = Path(cmd[3])
            ov.mkdir(parents=True, exist_ok=True)
            kb = kw.get("env", {}).get("ETH_IERISC_KB", "32")
            extra = "" if kb == "32" else f"ierisc_kb={kb}\n"
            (ov / e.MARK).write_text(f"src={Path(cmd[2]).resolve()}\nn_eth={cmd[4]}\n{extra}")
        return subprocess.CompletedProcess(cmd, rc, "", "make_overlay: refusing overlay X")
    return run, calls


def test_p150_default_makes_overlay_and_sets_eth(tmp_path):
    src = tmp_path / "tt-metal"; src.mkdir()
    env = {"TT_METAL_HOME": str(src), "GSPLAT_TT_ETH_OVERLAY": str(tmp_path / "ov")}
    run, calls = _fake_make_overlay()
    logs = []
    assert e.setup(env, ROOT, card="p150b", run=run, log=logs.append) == "eth"
    assert env["GSPLAT_TT_DISPATCH"] == "eth"
    assert env["TT_METAL_RUNTIME_ROOT"] == str(tmp_path / "ov")
    assert env["TT_METAL_CACHE_RENDER"] == str(tmp_path / "ov") + "-cache/render"
    assert len(calls) == 1 and calls[0][-1] == "12"
    # Second run reuses the matching overlay: no rebuild.
    env2 = {"TT_METAL_HOME": str(src), "GSPLAT_TT_ETH_OVERLAY": str(tmp_path / "ov")}
    assert e.setup(env2, ROOT, card="p150b", run=run, log=logs.append) == "eth"
    assert len(calls) == 1


def test_p100_stays_worker():
    env = {"TT_METAL_HOME": "/x"}
    run, calls = _fake_make_overlay()
    assert e.setup(env, ROOT, card="p100a", run=run, log=lambda m: None) == "worker"
    assert env["GSPLAT_TT_DISPATCH"] == "worker" and "TT_METAL_RUNTIME_ROOT" not in env
    assert not calls


def test_worker_override_on_p150():
    env = {"GSPLAT_TT_DISPATCH": "worker", "TT_METAL_HOME": "/x"}
    run, calls = _fake_make_overlay()
    assert e.setup(env, ROOT, card="p150a", run=run, log=lambda m: None) == "worker"
    assert "TT_METAL_RUNTIME_ROOT" not in env and not calls


def test_overlay_failure_falls_back_to_worker(tmp_path):
    env = {"TT_METAL_HOME": str(tmp_path), "GSPLAT_TT_ETH_OVERLAY": str(tmp_path / "ov")}
    run, _ = _fake_make_overlay(rc=2)
    logs = []
    assert e.setup(env, ROOT, card="p150a", run=run, log=logs.append) == "worker"
    assert env["GSPLAT_TT_DISPATCH"] == "worker" and "TT_METAL_RUNTIME_ROOT" not in env
    assert any("falling back to worker" in m for m in logs)
    assert e.setup({}, ROOT, card="p150a", run=run, log=logs.append) == "worker", "no TT_METAL_HOME"


def test_overlay_from_env_is_reused(tmp_path):
    ov = tmp_path / "ov"; ov.mkdir(); (ov / e.MARK).write_text("src=/s\nn_eth=12\n")
    env = {"TT_METAL_RUNTIME_ROOT": str(ov), "TT_METAL_CACHE_RENDER": "/c/render"}
    run, calls = _fake_make_overlay()
    assert e.setup(env, ROOT, card="p150b", run=run, log=lambda m: None) == "eth"
    assert env["TT_METAL_CACHE_RENDER"] == "/c/render" and not calls


def test_make_overlay_refuses_viewer_tree():
    # The real script refuses an overlay under the live viewer's tree: setup falls back.
    env = {"TT_METAL_HOME": str(ROOT), "GSPLAT_TT_ETH_OVERLAY": "/localdev/smarton/viewer/ttm-eth12"}
    assert e.setup(env, ROOT, card="p150b", log=lambda m: None) == "worker"


def test_profiler_uses_36kb_overlay_and_own_cache(tmp_path):
    # Task #731: with TT_METAL_DEVICE_PROFILER=1 the 32 KB overlay overflows (idle_erisc
    # cq_prefetch 0x4668 B > 0x4490 B); setup makes/reuses a p36 overlay with its own cache.
    src = tmp_path / "tt-metal"; src.mkdir()
    base = {"TT_METAL_HOME": str(src), "GSPLAT_TT_ETH_OVERLAY": str(tmp_path / "ov")}
    run, calls = _fake_make_overlay()
    assert e.setup(dict(base), ROOT, card="p150a", run=run, log=lambda m: None) == "eth"
    assert len(calls) == 1
    env = dict(base, TT_METAL_DEVICE_PROFILER="1", GSPLAT_TT_ETH_CACHE=str(tmp_path / "c"))
    assert e.setup(env, ROOT, card="p150a", run=run, log=lambda m: None) == "eth"
    assert env["TT_METAL_RUNTIME_ROOT"] == str(tmp_path / "ovp36")
    assert env["TT_METAL_CACHE_RENDER"] == str(tmp_path / "cp36") + "/render"
    assert len(calls) == 2 and e.overlay_info(str(tmp_path / "ovp36"))["ierisc_kb"] == "36"
    assert e.overlay_info(str(tmp_path / "ov")).get("ierisc_kb") is None, "default overlay untouched"
    env2 = dict(base, TT_METAL_DEVICE_PROFILER="1")
    assert e.setup(env2, ROOT, card="p150a", run=run, log=lambda m: None) == "eth"
    assert len(calls) == 2, "matching p36 overlay reused"
    assert env2["TT_METAL_CACHE"] == str(tmp_path / "ovp36") + "-cache/prod"
    off = dict(base, TT_METAL_DEVICE_PROFILER="0")
    assert e.setup(off, ROOT, card="p150a", run=run, log=lambda m: None) == "eth"
    assert off["TT_METAL_RUNTIME_ROOT"] == str(tmp_path / "ov") and len(calls) == 2


def test_profiler_replaces_32kb_overlay_from_env(tmp_path):
    src = tmp_path / "tt-metal"; src.mkdir()
    ov = tmp_path / "ov"; ov.mkdir(); (ov / e.MARK).write_text(f"src={src}\nn_eth=12\n")
    env = {"TT_METAL_RUNTIME_ROOT": str(ov), "TT_METAL_HOME": str(src),
           "GSPLAT_TT_ETH_OVERLAY": str(ov), "TT_METAL_DEVICE_PROFILER": "1"}
    run, calls = _fake_make_overlay()
    logs = []
    assert e.setup(env, ROOT, card="p150b", run=run, log=logs.append) == "eth"
    assert env["TT_METAL_RUNTIME_ROOT"] == str(ov) + "p36" and len(calls) == 1
    assert any("32 KB idle-ERISC bound" in m for m in logs)
    # A 36 KB overlay from env is kept as is.
    env3 = {"TT_METAL_RUNTIME_ROOT": str(ov) + "p36", "TT_METAL_DEVICE_PROFILER": "1"}
    assert e.setup(env3, ROOT, card="p150b", run=run, log=lambda m: None) == "eth" and len(calls) == 1
