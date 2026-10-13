"""Task #581: alternate-frame rendering on 2 chips (render/afr.py, run.py GSPLAT_TT_AFR=1),
CPU only: the view split and merge, worker argv/env, the coordinator against fake worker
processes (stagger, merged md5, a crashing worker, differing hero frames) and the run.py
_back_to_back worker hooks.

    python3 tests/unit/test_afr.py      (or pytest)
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "render"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

import afr  # noqa: E402

N_VIEWS = 30


def _digest(i):
    return hashlib.md5(f"view{i}".encode()).hexdigest()


# Fake worker: speaks the protocol like run.py --afr-worker K/N, renders view i as _digest(i).
_FAKE = r'''
import hashlib, os, sys, time
sys.path.insert(0, sys.argv[1])
import afr
k, n = afr.parse_worker(sys.argv[2])
n_views, n_passes = int(sys.argv[3]), int(sys.argv[4])
stamp = os.environ.get("FAKE_STAMPS")
mode = os.environ.get("FAKE_MODE", "")
def note(what):
    if stamp:
        with open(os.path.join(stamp, f"{what}{k}"), "w") as f:
            f.write(repr(afr.now()))
note("start")
print(f"worker {k} warming up B2B scene=fake", flush=True)  # forwarded, prefixed
time.sleep(0.05)
if mode == f"crash{k}":
    sys.exit(3)
idx = afr.split(n_views, n)[k]
link = afr.WorkerLink(k, n)
note("ready")
for p in range(n_passes):
    link.before_pass(p)
    time.sleep(0.002 * len(idx))
    link.mark_end()
    d = [hashlib.md5(f"view{i}".encode()).hexdigest() for i in idx]
    link.after_pass(p, d if p == 0 else None)
hero = "h1" if mode == "herodiff" and k == 1 else "h0"
link.done(0, hero)
'''


def _fake_cmds(tmp, n_passes, devs=(0, 1), env_extra=None):
    script = pathlib.Path(tmp) / "fake_worker.py"
    script.write_text(_FAKE)
    cmds = []
    for k, d in enumerate(devs):
        env = afr.worker_env(dict(os.environ, **(env_extra or {})), k, d)
        cmds.append(([sys.executable, str(script), str(ROOT / "render"),
                      f"{k}/{len(devs)}", str(N_VIEWS), str(n_passes)], env))
    return cmds


def test_split_and_merge_keep_the_sweep_md5():
    parts = afr.split(N_VIEWS, 2)
    assert parts[0] == list(range(0, 30, 2)) and parts[1] == list(range(1, 30, 2))
    assert afr.split(5, 2) == [[0, 2, 4], [1, 3]]
    full = [_digest(i) for i in range(N_VIEWS)]
    per_chip = [[full[i] for i in p] for p in parts]
    assert afr.merge(per_chip, N_VIEWS) == full
    assert afr.raw_md5(afr.merge(per_chip, N_VIEWS)) == \
        hashlib.md5("".join(full).encode()).hexdigest()[:8]
    try:
        afr.merge([per_chip[0], per_chip[1][:-1]], N_VIEWS)
        raise AssertionError("short chip list accepted")
    except ValueError:
        pass


def test_period_is_the_union_window_per_view():
    # chip 0 runs 1.0..1.06 s, chip 1 1.01..1.07 s, 30 views -> 70 ms / 30
    assert abs(afr.period_ms([(1.0, 1.06), (1.01, 1.07)], 30) - 70.0 / 30) < 1e-9


def test_devices_and_flags():
    assert not afr.enabled({}) and not afr.enabled({"GSPLAT_TT_AFR": "0"})
    assert afr.enabled({"GSPLAT_TT_AFR": "1"})
    assert afr.devices({}) == [0, 1]
    assert afr.devices({"GSPLAT_TT_AFR_DEVICES": "2,3"}) == [2, 3]
    for bad in ("0", "0,0", "a,b", "-1,0"):
        try:
            afr.devices({"GSPLAT_TT_AFR_DEVICES": bad})
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    assert afr.parse_worker("1/2") == (1, 2)
    for bad in ("2/2", "0/1", "x"):
        try:
            afr.parse_worker(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_worker_env_picks_one_chip():
    base = {"GSPLAT_TT_AFR": "1", "TT_VISIBLE_DEVICES": "0,1,2,3", "X": "y"}
    e = afr.worker_env(base, 1, 3)
    assert e["TT_VISIBLE_DEVICES"] == "3" and e["GSPLAT_TT_DEVICE_ID"] == "0"
    assert e["GSPLAT_TT_AFR"] == "0" and e["GSPLAT_TT_AFR_WORKER"] == "1" and e["X"] == "y"
    assert base["GSPLAT_TT_AFR"] == "1"  # not mutated
    e = afr.worker_env(dict(base, GSPLAT_TT_AFR_VISIBLE="0"), 1, 3)
    assert "TT_VISIBLE_DEVICES" not in e and e["GSPLAT_TT_DEVICE_ID"] == "3"


def test_worker_argv():
    argv = ["--back-to-back", "--iter-dir", "x", "--dump-views", "dv"]
    assert afr.worker_argv(argv, 0, 2) == \
        ["--back-to-back", "--dump-views", "dv", "--iter-dir", "x", "--afr-worker", "0/2"]
    assert afr.worker_argv(argv, 1, 2) == \
        ["--back-to-back", "--dump-views", "dv", "--no-ref", "--iter-dir", "x/afr1",
         "--afr-worker", "1/2"]
    assert afr.worker_argv(["--iter-dir=y", "--afr-worker", "0/2"], 1, 2) == \
        ["--no-ref", "--iter-dir", "y/afr1", "--afr-worker", "1/2"]
    assert afr.worker_argv([], 1, 2)[-4:] == ["--iter-dir", "render-clean/afr1",
                                               "--afr-worker", "1/2"]


def test_coordinator_two_fake_chips():
    logs = []
    with tempfile.TemporaryDirectory() as tmp:
        stamps = pathlib.Path(tmp) / "stamps"
        stamps.mkdir()
        cmds = _fake_cmds(tmp, 4, env_extra={"FAKE_STAMPS": str(stamps)})
        rc, info = afr.coordinate(cmds, 4, N_VIEWS, log=logs.append, timeout_s=60)
        t = {p.name: float(p.read_text()) for p in stamps.iterdir()}
    assert rc == 0, logs
    # Stagger: worker 1 starts only after worker 0 finished its warm-up.
    assert t["start1"] >= t["ready0"]
    assert info["digests"] == [_digest(i) for i in range(N_VIEWS)]
    assert len(info["periods"]) == 4 and all(p > 0 for p in info["periods"])
    assert info["hero_md5"] == ["h0", "h0"] and info["worker_rc"] == [0, 0]
    # Worker output is forwarded with a prefix, never as a bare B2B line.
    assert any(l.startswith("[afr1] worker 1 warming up") for l in logs)
    assert not any(l.startswith("B2B ") for l in logs)
    line, ms, med = afr.summary_line(info, "fake", N_VIEWS, 1, False, 0.0, [0, 1], "out")
    kv = dict(x.split("=", 1) for x in line.split()[1:])
    assert kv["raw_md5"] == afr.raw_md5([_digest(i) for i in range(N_VIEWS)])
    assert kv["passes"] == "2" and kv["afr_chips"] == "2" and kv["afr_hero_equal"] == "yes"
    assert kv["identical_across_passes"] == "yes"
    # AFR relation, not wall-clock bounds (host sleeps are coarse): two chips in parallel
    # give a period of about half of one chip's time per own view.
    chip = statistics.median(x for c in info["chip_ms"] for x in c)
    assert abs(2 * med - chip) <= 0.1 * chip, (med, chip)


def test_coordinator_broken_pipe_on_go_fails_cleanly():
    class _DeadStdin:
        def write(self, s):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

        def close(self):
            pass

    def popen(*a, **kw):
        p = subprocess.Popen(*a, **kw)
        p._real_stdin, p.stdin = p.stdin, _DeadStdin()  # worker stays alive, waiting for GO
        return p

    logs = []
    with tempfile.TemporaryDirectory() as tmp:
        cmds = _fake_cmds(tmp, 2)
        rc, info = afr.coordinate(cmds, 2, N_VIEWS, log=logs.append, popen=popen, timeout_s=60)
    assert rc == 8, (rc, info, logs)
    assert info["periods"] == []
    assert any("stdin closed" in l for l in logs)


def test_coordinator_worker_crash_stops_the_other():
    logs = []
    with tempfile.TemporaryDirectory() as tmp:
        cmds = _fake_cmds(tmp, 3, env_extra={"FAKE_MODE": "crash1"})
        rc, info = afr.coordinate(cmds, 3, N_VIEWS, log=logs.append, timeout_s=60)
    assert rc == 3, (rc, info, logs)
    assert info["periods"] == []


def test_coordinator_flags_differing_hero():
    with tempfile.TemporaryDirectory() as tmp:
        cmds = _fake_cmds(tmp, 2, env_extra={"FAKE_MODE": "herodiff"})
        rc, info = afr.coordinate(cmds, 2, N_VIEWS, log=lambda m: None, timeout_s=60)
    assert rc == 5 and info["hero_md5"] == ["h0", "h1"]
    line, _, _ = afr.summary_line(info, "fake", N_VIEWS, 0, False, 0.0, [0, 1], "out")
    assert "afr_hero_equal=NO" in line


def test_run_py_coordinator_end_to_end():
    """run.py _afr_coordinate: worker argv/env go to popen; the merged B2B line and the
    TTW_TIMING lines are printed once, by the coordinator."""
    from test_run_back_to_back import _load_run
    run = _load_run()
    seen = []
    with tempfile.TemporaryDirectory() as tmp:
        script = pathlib.Path(tmp) / "fake_worker.py"
        script.write_text(_FAKE)
        cams = pathlib.Path(tmp) / "cams.json"
        cams.write_text(json.dumps({"s": {"order": [f"v{i}" for i in range(N_VIEWS)]}}))

        def popen(argv, env, **kw):
            seen.append((argv, env))
            k = int(argv[argv.index("--afr-worker") + 1].split("/")[0])
            fake = [sys.executable, str(script), str(ROOT / "render"), f"{k}/2",
                    str(N_VIEWS), "4"]
            return subprocess.Popen(fake, env=env, **kw)

        args = argparse.Namespace(back_to_back=True, cameras=cams, scene="s", view_range=None,
                                  b2b_warmup=1, b2b_passes=2, b2b_drop=False,
                                  view_gap_ms=0.0, iter_dir="it")
        out = io.StringIO()
        env = dict(os.environ, GSPLAT_TT_AFR="1", GSPLAT_TT_AFR_DEVICES="4,5")
        with contextlib.redirect_stdout(out):
            rc = run._afr_coordinate(args, ["--back-to-back", "--iter-dir", "it"], env, popen)
    text = out.getvalue()
    assert rc == 0, text
    assert [e["TT_VISIBLE_DEVICES"] for _, e in seen] == ["4", "5"]
    assert seen[1][0][-5:] == ["--no-ref", "--iter-dir", "it/afr1", "--afr-worker", "1/2"]
    b2b = [l for l in text.splitlines() if l.startswith("B2B ")]
    assert len(b2b) == 1 and "afr_devices=4,5" in b2b[0]
    assert f"raw_md5={afr.raw_md5([_digest(i) for i in range(N_VIEWS)])}" in b2b[0]
    assert sum(l.startswith("TTW_TIMING b2b_ms_frame") for l in text.splitlines()) == 2

    args.back_to_back = False
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert run._afr_coordinate(args, [], env, popen) == 2


def test_back_to_back_worker_hooks():
    """run.py _back_to_back with a WorkerLink: waits for GO per pass, reports pass windows
    and pass 0's digests, dumps views under their sweep positions."""
    from test_run_back_to_back import _FakePipeline, _load_run
    run = _load_run()
    order = ["b", "d"]
    cam = {"views": {n: {"c2w": np.eye(4, dtype=np.float32) * (i + 2)}
                     for i, n in enumerate(order)}}
    args = argparse.Namespace(scene="t", b2b_passes=2, b2b_drop=False, b2b_warmup=1)
    inp = io.StringIO("".join(f"GO {p}\n" for p in range(4)))
    msgs = io.StringIO()
    link = afr.WorkerLink(1, 2, inp=inp, out=msgs)
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        d = pathlib.Path(d)
        (d / "dump").mkdir()
        rc = run._back_to_back(args, _FakePipeline(), None, cam, order, None, 8, 8, "a",
                               np.zeros((8, 8, 3), np.float32), d / "dump", d,
                               afr=link, idx=[1, 3])
        dumped = sorted(p.name for p in (d / "dump").iterdir())
    assert rc == 0
    assert dumped == ["view01_b.png", "view03_d.png"]
    evs = [json.loads(l[len(afr.MSG):]) for l in msgs.getvalue().splitlines()]
    assert [e["ev"] for e in evs] == ["ready", "pass", "pass", "pass", "pass"]
    assert [e["p"] for e in evs[1:]] == [0, 1, 2, 3]
    assert len(evs[1]["digests"]) == 2 and all(e["digests"] is None for e in evs[2:])
    assert all(e["t1"] >= e["t0"] > 0 for e in evs[1:])


def test_worker_stops_without_go():
    link = afr.WorkerLink(0, 2, inp=io.StringIO(""), out=io.StringIO())
    try:
        link.before_pass(0)
        raise AssertionError("ran without GO")
    except SystemExit:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
