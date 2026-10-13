"""opt/viewer/deploy_and_measure.py offline (task #613): no device, no network, no ssh.
renew.sh, viewer.sh status and the sweeps are faked at subprocess.run."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_and_measure as dm  # noqa: E402

HUD = ("Page: -- | Sent: 122.0 FPS | Render: 1024x1024 | Visible: 1,673,807 | "
       "Device render: {} ms | Host: 0.30 ms | Encode: 3.07 ms")
TAG_SHA = "445f9e404c25da2a17cbc7fa33ede667eec303a1"
NEW_SHA = "2346ba3194564ab1449079c2ac93bec49e90b496"


def sweep_line(fps, dev_ms):
    return "SWEEP " + json.dumps({"decoded_fps": fps, "frames": 2700, "interval_ms_p50": 8.2,
                                  "link_mbs": 12.3, "huds": [HUD.format(d) for d in dev_ms]})


def renew_line(rc=0, sha=NEW_SHA, status="ok"):
    return json.dumps({"role": "viewer", "host": "bh-30", "status": status, "exit": rc,
                       "deploy": f"current: {sha[:8]}", "deployed_sha": sha, "note": "",
                       "expiry_utc": "2026-10-14T06:00Z"})


def test_last_json_takes_the_json_line_among_noise():
    out = "[renew viewer] x\n{not json\n" + renew_line() + "\n"
    assert dm.last_json(out)["deployed_sha"] == NEW_SHA
    assert dm.last_json("no json here\n") is None


def test_parse_sweep_and_hud_median():
    s = dm.parse_sweep("noise\n" + sweep_line(121.0, [7.8, 7.9, 8.0]) + "\n")
    assert s["decoded_fps"] == 121.0
    assert dm.hud_device_ms(s["huds"]) == 7.9
    assert dm.parse_sweep("SWEEP {broken") is None
    assert dm.hud_device_ms(["no device here"]) is None


def test_hud_median_skips_warmup_when_enough_samples():
    huds = [HUD.format(10.5)] * 12 + [HUD.format(7.8)] * 30
    assert dm.hud_device_ms(huds) == 7.8


def test_status_sha():
    assert dm.status_sha(f"[viewer] running pid 123 sha={NEW_SHA}\n") == NEW_SHA
    assert dm.status_sha("[viewer] stopped sha=none") == ""


def test_choose_rev_keeps_a_deployed_descendant():
    assert dm.choose_rev(TAG_SHA, NEW_SHA, True)[0] == NEW_SHA
    assert dm.choose_rev(TAG_SHA, NEW_SHA, False) == (TAG_SHA, "")
    assert dm.choose_rev(TAG_SHA, "", False) == (TAG_SHA, "")
    assert dm.choose_rev(TAG_SHA, TAG_SHA, True) == (TAG_SHA, "")


def test_assemble_ok_and_error_shapes():
    ok = dm.assemble(NEW_SHA, [json.loads(sweep_line(f, [d])[6:]) for f, d in
                               ((121.0, 7.85), (120.9, 7.83), (122.4, 7.86))], 0, rev="best-iter-227")
    assert ok["page_fps"] == 121.0 and ok["device_ms"] == 7.85 and ok["rc"] == 0
    assert "error" not in ok and ok["rev"] == "best-iter-227"
    assert ok["page_fps_runs"] == [121.0, 120.9, 122.4]
    bad = dm.assemble("", [], 20, "viewer not up", ready="")
    assert bad == {"deployed_sha": None, "page_fps": None, "device_ms": None, "rc": 20,
                   "error": "viewer not up"}
    json.dumps(ok), json.dumps(bad)


class Fake:
    """subprocess.run stand-in: answers by the command's script name."""

    def __init__(self, renew_out="", renew_rc=0, status=f"[viewer] running pid 1 sha={NEW_SHA}",
                 sweeps=(), sweep_rc=0):
        self.renew_out, self.renew_rc, self.status = renew_out, renew_rc, status
        self.sweeps, self.sweep_rc, self.calls = list(sweeps), sweep_rc, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        s = " ".join(map(str, cmd))
        if s.startswith("git "):
            if "merge-base" in s:  # best-iter-227 is an ancestor of the deployed commit
                return SimpleNamespace(returncode=0 if cmd[-2] == TAG_SHA else 1, stdout="")
            if "rev-parse" in s:
                ref = cmd[-1].split("^")[0]
                sha = {"best-iter-227": TAG_SHA, "best-iter-999": ""}.get(ref, ref)
                if not sha:
                    raise subprocess.CalledProcessError(128, cmd)
                return SimpleNamespace(returncode=0, stdout=sha + "\n")
            return SimpleNamespace(returncode=0, stdout="")
        if s.endswith("viewer.sh status"):
            return SimpleNamespace(returncode=0, stdout=self.status + "\n", stderr="")
        if "renew.sh" in s:
            return SimpleNamespace(returncode=self.renew_rc, stdout=self.renew_out, stderr="")
        if " -c import" in s:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "ws_sweep_t532.py" in s:
            out = self.sweeps.pop(0) if self.sweeps else ""
            return SimpleNamespace(returncode=self.sweep_rc, stdout=out, stderr="boom")
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def live(monkeypatch, capsys):
    def go(fake, argv, ready=(True, "page 200, websocket 101")):
        monkeypatch.setattr(dm.subprocess, "run", fake)
        monkeypatch.setattr(dm, "wait_ready", lambda port, t: ready)
        monkeypatch.setattr(dm, "sweep_python", lambda: "/fake/python")
        monkeypatch.setenv("VIEWER_NO_LOCK", "1")
        rc = dm.main(argv)
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 1, lines  # exactly one JSON line
        out = json.loads(lines[0])
        assert out["rc"] == rc
        return out
    return go


def test_live_ok_keeps_deployed_descendant(live):
    fake = Fake(renew_out="[x]\n" + renew_line(), sweeps=[sweep_line(f, [7.85]) for f in (121, 120.9, 122.4)])
    out = live(fake, ["227"])
    assert out["rc"] == 0 and out["page_fps"] == 121 and out["device_ms"] == 7.85
    assert out["deployed_sha"] == NEW_SHA and "error" not in out
    renew = next(c for c in fake.calls if "renew.sh" in " ".join(map(str, c)))
    assert renew[-2:] == ["--rev", NEW_SHA]  # not the older tag
    assert "already contains" in out["deploy_note"]


def test_live_force_deploys_the_tag(live):
    fake = Fake(renew_out=renew_line(sha=TAG_SHA), sweeps=[sweep_line(120, [8.0])])
    out = live(fake, ["227", "--force", "--runs", "1"])
    renew = next(c for c in fake.calls if "renew.sh" in " ".join(map(str, c)))
    assert renew[-1] == TAG_SHA and out["deployed_sha"] == TAG_SHA and out["rc"] == 0


def test_live_renew_failure_passes_its_code(live):
    out = live(Fake(renew_out=renew_line(rc=5, status="no_free_box"), renew_rc=5), ["227"])
    assert out["rc"] == 5 and "no_free_box" in out["error"] and out["page_fps"] is None


def test_live_renew_without_json(live):
    out = live(Fake(renew_out="garbage", renew_rc=0), ["227"])
    assert out["rc"] == 1 and "no JSON" in out["error"]


def test_live_bad_rev(live):
    out = live(Fake(), ["999"])
    assert out["rc"] == 2 and "best-iter-999" in out["error"]


def test_live_not_ready(live):
    out = live(Fake(renew_out=renew_line()), ["227"], ready=(False, "page not 200, websocket 0"))
    assert out["rc"] == dm.RC_NOT_READY and "not up" in out["error"]


def test_live_sweep_failure(live):
    out = live(Fake(renew_out=renew_line(), sweeps=[sweep_line(121, [7.8]), ""]), ["227"])
    assert out["rc"] == dm.RC_SWEEP and "sweep r2" in out["error"]
    assert out["page_fps_runs"] == [121]


def test_live_lock_busy(live):
    out = live(Fake(renew_out=renew_line(), sweep_rc=75), ["227"])
    assert out["rc"] == 75 and "busy" in out["error"]


def test_usage_errors(live):
    assert live(Fake(), [])["rc"] == 2
    assert "8081" in live(Fake(), ["227", "--port", "8081"])["error"]


def test_no_deploy_skips_renew(live):
    fake = Fake(sweeps=[sweep_line(119, [8.1])])
    out = live(fake, ["227", "--no-deploy", "--runs", "1"])
    assert out["rc"] == 0 and out["deployed_sha"] == NEW_SHA
    assert not any("renew.sh" in " ".join(map(str, c)) for c in fake.calls)
