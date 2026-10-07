"""opt/sync_remote.sh with a fake ssh (task #324): no host is contacted.

The measurement path's ssh command lines stay as they were, the viewer box is
refused unless the viewer deploy asks for it, and the viewer deploy builds with
and links its own venv and scenes instead of /localdev/$USER/gstt2.
"""
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "opt" / "sync_remote.sh"

FAKE_SSH = """#!/usr/bin/env bash
n=$(find "$CAP" -name '*.argv' | wc -l | tr -d ' ')
printf '%s\\n' "$@" > "$CAP/$n.argv"
case "$*" in *"tar -m -x"*) cat > /dev/null; : > "$CAP/$n.stdin" ;; *) cat > "$CAP/$n.stdin" ;; esac
"""


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path, host, env_extra=None):
    fake = tmp_path / "fake"
    cap = tmp_path / "cap"
    fake.mkdir(exist_ok=True)
    cap.mkdir()
    _exe(fake / "ssh", FAKE_SSH)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GSTT2_", "SYNC_REMOTE_", "REMOTE_TT_METAL"))}
    env.update(CAP=str(cap), PATH=f"{fake}:{env['PATH']}", SYNC_ALL="0", **(env_extra or {}))
    proc = subprocess.run(["bash", str(SCRIPT), host, "/localdev/u/dst", "HEAD"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    calls = [((cap / f"{i}.argv").read_text().splitlines(), (cap / f"{i}.stdin").read_text(errors="replace"))
             for i in range(len(list(cap.glob("*.argv"))))]
    return proc, calls


def test_measurement_host_command_lines(tmp_path):
    proc, calls = _run(tmp_path, "yyzo-bh-04")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 2
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert calls[0][0] == ["-o", "BatchMode=yes", "yyzo-bh-04",
                           f"mkdir -p '/localdev/u/dst' && tar -m -x -C '/localdev/u/dst' && echo {sha} > '/localdev/u/dst/SHA'"]
    # Unchanged since before #324, including the double space of the empty expansions.
    assert calls[1][0] == ["-o", "BatchMode=yes", "yyzo-bh-04", "DIR='/localdev/u/dst'  bash -s"]


def test_viewer_host_refused_without_viewer_flag(tmp_path):
    proc, calls = _run(tmp_path, "bh-30")
    assert proc.returncode == 2
    assert "viewer box" in proc.stderr
    assert calls == []


def test_viewer_deploy_uses_viewer_dir(tmp_path):
    v = "/localdev/u/viewer"
    proc, calls = _run(tmp_path, "bh-30", dict(
        SYNC_REMOTE_VIEWER="1", GSTT2_BASE=v, GSTT2_VENV=f"{v}/venv", GSTT2_SCENES=f"{v}/scenes",
        REMOTE_TT_METAL_HOME=f"{v}/tt-metal"))
    assert proc.returncode == 0, proc.stderr
    assert calls[1][0][-1] == (f"DIR='/localdev/u/dst' TT_METAL_HOME='{v}/tt-metal' GSTT2_BASE='{v}' "
                               f"GSTT2_VENV='{v}/venv' GSTT2_SCENES='{v}/scenes' bash -s")
    assert "/localdev/smarton" not in calls[1][1]


def _remote(tmp_path, env_extra):
    """Run the remote half locally up to the (stubbed, failing) cmake configure."""
    _, calls = _run(tmp_path, "yyzo-bh-04")
    stub = tmp_path / "stub"
    stub.mkdir()
    _exe(stub / "cmake", "#!/bin/sh\nexit 1\n")
    _exe(stub / "md5sum", "#!/bin/sh\ncat \"$@\" >/dev/null 2>&1; echo 0 -\n")
    d = tmp_path / "dst"
    d.mkdir(exist_ok=True)
    env = dict(PATH=f"{stub}:/usr/bin:/bin", USER="u", HOME=str(tmp_path), DIR=str(d), **env_extra)
    proc = subprocess.run(["bash", "-s"], input=calls[1][1], env=env, capture_output=True, text=True)
    return proc, d


def test_remote_links_viewer_venv_and_scenes(tmp_path):
    v = tmp_path / "viewer"
    (v / "venv" / "bin").mkdir(parents=True)
    (v / "scenes").mkdir()
    (v / "venv" / "bin" / "activate").write_text("export ACTIVATED=1\n")
    d = tmp_path / "dst"
    d.mkdir()
    # A link left by an older deploy into a gstt2 tree is replaced.
    os.symlink("/localdev/u/gstt2/scenes", d / "scenes")
    proc, d = _remote(tmp_path, dict(GSTT2_BASE=str(v), GSTT2_VENV=str(v / "venv"),
                                     GSTT2_SCENES=str(v / "scenes")))
    assert proc.returncode == 1  # the stub cmake fails after the links are made
    assert os.readlink(d / ".venv") == str(v / "venv")
    assert os.readlink(d / "scenes") == str(v / "scenes")


def test_remote_default_links_gstt2_from_user(tmp_path):
    proc, d = _remote(tmp_path, {})
    assert proc.returncode == 1
    assert os.readlink(d / ".venv") == "/localdev/u/gstt2/.venv"
    assert os.readlink(d / "scenes") == "/localdev/u/gstt2/scenes"
