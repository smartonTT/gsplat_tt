"""opt/sync_remote.sh with a fake ssh and rsync (tasks #324, #676): no host is contacted.

The tree goes one way by rsync from the caller's own repo (refused outside the
project repos), with an include list; the viewer box is refused unless the viewer
deploy asks for it; the viewer deploy builds with and links its own venv and scenes.
"""
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "opt" / "sync_remote.sh"

FAKE_SSH = """#!/usr/bin/env bash
case "$*" in *"command -v rsync"*) exit 0 ;; esac
n=$(find "$CAP" -name '*.argv' | wc -l | tr -d ' ')
printf '%s\\n' "$@" > "$CAP/$n.argv"
case "$*" in
  *"bash -s"*) cat > "$CAP/$n.stdin" ;;
  *md5sum*) : > "$CAP/$n.stdin"; echo "${FAKE_MD5:-}" ;;
  *) : > "$CAP/$n.stdin" ;;
esac
"""

FAKE_RSYNC = """#!/usr/bin/env bash
n=$(find "$CAP" -name '*.argv' | wc -l | tr -d ' ')
printf '%s\\n' rsync "$@" > "$CAP/$n.argv"
: > "$CAP/$n.stdin"
for a in "$@"; do case $a in --files-from=*) cp "${a#--files-from=}" "$CAP/$n.list" ;; esac; done
echo "Number of regular files transferred: 2"
echo "Total transferred file size: 1,234 bytes"
"""


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _repo(tmp_path):
    """A small repo with files the list keeps and files it skips."""
    r = tmp_path / "repo"
    files = {
        "CMakeLists.txt": "x", "render/a.cpp": "x", "render/kernels/k.cpp": "x", "opt/s.py": "x",
        "docs/t1/remote.sh": "x", "docs/t1/out/run.log": "x", "docs/t1/big.json": "x" * 1048577,
        "opt/metal-screenshots/a.png": "x", "opt/profiler/ttw-1/p.csv": "x", "x/__pycache__/m.pyc": "x",
        "tests/fixtures/hero/h.npz": "x", "a.tracy": "x",
    }
    for f, t in files.items():
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        (r / f).write_text(t)
    subprocess.check_call(["git", "init", "-q", str(r)])
    _git(r, "add", "-A")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (r / "scenes").mkdir()
    (r / "scenes" / "bicycle.ply").write_text("ply")
    return r


def _run(tmp_path, host, *args, env_extra=None, repo=None, allowed=True):
    repo = repo or _repo(tmp_path)
    fake = tmp_path / "fake"
    cap = tmp_path / "cap"
    fake.mkdir(exist_ok=True)
    cap.mkdir(exist_ok=True)
    for f in cap.iterdir():
        f.unlink()
    _exe(fake / "ssh", FAKE_SSH)
    _exe(fake / "rsync", FAKE_RSYNC)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GSTT2_", "SYNC_", "REMOTE_TT_METAL"))}
    env.update(CAP=str(cap), PATH=f"{fake}:{env['PATH']}", USER="u", **(env_extra or {}))
    if allowed:
        env["SYNC_REMOTE_ALLOWED_ROOTS"] = str(repo.resolve())
    # args: options, then "--pos" and the positionals after the host
    i = args.index("--pos") if "--pos" in args else len(args)
    argv = ["bash", str(SCRIPT), *args[:i], host, *args[i + 1:]]
    proc = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True)
    calls = []
    for i in range(len(list(cap.glob("*.argv")))):
        lst = cap / f"{i}.list"
        calls.append(((cap / f"{i}.argv").read_text().splitlines(),
                      (cap / f"{i}.stdin").read_text(errors="replace"),
                      lst.read_text().splitlines() if lst.exists() else None))
    return proc, calls, repo


def test_sync_sends_include_list_one_way(tmp_path):
    proc, calls, repo = _run(tmp_path, "yyzo-bh-04", "--pos", "/localdev/u/dst", "HEAD",
                             env_extra=dict(FAKE_MD5="5b2b0b0a2c3e0b2f6b2a5a0f7c1c0d4e"))
    assert proc.returncode == 0, proc.stderr
    sha = _git(repo, "rev-parse", "HEAD")
    rs = calls[0]
    assert rs[0][0] == "rsync"
    assert any("StrictHostKeyChecking=yes" in a for a in rs[0]), rs[0]
    assert rs[0][-1] == "yyzo-bh-04:/localdev/u/dst/"
    assert sorted(rs[2]) == sorted(["CMakeLists.txt", "docs/t1/remote.sh", "opt/s.py", "render/a.cpp",
                                    "render/kernels/k.cpp", "SHA", "SYNC_STAMP"])
    stage = repo / "tmp" / "sync-stage" / sha
    assert (stage / "SHA").read_text().strip() == sha
    assert f"source={repo.resolve()}" in (stage / "SYNC_STAMP").read_text()
    assert "sent 2 of 7 files, 1234 bytes" in proc.stdout
    build = [c for c in calls if "bash -s" in c[0][-1]]
    assert len(build) == 1 and build[0][0][-1].startswith("DIR='/localdev/u/dst' ")
    # the scene md5 differs from the remote's: copied after the build step
    scene = [c for c in calls if c[0][0] == "rsync" and any(a.endswith("scenes/bicycle.ply") for a in c[0])]
    assert len(scene) == 1 and scene[0][0][-1] == "yyzo-bh-04:/localdev/u/dst/scenes/bicycle.ply"
    assert calls.index(scene[0]) > calls.index(build[0])


def test_scene_with_same_md5_is_not_copied(tmp_path):
    md5 = subprocess.run(["md5", "-q", "/dev/stdin"], input="ply", capture_output=True, text=True).stdout.strip()
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--pos", "/localdev/u/dst", env_extra=dict(FAKE_MD5=md5))
    assert proc.returncode == 0, proc.stderr
    assert "scene bicycle.ply ok" in proc.stdout
    assert [c for c in calls if c[0][0] == "rsync"] == [calls[0]]


def test_default_remote_dir_is_project_owned(tmp_path):
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert calls[0][0][-1] == "yyzo-bh-04:/localdev/u/gstt2-sync/"


def test_dry_run_and_list_do_not_build(tmp_path):
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1 and "-n" in calls[0][0]
    assert "would send 2 of 7 files" in proc.stdout
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--list", repo=tmp_path / "repo")
    assert proc.returncode == 0, proc.stderr
    assert calls == [] and "5 files" in proc.stdout


def test_source_outside_project_repos_refused(tmp_path):
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--pos", "/localdev/u/dst", allowed=False)
    assert proc.returncode == 2
    assert "refusing source" in proc.stderr
    assert calls == []


def test_home_and_dev_never_synced(tmp_path):
    repo = _repo(tmp_path)
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--pos", "/localdev/u/dst", repo=repo,
                          env_extra=dict(HOME=str(repo)))
    assert proc.returncode == 2 and "never sync" in proc.stderr
    assert calls == []


def test_pull_stays_inside_worktree(tmp_path):
    repo = _repo(tmp_path)
    proc, calls, _ = _run(tmp_path, "yyzo-bh-04", "--pull", "tmp/out", "res", "--pos", "/localdev/u/dst", repo=repo)
    assert proc.returncode == 0, proc.stderr
    assert calls[0][0][-2:] == ["yyzo-bh-04:/localdev/u/dst/tmp/out/", f"{(repo / 'res').resolve()}/"]
    for bad in (["--pull", "../x", "res"], ["--pull", "/etc", "res"], ["--pull", "out", str(tmp_path / "elsewhere")]):
        proc, calls, _ = _run(tmp_path, "yyzo-bh-04", *bad, "--pos", "/localdev/u/dst", repo=repo)
        assert proc.returncode == 2, bad
        assert calls == []


def test_viewer_host_refused_without_viewer_flag(tmp_path):
    proc, calls, _ = _run(tmp_path, "bh-30", "--pos", "/localdev/u/dst")
    assert proc.returncode == 2
    assert "viewer box" in proc.stderr
    assert calls == []


def test_viewer_list_is_local_only(tmp_path):
    proc, calls, _ = _run(tmp_path, "bh-30", "--list")
    assert proc.returncode == 0, proc.stderr
    assert calls == []


def test_viewer_deploy_uses_viewer_dir(tmp_path):
    v = "/localdev/u/viewer"
    proc, calls, _ = _run(tmp_path, "bh-30", "--pos", "/localdev/u/dst", env_extra=dict(
        SYNC_REMOTE_VIEWER="1", GSTT2_BASE=v, GSTT2_VENV=f"{v}/venv", GSTT2_SCENES=f"{v}/scenes",
        REMOTE_TT_METAL_HOME=f"{v}/tt-metal"))
    assert proc.returncode == 0, proc.stderr
    build = [c for c in calls if "bash -s" in c[0][-1]][0]
    assert build[0][-1] == (f"DIR='/localdev/u/dst' VIEWER_BUILD=1 JOBS='16' TT_METAL_HOME='{v}/tt-metal' "
                            f"GSTT2_BASE='{v}' GSTT2_VENV='{v}/venv' GSTT2_SCENES='{v}/scenes' bash -s")
    assert "/localdev/smarton" not in build[1]


def _remote(tmp_path, env_extra):
    """Run the remote half locally up to the (stubbed, failing) cmake configure."""
    _, calls, _ = _run(tmp_path, "yyzo-bh-04", "--pos", "/localdev/u/dst")
    script = [c for c in calls if "bash -s" in c[0][-1]][0][1]
    stub = tmp_path / "stub"
    stub.mkdir()
    _exe(stub / "cmake", "#!/bin/sh\nexit 1\n")
    _exe(stub / "md5sum", "#!/bin/sh\ncat \"$@\" >/dev/null 2>&1; echo 0 -\n")
    d = tmp_path / "dst"
    d.mkdir(exist_ok=True)
    env = dict(PATH=f"{stub}:/usr/bin:/bin", USER="u", HOME=str(tmp_path), DIR=str(d), **env_extra)
    proc = subprocess.run(["bash", "-s"], input=script, env=env, capture_output=True, text=True)
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


FAKE_SSH_NO_RSYNC = """#!/usr/bin/env bash
# A host without rsync: the remote command runs here, against a local remote_dir.
while [ "${1#-}" != "$1" ]; do case $1 in -o) shift 2 ;; *) shift ;; esac; done
shift  # host
case "$*" in *"command -v rsync"*) echo NO_RSYNC; exit 0 ;; esac
echo "$*" >> "$CAP/ssh.log"
exec bash -c "$*"
"""


def test_host_without_rsync_gets_changed_files_by_tar(tmp_path):
    repo = _repo(tmp_path)
    fake, cap, dst = tmp_path / "fake", tmp_path / "cap", tmp_path / "remote" / "dst"
    fake.mkdir()
    cap.mkdir()
    _exe(fake / "ssh", FAKE_SSH_NO_RSYNC)
    _exe(fake / "rsync", "#!/bin/sh\necho rsync must not run >&2; exit 9\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GSTT2_", "SYNC_", "REMOTE_TT_METAL"))}
    env.update(CAP=str(cap), PATH=f"{fake}:{env['PATH']}", USER="u",
               SYNC_REMOTE_ALLOWED_ROOTS=str(repo.resolve()))

    def sync(*a):
        return subprocess.run(["bash", str(SCRIPT), *a, "yyzo-bh-49", str(dst), "HEAD"],
                              cwd=repo, env=env, capture_output=True, text=True)

    p = sync("--no-build")
    assert p.returncode == 0, p.stderr
    assert "has no rsync" in p.stdout and "sent 7 of 7 files" in p.stdout, p.stdout
    assert (dst / "render/kernels/k.cpp").read_text() == "x"
    assert not (dst / "opt/metal-screenshots").exists() and not (dst / "a.tracy").exists()
    assert not list(dst.rglob("._*"))
    p = sync("--dry-run")
    assert p.returncode == 0, p.stderr
    assert "would send 1 of 7 files" in p.stdout, p.stdout   # only SYNC_STAMP's time changes
    (repo / "render" / "a.cpp").write_text("y")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "edit")
    p = sync("--no-build")
    assert p.returncode == 0, p.stderr
    assert "sent 3 of 7 files" in p.stdout, p.stdout   # a.cpp, SHA, SYNC_STAMP
    assert (dst / "render/a.cpp").read_text() == "y"
    (dst / "out").mkdir()
    (dst / "out" / "r.json").write_text("{}")
    p = subprocess.run(["bash", str(SCRIPT), "--pull", "out", "tmp/got", "yyzo-bh-49", str(dst)],
                       cwd=repo, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (repo / "tmp/got/r.json").read_text() == "{}"
