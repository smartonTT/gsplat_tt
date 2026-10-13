"""Per-landing viewer duty in one step (task #613): deploy best-iter-<N> to the live viewer
and measure end-to-end page FPS through the Mac tunnel (localhost:8091). See
opt/viewer/deploy_and_measure.sh for usage; this file holds the logic and the parsing that
opt/viewer/test_deploy_and_measure.py tests offline.

Steps:
  1. resolve best-iter-<N> (or --rev). If the viewer already runs a descendant of it (newer
     viewer-only commits on top of that best), keep the deployed commit: deploying the tag
     would roll those back. renew.sh then only extends the reservation.
  2. opt/ird/renew.sh viewer --rev <rev>: extend the viewer reservation, deploy when the tree
     differs (renew.sh takes `ttp lock viewer` itself for the deploy).
  3. wait until http://127.0.0.1:8091/ answers 200 and a websocket upgrade answers 101.
  4. --runs sweeps of opt/viewer/ws_sweep_t532.py --no-decode over the tunnel, each under
     `ttp lock viewer`, with the Mac venv that has msgspec/zstandard/websockets.
Prints exactly one JSON line on stdout: {deployed_sha, page_fps, device_ms, rc, error?, ...};
rc is also the exit code. Progress goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBPROTOCOL = "viser-v1.0.27"

# Exit codes besides renew.sh's own (2-6, 10-13, passed through unchanged).
RC_USAGE = 2
RC_NOT_READY = 20
RC_SWEEP = 21
RC_SWEEP_PY = 22
RC_LOCK_BUSY = 75

_HUD_DEV = re.compile(r"Device render:\s*([0-9.]+)\s*ms")
_STATUS_SHA = re.compile(r"\bsha=([0-9a-f]{7,40})\b")


def log(msg: str) -> None:
    print(f"[deploy_and_measure] {msg}", file=sys.stderr, flush=True)


def last_json(text: str) -> dict | None:
    """The last stdout line that is a JSON object (renew.sh prints exactly one)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_sweep(text: str) -> dict | None:
    """The SWEEP {...} line of ws_sweep.py / ws_sweep_t532.py."""
    for line in reversed(text.splitlines()):
        if line.startswith("SWEEP "):
            try:
                return json.loads(line[len("SWEEP "):])
            except ValueError:
                return None
    return None


def hud_device_ms(huds: list[str], skip: int = 12) -> float | None:
    """Median 'Device render' ms of the viewer's HUD lines (4 Hz). The first ``skip`` (the
    sweep's 3 s warm-up) are left out when enough remain."""
    vals = [float(m.group(1)) for h in huds if (m := _HUD_DEV.search(h))]
    if len(vals) > 2 * skip:
        vals = vals[skip:]
    return round(statistics.median(vals), 2) if vals else None


def status_sha(text: str) -> str:
    """sha=<...> from `opt/viewer/viewer.sh status` ('' if none)."""
    m = _STATUS_SHA.search(text)
    return m.group(1) if m else ""


def choose_rev(target: str, deployed: str, deployed_contains_target: bool) -> tuple[str, str]:
    """(rev for renew.sh, note). Keep the deployed commit when it already contains the target."""
    if deployed and deployed != target and deployed_contains_target:
        return deployed, f"deployed {deployed[:8]} already contains {target[:8]}: kept"
    return target, ""


def summarize(sweeps: list[dict]) -> dict:
    fps = [float(s["decoded_fps"]) for s in sweeps]
    dev = [d for s in sweeps if (d := hud_device_ms(s.get("huds") or [])) is not None]
    return {
        "page_fps": round(statistics.median(fps), 2) if fps else None,
        "device_ms": round(statistics.median(dev), 2) if dev else None,
        "page_fps_runs": fps,
        "device_ms_runs": dev,
        "interval_ms_p50": [s.get("interval_ms_p50") for s in sweeps],
        "link_mbs": [s.get("link_mbs") for s in sweeps],
    }


def assemble(deployed_sha: str, sweeps: list[dict], rc: int, error: str = "", **extra) -> dict:
    """The one output line. page_fps/device_ms are medians over the sweep runs."""
    out = {"deployed_sha": deployed_sha or None, "page_fps": None, "device_ms": None, "rc": rc}
    if sweeps:
        s = summarize(sweeps)
        out["page_fps"], out["device_ms"] = s.pop("page_fps"), s.pop("device_ms")
        out.update(s)
    if error:
        out["error"] = error
    out.update({k: v for k, v in extra.items() if v not in (None, "")})
    return out


# ---- live steps -------------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO,
                          capture_output=True).returncode == 0


def page_ok(port: int, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ws_status(port: int, timeout: float = 5.0) -> int:
    """HTTP status of a websocket upgrade request with the viewer's subprotocol (101 = up)."""
    req = (f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
           "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
           f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: {SUBPROTOCOL}\r\n\r\n")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall(req.encode())
            line = s.recv(256).split(b"\r\n", 1)[0].decode(errors="replace")
    except OSError:
        return 0
    parts = line.split()
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def wait_ready(port: int, timeout_s: float, every_s: float = 5.0) -> tuple[bool, str]:
    t_end = time.monotonic() + timeout_s
    while True:
        p, w = page_ok(port), ws_status(port)
        if p and w == 101:
            return True, "page 200, websocket 101"
        if time.monotonic() >= t_end:
            return False, f"page {'200' if p else 'not 200'}, websocket {w or 'no answer'}"
        time.sleep(every_s)


def sweep_python() -> str:
    env = os.environ.get("VIEWER_SWEEP_PY")
    if env:
        return env
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).parent
    for root in (REPO, common):
        p = root / "venv-macos" / "bin" / "python"
        if p.exists():
            return str(p)
    return sys.executable


def locked(cmd: list[str]) -> list[str]:
    return (["ttp", "lock", "viewer", "--", *cmd]
            if shutil.which("ttp") and not os.environ.get("VIEWER_NO_LOCK") else cmd)


def run(args) -> dict:
    extra: dict = {}
    try:
        subprocess.run(["git", "fetch", "-q", "--tags", "origin"], cwd=REPO, timeout=60,
                       capture_output=True)
    except subprocess.TimeoutExpired:
        log("git fetch timed out: using local refs")
    rev = args.rev or f"best-iter-{args.n}"
    extra["rev"] = rev
    try:
        target = git("rev-parse", "--verify", f"{rev}^{{commit}}")
    except subprocess.CalledProcessError:
        return assemble("", [], RC_USAGE, f"cannot resolve {rev}", **extra)

    deployed_sha = ""
    if not args.no_deploy:
        renew_rev = target
        if not args.force:
            st = subprocess.run([str(REPO / "opt/viewer/viewer.sh"), "status"], cwd=REPO,
                                capture_output=True, text=True, timeout=90)
            dep = status_sha(st.stdout)
            try:
                dep = git("rev-parse", "--verify", f"{dep}^{{commit}}") if dep else ""
            except subprocess.CalledProcessError:
                dep = ""
            renew_rev, note = choose_rev(target, dep, bool(dep) and is_ancestor(target, dep))
            if note:
                extra["deploy_note"] = note
                log(note)
        log(f"renew.sh viewer --rev {renew_rev}")
        r = subprocess.run([str(REPO / "opt/ird/renew.sh"), "viewer", "--rev", renew_rev],
                           cwd=REPO, capture_output=True, text=True)
        sys.stderr.write(r.stderr[-4000:])
        rj = last_json(r.stdout)
        if rj is None:
            return assemble("", [], r.returncode or 1,
                            f"renew.sh printed no JSON (exit {r.returncode})", **extra)
        deployed_sha = rj.get("deployed_sha", "")
        extra.update(host=rj.get("host"), renew_status=rj.get("status"), deploy=rj.get("deploy"),
                     expiry_utc=rj.get("expiry_utc"))
        if r.returncode != 0:
            return assemble(deployed_sha, [], r.returncode,
                            f"renew.sh exit {r.returncode}: {rj.get('status')}: "
                            f"{rj.get('note') or rj.get('deploy')}", **extra)
    else:
        st = subprocess.run([str(REPO / "opt/viewer/viewer.sh"), "status"], cwd=REPO,
                            capture_output=True, text=True, timeout=90)
        deployed_sha = status_sha(st.stdout)

    ok, why = wait_ready(args.port, args.ready_timeout)
    extra["ready"] = why
    if not ok:
        return assemble(deployed_sha, [], RC_NOT_READY,
                        f"viewer not up on localhost:{args.port} after {args.ready_timeout:.0f} s: {why}",
                        **extra)

    py = sweep_python()
    chk = subprocess.run([py, "-c", "import msgspec, zstandard, websockets, numpy"],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        return assemble(deployed_sha, [], RC_SWEEP_PY,
                        f"{py} lacks the sweep deps (msgspec, zstandard, websockets, numpy); "
                        "set VIEWER_SWEEP_PY", **extra)
    sweeps = []
    for i in range(args.runs):
        cmd = locked([py, str(REPO / "opt/viewer/ws_sweep_t532.py"),
                      "--url", f"ws://127.0.0.1:{args.port}", "--subprotocol", SUBPROTOCOL,
                      "--no-decode", "--secs", str(args.secs), "--label", f"r{i + 1}"])
        log(f"sweep r{i + 1}/{args.runs}")
        s = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if s.returncode == RC_LOCK_BUSY:
            return assemble(deployed_sha, sweeps, RC_LOCK_BUSY, "viewer lock busy", **extra)
        sj = parse_sweep(s.stdout)
        if s.returncode != 0 or not sj or not sj.get("frames"):
            tail = " | ".join((s.stderr or s.stdout).strip().splitlines()[-3:])
            return assemble(deployed_sha, sweeps, RC_SWEEP,
                            f"sweep r{i + 1} exit {s.returncode}: {tail}", **extra)
        sweeps.append(sj)
    return assemble(deployed_sha, sweeps, 0, **extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("n", nargs="?", type=int, help="iteration N of best-iter-<N>")
    ap.add_argument("--rev", default="", help="deploy this ref instead of best-iter-<N>")
    ap.add_argument("--force", action="store_true",
                    help="deploy the rev even when the viewer runs a descendant of it")
    ap.add_argument("--no-deploy", action="store_true",
                    help="skip renew.sh; only wait for the viewer and measure")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--secs", type=float, default=23.0)
    ap.add_argument("--port", type=int, default=int(os.environ.get("VIEWER_LOCAL_PORT", 8091)))
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    args = ap.parse_args(argv)
    if args.port == 8081:  # the user's LTX relay
        out = assemble("", [], RC_USAGE, "port 8081 is the LTX relay, never the viewer")
    elif args.n is None and not args.rev:
        out = assemble("", [], RC_USAGE, "give N (best-iter-<N>) or --rev REF")
    else:
        try:
            out = run(args)
        except Exception as e:  # one JSON line whatever happens
            out = assemble("", [], 1, f"{type(e).__name__}: {e}")
    print(json.dumps(out), flush=True)
    return int(out["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
