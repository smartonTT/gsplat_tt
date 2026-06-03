#!/usr/bin/env python3
"""Start (or restart) the gsplat viewer detached from the launching shell.

Agent/CI shells often exit with SIGTERM and take down background children.
This launcher uses ``start_new_session=True`` so the viewer survives after
the parent shell is torn down.

Usage:
  python3 scripts/run_viewer.py              # default port 8083
  python3 scripts/run_viewer.py --port 8083
  python3 scripts/run_viewer.py --stop --port 8083
  python3 scripts/run_viewer.py --status --port 8083
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT / "opt"
DEFAULT_PORT = 8083


def pid_path(port: int) -> Path:
    return OPT / f"viewer-{port}.pid"


def log_path(port: int) -> Path:
    return OPT / f"viewer-{port}.log"


def read_pid(port: int) -> int | None:
    p = pid_path(port)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def port_listener(port: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    return int(out.splitlines()[0])


def stop_viewer(port: int) -> None:
    for pid in filter(None, {read_pid(port), port_listener(port)}):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    pid_path(port).unlink(missing_ok=True)
    time.sleep(0.5)


def start_viewer(port: int, host: str, backend: str) -> int:
    OPT.mkdir(parents=True, exist_ok=True)
    stop_viewer(port)

    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"missing venv python: {python}")

    log = log_path(port)
    with log.open("a", encoding="utf-8") as logf:
        logf.write(
            f"\n--- viewer start {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"port={port} backend={backend} ---\n"
        )
        logf.flush()
        proc = subprocess.Popen(
            [
                str(python),
                "-m",
                "gsplat",
                "scenes/stitch_doll.ply",
                "--backend",
                backend,
                "--port",
                str(port),
                "--host",
                host,
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path(port).write_text(str(proc.pid))
    return proc.pid


def status(port: int) -> int:
    pid = read_pid(port)
    listen = port_listener(port)
    alive = pid is not None and is_alive(pid)
    print(f"port={port}  pidfile={pid}  alive={alive}  listener={listen}")
    if listen:
        print(f"url=http://127.0.0.1:{port}")
    return 0 if listen else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Detached gsplat viewer launcher")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", default="cpu_cpp")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.stop:
        stop_viewer(args.port)
        print(f"stopped viewer on port {args.port}")
        return

    if args.status:
        raise SystemExit(status(args.port))

    pid = start_viewer(args.port, args.host, args.backend)
    # Wait until the port is bound or the child exits.
    for _ in range(50):
        if port_listener(args.port) == pid:
            print(f"Viewer running at http://127.0.0.1:{args.port}  pid={pid}")
            print(f"log: {log_path(args.port)}")
            return
        if not is_alive(pid):
            tail = log_path(args.port).read_text().splitlines()[-20:]
            raise SystemExit(
                "viewer exited during startup; log tail:\n" + "\n".join(tail)
            )
        time.sleep(0.1)

    raise SystemExit(f"viewer pid {pid} started but port {args.port} not listening yet")


if __name__ == "__main__":
    main()
