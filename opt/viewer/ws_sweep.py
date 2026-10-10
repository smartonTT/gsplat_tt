"""Websocket sweep client for the live viewer (task #457, viser 1.0.27 protocol).

Connects like the page (subprotocol viser-v<version>), sends the bicycle bench
poses as ViewerCameraMessage at --hz (cycling through the 30 views, like a fast
camera drag), and decodes every BackgroundImageMessage JPEG it receives. Prints
one SWEEP line: decoded frames/s over the steady window (after --warm s), frame
interval percentiles, JPEG size and decode ms. Needs only the viewer venv
(websockets, msgspec, zstandard, simplejpeg); no device access.
  venv/bin/python opt/viewer/ws_sweep.py [--url ws://127.0.0.1:8080] [--hz 240] [--secs 20]
Wire format (viser/infra/_infra.py): client->server a plain msgpack dict with a
"type" key; server->client [u64 raw len][u64 zstd len][zstd msgpack {messages,..}]
[aligned binary buffers]; BackgroundImageMessage.rgb_data is inline bytes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def rot_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) of a 3x3 rotation (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = ((m[k, j] - m[j, k]) / s, v[0], v[1], v[2])
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    return tuple(float(x) for x in (q if q[0] >= 0 else -q))


def camera_messages(width: int, height: int) -> list[dict]:
    cam = json.loads((REPO / "benchmarks" / "cameras_v2.json").read_text())["bicycle"]
    fov = float(np.deg2rad(float(cam["fov_deg"])))
    out = []
    for name in cam["order"]:
        c2w = np.asarray(cam["views"][name]["c2w"], dtype=np.float64)
        pos = c2w[:3, 3]
        fwd = c2w[:3, 2]
        out.append({"type": "ViewerCameraMessage", "wxyz": rot_to_wxyz(c2w[:3, :3]),
                    "position": tuple(float(x) for x in pos), "fov": fov, "near": 0.01,
                    "far": 1000.0, "image_height": int(height), "image_width": int(width),
                    "look_at": tuple(float(x) for x in pos + fwd),
                    "up_direction": tuple(float(x) for x in -c2w[:3, 1])})
    return out


def parse_frame(raw: bytes, zd, huds: list[str] | None = None) -> list[bytes]:
    """JPEG payloads of the BackgroundImageMessages in one server window.
    The viewer's stats HUD text (GuiUpdateMessage _markdown) goes to ``huds``."""
    import msgspec
    n_raw = int.from_bytes(raw[0:8], "little")
    n_z = int.from_bytes(raw[8:16], "little")
    inner = msgspec.msgpack.decode(zd.decompress(raw[16:16 + n_z], max_output_size=n_raw))
    msgs = inner.get("messages", ())
    if huds is not None:
        for m in msgs:
            c = (m.get("updates") or {}).get("_markdown") if m.get("type") == "GuiUpdateMessage" else None
            if isinstance(c, str) and "Device render" in c:
                huds.append(c)
    return [m["rgb_data"] for m in msgs
            if m.get("type") == "BackgroundImageMessage" and m.get("rgb_data")]


async def sweep(args) -> dict:
    import msgspec
    import simplejpeg
    import viser
    import websockets.asyncio.client as wsc
    import zstandard

    msgs = [msgspec.msgpack.encode(m) for m in camera_messages(args.width, args.height)]
    zd = zstandard.ZstdDecompressor()
    t_frames: list[float] = []
    dec_ms: list[float] = []
    sizes: list[int] = []
    huds: list[str] = []
    sent = 0
    async with wsc.connect(args.url, subprotocols=[f"viser-v{viser.__version__}"],
                           max_size=64 * 1024 * 1024, compression=None) as ws:
        t_start = time.perf_counter()
        t_end = t_start + args.secs

        async def sender():
            nonlocal sent
            period = 1.0 / args.hz
            nxt = time.perf_counter()
            while time.perf_counter() < t_end:
                await ws.send(msgs[sent % len(msgs)])
                sent += 1
                nxt += period
                await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

        async def receiver():
            while time.perf_counter() < t_end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.05, t_end - time.perf_counter()))
                except asyncio.TimeoutError:
                    break
                if not isinstance(raw, bytes):
                    continue
                for jpg in parse_frame(raw, zd, huds):
                    t = time.perf_counter()
                    simplejpeg.decode_jpeg(jpg)
                    dec_ms.append((time.perf_counter() - t) * 1000.0)
                    t_frames.append(time.perf_counter())
                    sizes.append(len(jpg))

        await asyncio.gather(sender(), receiver())
    steady = [t for t in t_frames if t >= t_start + args.warm]
    span = (steady[-1] - steady[0]) if len(steady) > 1 else 0.0
    fps = (len(steady) - 1) / span if span > 0 else 0.0
    iv = np.diff(steady) * 1000.0 if len(steady) > 1 else np.zeros(1)
    return dict(url=args.url, hz=args.hz, secs=args.secs, sent=sent, frames=len(t_frames),
                steady_frames=len(steady), decoded_fps=round(fps, 2),
                interval_ms_p50=round(float(np.percentile(iv, 50)), 2),
                interval_ms_p90=round(float(np.percentile(iv, 90)), 2),
                interval_ms_p99=round(float(np.percentile(iv, 99)), 2),
                jpeg_kb_median=round(statistics.median(sizes) / 1024.0, 1) if sizes else 0.0,
                decode_ms_median=round(statistics.median(dec_ms), 2) if dec_ms else 0.0,
                hud=huds[-1].replace("**", "").replace("  \n", " | ") if huds else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080")
    ap.add_argument("--hz", type=float, default=240.0)
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--warm", type=float, default=3.0, help="seconds left out of the FPS window")
    ap.add_argument("--width", type=int, default=0, help="client canvas (default: bench image size)")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if not args.width or not args.height:
        cam = json.loads((REPO / "benchmarks" / "cameras_v2.json").read_text())["bicycle"]
        args.width, args.height = cam["image_size"]
    r = asyncio.run(sweep(args))
    r["label"] = args.label
    print("SWEEP " + json.dumps(r), flush=True)
    return 0 if r["frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
