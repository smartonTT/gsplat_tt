"""Compatibility shims for viser + nerfview internals.

We import this for its side effects from ``gsplat.viewer``.

**viser ``CameraHandle._update_wxyz`` divides by zero at the pole.**
When the camera's forward direction becomes parallel to ``up_direction``,
viser's Gram-Schmidt step yields a zero-length lateral vector and emits
``RuntimeWarning: invalid value encountered in divide``. The resulting NaN
view matrix paints the browser black and pins drag rotation to a
half-circle. We replace the routine with a gimbal-safe version that picks an
alternate basis through the singularity so the orbit keeps spinning.

nerfview's mid-render interrupt (``Renderer.submit`` + a ``sys.settrace``
hook) is not patched here: ``gsplat.nerfview_viewer.FastRenderer`` renders
without the trace hook, so the interrupt can never fire (task #271 removed a
patch that set ``_may_interrupt_render = False`` before the original submit,
which set it straight back).

**Connection logging and auto-reconnect (task #347).** viser runs with
``verbose=False`` and websockets logs keepalive closes only at debug level,
so a dropped page left nothing in the log. ``_patch_websocket_serve`` wraps
the handler viser passes to ``websockets.asyncio.server.serve`` and logs each
session (UTC, client id, remote, close codes sent/received, length, ping
latency) and any handler exception with its traceback. It also sets a
longer keepalive (``GSPLAT_VIEWER_WS_PING_INTERVAL``/``_TIMEOUT``, default
20 s / 60 s) for the two-hop ssh tunnel, and injects ``RECONNECT_SHIM`` into
the served page: the viser 1.0.27 client retries only while the page has
focus, so a page left in the background never came back by itself.

**Page FPS (task #457).** ``PAGEFPS_SHIM`` (injected with the reconnect shim)
counts the frames the page really shows: viser hands each JPEG frame to three.js
as a blob URL and revokes it once the image has loaded, so the shim times
``URL.createObjectURL`` -> ``URL.revokeObjectURL`` for image/jpeg blobs. Once a
second it shows "page N FPS" in a corner and GETs
``/gsplat/pagestats?fps=..&dec=..&n=..``; the server answers 204 and keeps the
numbers in ``PAGE_STATS`` for the HUD and the heartbeat log.
"""
from __future__ import annotations

import asyncio
import dataclasses
import gzip
import inspect
import os
import threading
import time
import traceback
import urllib.parse

import numpy as np

from gsplat.viewer_session import log

# Live websocket connections (heartbeat reads its length).
LIVE_CONNECTIONS: set = set()

_SHIM_MARKER = b"gsplat-reconnect-shim"

# Runs before viser's bundle. Keeps the client's 1 s retry loop alive when
# the page lacks focus (document.hasFocus), and spaces the retries 1, 2, 4,
# 8, 10, 10... s apart (reset on connect) so a long outage does not hammer
# the tunnel. Close reasons go to the browser console.
RECONNECT_SHIM = b"""<script>/* gsplat-reconnect-shim (task #347) */
(function () {
  try { document.hasFocus = function () { return true; }; } catch (e) {}
  var W = window.Worker;
  if (!W) return;
  function GsplatWorker(url, opts) {
    var w = new W(url, opts), delay = 1000, next = 0, post = w.postMessage.bind(w);
    w.addEventListener("message", function (ev) {
      var d = ev.data || {};
      if (d.type === "connected") {
        delay = 1000; next = 0;
        console.log("[gsplat] viewer connected " + new Date().toISOString());
      } else if (d.type === "closed") {
        console.warn("[gsplat] viewer disconnected " + new Date().toISOString() + ": " + d.closeReason);
      }
    });
    w.postMessage = function (msg) {
      if (msg && msg.type === "retry") {
        var now = Date.now();
        if (now < next) return;
        next = now + delay; delay = Math.min(delay * 2, 10000);
      }
      return post.apply(null, arguments);
    };
    return w;
  }
  GsplatWorker.prototype = W.prototype;
  window.Worker = GsplatWorker;
})();
</script>
"""


# Frames the page received and loaded per second (task #457). Wraps the blob URL
# viser makes for each BackgroundImageMessage; viser revokes it in the texture's
# onLoad, so create -> revoke is receive -> image loaded.
PAGEFPS_SHIM = b"""<script>/* gsplat-pagefps-shim (task #457) */
(function () {
  var U = window.URL;
  if (!U || !U.createObjectURL || !U.revokeObjectURL) return;
  var create = U.createObjectURL.bind(U), revoke = U.revokeObjectURL.bind(U);
  var pending = new Map(), done = [], decSum = 0, decN = 0, total = 0, last = 0, box = null;
  U.createObjectURL = function (obj) {
    var url = create(obj);
    if (obj && obj.type === "image/jpeg") {
      if (pending.size > 256) pending.clear();
      pending.set(url, performance.now());
    }
    return url;
  };
  U.revokeObjectURL = function (url) {
    var t0 = pending.get(url);
    if (t0 !== undefined) {
      pending.delete(url);
      var t = performance.now();
      done.push(t); decSum += t - t0; decN += 1; total += 1;
    }
    return revoke(url);
  };
  setInterval(function () {
    var now = performance.now();
    while (done.length && done[0] < now - 1000) done.shift();
    var fps = done.length, dec = decN ? decSum / decN : 0;
    decSum = 0; decN = 0;
    window.__gsplatPage = {fps: fps, dec: dec, total: total, t: Date.now()};
    if (!box && document.body) {
      box = document.createElement("div");
      box.id = "gsplat-pagefps";
      box.style.cssText = "position:fixed;left:8px;bottom:8px;z-index:99999;padding:2px 6px;" +
        "font:12px monospace;color:#fff;background:rgba(0,0,0,.55);border-radius:3px;" +
        "pointer-events:none";
      document.body.appendChild(box);
    }
    if (box) box.textContent = "page " + fps + " FPS \\u00b7 load " + dec.toFixed(1) + " ms";
    if (fps > 0 || last > 0) {
      fetch("/gsplat/pagestats?fps=" + fps + "&dec=" + dec.toFixed(2) + "&n=" + total,
            {cache: "no-store"}).catch(function () {});
    }
    last = fps;
  }, 1000);
})();
</script>
"""


def inject_reconnect_shim(html: bytes) -> bytes:
    """Insert ``RECONNECT_SHIM`` and ``PAGEFPS_SHIM`` right after ``<head>`` (idempotent)."""
    if _SHIM_MARKER in html:
        return html
    i = html.find(b"<head>")
    if i < 0:
        return html
    i += len(b"<head>")
    return html[:i] + b"\n" + RECONNECT_SHIM + PAGEFPS_SHIM + html[i:]


class PageStats:
    """Latest page-side frame rate reported through /gsplat/pagestats (task #457)."""

    STALE_S = 3.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: dict | None = None

    def update(self, query: str, remote: str = "?", now: float | None = None) -> bool:
        q = urllib.parse.parse_qs(query)
        try:
            fps = float(q["fps"][0])
            dec = float(q.get("dec", ["0"])[0])
            n = int(float(q.get("n", ["0"])[0]))
        except (KeyError, ValueError, IndexError):
            return False
        with self._lock:
            self._last = {"fps": fps, "dec": dec, "n": n, "remote": remote,
                          "t": time.monotonic() if now is None else now}
        return True

    def latest(self, now: float | None = None) -> dict | None:
        """The last report, or None once it is older than STALE_S."""
        now = time.monotonic() if now is None else now
        with self._lock:
            last = self._last
        if last is None or now - last["t"] > self.STALE_S:
            return None
        return dict(last)


PAGE_STATS = PageStats()
_PAGESTATS_PATH = "/gsplat/pagestats"


def _pagestats_response(connection, request):
    """204 for the page shim's report, or None for any other path."""
    path = getattr(request, "path", "") or ""
    if not path.startswith(_PAGESTATS_PATH):
        return None
    _, _, query = path.partition("?")
    PAGE_STATS.update(query, str(getattr(connection, "remote_address", "?")))
    import http
    return connection.respond(http.HTTPStatus.NO_CONTENT, "")


_shim_cache: dict = {}


def _shim_response(response):
    """Rewrite viser's index.html response (plain or gzip) to carry the shim."""
    if response is None or not str(response.headers.get("Content-Type", "")).startswith("text/html"):
        return response
    body = response.body
    out = _shim_cache.get(body)
    if out is None:
        gz = response.headers.get("Content-Encoding") == "gzip"
        html = inject_reconnect_shim(gzip.decompress(body) if gz else body)
        out = gzip.compress(html) if gz else html
        if len(_shim_cache) > 8:
            _shim_cache.clear()
        _shim_cache[body] = out
    headers = response.headers.copy()
    del headers["Content-Length"]  # Headers.__setitem__ appends
    headers["Content-Length"] = str(len(out))
    return dataclasses.replace(response, headers=headers, body=out)


def _close_info(conn) -> str:
    proto = getattr(conn, "protocol", None)
    rcvd = getattr(proto, "close_rcvd", None)
    sent = getattr(proto, "close_sent", None)

    def fmt(frame) -> str:
        return "-" if frame is None else f"{frame.code} {frame.reason!r}"

    first = ""
    if rcvd is not None and sent is not None:
        first = " closed_by=" + ("client" if getattr(proto, "close_rcvd_then_sent", False) else "server")
    elif rcvd is None and sent is None:
        first = " closed_by=none(abnormal, 1006)"
    return f"code={getattr(conn, 'close_code', None)} rcvd={fmt(rcvd)} sent={fmt(sent)}{first}"


def _wrap_handler(handler):
    async def logged_handler(conn):
        t0 = time.monotonic()
        LIVE_CONNECTIONS.add(conn)
        exc = None
        try:
            return await handler(conn)
        except BaseException as e:  # noqa: BLE001 - logged, then re-raised
            exc = e
            if not isinstance(e, asyncio.CancelledError):
                log(f"websocket handler failed: {e!r}\n{traceback.format_exc()}")
            raise
        finally:
            LIVE_CONNECTIONS.discard(conn)
            cid = getattr(conn, "_gsplat_client_id", "?")
            latency = getattr(conn, "latency", 0.0) or 0.0
            log(f"client {cid} disconnected after {time.monotonic() - t0:.1f}s "
                f"{_close_info(conn)} ping_latency={latency * 1000:.0f}ms "
                f"open_connections={len(LIVE_CONNECTIONS)}"
                + (f" exception={type(exc).__name__}" if exc is not None else ""))

    return logged_handler


def _wrap_process_request(process_request):
    def shimmed(connection, request):
        stats = _pagestats_response(connection, request)
        if stats is not None:
            return stats
        r = process_request(connection, request)
        if inspect.isawaitable(r):
            async def later():
                return _shim_response(await r)
            return later()
        return _shim_response(r)

    return shimmed


def _loop_exception_handler(loop, context) -> None:
    # Exceptions of fire-and-forget tasks (viser's per-message handlers)
    # otherwise surface late, if ever, and without a time.
    exc = context.get("exception")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else ""
    log(f"event loop error: {context.get('message')}\n{tb}")


def _patch_websocket_serve() -> None:
    import websockets.asyncio.server as ws_server

    if getattr(ws_server.serve, "_gsplat_wrapped", False):
        return
    orig_serve = ws_server.serve

    def serve(handler, *args, **kwargs):
        kwargs.setdefault("ping_interval", float(os.environ.get("GSPLAT_VIEWER_WS_PING_INTERVAL", "20")))
        kwargs.setdefault("ping_timeout", float(os.environ.get("GSPLAT_VIEWER_WS_PING_TIMEOUT", "60")))
        if kwargs.get("process_request") is not None:
            kwargs["process_request"] = _wrap_process_request(kwargs["process_request"])
        try:
            asyncio.get_running_loop().set_exception_handler(_loop_exception_handler)
        except RuntimeError:
            pass
        return orig_serve(_wrap_handler(handler), *args, **kwargs)

    serve._gsplat_wrapped = True  # type: ignore[attr-defined]
    ws_server.serve = serve  # type: ignore[assignment]


def _patch_message_producer() -> None:
    """Tag each connection with viser's client id and log the connect."""
    import viser.infra._infra as infra

    if getattr(infra._message_producer, "_gsplat_wrapped", False):
        return
    orig = infra._message_producer

    async def _message_producer(websocket, buffer, client_id):
        if getattr(websocket, "_gsplat_client_id", None) is None:
            websocket._gsplat_client_id = client_id
            log(f"client {client_id} connected from {getattr(websocket, 'remote_address', '?')} "
                f"open_connections={len(LIVE_CONNECTIONS)}")
        return await orig(websocket, buffer, client_id)

    _message_producer._gsplat_wrapped = True  # type: ignore[attr-defined]
    infra._message_producer = _message_producer


def _patch_message_buffer_flush() -> None:
    """Send each background image at once instead of on viser's 1/60 s window.

    viser 1.0.27's AsyncMessageBuffer.window_generator sleeps window_duration_sec
    (1/60 s) after every window unless flush() is pulsed, and frames pushed in
    that sleep collapse to the newest by redundancy key: the page got at most
    60 frames/s and saw 17.7/35 ms intervals (43 FPS, task #458). A flush per
    BackgroundImageMessage ends the sleep; other messages keep the window.
    GSPLAT_VIEWER_FRAME_FLUSH=0 restores viser's pacing.
    """
    from viser._messages import BackgroundImageMessage
    from viser.infra._async_message_buffer import AsyncMessageBuffer

    if getattr(AsyncMessageBuffer.push, "_gsplat_wrapped", False):
        return
    if os.environ.get("GSPLAT_VIEWER_FRAME_FLUSH", "1") == "0":
        return
    orig_push = AsyncMessageBuffer.push

    def push(self, message) -> None:
        orig_push(self, message)
        if isinstance(message, BackgroundImageMessage):
            self.flush()

    push._gsplat_wrapped = True  # type: ignore[attr-defined]
    AsyncMessageBuffer.push = push  # type: ignore[assignment]


def _patch_viser_camera_handle_gimbal() -> None:
    import viser._viser as viser_internals  # type: ignore[attr-defined]
    import viser.transforms as vt

    def _safe_update_wxyz(self) -> None:
        z = np.asarray(self._state.look_at, dtype=np.float64) - np.asarray(
            self._state.position, dtype=np.float64
        )
        z_norm = float(np.linalg.norm(z))
        if z_norm < 1e-9:
            return
        z = z / z_norm

        def _gram_schmidt(reference: np.ndarray) -> np.ndarray:
            rotated = vt.SO3.exp(z * np.pi) @ reference
            ortho = rotated - np.dot(z, rotated) * z
            return ortho

        up = np.asarray(self._state.up_direction, dtype=np.float64)
        y = _gram_schmidt(up)
        y_norm = float(np.linalg.norm(y))
        if y_norm < 1e-6:
            # Forward is (nearly) parallel to up_direction. Pick a fallback
            # reference that is guaranteed not parallel to z, run the same
            # Gram-Schmidt projection, and continue. This lets the orbit
            # keep rotating right through the pole instead of NaN-locking.
            alt = (
                np.array([1.0, 0.0, 0.0])
                if abs(float(z[0])) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            y = _gram_schmidt(alt)
            y_norm = float(np.linalg.norm(y))
            if y_norm < 1e-6:
                return  # truly degenerate; leave wxyz untouched
        y = y / y_norm
        x = np.cross(y, z)
        self._state.wxyz = vt.SO3.from_matrix(
            np.stack([x, y, z], axis=1)
        ).wxyz.astype(np.float64)

    viser_internals.CameraHandle._update_wxyz = _safe_update_wxyz  # type: ignore[assignment]


def install_all() -> None:
    _patch_viser_camera_handle_gimbal()
    _patch_websocket_serve()
    _patch_message_producer()
    _patch_message_buffer_flush()


install_all()
