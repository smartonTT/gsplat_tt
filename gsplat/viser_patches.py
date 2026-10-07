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
"""
from __future__ import annotations

import asyncio
import dataclasses
import gzip
import inspect
import os
import time
import traceback

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


def inject_reconnect_shim(html: bytes) -> bytes:
    """Insert ``RECONNECT_SHIM`` right after ``<head>`` (idempotent)."""
    if _SHIM_MARKER in html:
        return html
    i = html.find(b"<head>")
    if i < 0:
        return html
    i += len(b"<head>")
    return html[:i] + b"\n" + RECONNECT_SHIM + html[i:]


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


install_all()
