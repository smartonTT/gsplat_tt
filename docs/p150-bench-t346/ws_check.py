"""Open a viser websocket through the Mac tunnel (localhost:8091) and count what the viewer sends."""
import asyncio, sys, time
import websockets

async def main(url, secs):
    async with websockets.connect(url, subprotocols=["viser-v1.0.27"], max_size=None) as ws:
        print("connected status=101 subprotocol=", ws.subprotocol)
        n = b = 0; t0 = time.time()
        while time.time() - t0 < secs:
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=secs - (time.time() - t0))
            except asyncio.TimeoutError:
                break
            n += 1; b += len(m)
        print(f"messages={n} bytes={b} in {secs}s")

asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8091/", float(sys.argv[2]) if len(sys.argv) > 2 else 8))
