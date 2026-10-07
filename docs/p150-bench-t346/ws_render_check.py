"""Run on the viewer box with the viewer's venv: connect like the browser, send one camera pose,
and count the bytes the viewer streams back (a rendered frame is tens of kB of JPEG)."""
import asyncio, sys, time
import msgspec, websockets

CAM = {"type": "ViewerCameraMessage", "wxyz": [1.0, 0.0, 0.0, 0.0], "position": [0.0, 0.0, -4.0],
       "fov": 0.9, "near": 0.01, "far": 100.0, "image_height": 720, "image_width": 1280,
       "look_at": [0.0, 0.0, 0.0], "up_direction": [0.0, -1.0, 0.0]}

async def main(url, secs):
    async with websockets.connect(url, subprotocols=["viser-v1.0.27"], max_size=None) as ws:
        print("connected subprotocol", ws.subprotocol)
        await ws.send(msgspec.msgpack.encode(CAM))
        n = b = big = 0; t0 = time.time()
        while time.time() - t0 < secs:
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=secs - (time.time() - t0))
            except asyncio.TimeoutError:
                break
            n += 1; b += len(m); big += len(m) > 10000
        print(f"messages={n} bytes={b} large(>10kB)={big} in {secs}s")

asyncio.run(main(sys.argv[1], float(sys.argv[2])))
