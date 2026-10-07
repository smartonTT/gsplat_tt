// Page reconnect check for the live viewer (task #357), run on the Mac with node >= 22.
// Drives a headless Chrome over CDP: opens the viewer page, waits for frames, runs
// ARM_CMD (e.g. an ssh that arms the viewer's stall hook), drags the camera so the next
// render stalls, then waits for the page's reconnect shim to log "viewer connected" again
// and saves before/after screenshots.
//   node opt/viewer/page_reconnect_check.mjs CDP_PORT URL OUT_DIR WAIT_S "ARM_CMD"
// Start Chrome first: "Google Chrome" --headless=new --remote-debugging-port=CDP_PORT
//   --user-data-dir=tmp/chrome --window-size=1280,900 about:blank
import { execSync } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const [cdpPort, url, outDir, waitS, armCmd] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ts = () => new Date().toISOString();

const targets = await (await fetch(`http://127.0.0.1:${cdpPort}/json`)).json();
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
const consoleLines = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === "Runtime.consoleAPICalled") {
    const text = m.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
    if (text.includes("[gsplat]")) { consoleLines.push(`${ts()} ${text}`); console.log(`[page] ${text}`); }
  }
});
const send = (method, params = {}) => new Promise((r) => {
  const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params }));
});
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${outDir}/${name}`, Buffer.from(r.result.data, "base64"));
  console.log(`[check ${ts()}] saved ${outDir}/${name}`);
};
const drag = async (dx) => {
  const x = 640, y = 450;
  await send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  for (let k = 1; k <= 10; k++) {
    await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: x + (dx * k) / 10, y, button: "left", buttons: 1 });
    await sleep(30);
  }
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: x + dx, y, button: "left", clickCount: 1 });
};
const connectedCount = () => consoleLines.filter((l) => l.includes("viewer connected")).length;

await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url });
await sleep(20000);
await shot("page_before.png");
const before = connectedCount();
console.log(`[check ${ts()}] connected logs so far: ${before}; arming: ${armCmd}`);
execSync(armCmd, { stdio: "inherit" });
await drag(120);
const t0 = Date.now();
let ok = false;
while (Date.now() - t0 < Number(waitS) * 1000) {
  await sleep(2000);
  if (consoleLines.some((l) => l.includes("disconnected")) && connectedCount() > before) { ok = true; break; }
}
console.log(`[check ${ts()}] reconnect ${ok ? "OK" : "NOT SEEN"} after ${((Date.now() - t0) / 1000).toFixed(0)} s`);
if (ok) { await sleep(10000); await drag(-120); await sleep(8000); }
await shot("page_after.png");
writeFileSync(`${outDir}/page_console.log`, consoleLines.join("\n") + "\n");
ws.close();
process.exit(ok ? 0 : 1);
