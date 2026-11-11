// Overlay the session's MediaPipe landmarks (semi-transparent) onto the webcam panel of a rendered
// comparison video: <dir>/compare.mp4 -> <dir>/compare-landmarks.mp4. No avatar re-render.
// Usage: node overlay.mjs <video dir with compare.mp4> <session dir> [--size 600] [--from s]
//   --size and --from must match the render.mjs call that made compare.mp4.
// Pose: white skeleton, orange = the person's left, blue = right, dot opacity = visibility. Face mesh:
// faint green dots. Hands: yellow. Landmarks fade when the latest detector frame is over 300 ms old.
// Needs google-chrome, ffmpeg, ffprobe, python3, node >= 22. Run under tools/guarded-run.sh.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const size = Number(opt('--size', 600)), from = Number(opt('--from', 0));
const [dir, sessionDir] = args.map((a) => path.resolve(a));
const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.join(dir, 'compare.mp4');
const probe = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
  'stream=nb_read_packets', '-of', 'csv=p=0', src], { encoding: 'utf8' });
const n = Number(probe.stdout.trim());
if (!n) { console.error('cannot read', src); process.exit(1); }
// webcam panel: the session video scaled to the panel height (render.mjs: scale=-2:size)
const meta = JSON.parse(fs.readFileSync(path.join(sessionDir, 'session.json'), 'utf8'));
const camW = meta.processing?.width || 4, camH = meta.processing?.height || 3;
const W = 2 * Math.round((size * camW / camH) / 2), H = size;

const www = path.join(dir, 'www-overlay'), frames = path.join(dir, 'overlay-frames');
for (const d of [www, frames]) { fs.rmSync(d, { recursive: true, force: true }); fs.mkdirSync(d, { recursive: true }); }
fs.copyFileSync(path.join(here, 'overlay.html'), path.join(www, 'overlay.html'));
fs.symlinkSync(path.join(sessionDir, 'landmarks.jsonl'), path.join(www, 'landmarks.jsonl'));

const port = 8890 + Math.floor(Math.random() * 100), dbg = port + 1000;
const server = spawn('python3', ['-m', 'http.server', String(port), '-d', www], { stdio: 'ignore' });
const chrome = spawn('google-chrome', ['--headless=new', `--remote-debugging-port=${dbg}`, '--no-first-run',
  `--user-data-dir=${dir}/chrome-profile-overlay`, 'about:blank'], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill(); server.kill(); } catch {} };
process.on('exit', cleanup);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${dbg}/json`)).json()).find((t) => t.type === 'page'); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id) pending.get(d.id)?.(d); };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send('Runtime.enable');
await send('Page.navigate', { url: `http://localhost:${port}/overlay.html?w=${W}&h=${H}` });
let ready = false;
for (let i = 0; i < 120 && !ready; i++) { await sleep(250); ready = await ev('!!window.__ready'); }
if (!ready) { console.error('overlay page not ready'); process.exit(1); }
console.log('landmarks:', JSON.stringify(await ev('window.__info')), `panel ${W}x${H}, ${n} frames`);
const t0 = from * 1000, start = Date.now();
for (let k = 0; k < n; k++) {
  const b64 = await ev(`window.drawAt(${t0 + (k * 1000) / 60})`);
  fs.writeFileSync(path.join(frames, `${String(k).padStart(5, '0')}.png`), Buffer.from(b64, 'base64'));
  if (k % 600 === 0) console.log(`frame ${k}/${n}, ${((Date.now() - start) / 1000).toFixed(0)} s`);
}
ws.close(); cleanup();

const r = spawnSync('ffmpeg', ['-v', 'error', '-y', '-i', src, '-framerate', '60', '-i', path.join(frames, '%05d.png'),
  '-filter_complex', '[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]', '-map', '[v]', '-frames:v', String(n),
  '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-threads', '4', path.join(dir, 'compare-landmarks.mp4')],
  { stdio: 'inherit' });
console.log(r.status === 0 ? `wrote ${path.join(dir, 'compare-landmarks.mp4')}` : 'ffmpeg failed');
for (const d of [www, frames, path.join(dir, 'chrome-profile-overlay')]) fs.rmSync(d, { recursive: true, force: true });
process.exit(r.status === 0 ? 0 : 1);
