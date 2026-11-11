// Render a side-by-side avatar video: mirrored webcam | one avatar per system (tracks from prepare.py).
// Usage: node render.mjs <out dir with tracks.json> <avatar.vrm> <session dir> [--size 600] [--from s] [--to s] [--keep-frames]
// Frames at 60 Hz (the tracks' render rate) go to <out>/frames (deleted after a successful encode
// unless --keep-frames), the video to <out>/compare.mp4.
// Needs google-chrome, ffmpeg, python3, node >= 22 and network access (three.js from jsDelivr).
// Run under tools/guarded-run.sh: Chrome's software WebGL uses every core it gets.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const keep = args.includes('--keep-frames') && args.splice(args.indexOf('--keep-frames'), 1);
const size = Number(opt('--size', 600)), from = opt('--from'), to = opt('--to');
const [outDir, avatar, sessionDir] = args.map((a) => path.resolve(a));
const here = path.dirname(fileURLToPath(import.meta.url));
const www = path.join(outDir, 'www'), frames = path.join(outDir, 'frames');
fs.rmSync(www, { recursive: true, force: true });
fs.rmSync(frames, { recursive: true, force: true });
fs.mkdirSync(www, { recursive: true });
fs.mkdirSync(frames, { recursive: true });
fs.copyFileSync(path.join(here, 'render.html'), path.join(www, 'render.html'));
fs.symlinkSync(path.join(outDir, 'tracks.json'), path.join(www, 'tracks.json'));
fs.symlinkSync(avatar, path.join(www, 'avatar.vrm'));

const port = 8790 + Math.floor(Math.random() * 100), dbg = port + 1000;
const server = spawn('python3', ['-m', 'http.server', String(port), '-d', www], { stdio: 'ignore' });
const chrome = spawn('google-chrome', ['--headless=new', `--remote-debugging-port=${dbg}`, '--no-first-run',
  '--enable-unsafe-swiftshader', `--user-data-dir=${outDir}/chrome-profile`, 'about:blank'], { stdio: 'ignore' });
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
let id = 0; const pending = new Map(); const logs = [];
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id) pending.get(d.id)?.(d);
  else if (d.method === 'Runtime.consoleAPICalled') logs.push(d.params.args.map((a) => a.value ?? a.description).join(' '));
  else if (d.method === 'Runtime.exceptionThrown') logs.push('EXC ' + JSON.stringify(d.params.exceptionDetails).slice(0, 600));
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send('Runtime.enable');
await sleep(500);
await send('Page.navigate', { url: `http://localhost:${port}/render.html?size=${size}` });
let ready = false;
for (let i = 0; i < 240 && !ready; i++) { await sleep(500); ready = await ev('!!window.__ready'); }
if (!ready) { console.error('page not ready:', logs.join('\n')); process.exit(1); }
const info = await ev('window.__info');
console.log('info:', JSON.stringify(info));
const t0 = from !== undefined ? Number(from) * 1000 : 0;
const t1 = to !== undefined ? Number(to) * 1000 : info.end;
const n = Math.floor((t1 - t0) / (1000 / 60)) + 1;
const start = Date.now();
for (let k = 0; k < n; k++) {
  const b64 = await ev(`window.renderAt(${t0 + (k * 1000) / 60})`);
  if (!b64) { console.error('render failed at frame', k, logs.slice(-5).join('\n')); process.exit(1); }
  fs.writeFileSync(path.join(frames, `${String(k).padStart(5, '0')}.jpg`), Buffer.from(b64, 'base64'));
  if (k % 300 === 0) console.log(`frame ${k}/${n}, ${((Date.now() - start) / 1000).toFixed(0)} s`);
}
ws.close(); cleanup();

// Webcam panel: the recorded (unmirrored) camera video, mirrored like vspark's processing, resampled
// to 60 fps and cut to the same span; landmark times count from the recorder start (~1 frame offset).
const font = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';
const r = spawnSync('ffmpeg', ['-v', 'error', '-y',
  '-ss', String(t0 / 1000), '-i', path.join(sessionDir, 'video.webm'),
  '-framerate', '60', '-i', path.join(frames, '%05d.jpg'),
  '-filter_complex',
  `[0:v]fps=60,hflip,scale=-2:${size},drawbox=x=0:y=0:w=iw:h=${Math.round(size / 15)}:color=black@0.55:t=fill,` +
  `drawtext=fontfile=${font}:text='webcam (mirrored)':fontcolor=white:fontsize=${Math.round(size / 26)}:x=${Math.round(size / 40)}:y=${Math.round(size / 60)}[cam];` +
  `[cam][1:v]hstack=inputs=2,format=yuv420p[v]`,
  '-map', '[v]', '-frames:v', String(n), '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
  path.join(outDir, 'compare.mp4')], { stdio: 'inherit' });
console.log(r.status === 0 ? `wrote ${path.join(outDir, 'compare.mp4')} (${n} frames)` : 'ffmpeg failed');
fs.rmSync(path.join(outDir, 'chrome-profile'), { recursive: true, force: true });
fs.rmSync(www, { recursive: true, force: true });
if (r.status === 0 && !keep) fs.rmSync(frames, { recursive: true, force: true });
process.exit(r.status === 0 ? 0 : 1);
