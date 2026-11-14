// Render a side-by-side avatar video: mirrored webcam | one avatar per system (tracks from prepare.py).
// Usage: node render.mjs <out dir with tracks.json> <avatar.vrm> <session dir> [--size 600] [--from s] [--to s] [--fps 30] [--keep-frames]
//        [--arkit-map <vspark checkout>/packages/shared/src/arkit_tables.ts]
// Panels with blendshapes (prepare.py --blendshapes) drive the avatar's face like vspark's default
// face mapper: ARKit -> VRoid Fcl_* morph targets with vspark's ARKIT_TO_FCL table, read from a local
// vspark checkout at render time (default: a `vspark` folder next to this repo; not copied here).
// Frames at --fps (default 30; the tracks are sampled at 60 Hz) go to <out>/frames (deleted after a successful encode
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
const fps = Number(opt('--fps', 30));  // 30 is plenty to judge motion; 60 doubles the render time
// GPU rendering (owner, 2026-11-13: CPU SwiftShader took ~25 min per minute of video); MT_CHROME_GL=swiftshader
// falls back to the software renderer
const glFlags = process.env.MT_CHROME_GL === 'swiftshader' ? ['--enable-unsafe-swiftshader']
  : (process.env.MT_CHROME_GL_FLAGS ?? '--use-angle=vulkan --enable-features=Vulkan --ignore-gpu-blocklist --enable-gpu').split(' ');
const arkitMap = opt('--arkit-map');
const [outDir, avatar, sessionDir] = args.map((a) => path.resolve(a));
const here = path.dirname(fileURLToPath(import.meta.url));
const www = path.join(outDir, 'www'), frames = path.join(outDir, 'frames');
fs.rmSync(www, { recursive: true, force: true });
// resumable: frames of an interrupted run are kept when its inputs are the same (frames/key.json)
const tracksStat = fs.statSync(path.join(outDir, 'tracks.json'));
const key = JSON.stringify({ tracks: [tracksStat.size, tracksStat.mtimeMs], avatar, sessionDir, size, from, to, fps });
const keyFile = path.join(frames, 'key.json');
if (!fs.existsSync(keyFile) || fs.readFileSync(keyFile, 'utf8') !== key) fs.rmSync(frames, { recursive: true, force: true });
fs.mkdirSync(www, { recursive: true });
fs.mkdirSync(frames, { recursive: true });
fs.writeFileSync(keyFile, key);
fs.copyFileSync(path.join(here, 'render.html'), path.join(www, 'render.html'));
fs.symlinkSync(path.join(outDir, 'tracks.json'), path.join(www, 'tracks.json'));
fs.symlinkSync(avatar, path.join(www, 'avatar.vrm'));
if (JSON.parse(fs.readFileSync(path.join(outDir, 'tracks.json'), 'utf8')).panels.some((p) => p.arkit)) {
  // ARKIT_TO_FCL entries are `  name: [['Fcl_X', 1.0], ...],` lines in vspark's arkit_tables.ts.
  const src = fs.readFileSync(arkitMap ?? path.join(here, '../../../vspark/packages/shared/src/arkit_tables.ts'), 'utf8');
  const block = src.slice(src.indexOf('ARKIT_TO_FCL'), src.indexOf('};', src.indexOf('ARKIT_TO_FCL')));
  const table = {};
  for (const [, name, targets] of block.matchAll(/^\s*(\w+):\s*\[(.*)\],?\s*$/gm))
    table[name] = [...targets.matchAll(/\['(\w+)',\s*(-?[\d.]+)\]/g)].map((m) => [m[1], Number(m[2])]);
  if (!Object.keys(table).length) { console.error('no ARKIT_TO_FCL table found'); process.exit(1); }
  console.log(`ARKit -> Fcl table: ${Object.keys(table).length} shapes`);
  fs.writeFileSync(path.join(www, 'arkit_map.json'), JSON.stringify(table));
}

const port = 8790 + Math.floor(Math.random() * 100), dbg = port + 1000;
const server = spawn('python3', ['-m', 'http.server', String(port), '-d', www], { stdio: 'ignore' });
const chrome = spawn('google-chrome', ['--headless=new', `--remote-debugging-port=${dbg}`, '--no-first-run',
  ...glFlags, `--user-data-dir=${outDir}/chrome-profile-${process.pid}`, 'about:blank'], { stdio: 'ignore' });
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
console.log('GL renderer:', await ev(`(() => { const g = document.createElement('canvas').getContext('webgl2');
  const e = g && g.getExtension('WEBGL_debug_renderer_info'); return g ? g.getParameter(e ? e.UNMASKED_RENDERER_WEBGL : g.RENDERER) : 'none'; })()`));
const t0 = from !== undefined ? Number(from) * 1000 : 0;
const t1 = to !== undefined ? Number(to) * 1000 : info.end;
const n = Math.floor((t1 - t0) / (1000 / fps)) + 1;
const start = Date.now();
for (let k = 0; k < n; k++) {
  const file = path.join(frames, `${String(k).padStart(5, '0')}.jpg`);
  if (fs.existsSync(file)) continue;  // rendered before an interruption
  const b64 = await ev(`window.renderAt(${t0 + (k * 1000) / fps})`);
  if (!b64) { console.error('render failed at frame', k, logs.slice(-5).join('\n')); process.exit(1); }
  fs.writeFileSync(file + '.tmp', Buffer.from(b64, 'base64'));
  fs.renameSync(file + '.tmp', file);
  if (k % 300 === 0) console.log(`frame ${k}/${n}, ${((Date.now() - start) / 1000).toFixed(0)} s`);
}
ws.close(); cleanup();

// Webcam panel: the recorded (unmirrored) camera video, mirrored like vspark's processing, resampled
// to 60 fps and cut to the same span; landmark times count from the recorder start (~1 frame offset).
const font = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';
const r = spawnSync('ffmpeg', ['-v', 'error', '-y',
  '-ss', String(t0 / 1000), '-i', path.join(sessionDir, 'video.webm'),
  '-framerate', String(fps), '-i', path.join(frames, '%05d.jpg'),
  '-filter_complex',
  `[0:v]fps=${fps},hflip,scale=-2:${size},drawbox=x=0:y=0:w=iw:h=${Math.round(size / 15)}:color=black@0.55:t=fill,` +
  `drawtext=fontfile=${font}:text='webcam (mirrored)':fontcolor=white:fontsize=${Math.round(size / 26)}:x=${Math.round(size / 40)}:y=${Math.round(size / 60)}[cam];` +
  `[cam][1:v]hstack=inputs=2,format=yuv420p[v]`,
  '-map', '[v]', '-frames:v', String(n), '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
  path.join(outDir, 'compare.mp4')], { stdio: 'inherit' });
console.log(r.status === 0 ? `wrote ${path.join(outDir, 'compare.mp4')} (${n} frames)` : 'ffmpeg failed');
fs.rmSync(path.join(outDir, `chrome-profile-${process.pid}`), { recursive: true, force: true });
fs.rmSync(www, { recursive: true, force: true });
if (r.status === 0 && !keep) fs.rmSync(frames, { recursive: true, force: true });
process.exit(r.status === 0 ? 0 : 1);
