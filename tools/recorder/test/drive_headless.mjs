// Drive tools/recorder in headless Chrome over a test video via CDP; print a summary.
// Usage: node drive_headless.mjs <page url> <output dir>. Called by run_headless.sh.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
const [,, pageUrl, outDir] = process.argv;
const chrome = spawn('google-chrome', ['--headless=new', '--remote-debugging-port=9333', '--no-first-run',
  '--autoplay-policy=no-user-gesture-required', `--user-data-dir=${outDir}/chrome-profile`, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch('http://127.0.0.1:9333/json')).json()).find((t) => t.type === 'page'); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map(); const logs = [];
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id) pending.get(d.id)?.(d); else if (d.method === 'Runtime.consoleAPICalled') logs.push(d.params.args.map((a) => a.value ?? a.description).join(' ')); else if (d.method === 'Runtime.exceptionThrown') logs.push('EXC ' + JSON.stringify(d.params.exceptionDetails).slice(0, 400)); };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send('Runtime.enable');
await send('Page.navigate', { url: pageUrl });
let state = '';
for (let i = 0; i < 120; i++) { await sleep(500); state = await ev("document.getElementById('state')?.textContent"); if (state === 'Ready.' || state?.startsWith('Error')) break; }
console.log('state after load:', state);
if (state === 'Ready.') {
  await ev("document.getElementById('start').click()");
  for (let i = 0; i < 80; i++) { await sleep(500); if (await ev('!!window.__mtResult')) break; }
  const res = await ev('window.__mtResult');
  if (res) { fs.writeFileSync(`${outDir}/session.json`, res.session); fs.writeFileSync(`${outDir}/landmarks.jsonl`, res.landmarks); }
  console.log('status:', await ev("document.getElementById('status').textContent"));
  console.log('result:', res ? res.dirName : 'none');
}
console.log('console:', logs.slice(-8).join('\n  '));
ws.close(); chrome.kill();
