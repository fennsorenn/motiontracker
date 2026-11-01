// Session recorder: webcam video + MediaPipe Holistic landmarks, reproducing vspark's
// CameraCapture path (packages/frontend/src/media/CameraCapture.ts + mediapipeWorker.ts):
// mirrored frame, 320x240, inference throttled to 10 FPS, HolisticLandmarker in VIDEO mode with
// the same confidence options, and the same per-frame landmark selection (_dispatch).
//
// Source: the webcam by default; `?video=<url>` replays a video file through the same pipeline
// (landmarks only, timestamps = video time), e.g. to re-run detection on a recorded session.

import {
  FilesetResolver,
  HolisticLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/vision_bundle.mjs';
import { PROTOCOLS } from './protocols.js';

const TASKS_VISION = '0.10.17'; // same version as vspark
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION}/wasm`;
// vspark loads .../float16/latest/...; version 1 is byte-identical (same MD5) and pinned.
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task';
const MODEL_MD5_BASE64 = '3ZywDp2ZM9rHmI2M40e6ew==';
const PROC_W = 320;
const PROC_H = 240;
const TARGET_FPS = 10;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

const $ = (id) => document.getElementById(id);
const videoEl = $('video');
const params = new URLSearchParams(location.search);
const replayUrl = params.get('video');

let landmarker = null;
let stream = null;
let recorder = null;
let chunks = [];
let frames = [];
let segments = [];
let running = false;
let t0 = 0; // performance.now() at recording start (live mode)
let lastFrameAt = 0;
let protocolTimer = null;
let freeStillStart = null;
let stats = { frames: 0, pose: 0, face: 0, leftHand: 0, rightHand: 0, inferMs: 0 };

const canvas = new OffscreenCanvas(PROC_W, PROC_H);
const ctx = canvas.getContext('2d');

// ── setup ────────────────────────────────────────────────────────────────────────

async function init() {
  for (const [key, p] of Object.entries(PROTOCOLS)) {
    $('protocol').add(new Option(`${key}: ${p.description}`, key));
  }
  if (replayUrl) {
    $('camera').add(new Option(`video file: ${replayUrl}`, ''));
    $('camera').disabled = $('size').disabled = true;
    $('protocol').value = 'free';
    $('name').value = replayUrl.split('/').pop().replace(/\.[^.]+$/, '') + '-replay';
  } else {
    await listCameras();
  }
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  landmarker = await HolisticLandmarker.createFromModelPath(vision, MODEL_URL);
  await landmarker.setOptions({
    runningMode: 'VIDEO',
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
  });
  $('state').textContent = 'Ready.';
  $('start').disabled = false;
}

async function listCameras() {
  // Labels are only visible after a permission grant.
  const probe = await navigator.mediaDevices.getUserMedia({ video: true }).catch(() => null);
  probe?.getTracks().forEach((t) => t.stop());
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  for (const d of devices) $('camera').add(new Option(d.label || d.deviceId, d.deviceId));
}

// ── landmarks (mirrors vspark CameraCapture._dispatch) ───────────────────────────

const r5 = (v) => Math.round(v * 1e5) / 1e5;
const pack = (pts, withVis) =>
  pts.map((p) => (withVis ? [r5(p.x), r5(p.y), r5(p.z), r5(p.visibility ?? 0)] : [r5(p.x), r5(p.y), r5(p.z)]));

function toFrame(r, t) {
  const f = { t: Math.round(t * 10) / 10 };
  if (r.faceLandmarks?.[0]?.length) f.face = pack(r.faceLandmarks[0], false);
  if (r.poseWorldLandmarks?.[0]?.length) {
    // vspark: world coordinates, visibility from the image-space set.
    const img = r.poseLandmarks?.[0];
    f.pose = r.poseWorldLandmarks[0].map((p, i) => [
      r5(p.x), r5(p.y), r5(p.z), r5(img?.[i]?.visibility ?? p.visibility ?? 0),
    ]);
  } else if (r.poseLandmarks?.[0]?.length) {
    f.pose = pack(r.poseLandmarks[0], true);
  }
  // vspark: hand world landmarks, falling back to image space; no visibility.
  const left = r.leftHandWorldLandmarks?.[0]?.length ? r.leftHandWorldLandmarks[0] : r.leftHandLandmarks?.[0];
  const right = r.rightHandWorldLandmarks?.[0]?.length ? r.rightHandWorldLandmarks[0] : r.rightHandLandmarks?.[0];
  if (left?.length) f.leftHand = pack(left, false);
  if (right?.length) f.rightHand = pack(right, false);
  // Extra (not sent by vspark; ignored by the baseline): normalized image-space 2D landmarks of
  // the mirrored frame, for reprojection metrics.
  if (r.poseLandmarks?.[0]?.length) f.pose2d = pack(r.poseLandmarks[0], true);
  if (r.leftHandLandmarks?.[0]?.length) f.leftHand2d = pack(r.leftHandLandmarks[0], false);
  if (r.rightHandLandmarks?.[0]?.length) f.rightHand2d = pack(r.rightHandLandmarks[0], false);
  return f;
}

// ── capture loop (mirrors vspark's _scheduleNext / _tick / _drawMirror) ──────────

function scheduleNext() {
  if (!running) return;
  const wait = Math.max(0, FRAME_INTERVAL_MS - (performance.now() - lastFrameAt));
  setTimeout(tick, wait);
}

function tick() {
  if (!running) return;
  if (videoEl.readyState < 2) return scheduleNext();
  lastFrameAt = performance.now();
  ctx.setTransform(-1, 0, 0, 1, PROC_W, 0);
  ctx.drawImage(videoEl, 0, 0, PROC_W, PROC_H);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const t = replayUrl ? videoEl.currentTime * 1000 : lastFrameAt - t0;
  const before = performance.now();
  // VIDEO mode needs strictly increasing timestamps; wall clock in both modes.
  const r = landmarker.detectForVideo(canvas, lastFrameAt);
  stats.inferMs = 0.9 * stats.inferMs + 0.1 * (performance.now() - before);
  const f = toFrame(r, t);
  frames.push(f);
  stats.frames++;
  for (const k of ['pose', 'face', 'leftHand', 'rightHand']) if (f[k]) stats[k]++;
  showStatus(f);
  scheduleNext();
}

function showStatus(f) {
  const n = Math.max(1, stats.frames);
  const pct = (k) => `${((100 * stats[k]) / n).toFixed(0)}%`.padStart(4);
  const secs = frames.length ? (frames[frames.length - 1].t - frames[0].t) / 1000 : 0;
  $('status').textContent =
    `frames ${stats.frames}  (${(stats.frames / Math.max(secs, 1e-3)).toFixed(1)} fps)   ` +
    `inference ${stats.inferMs.toFixed(0)} ms\n` +
    `present: pose${pct('pose')}  face${pct('face')}  leftHand${pct('leftHand')}  rightHand${pct('rightHand')}\n` +
    `now:     ${['pose', 'face', 'leftHand', 'rightHand'].map((k) => (f[k] ? k : '-'.repeat(k.length))).join('  ')}`;
}

// ── protocol ─────────────────────────────────────────────────────────────────────

function now() {
  return replayUrl ? videoEl.currentTime * 1000 : performance.now() - t0;
}

function runProtocol(protocol) {
  const steps = [];
  let at = protocol.leadIn * 1000;
  for (const s of protocol.steps) {
    steps.push({ ...s, t0: at, t1: at + s.duration * 1000 });
    at += s.duration * 1000;
  }
  const end = at;
  const update = () => {
    const t = now();
    const cur = steps.find((s) => t >= s.t0 && t < s.t1);
    if (t < protocol.leadIn * 1000) {
      $('prompt').textContent = `Get ready… ${Math.ceil((protocol.leadIn * 1000 - t) / 1000)}`;
    } else if (cur) {
      $('prompt').textContent = `${cur.prompt}  (${Math.ceil((cur.t1 - t) / 1000)} s)`;
    } else if (steps.length && t >= end) {
      $('prompt').textContent = 'Done. Stopping…';
      clearInterval(protocolTimer);
      segments = steps.map((s) => ({ label: s.label, t0: s.t0, t1: s.t1, prompt: s.prompt }));
      stop();
    }
  };
  protocolTimer = setInterval(update, 100);
  update();
  return steps;
}

document.addEventListener('keydown', (e) => {
  if (!running || $('protocol').value !== 'free' || e.key.toLowerCase() !== 's') return;
  if (freeStillStart === null) {
    freeStillStart = now();
    $('prompt').textContent = 'Still segment… (S to end)';
  } else {
    segments.push({ label: 'still', t0: freeStillStart, t1: now() });
    freeStillStart = null;
    $('prompt').textContent = `${segments.length} still segment(s) marked.`;
  }
});

// ── start / stop / save ──────────────────────────────────────────────────────────

async function start() {
  $('start').disabled = true;
  frames = [];
  segments = [];
  chunks = [];
  freeStillStart = null;
  stats = { frames: 0, pose: 0, face: 0, leftHand: 0, rightHand: 0, inferMs: 0 };
  const protocol = PROTOCOLS[$('protocol').value];

  if (replayUrl) {
    videoEl.src = replayUrl;
    videoEl.style.transform = 'none'; // recorded videos are unmirrored; show them as recorded
    await videoEl.play();
    videoEl.onended = () => stop();
  } else {
    const [w, h] = $('size').value.split('x').map(Number);
    const deviceId = $('camera').value;
    stream = await navigator.mediaDevices.getUserMedia({
      video: { ...(deviceId ? { deviceId: { exact: deviceId } } : {}), width: { ideal: w }, height: { ideal: h } },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) =>
      MediaRecorder.isTypeSupported(m)
    );
    recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    await new Promise((resolve) => {
      recorder.onstart = resolve;
      recorder.start(1000);
    });
  }
  t0 = performance.now();
  lastFrameAt = 0;
  running = true;
  $('stop').disabled = false;
  $('state').innerHTML = '<span class="rec">● recording</span>';
  if (protocol.steps.length) runProtocol(protocol);
  else $('prompt').textContent = replayUrl ? 'Replaying video…' : 'Recording (free). Press S to mark still segments.';
  tick();
}

async function stop() {
  if (!running) return;
  running = false;
  clearInterval(protocolTimer);
  $('stop').disabled = true;
  if (freeStillStart !== null) segments.push({ label: 'still', t0: freeStillStart, t1: now() });
  let videoBlob = null;
  if (recorder) {
    await new Promise((resolve) => {
      recorder.onstop = resolve;
      recorder.stop();
    });
    videoBlob = new Blob(chunks, { type: recorder.mimeType });
  }
  const track = stream?.getVideoTracks()[0];
  const settings = track?.getSettings() ?? {};
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  recorder = null;
  videoEl.pause();

  const name = $('name').value.trim() || 'session';
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const session = {
    format: 'motiontracker-session',
    version: 1,
    name,
    created: new Date().toISOString(),
    protocol: $('protocol').value,
    segments,
    source: replayUrl
      ? { kind: 'video-replay', url: replayUrl }
      : { kind: 'webcam', label: track?.label ?? null, width: settings.width, height: settings.height, frameRate: settings.frameRate },
    processing: {
      mirrored: true,
      width: PROC_W,
      height: PROC_H,
      targetFps: TARGET_FPS,
      detector: { name: 'mediapipe-holistic-landmarker', tasksVision: TASKS_VISION, model: MODEL_URL, modelMd5: MODEL_MD5_BASE64, runningMode: 'VIDEO' },
      landmarkFormat: 'vspark CameraCapture: pose=world+image visibility, hands=world (image fallback), face=image; *2d = normalized image space of the mirrored frame',
    },
    landmarks: { file: 'landmarks.jsonl', frames: frames.length },
    video: videoBlob
      ? {
          file: 'video.webm',
          mimeType: videoBlob.type,
          mirrored: false,
          alignment: 'landmark t (ms) is measured from the MediaRecorder start event; expect about one frame of misalignment',
        }
      : null,
    userAgent: navigator.userAgent,
  };
  const files = { 'session.json': JSON.stringify(session, null, 1), 'landmarks.jsonl': frames.map((f) => JSON.stringify(f)).join('\n') + '\n' };
  if (videoBlob) files['video.webm'] = videoBlob;
  await save(`${stamp}-${name}`, files);
  $('start').disabled = false;
}

let sessionsDir = null;
async function save(dirName, files) {
  if (params.has('test')) {
    // Automated tests (`?test`): keep the result in the page instead of writing files.
    window.__mtResult = { dirName, session: files['session.json'], landmarks: files['landmarks.jsonl'] };
    $('state').textContent = 'Test mode: result kept in window.__mtResult.';
    return;
  }
  if (window.showDirectoryPicker) {
    try {
      sessionsDir ??= await window.showDirectoryPicker({ id: 'mt-sessions', mode: 'readwrite' });
      const dir = await sessionsDir.getDirectoryHandle(dirName, { create: true });
      for (const [name, data] of Object.entries(files)) {
        const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
        await w.write(data);
        await w.close();
      }
      $('state').textContent = `Saved to ${sessionsDir.name}/${dirName}/`;
      return;
    } catch (e) {
      console.warn('directory save failed, falling back to downloads', e);
    }
  }
  for (const [name, data] of Object.entries(files)) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(data instanceof Blob ? data : new Blob([data]));
    a.download = `${dirName}__${name}`;
    a.click();
  }
  $('state').textContent = `Downloaded ${dirName}__* — move them into sessions/${dirName}/ without the prefix.`;
}

$('start').onclick = () => start().catch((e) => { $('state').textContent = `Error: ${e.message}`; $('start').disabled = false; });
$('stop').onclick = () => stop();
init().catch((e) => ($('state').textContent = `Error: ${e.message}`));
