#!/usr/bin/env bash
# Headless smoke test of the recorder: builds a 12 s test video from MediaPipe's Apache-2.0 test
# image (a person, panned gently), replays it through the recorder in headless Chrome, and writes
# the resulting session to $OUT (default: a temp dir). Needs google-chrome, ffmpeg, node >= 22,
# python3 and network access (MediaPipe loads from CDN). Does not use the webcam.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$(mktemp -d)}"
PORT="${PORT:-8765}"
mkdir -p "$OUT"
curl -sf -o "$OUT/person.jpg" https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg
ffmpeg -v error -y -f lavfi -i color=c=0xd8d4cf:s=640x480:r=30 -loop 1 -i "$OUT/person.jpg" \
  -filter_complex "[1:v]scale=-2:600[p];[0:v][p]overlay=x=(W-w)/2+40*sin(2*PI*t/4):y=-20+15*sin(2*PI*t/3),format=yuv420p" \
  -t 12 -c:v libvpx-vp9 -b:v 2M "$OUT/test_person.webm"
# Serve the recorder plus the test video from a temp root so nothing lands in the repo.
ROOT="$OUT/www"; mkdir -p "$ROOT"
cp "$HERE"/../index.html "$HERE"/../recorder.js "$HERE"/../protocols.js "$OUT/test_person.webm" "$ROOT/"
python3 -m http.server "$PORT" -d "$ROOT" >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 1
node "$HERE/drive_headless.mjs" "http://localhost:$PORT/index.html?video=test_person.webm&test" "$OUT"
echo "session written to $OUT (session.json, landmarks.jsonl)"
