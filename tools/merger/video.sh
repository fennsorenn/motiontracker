# Comparison videos (webcam | hand-written combination | merger, face blendshapes on both) for seated-v1 and hands-v1.
# Needs data/videos/<session>-full/face.landmarks.jsonl (runtime dump with face) and a vspark checkout next to this repo.
# usage: mergervideo.sh <run name>
set -e
export OMP_NUM_THREADS=2
R=$1
cd "$(dirname "$0")/../.."
AV=${MT_AVATAR_VRM:-$PWD/../volleyball/tools/avatars/models/AvatarSample_F.vrm}
one() {  # name session
  D=data/videos/$1-$R; mkdir -p $D
  L=data/videos/$1-full/face.landmarks.jsonl
  [ -f $D/merger.poses.jsonl ] || (cd train && uv run python -m mttrain.merger.run ../data/merger/runs/$R/best.pt ../$L --out-dir ../$D/tmp --avatar AvatarSample_F && mv ../$D/tmp/*.jsonl ../$D/ && rmdir ../$D/tmp)
  [ -f $D/tracks.json ] || uv run --project eval python tools/avatar-video/prepare.py sessions/$2 $D --all-bones --hand-insets \
     --system combination-glide=$D/base.poses.jsonl --system merger-$R-glide=$D/merger.poses.jsonl \
     --blendshapes combination-glide=$L --blendshapes merger-$R-glide=$L
  [ -f $D/compare.mp4 ] || node tools/avatar-video/render.mjs $D $AV sessions/$2 --size 640
  [ -f $D/compare-landmarks.mp4 ] || node tools/avatar-video/overlay.mjs $D sessions/$2 --size 640
  [ -f $D/compare-landmarks-small.mp4 ] || ffmpeg -v error -y -i $D/compare-landmarks.mp4 -vf scale=1440:-2 -c:v libx264 -crf 28 -preset medium -threads 4 -movflags +faststart $D/compare-landmarks-small.mp4
  echo "$1 VIDEO DONE"
}
one seated-v1 2026-11-01T15-29-11-session
one hands-v1 2026-11-12T21-56-29-session
echo ALL VIDEOS DONE
