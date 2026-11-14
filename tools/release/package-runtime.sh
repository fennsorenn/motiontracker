#!/usr/bin/env bash
# Package the live runtime for download (vspark's Learned Tracking "Download runtime"): the runtime's source at a
# tag (core/ and service/, from git, so nothing untracked) plus the model bundles (package-models.sh) in one zip.
# Layout: motiontracker-<version>/{core,service,data/...,MODELS-NOTICE.md}; run with
#   uv run --project motiontracker-<version>/service mtservice ...
#
#   tools/release/package-runtime.sh v0.1.0 [OUT_DIR]   -> OUT_DIR/motiontracker-runtime-v0.1.0.zip (prints its SHA-256)
set -euo pipefail
VERSION=${1:?usage: package-runtime.sh VERSION [OUT_DIR]}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR=$(cd "${2:-.}" && pwd)
OUT=$OUT_DIR/motiontracker-runtime-$VERSION.zip
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
PREFIX=motiontracker-$VERSION
git -C "$ROOT" archive --format=tar --prefix="$PREFIX/" "$VERSION" core service | tar -x -C "$STAGE"
MODELS=$("$ROOT/tools/release/package-models.sh" "$VERSION" "$STAGE")
unzip -q "$MODELS" -d "$STAGE/$PREFIX" && rm "$MODELS"
rm -f "$OUT"
(cd "$STAGE" && zip -qrX "$OUT" "$PREFIX")
echo "$OUT"
sha256sum "$OUT" | cut -d' ' -f1
