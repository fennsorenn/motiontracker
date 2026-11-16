#!/usr/bin/env bash
# Package the live runtime's model bundles as a release asset (owner decision 2026-11-14: weights may be published as
# GitHub release assets, PLAN.md §9 #9). The zip mirrors the repo's data/ layout, so unpacking it into an extracted
# source archive (motiontracker-<version>/) puts the bundles where mtservice looks for them.
#
#   tools/release/package-models.sh v0.1.0 [OUT_DIR]   -> OUT_DIR/motiontracker-models-v0.1.0.zip
set -euo pipefail
VERSION=${1:?usage: package-models.sh VERSION [OUT_DIR]}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(cd "${2:-.}" && pwd)/motiontracker-models-$VERSION.zip
BUNDLES=(data/merger/runs/mg3/live data/m1/runs/m1-v10/live)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
for b in "${BUNDLES[@]}"; do
  [ -d "$ROOT/$b" ] || { echo "missing bundle $b" >&2; exit 1; }
  mkdir -p "$STAGE/$b" && cp "$ROOT/$b"/* "$STAGE/$b/"
done
cp "$ROOT/tools/release/MODELS-NOTICE.md" "$STAGE/"
rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" .)
echo "$OUT"
