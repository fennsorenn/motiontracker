# Merger evaluation on the rendered sets (resumable per sequence); run under tools/guarded-run.sh.
# usage: mergereval.sh <run name>   (rendered sets, resumable per sequence)
set -e
export OMP_NUM_THREADS=2
R=$1
cd "$(dirname "$0")/../.."
for set in ${SETS:-m1/v1 m1/rest1 m1/contact1 m1/head1 m4/pairs3 m4/hands3}; do
  for d in data/${set%%/*}/renders/${set#*/}/seq*; do
    O=data/merger/eval/$R/${set#*/}/$(basename $d)
    [ -f $O/report/report.json ] && continue
    mkdir -p $O
    [ -f $O/merger.poses.jsonl ] || (cd train && uv run python -m mttrain.merger.run ../data/merger/runs/$R/best.pt ../$d/landmarks.jsonl --out-dir ../$O)
    uv run --project eval mteval evaluate $d --system m1=$O/m1.poses.jsonl --system base=$O/base.poses.jsonl --system merger=$O/merger.poses.jsonl \
       --gt $d/gt.jsonl --no-calibration --out $O/report.tmp > /dev/null && rm -rf $O/report && mv $O/report.tmp $O/report
  done
done
cd train && uv run python -m mttrain.merger.evalsummary ../data/merger/eval/$R
echo EVAL DONE
