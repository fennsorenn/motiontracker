#!/bin/bash
# Run a heavy job without starving the machine (2026-11-08: an unbounded refit + training run
# froze the desktop and the machine crashed).
#
#   tools/guarded-run.sh [-c CORES] [-l MAX_LOAD] [-m MIN_AVAIL_GB] [-o LOG] -- command args...
#
# - CPU: the job and every thread it starts are pinned to CORES (default: the upper half of the
#   logical CPUs) and niced, so the other cores stay free whatever the job does. pymomentum's solver
#   (dispenso) starts one thread per logical CPU in every process, ignoring OMP/BLAS limits and CPU
#   affinity, so pinning is the only reliable cap. Pinned threads still count in the load average,
#   so load is only a runaway check here (default 4x the logical CPUs).
# - Watchdog: every 5 s, if available memory drops below MIN_AVAIL_GB (default 6), or the 1-minute
#   load average stays above MAX_LOAD for 3 checks in a row, the whole job (its process group) is
#   killed and the reason is logged.
set -u
N=$(nproc)
CORES="$((N / 2))-$((N - 1))"
MAX_LOAD=$((4 * N))
MIN_GB=6
LOG=/dev/stderr
while [ $# -gt 0 ]; do
  case "$1" in
    -c) CORES=$2; shift 2 ;;
    -l) MAX_LOAD=$2; shift 2 ;;
    -m) MIN_GB=$2; shift 2 ;;
    -o) LOG=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [-c CORES] [-l MAX_LOAD] [-m MIN_AVAIL_GB] [-o LOG] -- command..." >&2; exit 2; }

setsid nice -n 15 taskset -c "$CORES" "$@" >>"$LOG" 2>&1 &
PID=$!
PGID=$PID  # setsid makes the job its own process group leader
echo "[guard] pid $PID pgid $PGID cores $CORES max load $MAX_LOAD min avail ${MIN_GB}G: $*" >>"$LOG"
over=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 5
  load=$(cut -d' ' -f1 /proc/loadavg)
  avail=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
  if awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN { exit !(l > m) }'; then over=$((over + 1)); else over=0; fi
  reason=""
  [ "$over" -ge 3 ] && reason="load $load > $MAX_LOAD for 15 s"
  [ "$avail" -lt "$MIN_GB" ] && reason="available memory ${avail}G < ${MIN_GB}G"
  if [ -n "$reason" ]; then
    echo "[guard] killing job: $reason" >>"$LOG"
    kill -TERM -- "-$PGID" 2>/dev/null; sleep 5; kill -KILL -- "-$PGID" 2>/dev/null
    exit 99
  fi
done
wait "$PID"
code=$?
echo "[guard] job exited with $code" >>"$LOG"
exit $code
