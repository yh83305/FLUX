#!/usr/bin/env bash
# Restart an unexpectedly exited scheduler, never a blocked/circuit-open job.
set -u
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
child=""
stop=0
trap 'stop=1; if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi' TERM INT
for attempt in 1 2 3; do
  python3 -u "$script_dir/run_continuous_suite.py" --config "$1" &
  child=$!
  wait "$child"
  rc=$?
  if [[ "$stop" == 1 || "$rc" == 0 ]]; then exit "$rc"; fi
  echo "Scheduler unexpectedly exited rc=$rc; supervisor attempt=$attempt/3"
  [[ "$attempt" == 3 ]] && break
  sleep 30
done
echo "ALERT: scheduler restart circuit open; inspect queue.log"
exit 1
