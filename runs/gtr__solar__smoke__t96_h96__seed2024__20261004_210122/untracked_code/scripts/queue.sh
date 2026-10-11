#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
queue_python="${QUEUE_PYTHON:-$project_dir/.conda/bin/python}"
if [[ ! -x "$queue_python" ]]; then queue_python="$(command -v python3)"; fi
queue_command="${1:-status}"
queue_config="${2:-$project_dir/configs/queue.json}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$queue_command" in
  start|resume)
    runtime_dir="$($queue_python -c 'import json,pathlib,sys; c=json.load(open(sys.argv[1])); print(pathlib.Path(c.get("runtime_dir",sys.argv[2]+"/runtime")).resolve())' "$queue_config" "$project_dir")"
    mkdir -p "$runtime_dir" "$project_dir/logs"
    queue_log="$runtime_dir/queue.log"
    nohup "$queue_python" "$project_dir/scripts/queue.py" "$queue_command" --config "$queue_config" "$@" >> "$queue_log" 2>&1 < /dev/null &
    launcher_pid=$!
    # This is only the launcher PID; Python atomically records its verified identity.
    printf '%s\n' "$launcher_pid" > "$runtime_dir/queue.launcher.pid"
    printf 'Queue launcher PID %s; log %s\n' "$launcher_pid" "$queue_log"
    printf 'Observe: tail -f %q\n' "$queue_log"
    ;;
  status|stop)
    exec "$queue_python" "$project_dir/scripts/queue.py" "$queue_command" --config "$queue_config" "$@"
    ;;
  *) printf 'Usage: bash scripts/queue.sh {start|resume|status|stop} [queue.json] [--retry-failed ID]\n' >&2; exit 2 ;;
esac
