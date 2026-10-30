#!/usr/bin/env bash
# Starts or stops the app's server against a scratch data folder (never the real data/),
# seeding a few odors into its odor table first.
#   OLFDB_DATA_DIR=<scratch folder> .claude/skills/run-hong-data/serve.sh start [port]   (port 8765 by default)
#   .claude/skills/run-hong-data/serve.sh stop [port]
set -euo pipefail
skill=$(cd "$(dirname "$0")" && pwd)
cd "$skill/../../.."
port=${2:-8765}
case "${1:-}" in
  start)
    [ -n "${OLFDB_DATA_DIR:-}" ] || { echo "set OLFDB_DATA_DIR to a scratch folder (never the real data/)" >&2; exit 2; }
    data=$(realpath -m "$OLFDB_DATA_DIR")
    [ "$data" != "$(realpath -m data)" ] || { echo "OLFDB_DATA_DIR is the real data/" >&2; exit 2; }
    if lsof -ti:"$port" -sTCP:LISTEN >/dev/null; then echo "port $port is in use: serve.sh stop $port first" >&2; exit 1; fi
    export OLFDB_DATA_DIR=$data
    PYTHONPATH=$PWD ./venv/bin/python "$skill/seed.py"
    log="$data.log"
    nohup ./venv/bin/uvicorn app.main:app --port "$port" > "$log" 2>&1 &
    if ! timeout 60 bash -c "until curl -sf http://localhost:$port/api/health >/dev/null; do sleep 1; done"; then
      echo "the server did not answer in 60 s:" >&2; tail -20 "$log" >&2; exit 1
    fi
    echo "serving http://localhost:$port/ (data: $data, log: $log)"
    ;;
  stop)
    lsof -ti:"$port" -sTCP:LISTEN | xargs -r kill
    ;;
  *)
    echo "usage: OLFDB_DATA_DIR=<scratch folder> $0 start [port] | $0 stop [port]" >&2; exit 2
    ;;
esac
