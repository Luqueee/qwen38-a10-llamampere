#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER="$1"
ROOT="$2"

rm -f "$ROOT/READY"
rm -f "$ROOT/STOP"

echo "[target] restarting $CONTAINER"

docker restart "$CONTAINER" >/dev/null

echo "[target] waiting for /health"

for i in $(seq 1 480); do
    if curl -fsS \
        http://127.0.0.1:8000/health \
        >/dev/null 2>&1
    then
        echo "[target] healthy"
        touch "$ROOT/READY"
        break
    fi

    sleep 0.25
done

if [[ ! -f "$ROOT/READY" ]]; then
    echo "[target] ERROR: health check timed out" >&2
    exit 1
fi

while [[ ! -f "$ROOT/STOP" ]]; do
    sleep 0.1
done

echo "[target] STOP received"
