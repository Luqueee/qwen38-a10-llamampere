#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-lambdalabs}"
REMOTE="/home/ubuntu"
ACTION="${1:-help}"

ssh_run() {
    ssh "$HOST" "$@"
}

install_remote() {
ssh "$HOST" 'bash -s' <<'REMOTE_INSTALL'
set -euo pipefail

ROOT="/home/ubuntu"
WORK="$ROOT/tryton-replay"
VENV="$ROOT/capture-env"

mkdir -p "$WORK/results"

echo "==> Preparing Python..."

if [[ ! -x "$VENV/bin/python" ]]; then
    if ! python3 -m venv "$VENV" 2>/dev/null; then
        echo "python3-venv is not installed; installing it..."

        if [[ "$(id -u)" == "0" ]]; then
            apt-get update
            apt-get install -y python3-venv
        else
            sudo apt-get update
            sudo apt-get install -y python3-venv
        fi

        python3 -m venv "$VENV"
    fi
fi

"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q aiohttp

echo "==> Installing capture proxy..."

cat > "$ROOT/capture-proxy.py" <<'PY'
#!/usr/bin/env python3

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout


LISTEN_HOST = os.environ.get("CAPTURE_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("CAPTURE_PORT", "8001"))

UPSTREAM = os.environ.get(
    "CAPTURE_UPSTREAM",
    "http://127.0.0.1:8000",
)

OUTDIR = Path(
    os.environ.get(
        "CAPTURE_OUTDIR",
        "/home/ubuntu/tryton-replay",
    )
)

OUTDIR.mkdir(parents=True, exist_ok=True)

REQUESTS_FILE = OUTDIR / "requests.jsonl"

counter = 0
lock = asyncio.Lock()


async def next_id():
    global counter

    async with lock:
        counter += 1
        return counter


def safe_headers(headers):
    result = {}

    for key, value in headers.items():
        lk = key.lower()

        if lk in {
            "authorization",
            "cookie",
            "proxy-authorization",
        }:
            continue

        result[key] = value

    return result


async def on_startup(app):
    app["session"] = ClientSession(
        timeout=ClientTimeout(total=None),
        auto_decompress=False,
    )


async def on_cleanup(app):
    await app["session"].close()


async def proxy(request):
    req_id = await next_id()
    body = await request.read()

    upstream_headers = {}

    for key, value in request.headers.items():
        lk = key.lower()

        if lk in {
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
        }:
            continue

        upstream_headers[key] = value

    url = UPSTREAM + request.rel_url.path_qs

    record = {
        "id": req_id,
        "timestamp": time.time(),
        "method": request.method,
        "path": request.rel_url.path_qs,
        "headers": safe_headers(upstream_headers),
        "body_base64": base64.b64encode(body).decode("ascii"),
    }

    try:
        record["json"] = json.loads(body)
    except Exception:
        pass

    with REQUESTS_FILE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )

    session = request.app["session"]

    async with session.request(
        request.method,
        url,
        headers=upstream_headers,
        data=body,
    ) as upstream:

        response_headers = {}

        for key, value in upstream.headers.items():
            lk = key.lower()

            if lk in {
                "content-length",
                "transfer-encoding",
                "connection",
            }:
                continue

            response_headers[key] = value

        response = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=response_headers,
        )

        await response.prepare(request)

        try:
            async for chunk in upstream.content.iter_chunked(65536):
                await response.write(chunk)
        except ConnectionResetError:
            pass

        try:
            await response.write_eof()
        except ConnectionResetError:
            pass

        return response


app = web.Application(
    client_max_size=128 * 1024 * 1024
)

app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
app.router.add_route("*", "/{path:.*}", proxy)


if __name__ == "__main__":
    print(
        f"Listening: {LISTEN_HOST}:{LISTEN_PORT}",
        flush=True,
    )
    print(
        f"Upstream : {UPSTREAM}",
        flush=True,
    )
    print(
        f"Capture  : {REQUESTS_FILE}",
        flush=True,
    )

    web.run_app(
        app,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        access_log=None,
    )
PY

chmod +x "$ROOT/capture-proxy.py"


echo "==> Installing proxy control..."

cat > "$ROOT/capture-control.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu"
WORK="$ROOT/tryton-replay"
VENV="$ROOT/capture-env"

PIDFILE="$WORK/capture-proxy.pid"
LOGFILE="$WORK/capture-proxy.log"

mkdir -p "$WORK"

case "${1:-status}" in

start)
    if [[ -f "$PIDFILE" ]]; then
        PID="$(cat "$PIDFILE")"

        if kill -0 "$PID" 2>/dev/null; then
            echo "Proxy already active: PID=$PID"
            exit 0
        fi
    fi

    nohup \
        "$VENV/bin/python" \
        "$ROOT/capture-proxy.py" \
        > "$LOGFILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PIDFILE"

    for i in $(seq 1 30); do
        if python3 - <<'PY' >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", 8001))
s.close()
PY
        then
            echo "Proxy active: PID=$PID"
            echo "8001 -> 127.0.0.1:8000"
            exit 0
        fi

        sleep 1
    done

    echo "ERROR: proxy did not open port 8001"
    tail -100 "$LOGFILE" || true
    exit 1
    ;;

stop)
    if [[ -f "$PIDFILE" ]]; then
        PID="$(cat "$PIDFILE")"

        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            wait "$PID" 2>/dev/null || true
        fi

        rm -f "$PIDFILE"
    fi

    echo "Proxy stopped"
    ;;

reset)
    : > "$WORK/requests.jsonl"
    rm -f "$WORK/inference-requests.jsonl"

    echo "Capture cleared"
    ;;

status)
    echo "=== CAPTURE PROXY ==="

    if [[ -f "$PIDFILE" ]] &&
       kill -0 "$(cat "$PIDFILE")" 2>/dev/null
    then
        echo "status : RUNNING"
        echo "pid    : $(cat "$PIDFILE")"
    else
        echo "status : STOPPED"
    fi

    echo
    echo "=== PORTS ==="

    ss -ltnp 2>/dev/null |
        grep -E ':8000|:8001' ||
        true

    echo
    echo "=== CAPTURE ==="

    wc -l \
        "$WORK/requests.jsonl" \
        "$WORK/inference-requests.jsonl" \
        2>/dev/null ||
        true

    echo
    echo "=== CURRENT VOCAB ==="

    python3 - <<'PY'
import json
import subprocess

try:
    d = json.loads(
        subprocess.check_output(
            ["docker", "inspect", "model-serving"]
        )
    )[0]

    cmd = d["Config"].get("Cmd") or []

    try:
        i = cmd.index("--spec-draft-vocab-map")
        print(cmd[i + 1])
    except ValueError:
        print("FULL VOCAB")

except Exception as exc:
    print("Could not inspect model-serving:", exc)
PY
    ;;

*)
    echo "Usage: $0 start|stop|reset|status"
    exit 1
    ;;
esac
SH

chmod +x "$ROOT/capture-control.sh"


echo "==> Installing capture finalizer..."

cat > "$ROOT/finalize-capture.py" <<'PY'
#!/usr/bin/env python3

import json
import re
from collections import Counter
from pathlib import Path

WORK = Path("/home/ubuntu/tryton-replay")

SOURCE = WORK / "requests.jsonl"
TARGET = WORK / "inference-requests.jsonl"

pattern = re.compile(
    r"^/(?:v1/)?(?:"
    r"chat/completions|"
    r"responses|"
    r"completions"
    r")(?:\?|$)"
)

count = 0
paths = Counter()

with SOURCE.open(
    "r",
    encoding="utf-8",
) as src, TARGET.open(
    "w",
    encoding="utf-8",
) as dst:

    for line in src:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        path = record.get("path", "")

        if not pattern.search(path):
            continue

        dst.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        count += 1
        paths[path.split("?", 1)[0]] += 1


print(f"Inference requests: {count}")
print(f"Output: {TARGET}")

for path, n in sorted(paths.items()):
    print(f"  {path}: {n}")
PY

chmod +x "$ROOT/finalize-capture.py"


echo "==> Installing replay..."

cat > "$ROOT/replay-tryton.py" <<'PY'
#!/usr/bin/env python3

import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def discover_api_key():
    value = os.environ.get("MODEL_SERVING_API_KEY")

    if value:
        return value

    try:
        inspect = json.loads(
            subprocess.check_output(
                ["docker", "inspect", "model-serving"]
            )
        )[0]

        env = inspect["Config"].get("Env") or []

        wanted = {
            "MODEL_SERVING_API_KEY",
            "LLAMA_API_KEY",
        }

        for entry in env:
            if "=" not in entry:
                continue

            key, value = entry.split("=", 1)

            if key in wanted and value:
                return value

        cmd = inspect["Config"].get("Cmd") or []

        for flag in (
            "--api-key",
            "--api-key-file",
        ):
            if flag in cmd:
                i = cmd.index(flag)

                if (
                    i + 1 < len(cmd)
                    and flag == "--api-key"
                ):
                    return cmd[i + 1]

    except Exception:
        pass

    return None


def replay(record, server, api_key):
    body = base64.b64decode(
        record["body_base64"]
    )

    url = server.rstrip("/") + record["path"]

    headers = {
        "Content-Type": "application/json",
    }

    safe = record.get("headers") or {}

    if "Accept" in safe:
        headers["Accept"] = safe["Accept"]

    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}"
        )

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=record.get("method", "POST"),
    )

    started = time.perf_counter()
    first_byte = None

    digest = hashlib.sha256()
    total_bytes = 0

    try:
        response = urllib.request.urlopen(
            req,
            timeout=1800,
        )

        status = response.status

        first = response.read(1)

        if first:
            first_byte = (
                time.perf_counter()
                - started
            )

            digest.update(first)
            total_bytes += 1

        while True:
            chunk = response.read(65536)

            if not chunk:
                break

            digest.update(chunk)
            total_bytes += len(chunk)

        response.close()

    except urllib.error.HTTPError as exc:
        status = exc.code
        data = exc.read()

        digest.update(data)
        total_bytes = len(data)

    elapsed = time.perf_counter() - started

    return {
        "id": record.get("id"),
        "path": record.get("path"),
        "status": status,
        "elapsed_seconds": elapsed,
        "ttfb_seconds": first_byte,
        "response_bytes": total_bytes,
        "response_sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "/home/ubuntu/tryton-replay/"
            "inference-requests.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
    )

    args = parser.parse_args()

    records = []

    with Path(args.input).open() as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    if not records:
        raise SystemExit(
            "No inference requests to replay"
        )

    api_key = discover_api_key()

    print(f"Requests : {len(records)}")
    print(f"Server   : {args.server}")
    print(
        "Auth     : "
        + ("detected" if api_key else "none")
    )
    print()

    results = []

    all_started = time.perf_counter()

    for pos, record in enumerate(records, 1):
        print(
            f"[{pos:02d}/{len(records):02d}] "
            f"id={record.get('id')} ... ",
            end="",
            flush=True,
        )

        result = replay(
            record,
            args.server,
            api_key,
        )

        results.append(result)

        print(
            f"{result['elapsed_seconds']:.3f}s "
            f"HTTP={result['status']}"
        )

    total = time.perf_counter() - all_started

    ok = sum(
        1
        for result in results
        if 200 <= result["status"] < 300
    )

    output = {
        "total_seconds": total,
        "request_count": len(results),
        "successful_requests": ok,
        "requests": results,
    }

    Path(args.output).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("==============================")
    print(" REPLAY")
    print("==============================")
    print(f"Requests : {len(results)}")
    print(f"Success  : {ok}/{len(results)}")
    print(f"Total    : {total:.3f} s")
    print(f"Output   : {args.output}")

    if ok != len(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
PY

chmod +x "$ROOT/replay-tryton.py"


echo "==> Installing analyzer..."

cat > "$ROOT/parse-replay-benchmark.py" <<'PY'
#!/usr/bin/env python3

import csv
import json
import re
import statistics
import sys
from pathlib import Path


logfile = Path(sys.argv[1])
gpufile = Path(sys.argv[2])
replayfile = Path(sys.argv[3])
summaryfile = Path(sys.argv[4])
label = sys.argv[5]
vocab = sys.argv[6]


text = logfile.read_text(
    encoding="utf-8",
    errors="replace",
)

replay = json.loads(
    replayfile.read_text()
)


prompt_re = re.compile(
    r"prompt eval time =\s*"
    r"([0-9.]+) ms /\s*"
    r"([0-9]+) tokens .*?"
    r"([0-9.]+) tokens per second"
)

eval_re = re.compile(
    r"\|\s+eval time =\s*"
    r"([0-9.]+) ms /\s*"
    r"([0-9]+) tokens .*?"
    r"([0-9.]+) tokens per second"
)

draft_re = re.compile(
    r"draft acceptance =\s*([0-9.]+)\s*"
    r"\(\s*([0-9]+) accepted /\s*"
    r"([0-9]+) generated\),"
    r"\s*mean len =\s*([0-9.]+)"
)


prompts = []
evals = []
drafts = []

for line in text.splitlines():
    m = prompt_re.search(line)

    if m:
        prompts.append({
            "ms": float(m.group(1)),
            "tokens": int(m.group(2)),
        })
        continue

    if "prompt eval time" not in line:
        m = eval_re.search(line)

        if m:
            evals.append({
                "ms": float(m.group(1)),
                "tokens": int(m.group(2)),
            })

    m = draft_re.search(line)

    if m:
        drafts.append({
            "acceptance": float(m.group(1)),
            "accepted": int(m.group(2)),
            "generated": int(m.group(3)),
            "mean_len": float(m.group(4)),
        })


prompt_tokens = sum(x["tokens"] for x in prompts)
prompt_ms = sum(x["ms"] for x in prompts)

gen_tokens = sum(x["tokens"] for x in evals)
eval_ms = sum(x["ms"] for x in evals)

prefill_tps = (
    prompt_tokens / (prompt_ms / 1000)
    if prompt_ms else None
)

decode_tps = (
    gen_tokens / (eval_ms / 1000)
    if eval_ms else None
)

accepted = sum(
    x["accepted"]
    for x in drafts
)

draft_generated = sum(
    x["generated"]
    for x in drafts
)

acceptance = (
    accepted / draft_generated
    if draft_generated else None
)

mean_len = (
    sum(
        x["mean_len"] * x["generated"]
        for x in drafts
    )
    / draft_generated
    if draft_generated else None
)


mem = []
util = []
power = []

if gpufile.exists():
    with gpufile.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                mem.append(
                    float(row["memory_mib"])
                )
                util.append(
                    float(row["gpu_util_pct"])
                )
                power.append(
                    float(row["power_w"])
                )
            except Exception:
                pass


e2e = replay["total_seconds"]

avg_power = (
    statistics.mean(power)
    if power else None
)

energy_wh = (
    avg_power * e2e / 3600
    if avg_power is not None
    else None
)

summary = {
    "label": label,
    "vocab": vocab,

    "e2e_seconds": e2e,

    "requests": {
        "count": replay["request_count"],
        "successful": replay[
            "successful_requests"
        ],
    },

    "prompt": {
        "tokens": prompt_tokens,
        "seconds": prompt_ms / 1000,
        "tokens_per_second": prefill_tps,
    },

    "generation": {
        "tokens": gen_tokens,
        "seconds": eval_ms / 1000,
        "tokens_per_second": decode_tps,
    },

    "mtp": {
        "accepted": accepted,
        "draft_generated": draft_generated,
        "acceptance": acceptance,
        "mean_len_weighted": mean_len,
    },

    "gpu": {
        "memory_peak_mib": (
            max(mem) if mem else None
        ),
        "utilization_avg_percent": (
            statistics.mean(util)
            if util else None
        ),
        "power_avg_w": avg_power,
        "energy_wh": energy_wh,
    },

    "response_sha256": [
        x["response_sha256"]
        for x in replay["requests"]
    ],
}

summaryfile.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


def f(value, digits=2):
    if value is None:
        return "-"

    return f"{value:.{digits}f}"


print()
print("==================================================")
print(" REPLAY BENCHMARK")
print("==================================================")
print()
print(f"Label          : {label}")
print(f"Vocab          : {vocab}")
print(f"Requests       : {replay['request_count']}")
print(
    f"Successful     : "
    f"{replay['successful_requests']}/"
    f"{replay['request_count']}"
)
print(f"E2E            : {e2e:.3f} s")
print()
print(f"Prompt tokens  : {prompt_tokens}")
print(f"Prefill        : {f(prefill_tps)} tok/s")
print()
print(f"Generated      : {gen_tokens}")
print(f"Decode         : {f(decode_tps)} tok/s")
print()
print(
    "MTP acceptance : "
    + f(
        acceptance * 100
        if acceptance is not None
        else None
    )
    + " %"
)
print(f"MTP mean len   : {f(mean_len)}")
print()
print(
    f"VRAM peak      : "
    f"{f(max(mem) if mem else None)} MiB"
)
print(
    f"GPU util avg   : "
    f"{f(statistics.mean(util) if util else None)} %"
)
print(
    f"Power avg      : {f(avg_power)} W"
)
print(
    f"Energy         : {f(energy_wh, 3)} Wh"
)
print()
print(f"Summary        : {summaryfile}")
PY

chmod +x "$ROOT/parse-replay-benchmark.py"


echo "==> Installing replay benchmark..."

cat > "$ROOT/replay-benchmark.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-}"

if [[ -z "$LABEL" ]]; then
    echo "Usage: $0 <label>"
    exit 1
fi

ROOT="/home/ubuntu"
WORK="$ROOT/tryton-replay"
CONTAINER="model-serving"

CAPTURE="$WORK/inference-requests.jsonl"

if [[ ! -s "$CAPTURE" ]]; then
    echo "ERROR: finalized capture does not exist:"
    echo "  $CAPTURE"
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
    echo "ERROR: model-serving is not running"
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$WORK/results/${STAMP}_${LABEL}"

mkdir -p "$OUT"

VOCAB="$(
python3 - <<'PY'
import json
import subprocess

d = json.loads(
    subprocess.check_output(
        ["docker", "inspect", "model-serving"]
    )
)[0]

cmd = d["Config"].get("Cmd") or []

try:
    i = cmd.index("--spec-draft-vocab-map")
    print(cmd[i + 1])
except ValueError:
    print("FULL_VOCAB")
PY
)"

echo
echo "=========================================="
echo " FIXED REPLAY"
echo "=========================================="
echo
echo "Label : $LABEL"
echo "Vocab : $VOCAB"
echo "Out   : $OUT"
echo

docker inspect "$CONTAINER" \
    > "$OUT/docker-inspect.json"

BEFORE_LINES="$(
    docker logs "$CONTAINER" 2>&1 |
    wc -l
)"

GPUFILE="$OUT/gpu.csv"

echo \
"epoch,memory_mib,gpu_util_pct,power_w" \
> "$GPUFILE"

sample_gpu() {
    while true; do
        TS="$(date +%s.%N)"

        DATA="$(
            nvidia-smi \
                --query-gpu=memory.used,utilization.gpu,power.draw \
                --format=csv,noheader,nounits \
                2>/dev/null |
                head -n1 |
                tr -d ' '
        )"

        if [[ -n "$DATA" ]]; then
            echo "$TS,$DATA" >> "$GPUFILE"
        fi

        sleep 1
    done
}

sample_gpu &
GPU_PID=$!

cleanup() {
    kill "$GPU_PID" >/dev/null 2>&1 || true
    wait "$GPU_PID" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

"$ROOT/replay-tryton.py" \
    --input "$CAPTURE" \
    --output "$OUT/replay.json"

cleanup
trap - EXIT INT TERM

sleep 1

docker logs "$CONTAINER" 2>&1 |
    tail -n "+$((BEFORE_LINES + 1))" \
    > "$OUT/model-serving.log"

"$ROOT/parse-replay-benchmark.py" \
    "$OUT/model-serving.log" \
    "$GPUFILE" \
    "$OUT/replay.json" \
    "$OUT/summary.json" \
    "$LABEL" \
    "$VOCAB"
SH

chmod +x "$ROOT/replay-benchmark.sh"


echo "==> Installing comparator..."

cat > "$ROOT/compare-replays.py" <<'PY'
#!/usr/bin/env python3

import json
from pathlib import Path

root = Path(
    "/home/ubuntu/tryton-replay/results"
)

rows = []

for path in sorted(
    root.glob("*/summary.json")
):
    d = json.loads(path.read_text())

    rows.append({
        "label": d["label"],
        "vocab": d["vocab"],
        "e2e": d["e2e_seconds"],
        "requests": d["requests"]["count"],
        "decode": d["generation"][
            "tokens_per_second"
        ],
        "acceptance": d["mtp"][
            "acceptance"
        ],
        "mean_len": d["mtp"][
            "mean_len_weighted"
        ],
        "prefill": d["prompt"][
            "tokens_per_second"
        ],
        "energy": d["gpu"]["energy_wh"],
    })


if not rows:
    raise SystemExit(
        "No results yet"
    )


print()
print(
    f"{'LABEL':20}"
    f"{'REQ':>6}"
    f"{'E2E':>10}"
    f"{'DECODE':>11}"
    f"{'ACC%':>10}"
    f"{'MEAN':>9}"
    f"{'PREFILL':>11}"
    f"{'Wh':>9}"
)

print("-" * 86)


def f(v, n=2):
    if v is None:
        return "-"

    return f"{v:.{n}f}"


for r in rows:
    acc = (
        r["acceptance"] * 100
        if r["acceptance"] is not None
        else None
    )

    print(
        f"{r['label'][:20]:20}"
        f"{r['requests']:>6}"
        f"{f(r['e2e']):>10}"
        f"{f(r['decode']):>11}"
        f"{f(acc):>10}"
        f"{f(r['mean_len']):>9}"
        f"{f(r['prefill']):>11}"
        f"{f(r['energy'], 3):>9}"
    )

print()
PY

chmod +x "$ROOT/compare-replays.py"


echo "==> Starting proxy..."

"$ROOT/capture-control.sh" stop >/dev/null 2>&1 || true
"$ROOT/capture-control.sh" start

echo
echo "=============================================="
echo " INSTALLATION COMPLETED"
echo "=============================================="
echo
"$ROOT/capture-control.sh" status
REMOTE_INSTALL
}


case "$ACTION" in

install)
    install_remote
    ;;

start)
    ssh_run "$REMOTE/capture-control.sh start"
    ;;

stop)
    ssh_run "$REMOTE/capture-control.sh stop"
    ;;

reset)
    ssh_run "$REMOTE/capture-control.sh reset"
    ;;

status)
    ssh_run "$REMOTE/capture-control.sh status"
    ;;

finalize)
    ssh_run "$REMOTE/finalize-capture.py"
    ;;

replay)
    LABEL="${2:-}"

    if [[ -z "$LABEL" ]]; then
        echo "Usage:"
        echo "  $0 replay <label>"
        exit 1
    fi

    ssh "$HOST" \
        "$REMOTE/replay-benchmark.sh $(printf '%q' "$LABEL")"
    ;;

compare)
    ssh_run "$REMOTE/compare-replays.py"
    ;;

help|*)
    cat <<EOF

Usage:

  $0 install
  $0 start
  $0 stop
  $0 reset
  $0 status
  $0 finalize
  $0 replay <label>
  $0 compare

SSH host:
  $HOST

Architecture:

  Tryton -> :8001 -> :8000 model-serving

Replay runs directly against:
  127.0.0.1:8000

EOF
    ;;
esac
