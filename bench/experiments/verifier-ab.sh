#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="${REMOTE:-lambdalabs}"
ORIGINAL="${ORIGINAL:-model-serving}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${BASE_DIR:-$REPO_ROOT/results/experiments}"
REPLAY="${REPLAY:-$REPO_ROOT/bench/harness/replay.sh}"
REPEATS="${REPEATS:-1}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOCAL_ROOT="$BASE_DIR/verifier-ab/$RUN_ID"
REMOTE_BASE="/home/ubuntu/tryton-replay"
REMOTE_RESULTS="$REMOTE_BASE/results"
REMOTE_ROOT="$REMOTE_BASE/verifier-ab/$RUN_ID"

BACKUP="${ORIGINAL}-verifier-ab-${RUN_ID}"

CPU_LABEL="v04-n7-unlimited-cpuverify"
GPU_LABEL="v04-n7-unlimited-gpuverify"

MOVED=0

mkdir -p "$LOCAL_ROOT"/{cpu,gpu,meta}

log() {
    echo
    echo "[$(date '+%H:%M:%S')] $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

wait_health() {
    ssh "$REMOTE" '
        for i in $(seq 1 180); do
            if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
                echo READY
                exit 0
            fi
            sleep 1
        done
        docker logs --tail 100 model-serving >&2 || true
        exit 1
    '
}

restore() {
    if [[ "$MOVED" != "1" ]]; then
        return
    fi

    echo
    echo "Restoring the original model-serving..."

    ssh "$REMOTE" bash -s -- "$ORIGINAL" "$BACKUP" <<'REMOTE'
set +e
NAME="$1"
BACKUP="$2"

docker rm -f "$NAME" >/dev/null 2>&1 || true

if docker inspect "$BACKUP" >/dev/null 2>&1; then
    docker rename "$BACKUP" "$NAME"
    docker start "$NAME" >/dev/null
fi
REMOTE

    MOVED=0
    wait_health >/dev/null || true

    echo "Original model-serving restored."
}

trap restore EXIT INT TERM

echo
echo "=========================================="
echo " GPU VERIFIER A/B"
echo "=========================================="
echo
echo "Run     : $RUN_ID"
echo "Remote  : $REMOTE"
echo "Results : $LOCAL_ROOT"
echo

###############################################################################
# PRECHECK
###############################################################################

log "Precheck"

[[ -x "$REPLAY" ]] ||
    die "Cannot find $REPLAY"

ssh "$REMOTE" '
set -Eeuo pipefail

command -v docker
command -v curl
command -v python3
command -v nvidia-smi

docker inspect model-serving >/dev/null

echo
docker ps --filter name=model-serving \
    --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

echo
echo "Command:"
docker inspect model-serving \
    --format "{{json .Config.Cmd}}"

echo
echo "Env:"
docker inspect model-serving \
    --format "{{range .Config.Env}}{{println .}}{{end}}" |
grep -E "^(GGML_Q8|LLAMA_MTP|LLAMA_SPEC)" || true
'

###############################################################################
# SAVE ORIGINAL CONFIGURATION
###############################################################################

log "Saving original configuration"

ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT'"

ssh "$REMOTE" "docker inspect '$ORIGINAL'" \
    > "$LOCAL_ROOT/meta/original.json"

rsync -a \
    "$LOCAL_ROOT/meta/original.json" \
    "$REMOTE:$REMOTE_ROOT/original.json" \
    >/dev/null

###############################################################################
# VALIDATE BASELINE
###############################################################################

python3 - "$LOCAL_ROOT/meta/original.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))[0]
cmd = d["Config"]["Cmd"]
env = d["Config"].get("Env") or []

def val(flag):
    if flag not in cmd:
        raise SystemExit(f"Missing {flag}")
    return cmd[cmd.index(flag) + 1]

expected = {
    "--spec-draft-n-max": "7",
    "--cache-type-k": "q8_0",
    "--cache-type-v": "turbo3",
    "--spec-draft-type-k": "q8_0",
    "--spec-draft-type-v": "turbo3",
    "--temp": "0",
    "--top-k": "0",
    "--top-p": "1",
    "--min-p": "0",
}

for flag, wanted in expected.items():
    got = val(flag)
    if got != wanted:
        raise SystemExit(
            f"{flag}: expected {wanted}, got {got}"
        )

for x in env:
    if x.startswith("LLAMA_SPEC_ADAPT_COST="):
        raise SystemExit(
            "LLAMA_SPEC_ADAPT_COST is still active"
        )

print()
print("Baseline OK")
print("n_max     :", val("--spec-draft-n-max"))
print(
    "main KV   :",
    val("--cache-type-k"),
    "/",
    val("--cache-type-v"),
)
print(
    "draft KV  :",
    val("--spec-draft-type-k"),
    "/",
    val("--spec-draft-type-v"),
)
print(
    "budget    :",
    val("--reasoning-budget"),
    "-> temporarily -1",
)
PY

###############################################################################
# MOVE ORIGINAL ASIDE
###############################################################################

log "Moving the original model-serving container aside"

ssh "$REMOTE" bash -s -- "$ORIGINAL" "$BACKUP" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"
BACKUP="$2"

docker stop "$NAME" >/dev/null
docker rename "$NAME" "$BACKUP"

echo "$NAME -> $BACKUP"
REMOTE

MOVED=1

###############################################################################
# CREATE VARIANT
###############################################################################

create_variant() {
    MODE="$1"

    log "Creating variant $MODE"

    ssh "$REMOTE" \
        "python3 - '$REMOTE_ROOT/original.json' '$MODE' '$REMOTE_ROOT/$MODE.json'" \
        <<'PY'
import json
import sys

source, mode, output = sys.argv[1:4]

d = json.load(open(source))[0]

cfg = dict(d["Config"])
host = dict(d["HostConfig"])

cmd = list(cfg["Cmd"])
env = list(cfg.get("Env") or [])

i = cmd.index("--reasoning-budget")
cmd[i + 1] = "-1"

env = [
    x for x in env
    if not x.startswith("LLAMA_MTP_GPU_VERIFY=")
    and not x.startswith("LLAMA_SPEC_ADAPT_COST=")
    and not x.startswith("LLAMA_SPEC_ADAPT_FLOOR=")
]

if mode == "gpu":
    env.append("LLAMA_MTP_GPU_VERIFY=greedy")
elif mode != "cpu":
    raise SystemExit("Invalid mode")

cfg["Cmd"] = cmd
cfg["Env"] = env
cfg["HostConfig"] = host

with open(output, "w") as f:
    json.dump(cfg, f)

print("reasoning-budget=-1")

if mode == "gpu":
    print("LLAMA_MTP_GPU_VERIFY=greedy")
else:
    print("LLAMA_MTP_GPU_VERIFY=<unset>")
PY

    ssh "$REMOTE" bash -s -- \
        "$ORIGINAL" \
        "$REMOTE_ROOT/$MODE.json" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"
PAYLOAD="$2"

docker rm -f "$NAME" >/dev/null 2>&1 || true

API="$(docker version --format '{{.Server.APIVersion}}')"

RESP="${PAYLOAD}.response"

CODE="$(
    curl -sS \
        --unix-socket /var/run/docker.sock \
        -H 'Content-Type: application/json' \
        -X POST \
        --data-binary "@$PAYLOAD" \
        -o "$RESP" \
        -w '%{http_code}' \
        "http://localhost/v${API}/containers/create?name=${NAME}"
)"

if [[ "$CODE" != "201" ]]; then
    echo "Docker API error: HTTP $CODE"
    cat "$RESP"
    exit 1
fi

docker start "$NAME" >/dev/null
REMOTE

    wait_health
}

destroy_variant() {
    ssh "$REMOTE" \
        "docker rm -f '$ORIGINAL' >/dev/null 2>&1 || true"
}

###############################################################################
# RUN ARM
###############################################################################

run_arm() {
    MODE="$1"
    BASE_LABEL="$2"

    create_variant "$MODE"

    for N in $(seq 1 "$REPEATS"); do

        if [[ "$REPEATS" -eq 1 ]]; then
            LABEL="$BASE_LABEL"
        else
            LABEL="${BASE_LABEL}-r${N}"
        fi

        log "$MODE replay $N/$REPEATS: $LABEL"

        #
        # Cold start to avoid prompt cache reuse
        #
        ssh "$REMOTE" \
            "docker restart '$ORIGINAL' >/dev/null"

        wait_health >/dev/null

        "$REPLAY" replay "$LABEL" 2>&1 |
            tee "$LOCAL_ROOT/$MODE/$LABEL.replay.log"

        ssh "$REMOTE" \
            "docker logs '$ORIGINAL' 2>&1" \
            > "$LOCAL_ROOT/$MODE/$LABEL.docker.log"

        RESULT="$(
            ssh "$REMOTE" \
                "python3 - '$REMOTE_RESULTS' '$LABEL'" \
                <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
label = sys.argv[2]

m = [
    p for p in root.iterdir()
    if p.is_dir()
    and p.name.endswith("_" + label)
]

if not m:
    raise SystemExit("Result not found")

m.sort(
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

print(m[0])
PY
        )"

        mkdir -p "$LOCAL_ROOT/$MODE/$LABEL"

        rsync -aH \
            "$REMOTE:$RESULT/" \
            "$LOCAL_ROOT/$MODE/$LABEL/"

        echo
        echo "=== VERIFIER LOG ==="

        grep -E \
            "GPU .*verification|draft acceptance|round cost by draft width" \
            "$LOCAL_ROOT/$MODE/$LABEL.docker.log" |
            tail -100 || true

        if [[ "$MODE" == "gpu" ]]; then

            ENABLED="$(
                grep -c \
                    "GPU greedy verification enabled" \
                    "$LOCAL_ROOT/$MODE/$LABEL.docker.log" ||
                true
            )"

            INELIGIBLE="$(
                grep -c \
                    "GPU verification ineligible" \
                    "$LOCAL_ROOT/$MODE/$LABEL.docker.log" ||
                true
            )"

            echo
            echo "GPU verifier enabled : $ENABLED"
            echo "GPU ineligible       : $INELIGIBLE"

            if [[ "$ENABLED" -eq 0 ]]; then
                die "GPU verifier was NOT activated"
            fi
        fi

    done

    destroy_variant
}

###############################################################################
# CPU
###############################################################################

log "ARM A: CPU verifier"

run_arm \
    cpu \
    "$CPU_LABEL"

###############################################################################
# GPU
###############################################################################

log "ARM B: GPU verifier"

run_arm \
    gpu \
    "$GPU_LABEL"

###############################################################################
# RESTORE ORIGINAL
###############################################################################

restore

###############################################################################
# ANALYSIS
###############################################################################

log "Comparing results"

python3 - "$LOCAL_ROOT" <<'PY'
from pathlib import Path
import re
import statistics
import sys
import json

root = Path(sys.argv[1])

patterns = {
    "e2e":
        r"^E2E\s*:\s*([0-9.]+)",
    "prefill":
        r"^Prefill\s*:\s*([0-9.]+)",
    "generated":
        r"^Generated\s*:\s*([0-9]+)",
    "decode":
        r"^Decode\s*:\s*([0-9.]+)",
    "acceptance":
        r"^MTP acceptance\s*:\s*([0-9.]+)",
    "mean":
        r"^MTP mean len\s*:\s*([0-9.]+)",
    "vram":
        r"^VRAM peak\s*:\s*([0-9.]+)",
    "energy":
        r"^Energy\s*:\s*([0-9.]+)",
}

def parse(path):
    text = path.read_text(errors="replace")
    out = {}

    for k, p in patterns.items():
        m = re.search(p, text, re.M)

        if m:
            out[k] = float(m.group(1))

    return out

def group(mode):
    return [
        parse(x)
        for x in sorted(
            (root / mode).glob("*.replay.log")
        )
    ]

cpu = group("cpu")
gpu = group("gpu")

if not cpu or not gpu:
    raise SystemExit(
        "No CPU/GPU results found"
    )

def avg(data, key):
    return statistics.mean(
        x[key] for x in data
    )

metrics = [
    ("E2E s", "e2e"),
    ("Prefill t/s", "prefill"),
    ("Generated", "generated"),
    ("Decode t/s", "decode"),
    ("Acceptance %", "acceptance"),
    ("Mean len", "mean"),
    ("VRAM MiB", "vram"),
    ("Energy Wh", "energy"),
]

print()
print("=" * 68)
print("A/B RESULTS")
print("=" * 68)

print(
    f'{"Metric":<18}'
    f'{"CPU":>15}'
    f'{"GPU":>15}'
    f'{"Delta":>15}'
)

print("-" * 68)

for title, key in metrics:

    a = avg(cpu, key)
    b = avg(gpu, key)

    if key in ("decode", "prefill"):
        delta = (
            (b / a - 1) * 100
        )

        ds = f"{delta:+.2f}%"

    elif key in ("e2e", "energy"):
        delta = (
            (a / b - 1) * 100
        )

        ds = f"{delta:+.2f}%"

    else:
        ds = f"{b-a:+.2f}"

    print(
        f"{title:<18}"
        f"{a:>15.2f}"
        f"{b:>15.2f}"
        f"{ds:>15}"
    )

cpu_dec = avg(cpu, "decode")
gpu_dec = avg(gpu, "decode")

gain = (
    gpu_dec / cpu_dec - 1
) * 100

print()
print(
    f"CPU verifier : "
    f"{cpu_dec:.2f} tok/s"
)

print(
    f"GPU verifier : "
    f"{gpu_dec:.2f} tok/s"
)

print(
    f"Gain         : "
    f"{gain:+.2f}%"
)

print()

if gain >= 3:
    decision = (
        "PATCH_REASONING_BUDGET: "
        "GPU verifier implementation "
        "compatible with budget=512 is worth implementing."
    )

elif gain <= 1:
    decision = (
        "KERNEL_NEXT: "
        "the verifier adds little; "
        "focus on IQ4_XS width8."
    )

else:
    decision = (
        "REPEAT_AB: "
        "gain is 1-3%; "
        "repeat with REPEATS=3."
    )

print("Decision:")
print(decision)

(root / "ab-summary.txt").write_text(
    f"CPU decode : {cpu_dec:.2f} tok/s\n"
    f"GPU decode : {gpu_dec:.2f} tok/s\n"
    f"Gain       : {gain:+.2f}%\n"
    f"Decision   : {decision}\n"
)

(root / "ab-summary.json").write_text(
    json.dumps(
        {
            "cpu": cpu,
            "gpu": gpu,
            "decode_gain_pct": gain,
            "decision": decision,
        },
        indent=2,
    )
)
PY

echo
echo "=========================================="
echo " A/B COMPLETED"
echo "=========================================="
echo
echo "Results:"
echo "$LOCAL_ROOT"
echo
cat "$LOCAL_ROOT/ab-summary.txt"
echo
echo "model-serving:"
ssh "$REMOTE" \
    "docker ps --filter name=model-serving \
    --format '{{.Names}} -> {{.Status}}'"
