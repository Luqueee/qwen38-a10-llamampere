#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="${REMOTE:-lambdalabs}"
ORIGINAL="${ORIGINAL:-model-serving}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${BASE_DIR:-$REPO_ROOT/results/experiments}"
REPLAY="${REPLAY:-$REPO_ROOT/bench/harness/replay.sh}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOCAL_ROOT="$BASE_DIR/mmvq-width8-ab/$RUN_ID"

REMOTE_BASE="/home/ubuntu/tryton-replay"
REMOTE_RESULTS="$REMOTE_BASE/results"
REMOTE_ROOT="$REMOTE_BASE/mmvq-width8-ab/$RUN_ID"

BACKUP="${ORIGINAL}-mmvq-ab-${RUN_ID}"

MOVED=0

mkdir -p "$LOCAL_ROOT"/{baseline,mmq,meta}

log() {
    echo
    echo "[$(date '+%H:%M:%S')] $*"
}

die() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}

wait_health() {
    ssh "$REMOTE" '
        set -Eeuo pipefail

        for i in $(seq 1 180); do
            if curl -fsS \
                http://127.0.0.1:8000/health \
                >/dev/null 2>&1
            then
                echo READY
                exit 0
            fi

            sleep 1
        done

        docker logs --tail 120 model-serving >&2 || true
        exit 1
    '
}

restore() {
    if [[ "$MOVED" != "1" ]]; then
        return
    fi

    echo
    echo "Restoring the original model-serving..."

    ssh "$REMOTE" bash -s -- \
        "$ORIGINAL" \
        "$BACKUP" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"
BACKUP="$2"

docker rm -f "$NAME" >/dev/null 2>&1 || true

if ! docker inspect "$BACKUP" >/dev/null 2>&1; then
    echo "Cannot find backup $BACKUP" >&2
    exit 1
fi

docker rename "$BACKUP" "$NAME"
docker start "$NAME" >/dev/null
REMOTE

    MOVED=0
    wait_health >/dev/null || true

    echo "Original model-serving restored."
}

trap restore EXIT INT TERM

echo
echo "================================================="
echo " IQ4_XS WIDTH8 - MMVQ vs MMQ A/B/B/A"
echo "================================================="
echo
echo "Run     : $RUN_ID"
echo "Remote  : $REMOTE"
echo "Results : $LOCAL_ROOT"
echo

[[ -x "$REPLAY" ]] ||
    die "Cannot find $REPLAY"

###############################################################################
# PRECHECK
###############################################################################

log "Precheck"

ssh "$REMOTE" '
set -Eeuo pipefail

command -v docker
command -v curl
command -v python3

docker inspect model-serving >/dev/null

docker ps \
    --filter "name=^/model-serving$" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

echo
echo "GPU:"
nvidia-smi \
    --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
'

###############################################################################
# SAVE CONFIGURATION
###############################################################################

log "Saving original configuration"

ssh "$REMOTE" "docker inspect '$ORIGINAL'" \
    > "$LOCAL_ROOT/meta/original.json"

python3 - "$LOCAL_ROOT/meta/original.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))[0]

cmd = list(d["Config"].get("Cmd") or [])
env = list(d["Config"].get("Env") or [])

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
    "--reasoning-budget": "512",
    "--temp": "0",
    "--top-k": "0",
    "--top-p": "1",
    "--min-p": "0",
}

errors = []

for flag, wanted in expected.items():
    got = val(flag)

    if got != wanted:
        errors.append(
            f"{flag}: expected={wanted}, actual={got}"
        )

for item in env:
    if item.startswith("LLAMA_MTP_GPU_VERIFY="):
        errors.append(
            f"GPU verifier is active: {item}"
        )

    if item.startswith("LLAMA_SPEC_ADAPT_"):
        errors.append(
            f"Adaptive setting is active: {item}"
        )

if errors:
    print("Invalid baseline:")

    for error in errors:
        print("  -", error)

    raise SystemExit(1)

print()
print("Baseline OK")
print("  n_max    :", val("--spec-draft-n-max"))
print(
    "  main KV  :",
    val("--cache-type-k"),
    "/",
    val("--cache-type-v"),
)
print(
    "  draft KV :",
    val("--spec-draft-type-k"),
    "/",
    val("--spec-draft-type-v"),
)
print("  budget   :", val("--reasoning-budget"))

current = [
    x for x in env
    if x.startswith("GGML_MMVQ_NMAX=")
]

print(
    "  MMVQ env :",
    current[0] if current else "<unset>"
)
PY

###############################################################################
# PREPARE REMOTE
###############################################################################

ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT'"

rsync -a \
    "$LOCAL_ROOT/meta/original.json" \
    "$REMOTE:$REMOTE_ROOT/original.json" \
    >/dev/null

###############################################################################
# MOVE ORIGINAL ASIDE
###############################################################################

log "Moving the original model-serving container aside"

ssh "$REMOTE" bash -s -- \
    "$ORIGINAL" \
    "$BACKUP" <<'REMOTE'
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
    local mode="$1"

    log "Creating variant: $mode"

    ssh "$REMOTE" \
        "python3 - '$REMOTE_ROOT/original.json' '$mode' '$REMOTE_ROOT/create.json'" \
        <<'PY'
import json
import sys
from pathlib import Path

source, mode, output = sys.argv[1:4]

d = json.load(open(source))[0]

cfg = dict(d["Config"])
host = dict(d["HostConfig"])

env = list(cfg.get("Env") or [])

# Change only this knob.
env = [
    x for x in env
    if not x.startswith("GGML_MMVQ_NMAX=")
]

if mode == "mmq":
    env.append(
        "GGML_MMVQ_NMAX=iq4_xs=7"
    )
elif mode != "baseline":
    raise SystemExit(
        f"Invalid mode: {mode}"
    )

cfg["Env"] = env
cfg["HostConfig"] = host

Path(output).write_text(
    json.dumps(
        cfg,
        separators=(",", ":"),
    )
)

print(
    "GGML_MMVQ_NMAX="
    + (
        "iq4_xs=7"
        if mode == "mmq"
        else "<unset>"
    )
)
PY

    ssh "$REMOTE" bash -s -- \
        "$ORIGINAL" \
        "$REMOTE_ROOT/create.json" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"
PAYLOAD="$2"

docker rm -f "$NAME" >/dev/null 2>&1 || true

API="$(
    docker version \
        --format '{{.Server.APIVersion}}'
)"

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
    echo "Docker API HTTP=$CODE" >&2
    cat "$RESP" >&2
    exit 1
fi

docker start "$NAME" >/dev/null
REMOTE

    wait_health >/dev/null

    echo "Effective environment:"

    ssh "$REMOTE" "
        docker inspect '$ORIGINAL' \
            --format '{{range .Config.Env}}{{println .}}{{end}}' |
        grep '^GGML_MMVQ_NMAX=' ||
        echo 'GGML_MMVQ_NMAX=<unset>'
    "
}

destroy_variant() {
    ssh "$REMOTE" \
        "docker rm -f '$ORIGINAL' >/dev/null 2>&1 || true"
}

###############################################################################
# COPY RESULT
###############################################################################

copy_result() {
    local mode="$1"
    local label="$2"

    local remote_result

    remote_result="$(
        ssh "$REMOTE" \
            "python3 - '$REMOTE_RESULTS' '$label'" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
label = sys.argv[2]

matches = [
    p for p in root.iterdir()
    if p.is_dir()
    and p.name.endswith("_" + label)
]

if not matches:
    raise SystemExit(
        f"Cannot find result for {label}"
    )

matches.sort(
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

print(matches[0])
PY
    )"

    mkdir -p \
        "$LOCAL_ROOT/$mode/$label"

    rsync -aH \
        "$REMOTE:$remote_result/" \
        "$LOCAL_ROOT/$mode/$label/"
}

###############################################################################
# RUN
###############################################################################

run_one() {
    local mode="$1"
    local suffix="$2"

    local label="iq4-width8-${mode}-${RUN_ID}-${suffix}"

    create_variant "$mode"

    log "Replay $label"

    "$REPLAY" replay "$label" 2>&1 |
        tee "$LOCAL_ROOT/$mode/$label.log"

    ssh "$REMOTE" \
        "docker logs '$ORIGINAL' 2>&1" \
        > "$LOCAL_ROOT/$mode/$label.docker.log"

    copy_result "$mode" "$label"

    destroy_variant
}

###############################################################################
# A / B / B / A
###############################################################################

echo
echo "Experiment order:"
echo "  A1 baseline"
echo "  B1 IQ4_XS width8 -> MMQ"
echo "  B2 IQ4_XS width8 -> MMQ"
echo "  A2 baseline"
echo

run_one baseline a1
run_one mmq      b1
run_one mmq      b2
run_one baseline a2

###############################################################################
# RESTORE BEFORE ANALYSIS
###############################################################################

restore

###############################################################################
# ANALYSIS
###############################################################################

log "Analyzing A/B/B/A"

python3 - "$LOCAL_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import statistics
import sys

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
    "gpu":
        r"^GPU util avg\s*:\s*([0-9.]+)",
    "power":
        r"^Power avg\s*:\s*([0-9.]+)",
    "energy":
        r"^Energy\s*:\s*([0-9.]+)",
}

def parse(path):
    text = path.read_text(
        errors="replace"
    )

    out = {
        "path": str(path),
    }

    for key, pattern in patterns.items():
        m = re.search(
            pattern,
            text,
            re.M,
        )

        if m:
            out[key] = float(
                m.group(1)
            )

    return out

def load_group(mode):
    logs = sorted(
        (root / mode).glob("*.log")
    )

    logs = [
        x for x in logs
        if not x.name.endswith(
            ".docker.log"
        )
    ]

    return [
        parse(x)
        for x in logs
    ]

a = load_group("baseline")
b = load_group("mmq")

if len(a) != 2 or len(b) != 2:
    raise SystemExit(
        f"Expected A=2 B=2; "
        f"got A={len(a)} B={len(b)}"
    )

def avg(data, key):
    return statistics.mean(
        x[key] for x in data
    )

def stdev(data, key):
    values = [
        x[key] for x in data
    ]

    return statistics.stdev(
        values
    )

metrics = [
    ("E2E s", "e2e"),
    ("Prefill t/s", "prefill"),
    ("Generated", "generated"),
    ("Decode t/s", "decode"),
    ("Acceptance %", "acceptance"),
    ("Mean len", "mean"),
    ("VRAM MiB", "vram"),
    ("GPU util %", "gpu"),
    ("Power W", "power"),
    ("Energy Wh", "energy"),
]

print()
print("=" * 78)
print("IQ4_XS WIDTH8 ROUTING - A/B/B/A RESULTS")
print("=" * 78)

print(
    f'{"Metric":<18}'
    f'{"Baseline":>15}'
    f'{"MMQ width8":>15}'
    f'{"Delta":>15}'
)

print("-" * 78)

for title, key in metrics:
    av = avg(a, key)
    bv = avg(b, key)

    if key in (
        "decode",
        "prefill",
    ):
        delta = (
            bv / av - 1
        ) * 100

        ds = f"{delta:+.2f}%"

    elif key in (
        "e2e",
        "energy",
    ):
        delta = (
            av / bv - 1
        ) * 100

        ds = f"{delta:+.2f}%"

    else:
        ds = f"{bv-av:+.2f}"

    print(
        f"{title:<18}"
        f"{av:>15.2f}"
        f"{bv:>15.2f}"
        f"{ds:>15}"
    )

a_decode = avg(a, "decode")
b_decode = avg(b, "decode")

gain = (
    b_decode / a_decode - 1
) * 100

print()
print("Individual decode runs:")
print(
    "  Baseline :",
    ", ".join(
        f"{x['decode']:.2f}"
        for x in a
    ),
)
print(
    "  MMQ      :",
    ", ".join(
        f"{x['decode']:.2f}"
        for x in b
    ),
)

print()
print(
    f"Baseline mean : "
    f"{a_decode:.2f} tok/s"
)
print(
    f"MMQ mean      : "
    f"{b_decode:.2f} tok/s"
)
print(
    f"Gain          : "
    f"{gain:+.2f}%"
)

print()
print(
    f"Baseline SD   : "
    f"{stdev(a, 'decode'):.3f}"
)
print(
    f"MMQ SD        : "
    f"{stdev(b, 'decode'):.3f}"
)

same_generated = (
    len({
        x["generated"]
        for x in a + b
    }) == 1
)

same_acceptance = (
    len({
        x["acceptance"]
        for x in a + b
    }) == 1
)

same_mean = (
    len({
        x["mean"]
        for x in a + b
    }) == 1
)

print()
print("Sanity:")
print(
    "  Generated identical :",
    "YES" if same_generated else "NO",
)
print(
    "  Acceptance identical:",
    "YES" if same_acceptance else "NO",
)
print(
    "  Mean len identical  :",
    "YES" if same_mean else "NO",
)

if gain >= 2.0:
    decision = (
        "PROMISING: forced MMQ width8 "
        "improves by >=2%. Next step: "
        "profile the MMQ arm with Nsight and validate outputs."
    )

elif gain <= 0.5:
    decision = (
        "CLOSE: MMQ width8 does not provide a useful "
        "improvement. Keep MMVQ and close "
        "this experiment."
    )

else:
    decision = (
        "BORDERLINE: improvement is between 0.5% and 2%. "
        "Repeat before changing production."
    )

print()
print("Decision:")
print(" ", decision)

summary = {
    "baseline": a,
    "mmq": b,
    "baseline_decode_mean": a_decode,
    "mmq_decode_mean": b_decode,
    "gain_pct": gain,
    "same_generated": same_generated,
    "same_acceptance": same_acceptance,
    "same_mean_len": same_mean,
    "decision": decision,
}

(root / "summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

(root / "summary.txt").write_text(
    f"Baseline : {a_decode:.2f} tok/s\n"
    f"MMQ      : {b_decode:.2f} tok/s\n"
    f"Gain     : {gain:+.2f}%\n"
    f"Decision : {decision}\n"
)
PY

echo
echo "=========================================="
echo " EXPERIMENT COMPLETED"
echo "=========================================="
echo
cat "$LOCAL_ROOT/summary.txt"

echo
echo "Results:"
echo "  $LOCAL_ROOT"

echo
echo "model-serving:"
ssh "$REMOTE" \
    "docker ps \
    --filter 'name=^/${ORIGINAL}$' \
    --format '{{.Names}} -> {{.Status}}'"
