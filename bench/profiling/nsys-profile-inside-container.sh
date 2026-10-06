#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="lambdalabs"

# Name that bench/harness/replay.sh expects.
ORIGINAL="model-serving"

# During profiling:
#   model-serving-original = normal container parked
#   model-serving          = container instrumented with nsys
BACKUP_CONTAINER="model-serving-original"
PROFILE_CONTAINER="model-serving-nsys"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPLAY="$REPO_ROOT/bench/harness/replay.sh"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LABEL="nsys-inside-${RUN_ID}"

LOCAL_ROOT="$REPO_ROOT/results/profiling/nsys-inside/$RUN_ID"
REMOTE_ROOT="/home/ubuntu/tryton-replay/nsys-inside/$RUN_ID"
REMOTE_RESULTS="/home/ubuntu/tryton-replay/results"

mkdir -p "$LOCAL_ROOT"

exec > >(tee -a "$LOCAL_ROOT/run.log") 2>&1


###############################################################################
# HELPERS
###############################################################################

die() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}


restore_original() {
    set +e

    echo
    echo "Restoring original model-serving..."

    ssh "$REMOTE" "
        set +e

        #
        # If the backup exists, the swap has already happened.
        #
        if docker inspect '$BACKUP_CONTAINER' >/dev/null 2>&1; then

            # The current model-serving is the profiling container.
            docker rm -f '$ORIGINAL' >/dev/null 2>&1 || true

            # In case the profiling container still has its old name.
            docker rm -f '$PROFILE_CONTAINER' >/dev/null 2>&1 || true

            docker rename '$BACKUP_CONTAINER' '$ORIGINAL' \
                >/dev/null 2>&1 || true

            docker start '$ORIGINAL' \
                >/dev/null 2>&1 || true

        else

            # If we failed before the swap, just clean up the temporary container.
            docker rm -f '$PROFILE_CONTAINER' \
                >/dev/null 2>&1 || true

            if docker inspect '$ORIGINAL' >/dev/null 2>&1; then
                docker start '$ORIGINAL' \
                    >/dev/null 2>&1 || true
            fi
        fi
    " || true
}


cleanup() {
    restore_original
}

trap cleanup EXIT INT TERM


echo "=========================================="
echo " NSIGHT INSIDE CONTAINER"
echo "=========================================="
echo
echo "Run    : $RUN_ID"
echo "Label  : $LABEL"
echo "Remote : $REMOTE"
echo "Local  : $LOCAL_ROOT"
echo


###############################################################################
# 1. PRECHECK
###############################################################################

echo "[1/8] Precheck..."

command -v ssh >/dev/null ||
    die "ssh not found"

command -v rsync >/dev/null ||
    die "rsync not found"

[[ -x "$REPLAY" ]] ||
    die "Does not exist or is not executable: $REPLAY"


ssh "$REMOTE" "
    set -e

    command -v docker
    command -v nsys
    command -v nvidia-smi
    command -v curl
    command -v python3

    echo
    nsys --version

    echo
    echo 'Current containers:'

    docker ps -a \
        --format 'table {{.Names}}\t{{.Status}}' |
        grep -E '(^NAMES|model-serving)' || true

    echo
    echo 'NCOLS1_MIN:'

    docker inspect '$ORIGINAL' \
        --format '{{range .Config.Env}}{{println .}}{{end}}' |
        grep '^GGML_Q8_TURBO3_MMA_NCOLS1_MIN=' ||
        echo 'unset -> default 2'
"


#
# Do not start if leftovers from a previous attempt remain.
#
if ssh "$REMOTE" \
    "docker inspect '$BACKUP_CONTAINER' >/dev/null 2>&1"
then
    die "$BACKUP_CONTAINER exists on Lambda. Leftovers from a previous attempt remain."
fi

if ssh "$REMOTE" \
    "docker inspect '$PROFILE_CONTAINER' >/dev/null 2>&1"
then
    echo "Removing old profiling container..."

    ssh "$REMOTE" \
        "docker rm -f '$PROFILE_CONTAINER' >/dev/null"
fi


RUNNING="$(
    ssh "$REMOTE" \
        "docker inspect '$ORIGINAL' --format '{{.State.Running}}'"
)"

[[ "$RUNNING" == "true" ]] ||
    die "$ORIGINAL is not running before starting"


NCOLS="$(
    ssh "$REMOTE" "
        docker inspect '$ORIGINAL' \
            --format '{{range .Config.Env}}{{println .}}{{end}}' |
        sed -n \
          's/^GGML_Q8_TURBO3_MMA_NCOLS1_MIN=//p'
    "
)"

if [[ -n "$NCOLS" && "$NCOLS" != "2" ]]; then
    die "NCOLS1_MIN=$NCOLS; baseline 2 is required for profiling"
fi


echo
echo "Precheck OK."


###############################################################################
# 2. PREPARE REMOTE DIRECTORY
###############################################################################

echo
echo "[2/8] Preparing remote scratch..."

ssh "$REMOTE" "
    set -e

    rm -rf '$REMOTE_ROOT'
    mkdir -p '$REMOTE_ROOT'

    docker inspect '$ORIGINAL' \
        > '$REMOTE_ROOT/original-container.json'

    nvidia-smi -q \
        > '$REMOTE_ROOT/nvidia-smi-before.txt'

    nsys --version \
        > '$REMOTE_ROOT/nsys-version.txt' \
        2>&1
"


###############################################################################
# 3. CREATE INSTRUMENTED CONTAINER
###############################################################################

echo
echo "[3/8] Creating profiling container..."

ssh "$REMOTE" bash -s -- \
    "$ORIGINAL" \
    "$PROFILE_CONTAINER" \
    "$REMOTE_ROOT" <<'REMOTE_SCRIPT'

set -Eeuo pipefail

ORIGINAL="$1"
PROFILE_CONTAINER="$2"
REMOTE_ROOT="$3"

INSPECT="$REMOTE_ROOT/original-container.json"
PAYLOAD="$REMOTE_ROOT/create-profile-container.json"

NSYS_BIN="$(readlink -f "$(command -v nsys)")"

if [[ ! -x "$NSYS_BIN" ]]; then
    echo "ERROR: cannot find the actual nsys binary"
    exit 1
fi

#
# Example:
#
# /opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
#
# We want to mount the entire:
#
# /opt/nvidia/nsight-systems-cli/2026.5.1
#
NSYS_ROOT="$(dirname "$(dirname "$NSYS_BIN")")"

echo "Nsight binary : $NSYS_BIN"
echo "Nsight root   : $NSYS_ROOT"

python3 - \
    "$INSPECT" \
    "$PAYLOAD" \
    "$NSYS_BIN" \
    "$NSYS_ROOT" \
    "$REMOTE_ROOT" <<'PY'

import copy
import json
import sys

src, dst, nsys_bin, nsys_root, remote_root = sys.argv[1:]

with open(src) as f:
    old = json.load(f)[0]

config = copy.deepcopy(old["Config"])
host = copy.deepcopy(old["HostConfig"])

original_exec = old["Path"]
original_args = old.get("Args") or []

print("Original executable:", original_exec)
print("Original args:", len(original_args))


# ---------------------------------------------------------------------------
# The temporary container must NOT restart automatically.
# ---------------------------------------------------------------------------

host["RestartPolicy"] = {
    "Name": "no",
    "MaximumRetryCount": 0,
}


# ---------------------------------------------------------------------------
# Additional mounts.
# ---------------------------------------------------------------------------

binds = list(host.get("Binds") or [])

binds.append(
    f"{nsys_root}:{nsys_root}:ro"
)

binds.append(
    f"{remote_root}:/profiles:rw"
)

host["Binds"] = binds


# ---------------------------------------------------------------------------
# Permissions required for profiling.
# Affect the temporary container only.
# ---------------------------------------------------------------------------

caps = list(host.get("CapAdd") or [])

if "SYS_ADMIN" not in caps:
    caps.append("SYS_ADMIN")

host["CapAdd"] = caps


# ---------------------------------------------------------------------------
# nsys becomes PID 1.
#
# IMPORTANT:
# We do not use --delay or --duration.
#
# We want to ensure that we capture the entire request 15.
# ---------------------------------------------------------------------------

config["Entrypoint"] = [nsys_bin]

config["Cmd"] = [
    "profile",

    "--trace=cuda-sw,nvtx",

    # Show the individual kernels used by CUDA Graphs.
    "--cuda-graph-trace=node:host-only",

    "--sample=none",
    "--cpuctxsw=none",

    "--force-overwrite=true",

    "--output=/profiles/profile",

    original_exec,
    *original_args,
]


payload = {
    **config,
    "HostConfig": host,
}


#
# Preserve user-defined network configuration when present.
#
networks = old.get("NetworkSettings", {}).get("Networks", {})

endpoints = {}

for name, net in networks.items():

    ep = {}

    ipam = net.get("IPAMConfig")

    if ipam:
        ep["IPAMConfig"] = ipam

    aliases = net.get("Aliases")

    if aliases:

        old_id = old.get("Id", "")
        old_name = old.get("Name", "").lstrip("/")

        aliases = [
            a
            for a in aliases
            if a not in {
                old_id,
                old_id[:12],
                old_name,
            }
        ]

        if aliases:
            ep["Aliases"] = aliases

    endpoints[name] = ep


if endpoints:
    payload["NetworkingConfig"] = {
        "EndpointsConfig": endpoints,
    }


with open(dst, "w") as f:
    json.dump(payload, f)

PY


docker rm -f "$PROFILE_CONTAINER" \
    >/dev/null 2>&1 || true


HTTP="$(
    curl -sS \
        --unix-socket /var/run/docker.sock \
        -o "$REMOTE_ROOT/docker-create-response.json" \
        -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -X POST \
        --data-binary @"$PAYLOAD" \
        "http://localhost/containers/create?name=${PROFILE_CONTAINER}"
)"


if [[ "$HTTP" != "201" ]]; then

    echo "ERROR creating profiling container: HTTP $HTTP"

    cat "$REMOTE_ROOT/docker-create-response.json"

    exit 1
fi


echo "Profiling container created."

REMOTE_SCRIPT


###############################################################################
# 4. SWAP NAMES
###############################################################################

echo
echo "[4/8] Temporarily replacing model-serving..."

ssh "$REMOTE" "
    set -e

    echo 'Stopping normal model-serving...'

    docker stop '$ORIGINAL' >/dev/null


    echo 'Renaming original -> $BACKUP_CONTAINER'

    docker rename \
        '$ORIGINAL' \
        '$BACKUP_CONTAINER'


    echo 'Renaming profiler -> model-serving'

    docker rename \
        '$PROFILE_CONTAINER' \
        '$ORIGINAL'


    echo 'Starting instrumented model-serving...'

    docker start '$ORIGINAL' >/dev/null
"


echo
echo "Waiting for /health..."

HEALTHY=0

for i in $(seq 1 300); do

    if ssh "$REMOTE" \
        "curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1"
    then
        HEALTHY=1
        break
    fi


    RUNNING="$(
        ssh "$REMOTE" "
            docker inspect '$ORIGINAL' \
                --format '{{.State.Running}}' \
                2>/dev/null ||
            echo false
        "
    )"


    if [[ "$RUNNING" != "true" ]]; then

        echo
        echo "The profiling container has exited:"
        echo

        ssh "$REMOTE" \
            "docker logs '$ORIGINAL' 2>&1 || true"

        die "profiling container is not running"
    fi

    sleep 1
done


[[ "$HEALTHY" == "1" ]] ||
    die "Timed out waiting for /health"


echo
echo "Profiled server is ready:"

ssh "$REMOTE" "
    docker ps \
        --filter name='^/model-serving$' \
        --format '  {{.Names}} -> {{.Status}}'
"


#
# This is exactly the check that failed for us before.
#
CHECK_RUNNING="$(
    ssh "$REMOTE" "
        docker inspect '$ORIGINAL' \
            --format '{{.State.Running}}'
    "
)"

[[ "$CHECK_RUNNING" == "true" ]] ||
    die "The replay script would see model-serving as stopped"


###############################################################################
# 5. TELEMETRY + REPLAY
###############################################################################

echo
echo "[5/8] Starting GPU telemetry..."

ssh "$REMOTE" "
    nohup nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,memory.used,temperature.gpu \
        --format=csv,noheader,nounits \
        -lms 100 \
        > '$REMOTE_ROOT/gpu.csv' \
        2> '$REMOTE_ROOT/gpu-sampler.err' &

    echo \$! \
        > '$REMOTE_ROOT/gpu.pid'
"


echo
echo "Running complete replay..."
echo

"$REPLAY" replay "$LABEL" 2>&1 |
    tee "$LOCAL_ROOT/replay.stdout.log"


###############################################################################
# 6. FINALIZE NSIGHT
###############################################################################

echo
echo "[6/8] Finalizing Nsight..."

#
# Stop the nvidia-smi sampler.
#
ssh "$REMOTE" "
    set +e

    if test -f '$REMOTE_ROOT/gpu.pid'; then

        kill \
            \$(cat '$REMOTE_ROOT/gpu.pid') \
            2>/dev/null || true

        rm -f '$REMOTE_ROOT/gpu.pid'
    fi
"


#
# nsys is PID 1 in the container.
#
# SIGINT is equivalent to the usual Ctrl+C for nsys and allows it
# to close the capture and generate profile.nsys-rep.
#
echo "Sending SIGINT to nsys..."

ssh "$REMOTE" "
    docker kill \
        --signal=SIGINT \
        '$ORIGINAL' \
        >/dev/null
"


echo "Waiting for the profiling container to stop..."

STOPPED=0

for i in $(seq 1 180); do

    RUNNING="$(
        ssh "$REMOTE" "
            docker inspect '$ORIGINAL' \
                --format '{{.State.Running}}' \
                2>/dev/null ||
            echo false
        "
    )"

    if [[ "$RUNNING" == "false" ]]; then
        STOPPED=1
        break
    fi

    sleep 1
done


if [[ "$STOPPED" != "1" ]]; then

    echo "SIGINT did not stop the profiler; using docker stop..."

    ssh "$REMOTE" "
        docker stop \
            --time 30 \
            '$ORIGINAL' \
            >/dev/null || true
    "
fi


echo
echo "Waiting for profile.nsys-rep..."

PROFILE_READY=0

for i in $(seq 1 180); do

    if ssh "$REMOTE" \
        "test -s '$REMOTE_ROOT/profile.nsys-rep'"
    then
        PROFILE_READY=1
        break
    fi

    sleep 1
done


if [[ "$PROFILE_READY" != "1" ]]; then

    echo
    echo "=== docker logs profiler ==="

    ssh "$REMOTE" "
        docker logs '$ORIGINAL' \
            2>&1 || true
    "

    echo
    echo "=== remote files ==="

    ssh "$REMOTE" "
        ls -lah '$REMOTE_ROOT'
    "

    die "profile.nsys-rep was not generated"
fi


echo
echo "Profile generated:"

ssh "$REMOTE" "
    ls -lh \
        '$REMOTE_ROOT/profile.nsys-rep'
"


###############################################################################
###############################################################################
# 7. RESTORE NORMAL MODEL-SERVING
###############################################################################

echo
echo "[7/8] Restoring normal model-serving..."

ssh "$REMOTE" "
    set -e

    #
    # The current model-serving is the stopped profiling container.
    #
    docker rm -f '$ORIGINAL' \
        >/dev/null


    docker rename \
        '$BACKUP_CONTAINER' \
        '$ORIGINAL'


    docker start '$ORIGINAL' \
        >/dev/null


    until curl -fsS \
        http://127.0.0.1:8000/health \
        >/dev/null 2>&1
    do
        sleep 1
    done
"


echo "Original model-serving restored."

ssh "$REMOTE" "
    docker ps \
        --filter name='^/model-serving$' \
        --format '  {{.Names}} -> {{.Status}}'
"


###############################################################################
###############################################################################
# GENERATE NSIGHT REPORTS
###############################################################################

echo
echo "Generating Nsight reports..."

ssh "$REMOTE" "
    set +e

    sudo -n nsys stats \
        --force-export=true \
        --report cuda_api_sum \
        --report cuda_gpu_kern_sum \
        --report cuda_gpu_mem_time_sum \
        --report cuda_kern_exec_sum \
        '$REMOTE_ROOT/profile.nsys-rep' \
        > '$REMOTE_ROOT/nsys-stats.txt' \
        2>&1


    sudo -n nsys analyze \
        --rule cuda_api_sync \
        --rule gpu_gaps \
        '$REMOTE_ROOT/profile.nsys-rep' \
        > '$REMOTE_ROOT/nsys-analyze.txt' \
        2>&1 || true


    nvidia-smi -q \
        > '$REMOTE_ROOT/nvidia-smi-after.txt'


    docker inspect '$ORIGINAL' \
        > '$REMOTE_ROOT/model-serving-restored.json'


    sudo chown -R ubuntu:ubuntu \
        '$REMOTE_ROOT'
"


###############################################################################
# 8. DOWNLOAD TO THE LOCAL MACHINE
###############################################################################

echo
echo "[8/8] Downloading results..."

mkdir -p "$LOCAL_ROOT/nsys"


rsync -a \
    --info=stats1 \
    "$REMOTE:$REMOTE_ROOT/" \
    "$LOCAL_ROOT/nsys/"


[[ -s "$LOCAL_ROOT/nsys/profile.nsys-rep" ]] ||
    die "profile.nsys-rep did not reach the local machine"


###############################################################################
# DOWNLOAD REPLAY
###############################################################################

REMOTE_REPLAY="$(
    ssh "$REMOTE" "
        find '$REMOTE_RESULTS' \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name '*_${LABEL}' \
            -printf '%T@ %p\n' |
        sort -nr |
        head -n1 |
        cut -d' ' -f2-
    "
)"


if [[ -n "$REMOTE_REPLAY" ]]; then

    mkdir -p "$LOCAL_ROOT/replay"

    rsync -a \
        --info=stats1 \
        "$REMOTE:$REMOTE_REPLAY/" \
        "$LOCAL_ROOT/replay/"
else

    echo "WARNING: could not find the replay directory."
fi


###############################################################################
# VERIFY THAT NSIGHT ACTUALLY CAPTURED CUDA
###############################################################################

echo
echo "Checking that the report contains CUDA..."

if grep -q \
    'does not contain CUDA' \
    "$LOCAL_ROOT/nsys/nsys-stats.txt"
then

    echo
    echo "WARNING:"
    echo "Nsight generated a report, but it still does not contain CUDA."
    echo

    cat "$LOCAL_ROOT/nsys/nsys-stats.txt"

    exit 1
fi


echo "CUDA trace found."


###############################################################################
# DELETE REMOTE SCRATCH ONLY AFTER VERIFYING THE COPY
###############################################################################

ssh "$REMOTE" "
    rm -rf '$REMOTE_ROOT'

    if [[ -n '$REMOTE_REPLAY' ]]; then
        rm -rf '$REMOTE_REPLAY'
    fi
"


###############################################################################
# PACKAGE LOCALLY
###############################################################################

ARCHIVE="${LOCAL_ROOT}.tar.gz"

tar \
    -C "$(dirname "$LOCAL_ROOT")" \
    -czf "$ARCHIVE" \
    "$(basename "$LOCAL_ROOT")"


#
# Normal state has been restored. Cleanup is no longer needed.
#
trap - EXIT INT TERM


echo
echo "=========================================="
echo " PROFILING COMPLETED"
echo "=========================================="
echo
echo "Archive:"
echo
echo "  $ARCHIVE"
echo
echo "Size:"
ls -lh "$ARCHIVE"
echo
echo "Profile:"
ls -lh "$LOCAL_ROOT/nsys/profile.nsys-rep"
echo
echo "First CUDA statistics:"
echo

head -80 \
    "$LOCAL_ROOT/nsys/nsys-stats.txt" \
    || true

