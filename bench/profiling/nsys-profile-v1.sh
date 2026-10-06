#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="${REMOTE:-lambdalabs}"
CONTAINER="${CONTAINER:-model-serving}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPLAY="$REPO_ROOT/bench/harness/replay.sh"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LABEL="nsys-request15-${RUN_ID}"
SESSION="tryton${RUN_ID//-/}"

LOCAL_ROOT="$REPO_ROOT/results/profiling/nsys-qwen38/$RUN_ID"
REMOTE_ROOT="/home/ubuntu/tryton-replay/nsys/$RUN_ID"
REMOTE_RESULTS="/home/ubuntu/tryton-replay/results"

mkdir -p "$LOCAL_ROOT"

LOG="$LOCAL_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " Qwen3.8 / A10 / Nsight Systems"
echo "=========================================="
echo "Run       : $RUN_ID"
echo "Label     : $LABEL"
echo "Session   : $SESSION"
echo "Local     : $LOCAL_ROOT"
echo "Remote    : $REMOTE_ROOT"
echo


###############################################################################
# 1. Check baseline
###############################################################################

echo "[1/9] Checking NCOLS1_MIN..."

CURRENT_NCOLS="$(
    ssh "$REMOTE" "
        docker inspect '$CONTAINER' \
          --format '{{range .Config.Env}}{{println .}}{{end}}' |
        sed -n 's/^GGML_Q8_TURBO3_MMA_NCOLS1_MIN=//p'
    "
)"

if [[ -z "$CURRENT_NCOLS" ]]; then
    echo "NCOLS1_MIN not set -> default = 2"
elif [[ "$CURRENT_NCOLS" != "2" ]]; then
    echo "ERROR: current NCOLS1_MIN = $CURRENT_NCOLS"
    echo "Profiling must use baseline 2."
    exit 1
else
    echo "NCOLS1_MIN = 2 OK"
fi


###############################################################################
# 2. Install Nsight Systems CLI if needed
###############################################################################

echo
echo "[2/9] Checking Nsight Systems..."

ssh "$REMOTE" 'bash -s' <<'REMOTE'
set -Eeuo pipefail

if command -v nsys >/dev/null 2>&1; then
    nsys --version
    exit 0
fi

echo "Nsight Systems is not installed. Installing CLI..."

. /etc/os-release

if [[ "$ID" != "ubuntu" ]]; then
    echo "ERROR: automatic installation is prepared for Ubuntu; found: $ID"
    exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    ca-certificates

REL="$(echo "$VERSION_ID" | tr -d .)"
ARCH="$(dpkg --print-architecture)"

wget -qO- \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub |
sudo gpg --dearmor \
  --yes \
  -o /usr/share/keyrings/nvidia-devtools-keyring.gpg

echo \
"deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu${REL}/${ARCH}/ /" |
sudo tee /etc/apt/sources.list.d/nvidia-devtools.list >/dev/null

sudo apt-get update
sudo apt-get install -y nsight-systems-cli

nsys --version
REMOTE


###############################################################################
# 3. Prepare remote directory
###############################################################################

echo
echo "[3/9] Preparing remote profiling..."

ssh "$REMOTE" "
    rm -rf '$REMOTE_ROOT'
    mkdir -p '$REMOTE_ROOT'

    docker inspect '$CONTAINER' \
        > '$REMOTE_ROOT/container-inspect.json'

    nvidia-smi -q \
        > '$REMOTE_ROOT/nvidia-smi-before.txt'

    nsys --version \
        > '$REMOTE_ROOT/nsys-version.txt' 2>&1
"


###############################################################################
###############################################################################
# Cleanup on error
###############################################################################

SESSION_STARTED=0
SAMPLER_STARTED=0

cleanup() {
    set +e

    echo
    echo "Cleanup..."

    if [[ "$SAMPLER_STARTED" == "1" ]]; then
        ssh "$REMOTE" "
            if [[ -f '$REMOTE_ROOT/gpu.pid' ]]; then
                kill \$(cat '$REMOTE_ROOT/gpu.pid') 2>/dev/null || true
                rm -f '$REMOTE_ROOT/gpu.pid'
            fi
        " || true
    fi

    if [[ "$SESSION_STARTED" == "1" ]]; then
        ssh "$REMOTE" "
            sudo nsys stop \
                --session='$SESSION' \
                --keep=45 \
                >/dev/null 2>&1 || true
        " || true
    fi
}

trap cleanup ERR INT TERM


###############################################################################
# 4. Start Nsight BEFORE the CUDA process
###############################################################################

echo
echo "[4/9] Starting Nsight session..."

ssh "$REMOTE" bash -s -- "$SESSION" "$REMOTE_ROOT" <<'REMOTE'
set -Eeuo pipefail

SESSION="$1"
REMOTE_ROOT="$2"

GPU_ARGS=()

if sudo nsys profile \
    --gpu-metrics-devices=help \
    2>&1 |
    grep -qE '^[[:space:]]*0:'
then
    echo "GPU Metrics are supported on GPU 0"
    GPU_ARGS=(
        --gpu-metrics-devices=0
        --gpu-metrics-frequency=1000
    )
else
    echo "GPU Metrics are unavailable; continuing with CUDA trace."
fi

sudo nsys start \
    --session-new="$SESSION" \
    --stop-on-exit=false \
    --trace=cuda \
    --cuda-trace-scope=system-wide \
    --cuda-graph-trace=graph \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$REMOTE_ROOT/profile" \
    "${GPU_ARGS[@]}"

sudo nsys sessions list
REMOTE

SESSION_STARTED=1


###############################################################################
# 5. Restart model-serving AFTER starting Nsight
###############################################################################

echo
echo "[5/9] Restarting model-serving under profiling..."

ssh "$REMOTE" "
    docker restart '$CONTAINER' >/dev/null

    until curl -fsS \
        http://127.0.0.1:8000/health \
        >/dev/null 2>&1
    do
        sleep 1
    done

    echo 'model-serving healthy'
"


###############################################################################
###############################################################################
# 6. Additional NVIDIA telemetry
###############################################################################

echo
echo "[6/9] Starting GPU telemetry at 100 ms..."

ssh "$REMOTE" "
    nohup nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,memory.used,temperature.gpu \
      --format=csv,noheader,nounits \
      -lms 100 \
      > '$REMOTE_ROOT/gpu.csv' \
      2> '$REMOTE_ROOT/gpu-sampler.err' &

    echo \$! > '$REMOTE_ROOT/gpu.pid'
"

SAMPLER_STARTED=1


###############################################################################
# 7. COMPLETE replay
###############################################################################

echo
echo "[7/9] Running complete replay..."
echo

"$REPLAY" replay "$LABEL" \
    2>&1 |
tee "$LOCAL_ROOT/replay.stdout.log"


###############################################################################
# 8. Stop telemetry + retain ONLY the last 45 s of Nsight
###############################################################################

echo
echo "[8/9] Closing profiling..."

ssh "$REMOTE" "
    if [[ -f '$REMOTE_ROOT/gpu.pid' ]]; then
        kill \$(cat '$REMOTE_ROOT/gpu.pid') 2>/dev/null || true
        rm -f '$REMOTE_ROOT/gpu.pid'
    fi
"

SAMPLER_STARTED=0

ssh "$REMOTE" "
    sudo nsys stop \
        --session='$SESSION' \
        --keep=45
"

SESSION_STARTED=0


###############################################################################
# Generate reports
###############################################################################

echo
echo "Generating Nsight statistics..."

ssh "$REMOTE" "
    REPORT='$REMOTE_ROOT/profile.nsys-rep'

    if [[ ! -f \"\$REPORT\" ]]; then
        echo 'ERROR: profile.nsys-rep was not generated'
        ls -lah '$REMOTE_ROOT'
        exit 1
    fi

    sudo nsys stats \
        --report cuda_api_sum \
        --report cuda_gpu_kern_sum \
        --report cuda_gpu_mem_time_sum \
        \"\$REPORT\" \
        > '$REMOTE_ROOT/nsys-stats.txt' \
        2>&1 || true

    docker logs '$CONTAINER' \
        > '$REMOTE_ROOT/model-serving.log' \
        2>&1

    nvidia-smi -q \
        > '$REMOTE_ROOT/nvidia-smi-after.txt'

    sudo chown -R ubuntu:ubuntu '$REMOTE_ROOT'
"


###############################################################################
# 9. Copy EVERYTHING to the local machine
###############################################################################

echo
echo "[9/9] Copying profiler locally..."

rsync -a \
    --info=stats1 \
    "$REMOTE:$REMOTE_ROOT/" \
    "$LOCAL_ROOT/nsys/"


###############################################################################
# Also copy the replay result
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

    ssh "$REMOTE" "rm -rf -- '$REMOTE_REPLAY'"
else
    echo "WARNING: replay directory for $LABEL was not found"
fi


###############################################################################
# Remove remote scratch
###############################################################################

echo
echo "Removing remote profiling scratch..."

ssh "$REMOTE" "rm -rf '$REMOTE_ROOT'"


###############################################################################
# Package locally
###############################################################################

ARCHIVE="${LOCAL_ROOT}.tar.gz"

tar -C "$(dirname "$LOCAL_ROOT")" \
    -czf "$ARCHIVE" \
    "$(basename "$LOCAL_ROOT")"


echo
echo "=========================================="
echo " PROFILING COMPLETED"
echo "=========================================="
echo
echo "Results:"
echo "  $LOCAL_ROOT"
echo
echo "File to analyze:"
echo "  $ARCHIVE"
echo
echo "Main contents:"
find "$LOCAL_ROOT" -maxdepth 2 -type f -printf '  %p\n' | sort
