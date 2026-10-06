#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="lambdalabs"
CONTAINER="model-serving"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPLAY_SCRIPT="$REPO_ROOT/bench/harness/replay.sh"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LABEL="nsys-${RUN_ID}"

LOCAL_ROOT="$REPO_ROOT/results/profiling/nsys-qwen38/$RUN_ID"
REMOTE_ROOT="/home/ubuntu/tryton-replay/nsys/$RUN_ID"
REMOTE_RESULTS="/home/ubuntu/tryton-replay/results"

mkdir -p "$LOCAL_ROOT"

LOG="$LOCAL_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1

die() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}

echo "=========================================="
echo " Qwen3.8 / A10 / Nsight Systems"
echo "=========================================="
echo
echo "Run        : $RUN_ID"
echo "Label      : $LABEL"
echo "Remote     : $REMOTE"
echo "Container  : $CONTAINER"
echo "Local      : $LOCAL_ROOT"
echo "Remote tmp : $REMOTE_ROOT"
echo

###############################################################################
# 1. LOCAL PREFLIGHT
###############################################################################

echo "[1/8] Local preflight..."

command -v ssh >/dev/null || die "ssh not found"
command -v rsync >/dev/null || die "rsync not found"
command -v tar >/dev/null || die "tar not found"

[[ -x "$REPLAY_SCRIPT" ]] ||
    die "Does not exist or is not executable: $REPLAY_SCRIPT"

ssh "$REMOTE" true ||
    die "Cannot connect to $REMOTE over SSH"

echo "OK"


###############################################################################
# 2. REMOTE PREFLIGHT
###############################################################################

echo
echo "[2/8] Preflight Lambda..."

ssh "$REMOTE" "
    set -e

    command -v docker
    command -v curl
    command -v nsys
    command -v nvidia-smi

    echo
    nsys --version

    echo
    echo 'Container:'
    docker inspect '$CONTAINER' \
        --format 'name={{.Name}} user={{printf \"%q\" .Config.User}} image={{.Config.Image}}'

    echo
    echo 'NCOLS1_MIN:'
    docker inspect '$CONTAINER' \
        --format '{{range .Config.Env}}{{println .}}{{end}}' |
        grep '^GGML_Q8_TURBO3_MMA_NCOLS1_MIN=' ||
        echo 'unset (default=2)'
"

NCOLS="$(
    ssh "$REMOTE" "
        docker inspect '$CONTAINER' \
            --format '{{range .Config.Env}}{{println .}}{{end}}' |
        sed -n 's/^GGML_Q8_TURBO3_MMA_NCOLS1_MIN=//p'
    "
)"

if [[ -n "$NCOLS" && "$NCOLS" != "2" ]]; then
    die "NCOLS1_MIN=$NCOLS; baseline 2 is required for profiling"
fi

echo
echo "Remote preflight OK."


###############################################################################
# 3. PREPARE SCRATCH
###############################################################################

echo
echo "[3/8] Preparing remote directory..."

ssh "$REMOTE" "
    set -e

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
# 4. NSIGHT TARGET
###############################################################################

echo
echo "[4/8] Creating Nsight target..."

ssh "$REMOTE" "cat > '$REMOTE_ROOT/profile-target.sh'" <<'REMOTE_EOF'
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
REMOTE_EOF

ssh "$REMOTE" "
    chmod +x '$REMOTE_ROOT/profile-target.sh'
"


###############################################################################
###############################################################################
# 5. START NSIGHT
###############################################################################

echo
echo "[5/8] Starting Nsight Systems..."

ssh "$REMOTE" "
    set -e

    cd '$REMOTE_ROOT'

    nohup sudo -n nsys profile \
        --trace=cuda-sw \
        --cuda-trace-scope=system-wide \
        --sample=none \
        --cpuctxsw=none \
        --output='$REMOTE_ROOT/profile' \
        '$REMOTE_ROOT/profile-target.sh' \
        '$CONTAINER' \
        '$REMOTE_ROOT' \
        > '$REMOTE_ROOT/nsys-driver.log' \
        2>&1 &

    echo \$! > '$REMOTE_ROOT/nsys.pid'
"

echo "Waiting for READY..."

READY=0

for i in $(seq 1 240); do

    if ssh "$REMOTE" "test -f '$REMOTE_ROOT/READY'"; then
        READY=1
        break
    fi

    RUNNING="$(
        ssh "$REMOTE" "
            if test -f '$REMOTE_ROOT/nsys.pid' &&
               kill -0 \$(cat '$REMOTE_ROOT/nsys.pid') 2>/dev/null
            then
                echo yes
            else
                echo no
            fi
        "
    )"

    if [[ "$RUNNING" == "no" ]]; then
        echo
        echo "Nsight exited before READY:"
        ssh "$REMOTE" "
            cat '$REMOTE_ROOT/nsys-driver.log' 2>/dev/null || true
        "
        die "Nsight could not start"
    fi

    sleep 1
done

[[ "$READY" == "1" ]] ||
    die "Timed out waiting for READY"

echo "Nsight is active."
echo "model-serving started after capture began."


###############################################################################
###############################################################################
# 6. TELEMETRY + REPLAY
###############################################################################

echo
echo "[6/8] Running replay..."

ssh "$REMOTE" "
    nohup nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.current.sm,clocks.current.memory,memory.used,temperature.gpu \
        --format=csv,noheader,nounits \
        -lms 100 \
        > '$REMOTE_ROOT/gpu.csv' \
        2> '$REMOTE_ROOT/gpu-sampler.err' &

    echo \$! > '$REMOTE_ROOT/gpu.pid'
"

"$REPLAY_SCRIPT" replay "$LABEL" 2>&1 |
    tee "$LOCAL_ROOT/replay.stdout.log"


###############################################################################
# 7. FINALIZE NSIGHT
###############################################################################

echo
echo "[7/8] Finalizing profiling..."

ssh "$REMOTE" "
    set +e

    if test -f '$REMOTE_ROOT/gpu.pid'; then
        kill \$(cat '$REMOTE_ROOT/gpu.pid') 2>/dev/null
        rm -f '$REMOTE_ROOT/gpu.pid'
    fi

    touch '$REMOTE_ROOT/STOP'
"

echo "Waiting for profile.nsys-rep..."

FINISHED=0

for i in $(seq 1 300); do

    RUNNING="$(
        ssh "$REMOTE" "
            if test -f '$REMOTE_ROOT/nsys.pid' &&
               kill -0 \$(cat '$REMOTE_ROOT/nsys.pid') 2>/dev/null
            then
                echo yes
            else
                echo no
            fi
        "
    )"

    if [[ "$RUNNING" == "no" ]]; then
        FINISHED=1
        break
    fi

    sleep 1
done

[[ "$FINISHED" == "1" ]] ||
    die "Nsight did not finish"

if ! ssh "$REMOTE" "test -s '$REMOTE_ROOT/profile.nsys-rep'"; then

    echo
    echo "=== nsys-driver.log ==="

    ssh "$REMOTE" "
        cat '$REMOTE_ROOT/nsys-driver.log' 2>/dev/null || true
    "

    echo
    echo "=== files ==="

    ssh "$REMOTE" "
        ls -lah '$REMOTE_ROOT'
    "

    die "profile.nsys-rep was not generated"
fi

echo
echo "profile.nsys-rep generated:"
ssh "$REMOTE" "
    ls -lh '$REMOTE_ROOT/profile.nsys-rep'
"


###############################################################################
# STATISTICS
###############################################################################

echo
echo "Generating statistics..."

ssh "$REMOTE" "
    set +e

    sudo -n nsys stats \
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
        2>&1

    docker logs '$CONTAINER' \
        > '$REMOTE_ROOT/model-serving.log' \
        2>&1

    nvidia-smi -q \
        > '$REMOTE_ROOT/nvidia-smi-after.txt'

    sudo chown -R ubuntu:ubuntu '$REMOTE_ROOT'
"


###############################################################################
# 8. DOWNLOAD TO THE LOCAL MACHINE
###############################################################################

echo
echo "[8/8] Downloading to the local machine..."

mkdir -p "$LOCAL_ROOT/nsys"

rsync -a \
    --info=stats1 \
    "$REMOTE:$REMOTE_ROOT/" \
    "$LOCAL_ROOT/nsys/"

[[ -s "$LOCAL_ROOT/nsys/profile.nsys-rep" ]] ||
    die "The local copy does not contain profile.nsys-rep"


###############################################################################
###############################################################################
# COPY REPLAY
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
    echo "WARNING: could not find the remote replay directory."
fi


###############################################################################
###############################################################################
# ONLY NOW REMOVE REMOTE SCRATCH
###############################################################################

echo
echo "Local copy verified."

ssh "$REMOTE" "
    rm -rf '$REMOTE_ROOT'

    if [[ -n '$REMOTE_REPLAY' ]]; then
        rm -rf '$REMOTE_REPLAY'
    fi
"


###############################################################################
###############################################################################
# PACKAGE
###############################################################################

ARCHIVE="${LOCAL_ROOT}.tar.gz"

tar \
    -C "$(dirname "$LOCAL_ROOT")" \
    -czf "$ARCHIVE" \
    "$(basename "$LOCAL_ROOT")"


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
echo "To verify:"
echo
echo "  tar -tzf '$ARCHIVE' | head -50"
