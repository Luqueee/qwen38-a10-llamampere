#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE="${REMOTE:-lambdalabs}"
CONTAINER="${CONTAINER:-model-serving}"

REMOTE_RESULTS="/home/ubuntu/tryton-replay/results"
REPLAY="${REPLAY:-$REPO_ROOT/bench/harness/replay.sh}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOCAL_ROOT="${LOCAL_ROOT:-$REPO_ROOT/results/experiments/ncols1-ab/$RUN_ID}"

ARMS=(1 2 4)
RUNS_PER_ARM=3

mkdir -p "$LOCAL_ROOT"

LOG="$LOCAL_ROOT/ab.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " NCOLS1_MIN A/B"
echo "=========================================="
echo "Remote       : $REMOTE"
echo "Container    : $CONTAINER"
echo "Local output : $LOCAL_ROOT"
echo

if [[ ! -x "$REPLAY" ]]; then
    echo "ERROR: does not exist or is not executable:"
    echo "  $REPLAY"
    exit 1
fi

command -v ssh >/dev/null
command -v rsync >/dev/null


remote_env_value() {
    ssh "$REMOTE" "
        docker inspect '$CONTAINER' \
          --format '{{range .Config.Env}}{{println .}}{{end}}' |
        grep '^GGML_Q8_TURBO3_MMA_NCOLS1_MIN=' || true
    "
}


wait_health() {
    echo "Waiting for /health..."

    ssh "$REMOTE" '
        for i in $(seq 1 180); do
            if curl -fsS \
                http://127.0.0.1:8000/health \
                >/dev/null 2>&1
            then
                exit 0
            fi

            sleep 1
        done

        echo "ERROR: model-serving did not become healthy"
        docker logs --tail 200 model-serving >&2 || true
        exit 1
    '
}


restart_clean() {
    echo "Restarting $CONTAINER to clear state..."

    ssh "$REMOTE" "
        docker restart '$CONTAINER' >/dev/null
    "

    wait_health
}


set_ncols1() {
    local value="$1"

    echo
    echo "=========================================="
    echo " Setting NCOLS1_MIN=$value"
    echo "=========================================="

    #
    # Perform the entire recreation on the remote host.
    #
    # The current container is temporarily retained for rollback.
    # Use docker inspect -> Docker Engine API to preserve Config,
    # HostConfig, mounts, DeviceRequests/GPU, network mode, etc.
    #
    ssh "$REMOTE" bash -s -- "$CONTAINER" "$value" <<'REMOTE_SCRIPT'
set -Eeuo pipefail

CONTAINER="$1"
VALUE="$2"

OLD="${CONTAINER}-ncols1-old"
PAYLOAD="/tmp/${CONTAINER}-ncols1-create.json"
INSPECT="/tmp/${CONTAINER}-ncols1-inspect.json"

cleanup_files() {
    rm -f "$PAYLOAD" "$INSPECT"
}

trap cleanup_files EXIT

if ! docker inspect "$CONTAINER" >"$INSPECT"; then
    echo "ERROR: container $CONTAINER does not exist" >&2
    exit 1
fi

IMAGE="$(
    python3 - "$INSPECT" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)[0]

print(d["Config"]["Image"])
PY
)"

echo "Current image: $IMAGE"

#
# Generate a payload compatible with POST /containers/create.
#
python3 - "$INSPECT" "$PAYLOAD" "$VALUE" <<'PY'
import copy
import json
import sys

src, dst, value = sys.argv[1:4]

with open(src) as f:
    old = json.load(f)[0]

config = copy.deepcopy(old["Config"])
host = copy.deepcopy(old["HostConfig"])

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

key = "GGML_Q8_TURBO3_MMA_NCOLS1_MIN"
env = []

for item in config.get("Env") or []:
    if not item.startswith(key + "="):
        env.append(item)

env.append(f"{key}={value}")
config["Env"] = env

# ------------------------------------------------------------------
# Fields that belong to the container state and not to create.
# Config from docker inspect is almost directly reusable.
# ------------------------------------------------------------------

# Docker create accepts most of HostConfig as-is.
# These fields can contain calculated/runtime state, so it is better
# not to pass them through if they appear.
for key_to_drop in (
    "ContainerIDFile",
):
    if host.get(key_to_drop) in ("", None):
        host.pop(key_to_drop, None)

payload = {
    **config,
    "HostConfig": host,
}

# ------------------------------------------------------------------
# NetworkingConfig to preserve aliases/network configuration
# for user-defined networks where possible.
# ------------------------------------------------------------------

networks = old.get("NetworkSettings", {}).get("Networks", {})
endpoints = {}

for name, net in networks.items():
    ep = {}

    aliases = net.get("Aliases")
    if aliases:
        # Remove ephemeral aliases that are the current ID or name.
        old_id = old.get("Id", "")
        old_name = old.get("Name", "").lstrip("/")

        clean_aliases = [
            a for a in aliases
            if a not in {old_id, old_id[:12], old_name}
        ]

        if clean_aliases:
            ep["Aliases"] = clean_aliases

    ipam = net.get("IPAMConfig")
    if ipam:
        ep["IPAMConfig"] = ipam

    links = net.get("Links")
    if links:
        ep["Links"] = links

    endpoints[name] = ep

if endpoints:
    payload["NetworkingConfig"] = {
        "EndpointsConfig": endpoints,
    }

with open(dst, "w") as f:
    json.dump(payload, f)
PY

#
# Remove old rollback container if one was left from an interrupted
# run.
#
if docker inspect "$OLD" >/dev/null 2>&1; then
    echo "Removing old rollback container $OLD"
    docker rm -f "$OLD" >/dev/null
fi

echo "Stopping current container..."
docker stop "$CONTAINER" >/dev/null

echo "Renaming current container to $OLD..."
docker rename "$CONTAINER" "$OLD"

rollback() {
    echo "ROLLBACK: restoring previous container..." >&2

    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

    if docker inspect "$OLD" >/dev/null 2>&1; then
        docker rename "$OLD" "$CONTAINER"
        docker start "$CONTAINER" >/dev/null || true
    fi
}

trap 'rollback; cleanup_files' ERR

#
# Create the container with the SAME name and all previous
# configuration except NCOLS1_MIN.
#
HTTP_CODE="$(
    curl -sS \
        --unix-socket /var/run/docker.sock \
        -o /tmp/docker-create-response.json \
        -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -X POST \
        --data-binary @"$PAYLOAD" \
        "http://localhost/containers/create?name=${CONTAINER}"
)"

if [[ "$HTTP_CODE" != "201" ]]; then
    echo "ERROR creating container: HTTP $HTTP_CODE" >&2
    cat /tmp/docker-create-response.json >&2 || true
    false
fi

echo "Starting new container..."
docker start "$CONTAINER" >/dev/null

#
# Wait for health before destroying the rollback container.
#
echo "Waiting for health..."

healthy=0

for i in $(seq 1 180); do
    if curl -fsS \
        http://127.0.0.1:8000/health \
        >/dev/null 2>&1
    then
        healthy=1
        break
    fi

    sleep 1
done

if [[ "$healthy" != "1" ]]; then
    echo "ERROR: the new container did not start correctly" >&2
    docker logs --tail 200 "$CONTAINER" >&2 || true
    false
fi

ACTUAL="$(
    docker inspect "$CONTAINER" \
      --format '{{range .Config.Env}}{{println .}}{{end}}' |
    grep '^GGML_Q8_TURBO3_MMA_NCOLS1_MIN='
)"

echo "Active environment: $ACTUAL"

#
# The new container is confirmed to work.
#
echo "Removing rollback container..."
docker rm -f "$OLD" >/dev/null

trap - ERR

echo "NCOLS1_MIN=$VALUE applied successfully."
REMOTE_SCRIPT

    echo
    echo "Remote value:"
    remote_env_value

    #
    # Also save the resulting container inspect locally.
    #
    ssh "$REMOTE" "docker inspect '$CONTAINER'" \
        > "$LOCAL_ROOT/container-ncols1-${value}.json"
}


find_remote_result() {
    local label="$1"

    ssh "$REMOTE" "
        find '$REMOTE_RESULTS' \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name '*_${label}' \
            -printf '%T@ %p\n' |
        sort -nr |
        head -n1 |
        cut -d' ' -f2-
    "
}


copy_result_local() {
    local label="$1"
    local remote_dir="$2"

    local local_dir="$LOCAL_ROOT/$label"

    mkdir -p "$local_dir"

    echo "Copying result:"
    echo "  Lambda: $remote_dir"
    echo "  Local : $local_dir"

    rsync \
        -a \
        --info=stats1 \
        "${REMOTE}:${remote_dir}/" \
        "${local_dir}/"

    #
    # Minimal check before deleting the remote copy.
    #
    if [[ -z "$(find "$local_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: local result is empty; remote copy will NOT be deleted."
        return 1
    fi

    echo "Local copy completed."

    #
    # Lambda is only used as scratch space.
    #
    ssh "$REMOTE" "rm -rf -- '$remote_dir'"

    echo "Temporary remote result deleted."
}

run_replay() {
    local value="$1"
    local iteration="$2"

    local label="ncols1-min${value}-${iteration}"
    local stdout_log="$LOCAL_ROOT/${label}.stdout.log"

    echo
    echo "------------------------------------------"
    echo " Replay: $label"
    echo "------------------------------------------"

    #
    # Clean:
    #  - adaptive hot tail
    #  - prompt cache
    #
    restart_clean

    #
    # The replay is started from the local machine.
    #
    "$REPLAY" replay "$label" 2>&1 |
        tee "$stdout_log"

    remote_dir="$(find_remote_result "$label")"

    if [[ -z "$remote_dir" ]]; then
        echo "ERROR: cannot find remote result for $label"
        exit 1
    fi

    copy_result_local "$label" "$remote_dir"
}


# Save the initial container baseline locally.
#
echo "Saving initial docker inspect..."
ssh "$REMOTE" "docker inspect '$CONTAINER'" \
    > "$LOCAL_ROOT/container-before.json"

echo "Initial environment:"
remote_env_value
echo


#
# A/B
#
for value in "${ARMS[@]}"; do
    set_ncols1 "$value"

    for iteration in $(seq 1 "$RUNS_PER_ARM"); do
        run_replay "$value" "$iteration"
    done
done


# Final state.
ssh "$REMOTE" "docker inspect '$CONTAINER'" \
    > "$LOCAL_ROOT/container-after.json"


echo
echo "=========================================="
echo " A/B COMPLETED"
echo "=========================================="
echo
echo "All persistent results are stored in:"
echo
echo "  $LOCAL_ROOT"
echo
echo "Structure:"
find "$LOCAL_ROOT" -maxdepth 2 -type f | sort
