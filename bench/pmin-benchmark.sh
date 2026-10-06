#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_CONTAINER="${SOURCE_CONTAINER:-model-serving}"
TEST_CONTAINER="${TEST_CONTAINER:-model-serving-pmin-bench}"

HOST_PORT="${HOST_PORT:-8000}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
RESULTS="${RESULTS:-pmin-results.csv}"

PMINS=(
    "0.00"
    "0.40"
    "0.60"
    "0.75"
)

PROMPT='Write a comprehensive technical analysis of the design of an AI agent
system that uses external tools, transactional databases, retries, pagination,
concurrency control, context management, error recovery, and result validation.

Divide the analysis into at least 20 developed sections. Explain decisions,
alternatives, potential problems, and their implications. Do not write a short
summary or end prematurely. Continue developing the analysis in detail until
all sections are complete.'

# ============================================================
# HELPERS
# ============================================================

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: required command '$1' is missing"
        exit 1
    }
}

need_cmd docker
need_cmd curl
need_cmd jq
need_cmd sed
need_cmd awk

# ============================================================
# CHECK THE ORIGINAL CONTAINER
# ============================================================

if ! docker inspect "$SOURCE_CONTAINER" >/dev/null 2>&1; then
    echo "ERROR: container '$SOURCE_CONTAINER' does not exist"
    echo
    echo "Available containers:"
    docker ps -a --format '  {{.Names}}'
    exit 1
fi

IMAGE="$(
    docker inspect \
        -f '{{.Config.Image}}' \
        "$SOURCE_CONTAINER"
)"

NETWORK="$(
    docker inspect \
        -f '{{.HostConfig.NetworkMode}}' \
        "$SOURCE_CONTAINER"
)"

WAS_RUNNING="$(
    docker inspect \
        -f '{{.State.Running}}' \
        "$SOURCE_CONTAINER"
)"

# ============================================================
# COPY THE EXACT CONTAINER COMMAND
# ============================================================

mapfile -t BASE_CMD < <(
    docker inspect "$SOURCE_CONTAINER" |
        jq -r '.[0].Config.Cmd[]'
)

# ============================================================
# EXTRACT API KEY, ALIAS, AND INTERNAL PORT
# ============================================================

API_KEY=""
MODEL_ID="qwen3.8-27b"
INTERNAL_PORT="8000"

for i in "${!BASE_CMD[@]}"; do
    case "${BASE_CMD[$i]}" in

        --api-key)
            API_KEY="${BASE_CMD[$((i + 1))]:-}"
            ;;

        --alias)
            MODEL_ID="${BASE_CMD[$((i + 1))]:-qwen3.8-27b}"
            ;;

        --port)
            INTERNAL_PORT="${BASE_CMD[$((i + 1))]:-8000}"
            ;;

    esac
done

if [[ -z "$API_KEY" ]]; then
    echo "ERROR: --api-key was not found in the container command."
    exit 1
fi

AUTH_HEADER="Authorization: Bearer $API_KEY"

echo
echo "=================================================="
echo " llama.cpp p-min benchmark"
echo "=================================================="
echo
echo "Base container : $SOURCE_CONTAINER"
echo "Image          : $IMAGE"
echo "Network        : $NETWORK"
echo "Model          : $MODEL_ID"
echo "API key        : detected"
echo "Internal port  : $INTERNAL_PORT"
echo "Host port      : $HOST_PORT"
echo "Runs per p-min : $RUNS"
echo "Max tokens     : $MAX_TOKENS"
echo

# ============================================================
# CLEANUP
# ============================================================

cleanup() {
    echo
    echo "Cleaning up benchmark..."

    docker rm -f "$TEST_CONTAINER" \
        >/dev/null 2>&1 || true

    if [[ "$WAS_RUNNING" == "true" ]]; then
        if ! docker inspect "$SOURCE_CONTAINER" \
            -f '{{.State.Running}}' 2>/dev/null |
            grep -q true; then

            echo "Restoring $SOURCE_CONTAINER..."

            docker start "$SOURCE_CONTAINER" \
                >/dev/null 2>&1 || true
        fi
    fi
}

trap cleanup EXIT

# ============================================================
# STOP THE ORIGINAL SERVER
# ============================================================

if [[ "$WAS_RUNNING" == "true" ]]; then
    echo "Stopping $SOURCE_CONTAINER..."

    docker stop "$SOURCE_CONTAINER" \
        >/dev/null
fi

# ============================================================
# CSV
# ============================================================

echo \
"p_min,run,gen_tokens,tok_s,acceptance,mean_len,e2e_s" \
> "$RESULTS"

# ============================================================
# SET P-MIN
# ============================================================

set_pmin() {

    local value="$1"

    CMD=("${BASE_CMD[@]}")

    local found=0

    for i in "${!CMD[@]}"; do

        if [[ "${CMD[$i]}" == "--spec-draft-p-min" ]]; then

            CMD[$((i + 1))]="$value"

            found=1
            break
        fi

    done

    if [[ "$found" -eq 0 ]]; then

        CMD+=(
            "--spec-draft-p-min"
            "$value"
        )

    fi
}

# ============================================================
# WAIT FOR THE SERVER
# ============================================================

wait_server() {

    echo -n "Waiting for llama-server"

    for _ in $(seq 1 180); do

        if curl -sf \
            -H "$AUTH_HEADER" \
            "http://127.0.0.1:${HOST_PORT}/health" \
            >/dev/null 2>&1; then

            echo " OK"
            return 0
        fi

        echo -n "."
        sleep 1

    done

    echo
    echo "ERROR: timed out waiting for llama-server"
    echo
    docker logs "$TEST_CONTAINER" 2>&1 |
        tail -n 100

    return 1
}

# ============================================================
# GENERATE THE REQUEST BODY
# ============================================================

make_request() {

    jq -n \
        --arg model "$MODEL_ID" \
        --arg prompt "$PROMPT" \
        --argjson max_tokens "$MAX_TOKENS" \
        '{
            model: $model,

            messages: [
                {
                    role: "user",
                    content: $prompt
                }
            ],

            temperature: 0,
            seed: 1234,

            max_tokens: $max_tokens,

            stream: false,
            cache_prompt: false
        }'
}

# ============================================================
# BENCHMARK
# ============================================================
for PMIN in "${PMINS[@]}"; do

    echo
    echo "=================================================="
    echo " p-min = $PMIN"
    echo "=================================================="

    set_pmin "$PMIN"

    docker rm -f "$TEST_CONTAINER" \
        >/dev/null 2>&1 || true

    DOCKER_ARGS=(
        run
        -d
        --rm
        --name "$TEST_CONTAINER"
        --gpus all
        --volumes-from "$SOURCE_CONTAINER"
    )

    # ========================================================
    # NETWORK
    # ========================================================

    if [[ "$NETWORK" == "host" ]]; then

        DOCKER_ARGS+=(
            --network host
        )

    else

        DOCKER_ARGS+=(
            --network "$NETWORK"
            -p "${HOST_PORT}:${INTERNAL_PORT}"
        )

    fi

    # ========================================================
    # START THE CONTAINER
    # ========================================================
    docker "${DOCKER_ARGS[@]}" \
        "$IMAGE" \
        "${CMD[@]}" \
        >/dev/null

    wait_server

    echo "Model: $MODEL_ID"

    # ========================================================
    # WARM-UP
    # ========================================================

    echo "Warm-up..."

    jq -n \
        --arg model "$MODEL_ID" \
        '{
            model: $model,

            messages: [
                {
                    role: "user",
                    content: "Briefly explain speculative decoding."
                }
            ],

            temperature: 0,
            seed: 1234,
            max_tokens: 64,
            stream: false,
            cache_prompt: false
        }' |
        curl -fsS \
            -H "$AUTH_HEADER" \
            -H 'Content-Type: application/json' \
            -d @- \
            "http://127.0.0.1:${HOST_PORT}/v1/chat/completions" \
            >/dev/null

    sleep 1

    # ========================================================
    # RUNS
    # ========================================================

    for RUN in $(seq 1 "$RUNS"); do

        echo
        echo "----------------------------------------------"
        echo "p-min=$PMIN | run=$RUN/$RUNS"
        echo "----------------------------------------------"

        BODY="$(make_request)"

        # Record the current number of log lines to
        # simplify debugging if needed.
        BEFORE_LINES="$(
            docker logs "$TEST_CONTAINER" 2>&1 |
                wc -l
        )"

        E2E="$(
            curl -fsS \
                -o /dev/null \
                -w '%{time_total}' \
                -H "$AUTH_HEADER" \
                -H 'Content-Type: application/json' \
                -d "$BODY" \
                "http://127.0.0.1:${HOST_PORT}/v1/chat/completions"
        )"

        # Allow llama.cpp a little time to print
        # print_timing.
        sleep 1

        LOGS="$(
            docker logs "$TEST_CONTAINER" 2>&1 |
                tail -n "+$((BEFORE_LINES + 1))"
        )"

        # ====================================================
        # EXTRACT EVAL TIME
        # ====================================================
        EVAL_LINE="$(
            printf '%s\n' "$LOGS" |
                grep -E '\|[[:space:]]+eval time =' |
                tail -n 1 || true
        )"

        # ====================================================
        # EXTRACT MTP METRICS
        # ====================================================

        DRAFT_LINE="$(
            printf '%s\n' "$LOGS" |
                grep 'draft acceptance =' |
                tail -n 1 || true
        )"

        if [[ -z "$EVAL_LINE" ]]; then

            echo "ERROR: 'eval time' was not found in the logs."
            echo
            printf '%s\n' "$LOGS" | tail -n 100
            exit 1

        fi

        if [[ -z "$DRAFT_LINE" ]]; then

            echo "ERROR: 'draft acceptance' was not found in the logs."
            echo
            printf '%s\n' "$LOGS" | tail -n 100
            exit 1

        fi

        # ====================================================
        # PARSE METRICS
        # ====================================================

        GEN_TOKENS="$(
            printf '%s\n' "$EVAL_LINE" |
                sed -E \
                    's|.* /[[:space:]]*([0-9]+) tokens.*|\1|'
        )"

        TOK_S="$(
            printf '%s\n' "$EVAL_LINE" |
                sed -E \
                    's|.*,[[:space:]]*([0-9.]+) tokens per second.*|\1|'
        )"

        ACCEPTANCE="$(
            printf '%s\n' "$DRAFT_LINE" |
                sed -E \
                    's|.*draft acceptance = ([0-9.]+).*|\1|'
        )"

        MEAN_LEN="$(
            printf '%s\n' "$DRAFT_LINE" |
                sed -E \
                    's|.*mean len =[[:space:]]*([0-9.]+).*|\1|'
        )"

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================
        echo "tokens       : $GEN_TOKENS"
        echo "tok/s        : $TOK_S"
        echo "acceptance   : $ACCEPTANCE"
        echo "mean len     : $MEAN_LEN"
        echo "E2E          : ${E2E}s"

        # ====================================================
        # CSV
        # ====================================================

        echo \
"$PMIN,$RUN,$GEN_TOKENS,$TOK_S,$ACCEPTANCE,$MEAN_LEN,$E2E" \
        >> "$RESULTS"

    done

    echo
    echo "Stopping p-min=$PMIN server..."
    docker rm -f "$TEST_CONTAINER" \
        >/dev/null

    sleep 2

done

# ============================================================
# SUMMARY
# ============================================================
echo
echo " RESULTS"
echo

printf \
"%-8s %-12s %-14s %-12s %-12s\n" \
"p-min" \
"tok/s avg" \
"acceptance" \
"mean len" \
"E2E avg"

awk -F, '
NR > 1 {

    p=$1

    n[p]++

    tps[p]+=$4
    acc[p]+=$5
    len[p]+=$6
    e2e[p]+=$7
}

END {

    for (p in n) {

        printf "%-8s %-12.2f %-14.4f %-12.2f %-12.2f\n",
            p,
            tps[p]/n[p],
            acc[p]/n[p],
            len[p]/n[p],
            e2e[p]/n[p]
    }
}
' "$RESULTS" |
    sort -n

echo
echo "=================================================="
echo "Full CSV: $RESULTS"
echo "=================================================="
