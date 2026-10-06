#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="${REMOTE:-lambdalabs}"
ORIGINAL="${ORIGINAL:-model-serving}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${BASE_DIR:-$REPO_ROOT}"
REPLAY="${REPLAY:-$BASE_DIR/bench/harness/replay.sh}"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LABEL="hybrid-diag-budget512-${RUN_ID}"

LOCAL_ROOT="$REPO_ROOT/results/experiments/hybrid-verifier/$RUN_ID"

REMOTE_HOME="/home/ubuntu"
REMOTE_BASE="$REMOTE_HOME/tryton-replay"
REMOTE_RESULTS="$REMOTE_BASE/results"
REMOTE_ROOT="$REMOTE_BASE/hybrid-verifier/$RUN_ID"

REMOTE_WORKTREE="$REMOTE_HOME/.hybrid-diag-worktree-$RUN_ID"

IMAGE="local/llamampere:v0.4-hybrid-diag-$RUN_ID"
BACKUP_CONTAINER="${ORIGINAL}-hybrid-diag-original-${RUN_ID}"

ORIGINAL_MOVED=0
WORKTREE_CREATED=0

mkdir -p "$LOCAL_ROOT"/{meta,logs,result}

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

        for i in $(seq 1 240); do
            if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
                echo READY
                exit 0
            fi
            sleep 1
        done

        echo "Timeout waiting for /health" >&2
        docker logs --tail 150 model-serving >&2 || true
        exit 1
    '
}

restore_original() {
    if [[ "$ORIGINAL_MOVED" != "1" ]]; then
        return
    fi

    echo
    echo "Restoring original model-serving..."

    ssh "$REMOTE" bash -s -- \
        "$ORIGINAL" \
        "$BACKUP_CONTAINER" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"
BACKUP="$2"

docker rm -f "$NAME" >/dev/null 2>&1 || true

if ! docker inspect "$BACKUP" >/dev/null 2>&1; then
    echo "Backup not found: $BACKUP" >&2
    exit 1
fi

docker rename "$BACKUP" "$NAME"
docker start "$NAME" >/dev/null
REMOTE

    ORIGINAL_MOVED=0
    wait_health >/dev/null || true

    echo "Original model-serving restored."
}

cleanup() {
    set +e

    restore_original

    if [[ "$WORKTREE_CREATED" == "1" && -n "${REMOTE_SRC:-}" ]]; then
        ssh "$REMOTE" "
            git -C '$REMOTE_SRC' worktree remove --force '$REMOTE_WORKTREE' >/dev/null 2>&1 || true
            git -C '$REMOTE_SRC' worktree prune >/dev/null 2>&1 || true
        " || true
    fi

    set -e
}

trap cleanup EXIT INT TERM

echo
echo "================================================="
echo " HYBRID GPU VERIFIER - ROUND DIAGNOSTICS"
echo "================================================="
echo
echo "Run       : $RUN_ID"
echo "Label     : $LABEL"
echo "Remote    : $REMOTE"
echo "Local     : $LOCAL_ROOT"
echo

[[ -x "$REPLAY" ]] ||
    die "Cannot find $REPLAY"

###############################################################################
# FIND REMOTE REPOSITORY
###############################################################################

log "Searching for llamAmpere checkout on lambdalabs"

REMOTE_SRC="$(
    ssh "$REMOTE" 'bash -s' <<'REMOTE'
set -Eeuo pipefail

find /home/ubuntu -maxdepth 6 -type d -name .git -printf '%h\n' 2>/dev/null |
while IFS= read -r repo; do
    url="$(git -C "$repo" remote get-url origin 2>/dev/null || true)"

    if printf '%s\n' "$url" | grep -qi 'JakeATX/llamAmpere'; then
        printf '%s\n' "$repo"
        exit 0
    fi
done
REMOTE
)"

if [[ -z "$REMOTE_SRC" ]]; then
    echo
    echo "Similar repositories found on Lambda:"
    ssh "$REMOTE" \
        "find /home/ubuntu -maxdepth 6 -type d -iname '*llam*ampere*' -print 2>/dev/null" \
        || true

    die "Cannot find remote checkout of JakeATX/llamAmpere"
fi

echo "Remote repository: $REMOTE_SRC"

REMOTE_COMMIT="$(
    ssh "$REMOTE" "git -C '$REMOTE_SRC' rev-parse HEAD"
)"

REMOTE_BRANCH="$(
    ssh "$REMOTE" "git -C '$REMOTE_SRC' branch --show-current || true"
)"

echo "Branch     : ${REMOTE_BRANCH:-detached}"
echo "Commit     : $REMOTE_COMMIT"

ssh "$REMOTE" "git -C '$REMOTE_SRC' status --short" \
    > "$LOCAL_ROOT/meta/source-status.txt"

if [[ -s "$LOCAL_ROOT/meta/source-status.txt" ]]; then
    echo
    echo "The remote repository has local changes."
    echo "They will not be modified; using an isolated worktree:"
    cat "$LOCAL_ROOT/meta/source-status.txt"
fi

###############################################################################
# REMOTE PRECHECK
###############################################################################

log "Precheck"

ssh "$REMOTE" bash -s -- "$ORIGINAL" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"

command -v docker
command -v git
command -v python3
command -v curl
command -v nvidia-smi

docker inspect "$NAME" >/dev/null

echo
docker ps \
    --filter "name=^/${NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

echo
nvidia-smi \
    --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
REMOTE

###############################################################################
# SAVE ORIGINAL CONFIGURATION
###############################################################################

log "Saving original configuration"

ssh "$REMOTE" "docker inspect '$ORIGINAL'" \
    > "$LOCAL_ROOT/meta/original-container.json"

python3 - "$LOCAL_ROOT/meta/original-container.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))[0]

cmd = list(d["Config"].get("Cmd") or [])
env = list(d["Config"].get("Env") or [])

def val(flag):
    if flag not in cmd:
        raise SystemExit(f"Missing {flag}")

    i = cmd.index(flag)

    if i + 1 >= len(cmd):
        raise SystemExit(f"{flag} has no value")

    return cmd[i + 1]

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

bad = []

for flag, want in expected.items():
    got = val(flag)

    if got != want:
        bad.append(
            f"{flag}: expected={want}, actual={got}"
        )

for x in env:
    if x.startswith("LLAMA_MTP_GPU_VERIFY="):
        bad.append(
            f"GPU verifier active: {x}"
        )

    if x.startswith("LLAMA_SPEC_ADAPT_"):
        bad.append(
            f"Adaptive active: {x}"
        )

if bad:
    print("Incorrect baseline:")

    for x in bad:
        print("  -", x)

    raise SystemExit(1)

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
)
PY

###############################################################################
# REMOTE WORKTREE
###############################################################################

log "Creating isolated remote worktree"

ssh "$REMOTE" "
    rm -rf '$REMOTE_WORKTREE'

    git -C '$REMOTE_SRC' \
        worktree add \
        --detach \
        '$REMOTE_WORKTREE' \
        '$REMOTE_COMMIT'
"

WORKTREE_CREATED=1

###############################################################################
# DIAGNOSTIC PATCH
###############################################################################

log "Applying instrumentation"

ssh "$REMOTE" \
    "python3 - '$REMOTE_WORKTREE'" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

def replace_once(path, old, new):
    p = root / path
    s = p.read_text()

    n = s.count(old)

    if n != 1:
        raise SystemExit(
            f"{path}: expected anchor once, found {n}"
        )

    p.write_text(
        s.replace(old, new, 1)
    )

def insert_before(path, anchor, text):
    p = root / path
    s = p.read_text()

    n = s.count(anchor)

    if n != 1:
        raise SystemExit(
            f"{path}: expected anchor once, found {n}"
        )

    p.write_text(
        s.replace(
            anchor,
            text + anchor,
            1,
        )
    )

###############################################################################
# Lazy grammar getter
###############################################################################

insert_before(
    "include/llama.h",
    "    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow.",
    """    // Diagnostic helper for hybrid verifier investigation.
    LLAMA_API bool llama_sampler_grammar_is_awaiting_trigger(
            const struct llama_sampler * smpl);

"""
)

insert_before(
    "src/llama-sampler.cpp",
    "// penalties\n",
    """bool llama_sampler_grammar_is_awaiting_trigger(
        const struct llama_sampler * smpl) {

    if (!smpl) {
        return false;
    }

    const auto * ctx =
        (const llama_sampler_grammar *) smpl->ctx;

    return
        ctx->grammar != nullptr &&
        ctx->grammar->awaiting_trigger;
}

"""
)

###############################################################################
# Reasoning remaining getter
###############################################################################

replace_once(
    "common/reasoning-budget.h",
    """common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);
""",
    """common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

int32_t common_reasoning_budget_get_remaining(
        const struct llama_sampler * smpl);
"""
)

replace_once(
    "common/reasoning-budget.cpp",
    """common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

""",
    """common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

int32_t common_reasoning_budget_get_remaining(
        const struct llama_sampler * smpl) {

    if (!smpl) {
        return -1;
    }

    return ((const common_reasoning_budget_ctx *) smpl->ctx)->remaining;
}

"""
)

###############################################################################
# Common sampler diagnostic getters
###############################################################################

replace_once(
    "common/sampling.h",
    """// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers
""",
    """// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// Read-only hybrid-verifier diagnostics.
bool common_sampler_diag_has_grammar(
        const struct common_sampler * gsmpl);

bool common_sampler_diag_grammar_lazy(
        const struct common_sampler * gsmpl);

bool common_sampler_diag_grammar_awaiting_trigger(
        const struct common_sampler * gsmpl);

bool common_sampler_diag_has_reasoning_budget(
        const struct common_sampler * gsmpl);

int32_t common_sampler_diag_reasoning_state(
        const struct common_sampler * gsmpl);

int32_t common_sampler_diag_reasoning_remaining(
        const struct common_sampler * gsmpl);

// helpers
"""
)

replace_once(
    "common/sampling.cpp",
    """bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers
""",
    """bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

bool common_sampler_diag_has_grammar(
        const struct common_sampler * gsmpl) {

    return gsmpl && gsmpl->grmr;
}

bool common_sampler_diag_grammar_lazy(
        const struct common_sampler * gsmpl) {

    return
        gsmpl &&
        gsmpl->grmr &&
        gsmpl->params.grammar_lazy;
}

bool common_sampler_diag_grammar_awaiting_trigger(
        const struct common_sampler * gsmpl) {

    return
        gsmpl &&
        gsmpl->grmr &&
        gsmpl->params.grammar_lazy &&
        llama_sampler_grammar_is_awaiting_trigger(
            gsmpl->grmr
        );
}

bool common_sampler_diag_has_reasoning_budget(
        const struct common_sampler * gsmpl) {

    return gsmpl && gsmpl->rbudget;
}

int32_t common_sampler_diag_reasoning_state(
        const struct common_sampler * gsmpl) {

    if (!gsmpl || !gsmpl->rbudget) {
        return -1;
    }

    return (int32_t)
        common_reasoning_budget_get_state(
            gsmpl->rbudget
        );
}

int32_t common_sampler_diag_reasoning_remaining(
        const struct common_sampler * gsmpl) {

    if (!gsmpl || !gsmpl->rbudget) {
        return -1;
    }

    return
        common_reasoning_budget_get_remaining(
            gsmpl->rbudget
        );
}

// helpers
"""
)

###############################################################################
# One HYBRID_DIAG line per round
###############################################################################

old = """            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                auto accepted = !synth_probs.empty()
                    ? server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay)
                    : (slot.spec_draft_q.empty()
                        ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)
                        : common_sampler_sample_and_accept_n_pq(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_draft_q, false, slot.spec_is_replay));
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);
"""

new = """            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                const bool diag_g_before =
                    common_sampler_diag_has_grammar(
                        slot.smpl.get()
                    );

                const bool diag_lazy_before =
                    common_sampler_diag_grammar_lazy(
                        slot.smpl.get()
                    );

                const bool diag_await_before =
                    common_sampler_diag_grammar_awaiting_trigger(
                        slot.smpl.get()
                    );

                const bool diag_rb_before =
                    common_sampler_diag_has_reasoning_budget(
                        slot.smpl.get()
                    );

                const int diag_rs_before =
                    common_sampler_diag_reasoning_state(
                        slot.smpl.get()
                    );

                const int diag_rem_before =
                    common_sampler_diag_reasoning_remaining(
                        slot.smpl.get()
                    );

                GGML_ASSERT(
                    slot.spec_i_batch.size() ==
                    n_draft + 1
                );

                const auto & synth_probs =
                    common_speculative_get_synth_probs(
                        spec.get()
                    );

                auto accepted =
                    !synth_probs.empty()
                    ? server_sample_and_accept_synth(
                        slot.smpl.get(),
                        slot.ctx_tgt,
                        slot.spec_i_batch,
                        slot.spec_draft,
                        synth_probs,
                        slot.spec_synth_rng,
                        slot.spec_is_replay
                    )
                    : (
                        slot.spec_draft_q.empty()
                        ? common_sampler_sample_and_accept_n(
                            slot.smpl.get(),
                            slot.ctx_tgt,
                            slot.spec_i_batch,
                            slot.spec_draft
                        )
                        : common_sampler_sample_and_accept_n_pq(
                            slot.smpl.get(),
                            slot.ctx_tgt,
                            slot.spec_i_batch,
                            slot.spec_draft,
                            slot.spec_draft_q,
                            false,
                            slot.spec_is_replay
                        )
                    );

                const bool diag_g_after =
                    common_sampler_diag_has_grammar(
                        slot.smpl.get()
                    );

                const bool diag_lazy_after =
                    common_sampler_diag_grammar_lazy(
                        slot.smpl.get()
                    );

                const bool diag_await_after =
                    common_sampler_diag_grammar_awaiting_trigger(
                        slot.smpl.get()
                    );

                const bool diag_rb_after =
                    common_sampler_diag_has_reasoning_budget(
                        slot.smpl.get()
                    );

                const int diag_rs_after =
                    common_sampler_diag_reasoning_state(
                        slot.smpl.get()
                    );

                const int diag_rem_after =
                    common_sampler_diag_reasoning_remaining(
                        slot.smpl.get()
                    );

                const int diag_rows =
                    (int) n_draft + 1;

                const bool diag_grammar_safe =
                    !diag_g_before ||
                    (
                        diag_lazy_before &&
                        diag_await_before &&
                        diag_g_after &&
                        diag_lazy_after &&
                        diag_await_after
                    );

                // reasoning states:
                // 0 = IDLE
                // 1 = COUNTING
                // 2 = FORCING
                // 3 = WAITING_UTF8
                // 4 = DONE
                const bool diag_reason_start_safe =
                    !diag_rb_before ||
                    (
                        diag_rs_before != 2 &&
                        diag_rs_before != 3 &&
                        (
                            diag_rs_before != 1 ||
                            diag_rem_before > diag_rows
                        )
                    );

                const bool diag_reason_end_safe =
                    !diag_rb_after ||
                    (
                        diag_rs_after != 2 &&
                        diag_rs_after != 3
                    );

                const bool diag_safe =
                    diag_grammar_safe &&
                    diag_reason_start_safe &&
                    diag_reason_end_safe;

                SLT_INF(
                    slot,
                    "HYBRID_DIAG "
                    "round=%d "
                    "draft=%zu "
                    "accepted=%zu "
                    "g=%d "
                    "lazy=%d "
                    "await_before=%d "
                    "await_after=%d "
                    "rb=%d "
                    "rs_before=%d "
                    "rem_before=%d "
                    "rs_after=%d "
                    "rem_after=%d "
                    "safe=%d\\n",
                    (int) slot.stats.n_draft_verif_steps + 1,
                    n_draft,
                    accepted.size() - 1,
                    (int) diag_g_before,
                    (int) diag_lazy_before,
                    (int) diag_await_before,
                    (int) diag_await_after,
                    (int) diag_rb_before,
                    diag_rs_before,
                    diag_rem_before,
                    diag_rs_after,
                    diag_rem_after,
                    (int) diag_safe
                );

                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);
"""

replace_once(
    "tools/server/server-context.cpp",
    old,
    new,
)

print("Instrumentation patch applied successfully.")
PY

ssh "$REMOTE" "
    git -C '$REMOTE_WORKTREE' diff --check
    git -C '$REMOTE_WORKTREE' diff --stat
"

ssh "$REMOTE" \
    "git -C '$REMOTE_WORKTREE' diff" \
    > "$LOCAL_ROOT/meta/instrumentation.patch"

###############################################################################
# REMOTE BUILD
###############################################################################

log "Building instrumented image on Lambda"

ssh "$REMOTE" "
    set -Eeuo pipefail

    cd '$REMOTE_WORKTREE'

    docker build \
        --build-arg CUDA_DOCKER_ARCH=86 \
        --target server \
        -f .devops/cuda.Dockerfile \
        -t '$IMAGE' \
        .
" 2>&1 | tee "$LOCAL_ROOT/logs/docker-build.log"

###############################################################################
# CREATE PAYLOAD
###############################################################################

log "Preparing instrumented container"

ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT'"

rsync -a \
    "$LOCAL_ROOT/meta/original-container.json" \
    "$REMOTE:$REMOTE_ROOT/original-container.json" \
    >/dev/null

ssh "$REMOTE" \
    "python3 - '$REMOTE_ROOT/original-container.json' '$IMAGE' '$REMOTE_ROOT/create.json'" \
    <<'PY'
import json
import sys
from pathlib import Path

source, image, output = sys.argv[1:4]

d = json.load(open(source))[0]

cfg = dict(d["Config"])
host = dict(d["HostConfig"])

cmd = list(cfg.get("Cmd") or [])
env = list(cfg.get("Env") or [])

def val(flag):
    return cmd[
        cmd.index(flag) + 1
    ]

assert val("--reasoning-budget") == "512"
assert val("--spec-draft-n-max") == "7"

env = [
    x
    for x in env
    if not x.startswith(
        "LLAMA_MTP_GPU_VERIFY="
    )
    and not x.startswith(
        "LLAMA_SPEC_ADAPT_COST="
    )
    and not x.startswith(
        "LLAMA_SPEC_ADAPT_FLOOR="
    )
]

cfg["Image"] = image
cfg["Env"] = env
cfg["HostConfig"] = host

Path(output).write_text(
    json.dumps(
        cfg,
        separators=(",", ":"),
    )
)

print("Image :", image)
print("Budget:", val("--reasoning-budget"))
print("n_max :", val("--spec-draft-n-max"))
PY

###############################################################################
# REPLACE model-serving
###############################################################################

log "Starting instrumented image"

ssh "$REMOTE" bash -s -- \
    "$ORIGINAL" \
    "$BACKUP_CONTAINER" \
    "$REMOTE_ROOT/create.json" <<'REMOTE'
set -Eeuo pipefail

NAME="$1"
BACKUP="$2"
PAYLOAD="$3"

docker stop "$NAME" >/dev/null
docker rename "$NAME" "$BACKUP"

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
    echo "Docker create failed: HTTP=$CODE"
    cat "$RESP"

    docker rename "$BACKUP" "$NAME" || true
    docker start "$NAME" >/dev/null || true

    exit 1
fi

docker start "$NAME" >/dev/null
REMOTE

ORIGINAL_MOVED=1

wait_health

###############################################################################
# REPLAY REAL
###############################################################################

log "Running live replay budget=512"

"$REPLAY" replay "$LABEL" 2>&1 |
    tee "$LOCAL_ROOT/logs/replay.log"

ssh "$REMOTE" \
    "docker logs '$ORIGINAL' 2>&1" \
    > "$LOCAL_ROOT/logs/model-serving.log"

COUNT="$(
    grep -c 'HYBRID_DIAG' \
        "$LOCAL_ROOT/logs/model-serving.log" \
        || true
)"

echo
echo "HYBRID_DIAG rounds: $COUNT"

if [[ "$COUNT" -eq 0 ]]; then
    die "No HYBRID_DIAG logs were generated"
fi

###############################################################################
# COPY RESULTS
###############################################################################

REMOTE_RESULT="$(
    ssh "$REMOTE" \
        "python3 - '$REMOTE_RESULTS' '$LABEL'" \
        <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
label = sys.argv[2]

results = [
    p for p in root.iterdir()
    if p.is_dir()
    and p.name.endswith(
        "_" + label
    )
]

if not results:
    raise SystemExit(
        "Result not found"
    )

results.sort(
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

print(results[0])
PY
)"

rsync -aH \
    "$REMOTE:$REMOTE_RESULT/" \
    "$LOCAL_ROOT/result/"

###############################################################################
# RESTORE PRODUCTION
###############################################################################

restore_original

###############################################################################
# ANALYSIS
###############################################################################

log "Analyzing rounds"

python3 - \
    "$LOCAL_ROOT/logs/model-serving.log" \
    "$LOCAL_ROOT" <<'PY'
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys

log_path = Path(sys.argv[1])
root = Path(sys.argv[2])

lines = [
    x
    for x in log_path.read_text(
        errors="replace"
    ).splitlines()
    if "HYBRID_DIAG" in x
]

state_name = {
    -1: "NONE",
     0: "IDLE",
     1: "COUNTING",
     2: "FORCING",
     3: "WAITING_UTF8",
     4: "DONE",
}

task_re = re.compile(
    r"\btask\s+(-?\d+)\b"
)

kv_re = re.compile(
    r"([A-Za-z_]+)=(-?\d+)"
)

rows = []

for line in lines:
    tm = task_re.search(line)

    task = (
        int(tm.group(1))
        if tm
        else -999999
    )

    r = {
        k: int(v)
        for k, v in kv_re.findall(line)
    }

    r["task"] = task
    rows.append(r)

if not rows:
    raise SystemExit(
        "No HYBRID_DIAG data"
    )

total = len(rows)
safe = sum(r["safe"] for r in rows)

def pct(n, d=None):
    if d is None:
        d = total

    return (
        0.0
        if d == 0
        else 100.0 * n / d
    )

await_both = sum(
    r["g"]
    and r["lazy"]
    and r["await_before"]
    and r["await_after"]
    for r in rows
)

trigger_cross = sum(
    r["g"]
    and r["lazy"]
    and r["await_before"]
    and not r["await_after"]
    for r in rows
)

active_start = sum(
    r["g"]
    and not r["await_before"]
    for r in rows
)

near_budget = sum(
    r["rb"]
    and r["rs_before"] == 1
    and r["rem_before"] <= r["draft"] + 1
    for r in rows
)

forcing_waiting = sum(
    r["rs_before"] in (2, 3)
    or r["rs_after"] in (2, 3)
    for r in rows
)

reason_start = Counter(
    r["rs_before"]
    for r in rows
)

by_task = defaultdict(list)

for r in rows:
    by_task[r["task"]].append(r)

print()
print("=" * 84)
print("HYBRID VERIFIER DIAGNOSTIC")
print("=" * 84)
print()

print(
    f"Total MTP rounds                  : "
    f"{total}"
)

print(
    f"Conservative GPU-safe rounds      : "
    f"{safe} ({pct(safe):.2f}%)"
)

print(
    f"Conservative unsafe rounds        : "
    f"{total-safe} ({pct(total-safe):.2f}%)"
)

print()

print(
    f"Awaiting trigger start + end      : "
    f"{await_both} ({pct(await_both):.2f}%)"
)

print(
    f"Trigger crossed inside round      : "
    f"{trigger_cross} ({pct(trigger_cross):.2f}%)"
)

print(
    f"Grammar active at round start     : "
    f"{active_start} ({pct(active_start):.2f}%)"
)

print()

print(
    f"Reasoning near budget boundary    : "
    f"{near_budget} ({pct(near_budget):.2f}%)"
)

print(
    f"FORCING/WAITING touched           : "
    f"{forcing_waiting} ({pct(forcing_waiting):.2f}%)"
)

print()
print("Reasoning state at round start:")

for state in sorted(reason_start):
    n = reason_start[state]

    print(
        f"  {state_name.get(state, state):<12}"
        f": {n:5d} ({pct(n):6.2f}%)"
    )

print()
print("-" * 84)

print(
    f'{"TASK":>8} '
    f'{"ROUNDS":>8} '
    f'{"SAFE":>8} '
    f'{"SAFE%":>8} '
    f'{"AWAIT":>8} '
    f'{"TRIGGER":>8} '
    f'{"ACTIVE":>8}'
)

print("-" * 84)

task_summary = {}

for task in sorted(by_task):
    rr = by_task[task]

    n = len(rr)
    s = sum(x["safe"] for x in rr)

    aw = sum(
        x["g"]
        and x["lazy"]
        and x["await_before"]
        and x["await_after"]
        for x in rr
    )

    tr = sum(
        x["g"]
        and x["lazy"]
        and x["await_before"]
        and not x["await_after"]
        for x in rr
    )

    ac = sum(
        x["g"]
        and not x["await_before"]
        for x in rr
    )

    print(
        f"{task:8d} "
        f"{n:8d} "
        f"{s:8d} "
        f"{pct(s,n):7.2f}% "
        f"{aw:8d} "
        f"{tr:8d} "
        f"{ac:8d}"
    )

    task_summary[str(task)] = {
        "rounds": n,
        "safe": s,
        "safe_pct": pct(s, n),
        "await": aw,
        "trigger": tr,
        "active": ac,
    }

print()
print("=" * 84)

safe_pct = pct(safe)

if safe_pct >= 70:
    decision = (
        "HYBRID_PATCH_HIGH_VALUE: "
        ">=70% of rounds are GPU-safe."
    )

elif safe_pct >= 40:
    decision = (
        "HYBRID_PATCH_MAYBE: "
        "40-70% of rounds are GPU-safe."
    )

else:
    decision = (
        "KERNEL_MORE_PROMISING: "
        "<40% of rounds are GPU-safe."
    )

print(decision)

summary = {
    "total_rounds": total,
    "safe_rounds": safe,
    "safe_pct": safe_pct,
    "await_both": await_both,
    "trigger_crossed": trigger_cross,
    "grammar_active_start": active_start,
    "near_budget": near_budget,
    "forcing_waiting": forcing_waiting,
    "tasks": task_summary,
    "decision": decision,
}

(root / "hybrid-summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

(root / "hybrid-summary.txt").write_text(
    f"Total rounds : {total}\n"
    f"Safe rounds  : {safe}\n"
    f"Safe pct     : {safe_pct:.2f}%\n"
    f"Trigger      : {trigger_cross}\n"
    f"Active       : {active_start}\n"
    f"Near budget  : {near_budget}\n"
    f"Decision     : {decision}\n"
)
PY

echo
echo "Replay:"
grep -E \
    '^(Requests|Successful|E2E|Prompt tokens|Prefill|Generated|Decode|MTP acceptance|MTP mean len|VRAM peak|Energy)' \
    "$LOCAL_ROOT/logs/replay.log" \
    || true

echo
echo "Summary:"
cat "$LOCAL_ROOT/hybrid-summary.txt"

echo
echo "Results:"
echo "$LOCAL_ROOT"

echo
echo "model-serving:"
ssh "$REMOTE" \
    "docker ps \
    --filter 'name=^/${ORIGINAL}$' \
    --format '{{.Names}} -> {{.Status}}'"
