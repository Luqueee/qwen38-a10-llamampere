#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="${REMOTE:-lambdalabs}"
REMOTE_USER_HOME="/home/ubuntu"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-$REPO_ROOT/backups}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
DEST="$BACKUP_ROOT/$RUN_ID"

# 0 = do not copy large GGUF files.
# 1 = also copy /home/ubuntu/models.
BACKUP_MODELS="${BACKUP_MODELS:-0}"

# Image to preserve.
IMAGE="${IMAGE:-local/llamampere:v0.4-sm86}"

mkdir -p "$DEST"/{metadata,tryton-replay,docker,runtime,models}

log() {
    echo
    echo "================================================================"
    echo "$*"
    echo "================================================================"
}

remote_exists() {
    ssh "$REMOTE" "test -e '$1'"
}

###############################################################################
# 1. METADATA
###############################################################################

log "[1/8] Saving host metadata"

ssh "$REMOTE" '
set +e

echo "=== DATE ==="
date -Is

echo
echo "=== HOST ==="
hostname
uname -a

echo
echo "=== OS ==="
cat /etc/os-release

echo
echo "=== NVIDIA ==="
nvidia-smi

echo
echo "=== DOCKER ==="
docker version

echo
echo "=== NSYS ==="
nsys --version 2>&1 || true
' > "$DEST/metadata/system.txt"


###############################################################################
# 2. EXACT MODEL-SERVING CONFIGURATION
###############################################################################

log "[2/8] Saving exact model-serving configuration"

ssh "$REMOTE" '
docker inspect model-serving
' > "$DEST/docker/model-serving.inspect.json"

ssh "$REMOTE" '
docker inspect model-serving \
  --format "{{range .Config.Env}}{{println .}}{{end}}"
' > "$DEST/docker/model-serving.env.txt"

ssh "$REMOTE" '
docker inspect model-serving \
  --format "{{json .Config.Cmd}}"
' > "$DEST/docker/model-serving.cmd.json"

ssh "$REMOTE" '
docker logs model-serving 2>&1
' > "$DEST/docker/model-serving.log" || true

ssh "$REMOTE" '
docker inspect model-serving \
  --format "image={{.Config.Image}}
image_id={{.Image}}
created={{.Created}}
"
' > "$DEST/docker/model-serving-image.txt"


###############################################################################
# 3. COMPLETE TRYTON REPLAY
###############################################################################

log "[3/8] Copying tryton-replay"

if remote_exists "$REMOTE_USER_HOME/tryton-replay"; then
    rsync -aH \
        --partial \
        --info=progress2 \
        "$REMOTE:$REMOTE_USER_HOME/tryton-replay/" \
        "$DEST/tryton-replay/"
else
    echo "WARNING: $REMOTE_USER_HOME/tryton-replay does not exist"
fi


###############################################################################
# 4. MODEL VOCABULARIES / SMALL FILES
###############################################################################

log "[4/8] Saving inventory and vocabularies"

ssh "$REMOTE" '
set +e

echo "=== /home/ubuntu/models ==="
find /home/ubuntu/models \
    -maxdepth 3 \
    -type f \
    -printf "%s %TY-%Tm-%Td %TH:%TM:%TS %p\n" \
    | sort

echo
echo "=== SHA256 ==="

find /home/ubuntu/models \
    -maxdepth 3 \
    -type f \
    \( -name "*.gguf" -o -name "*.txt" -o -name "*.json" \) \
    -print0 |
while IFS= read -r -d "" f; do
    sha256sum "$f"
done
' > "$DEST/models/manifest.txt"


# Copy small shortlists/maps without copying all GGUF files yet.
rsync -a \
    --partial \
    --include='*/' \
    --include='*.txt' \
    --include='*.json' \
    --include='*.yaml' \
    --include='*.yml' \
    --exclude='*' \
    "$REMOTE:$REMOTE_USER_HOME/models/" \
    "$DEST/models/files/" || true


###############################################################################
# 5. REPOSITORIES / RUNTIME, IF PRESENT
###############################################################################

log "[5/8] Saving discovered runtime/repositories"

REMOTE_REPOS=(
    "$REMOTE_USER_HOME/llamAmpere"
    "$REMOTE_USER_HOME/llamAmpere-v0.4"
    "$REMOTE_USER_HOME/work/llamAmpere"
    "$REMOTE_USER_HOME/work/llamAmpere-v0.4"
)

for repo in "${REMOTE_REPOS[@]}"; do
    if remote_exists "$repo/.git"; then

        name="$(echo "$repo" | sed 's#^/##; s#/#_#g')"

        echo "Found remote repository: $repo"

        ssh "$REMOTE" "
            cd '$repo'
            {
                echo 'path=$repo'
                echo -n 'commit='
                git rev-parse HEAD
                echo -n 'branch='
                git branch --show-current
                echo
                git status --short
                echo
                git log -5 --oneline
            }
        " > "$DEST/runtime/${name}.git.txt"

        # Include the source; exclude large build directories.
        rsync -aH \
            --partial \
            --exclude='build/' \
            --exclude='build-*' \
            --exclude='.cache/' \
            "$REMOTE:$repo/" \
            "$DEST/runtime/$name/"
    fi
done


###############################################################################
# 6. EXPORT DOCKER IMAGE
###############################################################################

log "[6/8] Exporting Docker image: $IMAGE"

if ssh "$REMOTE" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then

    ssh "$REMOTE" "
        docker image inspect '$IMAGE'
    " > "$DEST/docker/llamampere-image.inspect.json"

    #
    # Stream directly from Lambda to the local machine.
    # Avoids creating a huge temporary file on Lambda.
    #
    if command -v zstd >/dev/null 2>&1 &&
       ssh "$REMOTE" 'command -v zstd >/dev/null 2>&1'; then

        echo "Using zstd..."

        ssh "$REMOTE" \
            "docker save '$IMAGE' | zstd -T0 -3 -c" \
            > "$DEST/docker/llamampere-v0.4-sm86.tar.zst"

    else

        echo "zstd unavailable on both hosts; using gzip..."

        ssh "$REMOTE" \
            "docker save '$IMAGE' | gzip -1 -c" \
            > "$DEST/docker/llamampere-v0.4-sm86.tar.gz"
    fi

else
    echo "WARNING: Docker image '$IMAGE' does not exist on Lambda."
fi


###############################################################################
# 7. OPTIONAL LARGE MODELS
###############################################################################

if [[ "$BACKUP_MODELS" == "1" ]]; then

    log "[7/8] Also copying GGUF models"

    rsync -aH \
        --partial \
        --append-verify \
        --info=progress2 \
        "$REMOTE:$REMOTE_USER_HOME/models/" \
        "$DEST/models/files/"

else

    log "[7/8] GGUF models OMITTED"

    echo "Large models were NOT copied." \
        > "$DEST/models/GGUF_NOT_COPIED.txt"

    echo "To include them, run:" \
        >> "$DEST/models/GGUF_NOT_COPIED.txt"

    echo \
        "BACKUP_MODELS=1 bash $0" \
        >> "$DEST/models/GGUF_NOT_COPIED.txt"
fi


###############################################################################
# 8. LOCAL MANIFEST
###############################################################################

log "[8/8] Generating backup checksums"

(
    cd "$DEST"

    find . \
        -type f \
        ! -name SHA256SUMS \
        -print0 |
    sort -z |
    xargs -0 sha256sum
) > "$DEST/SHA256SUMS"


cat > "$DEST/README.txt" <<README
Remote host backup
==================

Date:
  $(date -Is)

Remote:
  $REMOTE

Docker image:
  $IMAGE

Includes:
  - complete tryton-replay
  - benchmark results
  - Nsight profiles present under tryton-replay
  - exact model-serving configuration
  - logs
  - NVIDIA/Docker/Nsight metadata
  - llamAmpere Docker image
  - discovered repositories
  - small vocabularies/maps
  - checksums/model listing

GGUF models copied:
  $BACKUP_MODELS

Restore Docker image:

  .tar.zst:
    zstd -dc docker/llamampere-v0.4-sm86.tar.zst | docker load

  .tar.gz:
    gzip -dc docker/llamampere-v0.4-sm86.tar.gz | docker load

Verify integrity:

  sha256sum -c SHA256SUMS
README


echo
echo "================================================================"
echo " BACKUP COMPLETE"
echo "================================================================"
echo
echo "Destination:"
echo "  $DEST"
echo
du -sh "$DEST"
echo
echo "Main contents:"
du -sh "$DEST"/* 2>/dev/null | sort -h
echo
echo "To verify:"
echo "  cd '$DEST' && sha256sum -c SHA256SUMS"
