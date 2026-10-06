#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="${REMOTE:-lambdalabs}"
DATE="${DATE:-$(date +%Y-%m-%d)}"
STAMP="$(date +%Y%m%d-%H%M%S)"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${DEST:-$REPO_ROOT/backups/lambdalabs-backup-$STAMP}"

mkdir -p "$DEST"/{files,metadata}

echo "================================================="
echo " HOST BACKUP - TODAY'S CHANGES"
echo "================================================="
echo
echo "Remote : $REMOTE"
echo "Date   : $DATE"
echo "Dest.  : $DEST"
echo

echo "[1/5] Recording the inventory of files modified today..."

ssh "$REMOTE" bash -s -- "$DATE" > "$DEST/metadata/modified-files.txt" <<'REMOTE'
set -Eeuo pipefail

DATE="$1"

find /home/ubuntu \
    -xdev \
    -type f \
    -newermt "$DATE 00:00:00" \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
    2>/dev/null |
sort
REMOTE

echo
echo "Files found:"
wc -l "$DEST/metadata/modified-files.txt"

echo
echo "[2/5] Copying files modified today..."

ssh "$REMOTE" bash -s -- "$DATE" <<'REMOTE' | tar -C "$DEST/files" -xf -
set -Eeuo pipefail

DATE="$1"

cd /

find home/ubuntu \
    -xdev \
    -type f \
    -newermt "$DATE 00:00:00" \
    -print0 \
    2>/dev/null |
tar \
    --null \
    --files-from=- \
    --no-recursion \
    --numeric-owner \
    -cf -
REMOTE

echo
echo "[3/5] Recording Git repositories and state..."

ssh "$REMOTE" bash -s > "$DEST/metadata/git-state.txt" <<'REMOTE'
set -Eeuo pipefail

find /home/ubuntu \
    -maxdepth 7 \
    \( -type d -o -type f \) \
    -name .git \
    -print0 \
    2>/dev/null |
while IFS= read -r -d '' dotgit; do
    repo="${dotgit%/.git}"

    if [ -f "$dotgit" ]; then
        repo="$(dirname "$dotgit")"
    fi

    echo
    echo "================================================="
    echo "REPO: $repo"
    echo "================================================="

    git -C "$repo" remote -v 2>/dev/null || true
    echo
    git -C "$repo" status --short --branch 2>/dev/null || true
    echo
    git -C "$repo" log -1 --decorate --oneline 2>/dev/null || true
done
REMOTE

echo
echo "[4/5] Recording server and Docker metadata..."

ssh "$REMOTE" bash -s > "$DEST/metadata/system.txt" <<'REMOTE'
set +e

echo "===== HOST ====="
hostname
date
uname -a

echo
echo "===== DISK ====="
df -h

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== DOCKER PS ====="
docker ps -a

echo
echo "===== DOCKER IMAGES - LIST ONLY ====="
docker images

echo
echo "===== MODEL-SERVING INSPECT ====="
docker inspect model-serving

echo
echo "===== MODEL-SERVING LOGS ====="
docker logs --timestamps model-serving 2>&1

echo
echo "===== MODEL-SERVING VERSION ====="
docker exec model-serving /app/llama-server --version 2>&1

echo
echo "===== MODEL-SERVING ENV ====="
docker inspect model-serving \
    --format '{{range .Config.Env}}{{println .}}{{end}}'

echo
echo "===== MODEL-SERVING CMD ====="
docker inspect model-serving \
    --format '{{json .Config.Cmd}}'

echo
echo "===== IMAGE METADATA ====="
docker image inspect local/llamampere:v0.4-sm86
REMOTE

echo
echo "[5/5] Packaging backup..."

tar \
    -C "$(dirname "$DEST")" \
    -czf "${DEST}.tar.gz" \
    "$(basename "$DEST")"

echo
echo "================================================="
echo " BACKUP COMPLETE"
echo "================================================="
echo
echo "Directory:"
echo "  $DEST"
echo
echo "Tarball:"
echo "  ${DEST}.tar.gz"
echo
echo "Size:"
du -sh "$DEST" "${DEST}.tar.gz"

echo
echo "Main files copied today:"
find "$DEST/files/home/ubuntu" \
    -maxdepth 4 \
    -type f \
    -printf '  %p\n' \
    2>/dev/null |
head -100

echo
echo "IMPORTANT:"
echo "  No Docker image was copied."
echo "  Nothing was changed or deleted on the remote host."
