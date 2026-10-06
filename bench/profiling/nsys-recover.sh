#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE="lambdalabs"
REMOTE_BASE="/home/ubuntu/tryton-replay/nsys"
LOCAL_BASE="${LOCAL_BASE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/results/profiling/nsys-qwen38}"

REMOTE_DIR="$(
    ssh "$REMOTE" "
        find '$REMOTE_BASE' \
          -mindepth 1 \
          -maxdepth 1 \
          -type d \
          -printf '%T@ %p\n' 2>/dev/null |
        sort -nr |
        head -n1 |
        cut -d' ' -f2-
    "
)"

if [[ -z "$REMOTE_DIR" ]]; then
    echo "ERROR: there is no remote Nsight profile."
    exit 1
fi

RUN_ID="$(basename "$REMOTE_DIR")"
LOCAL_DIR="$LOCAL_BASE/$RUN_ID"

echo "Remote : $REMOTE_DIR"
echo "Local  : $LOCAL_DIR"

echo
echo "Remote contents:"
ssh "$REMOTE" "ls -lah '$REMOTE_DIR'"

if ! ssh "$REMOTE" "test -s '$REMOTE_DIR/profile.nsys-rep'"; then
    echo
    echo "ERROR: the directory exists, but profile.nsys-rep does not."
    echo
    echo "Driver log:"
    ssh "$REMOTE" "cat '$REMOTE_DIR/nsys-driver.log' 2>/dev/null || true"
    exit 1
fi

mkdir -p "$LOCAL_DIR"

rsync -a --info=stats1 \
    "$REMOTE:$REMOTE_DIR/" \
    "$LOCAL_DIR/"

echo
echo
echo "Verifying copy..."

test -s "$LOCAL_DIR/profile.nsys-rep"

echo "OK: profile.nsys-rep copied."

ARCHIVE="${LOCAL_DIR}.tar.gz"

tar -C "$(dirname "$LOCAL_DIR")" \
    -czf "$ARCHIVE" \
    "$(basename "$LOCAL_DIR")"

echo
echo "=========================================="
echo " RECOVERED"
echo "=========================================="
echo "Archive:"
echo "  $ARCHIVE"
echo
ls -lh "$ARCHIVE"
