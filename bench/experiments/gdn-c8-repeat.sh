#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPLAY="${REPLAY:-$REPO_ROOT/bench/harness/replay.sh}"
REMOTE="${REMOTE:-lambdalabs}"
ORIGINAL="model-serving-gdn-original"

restore_baseline() {
    echo
    echo "=========================================="
    echo " RESTORING ORIGINAL CONTAINER"
    echo "=========================================="

    ssh "$REMOTE" bash <<'REMOTE'
set -e

docker rm -f model-serving >/dev/null 2>&1 || true

if docker inspect model-serving-gdn-original >/dev/null 2>&1; then
    docker rename model-serving-gdn-original model-serving
    docker start model-serving >/dev/null
fi

echo "Waiting for baseline..."

until curl -fsS \
    http://127.0.0.1:8000/health \
    >/dev/null 2>&1
do
    sleep 2
done

echo "Baseline restored."
REMOTE
}

trap restore_baseline EXIT INT TERM

echo
echo "=========================================="
echo " PREPARING GDN A/B"
echo "=========================================="

ssh "$REMOTE" bash <<'REMOTE'
set -Eeuo pipefail

# Clean up leftovers from a previously aborted test.
if docker inspect model-serving-gdn-original >/dev/null 2>&1; then
    echo "ERROR: model-serving-gdn-original already exists."
    echo "Restore the baseline first."
    exit 1
fi

docker inspect model-serving >/dev/null

echo "Saving inspect..."
docker inspect model-serving \
    > /home/ubuntu/model-serving-before-gdn.json

echo "Stopping baseline..."
docker stop model-serving >/dev/null

echo "Preserving original container..."
docker rename model-serving model-serving-gdn-original

cat > /home/ubuntu/create-gdn-variant.py <<'PY'
#!/usr/bin/env python3

import json
import subprocess
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: create-gdn-variant.py COLS PREFETCH")

cols = sys.argv[1]
prefetch = sys.argv[2]

if cols not in {"1", "2", "4", "8"}:
    raise SystemExit(f"invalid COLS={cols}")

if prefetch not in {"0", "1"}:
    raise SystemExit(f"invalid PREFETCH={prefetch}")

def run(args, **kwargs):
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, check=True, **kwargs)

raw = subprocess.check_output(
    ["docker", "inspect", "model-serving-gdn-original"],
    text=True,
)

data = json.loads(raw)[0]

cfg = data["Config"]
host = data["HostConfig"]

subprocess.run(
    ["docker", "rm", "-f", "model-serving"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

args = [
    "docker",
    "create",
    "--name",
    "model-serving",
]

#
# GPU
#
device_requests = host.get("DeviceRequests") or []

has_gpu = False

for req in device_requests:
    caps = req.get("Capabilities") or []

    if req.get("Driver") == "nvidia":
        has_gpu = True

    for group in caps:
        if "gpu" in group:
            has_gpu = True

if has_gpu:
    args += ["--gpus", "all"]

runtime = host.get("Runtime")

if runtime and runtime not in {"", "runc"}:
    args += ["--runtime", runtime]

#
# Network
#
network = host.get("NetworkMode")

if network and network not in {"", "default"}:
    args += ["--network", network]

#
# Published ports
#
if network != "host":
    seen_ports = set()

    for container_port, bindings in (host.get("PortBindings") or {}).items():
        if not bindings:
            continue

        for binding in bindings:
            host_port = binding.get("HostPort")
            host_ip = binding.get("HostIp", "")

            if not host_port:
                continue

            if host_ip in {"", "0.0.0.0", "::"}:
                spec = f"{host_port}:{container_port}"
            else:
                spec = f"{host_ip}:{host_port}:{container_port}"

            if spec not in seen_ports:
                args += ["-p", spec]
                seen_ports.add(spec)

#
# Reuse the original container's mounts exactly.
#
args += [
    "--volumes-from",
    "model-serving-gdn-original",
]

#
# Relevant host config
#
if host.get("Privileged"):
    args.append("--privileged")

ipc_mode = host.get("IpcMode")

if ipc_mode and ipc_mode not in {"", "private"}:
    args += ["--ipc", ipc_mode]

pid_mode = host.get("PidMode")

if pid_mode:
    args += ["--pid", pid_mode]

shm_size = host.get("ShmSize")

if shm_size and shm_size != 67108864:
    args += ["--shm-size", str(shm_size)]

for ulimit in host.get("Ulimits") or []:
    args += [
        "--ulimit",
        f'{ulimit["Name"]}={ulimit["Soft"]}:{ulimit["Hard"]}',
    ]

for extra_host in host.get("ExtraHosts") or []:
    args += ["--add-host", extra_host]

#
# Copy the current environment exactly, removing only the GDN knobs.
#
env = []

for item in cfg.get("Env") or []:
    if item.startswith("GGML_CUDA_SM86_GDN_COLS="):
        continue

    if item.startswith("GGML_CUDA_SM86_GDN_PREFETCH="):
        continue

    env.append(item)

env.append(f"GGML_CUDA_SM86_GDN_COLS={cols}")

if prefetch == "1":
    env.append("GGML_CUDA_SM86_GDN_PREFETCH=1")

for item in env:
    args += ["-e", item]

#
# Same image and same CMD.
#
args.append(cfg["Image"])
args.extend(cfg.get("Cmd") or [])

run(args)
run(["docker", "start", "model-serving"])

print()
print(f"GDN_COLS={cols}")
print(f"GDN_PREFETCH={prefetch}")
PY

chmod +x /home/ubuntu/create-gdn-variant.py

echo "Original preserved as:"
docker ps -a \
    --filter name=model-serving-gdn-original \
    --format '  {{.Names}}  {{.Image}}  {{.Status}}'
REMOTE

run_variant() {
    local cols="$1"
    local prefetch="$2"
    local label="$3"

    echo
    echo "=========================================="
    echo " $label"
    echo " COLS=$cols PREFETCH=$prefetch"
    echo "=========================================="

    ssh "$REMOTE" \
        "/home/ubuntu/create-gdn-variant.py '$cols' '$prefetch'"

    ssh "$REMOTE" bash <<'REMOTE'
set -e

echo "Waiting for model-serving..."

until curl -fsS \
    http://127.0.0.1:8000/health \
    >/dev/null 2>&1
do
    sleep 2
done

echo
echo "Active GDN environment:"

docker inspect model-serving \
    --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep 'GGML_CUDA_SM86_GDN' \
    | sort

echo
REMOTE

    "$REPLAY" replay "$label"

    echo
    echo "Cooling down 15s..."
    sleep 15
}

#
# Previously measured baseline:
#   cols=4
#   prefetch=0
#   mean decode = 75.72 tok/s
#
run_variant 8 0 "gdn-c8-p0-r1"
run_variant 8 0 "gdn-c8-p0-r2"
run_variant 8 0 "gdn-c8-p0-r3"

echo
echo "=========================================="
echo " GDN SCREENING COMPLETE"
echo "=========================================="
echo
echo "Known baseline:"
echo "  cols=4 prefetch=0 -> 75.72 tok/s"
echo
echo "The trap will now restore the original model-serving container."
