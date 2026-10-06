#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="${1:-lambda}"
SSH_KEY="${SSH_KEY:-}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_FILE="${MONITOR_LOG_FILE:-${REPO_ROOT}/results/telemetry/llm-monitor.log}"

ssh_args=(
  -T
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

if [[ -n "$SSH_KEY" ]]; then
  ssh_args+=(-i "$SSH_KEY")
fi

echo "Connecting to ${SSH_TARGET}..."
mkdir -p "$(dirname -- "$LOG_FILE")"
touch "$LOG_FILE"
echo "Diagnostic log: ${LOG_FILE}"

ssh "${ssh_args[@]}" "${SSH_TARGET}" python3 -u - 2>>"$LOG_FILE" <<'PY'
import json
import os
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/slots"
INTERVAL = 0.2
RESPONSE_END_GRACE = 1.0
MONITOR_STARTED_AT = time.monotonic()
SSH_PARENT_PID = os.getppid()


def container_api_key():
    try:
        result = subprocess.run(
            [
                "sudo", "-n", "docker", "inspect", "--format",
                "{{json .Config.Cmd}}", "model-serving",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        raise SystemExit("Could not inspect the model-serving container.")

    command = json.loads(result.stdout)
    if "--api-key" not in command:
        return None
    key_index = command.index("--api-key") + 1
    if key_index >= len(command):
        raise SystemExit("model-serving has --api-key without a value.")
    return command[key_index]


API_KEY = container_api_key()


def log_event(event, **fields):
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed": round(time.monotonic() - MONITOR_STARTED_AT, 3),
        "event": event,
    }
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)


def decoded_tokens(slot):
    values = slot.get("next_token", [])
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return 0
    return sum(
        item.get("n_decoded", 0)
        for item in values
        if isinstance(item, dict)
    )


def context_status(used, capacity):
    if capacity <= 0:
        return "unknown context"
    used = max(0, min(used, capacity))
    percentage = used * 100 / capacity
    remaining = capacity - used
    width = 16
    filled = min(width, round(percentage * width / 100))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"ctx [{bar}] {used}/{capacity} "
        f"({percentage:5.1f}%, {remaining} free)"
    )


previous_tokens = None
previous_time = None
previous_task = None
last_context_used = None
last_context_capacity = None
task_started_at = None
task_ttft = None
task_ttft_missed = False
last_ttft = None
last_task_duration = None
was_processing = False
idle_started_at = None

log_event(
    "monitor_started",
    url=URL,
    interval=INTERVAL,
    response_end_grace=RESPONSE_END_GRACE,
)
print(
    "Connected. TTFT = first token; response = through the last token "
    "(Ctrl+C to exit)."
)

try:
    while True:
        if os.getppid() != SSH_PARENT_PID:
            sys.exit(0)
        try:
            request = urllib.request.Request(URL)
            if API_KEY:
                request.add_header("Authorization", f"Bearer {API_KEY}")
            with urllib.request.urlopen(request, timeout=2) as response:
                slots = json.load(response)

            active = next(
                (
                    slot
                    for slot in slots
                    if isinstance(slot, dict) and slot.get("is_processing")
                ),
                None,
            )

            now = time.monotonic()

            log_event(
                "slots_polled",
                slots=[
                    {
                        "id": slot.get("id"),
                        "id_task": slot.get("id_task"),
                        "is_processing": bool(slot.get("is_processing")),
                        "state": slot.get("state"),
                        "n_prompt_tokens": slot.get("n_prompt_tokens"),
                        "n_prompt_tokens_processed": slot.get(
                            "n_prompt_tokens_processed"
                        ),
                        "n_prompt_tokens_cache": slot.get("n_prompt_tokens_cache"),
                        "n_decoded": decoded_tokens(slot),
                    }
                    for slot in slots
                    if isinstance(slot, dict)
                ],
            )

            if active is None:
                if was_processing and task_started_at is not None:
                    if idle_started_at is None:
                        idle_started_at = now
                        log_event(
                            "response_end_pending",
                            task=previous_task,
                            grace=RESPONSE_END_GRACE,
                        )
                    idle_duration = now - idle_started_at
                    if idle_duration < RESPONSE_END_GRACE:
                        task_duration = now - task_started_at
                        print(
                            f"\rInternal transition | "
                            f"response in progress≈{task_duration:.2f}s | "
                            f"confirming end {idle_duration:.2f}/"
                            f"{RESPONSE_END_GRACE:.2f}s    ",
                            end="",
                            flush=True,
                        )
                        previous_tokens = None
                        previous_time = None
                        time.sleep(INTERVAL)
                        continue

                    last_task_duration = idle_started_at - task_started_at
                    log_event(
                        "response_finished",
                        task=previous_task,
                        duration=round(last_task_duration, 3),
                        ttft=None if task_ttft is None else round(task_ttft, 3),
                    )
                idle_started_at = None
                was_processing = False
                task_started_at = None

                idle_slot = next(
                    (slot for slot in slots if isinstance(slot, dict)),
                    None,
                )
                if idle_slot is not None:
                    prompt_minimum = int(idle_slot.get("n_prompt_tokens", 0) or 0)
                    capacity = int(idle_slot.get("n_ctx", 0) or 0)
                    if last_context_used is None:
                        last_context_used = prompt_minimum
                        last_context_capacity = capacity
                    status = context_status(
                        max(last_context_used, prompt_minimum),
                        last_context_capacity or capacity,
                    )
                    waiting = f"Waiting | last {status}"
                    if last_ttft is not None:
                        waiting += f" | last TTFT≈{last_ttft:.2f}s"
                    if last_task_duration is not None:
                        waiting += (
                            f" | last complete response≈{last_task_duration:.2f}s"
                        )
                else:
                    waiting = "Waiting | no slots available"
                print(
                    f"\r{waiting:<180}",
                    end="",
                    flush=True,
                )
                previous_tokens = None
                previous_time = None
                previous_task = None
            else:
                task = active.get("id_task")
                total = decoded_tokens(active)
                prompt_total = active.get("n_prompt_tokens", 0)
                prompt_done = active.get("n_prompt_tokens_processed", 0)
                prompt_cached = active.get("n_prompt_tokens_cache", 0)
                context_capacity = int(active.get("n_ctx", 0) or 0)
                context_used = int(prompt_total or 0)
                last_context_used = context_used
                last_context_capacity = context_capacity

                if idle_started_at is not None:
                    log_event(
                        "response_continued",
                        previous_task=previous_task,
                        task=task,
                        idle_duration=round(now - idle_started_at, 3),
                    )
                    idle_started_at = None

                if not was_processing:
                    task_started_at = now
                    task_ttft = None
                    task_ttft_missed = total > 0
                    log_event(
                        "response_started",
                        slot=active.get("id"),
                        task=task,
                        decoded_at_detection=total,
                    )
                elif task != previous_task:
                    log_event(
                        "task_id_changed",
                        slot=active.get("id"),
                        previous_task=previous_task,
                        task=task,
                    )
                was_processing = True

                task_duration = now - task_started_at

                if (
                    total > 0
                    and task_ttft is None
                    and not task_ttft_missed
                    and task_started_at is not None
                ):
                    task_ttft = now - task_started_at
                    last_ttft = task_ttft

                if task_ttft is not None:
                    ttft_status = f"TTFT≈{task_ttft:.2f}s"
                elif task_ttft_missed:
                    ttft_status = "TTFT not observed"
                else:
                    waiting_ttft = now - task_started_at
                    ttft_status = f"Waiting for TTFT {waiting_ttft:.2f}s"

                same_task = (
                    task == previous_task
                    and previous_tokens is not None
                    and previous_time is not None
                    and total >= previous_tokens
                )

                speed = 0.0
                if same_task:
                    elapsed = now - previous_time
                    speed = (total - previous_tokens) / elapsed if elapsed else 0.0

                print(
                    f"\rSlot {active.get('id', '?')} | "
                    f"{context_status(context_used, context_capacity)} | "
                    f"processed prompt {prompt_done} "
                    f"(cached {prompt_cached}) | "
                    f"{ttft_status} | "
                    f"response in progress≈{task_duration:.2f}s | "
                    f"generated {total:5d} | "
                    f"{speed:6.2f} tok/s    ",
                    end="",
                    flush=True,
                )

                previous_tokens = total
                previous_time = now
                previous_task = task

        except Exception as error:
            log_event("poll_error", error=repr(error))
            print(f"\rError querying llama.cpp: {error}          ", end="", flush=True)

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    log_event("monitor_stopped", reason="keyboard_interrupt")
    print("\nMonitor stopped.")
    sys.exit(0)
PY
