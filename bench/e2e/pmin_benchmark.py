#!/usr/bin/env python3
"""A/B/C runtime-only MTP benchmark using the installed Tryton replay."""
import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/ubuntu")
CONTAINER = "model-serving"
OWNER = "tryton-pmin-benchmark"


class DockerConnection(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=180)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


def docker_api(method, path, data=None):
    connection = DockerConnection()
    try:
        connection.request(method, path,
                           body=None if data is None else json.dumps(data),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read()
        if response.status >= 300:
            raise RuntimeError(f"Docker {method} {path}: HTTP {response.status}: {body.decode()}")
        return json.loads(body) if body else None
    finally:
        connection.close()


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def inspect(name):
    return json.loads(subprocess.check_output(["docker", "inspect", name]))[0]


def flag(command, name):
    return command[command.index(name) + 1]


def authentication(config):
    env = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    command = config.get("Cmd") or []
    key = env.get("MODEL_SERVING_API_KEY") or env.get("LLAMA_API_KEY")
    if "--api-key" in command:
        key = flag(command, "--api-key")
    return {"Authorization": "Bearer " + key} if key else {}


def wait_ready(config, deadline_seconds=180):
    deadline = time.monotonic() + deadline_seconds
    headers = authentication(config)
    while time.monotonic() < deadline:
        state = inspect(CONTAINER)["State"]
        if not state["Running"]:
            raise RuntimeError(f"Container stopped: exit={state['ExitCode']}")
        try:
            request = urllib.request.Request("http://127.0.0.1:8000/health", headers=headers)
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise RuntimeError("Server did not become ready in 180 seconds")


def canonical_response(body):
    choices = []
    for choice in body["choices"]:
        message = choice.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append({"type": call.get("type", "function"),
                          "name": function["name"], "arguments": arguments})
        choices.append({"index": choice.get("index", 0),
                        "finish_reason": choice.get("finish_reason"),
                        "role": message.get("role", "assistant"),
                        "content": message.get("content") or "",
                        "reasoning_content": message.get("reasoning_content") or "",
                        "refusal": message.get("refusal"), "tool_calls": calls})
    return choices


def hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def enrich(output):
    replay = json.loads((output / "replay.json").read_text())
    log = (output / "model-serving.log").read_text()
    prompt = re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", log)
    decode = re.findall(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens", log)
    draft = re.findall(r"draft acceptance =\s*([\d.]+)\s*\(\s*(\d+) accepted /\s*(\d+) generated\),\s*mean len =\s*([\d.]+)", log)
    round_lines = re.findall(r"round cost by draft width = (.*)", log)
    if any(len(values) != 15 for values in [prompt, decode, draft, round_lines]):
        raise RuntimeError(f"Expected 15 isolated timing records: {list(map(len, [prompt, decode, draft, round_lines]))}")
    rows = []
    for index, result in enumerate(replay["requests"]):
        body = json.loads((output / "responses" / f"{index + 1:02d}.json").read_text())
        normalized = canonical_response(body)
        (output / "responses" / f"{index + 1:02d}.normalized.json").write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True))
        widths = []
        for match in re.finditer(r"w(\d+): (\d+) rounds, ([\d.]+) tok/round, ([\d.]+) ms/round \(draft ([\d.]+), verify ([\d.]+), other ([\d.]+)\)", round_lines[index]):
            w, n, tokens, ms, d, v, o = match.groups()
            widths.append({"width": int(w), "rounds": int(n), "tokens_per_round": float(tokens),
                           "ms_per_round": float(ms), "draft_ms": float(d),
                           "verify_ms": float(v), "other_ms": float(o)})
        if not widths:
            raise RuntimeError(f"Missing round cost data for request {index + 1}")
        row = {"request": index + 1, "e2e_seconds": result["elapsed_seconds"],
               "status": result["status"], "prefill_seconds": float(prompt[index][0]) / 1000,
               "prompt_tokens": int(prompt[index][1]), "decode_seconds": float(decode[index][0]) / 1000,
               "generated_tokens": int(decode[index][1]), "acceptance": float(draft[index][0]),
               "accepted": int(draft[index][1]), "draft_generated": int(draft[index][2]),
               "mean_draft_len_logged": float(draft[index][3]), "widths": widths,
               "rounds": sum(w["rounds"] for w in widths),
               "normalized_sha256": hash_json(normalized),
               "functional_sha256": hash_json([{k: v for k, v in c.items() if k != "reasoning_content"}
                                                for c in normalized]),
               "usage": body.get("usage"), "timings": body.get("timings")}
        for name in ["draft", "verify", "other"]:
            row[name + "_seconds"] = sum(w["rounds"] * w[name + "_ms"] for w in widths) / 1000
        rows.append(row)
    result = {"e2e_seconds": replay["total_seconds"], "requests": rows,
              "successful": replay["successful_requests"], "count": replay["request_count"],
              "totals": {key: sum(r[key] for r in rows) for key in
                         ["prefill_seconds", "decode_seconds", "prompt_tokens", "generated_tokens",
                          "rounds", "draft_seconds", "verify_seconds", "other_seconds", "accepted", "draft_generated"]}}
    (output / "detailed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--p-min", nargs="+", choices=["0", "0.35", "0.55"],
                        default=["0", "0.35", "0.55"])
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override target top-k only; retain the draft sampler")
    args = parser.parse_args()
    os.umask(0o077)
    lock = open("/tmp/tryton-pmin-benchmark.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = ROOT / "tryton-replay" / "pmin-ab" / stamp
    destination.mkdir(parents=True)
    reference = inspect(CONTAINER)
    config = reference["Config"]
    command = config.get("Cmd") or []
    expected = {"--spec-draft-n-max": "7", "--spec-draft-p-min": "0",
                "--cache-type-k": "q8_0", "--cache-type-v": "turbo3",
                "--spec-draft-type-k": "q8_0", "--spec-draft-type-v": "turbo3",
                "--reasoning": "on", "--reasoning-budget": "512", "--parallel": "1"}
    for name, value in expected.items():
        if flag(command, name) != value:
            raise RuntimeError(f"Reference flag {name} differs from agreed configuration")
    capture = ROOT / "tryton-replay" / "inference-requests.jsonl"
    records = [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]
    if len(records) != 15:
        raise RuntimeError(f"Expected fixed 15-request replay, got {len(records)}")
    import base64
    if any(json.loads(base64.b64decode(r["body_base64"])).get("stream", False) for r in records):
        raise RuntimeError("This capture wrapper expects the actual non-streaming replay")
    if any(m.get("Type") == "volume" for m in reference["Mounts"]):
        raise RuntimeError("Anonymous/named volume clone needs explicit mapping")
    slots_request = urllib.request.Request("http://127.0.0.1:8000/slots", headers=authentication(config))
    with urllib.request.urlopen(slots_request, timeout=5) as response:
        slots = json.load(response)
    if any(slot.get("is_processing") for slot in slots):
        raise RuntimeError("Reference is currently processing a request")
    benchmark = (ROOT / "replay-benchmark.sh").read_text()
    old = '"$ROOT/replay-tryton.py" \\\n'
    new = 'python3 "$ROOT/capture-tryton-replay.py" --response-dir "$OUT/responses" \\\n'
    if benchmark.count(old) != 1:
        raise RuntimeError("Installed benchmark replay entrypoint differs")
    benchmark = benchmark.replace(old, new)
    (destination / "reference-inspect.json").write_text(json.dumps(reference, indent=2))
    manifest = {"reference_id": reference["Id"], "image_id": reference["Image"],
                "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                "blocks": args.blocks, "gdn_cols": "8", "gdn_prefetch": "off",
                "n_max": 7, "p_min": args.p_min, "target_top_k_override": args.top_k, "results": []}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    api_version = docker_api("GET", "/version")["ApiVersion"]
    original_name = "model-serving-pmin-reference-" + stamp
    moved = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(f"RESULTS_ROOT={destination}", flush=True)
    try:
        run(["docker", "stop", CONTAINER], stdout=subprocess.DEVNULL)
        run(["docker", "rename", CONTAINER, original_name])
        moved = True
        orders = [("0", "0.35", "0.55"), ("0.55", "0.35", "0"),
                  ("0.35", "0", "0.55"), ("0.55", "0", "0.35")]
        for block in range(args.blocks):
            for p_min in orders[block % len(orders)]:
                if p_min not in args.p_min:
                    continue
                label = f"pmin-c8-{p_min.replace('.', '')}-b{block + 1}-{stamp}"
                if args.top_k is not None:
                    label += f"-target-tk{args.top_k}"
                create = copy.deepcopy(config)
                create["Image"] = reference["Image"]
                create["Env"] = [e for e in config.get("Env", []) if not e.startswith(
                    ("GGML_CUDA_SM86_GDN_COLS=", "GGML_CUDA_SM86_GDN_PREFETCH="))]
                create["Env"].append("GGML_CUDA_SM86_GDN_COLS=8")
                create["Cmd"][create["Cmd"].index("--spec-draft-p-min") + 1] = p_min
                if args.top_k is not None:
                    if "--top-k" in create["Cmd"]:
                        create["Cmd"][create["Cmd"].index("--top-k") + 1] = str(args.top_k)
                    else:
                        create["Cmd"].extend(["--top-k", str(args.top_k)])
                create["Labels"] = {**(create.get("Labels") or {}), OWNER: stamp}
                create["HostConfig"] = copy.deepcopy(reference["HostConfig"])
                docker_api("POST", f"/v{api_version}/containers/create?name={CONTAINER}", create)
                run(["docker", "start", CONTAINER], stdout=subprocess.DEVNULL)
                wait_ready(config)
                current = inspect(CONTAINER)
                if current["Image"] != reference["Image"]:
                    raise RuntimeError("Variant image is not identical to reference")
                print(f"RUN block={block + 1} p_min={p_min} label={label}", flush=True)
                # No warmup: every arm starts with fresh slot, RAM cache, shortlist and graphs.
                before = set((ROOT / "tryton-replay" / "results").iterdir())
                with (destination / f"{label}.stdout.log").open("w") as log:
                    run(["bash", "-c", benchmark, "replay-benchmark.sh", label], stdout=log,
                        stderr=subprocess.STDOUT)
                after = set((ROOT / "tryton-replay" / "results").iterdir()) - before
                outputs = [p for p in after if p.name.endswith("_" + label)]
                if len(outputs) != 1:
                    raise RuntimeError(f"Expected one replay result directory, got {outputs}")
                output = outputs[0]
                detailed = enrich(output)
                entry = {"block": block + 1, "p_min": p_min, "path": str(output),
                         "e2e_seconds": detailed["e2e_seconds"], "totals": detailed["totals"],
                         "normalized_sha256": [r["normalized_sha256"] for r in detailed["requests"]],
                         "functional_sha256": [r["functional_sha256"] for r in detailed["requests"]]}
                manifest["results"].append(entry)
                (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
                print(f"DONE block={block + 1} p_min={p_min} E2E={detailed['e2e_seconds']:.3f} "
                      f"generated={detailed['totals']['generated_tokens']} rounds={detailed['totals']['rounds']}", flush=True)
                docker_api("DELETE", f"/v{api_version}/containers/{current['Id']}?force=true")
                time.sleep(args.cooldown)
        manifest["completed"] = True
    finally:
        if moved:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            found = json.loads(subprocess.check_output(["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{json .}}"], text=True).strip() or "null")
            if found:
                temporary = inspect(CONTAINER)
                if temporary["Config"].get("Labels", {}).get(OWNER) != stamp:
                    raise RuntimeError("Refusing to remove an unrelated model-serving; reference remains preserved")
                docker_api("DELETE", f"/v{api_version}/containers/{temporary['Id']}?force=true")
            run(["docker", "rename", original_name, CONTAINER])
            if reference["State"]["Running"]:
                run(["docker", "start", CONTAINER], stdout=subprocess.DEVNULL)
                wait_ready(config)
            restored = inspect(CONTAINER)
            if restored["Id"] != reference["Id"] or restored["Config"] != reference["Config"]:
                raise RuntimeError("Restored reference does not match original container")
            manifest["restored_reference_id"] = restored["Id"]
            manifest["restored"] = True
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"RESTORED_REFERENCE={restored['Id']}", flush=True)
    print(f"COMPLETED_RESULTS={destination}", flush=True)


if __name__ == "__main__":
    main()
