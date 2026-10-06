#!/usr/bin/env python3
"""Replay the frozen history; stream only request15 to measure output phases."""
import argparse
import base64
import json
import os
import runpy
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/home/ubuntu/tryton-replay/inference-requests.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-final", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(records) != 15:
        raise RuntimeError("Expected the fixed15-request capture")
    replay = runpy.run_path("/home/ubuntu/replay-tryton.py")
    canonical = runpy.run_path("/home/ubuntu/test-pmin-tryton.py")["canonical_response"]
    server = "http://127.0.0.1:8000"
    key = replay["discover_api_key"]()
    prior = []
    whole_started = time.perf_counter()
    for index, record in enumerate(records[:-1], 1):
        result = replay["replay"](record, server, key)
        if result["status"] != 200:
            raise RuntimeError(f"Request{index} failed: {result['status']}")
        prior.append(result)
        print(f"WARM_HISTORY {index}/14 E2E={result['elapsed_seconds']:.3f}", flush=True)
    record = records[-1]
    payload = json.loads(base64.b64decode(record["body_base64"]))
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(server + record["path"], data=json.dumps(payload).encode(),
                                     headers=headers, method=record.get("method", "POST"))
    events = []
    reasoning = []
    content = []
    first_reasoning = first_content = done = None
    finish = None
    usage = timings = None
    role = "assistant"
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        if response.status != 200:
            raise RuntimeError(f"Final streaming request failed: {response.status}")
        while True:
            line = response.readline()
            elapsed = time.perf_counter() - started
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = elapsed
                break
            event = json.loads(data)
            events.append({"elapsed_seconds": elapsed, "data": event})
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                role = delta.get("role") or role
                if delta.get("tool_calls"):
                    raise RuntimeError("Final request unexpectedly emitted toolcalls; compare separately")
                if delta.get("reasoning_content"):
                    if first_reasoning is None:
                        first_reasoning = elapsed
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    if first_content is None:
                        first_content = elapsed
                    content.append(delta["content"])
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - started
    if done is None or first_reasoning is None or first_content is None or finish != "stop":
        raise RuntimeError("Incomplete stream or absent reasoning/final response boundary")
    reconstructed = {"choices": [{"index": 0, "finish_reason": finish, "message": {
        "role": role, "content": "".join(content), "reasoning_content": "".join(reasoning)}}],
        "usage": usage, "timings": timings}
    expected = json.loads(args.expected_final.read_text())
    equivalent = canonical(reconstructed) == canonical(expected)
    result = {"total_replay_seconds": time.perf_counter() - whole_started,
              "final_request_seconds": elapsed, "final_stream_complete_seconds": done,
              "first_reasoning_delta_seconds": first_reasoning,
              "first_visible_delta_seconds": first_content,
              "reasoning_stream_phase_seconds": first_content - first_reasoning,
              "visible_stream_phase_seconds": done - first_content,
              "normalized_final_equal_baseline": equivalent,
              "prior_requests": prior, "usage": usage, "timings": timings,
              "method": "First14 original raw requests; final changes only stream transport/include_usage. Times are arrival boundaries, not exact GPU token timestamps."}
    (args.output / "final-response.json").write_text(json.dumps(reconstructed, indent=2, ensure_ascii=False))
    (args.output / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    (args.output / "trace.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ["prior_requests", "usage", "timings"]}, indent=2), flush=True)
    if not equivalent:
        raise RuntimeError("Streaming final differs from captured baseline; phase times are not equivalent-work timing")


if __name__ == "__main__":
    main()
