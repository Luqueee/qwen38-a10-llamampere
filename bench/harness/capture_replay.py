#!/usr/bin/env python3
"""Capture response bytes while running the installed, unchanged Tryton replay."""
import argparse
import json
import runpy
import sys
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--replay-script", default="/home/ubuntu/replay-tryton.py")
    args, remaining = parser.parse_known_args()
    destination = Path(args.response_dir)
    destination.mkdir(parents=True, exist_ok=True)
    namespace = runpy.run_path(args.replay_script)
    original_replay = namespace["replay"]
    original_urlopen = urllib.request.urlopen
    position = 0

    def captured_replay(record, server, api_key):
        nonlocal position
        position += 1
        chunks = []

        class RecordedResponse:
            def __init__(self, response):
                self.response = response
                self.status = response.status

            def read(self, size=-1):
                chunk = self.response.read(size)
                if chunk:
                    chunks.append(chunk)
                return chunk

            def close(self):
                self.response.close()

        def urlopen(*values, **options):
            return RecordedResponse(original_urlopen(*values, **options))

        urllib.request.urlopen = urlopen
        try:
            result = original_replay(record, server, api_key)
        finally:
            urllib.request.urlopen = original_urlopen
        # The original replay measures request latency before this write.
        # Whole-replay E2E includes this small, identical capture cost in all arms.
        path = destination / f"{position:02d}.json"
        body = b"".join(chunks)
        path.write_bytes(body)
        result["response_file"] = str(path)
        if 200 <= result["status"] < 300:
            response = json.loads(body)
            if not response.get("choices"):
                raise RuntimeError(f"Request {position} returned no completion choices")
        return result

    namespace["main"].__globals__["replay"] = captured_replay
    sys.argv = [args.replay_script, *remaining]
    namespace["main"]()


if __name__ == "__main__":
    main()
