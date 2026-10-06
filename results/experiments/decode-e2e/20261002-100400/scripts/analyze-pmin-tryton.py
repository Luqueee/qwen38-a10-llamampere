#!/usr/bin/env python3
"""Summarize paired fixed-replay MTP trials and normalized output equivalence."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def stats(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0}


def interval95(values):
    # Student-t two-sided critical values for small paired samples.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if len(values) < 2:
        return None
    center = statistics.mean(values)
    margin = critical.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return [center - margin, center + margin]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    trials = manifest["results"]
    by_variant = {p: [r for r in trials if r["p_min"] == p] for p in ["0", "0.35", "0.55"]}
    if not by_variant["0"]:
        raise RuntimeError("No baseline trial")
    report = {"source": str(args.manifest), "restored": manifest.get("restored"),
              "completed": manifest.get("completed"), "variants": {}, "comparisons": {}}
    detailed = {r["path"]: json.loads((Path(r["path"]) / "detailed.json").read_text()) for r in trials}
    for p, group in by_variant.items():
        if not group:
            continue
        report["variants"][p] = {"runs": len(group), "e2e": stats([r["e2e_seconds"] for r in group]),
                                   "run_e2e": [r["e2e_seconds"] for r in group],
                                   "totals": {key: stats([r["totals"][key] for r in group])
                                              for key in group[0]["totals"]},
                                   "requests": {str(i): {
                                       "e2e_seconds": stats([detailed[r["path"]]["requests"][i - 1]["e2e_seconds"] for r in group]),
                                       "normalized_distinct_outputs": len({r["normalized_sha256"][i - 1] for r in group}),
                                       "functional_distinct_outputs": len({r["functional_sha256"][i - 1] for r in group})}
                                       for i in range(1, 16)}}
    controls = {r["block"]: r for r in by_variant["0"]}
    for p in ["0.35", "0.55"]:
        paired = [(controls[r["block"]], r) for r in by_variant[p] if r["block"] in controls]
        if not paired:
            continue
        gains = [a["e2e_seconds"] - b["e2e_seconds"] for a, b in paired]
        pairs = []
        for a, b in paired:
            pairs.append({"block": a["block"], "saved_seconds": a["e2e_seconds"] - b["e2e_seconds"],
                          "changed_normalized_requests": [i + 1 for i, (x, y) in enumerate(zip(a["normalized_sha256"], b["normalized_sha256"])) if x != y],
                          "changed_functional_requests": [i + 1 for i, (x, y) in enumerate(zip(a["functional_sha256"], b["functional_sha256"])) if x != y]})
        report["comparisons"][p] = {"pairs": pairs, "paired_saved_seconds": stats(gains),
                                     "paired_saved_95ci": interval95(gains),
                                     "request_saved_seconds": {str(i): statistics.mean([
                                         detailed[a["path"]]["requests"][i - 1]["e2e_seconds"] -
                                         detailed[b["path"]]["requests"][i - 1]["e2e_seconds"]
                                         for a, b in paired]) for i in range(1, 16)}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    with (args.output / "requests.csv").open("w", newline="") as stream:
        fields = ["block", "p_min", "request", "e2e_seconds", "prefill_seconds", "decode_seconds",
                  "prompt_tokens", "generated_tokens", "acceptance", "mean_draft_len_logged", "rounds",
                  "draft_seconds", "verify_seconds", "other_seconds", "normalized_sha256", "functional_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for trial in trials:
            for row in detailed[trial["path"]]["requests"]:
                writer.writerow({**row, "block": trial["block"], "p_min": trial["p_min"]})
    print("p_min    mean E2E    stdev     runs")
    for p, variant in report["variants"].items():
        print(f"{p:>5}    {variant['e2e']['mean']:8.3f}    {variant['e2e']['stdev']:6.3f}    {variant['runs']}")
    for p, comparison in report["comparisons"].items():
        print(f"p_min {p}: saved={comparison['paired_saved_seconds']['mean']:.3f}s "
              f"95CI={comparison['paired_saved_95ci']} pairs={comparison['pairs']}")
    print(f"Restored={report['restored']} Completed={report['completed']}")


if __name__ == "__main__":
    main()
