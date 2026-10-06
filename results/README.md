# Results

Recorded output of the measurements and experiments described in [`docs/`](../docs/README.md). Treat this directory as an archive: scripts write new runs here, but nothing reads it back.

## Layout

| Path | Contents | Produced by |
| --- | --- | --- |
| `telemetry/` | Created on first run of `ops/monitor/llm-monitor.sh`: JSON-lines samples from the live monitor | `ops/monitor/llm-monitor.sh` |
| `experiments/<name>/<run-id>/` | One directory per experiment run | `bench/experiments/*.sh`, `bench/e2e/*` |
| `profiling/<name>/<run-id>/` | Nsight Systems run logs, text reports, GPU samples and replay timings | `bench/profiling/*.sh` |

`<run-id>` is the start time, `YYYYMMDD-HHMMSS`.

| Experiment directory | Script |
| --- | --- |
| `experiments/ncols1-ab/` | `bench/experiments/ncols1-ab.sh` |
| `experiments/verifier-ab/` | `bench/experiments/verifier-ab.sh` |
| `experiments/mmvq-width8-ab/` | `bench/experiments/iq4-width8-routing.sh` |
| `experiments/hybrid-verifier/` | `bench/experiments/hybrid-verifier.sh` |
| `experiments/decode-e2e/` | `bench/e2e/pmin_benchmark.py`, `trace_final.py`, `analyze_pmin.py` |
| `profiling/nsys-qwen38/` | `bench/profiling/nsys-profile.sh`, `nsys-profile-v1.sh` |
| `profiling/nsys-inside/` | `bench/profiling/nsys-profile-inside-container.sh` |

A typical experiment run directory contains `meta/` (the configuration that was run), per-arm directories with `replay.json`, `gpu.csv`, `model-serving.log`, and a `*-summary.json` / `*-summary.txt` pair with the verdict.

## What is not here

To keep the archive small and shareable, the following are not stored:

- binary Nsight traces (`.nsys-rep`, `.sqlite`) and compressed run archives;
- raw prompt and response captures from the replay;
- container configuration dumps (`docker inspect` output);
- host backups (the backup scripts write to a separate, git-ignored `backups/` directory);
- per-run `summary.json` files that embed container or file-system details. Where such a file was removed, the run directory keeps its logs and reports.

## Reading notes

- Artifacts were recorded before the project moved to its current layout, so a few recorded paths (for example `Local output :` lines in logs) point to the earlier checkout location. They are historical and not used by any script.
- Human-readable strings in these artifacts (verdict lines, status labels, log messages) were translated to English to match the scripts. Numbers, keys and structure are unchanged.
- The archived runs used the original Spanish-language benchmark prompts; see [reproducibility notes](../docs/benchmarking/experiments.md#reproducibility-notes).
- `experiments/decode-e2e/*/scripts/` is a snapshot of the E2E tooling as it was when that run was made, under its historical file names. The maintained copies are in `bench/e2e/` and `bench/harness/`.
