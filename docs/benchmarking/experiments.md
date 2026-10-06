# Benchmarking experiments

All experiments run from the repository root against the `model-serving` container on the GPU host (SSH alias `lambdalabs`, overridable with `REMOTE`/`HOST`). They share the preserve-and-restore pattern described in [architecture](../architecture.md#measurement-model) and use [`bench/harness/replay.sh`](../../bench/harness/replay.sh) as the workload driver.

> **Prerequisites not in this repository.** The experiments replay a frozen set of 15 real agentic requests that must already be installed on the host (`/home/ubuntu/tryton-replay/`, created by `bench/harness/replay.sh install` and populated through its capture proxy). Model files, the llamAmpere runtime image and the draft-vocabulary shortlist are also host-side assets.

## Workload driver

`bench/harness/replay.sh` is a subcommand-style tool:

| Subcommand | Effect |
| --- | --- |
| `install` | Create the replay directory, Python environment and capture proxy on the host |
| `start` / `stop` / `reset` / `status` | Control the capture proxy and its recorded requests |
| `finalize` | Reduce the capture to the 15 inference requests |
| `replay <label>` | Replay the requests directly against `127.0.0.1:8000` and store timings, logs and GPU telemetry |
| `compare` | Compare stored replays |

The capture path is `client -> :8001 (capture proxy) -> :8000 (model-serving)`; replays bypass the proxy.

## Experiment catalogue

| Script | Question | Knob under test | Design |
| --- | --- | --- | --- |
| `bench/experiments/ncols1-ab.sh` | Does the minimum column count for the Q8 turbo MMA kernel matter? | `GGML_Q8_TURBO3_MMA_NCOLS1_MIN` = 1, 2, 4 | 3 replays per arm |
| `bench/experiments/verifier-ab.sh` | Is verifying MTP drafts on the GPU faster than on the CPU? | `LLAMA_MTP_GPU_VERIFY=greedy` | CPU arm vs GPU arm |
| `bench/experiments/iq4-width8-routing.sh` | Is the MMQ kernel faster than MMVQ for IQ4_XS at width 8? | forced MMQ routing | A/B/B/A interleave |
| `bench/experiments/hybrid-verifier.sh` | How many MTP rounds could a hybrid GPU verifier handle safely? | none (diagnostic build) | Instrumented image built on the host; per-round analysis |
| `bench/experiments/gdn-runtime.sh` | Which GDN kernel column and prefetch settings are fastest? | `GGML_CUDA_SM86_GDN_*` (cols, prefetch) | Screening of variants against a known baseline |
| `bench/experiments/gdn-c8-repeat.sh` | How reproducible is the `cols=8, prefetch=0` result? | same | Three repeats of one variant; same code as `gdn-runtime.sh` |
| `bench/e2e/pmin_benchmark.py` | Does the draft acceptance threshold `p_min`, or target `top-k 1`, reduce end-to-end time? | `--p-min`, `--top-k` | Cold-start A/B/C blocks; see [replay E2E](replay-e2e.md) |
| `bench/e2e/trace_final.py` | When does reasoning end and visible output begin? | none | Streams request 15 after replaying the first 14 |
| `bench/pmin-benchmark.sh` | Throughput versus `p_min` on a long generation | `p_min` values | Standalone benchmark container |

Profiling (Nsight Systems) lives in [`bench/profiling/`](../../bench/profiling/):

| Script | Use |
| --- | --- |
| `nsys-profile.sh` | Profile the replay through a restarted `model-serving` target |
| `nsys-profile-v1.sh` | Earlier variant that also installs Nsight Systems and samples GPU metrics at 100 ms |
| `nsys-profile-inside-container.sh` | Run the server under `nsys` inside a purpose-built container, then restore the original |
| `nsys-recover.sh` | Pull an interrupted profile back from the host |

## Recorded outcomes

The runs below are kept under [`results/experiments/`](../../results/experiments/). Each run directory holds the logs, per-arm replay timings and a summary; the verdict string comes from the script itself.

| Experiment | Result | Verdict |
| --- | --- | --- |
| GPU vs CPU verifier (`verifier-ab`) | decode 81.68 vs 82.29 tok/s (-0.74%) | `KERNEL_NEXT`: the verifier adds little; focus on IQ4_XS width 8 |
| IQ4_XS width-8 routing (`mmvq-width8-ab`) | MMQ 76.74 vs baseline 75.58 tok/s (+1.53%) | `BORDERLINE`: repeat before changing production |
| Hybrid verifier diagnostic (`hybrid-verifier`) | 902 MTP rounds, 549 (60.86%) conservatively GPU-safe | `HYBRID_PATCH_MAYBE`: 40-70% of rounds are GPU-safe |
| `p_min` and top-k 1 (`decode-e2e`) | mean E2E 94.02 s to 87.63 s with top-k 1 on the target only; all 15 normalised responses identical | See [replay E2E](replay-e2e.md) |

The `ncols1-ab` and profiling directories keep logs and environment dumps; their detailed per-run summaries were not retained.

## Running an experiment

```bash
# Run it; the script prints its configuration, swaps the container, and restores it
./bench/experiments/verifier-ab.sh

# Results appear in a new timestamped directory
ls results/experiments/verifier-ab/
```

Experiments replace the live `model-serving` container for the duration of the run and restore it afterwards. Do not run two experiments at the same time, and do not run them while the host serves traffic you care about.

## Reproducibility notes

- The recorded runs in `results/` were produced by earlier, Spanish-language versions of these scripts. Their human-readable strings were translated for consistency; measurements are unchanged.
- The English migration also translated the embedded benchmark prompts (for example the long-generation prompt in `bench/pmin-benchmark.sh`). A new run therefore produces different generated text, token counts and acceptance rates than the archived Spanish-prompt runs, and the two are not directly comparable.
