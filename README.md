# Qwen3.8-27B on a single NVIDIA A10 with llamAmpere

> A research project I did during my internship at [nan-tic.com](https://nan-tic.com).

I'm doing my internship (*prácticas*) at nan-tic, and this repository is the write-up of the biggest piece of work I did there: **how fast can one 24 GB NVIDIA A10 serve Qwen3.8-27B for a real, tool-calling ERP workload, and what actually makes it faster?**

It started as a vague "make the model quicker" and turned into a few weeks of measuring, being wrong, and measuring again. This repo keeps the part that survived: the final configuration, the tooling I built to trust my own numbers, and the recorded results.

**If you only read one page, read the [final configuration](docs/final-configuration.md).**

## The story in short

- **Pick a model that can do the job.** Speed means nothing if the agent makes tool errors, so I judged candidates on a real multi-step ERP task first. Qwen3.8-27B was the one that handled it reliably.
- **Make it fast.** Native MTP speculative decoding was the single biggest win. At the start I was getting around 24 tok/s with a conservative setup; the final configuration decodes at roughly 75-82 tok/s in my replay experiments.
- **Stop trusting vibes.** I froze 15 real requests into a replay harness so every change is compared on identical work, with outputs hashed to prove a faster run still says the same thing.
- **Try things, keep what pays off, write down what doesn't.** Kernel knobs, a GPU-side verifier, quantization choices, Nsight profiling. Several ideas did nothing, and I think that's worth recording as much as the wins.

## What I found

| Finding | Evidence |
| --- | --- |
| Greedy decoding with `top-k 1` cut end-to-end time on the 15-request replay from 94.02 s to 87.63 s (about 7%), with identical outputs | [`results/experiments/decode-e2e`](results/experiments/decode-e2e/) |
| A non-zero MTP `p_min` did not help (94.55 s and 93.85 s against 94.06 s) and changed some outputs | same run |
| Verifying MTP drafts on the GPU instead of the CPU was a wash (-0.74%) | [`results/experiments/verifier-ab`](results/experiments/verifier-ab/) |
| Forcing the MMQ kernel for IQ4_XS width 8 gave +1.53%, which is within what I'd call "repeat before believing" | [`results/experiments/mmvq-width8-ab`](results/experiments/mmvq-width8-ab/) |

An honest caveat: these are measurements from one machine, usually two replays per arm. They are good enough to make decisions, not to claim statistical significance, and the docs say so where it matters.

## The deployment in one paragraph

A llamAmpere v0.4 image (`local/llamampere:v0.4-sm86`, built from a pinned commit with `CUDA_DOCKER_ARCH=86`) serves `Qwen3.8-27B-ATX-4-XS.gguf` as the `model-serving` container, with two 81,920-token slots, a `q8_0` K and `turbo3` V cache, native reasoning limited to 512 tokens, greedy sampling and MTP drafting of up to 7 tokens. It listens on port 8000, bound to the host loopback and the Docker bridge, behind a bearer key. Every flag, environment variable and pin is in [`docs/final-configuration.md`](docs/final-configuration.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| [`docs/`](docs/README.md) | Final configuration, architecture, benchmarking guide, research notes |
| [`deploy/llamampere/`](deploy/llamampere/model-serving.yml) | Declarative spec of the deployed container |
| [`ops/`](ops/) | Live monitor and host backup tooling |
| [`bench/`](bench/) | Replay harness, E2E benchmarks, A/B experiments and Nsight profiling |
| [`results/`](results/README.md) | Recorded experiment and profiling output |

```text
deploy/llamampere/  model-serving.yml
ops/
  monitor/    llm-monitor.sh
  backup/     backup-host.sh  backup-host-today.sh
bench/
  harness/    replay.sh  capture_replay.py
  e2e/        pmin_benchmark.py  trace_final.py  analyze_pmin.py
  experiments/  ncols1-ab.sh  verifier-ab.sh  hybrid-verifier.sh
                iq4-width8-routing.sh  gdn-runtime.sh  gdn-c8-repeat.sh
  profiling/  nsys-profile.sh  nsys-profile-v1.sh
              nsys-profile-inside-container.sh  nsys-recover.sh
  pmin-benchmark.sh
```

## Quick start

Run everything from the repository root. The scripts drive a remote GPU host over SSH and expect an alias for it (`lambda` for the monitor, `lambdalabs` for the harness and experiments).

```bash
# Static checks (bash -n and Python syntax); never touches a remote host
make check

# Live view of the running server
./ops/monitor/llm-monitor.sh
```

## Things worth knowing

- **It was all written in Spanish first.** I wrote the scripts and notes in Spanish while working, then translated everything to English for this repo. Recorded results come from the Spanish versions, and I translated the benchmark prompts too, so a fresh run won't be directly comparable to the archived numbers (details in [`docs/benchmarking/experiments.md`](docs/benchmarking/experiments.md#reproducibility-notes)).
- **Local output is repo-relative.** Scripts find the repository root from their own location, so they work from any directory. Results land in `results/`; host backups land in `backups/` (git-ignored).
- **Remote names are a contract.** Remote paths (`/home/ubuntu/...`), the container name (`model-serving`) and uploaded file names are those of the provisioned host, even where a local file has been renamed.
- **Pinned and verified artifacts.** The runtime commit, model revision and model SHA-256 are pinned; a mismatch fails the deployment.
- **Secrets stay out of the repository.** The API key is a placeholder in the spec. Model weights, the draft-vocabulary map and the frozen replay requests live on the host and are not included.

## Where to start reading

1. [`docs/final-configuration.md`](docs/final-configuration.md): what is deployed and why.
2. [`docs/architecture.md`](docs/architecture.md): how the pieces fit together.
3. [`docs/benchmarking/experiments.md`](docs/benchmarking/experiments.md): how experiments are run and what they found.
4. [`docs/research/a10-model-survey.md`](docs/research/a10-model-survey.md): the investigation that led here (historical).
