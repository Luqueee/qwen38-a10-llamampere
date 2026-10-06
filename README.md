# Qwen3.8-27B on a single NVIDIA A10 with llamAmpere

Research, final configuration and measurements for serving **Qwen3.8-27B-ATX-4-XS** with the **llamAmpere** `llama.cpp` runtime and MTP speculative decoding on one NVIDIA A10 (24 GB) host, driven over SSH from a local machine.

**Start with the [final configuration](docs/final-configuration.md).**

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

## The deployment in one paragraph

A llamAmpere v0.4 image (`local/llamampere:v0.4-sm86`, built from a pinned commit with `CUDA_DOCKER_ARCH=86`) serves `Qwen3.8-27B-ATX-4-XS.gguf` with two 81,920-token slots, a `q8_0` K and `turbo3` V cache, native reasoning limited to 512 tokens, greedy sampling and MTP drafting of up to 7 tokens. It listens on port 8000, bound to the host loopback and the Docker bridge, behind a bearer key. Every flag, environment variable and pin is listed in [`docs/final-configuration.md`](docs/final-configuration.md).

## Quick start

Run everything from the repository root. Scripts expect an SSH host alias (`lambda` for the monitor, `lambdalabs` for the harness and experiments) that reaches the GPU host.

```bash
# Static checks (bash -n and Python syntax); never touches a remote host
make check

# Live view of the running server
./ops/monitor/llm-monitor.sh
```

## Conventions

- **Local output is repo-relative.** Scripts derive the repository root from their own location, so they work from any directory. Results land in `results/`; host backups land in `backups/` (git-ignored).
- **Remote names are a contract.** Remote paths (`/home/ubuntu/...`), the container name (`model-serving`) and uploaded file names are those of the provisioned host, even where a local file has been renamed.
- **Pinned and verified artifacts.** The runtime commit, model revision and model SHA-256 are pinned; a mismatch fails the deployment.
- **Secrets stay out of the repository.** The API key is a placeholder in the spec.

## Where to start reading

1. [`docs/final-configuration.md`](docs/final-configuration.md): what is deployed and why.
2. [`docs/architecture.md`](docs/architecture.md): how the pieces fit together.
3. [`docs/benchmarking/experiments.md`](docs/benchmarking/experiments.md): how experiments are run and what they found.
4. [`docs/research/a10-model-survey.md`](docs/research/a10-model-survey.md): the investigation that led here (historical).
