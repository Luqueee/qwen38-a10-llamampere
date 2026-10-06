# Final configuration: Qwen3.8-27B on llamAmpere

This is the deployed model server on the NVIDIA A10 (24 GB): one model, one runtime, one container. The machine-readable specification is [`deploy/llamampere/model-serving.yml`](../deploy/llamampere/model-serving.yml); this page explains it and records how it is built and verified.

## At a glance

| Item | Value |
| --- | --- |
| Model | `Qwen3.8-27B-ATX-4-XS.gguf` (IQ4_XS-M quantization) |
| Model source | Hugging Face `jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF`, revision `50a84c19dc5275b7b5e8ce59d664a4e448866fcb` |
| Model SHA-256 | `5cf05ad901dcaa76f41db13a5629146ed882219339377a80e37b12a8528d963b` |
| Runtime | llamAmpere v0.4, SM86 server image `local/llamampere:v0.4-sm86` |
| Runtime source | `https://github.com/JakeATX/llamAmpere.git`, commit `a48bf5ee3c09458018559a57b1dc6215c464982a` |
| Context | 163,840 tokens total: 2 slots of 81,920 |
| KV cache | K `q8_0`, V `turbo3` |
| Speculative decoding | MTP (`draft-mtp`), up to 7 draft tokens, `p-min` 0 |
| Reasoning | On, budget 512 tokens, `deepseek` response format |
| Sampling | Greedy: temperature 0, top-k 1, top-p 1, min-p 0 |
| API | OpenAI-compatible on port 8000, bearer key required |
| Served model name | `qwen3.8-27b` |
| Container name | `model-serving` |

## Deployment order

The host is provisioned in three idempotent steps; each step is skipped when its result is already correct.

1. **Build the runtime image** from the pinned llamAmpere commit ([Runtime build](#runtime-build)).
2. **Install and verify the model assets** ([Model assets](#model-assets)).
3. **Create the container** from [`deploy/llamampere/model-serving.yml`](../deploy/llamampere/model-serving.yml). It is restarted only if step 1 or 2 changed something.

Host prerequisites: Docker with the NVIDIA container runtime, `git` (to fetch the llamAmpere source), and `python3-venv` (isolated Hugging Face client). Image builds run as root through Docker BuildKit; model files are owned by the `ubuntu` user.

## Runtime build

The image is built on the GPU host from a pinned commit and is rebuilt only when the image's `org.opencontainers.image.revision` label differs from that commit.

```bash
git clone https://github.com/JakeATX/llamAmpere.git /var/tmp/llamampere
git -C /var/tmp/llamampere checkout a48bf5ee3c09458018559a57b1dc6215c464982a

cd /var/tmp/llamampere
DOCKER_BUILDKIT=1 docker build --pull --progress=plain \
  --target server \
  --build-arg CUDA_DOCKER_ARCH=86 \
  --build-arg APP_VERSION=v0.4 \
  --build-arg APP_REVISION=a48bf5ee3c09458018559a57b1dc6215c464982a \
  --file .devops/cuda.Dockerfile \
  --tag local/llamampere:v0.4-sm86 .
```

`CUDA_DOCKER_ARCH=86` targets the A10 (Ampere, compute capability 8.6).

## Model assets

Assets live in `/home/ubuntu/models/qwen3.8-27b-atx` on the host and are mounted read-only at `/models`.

1. **GGUF.** Downloaded with the Hugging Face client (`huggingface_hub[hf_xet]`, installed in an isolated virtual environment, `HF_XET_HIGH_PERFORMANCE=1`) at the pinned revision. The download is skipped when the file already exists with the pinned SHA-256, and forced when it is missing or corrupt.
   ```bash
   hf download jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF Qwen3.8-27B-ATX-4-XS.gguf \
     --revision 50a84c19dc5275b7b5e8ce59d664a4e448866fcb \
     --local-dir /home/ubuntu/models/qwen3.8-27b-atx --force-download
   ```
2. **Draft-vocabulary map.** A text file passed to `--spec-draft-vocab-map`. It is installed next to the GGUF and verified against a pinned SHA-256. It is a host-side asset and is not part of this repository.

The deployment fails if either checksum does not match. The container is restarted only when the image, the GGUF or the vocabulary map changed.

## Server flags

Flags are grouped as in the specification.

| Group | Flags | Effect |
| --- | --- | --- |
| Model and API | `--model`, `--alias qwen3.8-27b`, `--host 0.0.0.0`, `--port 8000`, `--api-key` | Serves the GGUF under the alias `qwen3.8-27b`; every request needs the key |
| Context and slots | `--ctx-size 163840`, `--parallel 2`, `--kv-unified-per-slot 81920` | Two concurrent conversations, each with 81,920 tokens of context |
| Throughput | `--batch-size 4096`, `--ubatch-size 1024`, `--threads 8`, `--threads-batch 8`, `--n-gpu-layers 99`, `--flash-attn on`, `--fit off` | Full GPU offload with Flash Attention; automatic memory fitting disabled so the configuration is exactly what is written |
| KV cache | `--cache-type-k q8_0`, `--cache-type-v turbo3` | Quantized K cache (`q8_0`) and the llamAmpere `turbo3` V cache type; the draft model uses the same types |
| Prompt reuse | `--jinja`, `--cache-prompt`, `--cache-ram 8192`, `--ctx-checkpoints 24`, `--checkpoint-min-step 10240` | Chat template from the model, prompt cache with an 8,192 MiB host-RAM tier and context checkpoints so repeated prefixes are not recomputed |
| Reasoning | `--reasoning on`, `--reasoning-budget 512`, `--reasoning-format deepseek` | Native reasoning capped at 512 tokens per request and returned separately in `reasoning_content` |
| Sampling | `--temp 0`, `--top-k 1`, `--top-p 1`, `--min-p 0` | Deterministic greedy decoding |
| MTP speculative decoding | `--spec-type draft-mtp`, `--spec-draft-n-max 7`, `--spec-draft-p-min 0`, `--spec-draft-type-k q8_0`, `--spec-draft-type-v turbo3`, `--spec-draft-vocab-map`, `--spec-draft-vocab-hot 2048` | Drafts up to 7 tokens per round with the model's own MTP head, keeps every draft token regardless of confidence, and uses a pinned vocabulary map with 2,048 hot slots |
| Observability | `--metrics`, `--perf` | Prometheus metrics at `/metrics` and llama.cpp timing lines in the log |

### Environment

| Variable | Value | Meaning |
| --- | --- | --- |
| `GGML_Q8_TURBO3_MMA_FUSED` | `1` | llamAmpere kernel switch for the fused `q8_0`/`turbo3` MMA path |
| `GGML_Q8_TURBO3_MMA_NCOLS1_MIN` | `2` | Minimum column count for that kernel; screened over 1, 2, 4 by [`ncols1-ab.sh`](../bench/experiments/ncols1-ab.sh) |
| `GGML_CUDA_SM86_GDN_COLS` | `8` | SM86 GDN kernel column count; screened by [`gdn-runtime.sh`](../bench/experiments/gdn-runtime.sh) and [`gdn-c8-repeat.sh`](../bench/experiments/gdn-c8-repeat.sh) |

### Container

- **Ports:** `127.0.0.1:8000` on the host and `172.17.0.1:8000` (the Docker bridge gateway) so other containers on the host can reach it. Nothing is published on a public interface.
- **GPU:** all NVIDIA GPUs (`driver: nvidia`, `count: -1`), `ipc_mode: host`.
- **Health check:** `curl -f http://localhost:8000/health`, 3 retries.
- **Volumes:** the model directory, read-only.

## Why these values

Evidence from the recorded runs in [`results/experiments/decode-e2e`](../results/experiments/decode-e2e/) (15-request frozen replay, `GDN_COLS=8`, `n_max=7`; see [replay E2E](benchmarking/replay-e2e.md)):

| Choice | Evidence |
| --- | --- |
| `--spec-draft-p-min 0` | `p_min` 0.35 and 0.55 did not reduce end-to-end time (94.55 s and 93.85 s against 94.06 s) and changed 5 and 7 of the 15 normalized responses (2 and 3 of them functionally) |
| `--top-k 1` with `--temp 0` | Mean E2E fell from 94.02 s to 87.63 s and the decode of the last request from 20.48 s to 18.46 s, with all 15 normalized responses, including reasoning and tool arguments, identical across the four replays. Valid only at temperature 0 |
| `--reasoning-budget 512` | See [Reasoning behaviour](#reasoning-behaviour) |

The kernel and verifier experiments that shaped the build are summarized in [experiments](benchmarking/experiments.md#recorded-outcomes). For the investigation that led to this configuration, see the [model survey](research/a10-model-survey.md).

## Reasoning behaviour

These observations were made on an earlier deployment of Qwen3.8-27B (Q4_K_M quantization on the upstream llama.cpp image) with the same reasoning flags and a maximum output of 6,144 tokens per request. The final deployment keeps the reasoning flags but sets no server-side output limit, so clients control `max_tokens`.

- **Unlimited reasoning can leave the answer empty.** With the budget at `-1`, a real request spent all 6,144 generated tokens on reasoning (146.5 s of generation) and returned `content: null`. With the budget at 512 and the same limit, a controlled request finished with `finish_reason: stop`, separate `reasoning_content` and a non-empty answer.
- **The budget is per request**, counts toward `max_tokens`, and does not limit what the model reads or the length of the final answer. Keep `max_tokens` well above the budget.
- **`deepseek` is only a response format.** It separates `reasoning_content` from `content`; it does not change how the model reasons. The other formats mix or tag the reasoning instead.

Per-request control:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 6144
}
```

Use `chat_template_kwargs.enable_thinking: false` to switch reasoning off for mechanical calls (extraction, formatting, tool calls). On the earlier deployment `reasoning_effort: "none"` and `reasoning_budget_tokens: 0` still produced reasoning, while `enable_thinking: false` returned `reasoning_content: null`, so do not rely on `reasoning_effort`. To enable reasoning with a specific budget, send `enable_thinking: true` and `reasoning_budget_tokens` (for example 128, 256, 512 or 1024).

## Operating the server

```bash
# Health and metrics, from the host
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics

# From the local machine
ssh -L 8000:127.0.0.1:8000 <host>
curl -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8000/v1/models

# Live view: context usage, cache reuse, time to first token, tokens per second
./ops/monitor/llm-monitor.sh <host>
```

Useful endpoints: `GET /health`, `GET /metrics`, `GET /slots`, `GET /v1/models`, `POST /v1/chat/completions`. The monitor reads the API key from the container command line over SSH and logs samples to `results/telemetry/llm-monitor.log`.

Diagnosing an empty answer (`content: null`):

1. Look in the log for a generation that stops exactly at the request's `max_tokens`.
2. Check whether the request overrode `reasoning_budget_tokens` with `-1`.
3. Check whether the client enabled reasoning without a limit.
4. Check that `reasoning_content` did not absorb the whole generation.
5. Confirm the container still has `--reasoning-budget 512`.

## Not in this repository

The model weights, the draft-vocabulary map, the built image, the API key and the frozen replay requests are host-side assets. Public access, if any, is handled outside this repository.
