# Research: Qwen3.8-27B on NVIDIA A10 24 GB

> **Status: historical investigation.** It records how the configuration was reached up to September 24, 2026. The deployed configuration is documented in [final-configuration.md](../final-configuration.md); where the two differ, the final configuration wins.

**Date:** September 24, 2026  
**Objective:** find a local configuration for Qwen3.8-27B that maximizes throughput on an NVIDIA A10 with 24 GB without appreciably degrading agentic/tool-calling behavior on Tryton.

---

## 1. Executive summary

After testing different versions of `llama.cpp`, MTP with different draft lengths, and several Qwen3.8-27B quantizations, the conclusions at that date were:

- **Qwen3.8-27B** was the reference for the Tryton agentic benchmark.
- For Qwen3.8, **MTP/native speculative decoding is the optimization that provides the greatest performance gain**.
- The best point found for the real workload is **`--spec-draft-n-max 8`**.
- Increasing to `n=10` produces very high peaks, even around 87 tok/s, but worsens long generations when acceptance drops.
- The older `llama.cpp` version tested, **build 10991**, performs better on this workload than **build 11151**.
- Reducing from `Q4_K_M` to `UD-Q4_K_S` or `UD-IQ4_XS` **does not increase sustained throughput much**, because MTP acceptance becomes the dominant factor.
- **UD-IQ4_XS does provide a significant memory advantage:** observed usage falls from approximately **20 GiB to 17.71 GiB of VRAM**, without a clear performance loss.
- With `UD-IQ4_XS`, **128K context appears theoretically feasible** on the A10, although it still needs to be measured in practice.
- If `UD-IQ4_XS` maintains the same agentic quality as `Q4_K_M`, it is probably the most interesting practical configuration because of the freed VRAM.

---

## 2. Hardware and environment

### GPU

- **NVIDIA A10**
- Physical VRAM: 24 GB
- Observed usable VRAM: ~22.48 GiB
- CUDA: 12.8
- Observed driver: 570.148.08

### Host

- 30 vCPU
- 200 GiB RAM
- x86_64
- Docker 28.x

### Runtime

`llama.cpp` is being used through Docker:

```text
ghcr.io/ggml-org/llama.cpp
```

Execution uses a single slot:

```text
--parallel 1
```

---

## 3. Quality target: Tryton benchmark

The primary benchmark used to evaluate agentic behavior is a real multi-step tool-calling task over the Tryton ERP mailing-message model, `marketing.email.message`. The task involves published messages (`state = 'sent'`) across two language-specific mailing lists, retrieving the messages, and semantically pairing equivalent messages across languages in a table.

### Expected correct interpretation

The task's reference to **published** messages corresponds operationally to:

```text
marketing.email.message.state = 'sent'
```

The agent must:

1. correctly detect `state = 'sent'`;
2. retrieve **38 Catalan-language messages**;
3. retrieve **38 Spanish-language messages**;
4. handle responses where `ids=[]` but `count=38`;
5. paginate when not all records are returned;
6. semantically pair the 38 messages;
7. make zero tool errors;
8. produce the complete final table.

This benchmark has proved much more useful than a synthetic benchmark for distinguishing apparently fast models from models that are genuinely reliable as agents.

---

## 4. Model under study

### Qwen3.8-27B

It is the current quality reference.

Observed strengths:

- correctly interprets `published` as `state='sent'`;
- responds correctly when a search returns `count=38` but no IDs;
- paginates correctly;
- retrieves the 38 messages in each language;
- performs the matching correctly;
- reliable agentic behavior on the benchmark task.

Initial main problem:

- decode was too slow with a conservative configuration;
- around ~24 tok/s before MTP optimization;
- ~42 tok/s with an initial MTP configuration;
- desired target: approach 70 tok/s without losing reliability.

---

## 5. Qwen3.8-27B baseline configuration

Initially used model:

```yaml
repository: Abiray/Qwen3.8-27B-Q4_K_M-GGUF

files:
  - name: Qwen3.8-27B-Q4_K_M.gguf
    sha256: 6a96cc760fc65cc6b9ea7f1922e3ff8b2a48c09f209456f23bbae9ef7032c867

  - name: mmproj-F16.gguf
    sha256: 28891162bb6e6ad6ec94edb048cfc8612ea78e76bcd5fb0d5a4208d5f90e0b5a
```

Relevant configuration:

```yaml
command:
  - --model
  - /models/Qwen3.8-27B-Q4_K_M.gguf

  - --n-gpu-layers
  - all

  - --ctx-size
  - "65536"

  - --parallel
  - "1"

  - --batch-size
  - "2048"

  - --ubatch-size
  - "512"

  - --cache-type-k
  - q8_0

  - --cache-type-v
  - q8_0

  - --flash-attn
  - "on"

  - --threads
  - "16"

  - --threads-batch
  - "30"

  - --jinja

  - --reasoning
  - "on"

  - --reasoning-budget
  - "512"

  - --reasoning-format
  - deepseek

  - --metrics
  - --perf

  - --spec-type
  - draft-mtp
```

For text-only use, avoid loading `mmproj-F16.gguf`, which takes around 928 MB on disk and provides nothing to the agent if it does not use vision.

---

## 6. MTP / speculative decoding

Qwen3.8 supports native MTP, and `llama.cpp` can use it through:

```text
--spec-type draft-mtp
```

Current `llama.cpp` documentation defines:

```text
--spec-draft-n-max N
```

as the maximum number of draft tokens per iteration.

There is also:

```text
--spec-draft-p-min P
```

as the minimum probability for speculative decoding in greedy mode.

Configuration maintained during testing:

```yaml
- --spec-type
- draft-mtp

- --spec-draft-p-min
- "0.75"

- --spec-draft-ngl
- all
```

---

## 7. MTP performance progression

### Initial baseline

Before tuning MTP:

```text
~24 tok/s
```

With initial MTP:

```text
~42 tok/s
```

---

### MTP `n=4`

Configuration:

```yaml
- --spec-draft-n-max
- "4"

- --spec-draft-p-min
- "0.75"
```

Representative result:

```text
eval           47.11 tok/s
acceptance     95.43 %
mean len       4.48
```

Conclusion:

- clear improvement over `n=2`;
- acceptance was high enough to justify expanding the draft.

---

### MTP `n=6`

Result with the older llama.cpp build:

```text
eval           53.47 tok/s
acceptance     96.70 %
mean len       6.47
```

Compared with `n=4`, the improvement was notable.

The extremely high acceptance indicated that the model was still hitting the draft-length limit.

---

### MTP `n=8`

This is the best balance found.

In a long generation:

```text
2839 tokens
64.58 tok/s
acceptance 92.26 %
mean len 7.01
```

Very fast stretches also occurred:

```text
~78 tok/s
~81 tok/s
```

when acceptance approached 97–98%.

Interpretation:

- `n=8` makes very good use of highly predictable segments;
- performance drops when acceptance falls;
- even so, it clearly outperforms `n=6` on long workloads.

Approximate comparison:

```text
n=6  -> 53.47 tok/s
n=8  -> 64.58 tok/s
```

Improvement over that comparable call:

```text
~20.8 %
```

---

### MTP `n=10`

It produces very high peaks:

```text
86.92 tok/s
86.53 tok/s
```

with:

```text
acceptance ~95-97 %
mean len   ~9.7-9.9
```

However, in long generations where acceptance drops:

```text
58.22 tok/s
82.39 % acceptance
mean len 6.54
```

and in another:

```text
59.46 tok/s
86.15 % acceptance
mean len 6.48
```

Conclusion:

```text
n=10 has better peak throughput,
but n=8 provides better sustained behavior for this workload.
```

### Current sweet spot

```yaml
- --spec-type
- draft-mtp

- --spec-draft-n-max
- "8"

- --spec-draft-p-min
- "0.75"

- --spec-draft-ngl
- all
```

---

## 8. llama.cpp versions

### Older version

```text
version: 0.4.1-dev
build:   10991
commit:  930e2fa59
```

Previously pinned image:

```text
ghcr.io/ggml-org/llama.cpp@sha256:d4bdfe78ad26a1ef3ccc834fc4e4a106d882e0f2163dd9a06e067c580c742101
```

With `n=6`:

```text
53.47 tok/s
96.70 % acceptance
mean len 6.47
```

---

### Newer version tested

```text
version: 0.5.0-dev
build:   11151
commit:  bd4f514db
```

With `n=6`:

```text
47.38 tok/s
93.85 % acceptance
mean len 5.71
```

Prefill:

```text
634.72 tok/s
```

Prefill did not worsen significantly, but decode/MTP did.

Comparison:

```text
53.47 -> 47.38 tok/s
```

Approximate regression:

```text
-11.4 %
```

Conclusion:

```text
For this specific combination of Qwen3.8 + native MTP + A10,
build 10991 performed better than build 11151.
```

Do not assume that a newer llama.cpp build will automatically be faster.
## 9. Qwen3.8 quantization research

Specific data for the Qwen3.8-27B model was sought to avoid downgrading quantization through trial and error.

### Bartowski fidelity data

The most useful metrics found compare the quantizations against BF16 using the same procedure.

| Quantization | Approx. size | Mean KLD ↓ | Top-token agreement ↑ |
|---|---:|---:|---:|
| Q4_K_M | 17.44 GB | 0.013878 | 94.996 % |
| Q4_K_S | 16.36 GB | 0.015601 | 94.773 % |
| IQ4_XS | 15.48 GB | 0.018841 | 94.141 % |
| IQ3_M | ~14.9 GB | 0.040647 | 91.055 % |
| Q3_K_L | 14.12 GB | 0.0432 | ~90.7 % |
| Q3_K_M | 13.40 GB | 0.0564 | ~89.8 % |

The key conclusion is the sharp increase in degradation when moving from the 4-bit range to the 3-bit range.

For this reason, the following order was chosen:

```text
Q4_K_M -> Q4_K_S -> IQ4_XS
```

and not to drop directly to Q3.

### External reference for agentic quality

Quesma published a comparison in which Qwen3.8-27B `Q4_K_M` closely matches the BF16 model's performance and, in their test, matches the full model on Terminal-Bench 2.1.

This reinforces the decision to consider ~4-bit quantizations the appropriate range for agents.

---

## 10. Unsloth Dynamic V3

Unsloth Dynamic quants were investigated.

Repository:

```text
unsloth/Qwen3.8-27B-GGUF
```

Currently published sizes:

```text
UD-Q4_K_M    ~16.5 GB
UD-Q4_K_S    ~15.4 GB
UD-IQ4_XS    ~14.3 GB
UD-Q3_K_XL   ~13.1 GB
```

The two variants tested were:

```text
UD-Q4_K_S
UD-IQ4_XS
```

---

## 11. UD-Q4_K_S

File:

```yaml
repository: unsloth/Qwen3.8-27B-GGUF

files:
  - name: Qwen3.8-27B-UD-Q4_K_S.gguf
    sha256: 75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017
```

Exact published size:

```text
15,358,213,024 bytes
```

Configuration:

```text
MTP n=8
p-min=0.75
```

Long-run observed results:

```text
65.29 tok/s
92.38 % acceptance
mean len 6.89
```

and:

```text
62.31 tok/s
88.00 % acceptance
mean len 6.58
```

There are also stretches around:

```text
82 tok/s
```

when MTP acceptance is very high.

Conclusion:

```text
It does not consistently increase throughput compared with Q4_K_M.
Its main advantage is reduced VRAM usage.
```

---

## 12. UD-IQ4_XS

File:

```yaml
repository: unsloth/Qwen3.8-27B-GGUF

files:
  - name: Qwen3.8-27B-UD-IQ4_XS.gguf
    sha256: 40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199
```

Exact published size:

```text
14,252,845,984 bytes
```

Configuration:

```text
MTP n=8
p-min=0.75
```

### Peaks

With very high acceptance:

```text
87.19 tok/s
98.43 % acceptance
mean len 8.60
```

and:

```text
87.98 tok/s
99.09 % acceptance
mean len 8.67
```

### Long generations

Test 1:

```text
1823 tokens
63.44 tok/s
88.38 % acceptance
mean len 6.28
```
Test 2:

```text
1716 tokens
66.35 tok/s
89.06 % acceptance
mean len 6.63
```

Practical comparison:

```text
Q4_K_M      ~64.6 tok/s
UD-Q4_K_S   ~62-65 tok/s
UD-IQ4_XS   ~63-66 tok/s
```

Performance conclusion:

```text
Reducing the weight size below Q4_K_M is not substantially increasing
the agent's sustained throughput.
```

The observed reason is that throughput is strongly correlated with:

```text
draft acceptance
mean accepted length
```

When MTP is highly accurate, any of these quants can reach >80 tok/s.  
When acceptance falls and `mean len` is around ~6, throughput returns to ~60-66 tok/s.

---

## 13. UD-IQ4_XS's real advantage: VRAM

Approximate usage previously observed with `Q4_K_M`:

```text
~20 GiB VRAM
```

With `UD-IQ4_XS`:

```text
17.71 GiB VRAM
```

Savings:

```text
~2.29 GiB
~11.5 %
```

Therefore, even without a large increase in tok/s:

```text
UD-IQ4_XS frees ~2.3 GiB without a clear throughput loss.
```

This is an important advantage on a 24 GB GPU.

---

## 14. Actual VRAM breakdown with UD-IQ4_XS

With:

```text
ctx = 65536
Main KV = q8_0
MTP enabled
```

`llama.cpp` reported:

```text
CUDA0 model buffer size       = 13061.10 MiB
CUDA0 KV buffer size          =  2176.00 MiB
CUDA0 RS buffer size          =  1346.62 MiB
CUDA0 compute buffer size     =   400.28 MiB

draft/MTP:
CUDA0 KV buffer size          =   256.00 MiB
CUDA0 compute buffer size     =   132.02 MiB
```

Approximate sum of visible GPU buffers:

```text
17372 MiB
~16.96 GiB
```

Actual usage observed by `nvidia-smi` is higher, ~17.71 GiB, due to:

- CUDA context;
- allocator;
- small reservations;
- other buffers not reflected in that `grep`.

### Approximate breakdown

| Component | VRAM |
|---|---:|
| Weights/model buffer | ~12.76 GiB |
| Main KV | ~2.13 GiB |
| RS/recurrent state | ~1.32 GiB |
| Main compute | ~0.39 GiB |
| MTP KV | ~0.25 GiB |
| MTP compute | ~0.13 GiB |

The recurrent buffer (`RS`) is surprisingly significant:

```text
~1.32 GiB
```

It is not simply KV and should not be treated as a disposable buffer.

---

## 15. MTP draft KV cache

Current `llama.cpp` documentation indicates that the defaults are:

```text
--spec-draft-type-k = f16
--spec-draft-type-v = f16
```

One could experiment with:

```yaml
- --spec-draft-type-k
- q8_0

- --spec-draft-type-v
- q8_0
```

However, the draft KV currently observed occupies only:

```text
256 MiB
```

Therefore, the potential savings would be relatively small and are not considered a priority over preserving current behavior.

---

## 16. 65K context and the possibility of 128K

Currently:

```text
ctx-size = 65536
Main KV = 2176 MiB
```

KV scales approximately linearly with context length.

Estimate:

```text
65,536 tokens   -> 2176 MiB KV
131,072 tokens  -> ~4352 MiB KV
```

Expected increase:

```text
+2176 MiB
~2.13 GiB
```

With `UD-IQ4_XS`:

```text
Current VRAM ~17.71 GiB
+ KV extra  ~ 2.13 GiB
-------------------------
estimate  ~19.84 GiB
```

On an A10 with approximately 22.48 GiB usable, this leaves a theoretical margin of about:

```text
~2.6 GiB
```

Therefore:

```text
128K context seems viable,
but this must be confirmed with a real test.
```

### 192K

An approximate extrapolation would put KV above 6 GiB and total usage too close to the VRAM limit.

This is not recommended without reducing other buffers or quantizing KV further.

---

## 17. Real issue detected: context growth

During testing, a request for:

```text
162801 tokens
```

against an available context of:

```text
65536 tokens
```

Error:

```text
request (162801 tokens) exceeds the available context size (65536 tokens)
```

This indicates that the agent or framework is accumulating an extremely large history.

Possible sources:

- preserved reasoning;
- complete tool calls and responses;
- accumulated conversation history;
- query results repeatedly forwarded.

Although increasing the context to 128K helps, the history compression/truncation policy should also be investigated.

---

## 18. Final configuration

The recommendation recorded here on September 24 was superseded by the deployed configuration, which uses the `Qwen3.8-27B-ATX-4-XS` quantization on the llamAmpere runtime with a `turbo3` V cache, two slots, `--spec-draft-n-max 7`, `--spec-draft-p-min 0` and a pinned draft vocabulary map.

The complete, current flag list, environment, build and model pins are in [final-configuration.md](../final-configuration.md).

---

## 19. What is not worth testing further for now

### `n=10`

It can reach higher peaks, but sustained performance falls when acceptance drops.

### Q3

External data show a considerable increase in divergence compared with Q4/IQ4.

For an agent sensitive to:

- tool selection;
- arguments;
- pagination;
- state interpretation;
- procedural reasoning;

seeking speed at the cost of that fidelity is not worthwhile without a strong need for memory.

### llama.cpp moving tag in production

Do not use:

```text
ghcr.io/ggml-org/llama.cpp:server-cuda
```

without pinning the digest after validation.

An update has already caused an ~11% regression in MTP decoding.

---

## 20. Recommended next experiments

Recommended order:

1. **Validate UD-IQ4_XS agentic quality** using exactly the 38 + 38 message Tryton benchmark.
2. If quality holds, keep `UD-IQ4_XS + n=8`.
3. Test `ctx-size=131072` and measure actual VRAM, not just the extrapolated value.
4. Investigate why the agent sent a request for 162K tokens.
5. Only then, experiment with `--spec-draft-p-min` around `0.75` to see whether the cost of poor drafts can be reduced.
6. Do not change several variables simultaneously: keep A/B testing.

---

## 21. Metrics to save for each test

To enable proper comparisons:

```text
model / quant
llama.cpp build + commit
ctx-size
KV type
n-max
p-min
n_gen
prompt tokens
prompt tok/s
eval tokens
eval tok/s
total time
draft accepted
draft generated
draft acceptance
mean len
VRAM peak
tool calls
tool errors
result correct/incorrect
```

Throughput alone is not enough.

For this project, the primary metric should be:

```text
correct agentic quality
+
end-to-end latency
+
throughput
```

---

## 22. Final table of key results

| Configuration | Long-run throughput | Observed peak | Long-run acceptance | Approx. VRAM | Quality |
|---|---:|---:|---:|---:|---|
| Qwen3.8 Q4_K_M unoptimized | ~24 tok/s | — | — | — | high |
| Q4_K_M + initial MTP | ~42 tok/s | — | — | ~20 GiB | high |
| Q4_K_M + n=4 | 47.11 tok/s | ~50 | 95.43 % | ~20 GiB | high |
| Q4_K_M + n=6 | 53.47 tok/s | ~56 | 96.70 % | ~20 GiB | high |
| **Q4_K_M + n=8** | **64.58 tok/s** | **~80+** | **92.26 %** | ~20 GiB | high |
| Q4_K_M + n=10 | ~58-59.5 tok/s | ~87 | ~82-86 % | ~20 GiB | high |
| UD-Q4_K_S + n=8 | ~62-65 tok/s | ~82 | ~88-92 % | lower | to be validated |
| **UD-IQ4_XS + n=8** | **~63-66 tok/s** | **~88** | **~88-89 %** | **17.71 GiB** | to be validated |

---

## 23. Conclusion

The bottleneck no longer appears to be simply the size of the weights.

Qwen3.8-27B on the A10 shows two regimes:

```text
MTP with high acceptance:
80-88 tok/s

MTP with moderate acceptance:
60-66 tok/s
```

Reducing the quantization from Q4_K_M to UD-IQ4_XS does not substantially change this relationship. The determining factor becomes how much of the MTP draft the target can accept.

Therefore, the most promising current state is:

```text
Qwen3.8-27B
UD-IQ4_XS
MTP n=8
p-min=0.75
Q8 KV
llama.cpp build 10991
```

provided the Tryton benchmark confirms that the lower quantization does not degrade agentic behavior.

Its advantage is not so much an increase in throughput as this combination:

```text
~same speed
+
~2.3 GiB less VRAM
+
realistic possibility of 128K context
```

---

## 24. External sources

### llama.cpp

Server and speculative decoding options:

https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Relevant points:

- `--spec-type draft-mtp`
- `--spec-draft-n-max`
- `--spec-draft-p-min`
- `--spec-draft-ngl`
- `--spec-draft-type-k`
- `--spec-draft-type-v`
- verbosity/logging

### Bartowski — Qwen3.8-27B GGUF

https://huggingface.co/bartowski/Qwen3.8-27B-GGUF

Used for comparison:

- sizes;
- Mean KLD;
- top-token agreement;
- degradation between Q4, IQ4, and Q3.

### Unsloth — Qwen3.8-27B GGUF

https://huggingface.co/unsloth/Qwen3.8-27B-GGUF

Files tested:

```text
Qwen3.8-27B-UD-Q4_K_S.gguf
Qwen3.8-27B-UD-IQ4_XS.gguf
```

### Quesma — Qwen3.8 quantization benchmark

https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/

Useful reference because it compares Qwen3.8-27B on quality benchmarks and shows that Q4_K_M closely matches BF16 performance, including on Terminal-Bench 2.1.

---

## 25. Notes

- Tok/s from different calls are not directly comparable if MTP acceptance or generated content changes substantially.
- The `eval time ... tokens per second` metric from `llama.cpp` is the main reference used here for decoding.
- Periodic `tg` values help observe changes during generation.
- `draft acceptance` depends heavily on the content.
- Peaks around ~88 tok/s do not mean the agent sustains 88 tok/s throughout the session.
- VRAM estimates for 128K are extrapolations. They should be verified with `nvidia-smi` and buffer logs.
- External quality results for quantization do not replace the Tryton benchmark; they help narrow the search space.

