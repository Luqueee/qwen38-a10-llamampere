# E2E measurement and Tryton decode on `lambdalabs`

This page documents the frozen replay measurements and final-response tracing workflow.

The local sources live in `bench/e2e/` and `bench/harness/`. They are copied by hand to the GPU host and keep their historical remote names, which the scripts reference at runtime: `pmin_benchmark.py` is installed as `/home/ubuntu/test-pmin-tryton.py`, `trace_final.py` as `/home/ubuntu/trace-final-tryton.py`, and `capture_replay.py` as `/home/ubuntu/capture-tryton-replay.py`.

`bench/e2e/pmin_benchmark.py` uses the fixed replay of 15 requests installed at `/home/ubuntu/tryton-replay/inference-requests.jsonl`. It preserves the original container, creates a new variant for each replay, and restores the reference when finished. It keeps `GDN_COLS=8`, PREFETCH disabled, and the reference prompts, tools, reasoning, image, mounts, and caches. Full responses, their normalization, and the draft/verify/other costs are saved in the results.

```bash
# One A/B/C p_min run, without repeating twelve replays.
ssh lambdalabs 'python3 /home/ubuntu/test-pmin-tryton.py --blocks 1'

# Isolated control and greedy-sampling candidate, each starting from a cold state.
ssh lambdalabs 'python3 /home/ubuntu/test-pmin-tryton.py --blocks 1 --p-min 0'
ssh lambdalabs 'python3 /home/ubuntu/test-pmin-tryton.py --blocks 1 --p-min 0 --top-k 1'
```

`--top-k` changes only the target; the draft MTP sampler backend retains its top-k 10. Top-k 1 is evaluated only at temperature 0: with a positive temperature it would change the distribution. Compare normalized outputs, including reasoning and tool arguments, before considering an improvement equivalent. Higher MTP acceptance or fewer generated tokens is not sufficient.

The measured E2E corresponds to inference calls in the frozen replay, not the ERP tool execution time. The draft/verify/other categories come from per-round histograms and are rounded; “other” includes sampling and host work and does not identify a function by itself. Token counts obtained by retokenizing parts of a response do not replace the server's total generated-token count.

To measure when reasoning ends and the visible response begins, `bench/e2e/trace_final.py` replays the first 14 requests without changing their bytes and sends the last one using SSE:

```bash
ssh lambdalabs 'python3 /home/ubuntu/trace-final-tryton.py --output /home/ubuntu/tryton-replay/final-trace/new-measurement --expected-final /path/to/reference-replay/responses/15.json'
```

The output directory must be new. Events and their arrival times are saved, and normalized equality of the final response, including reasoning, is required. These boundaries include transport and chunk size; they are not exact GPU execution timestamps. Streaming can present content earlier, but does not imply less decode work.

Measurement on 2026-10-02: the real reference had temperature 0, top-k 0, top-p 1, and min-p 0. With two replays per profile, top-k 1 on the target alone reduced mean E2E from 94.02 to 87.63 s and the decode of request 15 from 20.48 to 18.46 s. All 15 normalized responses, including reasoning and tools, were identical across the four replays. This validates that workload, not all prompts or requests with positive temperature.

The reference trace received the first reasoning at 4.60 s and the first visible content at 10.29 s; it finished at 25.15 s. The reasoning and visible-content arrival phases took 5.69 and 14.86 s. The original container was restored, without permanently applying top-k 1.
