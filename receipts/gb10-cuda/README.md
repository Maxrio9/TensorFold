# Four CUDA experiments on a GB10

All four variants missed the 3% served-speed criterion set before timing. Each patch applies independently to TensorFold 1.2.9 at `e7ff97e132f2268c7c52bf55d1b8bdbb033462f3`. This branch contains the experiment record. Mainline code is unchanged.

| Patch | Primary measure, one greedy stream | Baseline | Candidate | Median paired gain |
|---|---|---:|---:|---:|
| Qwen text metadata and snapshots | decode, tokens/s | 15.2475 | 15.2418 | −0.037% |
| Qwen normalization and head | time to first token, seconds | 0.8619 | 0.8659 | −0.519% |
| Qwen FP8 KV prefill waits | time to first token, seconds | 0.8675 | 0.8642 | +0.240% |
| dots.mocr greedy readback | decode, tokens/s | 64.6319 | 65.4191 | +1.200% |

The absolute values are medians of the three round-level baseline and candidate means. The last column is the median of the three paired ratios. Positive indicates a benefit. For latency the ratio is baseline divided by candidate. For throughput it is candidate divided by baseline. Each patch matched all generated IDs and SHA-256 digests on 196 paired requests, 3,136 generated tokens per patch. Every timed output also matched. This measures served tokens. The separate FP8 device checker additionally compared raw output bytes and output guards on ten paged and contiguous shapes.

## Verify the measurements

Run `python3 verify.py` in this directory. The verifier reads `arms.jsonl.gz`, checks each timed request and token stream, and recomputes clock qualification, round selection and all three metrics. The archive contains the exact public input arrays, generated tokens, hashes, raw per-request timings and clock data for each attempted round. Sample ordinal replaces local wall-clock time. `summary.json` includes all rounds, exclusions and selections. `conformance.json` contains the drafted, repeat and concurrent checks and per-request hashes. `trace-summary-121.json` contains the Nsight copy counts and kernel launch shapes. The four text logs contain the focused device checks. Binary hashes and experiment revision IDs are in `binary-sha256.json` and `revisions.json`.

## Reproduce

One NVIDIA GB10, SM121, 128 GB shared-memory class, Ubuntu 24.04.5 LTS, NVIDIA driver 595.99.02, CUDA compiler 13.0.88 and Zig 0.17.0. One rank and one server at a time. Clocks and power settings stayed at their configured values.

Use separate clean checkouts of the baseline for each build. Apply exactly one named patch to each candidate. With `BIN` pointing to a writable directory, build into `$BIN/121/baseline` or `$BIN/121/<variant>` as appropriate. The numeric architecture directory is used by the harness to check the physical device.

```sh
zig build -j2 -Dgpu=sm_121 -Dnvcc="$(command -v nvcc)" -Dtarget=aarch64-linux-gnu.2.39 --prefix "$BIN/121/$VARIANT"
md5sum "$BIN/121/$VARIANT/native/bin/tensorfold-native" > "$BIN/121/$VARIANT/build-ok"
```

Set `QWEN` to TensorFold/Qwen3.8-27B-MLX-4bit at `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` (affine 4-bit, group 64) and `DOTS` to dots-studio/dots.mocr BF16 at `e539fbb52280393adc081b289ec597430a0f9031`. Use `PROMPTS` for the baseline's `tools/bench_prompts.json`. For Qwen drafted conformance the drafter was z-lab/Qwen3.8-27B-DFlash2 at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Dots conformance used its copy drafter.

```sh
python3 series.py --bin "$BIN/121" --qwen "$QWEN" --dots "$DOTS" --prompts "$PROMPTS" --out "$RESULTS"
```

The harness uses the native server, a 4,096-token context, disabled drafts for ordinary performance, Qwen thinking off, BF16 KV except in the FP8 pair, and cache-off or the family's native no-retention path. Four fixture rotations yield independent stream prompts. Edge checks cover 14 lengths from 1 through 1,024, with 16 output tokens, greedy, seeded sampling and mixed four-stream batches. Timing uses 1,024 input and 48 output tokens, one and four streams, greedy or temperature 0.8, top-p 0.95 and seed 193 plus lane number. Every request returned zero cached tokens. Files were evicted individually and verified nonresident before timed server startup. Whole-host cache eviction, long-context 2k/8k/32k/64k timing, peak process and pinned memory, previous-release and external-server comparisons were outside this experiment.

Each round starts four fresh servers in baseline/candidate/candidate/baseline order. Keep the first three eligible rounds per mode. Exclude throttling, fewer than three busy samples per arm or a spread greater than 1% in median busy SM clocks. The original sampler interval was 150 ms. Replacement rounds used the same workload. Only one-stream dots replacements sampled at 25 ms in both arms to capture the short requests. All eligible one-stream dots greedy rounds used 25 ms. The declared material-win rule required positive paired gains in all three eligible rounds and a median of at least 3%. Nsight profiling ran in separate processes from timing. SM120 compilation is available for these patches. This receipt covers SM121.
