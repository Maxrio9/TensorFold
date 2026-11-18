# Four isolated CUDA experiments on SM120

Each of the four patches matched **196 paired requests and 3,136 generated tokens** on a physical RTX PRO 4500 Blackwell. The comparison includes greedy, seeded and mixed batches over 14 prompt sizes at one and four streams. Each patch and its baseline also passed the native drafted/plain, repeat and concurrent checks. The direct FP8 test passed all ten output-byte and guard checks.

| Patch | Primary, one greedy stream | Qualified median gain | Qualified A-B-B-A rounds | Range across all observed attempts |
|---|---|---:|---:|---:|
| metadata and snapshots | decode | Inconclusive | 0/3 | -0.03% to +0.15% |
| normalization and head | TTFT | Inconclusive | 0/3 | -0.32% to +0.68% |
| FP8 KV prefill waits | TTFT | Inconclusive | 0/3 | -1.75% to -1.06% |
| greedy readback | decode | Inconclusive | 0/3 | -0.44% to +0.81% |

The range includes every attempt, including excluded rounds, and is descriptive only. The target was ≥3% median paired improvement in three positive eligible rounds. Each trial used a fresh native server, a cold file cache and identical requests. We excluded rounds with insufficient busy telemetry, active power or thermal flags, or busy SM-clock medians differing across arms by more than 1%. Clock settings and the normal power policy were left as configured.

The baseline is TensorFold 1.2.9 (`e7ff97e132f2268c7c52bf55d1b8bdbb033462f3`). Each candidate applies exactly one patch. The baseline and all candidates also share [the headless-reserve compatibility change](discrete-zero-reserve.patch). The native CUDA build used driver 615.78.08 and CUDA compiler 13.0.88. Metadata and normalization used BF16 KV on Qwen3.8-27B affine-4, the FP8 patch used FP8 KV on that same checkpoint, and greedy readback used dots.mocr BF16. Immutable checkpoint identifiers and the conformance setup are in the [original study](../gb10-cuda/README.md). The served timing requests used 1,024 input tokens and 48 output tokens, no drafting, and zero reused prompt tokens.

Run `python3 verify.py` to check input and output equality, clock decisions, and the published A-B-B-A measurements against the raw records. [Full summary](summary.json), [arm records](arms.jsonl.gz), [native conformance](conformance.json), [kernel profiles](trace-summary-120.json) and [binary digests](binary-sha256.json).
