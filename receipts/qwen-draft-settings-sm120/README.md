# Qwen draft settings and metadata uploads on SM120

The target was Qwen3.8-27B MLX affine 4-bit with BF16 KV on a physical RTX PRO 4500 Blackwell. Every baseline and candidate used drafting. Each arm launched a fresh TensorFold 1.2.9 native server (e7ff97e132f2268c7c52bf55d1b8bdbb033462f3), dropped file caches, and compared 1,024-token inputs and 48 output IDs, greedy and sampled at seed 193. Each study held target, drafter, context (4,096) and inputs fixed. The [previous tests](../rtx-sm120/README.md) document the metadata patch, shared headless reserve overlay, driver and compiler. The numeric settings following the binary name are maximum draft-tree nodes (default 15) and, for DSpine, the draft attention window (default 8,192).

All six configurations ran forward and in reverse. Each exploratory percentage compares a candidate against its own drafter's default in the same direction. The gate required a 3% one-stream greedy throughput improvement in both directions with exact tokens to advance to independent A-B-B-A tests. Reporting a validated improvement would require three positive clock-qualified confirmation rounds. The raw clock flags are retained for each request.

| Study | Configuration | Greedy gain, forward | Greedy gain, reverse | Greedy target rounds, forward / reverse |
|---|---|---:|---:|---:|
| DFlash2, repeated text | baseline:15 | +0.00% | +0.00% | 1 / 1 |
| DFlash2, repeated text | metadata:7 | +0.02% | -0.36% | 1 / 1 |
| DFlash2, repeated text | baseline:31 | -0.72% | +1.37% | 1 / 1 |
| DFlash2, repeated text | metadata:15 | -1.23% | +1.37% | 1 / 1 |
| DFlash2, repeated text | baseline:7 | -0.80% | +1.09% | 1 / 1 |
| DFlash2, repeated text | metadata:31 | -0.68% | +1.16% | 1 / 1 |
| DFlash2, question | baseline:15 | +0.00% | +0.00% | 12 / 12 |
| DFlash2, question | metadata:7 | -8.06% | -8.00% | 14 / 14 |
| DFlash2, question | baseline:31 | -0.15% | +1.43% | 12 / 12 |
| DFlash2, question | metadata:15 | -0.27% | -5.82% | 12 / 12 |
| DFlash2, question | baseline:7 | -7.97% | -6.67% | 14 / 14 |
| DFlash2, question | metadata:31 | -0.80% | -1.91% | 12 / 12 |
| DSpine, question | baseline:15:8192 | +0.00% | +0.00% | 17 / 17 |
| DSpine, question | metadata:31:256 | -1.02% | -0.65% | 18 / 18 |
| DSpine, question | baseline:31:8192 | -8.73% | -2.87% | 18 / 18 |
| DSpine, question | baseline:15:256 | -1.79% | -3.03% | 18 / 18 |
| DSpine, question | baseline:31:256 | -1.66% | -0.52% | 18 / 18 |
| DSpine, question | metadata:15:8192 | +0.78% | +0.25% | 17 / 17 |

Each configuration matched its baseline on both returned token arrays and SHA-256. Both drafters produced identical greedy and sampled tokens on the question. The repeated-prompt workload allowed one target round for all 48 tokens, with 46 copied tokens accepted. The question follows a single slice of the public README and comes from the public TensorFold conformance profile. See [per-request records](arms.jsonl.gz) for full input arrays, outputs, token hashes, acceptance, timings, clock flags and binary identity. Run python3 verify.py to check screening results.

Model pins were TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538154e0e5b6f8dcb1fcc74a60a0798104c, z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4, and tencent/Qwen3.8-27B-DSpine-b16@ea909c0f5fab8865f9e14600334f55f7ce47308f. All screens used text completions with thinking off, prompt-cache retention disabled, and zero reused tokens. See the phase plans. Each arm harness takes the binary, target, public fixture path, output directory, timing, metadata, stream count, --drafter DRAFTER, and --host-drop as arguments. The question and DSpine harnesses take the pinned repository README as their fixture, and read that repository's conformance profile. The copy arm takes tools/bench_prompts.json. Each binary lives under a directory named for its physical target (120 or 121).
