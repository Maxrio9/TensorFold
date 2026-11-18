# Dots.mocr decode on SM121

Single-stream greedy decode gained **0.25%**. The prespecified target was 3%.

Each number uses the same 1,024 public input tokens and 48 output tokens. Positive gain means faster.

## Exploratory single-stream measurements

| Binary | Tile threshold | Forward / reverse (tokens/s) | Mean (tokens/s) |
|---|---:|---:|---:|
| baseline | 96 | 66.832 / 62.704 | 64.768 |
| baseline | 97 | 65.808 / 64.777 | 65.293 |
| baseline | 281 | 66.165 / 63.455 | 64.810 |
| greedy | 96 | 64.832 / 64.657 | 64.744 |
| greedy | 97 | 65.226 / 64.969 | 65.098 |
| greedy | 281 | 64.902 / 65.997 | 65.450 |

The selected configuration was the greedy binary with TF_QMMF_HALF_BELOW=281. The untouched baseline used 96.

## Held-out measurements

| Streams | Mode | Metric | Baseline | Combined | Median paired gain | Valid rounds | Round gains |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | greedy | decode tok/s | 64.1573 | 64.3191 | +0.25% | 3/3 | +0.25%, +0.14%, +0.42% |
| 1 | greedy | TTFT seconds | 0.0944 | 0.0939 | +0.47% | 3/3 | +0.47%, +4.40%, +0.19% |
| 1 | greedy | aggregate tok/s | 57.9102 | 58.0783 | +0.47% | 3/3 | +0.29%, +0.60%, +0.47% |
| 1 | sampled | decode tok/s | 64.4636 | 64.7056 | +0.24% | 3/3 | +0.61%, +0.24%, -0.61% |
| 1 | sampled | TTFT seconds | 0.0919 | 0.0922 | -0.23% | 3/3 | -0.34%, +0.14%, -0.23% |
| 1 | sampled | aggregate tok/s | 58.3880 | 58.5373 | +0.20% | 3/3 | +0.53%, +0.20%, -0.54% |
| 4 | greedy | decode tok/s | 52.6577 | 52.8012 | +0.56% | 3/3 | +0.56%, +0.17%, +2.02% |
| 4 | greedy | TTFT seconds | 0.2505 | 0.2512 | -0.35% | 3/3 | -0.35%, -1.28%, -0.15% |
| 4 | greedy | aggregate tok/s | 164.0361 | 164.3114 | +0.31% | 3/3 | +0.31%, -0.13%, +1.52% |
| 4 | sampled | decode tok/s | 51.6998 | 51.4807 | -0.62% | 3/3 | +2.28%, -0.62%, -0.70% |
| 4 | sampled | TTFT seconds | 0.2486 | 0.2487 | +0.01% | 3/3 | +0.01%, +0.02%, -0.22% |
| 4 | sampled | aggregate tok/s | 162.0826 | 161.5262 | -0.44% | 3/3 | +1.82%, -0.44%, -0.54% |

Confirmation rounds meeting the clock rule before the three-round cap: 1 stream(s), greedy: 3/3, 1 stream(s), sampled: 3/3, 4 stream(s), greedy: 3/3, 4 stream(s), sampled: 3/3.

## Four-stream lane details

| Mode | Lane | TTFT gain | Decode gain | Rounds |
|---|---:|---:|---:|---:|
| greedy | 0 | +52.71% | -8.54% | 3 |
| greedy | 1 | +61.89% | -10.88% | 3 |
| greedy | 2 | +18.90% | -5.47% | 3 |
| greedy | 3 | -51.87% | +19.45% | 3 |
| sampled | 0 | +0.52% | -0.96% | 3 |
| sampled | 1 | -0.28% | -0.50% | 3 |
| sampled | 2 | +26.73% | -6.56% | 3 |
| sampled | 3 | -17.15% | +5.94% | 3 |

All 196 edge requests and all timed requests matched the baseline input IDs, output IDs and token SHA-256. Drafted/plain, repeat and concurrent gates passed on both binaries. The receipt tests one checkpoint and text-only requests. Each timed arm started a new server, verified zero cached input tokens and cold file pages, and sampled device clocks at 25 ms. A material gain required three positive qualified primary rounds and a median of at least 3%.

GLM restoration and the original two container identities were checked separately after measurement.

## Reproduce this comparison

Run `python3 verify.py` from this directory to recheck the generated IDs and hashes, the six screening settings, all of the clock-qualified rounds and the paired native conformance outputs from the published measurements. `arms.jsonl.gz` contains the actual requests, generated IDs, timestamps, GPU telemetry and executable hashes. The input was a BF16 dots.mocr checkpoint on one physical NVIDIA GB10 (SM121). All six settings used TensorFold 1.2.9 at `e7ff97e132f2268c7c52bf55d1b8bdbb033462f3`. The candidate applies the [isolated greedy readback patch](../gb10-cuda/greedy.patch) and sets `TF_QMMF_HALF_BELOW=281`. All CUDA kernels are identical between the baseline and patched binaries. The tile change selects an existing kernel. The held-out primary measures text-only, single-stream, 1,024-token input and 48-token greedy output.

| Binary | MD5 | SHA-256 |
|---|---|---|
| baseline | `1add2bd538dc54bb6513ffd7c89f8176` | `42a1198769f75597d4e48c8da4529ebdbd724b7b7984e63e379eb3d99e8f66e9` |
| greedy readback | `4f9cedb3596ec4645f5ef8557c0e051c` | `c9b5435f9d0d9ab0f63cb2a73ee0d819813daea27fd2a8560e97e1e3f8bafe67` |
