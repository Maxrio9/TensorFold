# Dots.mocr decode on SM120

None of the five single-stream or five four-stream A-B-B-A rounds met the clock and telemetry criteria. The served speed result is **inconclusive**. All measured tokens matched.

Each number uses the same 1,024 public input tokens and 48 output tokens. Positive gain means faster.

## Exploratory single-stream measurements

| Binary | Tile threshold | Forward / reverse (tokens/s) | Mean (tokens/s) |
|---|---:|---:|---:|
| baseline | 96 | 174.392 / 175.078 | 174.735 |
| baseline | 97 | 174.165 / 173.382 | 173.774 |
| baseline | 281 | 174.601 / 173.969 | 174.285 |
| greedy | 96 | 175.243 / 175.641 | 175.442 |
| greedy | 97 | 175.314 / 173.675 | 174.494 |
| greedy | 281 | 174.452 / 173.519 | 173.986 |

The selected configuration was the greedy binary with TF_QMMF_HALF_BELOW=97. The untouched baseline used 96.

## Held-out measurements

| Streams | Mode | Metric | Baseline | Combined | Median paired gain | Valid rounds | Round gains |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 1024-0 | decode tok/s | — | — | — | 0 | — |
| 1 | 1024-0 | TTFT seconds | — | — | — | 0 | — |
| 1 | 1024-0 | aggregate tok/s | — | — | — | 0 | — |
| 1 | 1024-1 | decode tok/s | — | — | — | 0 | — |
| 1 | 1024-1 | TTFT seconds | — | — | — | 0 | — |
| 1 | 1024-1 | aggregate tok/s | — | — | — | 0 | — |
| 4 | 1024-0 | decode tok/s | — | — | — | 0 | — |
| 4 | 1024-0 | TTFT seconds | — | — | — | 0 | — |
| 4 | 1024-0 | aggregate tok/s | — | — | — | 0 | — |
| 4 | 1024-1 | decode tok/s | — | — | — | 0 | — |
| 4 | 1024-1 | TTFT seconds | — | — | — | 0 | — |
| 4 | 1024-1 | aggregate tok/s | — | — | — | 0 | — |

Confirmation rounds meeting the clock rule before the three-round cap: 1 stream(s), greedy: 0/5, 1 stream(s), sampled: 0/5, 4 stream(s), greedy: 0/5, 4 stream(s), sampled: 0/5.

## Excluded clock observations

These raw decode observations are descriptive. They do not count toward the 3% decision.

| Streams | Mode | Excluded decode gains | Reason for exclusion |
|---:|---|---|---|
| 1 | greedy | +0.15%, -0.11%, +0.33%, -0.06%, +0.24% | busy clocks differed by over 1%, fewer than three busy samples, power, thermal or hardware clock event |
| 1 | sampled | -0.22%, -0.23%, -0.27%, -0.23%, -0.43% | busy clocks differed by over 1%, fewer than three busy samples, power, thermal or hardware clock event |
| 4 | greedy | -0.41%, -0.22%, -0.43%, -0.41%, -0.31% | busy clocks differed by over 1%, fewer than three busy samples, power, thermal or hardware clock event |
| 4 | sampled | -0.39%, -0.33%, -0.51%, -0.33%, -0.46% | busy clocks differed by over 1%, power, thermal or hardware clock event |

## Four-stream lane details

| Mode | Lane | TTFT gain | Decode gain | Rounds |
|---|---:|---:|---:|---:|

All 196 edge requests and all timed requests matched the baseline input IDs, output IDs and token SHA-256. Drafted/plain, repeat and concurrent gates passed on both binaries. The receipt tests one checkpoint and text-only requests. Each timed arm started a new server, verified zero cached input tokens and cold file pages, and sampled device clocks at 25 ms. A material gain required three positive qualified primary rounds and a median of at least 3%.

The SM120 builds both used the same reviewed discrete zero-reserve compatibility overlay. Live desktop reserve was enforced separately on every arm.

## Reproduce this comparison

Run `python3 verify.py` to recheck the 196 exact request pairs, six screened settings, all 40 confirmation arms and native conformance from the published receipts. Each server used TensorFold 1.2.9 at `e7ff97e132f2268c7c52bf55d1b8bdbb033462f3`. The candidate applies the [greedy readback patch](../gb10-cuda/greedy.patch) and sets `TF_QMMF_HALF_BELOW=97`. The baseline uses 96. Both include the same headless reserve compatibility overlay. This comparison used one physical RTX PRO 4500 Blackwell (SM120) on driver 615.78.08.

| Binary | MD5 | SHA-256 |
|---|---|---|
| baseline | `b48aefcb6682c792ecffab9e9354af23` | `8332c36df3912d8bc96699707da8a34168005cc2a0bb4ab40ade66a349d5a442` |
| greedy readback | `8950ff63b608a1e2f440a13ac7e2def4` | `e9c9ffa8de53b5f2aadb4db78a5890cc6a8bc53dbe507e6216e28153202b8b4e` |
