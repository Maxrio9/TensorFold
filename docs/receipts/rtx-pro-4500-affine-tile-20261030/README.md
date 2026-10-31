# RTX PRO 4500 affine-tile: draft qualification receipt

**Draft: the full CUDA kernel-contract SUMMARY gate is outstanding.** The queued full runs were canceled after a scope review; no result below substitutes for that gate. No installed runtime was changed.

A 128-column/eight-warp MLP affine tile for N=17408, K=5120, one K slice and 1–16 rows, gated to SM120/82 SMs. Existing GEMV, grouped projections and other shapes retain their dispatch.

Base `01e25d3aebc0ebb686670c69e37b7c269f261b3d`; candidate code `aa8ec483b005b94c20dae85a3129a71387ef5f68`. The explicit build label 1.1.6 is retained recipe/client compatibility metadata; source SHAs identify code. See [environment.json](environment.json) for exact checkpoint/drafter revisions, config/tokenizer hashes, driver, libraries and hardware. Native Zig/CUDA inference, no MLX/PyTorch serving runtime.

Normal serving settings remain 131072-token context, FP8 KV, int8 prefill, vision, BF16 DFlash2, 8 GiB host prompt cache and parallel 1. Clean measurements use upstream-supported0.5 GiB reserve in every arm; historical private 0 GiB overlay measurements are not pooled with these.

## Completed decode comparison

Stock/candidate/stock, fresh calibration every boot, identical request history, within-cell medians and a geometric mean of bracketing stock medians. One candidate boot does not establish a confidence interval, particularly for small differences. All rates are tokens/s.

Standard `tools/bench_openai.py`, greedy prose/code, 256 tokens, 3 repetitions:

| Workload | Stock before | Candidate | Stock after | Gain |
|---|---:|---:|---:|---:|
| fibonacci-raw | 214.81 | 216.36 | 214.59 | +0.774% |
| gpu-chat-no-think | 164.19 | 165.32 | 164.08 | +0.724% |

Retained community-v1 client, 256 tokens, 5 repetitions per cell:

| Workload | Stock before | Candidate | Stock after | Gain |
|---|---:|---:|---:|---:|
| code/T1 | 232.32 | 232.52 | 230.48 | +0.484% |
| chat/T1 | 144.84 | 145.00 | 143.80 | +0.468% |
| code/T0 | 216.52 | 216.90 | 215.01 | +0.526% |
| chat/T0 | 165.23 | 165.56 | 164.18 | +0.520% |

Raw client metadata that was unavailable remains unknown. Native manifest and environment sidecars provide separately observed identities; these attached diagnostics are not a leaderboard submission.

## Completed exactness, feature and host checks

Every completed stock/candidate/stock token-hash comparison matches. Drafted/plain, resumed/fresh and queued concurrent/solo outputs match. PNG/JPEG/WebP, required tools, JSON schema, observed reasoning, cancellation recovery and context reuse through 122779 input tokens pass in every serving arm. Reasoning-off speed requests remain separate from reasoning feature checks.

- Build Summary: 478/478 steps succeeded; 311/313 tests passed (2 skipped)
- Build Summary: 167/167 steps succeeded

Zig cache hits are explicit in the logs; these counts describe tests actually reported, not a sum of overlapping runs. Final formatting passes; lean reports 425 inherited findings and 0 new. Boundary/ABI/compiler-selection tests, the failing GEMV regression receipt and its passing correction are attached. A preexisting scratch-memory host fixture crash was reproduced on the untouched baseline and fixed in the test setup.

Four final sanitizer runs pass with zero errors: memcheck and synccheck for actual native 13.3 images with Compute Sanitizer 2026.2.1, and independently compiled 13.0 images with matching 2025.3.1. Each run has 744 byte comparisons plus graph replay, shuffled-page/whole attention, input-preservation and guards. Exact CUDA images were extracted from native ELF headers; identities are attached. A newer-checker/older-image attempt completed all byte checks but returned a selective-recompilation API warning; that nonzero attempt is retained separately and is not counted as clean. No warnings or checks were suppressed.

## Outstanding and partial gates

The full backend contract was **not completed**. The initial baseline attempt timed out after 1100 seconds while computing references; no SUMMARY was produced. Later full baseline/candidate runs were canceled before start to end the excessive validation scope. New kernels are registered, but full FP64/controls/SUMMARY execution remains required before ready-for-review qualification.

Cold-prefill evidence is explicitly partial. Stock and MLP completed exact 2K/8K/32K/64K/130816-token arrays with cache retention disabled and zero cached counters. Attention completed only the cases present in cold-partial/state.json; the final stock bracket was not run. No complete cold comparison or release non-regression claim is made.

A separate unmodified-baseline edge case—131056 input tokens plus 16 output tokens—returned QwenContextFull after four tokens. Its failed receipt, exact input and server log are retained. The subsequent 130816 input leaves 256 positions for output and speculative lookahead, with configured context 131072 unchanged. This does not qualify the rejected edge case.

Physical Metal/HIP/XPU/other NVIDIA devices, other checkpoints, a full 13.0 served model gate and a previous-tag/standard-server release comparison were not run. SM80/86/89 and SM121 evidence is compile-only. Physical host RAM 31.24 GiB and the 24 GiB model-test container limit are separate facts; this is not a minimum-RAM claim. Per-process and GPU memory observations are retained in served/state.json.

## Reproduction and provenance

Use Zig 0.17.0 and the pinned source/checkpoint revisions. Build with `zig build native -Dsm=120 -Dversion=1.1.6 -Dnvcc=/usr/local/cuda-13.3/bin/nvcc -j6`; provide the matching grammar/media libraries. Set TF_SRC and MODELS, then run reproduction/reproduce_serve.sh. The standard benchmark command is `python3 tools/bench_openai.py http://127.0.0.1:18080 bench --tokens 256 --reps 3 --temperatures 0 --output decode.json`, with spaces between each option and value. Reproduction workers retain exact request order and use conventional /task and /submission container mount points. Use a fresh process-local calibration HOME for each boot.

For device qualification run `zig build -Dsm=120 -Dnvcc=/usr/local/cuda-13.3/bin/nvcc`, then `zig-out/bin/tf-cuda-test kernel-contract` on an isolated/admitted GPU and attach the actual SUMMARY. This is still outstanding here.

[Coordination and supplemental overlap checks](tile-coordination.json). Every file except the manifest itself is covered by [sha256.json](sha256.json). Kernel resource records use live-driver attributes on SM120; raw cuobjdump totals and their accounting difference are retained. SM121 records do not establish a physical device qualification.

## Additional GPU testing

[Independent 5090/RTX PRO 6000 Blackwell testing instructions](testing/REQUEST.md) document the opt-in `TF_CUDA_TEST_MLP128` switch. The override was added after these measurements; defaults are unchanged and no new GPU qualification is claimed.
