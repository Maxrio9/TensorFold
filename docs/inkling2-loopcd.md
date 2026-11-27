# LoopCD-Hidden for Inkling 2 Mini (`--loopcd-hidden W`, CUDA, default off)

The shared `loop_round.forward` handles prompt rows, single-token decode, speculative
verification and multi-stream rounds; graph capture records this same forward. Visit
slots are private for the 48 visits. At the beginning of passes 2 and 3 the running
partial is reset to the saved prelude output and the completed-block count is reset
to one (the embedding). The previous pass's B12 and P18 supply the four cross-pass
messages for every core layer in the next pass; its controller gates those messages
into the layer's query-specific attention-residual results.

The core consists of two six-layer attention-residual blocks. At the end of each
pass, the coda's first attention sees four per-row sources: the embedding B0,
prelude output B6, that pass's B12, and its P18. Its learned query scores the RMS
normalized sources and produces a weighted retrieval. Layer 18 is a block boundary:
after its attention, P18 is also appended as B18, and its MLP and later coda layers
retrieve from that history. The final output query uses the same history plus the
coda output. There is therefore no one common coda input vector: attention, MLP,
and output use different learned queries, and later queries also see coda-produced
states. Pass 1's B12 and P18 are overwritten as later passes execute, although they
first serve as the pass-2 message bank. Pass 3 already incorporates pass-1 and
pass-2 information through those messages.

For a consistent hidden-state analogue, define h1 as the *tuple* (B12_1, P18_1)
that the coda would read after one pass, and h3 as (B12_3, P18_3). B0 and B6 are
identical in both histories. Before layer 18, mix each tuple member per row in FP32,
round once to BF16, and replace its final-pass source by h3 + W*(h3-h1). All coda
queries and the output query then see the same guided history, including B18 when
it is materialized, while the coda's own state and caches are computed once. This
is a source-space analogue, not an exact mix of attention-residual retrievals: the
retrieval applies RMS normalization, query scoring and softmax, so it is nonlinear.
An alternative is to form pass-1 and pass-3 retrievals separately for every coda
attention, MLP and output query and mix those retrievals; that incurs extra work,
requires a definition for combining the shared coda-produced history, and does not
correspond to applying one common input vector before the coda.

The guidance weight is fixed for a server process. Mini's shared prefix pages and
snapshot buffers are process-local; their contents must not be shared across servers
with different weights. The flag-off path must retain its original allocation and
kernel sequence.

## Verification

- `nvcc -fatbin -O3 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121 zig/kernels/cuda/inkling2/loop_math.cu -o zig-out/fatbin/inkling2_loop_math.fatbin`: passed.
- `zig build tf-inkling2-mini-exl3-test -Dfatbins=zig-out/fatbin -Dsm=120,121 -Doptimize=ReleaseFast -j3 --summary all`: 4/4 build steps passed.
- `zig build tf-inkling2-mini-exl3-test -Dfatbins=<sm121-images> -Dsm=121 -Doptimize=ReleaseFast -j2 --prefix <sm121-build> --summary all`: 4/4 build steps passed.
- `zig build native -Dfatbins=zig-out/fatbin -Dsm=120,121 -Doptimize=ReleaseFast -j3 --summary all`: 9/9 build steps passed.
- `zig build test-inkling2-host test-inkling2-cuda-host -Doptimize=ReleaseFast -j3 --summary all`: 118 passed, 4 skipped.
- `zig build test-qwen35-prompt-rows-host test-native-options-host -j3 --summary all`: 34 passed.
- Desktop `inkling2-loopcd-mix`: 1,554/1,554 values passed against host reference. `inkling2-mini-regressions`: 4/4 kernels and 3/3 controls passed.
- The desktop C1 gate matched 400 release tokens at W=0 on code and prose. W=0.3 changed the prose but not the code, and matched plain decoding in both samples. The prose is readable but begins with a webpage-style preamble.

The complete commands, model and server arguments, output text, token IDs, timings, acceptance counts, initial allocation refusal, and logs are in `/home/lasse/Work/runs/inkling-loopcd-20261125/gate/`.
- `python3 tools/zig/lean_check.py`: 170 findings, identical to the clean `353717987a` baseline; no new findings.

## Usage and measured effect

`tensorfold-native serve MODEL --loopcd-hidden 0.3 ...` turns it on for the whole server process; `0` (the default) runs the unchanged release path, token-identical to inkling2-mini-exl3-v2. Method: arXiv 2610.02185 (Liu, Zheng, Chen, Dilip, Bai, Jiao, Wang, Zhang).

Local evaluation, 3.3 bpw on an RTX PRO 4500, thinking high, T=1.0 (2026-11-26): at W=0.3 no difference was detected against W=0 on IFEval (541), HumanEval+ (164), MBPP+ (378), AIME25+26 (60 x 4 samples, +2.9 pp, p=0.09) or a 72-item multi-needle long-context set. IFEval-200 at W 0.15/0.5 was flat. Single-stream decode cost was -0.7% (code) and -3.4% (prose). It is experimental: treat it as a research switch, not a quality setting.
