# nvcc 13.0 contract run for #1730 (MLP128) and #1731 (MT2 FP8 attention)

Requested by the maintainer: the release fatbins are built with nvcc 13.0, so that compiler gates the limit records.

**Scope.** One *combined* test binary on current main `37a4a84a` = standalone test-fixture fix (own PR) + the MLP128 landing commit + the MT2 landing commit + a local limits-only CLI hook (5 lines, not submitted). Control: untouched main `37a4a84a` + the same 5-line hook. This is not a single-contribution run.

**Toolchain.** nvcc 13.0.88 (`Build cuda_13.0.r13.0/compiler.36424714_0`) from image `codex-bench/native-tune:cu133` (`sha256:3e80f691…`), Zig 0.17.0, `-Dsm=120`, built without network. Host: RTX PRO 4500 Blackwell (SM120, 82 SMs), driver 615.78.08, graphical desktop active, 4 GiB reserve (not a headless run; the contract is deterministic and does not time anything).

## Result

```
SUMMARY 511 kernels: 511 pass gates 1-4, 0 fail, 8 known findings; 340 controls: 340 red as planted, 0 missed; 213 limit problems; gates 5503142 ms, limits 1037963 ms
```

- All 511 registered kernels pass gates 1-4, including MLP128 and both MT2 forms; 340/340 controls caught, none missed. The 8 known findings are the existing NaN-key findings (the MT2 forms inherit `tf_q35_attn_merge`'s behaviour; no fix is claimed).
- **Limit records: the three new launches pass their live checks under nvcc 13.0**, matching the records already in the PRs:

  | launch | live (13.0) | recorded |
  |---|---|---|
  | `tf_q35_attn_rows8_mt2` | 242 regs, 36992 B static, 128 threads | 242 / 36992 |
  | `tf_q35_attn_rows8_mt2_whole` | 239 regs, 36992 B static, 128 threads | 239 / 36992 |
  | `qmm_group MLP128` | 54 regs, 0 static + 26880 dynamic, 256 threads | 54 / 0 |

  The MLP128 base record therefore already equals the 13.0 compile; no record changed.
- **The 213 limit problems (168 missing static records + 45 live mismatches) are not ours.** The control with untouched main under the same nvcc 13.0 reports `LIMITS_SUMMARY: 213 problems` and the same 168 static problems; the sorted failure lists are identical except that the combined build registers 1387 launches against main's 1384 (the 3 new passing ones). They sit in `qwen4_*`, `nvfp4_experts_tile`, and the older `qwen35` attention kernels (`tf_q35_attn_partial_legacy`, `..._paged`, `..._contiguous`, `tf_q35_attn_rows`, `tf_q35_attn_partial8`, `tf_q35_attn_rows8`, `tf_q35k_compact*`).
  So the full backend gate is still red on main's own records under 13.0 as well as under 13.3. Please confirm whether CI expects something different from this entry point.

Host tests (CPU only, per branch): fmt, kernel-symbols, launch-args, attn-contract, attention-path, decode-records, formats and `compile-cuda-tests` pass. `test-qwen35-draft-trees-host` crashes 3 of 4 tests on untouched main and on both branches alike (inherited).

## Files
`logs/` gzipped full run, both limits-only runs and both build logs; `identities.txt`; `SHA256SUMS`.
Binaries: combined `bf7c1a8d6aa94c21027a01f29e76123434b8b3a32836489da96497a3c8ba5df1`, main control `babc8a11a4b91838e1f8da45668efbf20a304b4f28186acc02098204028f865e`.
Source: fixture `25fc2c7d`, MLP `389a6c2a` (tree 40e0df42), MT2 `c23a5417` (tree 413e1c06).
