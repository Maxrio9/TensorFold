# Tile size and block order on SM120

Changing block order in addition to tile size missed the 3% incremental target. On held-out synthetic inputs, the four projections together took 1.221810 ms with the stock 64×64 tile, 0.876658 ms with the existing large prompt kernels and default order, and 0.952789 ms with the large kernels and the selected shape-specific order. Relative to stock, this is +39.37% kernel throughput for the large kernels alone and +28.24% for the combination. Adding the block-order changes reduced throughput by 7.99% relative to the large kernels alone. Times sum one median call at each of the four shapes. Served latency remains unmeasured.

The GPU was an RTX PRO 4500 Blackwell (SM120). The fatbin, matrix shapes, output precision, pinned K slices and 192 MiB rotation recipe are the same as the [first projection test](../dots-prompt-sm120/README.md). Each alternative passed five guarded whole-output byte comparisons per shape and seed against the stock calculation. The launch order rotated across trials. The exact standalone command for discovery after compiling with the CUDA driver library is below. For confirmation, use seed 503 and the selected four IDs as shown. The argument after the fatbin is the host's required reserve in GiB.

```sh
g++ -std=c++17 -O2 -I"$CUDA_HOME/include" prompt_groups.cpp -L"$CUDA_HOME/lib64/stubs" -lcuda -o prompt_groups
./prompt_groups "$QMMF_FATBIN" "$RESERVE_GIB" 299 -1 -1 -1 -1
./prompt_groups "$QMMF_FATBIN" "$RESERVE_GIB" 503 8 8 6 9
```

The output columns are projection, seed, trial, configuration ID, and milliseconds per call. ID 0 is stock. ID 1 uses the large kernels with default order. IDs 2–5 use the stock tile at row-group heights 1, 2, 4, 8, and IDs 6–9 combine the large kernels with those same four heights. [Screen](screen.txt), [separate confirmation](holdout.txt), [predeclared method](plan.txt), and [held-out medians](results.json). Aggregate device clocks and clock-event samples accompany both runs. These short CUDA-event trials do not establish clock-qualified server performance.
