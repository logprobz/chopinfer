# Performance

How to measure, profile and submit a speed change. Accuracy comes first: a
change that can alter served numerics follows the
[engine rules](numerics.md#engine-rules) and their
[checklist](numerics.md#a-speedup-that-changes-bits) before any timing
counts. Which evidence a contributor produces and which the maintainers
produce is in [contribution tiers](contributing.md#contribution-tiers).

Timings are evidence only on otherwise idle GPUs and an otherwise idle host.
CPU results, skipped GPU tests and measurements on a shared device are not
performance evidence.

## Hardware

Chopinfer is tested only on the RTX PRO 6000 Blackwell 96 GB (Max-Q),
compute capability 12.0 with 188 SMs. The host refuses any GPU that is not
compute capability 12.0 (`runtime-cuda`), so SM121 parts such as DGX Spark
do not run. The kernels and all-reduce protocols are tuned for 188 SMs
(`SM_COUNT` in `kernels_sm120::gemm` and `kernels_sm120::moe`, and
`kernels_sm120::device::TARGET`), and packages plan each rank against 94 GiB
of device memory (`model_compiler_sdk::device::DEVICE_BYTES`). Other
compute capability 12.0 parts, such as the RTX 5090, are untested: a
package may not fit their memory, and a bundle's comm residency check may
refuse them at create.

## Measure a kernel

Kernel benchmarks are opt-in `kernel-lab` tests. Set
`CHOPINFER_KERNEL_BENCH=1` and select an idle GPU with
`CHOPINFER_KERNEL_GPU` (or several with `CHOPINFER_KERNEL_GPUS`):

```sh
CHOPINFER_KERNEL_BENCH=1 CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release \
    --test comm_bench --locked --offline -- --nocapture
```

The [kernel-lab README](../crates/kernel-lab/README.md#benchmarks) lists the
benchmarks and their selection variables; each engine's README lists its own
switches ([GEMM](../crates/kernels-sm120/cuda/gemm/README.md),
[MoE](../crates/kernels-sm120/cuda/moe/README.md),
[sparse attention](../crates/kernels-sm120/cuda/attention/SPARSE.md)).
Without a GPU selection a benchmark prints a skip and Cargo reports it as
passed; add `CHOPINFER_KERNEL_REQUIRE=1` so a skip fails instead.

The timing helpers live in `kernel-harness` (re-exported by kernel-lab):

- `time` gates the stream, evicts L2 before each iteration (a read sweep by
  default) and returns the median and minimum of at least three measured
  rounds. Each round records SM and memory clocks, clock event reasons,
  power and temperature, and flags power or thermal throttling and an SM
  clock more than 5% from the session's opening clock. Flagged rounds stay
  in the statistics; review them before any claim.
- `time_ab` compares two arms in one session: it evicts before each arm,
  alternates A/B and B/A each round, and returns each arm's median and the
  median of the paired B/A ratios. Treat a paired ratio more than 2% from 1
  as a material difference, after checking the flags.
- CUDA event timing on this GPU quantizes near 2.048 us. Differences below
  2 us on small kernels are not resolvable and are not improvements.

Generate a `kernel-lab probe` receipt before benchmarking. It records the
device's measured ceilings (DRAM and L2 bandwidth, MMA throughput per
format, PDL overlap, peer latency), which rooflines read in preference to
nominal values:

```sh
CHOPINFER_KERNEL_GPU=0 cargo run -p kernel-lab --release --locked --offline -- probe
```

Receipts go to `$CHOPINFER_KERNEL_RECEIPTS`, else `$CHOPINFER_DATA/kernel-lab`,
never the source tree. There is no kernel-level baseline table: compare a
change against its base build on the same GPU, with `time_ab` or with the
same benchmark run on both builds.

## Measure serving

The published serving baseline is the
[measured speed snapshot](serving-profiles.md#measured-speed)
(`v0.1.0-alpha.2`, 2026-10-03, four RTX PRO 6000 Blackwell Max-Q, graphics
clocks locked at 1,600 MHz and power limited to 250 W per GPU). No single
command reproduces it: its decode rates are medians of three passes of the
`spec_bench` chat cohort, and its 8K prefill at concurrency 8 has no tool.
The commands below come close; treat their results as approximate and
compare only runs made the same way on the same machine.

Set the same clocks and power first (root required; `nvidia-smi -rgc`
resets the clocks):

```sh
sudo nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1600,1600
sudo nvidia-smi -i 0,1,2,3 --power-limit=250
```

Serve one profile per model with the [quickstart helper](serving-profiles.md#quickstart-helper),
on an otherwise idle host:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
tools/quickstart.sh mimo26 tp4-dflash "$CHOPINFER_DATA" --listen 127.0.0.1:8101
# or: tools/quickstart.sh qwen38 tp4-mtp ...; tools/quickstart.sh glm5next tp4-mtp ...
```

Then, from another shell, run the decode cohorts three times (a new run ID
each pass keeps the prompts out of the prefix cache) and one 8K prefill:

```sh
export CHOPINFER_QUAL_CONFIG=models/mimo26/qual.json   # the served model's
for pass in 1 2 3; do
    python3 -B tools/qual/spec_bench.py --arm chopinfer-spec --speculation adaptive \
        --url http://127.0.0.1:8101 --run-id "speed-$(date +%Y%m%d)-$pass" --concurrencies 1 16 32
done
python3 -B tools/qual/bench.py --url http://127.0.0.1:8101 --concurrencies 1 \
    --prefill-tokens 8192 --prefill-prefix "speed-$(date +%s) "
```

Each `spec_bench.py` receipt (under `$CHOPINFER_DATA/qual/<model>`) records
every cohort's `aggregate_tokens_per_second`, the snapshot's rate:
concurrency times output tokens over the longest request, time to first
token included. Take the median of the three passes' chat cohorts per
concurrency, from receipts whose `host_load.timing` is `evidence`: the
[host load rule](../tools/qual/README.md#gsm8k-needles-and-soak), judged
over the cohorts' windows. The tool also runs a GSM8K cohort, so it needs
the frozen GSM8K inputs the model's `qual.json` declares. For an autoregressive
profile use `--arm chopinfer-ar` without `--speculation`. `bench.py`'s
prefill is one repeated-text prompt at concurrency 1; its decode figures
use another workload and are not the snapshot's. The
[load tools](../tools/qual/README.md#speculative-decoding-comparison) and the
development [benchmark loop](development-loop.md#benchmark-loop) describe
these workloads.

To compare a change with its base, use the counterbalanced
[serving A/B](../tools/ab/serving/README.md) instead: it serves both
installs in ABBA order on the same GPUs, gates on bitwise equal outputs
(or reports them under `"base_new": "report-only"` for a declared numeric
change) and reports per-phase timing, clocks and power.

## Profile

- **Engine steps.** `CHOPINFER_STEP_PROFILE_EVERY=N` samples host wall time
  per engine step ([step profiles](operations-observability.md#inspect-sampled-steps)).
  `chop_step_gpu_seconds` is absent with the native bundle executor, which
  reports no device time.
- **Serving kernels.** Capture a finite serving window with
  [Nsight Systems](operations-observability.md#capture-a-serving-window-with-nsight-systems),
  then budget each decode step's kernels from the SQLite export:

  ```sh
  python3 -B tools/profile/serving_step_nsys.py "$profile_prefix.sqlite" --device 0 \
      --json "$CHOPINFER_DATA/profiles/step-budget.json"
  ```

  A step is every kernel up to and including the next decode head GEMM,
  grouped by row bucket. Kernels overlap under PDL, so each kernel is
  charged the advance it causes in the step's latest end time, and the
  charges sum to the step span. The default `--head` pattern matches MiMo's
  and Qwen's head GEMMs (`chop_gemm_<model>_head_m<rows>_v1`); for another
  model pass a regular expression whose first group is the row count.
  The code emits no NVTX ranges, so the timeline has kernels but no
  engine phases.
- **Single kernels.** Build with `CHOPINFER_KERNEL_LINEINFO=1` for Nsight
  Compute source correlation. Profiling builds add `-lineinfo`, write
  `.profile.cubin` files and never compare or write kernel locks
  ([kernel-build](../crates/kernel-build/README.md)). Profile the kernel
  through its kernel-lab test, for example with `ncu --target-processes all`
  around the `cargo test` command, and keep reports outside the tree.

## What "faster" must show

1. **Unchanged or justified bits.** A speedup that should not change
   outputs shows bitwise equality: the affected kernel-lab tests pass under
   `CHOPINFER_KERNEL_REQUIRE=1`, and a serving A/B passes its bitwise gate.
   One that changes bits follows the
   [checklist](numerics.md#a-speedup-that-changes-bits) first.
2. **A kernel gain beyond noise.** A paired `time_ab` ratio more than 2%
   from 1, no unexplained flagged rounds, and a difference above 2 us.
3. **A serving gain where it matters.** The serving A/B's timing for every
   phase it measured, not only the improved one: a kernel retuned for one
   row bucket can slow another.
4. **The conditions.** The commit pair, profile, clocks and power, GPU
   model and driver, date, and the host's load for the measurement window
   (`uptime` and `sar -u`). Measurements on a busy host are not evidence.
