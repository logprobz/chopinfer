This job profiles K3-cap serving first, then adaptive serving with the cost
refresh fix. The two server processes run sequentially. Each owns one Nsight
session, started once after readiness and stopped once after C8 and C16.
Both policies use the same sampled requests, device results and resident
cache. Token IDs and selected and top logprobs compare exactly.

Catch-up converts committed target features into drafter KV. Flash retains
BF16 attention-input features from target layers 37, 38 and 39. The shared
main projection reads their concatenation, followed by main normalization
and the trained quantizer. Each of the three drafter stages then projects
its KV, normalizes and applies RoPE, and stores the committed ring. No target
transformer, drafter attention, MoE or LM head is rerun by this operation.

Catch-up precedes draft generation because the draft reads the committed
ring. Device results do not remove that dependency. The batched path gathers
rows across requests, shares the main projection across stages, skips rows
outside the readable window and uses exact-row graphs. Earlier eager
per-stage measurements do not measure this path.

Moving that projection after the next target verify would break its
dependency on drafting. A possible separate experiment is to project
provisional target features during the preceding verify and publish only
the accepted prefix. It would need matched arithmetic, rounding, storage
lifetime and rejection tests, and it would project some rejected rows.
This job changes no catch-up implementation.

The analyzer uses `utcEpochNs` from Nsight to align engine completion
timestamps. It retains consecutive full-concurrency cycles, discards the
first four and requires at least eight. It checks target geometry, a
complete target head, drafter markers and rank-zero device acceptance.
Graph IDs are discovered from kernel markers within each cycle. All target
pieces between embedding and the final head are included. Eager projection
families are recognized explicitly; other device kernels remain separate.
Exclusive kernel charges advance the overlap frontier before grouping.

Each cycle reports target, drafter, catch-up, sampling, other device work
and no-recorded-kernel time. Drafter sampling kernels belong to sampling.
Copies and memsets overlapping kernel gaps are reported separately, along
with host API/OSRT overlap, host sampling and commit timers, catch-up rows,
policy decisions and process fault counts. Host categories overlap and
must not be added. Gaps do not prove host idle or CPU acceptance cost.
Capture warnings are retained. Profiling overhead prevents using these
timings as a throughput comparison.

CPU preflight runs the analyzer on recorded and synthetic SQLite activity,
including overlap, eager work, wrong clock alignment and missing markers.
A real HTTP fixture exercises both server lifecycles and verifies cleanup
after start, stop, export, geometry and acceptance failures. It cannot run
unless the explicit CPU dry-run flag selects it. Staged GPU launchers never
set that flag.

Use the shared CPU admission wrapper for these checks:

```bash
python3 -B models/deepseek41/lab/dspark_cycle_profile/test_capture.py
python3 -B models/deepseek41/lab/dspark_cycle_profile/test_capture.py --dry-run "$CPU_OUTPUT"
python3 -B models/deepseek41/lab/verify_width_matrix/test_job.py --native "$STAGE"
python3 -B models/deepseek41/lab/dspark_cycle_profile/stage_profile.py "$STAGE" --seal
```

The frozen installation, harness and CPU receipts are hashed. Generated
bytecode and run outputs are excluded using the matrix inventory guard.
Each launcher explicitly selects four GPUs, checks 250 W limits and Xids,
and has a 1200-second limit. The shared work deadline is 1020 seconds,
leaving cleanup time before the outer 1170-second timeout. Large captures
stay under a unique root-volume path recorded with the results. Sealing
does not queue the job.
