# Runyard pinned benchmark inputs

Runyard is a project by the Chopinfer maintainer. These six fixture files
are included with the author's permission, copied from their Runyard checkout
at `9e6022c0fb7e075734a9602e227ea08d9b1fe442`. They are distributed as part
of this repository under its Apache-2.0 license. Existing source notices and
fixture bytes are preserved unchanged.

`manifest.json` lists each fixture's original path and SHA256. The coding
corpus digest follows Runyard's ordered hash of path, NUL, contents, NUL.
The three decode snapshots are hashed individually. Loading the workload
verifies every file before contacting a server.

Port sources in Runyard:

- `internal/bench/agentic_profile.go`: revision `runyard.agentic-coding.r6`,
  nine task instructions, SHA256 task selection and initial repository context.
- `internal/bench/agentic_workspace.go`: tool schemas, read pagination,
  UTF-8 byte bounds, literal search and unique single replacement.
- `internal/bench/agentic_session.go`: system prompt, tool loop and validity.
- `internal/bench/agentic_sweep.go`: warmup, replenishment, fixed measurement
  interval, drain classification, floor-rank percentiles and counter coverage.
- `internal/bench/agentic_decode.go`: separate approximate 32K context probe.
- `internal/bench/agentic_model.go`: advertised context and output cap handling.
- `internal/bench/agentic_version.go`: corpus and profile identity rationale.
- `internal/bench/executor_http.go`, `internal/proxy/generate.go` and
  `internal/proxy/observe.go`: generated output and per-request usage counters.
- `internal/cli/bench_agentic.go` and `docs/benchmarks.md`, section
  "Built-in agentic coding sweep": CLI defaults and measurement semantics.

The `.txt` files are byte-for-byte copies from Runyard's
`internal/bench/agentic_corpus/` and `internal/bench/agentic_decode_corpus/`.
They are inert benchmark data. The benchmark never executes their code.

The Python port does not implement Runyard's managed-run sealing or UI.
It uses Chopinfer qualification receipts and adds inter-delta distributions
and Prometheus snapshots. It requires a selected `/v1/models` identity,
and requires a complete SSE stream. Its `output_deltas`
counts generated SSE events (several generated fields in one event count
once), while Runyard counts generated field values. First/last output timing
and reported-token rate formulas are unchanged. Warmup validity is recorded;
invalid drain sessions also fail the port's overall validity gate.

Receipt v2 groups client output into configurable bursts, treats server step
histograms as primary timing evidence, uses greedy sampling with deterministic
per-case/turn request seeds, and assigns cases in fixed per-slot sequences.
These intentional differences from Runyard reduce A/B trajectory noise. The
corpus, task generation and scorer semantics are unchanged.
