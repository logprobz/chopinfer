# Prepared projection jobs

CPU preparation freezes a source revision, builds the test executables and
cubins, and records hashes of the binaries, case inventory, and Python runner.
The GPU job only executes these artifacts. It reserves two devices and uses
the first assigned device. GPU correctness still requires the resulting
oracle and invariance receipts.

The default selects one case per distinct stage symbol, choosing the largest
row set, then the largest live row count, then the first inventory index.
A case with several stages may represent more than one symbol. The prepared
manifest records those representatives. `--all` runs every use, and
`--symbols FILE` limits either mode to a JSON array of requested stage symbols.
The runner sets `CHOPINFER_PROJECTION_CASES` to the selected inventory indices
and invokes only test families with assigned cases. It requires exactly one
valid final receipt for every assigned index and rejects missing, duplicate,
mismatched, and unassigned evidence. Direct test invocation without that
environment variable also selects the symbol representatives.

For a CPU-only byte comparison after a declaration or derivation change, export
just the case manifest into a fresh external directory:

```sh
cargo --locked --offline run -p kernel-lab --example prepare_projections -- \
  "$CHOPINFER_DATA/projection-case-comparison" --cases-only
cmp "$PREPARED/artifacts/cases.json" \
  "$CHOPINFER_DATA/projection-case-comparison/cases.json"
```

This uses the same serializer and inventory as full preparation, without
building a new cubin selection or writing a private kernel lock. Regenerate
and compare the model adoption ledgers separately to check generated identities.

The GPU wrapper accepts the prepared directory followed by optional `--all`
and `--symbols FILE` arguments. It enforces a fifteen-minute test limit.
Job lists and timing estimates belong with the external qualification data.

Generated dense cases reuse one immutable packed weight shape and one seeded
BF16 activation shape per test thread. The activation's inactive tail remains
zero. Changing shape releases the previous fixture before building its
replacement. Non-BF16 activation calibration keeps the original live-prefix
encoding. Prepared modules are retained in their owning CUDA context; crossing
contexts is refused. Each context verifies its report once. The f64 oracle
splits output columns across at most eight workers while retaining each dot
product's serial reduction order. SwiGLU runs after restoring output order.
Every sampled index, numerical gate, poison check, replay, and chunk comparison
is retained. Indexed dense receipts record preparation/upload, module lookup,
replay/receipt, and chunk durations for subsequent GPU diagnosis.

The CPU preparation job optimizes `kernel-core` and `kernel-lab` in the test
profile. Timing probes and reports stay with the external qualification data.

For qualification of changes to the harness, use `--cases FILE` with an explicit
JSON array of prepared case indices. This preserves separate phase classes even
when several classes use the same stage symbol. The manifest's `dense_families`
records the kernel family and numerical tile-edge row count for each dense use.
The runner requires eager-versus-graph evidence and a rejected finite wrong
output on row 47 whenever selected dense cases reach that row. The negative
regression modifies the device output and applies the unchanged numerical gate.

Full per-symbol qualification and `--all` are deferred to Stage 5 for genuinely
new instances. Existing adopted symbols retain their original lab coverage;
a harness qualification job uses the selected family representatives.
