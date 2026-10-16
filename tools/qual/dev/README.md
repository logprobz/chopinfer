# Developer diagnostics

Measurement and inspection helpers outside the main qualification command
list. Run them from the workspace root, for example
`python3 tools/qual/dev/audio_check.py --help`. None of them qualifies a
model. KV preemption swap qualification now uses `kv_offload.py` through
the correctness runner's full suite; see the [qualification tools](../README.md#correctness-runner).

- `prefill_budget_soak.py`: compare queued load under prefill budgets.
- `encoder_parallel.py`: compare encoder geometry, tokens and timings across
  ranks and parallel modes.
- `spec_summary.py`: summarize speculative benchmark arms.
- `trace_inputs.py`: trace fixture reads offline for refactor parity.
- `audio_check.py`: inspect audio spread and layer receipts.

`encoder_parallel.py` needs `CHOPINFER_QUAL_CONFIG` set to the model's
`qual.json`. Drivers that launch servers need otherwise idle GPUs.

To check that a refactor of `tools/qual` reads the same inputs, trace the
gate modules of a baseline export and of the candidate against the same
fixture tree. Network requests are refused, so this shows input and policy
parity, not accuracy:

```sh
scratch="$CHOPINFER_DATA/qual-tools-trace"
mkdir -p "$scratch/baseline" "$scratch/tmp"
git archive main tools/qual | tar -x -C "$scratch/baseline"
ln -s "$PWD/fixtures" "$scratch/baseline/fixtures"
for family in $(ls models); do
  python3 -B tools/qual/dev/trace_inputs.py --source "$scratch/baseline" --root "$PWD" \
    --gate "$family" --scratch "$scratch/tmp" > "$scratch/before-$family.json"
  python3 -B tools/qual/dev/trace_inputs.py --source "$PWD" --root "$PWD" \
    --gate "$family" --scratch "$scratch/tmp" > "$scratch/after-$family.json"
  diff -u "$scratch/before-$family.json" "$scratch/after-$family.json"
done
```
