# Reproducing the prefill adoption proof stages

Like the [compact candidate-on control](sw6-compact-allocation.md), these
comparisons keep their capture-only controls separate from production
history. The committed patches below reproduce both source stages from
ancestors of the adoption series. The original unreferenced commits are
not needed, and no branch or tag is created.

| Stage | Base commit | Source patch | Expected tree |
|---|---|---|---|
| Template exposure | `641eccbacc3f41273f1bd1c1594262ea963048fc` | [template checks](prompt-end-anchors-fixtures-stage.patch) | `b6739a33bb7b0e9740a82c3fb3ba891a19195a60` |
| Workspace reservation | `b8b736b056f9d0997a91997a4943d574553678b0` | [workspace control](prefill-tail-folds-workspace-stage.patch) | `4b6852cef85a2855089a0b2755ac90e787b9d12b` |

Template exposure replaces two direct digest call sites with named
`check_journal` calls, retaining all original expected pins. It exposes four
long-context bodies from two pins, plus two failed-template bodies from
two pins.
Workspace reservation grows the Flash TP4 scratch bounds while retaining
the previous dense routes. Its pins include the attention arguments that
pass a whole workspace. The patch contains the final sealed control,
including those corrected pins; the earlier unsealed preparation is not a
proof input.

Run this recipe from the repository root. The example selects template
exposure; use the second table row's base, patch and expected tree for
workspace reservation.

```sh
proof_base=641eccbacc3f41273f1bd1c1594262ea963048fc
proof_patch=models/deepseek41/journal-changes/prompt-end-anchors-fixtures-stage.patch
proof_expected_tree=b6739a33bb7b0e9740a82c3fb3ba891a19195a60
proof_stage_dir=$(mktemp -d "${CHOPINFER_DATA:?}/ds41-proof-stage.XXXXXX")
proof_index="$proof_stage_dir/index"
GIT_INDEX_FILE="$proof_index" git read-tree "$proof_base"
GIT_INDEX_FILE="$proof_index" git apply --cached --check "$proof_patch"
GIT_INDEX_FILE="$proof_index" git apply --cached "$proof_patch"
proof_tree=$(GIT_INDEX_FILE="$proof_index" git write-tree)
test "$proof_tree" = "$proof_expected_tree"
proof_stage=$(git commit-tree "$proof_tree" -p "$proof_base" \
    -m 'Reproduce a prefill adoption proof stage')
printf '%s\n' "$proof_stage" > "$proof_stage_dir/revision"
```

Only the temporary index and unreferenced Git objects are written. The
working tree and its index stay unchanged. The tree check authenticates
all source bytes. The new commit ID can differ because author and time
metadata are not part of the source tree. Use the newly created revision
in fresh captures and reports; do not relabel an old capture.

```sh
cat > "$proof_stage_dir/ignored-tests" <<'IGNORED'
fake_host_tests::query_prefill::export_query_prefill_lifetimes
fake_host_tests::router_gaps::export_router_gap_lifecycles
fake_host_tests::short_prefill::short_prefill_journals_preserve_every_other_operation
lower::moe::scales_tests::graph_export::export_wide_moe_graphs
IGNORED
python3 -B tools/journal_proof/capture.py --repo . --model deepseek41 \
    --revision "$proof_stage" --runner nextest --jobs 4 \
    --allow-ignored "$proof_stage_dir/ignored-tests" \
    --out "$proof_stage_dir/capture"
```

Capture the corresponding base and adoption head with the same settings.
The two adoption heads are the commits named "Enable DeepSeek prompt-end
anchors by default" and "Enable DeepSeek dense prefill tail folds by
default", respectively. The proof chains are:

| Comparison | Committed proof input |
|---|---|
| Template base to template exposure | [fixture specification](prompt-end-anchors-fixtures.json) |
| Template exposure to anchor default | [anchor layout declaration](prompt-end-anchors-default.json), with a witness from `specs/layout_spec.py` |
| Workspace base to workspace reservation | [workspace specification](prefill-tail-folds-workspace.json) |
| Workspace reservation to folds default | [routing specification](prefill-tail-folds-default.json) |

Use `tools/journal_proof/proof.py` with each pair's sealed capture indexes
and the listed specification. Generate the anchor layout witness with its
recreated stage as `--base` and the anchor adoption commit as `--head`.
The [journal-proof instructions](../../../tools/journal_proof/README.md)
describe those commands and the capture sealing checks. Both patches have
been applied to separate temporary indexes and checked against the trees
above. These source controls change no production default or kernel lock.
