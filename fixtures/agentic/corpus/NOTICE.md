# Pinned coding benchmark inputs

These three Rust source snapshots are copied unchanged from Chopinfer and
are distributed under this repository's Apache-2.0 license. They are inert
benchmark inputs; the benchmark never compiles or executes their code.

`manifest.json` records their source commit, original paths, and SHA256
values. The coding corpus digest hashes each path, NUL, contents, NUL in
manifest order. Loading the workload verifies each file and the combined
digest before contacting a server. Snapshots stay fixed when live source
changes; replacing them requires an explicit new corpus revision.

Each file has a bug-fix, refactoring, and documentation task. The separate
decode probe repeats these same sources to its approximate context target,
even when a larger staged corpus supplies the coding tasks. Its receipt
records the exact context digest. Larger corpora, generated receipts, and
benchmark outputs belong outside the repository under `$CHOPINFER_DATA`.

Revision `chopinfer.agentic-coding.v1` and seed `chopinfer-coding-v1` define
the new default baseline. Rerun both comparison arms with these inputs;
results from earlier default corpora are not comparable.

The scorer checks tool flow: a session must read a file, apply one unique
replacement, and finish without failure. Corrected tool errors and a no-op
replacement are allowed. A passing session does not establish semantic
correctness of the edit.
