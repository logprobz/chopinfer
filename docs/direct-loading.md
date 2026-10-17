# Direct weight loading

Serving loads every weight directly from a Hugging Face checkpoint. The
bundle's weight plan (`chop_bundle_weight_plan_v1`, see
[the bundle architecture](../BUNDLE_ARCHITECTURE.md#direct-weight-loading))
states what each rank arena holds, and the host executes it
(`crates/runtime-host/src/direct.rs`). Packages carry no weights.
`chopinfer serve --model DIR` prepares a package with the family helper (see
[compatible checkpoints](compatible-checkpoints.md)); `chopinfer serve
--bundle P --checkpoint C` serves an explicit one.

## Planned components

A package declares each image it serves as a planned component
(`planned_components` in the manifest): its id, role and format, with no
path, size or digest, since the plan gives sizes and alignment. A planned
component requires the weight-plan capability (`chop.weight-plan.v1`); graph
buckets and the speculation drafter may name it. A manifest that names
`compiled_images` is refused with a pointer to `serve --model DIR`.

The host describes the checkpoint to the bundle: the safetensors headers, the
JSON documents (up to 64 MiB) and other files up to 64 KiB of the directory
and its immediate subdirectories, named `dir/file`, and the package's
bundle-metadata components. It asks for one plan per planned role and rank. A
package without planned components is asked once, with an empty role. A load
refuses a package whose bundle declines a planned component. A package with
planned components, or one that requires the weight-plan capability, is
refused without `--checkpoint DIR`. A package with neither (the
[external SDK example](external-models.md)) serves without one.

`qwen38-package` and `mimo26-package` emit such packages from the checkpoint
alone. Qwen plans its text image, whose views include the MTP layer except
under a mixed precision policy, and with `--encoders on` the vision encoder as
a second planned component; the encoder plan refuses a checkpoint without the
vision tower or with a quantized or differently shaped vision tensor. MiMo
plans its text image and, with `--drafter mtp|dflash` and `--encoders on`, the
drafter's and the encoders' images; a small bundle-metadata component names
the drafter's kind.

## Host-placed views

A plan may place a view in host memory (`ViewPlacementV1::Host`). The host
then keeps that span in page-locked, device-mapped host memory instead of the
rank arena, and the package must require `chop.host-resident-views.v1`. A host
span starts on the image alignment; the host cuts it out of the arena,
rounded up to that alignment, and later bytes move down by the same length.
Admission refuses an image with host views whose views are aligned above the
image alignment, and cuts that would leave a rank no device bytes. Every
rank's plan must agree on each view's alignment and placement. Qwen's TP2
plan keeps the n-gram table in host memory; see
[pinned host memory](deploy-common.md#pinned-host-memory) for the host limits.

## Host-mapped views

A plan may instead map a view from the checkpoint
(`ViewPlacementV1::Mapped`, capability `chop.host-mapped-rows.v1`): a table
read a few rows at a time, too large to keep in device or page-locked
memory. Exactly one record covers such a view, a plain copy of one tensor,
whole or one contiguous range of its leading axis, stored unchanged; the
host refuses any other record by name. It rechecks the file against the
header it opened, then maps that byte range read-only from the inode it
opened without following links. Every rank and image of the process that
names the same bytes shares one mapping. The span is cut out of the rank
arena as a host view is, and no upload reads it; the page cache holds the
bytes, and neither the device nor any page-locked budget counts them.
`startup direct_load` states their total as `mapped_bytes`.

A bundle reads a mapped view on the CPU (`with_mapped_view`) and gathers the
rows a step needs into a staging buffer: page-locked, device-mapped host
memory it allocates and writes in place (`allocate_staging_buffer`,
`with_staging_buffer`), one per rank, which device copies and kernels take
like any buffer. A step launches the work that does not need the rows,
gathers them on the CPU meanwhile, copies them on a stream of their own, and
has the consuming work wait on an event recorded after the copy.

The host keeps every loan. A closure's loan of a mapped view keeps its image
from being released; a staging buffer's loan first waits for the device work
that named the buffer since its last loan (an event recorded behind it on
each such stream, so copy on a stream that does little else), and while it
lasts the host refuses a second loan, the buffer's release and any device
work naming it. A graph capture and a device pointer table may not name a
staging buffer, since their work would reach it where no loan waits.

The checkpoint must not change while it is served. A mapping reads what the
file holds, and a file truncated while mapped makes reads of the lost pages
fault (`SIGBUS`), which the host does not handle.

## Streaming a plan

Each rank uploads through two alternating pinned 64 MiB chunks (512 MiB
pinned at TP4 during the upload, freed afterwards). The rank's worker threads
run the plan records a chunk covers and write straight into it; larger records
are produced in parts. Records that every rank places identically are
produced once and copied into each rank, kept between consumers within
`CHOPINFER_DIRECT_LOAD_SHARED_BYTES` (1 GiB by default).

Ranges the page cache holds are copied from it; cold ranges are read with
`O_DIRECT` (residency from `cachestat`, Linux 6.5; without it everything goes
through the cache). A file system that refuses direct reads falls back to the
cache for that file; other read errors fail the load.
`CHOPINFER_DIRECT_LOAD_DIRECT_READS=0` turns direct reads off, which enables a
per-rank read-ahead thread; `CHOPINFER_DIRECT_LOAD_READ_AHEAD=0` or `=1`
overrides that choice.

Workers per rank default to twice the CPUs the process may run on (its
affinity set, less any cgroup quota) divided by the rank count, at most 16,
since workers mostly wait on reads. `serve --load-threads N` or
`CHOPINFER_DIRECT_LOAD_THREADS` sets the count. As a guide, a cold CPU-side
load of the MiMo V2.6 Flash text image at TP4 (183 GB) took 27 to 39 seconds
with direct reads (2026-09-29).

Startup prints one `startup weight_plan` line per planned image (plan time,
record count, shared outputs, checkpoint identity, workers, CPUs, read-ahead
and direct reads) and one `startup direct_load` line (upload time, bytes
produced, read and read directly, read count, advised bytes).
`CHOPINFER_DEBUG_ARENA_DIGEST=1` reads each arena back after the upload and
prints its SHA-256 as `startup arena_digest` (about 26 s per 48 GB rank,
reported as `digest_seconds`). The `startup verify:` line reports
`component_bytes_hashed`: package verification reads every component in full,
hashes it and keeps it in memory, so a file rewritten after verification
cannot change what is served.

## Checkpoint and rank identity

The checkpoint's identity is the SHA-256 of every safetensors file's name,
length and header. It does not authenticate payload bytes: a tensor rewritten
in place with the same length passes it. Release qualification therefore
needs a content check: `chopinfer verify --model DIR` (see
[compatible checkpoints](compatible-checkpoints.md)). After the last arena is
uploaded the host checks that every checkpoint file still has its length and
header, and refuses the load otherwise.

A planned rank has no file, so its rank index carries `direct_identity`
instead of `sha256`: the SHA-256 of the checkpoint identity, the
implementation that produces the bytes and the rank's plan (arena, views and
records, without components). The implementation is a digest of every source
on the host's byte path (the source trees, build scripts and manifests of
`runtime-host` and every workspace crate it links, plus `Cargo.lock`, listed
by `runtime-host/build.rs`), and, when a record uses a bundle-owned
transform, the SHA-256 of the bundle's native module. A test fails if a module
on the byte path is outside that set. Any edit to those sources therefore
changes every planned identity. The host reports the same value as the rank's
identity, so a bundle can check its rank indexes against it (MiMo does).

## Proving device bytes unchanged

Since any edit on the byte path changes every `direct_identity`, identities
cannot show that a change left the uploaded bytes alone. Digest ledgers can.
`qwen38-load-direct` and `mimo26-load-direct` (package `qwen38-load-proto` in
`tools/load-proto`) plan every rank through the bundle's planner and stream it
on the CPU through the host's executor. `digest` prints a ledger entry for one
image role and rank count: per rank the arena length, the plan's SHA-256, each
component's SHA-256 without identity fields, and the SHA-256 of the produced
bytes. `verify --reference LEDGER` loads again and compares; it exits 2 on any
difference. Equal ledgers for a checkpoint mean equal arenas.

```sh
cargo run --release -p qwen38-load-proto --bin qwen38-load-direct -- \
  digest --checkpoint "$CHOPINFER_QWEN38_CHECKPOINT" --ranks 4 > "$OUT/before.json"
# rebuild with the change, then
cargo run --release -p qwen38-load-proto --bin qwen38-load-direct -- \
  verify --checkpoint "$CHOPINFER_QWEN38_CHECKPOINT" --ranks 4 --reference "$OUT/before.json"
```

`--image encoders` (Qwen) and `--image text|drafter|encoders` (MiMo) select
the image; `--package DIR` passes a package's bundle metadata (Qwen's
precision record, MiMo's drafter kind with `--package-index`) as the host
does. `plan` checks that each rank's records tile every view of its rank
index without producing bytes, and `create` runs the bundle's create over a
package on the host's fake device. On a GPU, `CHOPINFER_DEBUG_ARENA_DIGEST=1`
prints the device-side digest of each arena to compare with a ledger.
