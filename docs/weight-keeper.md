# Weight keeper

A keeper owns immutable rank images after a serving process exits. Servers map
those images read-only and hold a lease until their mappings are released.
Stopping a borrower releases its lease; it must not stop a keeper owned by its
launcher. The launcher stops the keeper only for a different weight image,
invalidated checkpoint, an explicit shutdown, or an idle-timeout policy.

A normal-load exactness control needs a separate allocation. Run those controls
before the keeper-backed block and retain their receipts for matched borrowed
controls. Do not silently replace a normal-load control with a borrowed one.
Timing runs remain exclusive. Sharing memory does not establish that concurrent
processes provide valid performance measurements.

## Identity across builds

Without a content proof, the existing recipe identity remains the matching
rule. It includes the checkpoint identity, encoded weight plan, shared transform
implementation and any bundle-owned transform implementation. Install directory
names are not part of that runtime identity.

For cross-build lending, prepare a CPU proof with the candidate's own binary,
checkpoint, package options and weight-transform environment:

```sh
bash tools/dev/heavy.sh --label weight-content \
  "$INSTALL/bin/chopinfer" weights identity --content \
  --output "$CHOPINFER_DATA/weight-content/candidate.json" \
  --model "$CHECKPOINT" --tensor-parallel-size 4
```

The output parent must exist; the command refuses to overwrite a file. Without
`--content`, it emits recipe and layout information only, not a byte-equality
proof. Identity planning never constructs a device backend. Hashing a large
checkpoint is substantial CPU and storage work; do it before reserving GPUs.

The proof hashes the compacted device arena, including zero padding and
excluding host and mapped views. Its layout digest includes placement,
alignment, offsets, lengths and component semantics such as dtype, shape and
layout IDs. An authenticated rank index's producer identity is normalized only
for this layout digest. Each server retains its own original recipe identity
for bundle authentication. A component-only layout change invalidates a proof.

Use the proof for both the keeper and its corresponding server build:

```sh
"$INSTALL/bin/chopinfer" weights keep --keeper-dir "$KEEPER_DIR" \
  --model "$CHECKPOINT" --tensor-parallel-size 4 \
  --weight-keeper-identity "$CHOPINFER_DATA/weight-content/candidate.json"

"$INSTALL/bin/chopinfer" serve --model "$CHECKPOINT" --tensor-parallel-size 4 \
  --weight-keeper "$KEEPER_DIR" --weight-keeper-policy require \
  --weight-keeper-identity "$CHOPINFER_DATA/weight-content/candidate.json"
```

Each build supplies its own proof. The keeper verifies its proof against a
hash of the actual uploaded allocation once, before lending it. A borrower
checks that its CPU proof still names its exact executable, fresh recipe,
layout and unchanged checkpoint files. The content protocol requires equal content and layout hashes,
image role, rank, device and byte length, and permits only read-only mappings.
It retains private-directory, no-follow, peer-identity and lease checks.

Legacy clients keep the v1 recipe comparison. A content-capable client detects
whether the keeper supports the explicitly versioned content extension.
`prefer` loads refused images normally; `require` fails instead. A malformed or
stale proof is an input error, never evidence for sharing. Cross-install reuse
still requires the same unchanged checkpoint files; copied checkpoints with
different file identities are not admitted merely because their headers match.

A launcher may compare sealed CPU proofs with the keeper's verified registry
before starting a job. A matching install path alone is insufficient, and a
different install path alone is not a reason to discard verified matching
images. Record loaded versus borrowed bytes and the content/layout identities
with the startup receipt.
