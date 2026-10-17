# Journal proofs

`proof.py` proves that every pinned fake-host journal of a model's bundle
changed between two revisions only as a reviewable JSON change spec
declares. A served-layout change (added allocations, new collectives,
changed grids) is declared in a layout declaration the spec names (see
[Layout declarations](#layout-declarations)). It compares two sealed captures (one per revision) and reads
sources only through `git show` at the resolved commits. It does not build,
run kernels or use GPUs. Python 3.11 or later and git are the only
dependencies.

```sh
python3 -B tools/journal_proof/proof.py --repo "$REPOSITORY" --model MODEL \
  --base "$BASE_COMMIT" --head "$HEAD_COMMIT" --spec "$CHANGE_SPEC" \
  --base-capture-index "$BASE_CAPTURE/index.json" \
  --head-capture-index "$HEAD_CAPTURE/index.json" \
  --report "$RESULTS/proof.json" [--require identity] [--declaration-revision REV]
```

Exit 0 (PROVEN) means every journal is IDENTICAL, PROVEN or a declared
addition or removal. Exit 1 means NOT PROVEN, and exit 2 means invalid input.
The report and its `.txt` sibling hold:
- the resolved commits and the sources' SHA-256 values;
- each journal's verdict, pins and body digests;
- the fixture mappings, history records and counts of authorized changes;
- `coverage`;
- `claim`, which states what was compared and what was not.

`specs/identity_spec.py --repo REPO --model MODEL --revision REV --index
INDEX` writes the identity spec of a capture: every fixture it binds, with the
empty rule. `--require identity` adds the byte-identical checklist (`identity_checklist`
in the report). Exit is then 0 only when all of the following hold:
- the proof passed with `coverage.full_history_proof` true;
- every journal is IDENTICAL;
- every fixture is COMPARED or TEXT;
- the spec declares nothing: empty auxiliary rules, no journal policies,
  module replacements or graph piece policy;
- no change counts beyond `compared_operations` and
  `trailing_uncompared_notes`.

## Pins and fixtures

`--model` names `models/MODEL`. Its `model.toml` `[bundle] package` must be
the crate in `models/MODEL/bundle`. A capture index names every journal the
revision's tests checked and seals each body's SHA-256. Each body's line
count and FNV-1a is its pin.

The revision's sources must hold exactly those pins: the literal
`(lines, 0x<fnv>)` tuples of the Rust files under the bundle directory that
call `check_journal` must equal, as a multiset, the captured journals' pins.
An unreached literal pin, a captured pin that is not a literal, a missing
body and an extra body are all invalid input. A journal of zero lines (a guard
such as "no gaps remain") is pinned `(0, 0xcbf29ce484222325)`, the FNV-1a of no
bytes; any other zero-line tuple is refused. An empty journal is always a text
journal, since no device line can witness it. Every `check_journal` call site
must pass a literal pin or a plain place (a variable, field, index or deref,
or a tuple of those). A call that computes its pin in place, a pin-like tuple
`(N, 0x...` that is not a whole literal, an aliased checker and any use other
than a direct call are refused.

Plain-place syntax does not establish where a place's value came from, and
the multiset alone does not show which call checked which pin: two journals
checked through one table read could stand for an unreached literal call
with the same pin. A v3 capture (`capture.py`) records the call site of every
check, and the capture and the proof both require:
- every captured journal was checked at a found call site;
- a call with a literal pin checked exactly that pin;
- every call site, literal or table read, checked at least one captured
  journal. There is no allowlist: a call the lib tests do not reach (in an
  ignored test, for example) fails the capture.

A call such as `let expected = pin_of(...); check_journal(&lines, expected,
file)` that the tests never reach is therefore refused. A v2 capture (made
by the earlier external helper) names no call sites; its helper authenticated
the bodies against a source manifest that covered every call. Each report's
`auxiliary_provenance.SIDE.call_sites` says which applies. A `.pin` sidecar
written by the capture must agree with its body. It is never the authority.

A fixture is one device lifetime: the journals one test's fake device
recorded. A journal that no device of its test recorded is a text journal,
and its fixture is its test. A spec must group journals exactly as these
fixtures do. A fixture split across spec fixtures, or spec fixtures that
combine source fixtures, are refused.

Binding a journal to its device (`auxiliary.collect`):
- The device is the one whose recorded lines contain the journal. When
  several do (a journal byte-identical to another package's, or an identical
  device copy), a device where a journal that only one device recorded
  already owns those operations is not a candidate; of the rest the lowest
  device ID is bound. The others are recorded with the binding: claimed by
  another journal, an identical copy (the same records), or, with different
  records, listed in `coverage.uncovered_devices` and never counted as
  covered.
- The create is the instance-create marker before the journal, around it,
  or, for a journal of the whole device lifetime, inside it. A journal of the
  whole lifetime of a device that ran no instance create (an encoder device)
  binds with an empty create (`whole-lifetime`); the proof checks both
  conditions against the trace.
- A resolve that names a module handle the device never loaded (a handle the
  host passed in) keeps that handle literally. Its module bytes are not
  authenticated; `coverage.unloaded_module_fixtures` names the fixtures.

## What is compared

For each device fixture, the proof divides every operation the device
recorded through the fixture's last pinned checkpoint among three owners:
- the pinned journals;
- the fixture's own create range;
- the history: setup, work between pinned captures, and operations inside a
  capture span that the journal does not contain.

Each owner is compared line for line after its declared rewrites and the
fixture's one typed mapping. The mapping covers issued IDs, allocation
returns, streams, events, graphs and functions, and comes from the
authenticated trace, never from first appearance. Create, pinned and history
lines are mapped by the same mapper (`history.Mapper`), which maps every
typed identity of the line's operation form, including the stream and event
of `record` and `wait` and the graph of `captured`, `replay` and `release`.
The operation forms are every form the fake hosts journal, including a
model's `readback S SPAN BYTES` and `peer copy S C SPAN C SPAN`; a line's
stream is its form's (`history.stream_of`). Anything the mapper does not
model fails:
- an unmapped function, event or graph ID;
- a stream with no recorded creation that equals the image of a created
  stream (such a stream otherwise stays fixed);
- a raw handle that is no base allocation (a host buffer) and is the image
  of an allocation or a declared added allocation's handle.

History lines are a closed grammar: every line must have an operation form,
and a stream with no recorded creation may not share its number with any
issued ID. Create and pinned journal bodies also hold test annotations, and
each must match one exact grammar: `step NAME`, `NAME view
BundleLogitsViewV1 {...}` or `NAME drafts OwnedDraftTokensV1 {...}`. A name
is words of letters, digits, `_`, `.` and `-`, compared literally, and only
a view's buffer handle maps. Any other line fails. In a body, an encoder's
host stream may share its number with an issued ID that is not a stream
(an encoder can launch on host stream 1 while its first resolve has ID 1).

The cross-owner order is then checked: a history operation cannot move
across a pinned launch or the create range. History lines use a closed
grammar, and an unknown form fails closed. Any failure marks every journal of
the fixture UNPROVEN.

Not compared:
- operations after a fixture's last pinned checkpoint, reported as
  `trailing_uncompared_notes`;
- upload contents beyond the recorded FNV-1a of their payload, and an arena
  upload's contents (FakeDevice journals it by rank only);
- which buffer a bare `release` frees (FakeDevice journals no handle; its
  place and count are compared);
- host state the fake device does not journal;
- kernel numerics, performance and source correctness.

## Text journals

Some journals are built by a test without a fake device, for example a list
of asks or invariance claims. A text journal is compared only by its body:
- With no policy, the base and head bodies must be byte-identical. The
  verdict is IDENTICAL.
- A change is PROVEN only under `text_edits` in the journal's policy. This is
  a list of `{"remove": {"line": N, "text": ...}}` (one-based base line) and
  `{"add": {"line": M, "text": ...}}` (one-based head line) entries, with exact
  text. The base without its removed lines must equal the head without its
  added lines.

`specs/text_edits.py BASE HEAD` writes such a policy for a reviewer to read.
The proof never derives one.

Rules:
- A text fixture takes the empty auxiliary rule.
- Text journals take no other policy, and `text_edits` apply to no device
  journal.
- A journal whose kind changes between the revisions (device to text, or
  back) is UNPROVEN.
- `coverage.text_fixtures` lists text fixtures. They count toward
  `full_history_proof`, and the claim states that they have no device history.

## Change specs

```json
{
  "schema": "chopinfer.journal-proof.spec.v1",
  "fixtures": [{"name": "host", "create": "m-create.txt", "journals": ["m-create.txt", "m-step.txt"]}],
  "journals": {"m-step.txt": {"launch_rules": []}},
  "auxiliary_rules": {"host": {"added_resolves": [], "added_modules": [], "create_policy": {}}},
  "added_journals": [],
  "removed_journals": []
}
```

Optional top-level fields are `description`, `module_replacements`,
`graph_piece_policy` and `layout` (a layout declaration, below). Every
fixture needs exactly one `auxiliary_rules` entry. A fixture of only declared added or removed journals, and every text
fixture, takes the empty rule shown above.

Journal policies (`journals`), the auxiliary `create_policy` and
`history_policy` accept these keys:
- `launch_rules`, `symbol_substitutions`, `cubin_substitutions`;
- `state_sizes`, `moves`, `merges`, `removals`, `expansions`;
- `added_resolves`, `added_modules`, `added_configures`;
- `span_changes`, `layout_spans`, `layout_strides`, `upload_payloads`.

`history_policy` also accepts `graph_nodes` and `graph_pieces`.

An auxiliary rule may also declare:
- `removed_resolves`;
- `allocation_resizes`, `allocation_bindings`, `allocation_layouts`.

Every rewrite names exact lines, symbols or counts. `count: "all"` transforms
every matching base launch and never selects by inspecting the head.
Overlapping rewrites are refused. Every span of a rewritten journal must lie
inside its allocation's head size. The one exception is an oversized span the
base already held (a fake host does not bound spans) in an argument kept
identical: the whole argument (its kind and every element, empty entries
included) equals the kept base launch's argument at the same index, and every
allocation it references keeps its size. Any difference in the argument bounds
all of its spans, and a record other than a launch is exempt only when kept
whole. Spans of merged or expanded launches are always bounded.

`module_replacements` authenticate a changed cubin against the
`kernels.lock.d` locks of both revisions:
- exact renames, removed symbols, configure byte changes and resolve order;
- each captured payload must match its locked SHA-256.

`graph_piece_policy` (`{"source": "crates/bundle-runtime/src/graphs.rs",
"divisors": [32, 6]}`) lets graph programs re-split at `[0, L/a, L/b, L]`. It
is accepted only when both revisions' `pieces` function is the recognized
implementation. The pre-rename path `crates/bundle-runtime-sm120/src/graphs.rs`
is also read, at revisions that have it.

The tests are the executable reference for every field:
- `test_proof.py`: the recorded reviewer false acceptances, pins and the
  identity checklist;
- `test_auxiliary.py`: trace authentication and the typed mapping;
- `test_extensions.py`: replacements, expansions and allocation-bound rewrites;
- `test_history.py`: history, layouts and graph pieces;
- `test_text.py`: text journals;
- `test_layout.py`: layout declarations and their witness writer.

## Writing specs

Specs are ordinary JSON. The writers under `specs/` derive every count, ID
and position from the captures, so a reviewer reads a short declaration and
the generated spec instead of writing counts by hand:
- `identity_spec.py`: every fixture of a capture with the empty rule.
- `build.py --declaration CHANGE.json`: a labelled kernel change, declared
  in schema `chopinfer.journal-proof.change.v1`:
  - a locked module dropping `removed_symbols`;
  - an `added_module` that owns new shared symbols;
  - `launch_rules` moving an old symbol's launches to a new one with
    appended literals.

  It checks the locks of both revisions, refuses any other appearance of a
  removed or added ID, and leaves text fixtures with the empty rule. See
  its docstring.
- `graph_nodes.py [--piece-policy A B]`: adds `graph_nodes` (and
  `graph_pieces`) to a spec. It reads the base history and the spec's own
  policies, never the head's counts.
- `text_edits.py BASE HEAD`: a text journal's `text_edits`.
- `layout_spec.py --declaration PATH [--declaration-revision REV]`: the
  spec of a declared served-layout change (below), with each fixture's
  witness.

### Without a layout declaration

Issued device IDs pair one to one between the revisions. The only
exceptions are the exact `added_modules` and `added_resolves` entries of an
auxiliary rule and the resolves a locked module replacement removes. Any
other addition or removal (a device allocation, a module or resolve beyond
those entries, or uploads or zeros of new buffers at create) is refused with
"issued ID count differs beyond exact module/resolve additions/removals"
(`test_auxiliary.py` pins this). Such a served-layout change is proven from
a layout declaration.

## Layout declarations

A layout declaration (schema `chopinfer.journal-proof.layout.v1`, read by
`layout.py`) states a served-layout change in reviewable items. The proof
applies it to every device fixture: the create range, every pinned journal
and the history. Every selection is evaluated on the base, never by reading
the head, and anything the declaration does not cover must be exact, as
without one.

```json
{"schema": "chopinfer.journal-proof.layout.v1",
 "description": "...",
 "modules": {"removed": ["old_cubin"],
             "added": [{"name": "new_cubin", "loads_like": "old_cubin"},
                       {"name": "comm_cubin", "loads_like": "core_cubin",
                        "configures": {"chop_new_reduce_v1": [65536]}}]},
 "resources": {
   "added": [{"name": "comm.inbox", "bytes": 256, "alignment": 256, "per_rank": 1,
              "lifetime": "instance", "init": "zero",
              "alongside": {"bytes": 128, "alignment": 256}}],
   "resized": [{"name": "moe.route", "from": 32, "to": 64, "alignment": 256, "per_rank": 1,
                "layout": {"planes": 2, "from_stride": 16, "to_stride": 32, "from_tail": 0, "to_tail": 0}}]},
 "collectives": [{"name": "MoE reduce", "from_symbol": "chop_old_reduce_v1", "to_symbol": "chop_new_reduce_v1",
                  "ranks": 2, "when": {"after": {"symbols": ["chop_moe_v1"]}},
                  "peers": {"1": "comm.inbox"}, "buffers": {}}],
 "kernels": [{"name": "MoE", "from_symbol": "chop_moe_v1", "to_symbol": "chop_moe_split_v1",
              "args": {"2": {"ScalarU32(1)": "ScalarU32(0)"}},
              "lengths": {"4": {"576": 2304}}, "strides": {"3": 0},
              "cases": [{"when": {"grid_x": 8}, "then": {"grid_x": 2}}]}],
 "text_journals": ["model-invariance-asks.txt"]}
```

The rules:
- **Added allocations.** Each head allocation that pairs with no base
  allocation must be listed in the fixture's witness and be a declared added
  resource: bytes and alignment. The ranks that must hold it come from the
  base, never from the head or the witness: `alongside` names a base
  allocation (bytes, alignment) the resource is added next to, and every
  rank that holds one in the fixture's base holds exactly `per_rank` of the
  resource in the head. Every other rank holds none. A rank whose added
  buffers are all missing therefore fails. Its lifetime is `instance`:
  allocated inside the fixture's create range. Its `allocate` note (the note
  before its allocation event) is accepted, and with `init: zero` exactly one
  whole zero of it inside create, on the stream of its own allocation, when
  the device ran an instance create (none in a create that is only the
  allocation prefix). The zero is a declared addition and is not compared,
  so its place is checked instead: it follows the allocation, nothing before
  it names the buffer, and every later record that names the buffer is
  ordered after it. A record is ordered when it runs on the zero's stream,
  on a stream that waited for an event recorded on an ordered stream after
  the zero, or after the host synchronized an ordered stream (a record with
  no stream needs the last). Graph capture follows FakeDevice for one open
  capture at a time per stream: an event recorded on a captured stream
  belongs to the capture, a stream that waits on it joins the capture, and
  the graph holds the launches of the capturing stream and of every stream
  that joined it. Those launches run at each replay of the graph, so each
  replay must come after the zero and be ordered on the replay stream. A
  member's launches issued before it joined are checked both when issued and
  at replay. Record and wait inside a capture order nothing outside the
  graph. What the model does not cover fails as not modelled:
  - a stream that begins or joins a capture while in another open capture;
  - a captured zero;
  - any other operation on a captured stream that names a checked buffer;
  - a replay inside a capture of a graph that uses one;
  - a replay of a graph that was never captured.
  On a device whose create is only its allocation prefix (it never runs the
  instance create that zeroes the buffer), the head may use the buffer
  unzeroed only where it keeps the base's initialization. Every record naming
  it must come from a declared collective argument, and no base record before
  that argument may write the base buffer it replaces there. Only a download,
  a readback, the source of a copy or peer copy, and an earlier declared
  argument where the same added buffer replaces every argument of that
  launch naming the buffer leave it unwritten (the head makes that write to
  the added buffer). Any other record naming it counts as a write: a zero,
  an upload, a copy into it, a launch, an annotation, or an earlier declared
  argument of another added buffer or of a launch that also names the
  buffer elsewhere. A use that no declared argument explains, or that
  replaces a buffer the base writes first, fails. The one exception is a
  plan-only device: when every note on it in both revisions is an `allocate`
  or a `launch rN` plan record (which lowering tests write with
  `device.note`), nothing executes, so its plan records are not writes. Any
  other record (a stream, zero, upload, copy, synchronize, record, wait,
  readback or an executed launch) makes the device executing, and every
  record counts.
  Any other record naming an added buffer must come from a declared
  collective argument.
- **New collectives.** Every base launch of `from_symbol` selected by
  `when` becomes `to_symbol`. Each argument in `peers` must be a peer array
  of exactly `ranks` entries over `ranks` distinct ranks; each entry
  becomes the same rank's added
  buffer of the named resource, whole, in the same order. Each argument in
  `buffers` (a single span) becomes its rank's added buffer. Launches of
  `from_symbol` that `when` does not select stay as they are.
- **Grid changes of a named kernel family.** Every base launch of
  `from_symbol` (selected by `when`) becomes `to_symbol`. `cases` set grid
  fields (and the symbol) of the launches their `when` selects; a launch no
  case selects keeps its grid, and two matching cases fail the fixture. A
  case's argument literal may replace only a scalar literal.
- **Selectors** (`when`) compare base config fields, `rows`, `prefix`,
  argument literals (`args`) and the previous launch on the same stream
  (`after`: its symbol in `symbols`, optionally its config fields).
- **Argument edits.** `args` maps exact scalar literals of an argument to
  new literals (values not in the map stay). `lengths` sets the length of a
  span in a buffer the device never allocated (a host-provided buffer such
  as the weights arena); these are exact but bounded by no allocation, and
  `coverage.unbounded_host_span_lengths` counts them. `strides` ties a
  scalar argument to the argument whose span it strides: when that span is a
  resized allocation with a layout, the scalar must be the base plane stride
  and becomes the head one.
- **Modules.** A removed module must be locked in the base and not in the
  head, and an added one the reverse (`kernels.lock.d`). Every load of
  either must carry its captured payload with the locked SHA-256, and every
  resolve it owns must be a locked entry of it. Their resolve and configure
  records are dropped from the base. Each such record must have its exact
  operation form, and a resolve must name the symbol its trace row resolves. In the head, an added module is loaded
  as many times as its `loads_like` module (a base module) is in the base,
  every load resolves the same symbols once each, and each resolve's
  configure notes follow it directly and carry exactly the values
  `configures` lists for its symbol (none when the symbol is not listed).
- **Resized allocations.** Each base allocation whose size changes must be
  listed in the witness and be a declared resized resource (from, to,
  alignment, `per_rank`). Every rank holding a base allocation of `from`
  bytes and that alignment resizes exactly `per_rank` of them, so a rank
  missing its resize fails. Its `allocate` note and every span of it are
  mapped: a whole span to the whole new allocation, and with a `layout`, a
  span by its form (one whole plane, inside one plane, one entry in every
  plane, or inside the tail). Any other span of it fails.
- **Text journals.** Only journals in `text_journals` may carry
  `text_edits`.

A spec names the declaration as `"layout": {"path": ..., "sha256": ...}`.
The proof reads it with `git show` at the head commit, so it is committed and
reviewed with the change. A change declared after it landed names the
revision that holds the declaration with `--declaration-revision`; the
report's `layout_declaration` records the path, revision, digest and
`read_at_head`, and its claim says when it was not read at the head. A
proof offered to land a served change must have `read_at_head` true: the
declaration is committed in the change itself. `--declaration-revision` is
only for replaying a change that has already landed, such as checking an old
change against a declaration written afterwards. A
layout spec carries no journal policies other than declared `text_edits`, no
create, history or allocation policies, module replacements or piece policy.

Each fixture's auxiliary rule holds the witness: `added_allocations`
(resource, head handle, rank), `resized_allocations` (resource, base and
head handles, rank), `removed_modules` and `added_modules` with
`removed_resolves` and `added_resolves`. The proof checks every entry against
the declaration and the traces; the witness chooses which IDs, never what is
allowed. `specs/layout_spec.py` writes it: it aligns the base and head issue
sequences, pairing every kept ID in order (role, rank, alignment, size or a
declared resize) and taking a head allocation as added only when it is a
declared added resource.

A device that records no allocation ranks (a test that never registers
its synthetic streams) is keyed per rank by its synthetic stream, which the
typed mapping keeps fixed; a stream shared by several ranks fails the per-rank
counts. Without a layout declaration such a device still refuses any shifted
allocation.

A declaration is model-specific, so it lives with its model:
`models/<model>/journal-changes/<change>.json`, committed with the change.
A declaration written after its change landed names the revision that holds
it with `--declaration-revision`. Every committed declaration must parse
(`test_layout.py`).

## Producing a capture

```sh
python3 -B tools/journal_proof/capture.py --repo "$REPOSITORY" --model MODEL \
  --revision "$COMMIT" --out "$NEW_CAPTURE_DIRECTORY" \
  [--runner libtest|nextest] [--jobs N] [--allow-ignored FILE]
```

The helper resolves the commit and writes its tree into a scratch directory
outside every worktree, straight from the blob objects, so no export
attribute (export-subst, export-ignore) can change or drop a file. Every git
read ignores replace refs. It patches only that copy:
- `check_journal` dumps every reached journal into `checks/TEST/` with its
  `.pin` (observed and expected) and `.site` (call site, through
  `#[track_caller]`) sidecars, then checks it as before, so a mismatch still
  fails its test. A dump that fails, such as one test checking a journal name
  twice, aborts the test process without replacing the first dump (a panic
  could be caught). Two `check_journal` calls on one source line are refused,
  since a call site is recorded as a line;
- the bundle runtime's fake host traces every test's devices;
- the model's instance create marks its boundary.

It then builds the model's bundle lib tests (`--release --locked --offline
--lib`) in a new target directory under `--target-root` (default
`$CHOPINFER_SCRATCH`). The tests run with no GPU selectors and no lock
update.

Runners:
- `libtest` (default): one test process with `--jobs` test threads.
- `nextest`: one process per test, `--jobs` at once. `nextest list` with the
  same profile gives the package's lib test inventory; its JUnit report must
  hold every listed test that is not ignored, once each.

The build and the run go through `$CHOPINFER_HEAVY_WRAPPER` when it is set
(`--heavy`). The run fails on any failed, filtered or ignored test, unless
`--allow-ignored` lists exactly the tests that may be ignored. A libtest
log's last summary is the run's own and must count every test its first
`running N tests` line announced; earlier summaries come from child runs that
a test starts itself (one test, the rest filtered out) and must not fail.

Several tests may check one journal (for example a test that reruns another
test's checks on real plans). After the run the capture merges the checks:
every check of a name must hold the same body and pin, or the capture fails.
The dump and the device binding come from the test last in name order (the
check a serial run makes last, which the earlier helper kept by overwriting),
so scheduling never changes the result. The index seals the other checks as
`repeated_checks` (test and call site); their call sites count as reached.

Disk and locking:
- The capture refuses to start below `--start-free-gib` (180) free on the
  target volume, and stops the build below `--minimum-free-gib` (150).
- It holds `--lock` (default `$CHOPINFER_DATA/journal-proof/capture.lock`)
  for the whole run, so captures run one at a time.
- Archive and target are removed on success, failure, timeout or
  interruption. Only the helper's own process group is signalled.

Every helper file must hold the bytes of its blob at the tools checkout's
`HEAD` (compared directly, so index flags such as assume-unchanged cannot hide
a change). The index records that commit. If a helper file
changes during the capture, no seal is issued.

A trace never depends on test order or scheduling:
- each test's devices are numbered from 1 in creation order, under
  `aux/tests/TEST/`;
- module payloads are named by their bytes (`aux/modules/FNV-LENGTH.bin`);
- `index.json` holds only relative paths, sorted keys and nothing about the
  runner.

So serial, threaded and per-process captures of one revision write
byte-identical journals, `aux/` and `index.json`. Only `capture.json`, the
logs and the timing files differ.

`capture.json` records:
- runner and jobs;
- build and test seconds with peak RSS;
- the patches;
- the outcome.

`test-processes.tsv` lists each test process's seconds, peak RSS and
command.

## Captures and the seal

A capture is a directory holding:
- the journal bodies, each with its `.pin` and (v3) `.site` sidecar;
- `aux/`: device traces, checkpoints, create exports, module payloads and
  `manifest.json` (the bindings);
- `index.json`: schema `chopinfer.journal-captures.v3`, or `.v2` from the
  earlier external helper.

The index records:
- the model and revision;
- the helper's commit and file digests;
- every body's pin, call site and SHA-256;
- every auxiliary file's SHA-256;
- the passing test count.

A local HMAC seals the index. The authority key is maintainer tooling:
it authenticates captures within one maintainer's machine, and
contributors do not need one. The key lives in
`$CHOPINFER_JOURNAL_AUTHORITY` (default `$CHOPINFER_DATA/journal-proof/authority`):
a 0700 directory holding a 0600 `key` file, outside the repository. A
capture never names its own signer. Sealing never creates the key: when the
directory or key is missing, a capture refuses to seal and names the
configured key path. Create the key once, with the same environment the
captures use:

```sh
python3 tools/journal_proof/provenance.py init
```

`init` creates the directory and a fresh 32-byte key and prints only the key
path. It refuses to replace an existing key. It refuses a symlink at the
authority directory itself or at the key; parent directories of the
authority are followed. A new key cannot verify captures sealed under an
earlier one, so move an existing authority rather than creating a new one.

The proof rejects unsigned, modified or foreign indexes, and any auxiliary
file whose digest changed. This trusts the local account and helper
installation. It does not protect against that account forging receipts.
