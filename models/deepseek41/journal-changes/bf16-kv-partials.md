# BF16 KV prefill partials journal declaration

BF16 KV storage at Flash TP4 now takes the native BF16 prefill partials, so
comparing the two KV layouts compares their stored KV alone. BF16 KV create
therefore needs the same resident BF16-partial grids as native, and refuses
the instance without them. The rounded F32 prefill gather BF16 KV prepared
before is retired; no configuration reaches it. Every native journal, and
every TP1 and TP2 journal, is byte-identical: 41 of the 44 pinned journals.

Three BF16 KV Flash TP4 journals change, each one fake-device lifetime:
`deepseek41-flash-tp4.bf16.journal`,
`deepseek41-flash-tp4.native-moe.bf16.journal` and
`deepseek41-flash-tp4.bf16-phased.journal`. In each, the declaration:

- replaces each rank's 184549376-byte gather inbox with a 92274688-byte
  BF16-partial inbox and its 33800-byte flags with 33864 bytes, the sizes
  native allocates; the four-byte epochs keep their identity;
- activates the two unchanged locked BF16-partial modules, with their three
  resolves per rank, and deactivates the rounded module with its two
  resolves, except in the phased lifetime, where the `phased_decode_reduce`
  candidate keeps loading it;
- replaces each of the 1620 prefill F32 reduce-scatter, all-gather and round
  triples (405 per rank) with one BF16-partial collective: 972 two-shot and
  648 one-shot launches.

The exact operation edits also cover the zeroes of the replaced buffers
and the releases of the added modules and resolves. An audit of the edits
pairs every removed triple with the added collective on its stream: it
reads the scatter's partial rows and writes the round's output rows. The
1620 added collectives equal the native journal's, in order, by stream,
launch configuration, scalars and span offsets and lengths. Decode,
verification and every other operation keep their records.

Creator declarations were derived from sealed captures of the base and of
the change, constrained to the resources above.
`tools/journal_proof/specs/operation_edits.py` generated the compact exact
operation edits. `layout_spec.py` and `proof.py` read this declaration from
the commit that adds it and check all pinned journals, creation records and
device history through each last pinned checkpoint.
