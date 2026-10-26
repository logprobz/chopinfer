# DSpark indexed retention default

Indexed retention now defaults on in serving and tests. A decode or verify
commit with at least eight retained-copy segments per rank uses one indexed
byte-copy launch per rank. Prefill, mixed prefill and smaller commits keep
the ordered scalar copy list. Every rank's source, destination and segment
mapping is validated before the first write.

Each pinned TP4 verify fixture adds one 768-byte segment buffer per rank,
with 256-byte alignment, during instance creation. The declaration binds
each buffer's issue ordinal, creator records, rank, size and stream. No
module, resolve, kernel lock or retained operation changes. These fixtures
commit their prompts through the scalar path and stop at the verify
checkpoint, before an indexed commit uses the new buffers.

The fake-host retention tests check exact wrapped segment mappings,
partially accepted cursors, the small-copy fallback and failed submission
before publication. The retention pair lab remains useful: it checks the
served primitive against an independent row oracle and scalar copies,
including poison preservation, consumers and replay.

The declaration comes from an ordered handle audit of all six complete
fixture histories. The sealed proof checks the exact creator positions
and every retained operation. Regenerate the positional declaration when
preceding resource creation changes.
