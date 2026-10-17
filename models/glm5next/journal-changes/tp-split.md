# TP-split declaration: BF16 KV coverage

`tp-split.json` replaces the MoE all-reduce inboxes, flags and epochs with
added F32 buffers. Two kinds of device journal exercise those buffers:

- the lowering journals (`journal_tests.rs`, `context_parallel_tests.rs`)
  are plan-only devices: they never run the instance create, record only
  allocations and `launch rN` plan records, and are admitted by the
  plan-only exception of the unzeroed-use rule
  (tools/journal_proof/README.md, "Added allocations");
- the served and fake-host journals run the instance create and execute
  their launches. The declared zeros and their order are checked there.

Every executing journal uses the FP8 latent KV cache. The BF16 KV format
appears only in the lowering journals, so no executing journal checks the
BF16 program under the executing rules. The FP8 journals cover it because the
KV format changes no record that touches a replaced buffer.

Checked on 2026-10-17 on the base captures of the change, over the three
lowering devices of each KV format (rows 2048, 8 and 32): handles were
normalized to their allocation size and ordinal, and each journal was
projected onto the records that name a buffer any declared collective
replaces. The projections are identical between the formats (364 records
each, over the same 12 replaced buffers). Of the 352 to 440 records that
differ between the formats, none names a replaced buffer. The KV format
changes attention and KV-cache launches only, and the readers and writers of
the replaced buffers, in their order, are the same as in the FP8 program that
the executing journals check.

The mutation harness (`tools/journal_proof/mutations.py pair`) also runs the
BF16 lowering journals made executing by one inserted executed launch: the
proof then refuses them, since their plan records write the replaced buffers
before the MoE reduce reads them.
