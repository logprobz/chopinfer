# Latent glue row status

The plain 256-thread F32 RMSNorm, latent query rotation and latent window
writer use the shared block quantizer's U32 flags. `status_row_offset` is
in words. Allocate `(offset + planned_rows) * 4` bytes, clear the selected
words once before the step, order launches that share those rows, and check
every selected word before accepting the step's outputs. Earlier flags and
inactive words survive. The status allocation must not alias any other role.

`NONFINITE_INPUT` (1) reports nonfinite source values, weights or rotation
tables. `UNREPRESENTABLE_OUTPUT` (4) reports nonfinite F32 arithmetic or a
nonfinite BF16 query result. The window's shared trained quantizer can add
its own flags, including a nonfinite BF16 quantizer input. These paths do
not trap on device data.

The failure behavior is explicit:

- Query rotation preserves the complete F32 query row and writes positive
  zero to every BF16 query head when any head or an earlier operation flags
  that row. One CTA validates all heads before writing them.
- The plain F32 norm writes a positive-zero output row on failure.
- Window normalization writes positive zero when it flags a row. The shared
  quantizer writes zero codes and finite scale 1 for any flagged row.
- The window store skips a flagged row. Both destination planes retain their
  previous bytes, including a previous circular cell. That cell does not
  become a valid key for the failed position. Do not advance retained bounds
  or accept a staged attention result for a failed step.

`LatentQueryRopeBuffers` and `LatentWindowWriteBuffers` require `status` and
`status_row_offset`. `RmsNormBuffers` takes `Some(status)` and the offset for
its plain F32 mode, selected by `numerics = Some(RmsNormNumerics::TREE_256)`.
Legacy BF16 entries use `None` and zero and retain their original ABI. The
existing pooling norm has its separate, backend-chosen numerical contract;
it does not use this plain norm entry.

Conformance declares sticky status as live-row state, moving its initial
flags with the data row for row, chunk and batch checks. Nonzero offsets are
also replayed separately with fixed guard words. NaN and infinity inputs,
prior flags, multi-head query failures and dropped-flag device controls are
part of the shared family cases.
