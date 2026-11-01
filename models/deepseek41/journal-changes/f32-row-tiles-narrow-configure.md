# Configure the narrow compressor row tile wherever it loads

A target projection's verify row tiles now hold the site's measured tiles
from the 33rd row to the instance's last admitted row, and still serve only
the admitted rows. An instance with the wide verify buckets and without
`f32_row_tiles` admits 65 rows and up, so its compressor op now also holds
the 33- to 64-row tile,
`chop_deepseek41_verify_f32_n512_k5120_rows_f64_r32_w32x16_s3_k64_z8_v1`,
and the instance configures that function's 61440 bytes of dynamic shared
memory when it loads the module. Nothing launches the tile there: the op
serves 65 rows and up as before, so the configure only raises a limit no
launch reads. The head's tile already covered 33 to 192 rows and the router
has no tile below 65, so their ops are unchanged, and so are every
workspace size, launch and route.

The [spec](f32-row-tiles-narrow-configure.json) declares the whole change as
one module replacement of `f32_verify_rows_deepseek41_sm120` by itself: both
revisions' lock entries and payload SHA-256 are identical, its resolves keep
their locked order, and that one symbol's configure list goes from none to
61440. Every fixture takes the empty auxiliary rule, with no layout, journal
policy, handle or operation, so every other record must match exactly. The
change reaches the four fixtures that load the module, the C16 odd-width and
C32 maximum allocation fixtures with `wide_verify` and with `wide_draft`
too: one configure note on each of their four ranks, four lines a journal.
Every other journal, create range and device history is identical.

The `f32_row_tiles` flip that follows serves this tile at 33 to 64 rows, so
it configures nothing this change has not already configured.
