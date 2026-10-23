# Packed fold weight codes

`mxfp8_fold_packed.cuh` addresses E4M3 weight codes stored as
`mxfp8-fold-packed-8x128-v1`, the layout the stream, direct and dual folds
read. The load-time transform and its CPU oracle are
`kernel_core::weight_transform::mxfp8_fold`; `Mxfp8FoldLayout::word_offset`
is the host twin of the header's `word_offset`, and its test checks every
word of several shapes against the packer.

Rows count across groups, N is a multiple of 8 and K a multiple of 64. An
eight-row strip is 8 * K contiguous bytes: its 128-K folds in order, 1024
bytes each, then a 512-byte fold when K % 128 == 64. Within a fold, strip
row r owns the 128-byte segment at r * 128 (64 bytes at r * 64 in a final
half fold). The word of K bytes [16c + 4l, 16c + 4l + 4) of the fold sits
at byte 32l + 4c of the segment (16l + 4c in a half fold), so the four
lanes of an MMA quad that read one row's K words own contiguous 32-byte
runs of it. Scales stay row-major UE8M0 rows.

## Porting a row-major reader

A packed entry is its row-major entry with only the weight-code address
changed. Nothing else moves: parameters, launch plan, dynamic shared
memory, arithmetic and accumulation order stay the same, so the packed
entry's outputs, status words, workspace and state equal the row-major
entry's bit for bit. The v2 fold (`mxfp8_fold.cuh`, `run<..., PackedWeights>`)
is the reference port.

- Device code takes a `bool PackedWeights = false` template parameter.
  Row-major instantiations keep their source text and must compile to the
  same cubin bytes, so existing locks stay unchanged; check them before
  anything else.
- A kernel that loads K words from global memory replaces `row * K + k`
  with `word_offset(row, k, K)` (or `load_word`). A kernel that stages
  16-byte row chunks in shared memory stages the strip's contiguous fold
  instead (64 16-byte copies per full fold) and reads its words at
  `segment_word`. A 64-K final fold has half-width segments.
- On the host, the family's `Entry` gains `packed_spec()`, which refuses
  shapes the layout cannot hold (`gemm::mxfp8_fold_packed::check`) and
  returns the spec named `gemm::mxfp8_fold_packed::symbol(&symbol)`, the
  row-major symbol plus `_packed_v1`. A prepared op then moves to packed
  codes by renaming its weight-reading launches
  (`gemm::mxfp8_fold_packed::launches`) and binding the packed view. An op
  whose launch arguments must change (a TMA tensor map over the codes)
  states its packed arguments itself.
- Row-major entries and cubins are not edited. Packed entries go in new
  cubins with new lock families, beside row-major twins at the same shapes.

## Serving packed codes

A model whose weight plan stores packed codes builds each serving fold
cubin with its weight readers replaced by their packed twins
(`gemm::mxfp8_fold_packed::replace`) and serves them through
`Sm120::with_packed_fold_weights`, which renames each registered entry's
weight-reading launches and declares the packed layout for its weight
roles. An op the model prepares itself takes `backend::packed_op`. A
registered entry with no packed form is refused, never run on packed
codes.

## Lab evidence

Each ported family adds a kernel-lab family `<family>_packed`, whose cubin
holds row-major entries and their packed twins at shapes that cover the
family's tilings, a final 64-K fold, several strips and several groups.
Its tests (see `crates/kernel-lab/families/tests/mxfp8_fold_packed.rs`):

- a CPU test that each packed spec differs from its row-major spec only in
  the packed template argument and its symbol;
- a GPU test that runs both through `kernel_harness::packed_fold::compare`
  at several row counts, with codes whose every word is distinct, so a read
  from the wrong strip, group, fold or lane changes an output;
- a GPU negative control: packed codes with one strip moved must fail the
  comparison, which the eight-row oracle cases cannot detect.

Speed is checked against the row-major entry at the consuming model's
shapes; a packed entry must be no slower. The conformance cases of the
row-major entry remain its correctness evidence.
