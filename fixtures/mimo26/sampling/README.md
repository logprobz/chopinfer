# Real MiMo logits rows for sampling tests

`real-logits-4x152576.bf16` holds four rows of MiMo V2.6 Flash logits,
152,576 little-endian BF16 values each (round to nearest even from F32).
They come from the numerics audit's full-model prompt positions
(`$CHOPINFER_DATA/numerics-audit/full`): text 0 position 0, the flattest row
(entropy 9.4 nats, 47,269 tokens in the top-p 0.95 set); text 2 position 37,
the most peaked (one token); text 2 position 32, a median row (60 kept
tokens); and text 2 position 42 (601 kept tokens).

`sampling_filtered_real_rows_match_cpu` in `crates/kernel-lab/tests/ops/sampling.rs`
compares the filtered device path with the CPU oracle on these rows.
