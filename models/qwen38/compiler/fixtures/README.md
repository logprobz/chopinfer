# Qwen3.8 checkpoint captures

Configuration and tensor families come from revision
`236dfdf285828023ca3bcd3f37366c58a3469b13` of
`Qwen/Qwen3.8-Flash-Next-FP8`. Tensor families were captured by reading only
safetensors headers. The captured tensor families form an independent golden for the catalog
generated from execution geometry. Real headers are admitted against that catalog.

Tokenizer goldens were captured with the cached Hugging Face Transformers
5.16.1 `PreTrainedTokenizerFast(tokenizer_file=...)` loading the pinned
`tokenizer.json` unchanged, and `encode(text, add_special_tokens=False)`.
`AutoTokenizer` in this cache reconstructs an older Qwen2 regex without
combining-mark grouping, so it is not the authority for this capture.
No weights or GPU were used. The tokenizer contract records SHA-256 hashes
and lengths of both files. The regex includes Unicode combining marks.

Chat goldens use the same fast tokenizer with the checkpoint's
`chat_template.jinja` assigned to `chat_template`, then
`apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
The goldens include the full message inputs, tools, keyword arguments,
rendered text and token IDs. They cover default/xhigh/medium/low reasoning,
disabled thinking, system trimming, preserved history, omitted tool fields,
multiple XML calls and grouped tool replies. They render text only.

Chat template SHA-256: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

Checkpoint-specific pins and tokenizer/chat captures live under
`checkpoints/qwen3.8-flash-next-fp8/`. `checkpoints/index.json` lists each
checkpoint directory, its checkpoint and tokenizer environment variables,
the execution geometry, and the captured tokenizer and main chat case counts.
Tests iterate over the index and load tokenizers through LocalFrontend.
The tokenizer contract pins remain provenance and match the served contract
for the reference revision. `catalog-contract.json` records the captured
header count. Add a directory and index entry to test another checkpoint of this template family without
changing test code. Set its environment variables to execute the asset checks.

`tokenizer-semantics.json` stays at family level. The `contributor-*.json`
files are synthetic exporter metadata and small tensor inputs, so they also
stay at family level. `tensor-families.json` is the independent FLASH catalog
golden, including the captured dtype and shape of every tensor family.
