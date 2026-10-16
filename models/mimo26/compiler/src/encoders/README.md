# MiMo V2.6 Flash encoder images

The encoder image is planned apart from the text image, and every rank holds
the same image, laid out as the encoder kernels read it
(`model_mimo26_contract::encoders::prepared_for`). The bundle binds its views in
place, with no copy of the weights. `encoders::image::estimated_rank_bytes`
sizes it from the checkpoint's 848 inference weights.

The catalog validates 364 vision tensors, 75 audio encoder tensors, 20 speech
embedding tables, and 389 audio tokenizer encoder tensors. It excludes the
audio decoder and EMA codebook buffers. All views are replicated on every
rank. The image holds 676 tensors (schema `chopinfer.mimo26.encoder-image.v2`):

- For each GEMM step of the encoder program, one tensor of its weights,
  stacked as stored and named for the first. With it, one tensor of their
  biases, named for the first weight's bias, with zeros for a weight that
  has none.
- The 20 `_codebook.embed` tensors, rounded from F32 to BF16 as the vLLM fork
  does after loading, as one `@codebooks` tensor. The 20 speech embedding
  tables, as one `@embeddings` tensor.
- Every other weight as stored, with two exceptions. Convolution weights
  (`(out, channel, tap)`) put their taps before their channels, the im2col
  column order. Attention sinks are widened to F32 (`f32_row_major_v1`).

The merger has an RMSNorm weight and two bias-free linear weights. This follows
the checkpoint and vLLM implementation; the HF merger's LayerNorm and biases
would request tensors that are absent (see
[serving-mimo26.md](../../../../../docs/serving-mimo26.md)). The image is about
2.6 GB per rank for the pinned checkpoint, with 4 KiB view alignment.
