# kernel-core

Pure Rust, model-neutral SM120 kernel contracts and reference arithmetic. This
crate has no CUDA runtime dependency and does not launch kernels.

- `contract`: typed op queries, refusal rules, static compile keys, launch and ABI arguments.
- `codec`: BF16, FP8, FP6, FP4 and shared-scale scalar codecs; `kv_fp8`: FP8 E4M3 KV cache storage.
- `quantize`: MXFP8, MXFP4, NVFP4 and block/group FP8 reference quantizers.
- `precision`: portable checkpoint precision descriptions a consumer binds to an operation contract.
- `oracle` and the `*_oracle` modules: decoded-input CPU (mostly f64) oracles for attention, sparse
  attention, gated attention, recurrent (linear attention), MoE, hyper-connection, n-gram/PLE and
  multimodal encoder contracts, with numerical checks.
- `attend`, `rotary`, `hf_rope`, `rmsnorm`, `select`, `head`: shared reference building blocks
  (index-list attention core, rotary tables, RMSNorm requirements, top-k with explicit tie and NaN
  rules, streaming vocabulary-head scoring).
- `transform` and `weight_transform`: chunked scale grids, MMA scale tiles, and the closed set of
  exact weight transforms a direct checkpoint load executes.
- `topology` and `rank_shape`: TP1/2/4/8/16 source and padded storage ranges, index ownership,
  KV duplication groups, and per-rank shapes derived from a model's configuration.
- `row_exchange`: buffer layout, launch geometry and arguments of the row exchange.

Numerical IDs are stable strings ending in `.v<N>`. GEMM IDs use
`gemm.w-<weight>.a-<activation>.acc-f32.v1`, for example
`gemm.w-mxfp4.a-bf16.acc-f32.v1`. A16 and A8 paths have different activation
segments. Exact F32 and power-of-two FP8 scale policies have different IDs.

MX formats and scales follow the [OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). NVFP4 uses the [NVIDIA hierarchical scale formula](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) with deterministic RNE for this inference reference.
