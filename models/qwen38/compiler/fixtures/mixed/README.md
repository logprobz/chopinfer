# Mixed export metadata

These fixtures retain selected configuration rules and tensor header shapes
from the NVIDIA text-only NVFP4 export (ModelOpt 0.46.0.dev281) and the
local-inference-lab export at revision `6909a5bed`. Neither declares
`quantization_config.weight_block_size`.

Each synthetic index assigns the selected tensors to one shard. Tests build
that shard from `tensors.json` as a sparse zero file outside the repository
and read only metadata. Configurations keep the checkpoint geometry; their
quantization rules cover only the selected roles.

The NVIDIA fixture includes NVFP4 decoder experts, BF16 projections and
norms, per-tensor FP8 PLE tables, and block FP8 MTP experts. The local-lab
fixture includes MXFP8 projections, NVFP4 decoder experts and PLE tables,
W4A16 NVFP4 MTP experts, and an F32 hyper-connection norm that admission
keeps in F32.
These fixtures test import contracts, not numerical or serving qualification.
