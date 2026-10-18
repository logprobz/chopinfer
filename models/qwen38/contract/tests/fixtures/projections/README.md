# Projection header fixtures

`fp8.json` records names, source dtypes, stored shapes and the precision
rules of the active FP8 export's headers, surveyed on 2026-10-04. It
contains no weights or measured outputs. Family axes expand original layer,
expert and shard selections independently of the declaration. The vision
tower families were added after the survey: the pinned FP8 revision's index
(checkpoint-files.json) names them.

Tests derive two mixed layouts from it at test time
(`../../projections_support/exports.rs`), each with every weight at its
logical shape and the quantization_config its rules imply:

- `nvfp4-routed-fp8-mtp`: NVFP4 routed experts with input scales,
  per-tensor FP8 lookup tables, block FP8 MTP experts with BF16 inverse
  scales, BF16 elsewhere and no vision tower.
- `mxfp8-dense-nvfp4`: MXFP8 attention, linear-attention and shared-expert
  projections, NVFP4 routed experts and lookup tables, weight-only NVFP4 MTP
  experts, 100 F32 hyper-connection and PLE norm weights, and the BF16 vision
  tower.

Counts are 152089, 299212 and 301593 tensors. FP8 and the MXFP8 layout hold
the 333 BF16 vision tower tensors. Optional encoder ownership admits an
entirely absent tower and refuses a partial or unattested tower. Bound norm
views retain the source dtype. Shared norm arithmetic applies only to the
explicitly declared HC/PLE and pre-FC paths; other kernels own their norm
arithmetic.

Shared HeaderCompanionRules imports the stored weight recipes and bounded part
views. Declarative hints identify BF16 roles and table-wide companion names.
Format-keyed lookup scope selects table-local FP8 or shard-local NVFP4 scales.
The lookup kind itself requires weight-only import. No quantization config is
rewritten by the tests.

Full-header tests use CHOPINFER_PROJECTION_HEADERS_QWEN38, selecting one
checkpoint or a directory whose children are checkpoints. They read confined
headers and bind closed coverage. The FP8 export's tensor names, dtypes,
shapes and quantization config must equal `fp8.json`; a checkpoint with a
mixed layout's tensor count must hold exactly that layout's tensors. They
print a skip reason when unset. These checks are metadata evidence; the
declarations execute no arithmetic or serving.
