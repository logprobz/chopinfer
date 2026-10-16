# Projection header fixtures

These fixtures record names, source dtypes, stored shapes and exporter precision
rules from the active FP8, NVIDIA mixed and local-lab mixed header copies
surveyed on 2026-10-04. They contain no weights or measured outputs. Family axes
expand original layer, expert and shard selections independently of the
declaration. Tests also compare the existing compiler mixed examples.

Counts are 152089, 299212 and 301593 tensors. FP8 and local-lab hold the 333
BF16 vision tower tensors, and local-lab also 100 F32 norm weights. The FP8
tower families were added after the survey: the pinned FP8 revision's index
(checkpoint-files.json) names them, and their headers equal local-lab's.
Optional encoder ownership admits an entirely absent tower (NVIDIA) and refuses
a partial or unattested tower. Bound norm views retain the observed source
dtype. Shared norm arithmetic applies only to the explicitly
declared HC/PLE and pre-FC paths; other kernels own their norm arithmetic.

Shared HeaderCompanionRules imports the stored weight recipes and bounded part
views. Declarative hints identify BF16 roles and table-wide companion names.
Format-keyed lookup scope selects table-local FP8 or shard-local NVFP4 scales.
The lookup kind itself requires weight-only import. No quantization config is
rewritten by the tests.

Full-header tests use CHOPINFER_PROJECTION_HEADERS_QWEN38, selecting one checkpoint
or a parent with fp8, nvidia and local-lab children. They read confined headers,
compare every included tensor name/dtype/shape and quantization config with the
fixture, then bind closed coverage. They print a skip reason when unset. These
checks are metadata evidence; the declarations execute no arithmetic or serving.
