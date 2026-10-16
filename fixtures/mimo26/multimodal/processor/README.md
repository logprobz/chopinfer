# Fixed inputs and HF processor arrays

`video-rgb.npy` is fixed input data: eight 256 by 256 RGB frames, one per
second, with four identical temporal pairs. Its full file SHA-256 remains
`97554e4fd1d7ece4b24fca8a9fde96c79f1b422193b82ce44ffea8d7edeef168`.
The original capture decoded the functional `red-then-blue.mp4` with OpenCV
in the vLLM capture environment. On 2026-10-12 these bytes were retained as
raw input, not as a correctness judgment. No replacement video decoder is
used during regeneration.

`capture/frontend_media.py` authenticates these bytes before feeding the
frames to the HF image processor. Identical frame pairs make its temporal
patchification equal to processing each pair's first image. The processor
arrays, grids and expanded prompt IDs are regenerated on CPU. ImageNet
normalization and pixel bounds remain the declared serving choices. No
audio input is kept: MiMo's audio is served but not qualified in-tree.
F32 greedy answers and forced F32/BF16/F64 scores come from `tools/refrun`.

The served API accepts the mp4 and decodes it with NVDEC. Existing CPU tests
check these fixed RGB bytes and subsequent preprocessing, but do not prove
that NVDEC's decoded RGB equals these frames. That equality remains an open
item; no extra GPU decode job was run for this migration.

The producer manifests describe its complete output. This directory stores
only the arrays read by compiler tests; other generated arrays stay with the
external reference workload. Their absence here does not change the producer
identity or the authenticated hashes of the stored test inputs.
