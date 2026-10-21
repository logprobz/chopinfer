# Qwen frontend reference producers

Use the configured `CHOPINFER_REFRUN_PYTHON` with local checkpoint files and
network access disabled. `images.py` produces the deterministic image inputs;
`prompts.sh` prepares the frozen teacher and needle prompts.

`python -m tools.refrun.frontend --checkpoint "$CHOPINFER_QWEN38_CHECKPOINT"`
with `--inputs fixtures/qwen38/frontend --output OUTPUT --tool-reasoning`
and `--images IMAGE_DIRECTORY` regenerates chat, tokenizer and image prompt
goldens on CPU. The image directory contains the named deterministic probes,
including blue-square.png and frame-00.png.

`frontend_api.py prepare` reads `frontend/api-requests.json` and writes an
HF token workload. Run it with `models/qwen38/reference.py` in F32 through
the GPU queue; `frontend_api.py shapes` derives the answer-dependent shape
expectations. See [the reference protocol](../frontend/README.md).

`image_decode.py` writes `../image-decode.json`, which pins image decoding
to transformers' `load_image` over runtime-media's image decode corpus and
checks that the image processor and the image probes add nothing to it.

`image_rows.py` and the model-owned `reference_workloads.py` retain the
authenticated image-probe workload producer. Accuracy answers are generated
and bound by `tools/refrun`, never by a serving-system baseline.
