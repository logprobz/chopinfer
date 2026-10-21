"""Pins MiMo's image decoding to transformers' `load_image`.

tools/refrun/image_decode.py's `model_capture` runs `load_image` (the
served reference loader) over runtime-media's image decode corpus, checks
that the checkpoint's image processor, with the qualification's declared
overrides, adds no decode step to `load_image`'s image, and checks the
image probes of `multimodal_cases.py`, read from the functional fixtures and
authenticated by their manifest. It writes fixtures/mimo26/image-decode.json.
CPU only; no model runs:

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B fixtures/mimo26/capture/image_decode.py \
        --checkpoint $CHOPINFER_MIMO_CHECKPOINT --fixtures $CHOPINFER_MIMO_FIXTURES/functional

`frontend_media.py` is left alone: its digest is part of the media
references' processor identity.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.refrun import image_decode, preprocessing
from multimodal_cases import cases, verified_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "fixtures"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    overrides = preprocessing.declared_overrides(json.loads((ROOT / "models/mimo26/qual.json").read_text()))
    processor = preprocessing.processor(args.checkpoint, overrides)
    probes = {case["file"]: verified_bytes(case, args.fixtures)[0] for case in cases() if case["kind"] == "image"}
    image_decode.model_capture("fixtures/mimo26/capture/image_decode.py", processor, probes,
                               ROOT / "fixtures/mimo26/image-decode.json")


if __name__ == "__main__":
    main()
