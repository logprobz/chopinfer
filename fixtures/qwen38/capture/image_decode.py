"""Pins Qwen3.8's image decoding to transformers' `load_image`.

tools/refrun/image_decode.py's `model_capture` runs `load_image` (the
served reference loader) over runtime-media's image decode corpus, checks
that the checkpoint's image processor adds no decode step to `load_image`'s
image, and checks the image probes `images.py` builds (its sizes and the
functional blue-square and frame-00). It writes
fixtures/qwen38/image-decode.json. CPU only; no model runs:

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B fixtures/qwen38/capture/image_decode.py \
        $CHOPINFER_QWEN38_CHECKPOINT
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.refrun import image_decode, preprocessing
import images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    processor = preprocessing.processor(args.checkpoint)
    probes = {**{name: images.png(*size) for name, size in images.SIZES.items()}, **images.functional()}
    image_decode.model_capture("fixtures/qwen38/capture/image_decode.py", processor, probes,
                               ROOT / "fixtures/qwen38/image-decode.json")


if __name__ == "__main__":
    main()
