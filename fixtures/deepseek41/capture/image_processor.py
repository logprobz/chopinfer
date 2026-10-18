"""Goldens for runtime-media's `image.centered-pad.v1` recipe.

Runs DeepSeek V4.1 Flash's reference image preprocessing (the checkpoint's
inference/image_processor.py: plan_image_grid, ImageOps.pad with gray 127,
the (v / 255 - 0.5) / 0.5 normalization and the patch reshape) on CPU over
synthetic images, and writes crates/runtime-media/tests/fixtures/centered-pad.json
with the SHA-256 of each image's F32 rows and of the reference's BF16 patches,
its plan and token grid, plus the plan of a sweep of source sizes. The plan
sweep calls the reference's plan_image_grid and Pillow's own ImageOps.pad on
stand-in images that record sizes and paste offsets, so no pixels are
resampled for it. CPU preprocessing only; no model runs:

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B fixtures/deepseek41/capture/image_processor.py \
        $CHOPINFER_DEEPSEEK_CHECKPOINT

The synthetic patterns match `pattern()` in the runtime-media test.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import random
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import PIL
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "crates/runtime-media/tests/fixtures/centered-pad.json"

# (name, pattern, width, height)
CASES = [
    ("keep-560x560", "noise", 560, 560),
    ("keep-wide-1386x700", "blocks", 1386, 700),
    ("upscale-1x1", "noise", 1, 1),
    ("upscale-odd-33x17", "noise", 33, 17),
    ("upscale-pad-rows-200x100", "blocks", 200, 100),
    ("upscale-pad-columns-150x400", "noise", 150, 400),
    ("upscale-content-1380x690", "noise", 1380, 690),
    ("downscale-pad-901x613", "noise", 901, 613),
    ("solve-2000x1500", "noise", 2000, 1500),
    ("solve-tall-1100x2600", "blocks", 1100, 2600),
    # 79 * (42 / 79) falls just under 42, so the solved canvas is 2
    # patches wide.
    ("solve-rounding-79x15000", "noise", 79, 15000),
    ("long-row-10000x10", "noise", 10000, 10),
    ("solve-thin-45000x60", "noise", 45000, 60),
    ("collapse-row-50000x40", "blocks", 50000, 40),
    ("collapse-column-10x30000", "noise", 10, 30000),
    # Pillow's Image.resize shrinks a source more than 100 times taller
    # than wide vertically first.
    ("tall-ratio-100-300x30000", "noise", 300, 30000),
    ("tall-ratio-over-100-299x30000", "noise", 299, 30000),
]


def pattern(kind, width, height):
    y, x = np.mgrid[0:height, 0:width].astype(np.uint64)
    channels = []
    for c in range(3):
        c = np.uint64(c)
        if kind == "noise":
            h = ((x * np.uint64(0x9E3779B1)) ^ (y * np.uint64(0x85EBCA77)) ^ (c * np.uint64(0xC2B2AE3D))) & np.uint64(
                0xFFFFFFFF
            )
            h ^= h >> np.uint64(15)
            h = (h * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF)
            h ^= h >> np.uint64(12)
            channels.append((h & np.uint64(255)).astype(np.uint8))
        elif kind == "blocks":
            channels.append(((x // np.uint64(7) * np.uint64(37) + y // np.uint64(5) * np.uint64(59) + c * np.uint64(85)) % np.uint64(256)).astype(np.uint8))
        else:
            raise ValueError(kind)
    return Image.fromarray(np.stack(channels, axis=-1), "RGB")


class Stand:
    """An image of a given size that records resizes and pastes."""

    palette = None
    mode = "RGB"

    def __init__(self, size):
        self.size = tuple(size)
        self.width, self.height = self.size
        self.pasted = None

    def resize(self, size, resample=None):
        return Stand(size)

    def paste(self, image, box):
        self.pasted = (box, image.size)


def planned(module, args, width, height):
    """The reference's plan and Pillow's pad geometry for a source size."""
    n_llm_h, n_llm_w, best_height, best_width = module.plan_image_grid(width, height, args)
    with patch.object(Image, "new", lambda mode, size, color=None: Stand(size)):
        out = ImageOps.pad(Stand((width, height)), (best_width, best_height), color=(127, 127, 127))
    if out.pasted is None:
        (x, y), (content_w, content_h) = (0, 0), out.size
    else:
        (x, y), (content_w, content_h) = out.pasted
    return [best_height, best_width, content_h, content_w, y, x, n_llm_h, n_llm_w]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    options = parser.parse_args()
    source = options.checkpoint / "inference/image_processor.py"
    spec = importlib.util.spec_from_file_location("reference_image_processor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((options.checkpoint / "inference/config.json").read_text())
    args = types.SimpleNamespace(**{k: v for k, v in config.items() if k.startswith("vision_")})
    assert args.vision_max_wh_ratio is None
    p = args.vision_patch_size

    rng = random.Random(41)
    sides = [1, 2, 13, 14, 15, 41, 42, 43, 100, 383, 384, 543, 544, 545, 587, 588, 589, 1000, 1343, 1344, 1345, 2048, 4097, 9000]
    sizes = {(w, h) for w in sides for h in sides}
    while len(sizes) < 900:
        w = int(10 ** rng.uniform(0, 4.3))
        h = int(10 ** rng.uniform(0, 4.3))
        if max(w, h) / min(w, h) <= 30000:
            sizes.add((w, h))
    sweep = [[h, w] + planned(module, args, w, h) for (w, h) in sorted(sizes)]

    cases = []
    for name, kind, width, height in CASES:
        image = pattern(kind, width, height)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = module.load_image({"data": buffer.getvalue()}, args)
        # The reference's F32 values before its BF16 conversion, by the same
        # operations.
        plan = planned(module, args, width, height)
        padded = ImageOps.pad(image, (plan[1], plan[0]), color=(127, 127, 127))
        x = torch.from_numpy(np.asarray(padded, dtype=np.float32)).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        rows = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p)
        assert torch.equal(rows.to(torch.bfloat16).reshape(patches.shape), patches), name
        bf16 = patches.contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
        cases.append(
            {
                "name": name,
                "pattern": kind,
                "width": width,
                "height": height,
                "plan": plan,
                "grid": [1, n_vit_h, n_vit_w],
                "tokens": n_llm_h * n_llm_w,
                "span_tokens": module.num_image_tokens(n_llm_h, n_llm_w),
                "rows_sha256": sha256(rows.contiguous().numpy().astype("<f4").tobytes()),
                "bf16_sha256": sha256(bf16),
            }
        )

    golden = {
        "source": f"reference inference image_processor.py load_image, Pillow {PIL.__version__}, torch {torch.__version__}, numpy {np.__version__}, on CPU; SHA-256 of the F32 rows and of the reference's BF16 patches",
        "image_processor_sha256": sha256(source.read_bytes()),
        "recipe": {
            "patch": p,
            "downsample": args.vision_downsample_ratio,
            "minimum_pixels": args.vision_min_pixels,
            "maximum_tokens": args.vision_max_n_token,
            "pad": [127, 127, 127],
            "value_divisor": 255.0,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "plan_columns": ["height", "width", "canvas_h", "canvas_w", "content_h", "content_w", "offset_y", "offset_x", "token_rows", "token_columns"],
        "cases": cases,
        "sizes": sweep,
    }
    # One size per line keeps the sweep readable.
    rows = ",\n  ".join(json.dumps(row) for row in golden.pop("sizes"))
    text = json.dumps(golden, indent=1)
    text = text[: text.rindex("}")].rstrip() + ',\n "sizes": [\n  ' + rows + "\n ]\n}"
    OUTPUT.write_text(text + "\n")
    print(f"wrote {OUTPUT} with {len(cases)} cases and {len(sweep)} sizes")


if __name__ == "__main__":
    main()
