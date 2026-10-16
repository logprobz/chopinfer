"""Goldens for runtime-media's `image.zero-padded-token-budget.v1` recipe.

Runs GLM 5.3 Flash's pinned image processor (transformers glm5_next,
AutoImageProcessor from the checkpoint's processor_config.json, which picks
the torchvision backend) on CPU over synthetic images, and writes
crates/runtime-media/tests/fixtures/zero-padded-token-budget.json with the
SHA-256 of each image's F32 pixel rows, its grid and placeholder count, plus
the canvas and content sizes of a sweep of source sizes. Grayscale, alpha
and tRNS cases are written as PNG files beside it and read back through PIL
as the processor's callers do; PIL's convert("RGB") discards their alpha.
CPU preprocessing only; no model runs:

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B fixtures/glm5next/capture/image_processor.py \
        $CHOPINFER_GLM5NEXT_CHECKPOINT [--dump DIR]

`--dump` also saves each case's rows as raw little-endian F32 for diffing.
The synthetic patterns match `pattern()` in the runtime-media test.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoImageProcessor
from transformers.image_processing_backends import TorchvisionBackend

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "crates/runtime-media/tests/fixtures"
NAME = "zero-padded-token-budget"

# (name, pattern, width, height, mode); mode "RGB" cases are built in memory,
# the others are written as PNG files. "RGBA" and "LA" take their alpha from
# the pattern's next channel unless named opaque; "P", "RGB-tRNS" and
# "L-tRNS" carry a tRNS chunk.
CASES = [
    ("upscale-10x10", "noise", 10, 10, "RGB"),
    ("upscale-odd-33x17", "noise", 33, 17, "RGB"),
    ("upscale-100x60", "blocks", 100, 60, "RGB"),
    ("upscale-content-100x100", "noise", 100, 100, "RGB"),
    ("below-12544-113x111", "noise", 113, 111, "RGB"),
    ("at-12544-112x112", "noise", 112, 112, "RGB"),
    ("at-12544-odd-196x64", "blocks", 196, 64, "RGB"),
    ("above-12544-120x105", "noise", 120, 105, "RGB"),
    ("pad-odd-301x199", "noise", 301, 199, "RGB"),
    ("aligned-280x196", "blocks", 280, 196, "RGB"),
    ("wide-3000x20", "noise", 3000, 20, "RGB"),
    ("tall-15x2000", "blocks", 15, 2000, "RGB"),
    ("budget-exact-2800x2240", "blocks", 2800, 2240, "RGB"),
    ("budget-search-2801x2240", "noise", 2801, 2240, "RGB"),
    ("budget-search-wide-7000x900", "blocks", 7000, 900, "RGB"),
    ("budget-search-4000x3000", "noise", 4000, 3000, "RGB"),
    ("gray-45x31", "noise", 45, 31, "L"),
    ("gray-alpha-opaque-150x90", "blocks", 150, 90, "LA"),
    ("rgba-opaque-64x50", "noise", 64, 50, "RGBA"),
    ("rgba-translucent-64x50", "noise", 64, 50, "RGBA"),
    ("rgba-transparent-black-112x112", "zero", 112, 112, "RGBA"),
    ("gray-alpha-translucent-45x31", "noise", 45, 31, "LA"),
    ("palette-trns-48x36", "noise", 48, 36, "P"),
    ("rgb-trns-50x20", "blocks", 50, 20, "RGB-tRNS"),
    ("gray-trns-33x21", "blocks", 33, 21, "L-tRNS"),
]


def pattern(kind, width, height, channels):
    """uint8 [height, width, channels], from integer math only."""
    y, x, c = np.meshgrid(np.arange(height, dtype=np.uint64), np.arange(width, dtype=np.uint64),
                          np.arange(channels, dtype=np.uint64), indexing="ij")
    mask = np.uint64(0xFFFFFFFF)
    if kind == "noise":
        h = (x * np.uint64(0x9E3779B1) ^ y * np.uint64(0x85EBCA77) ^ c * np.uint64(0xC2B2AE3D)) & mask
        h ^= h >> np.uint64(15)
        h = (h * np.uint64(0x2C1B3C6D)) & mask
        h ^= h >> np.uint64(12)
        return (h & np.uint64(255)).astype(np.uint8)
    if kind == "zero":
        return np.zeros((height, width, channels), np.uint8)
    if kind == "blocks":
        return ((x // np.uint64(7) * np.uint64(37) + y // np.uint64(5) * np.uint64(59)
                 + c * np.uint64(85)) % np.uint64(256)).astype(np.uint8)
    raise ValueError(kind)


def image(name, kind, width, height, mode):
    if mode == "RGB":
        return Image.fromarray(pattern(kind, width, height, 3), "RGB"), None
    buffer = io.BytesIO()
    if mode == "P":
        indices = pattern(kind, width, height, 1)[:, :, 0]
        picture = Image.fromarray(indices, "P")
        picture.putpalette(pattern("noise", 256, 1, 3).reshape(-1).tolist())
        picture.save(buffer, format="PNG", transparency=bytes(i * 7 % 256 for i in range(256)))
    elif mode in ("RGB-tRNS", "L-tRNS"):
        gray = mode == "L-tRNS"
        pixels = pattern(kind, width, height, 1 if gray else 3)
        key = int(pixels[0, 0, 0]) if gray else tuple(int(v) for v in pixels[0, 0])
        picture = Image.fromarray(pixels[:, :, 0] if gray else pixels, "L" if gray else "RGB")
        picture.save(buffer, format="PNG", transparency=key)
    else:
        color = 1 if mode in ("L", "LA") else 3
        pixels = pattern(kind, width, height, color + (mode in ("LA", "RGBA")))
        if "opaque" in name:
            pixels[:, :, color] = 255
        if mode == "L":
            pixels = pixels[:, :, 0]
        Image.fromarray(pixels, mode).save(buffer, format="PNG")
    data = buffer.getvalue()
    return Image.open(io.BytesIO(data)), data


def sizes(processor):
    """(height, width, canvas height, canvas width, content height, content width)."""
    chosen = {(1, 1), (1, 2000), (2000, 1), (27, 29), (111, 113), (112, 112), (2240, 2800), (2240, 2801)}
    for side in range(1, 230, 3):
        chosen.update({(side, side), (side, 2 * side + 1), (3 * side + 2, side)})
    generator = random.Random(28)
    while len(chosen) < 400:
        chosen.add((generator.randint(1, 6000), generator.randint(1, 6000)))
    captured = []

    def record(self, images, size, resample=None, **kwargs):
        captured.append((size.height, size.width))
        return torch.zeros(*images.shape[:-2], size.height, size.width, dtype=images.dtype)

    rows = []
    with patch.object(TorchvisionBackend, "resize", record):
        for height, width in sorted(chosen):
            captured.clear()
            canvas = processor.resize(torch.zeros(1, 3, height, width, dtype=torch.uint8),
                                      resample=processor.resample, factor=28, temporal_factor=2,
                                      min_image_tokens=processor.min_image_tokens,
                                      max_image_tokens=processor.max_image_tokens)
            content = captured[0] if captured else (height, width)
            rows.append([height, width, *canvas.shape[-2:], *content])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    if not isinstance(processor, TorchvisionBackend):
        raise SystemExit(f"expected the torchvision backend, got {type(processor).__qualname__}")
    settings = (processor.patch_size, processor.temporal_patch_size, processor.merge_size,
                processor.patch_expand_factor)
    if settings != (14, 2, 2, 1):
        raise SystemExit(f"unexpected processor geometry {settings}")
    config = json.loads((args.checkpoint / "processor_config.json").read_text())["image_processor"]
    golden = {
        # The fixture lives in a model-neutral crate, so it names no model.
        "source": (f"transformers {transformers.__version__} image processor, torchvision backend, torch "
                   f"{torch.__version__}, on CPU; SHA-256 of the F32 pixel_values rows"),
        "processor_config_sha256": hashlib.sha256((args.checkpoint / "processor_config.json").read_bytes()).hexdigest(),
        "recipe": {
            "patch": processor.patch_size, "temporal_patch": processor.temporal_patch_size,
            "merge": processor.merge_size, "minimum_tokens": processor.min_image_tokens,
            "maximum_tokens": processor.max_image_tokens, "value_scale": processor.rescale_factor,
            "mean": list(config["image_mean"]), "std": list(config["image_std"]),
        },
        "cases": [],
        "sizes": sizes(processor),
    }
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
    for name, kind, width, height, mode in CASES:
        source, png = image(name, kind, width, height, mode)
        entry = {"name": name, "pattern": kind, "width": width, "height": height,
                 "transparent": mode in ("P", "RGB-tRNS", "L-tRNS") or (
                     mode in ("LA", "RGBA") and "opaque" not in name)}
        if png is not None:
            entry["png"] = f"{NAME}-{name}.png"
            (OUTPUT / entry["png"]).write_bytes(png)
        out = processor(images=[source], return_tensors="pt")
        pixels = out["pixel_values"].numpy()
        if pixels.dtype != np.float32:
            raise SystemExit(f"{name}: pixel_values are {pixels.dtype}")
        pixels = np.ascontiguousarray(pixels)
        grid = out["image_grid_thw"].tolist()[0]
        entry.update({"grid": grid, "tokens": grid[0] * grid[1] * grid[2] // 4,
                      "rows_sha256": hashlib.sha256(pixels.astype("<f4").tobytes()).hexdigest()})
        if args.dump:
            pixels.astype("<f4").tofile(args.dump / f"{name}.f32")
        golden["cases"].append(entry)
        print(name, grid, entry["tokens"], flush=True)
    text = json.dumps(golden, indent=1)
    text = re.sub(r"\[\s+([^\[\]{}\"]+?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]",
                  text)
    (OUTPUT / f"{NAME}.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
