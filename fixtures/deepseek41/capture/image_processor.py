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

Sources go to the reference as encoded bytes: "RGB" cases as PNG built in
memory; grayscale, alpha, palette and tRNS PNGs, JPEGs, GIFs and WebPs are
also written beside the golden for the test to decode, as the reference's
Image.open and convert("RGB") read them (alpha and transparency discarded,
an EXIF orientation tag kept as stored, an animation's first frame). The
synthetic patterns match `pattern()` in the runtime-media test.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import random
import struct
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import PIL
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "crates/runtime-media/tests/fixtures"
OUTPUT = FIXTURES / "centered-pad.json"
NAME = "centered-pad"

# (name, pattern, width, height[, mode]); the mode defaults to "RGB".
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
    # Over 100 times taller than wide, growing: the horizontal pass runs
    # first.
    ("tall-grows-3x400", "noise", 3, 400),
    ("tall-grows-2x1000", "blocks", 2, 1000),
    # The span at exactly the 1024-token budget: 1 x 1021 tokens.
    ("exact-budget-42882x42", "noise", 42882, 42),
    ("gray-45x31", "noise", 45, 31, "L"),
    ("gray-alpha-translucent-45x31", "noise", 45, 31, "LA"),
    ("rgba-translucent-64x50", "noise", 64, 50, "RGBA"),
    ("rgba-transparent-black-112x112", "zero", 112, 112, "RGBA"),
    ("palette-trns-48x36", "noise", 48, 36, "P"),
    ("rgb-trns-50x20", "blocks", 50, 20, "RGB-tRNS"),
    ("gray-trns-33x21", "blocks", 33, 21, "L-tRNS"),
    ("jpeg-444-301x199", "blocks", 301, 199, "JPEG444"),
    ("jpeg-420-160x96", "blocks", 160, 96, "JPEG420"),
    ("jpeg-gray-64x48", "blocks", 64, 48, "JPEG-L"),
    ("jpeg-cmyk-256x256", "blocks", 256, 256, "JPEG-CMYK"),
    ("jpeg-cmyk-257x255", "noise", 257, 255, "JPEG-CMYK"),
    ("jpeg-progressive-444-301x199", "blocks", 301, 199, "JPEG-PROGRESSIVE444"),
    ("jpeg-progressive-420-161x97", "noise", 161, 97, "JPEG-PROGRESSIVE420"),
    # An EXIF orientation of 6 (rotate 90) stays 8 x 16 as stored.
    ("jpeg-exif6-8x16", "blocks", 8, 16, "JPEG-EXIF6"),
    ("jpeg-exif6-64x40", "blocks", 64, 40, "JPEG-EXIF6"),
    # Pillow interlaces a GIF unless told not to.
    ("gif-interlaced-64x48", "blocks", 64, 48, "GIF"),
    ("gif-plain-50x40", "noise", 50, 40, "GIF-PLAIN"),
    ("gif-trns-45x31", "noise", 45, 31, "GIF-TRNS"),
    # Frame 0 is the lower right quarter of a larger logical screen.
    ("gif-offset-frame-60x40", "blocks", 60, 40, "GIF-OFFSET"),
    ("gif-animated-48x32", "noise", 48, 32, "GIF-ANIMATED"),
    ("webp-lossy-301x199", "blocks", 301, 199, "WEBP-LOSSY"),
    ("webp-lossless-64x50", "noise", 64, 50, "WEBP-LOSSLESS"),
    ("webp-alpha-64x50", "noise", 64, 50, "WEBP-ALPHA"),
    ("webp-animated-48x32", "blocks", 48, 32, "WEBP-ANIMATED"),
]


def pattern(kind, width, height, channels=3):
    """uint8 [height, width, channels], from integer math only."""
    y, x, c = np.meshgrid(
        np.arange(height, dtype=np.uint64),
        np.arange(width, dtype=np.uint64),
        np.arange(channels, dtype=np.uint64),
        indexing="ij",
    )
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
        return ((x // np.uint64(7) * np.uint64(37) + y // np.uint64(5) * np.uint64(59) + c * np.uint64(85)) % np.uint64(256)).astype(np.uint8)
    raise ValueError(kind)


def palette_image(kind, width, height):
    picture = Image.fromarray(pattern(kind, width, height, 1)[:, :, 0], "P")
    picture.putpalette(pattern("noise", 256, 1, 3).reshape(-1).tolist())
    return picture


def relocated_gif(data, screen, offset):
    """A one-frame GIF with its logical screen set to `screen` and its frame
    moved to `offset`."""
    data = bytearray(data)
    data[6:10] = struct.pack("<HH", *screen)
    pos = 13 + (3 << ((data[10] & 7) + 1) if data[10] & 0x80 else 0)
    while data[pos] == 0x21:
        pos += 2
        while data[pos]:
            pos += data[pos] + 1
        pos += 1
    assert data[pos] == 0x2C
    data[pos + 1:pos + 5] = struct.pack("<HH", *offset)
    return bytes(data)


def encoded(name, kind, width, height, mode):
    """The source's bytes and its file extension, or None for an in-memory PNG."""
    buffer = io.BytesIO()
    if mode == "RGB":
        Image.fromarray(pattern(kind, width, height), "RGB").save(buffer, format="PNG")
        return buffer.getvalue(), None
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
    elif mode in ("L", "LA", "RGBA"):
        color = 1 if mode in ("L", "LA") else 3
        pixels = pattern(kind, width, height, color + (mode != "L"))
        Image.fromarray(pixels[:, :, 0] if mode == "L" else pixels, mode).save(buffer, format="PNG")
    elif mode in ("JPEG444", "JPEG420", "JPEG-EXIF6", "JPEG-PROGRESSIVE444", "JPEG-PROGRESSIVE420"):
        exif = Image.Exif()
        if mode == "JPEG-EXIF6":
            exif[0x0112] = 6
        Image.fromarray(pattern(kind, width, height), "RGB").save(
            buffer, format="JPEG", quality=92, subsampling=2 if mode.endswith("420") else 0,
            progressive=mode.startswith("JPEG-PROGRESSIVE"), exif=exif
        )
        return buffer.getvalue(), "jpg"
    elif mode == "JPEG-L":
        Image.fromarray(pattern(kind, width, height, 1)[:, :, 0], "L").save(buffer, format="JPEG", quality=92)
        return buffer.getvalue(), "jpg"
    elif mode == "JPEG-CMYK":
        Image.fromarray(pattern(kind, width, height, 4), "CMYK").save(buffer, format="JPEG", quality=92)
        return buffer.getvalue(), "jpg"
    elif mode in ("GIF", "GIF-PLAIN", "GIF-TRNS"):
        picture = palette_image(kind, width, height)
        options = {"interlace": mode != "GIF-PLAIN"}
        if mode == "GIF-TRNS":
            options["transparency"] = picture.getpixel((0, 0))
        picture.save(buffer, format="GIF", **options)
        return buffer.getvalue(), "gif"
    elif mode == "GIF-OFFSET":
        palette_image(kind, width // 2, height // 2).save(buffer, format="GIF")
        offset = (width - width // 2, height - height // 2)
        return relocated_gif(buffer.getvalue(), (width, height), offset), "gif"
    elif mode == "GIF-ANIMATED":
        frames = [palette_image(k, width, height) for k in (kind, "blocks")]
        frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100)
        return buffer.getvalue(), "gif"
    elif mode in ("WEBP-LOSSY", "WEBP-LOSSLESS"):
        Image.fromarray(pattern(kind, width, height), "RGB").save(
            buffer, format="WEBP", quality=90, lossless=mode == "WEBP-LOSSLESS"
        )
        return buffer.getvalue(), "webp"
    elif mode == "WEBP-ALPHA":
        Image.fromarray(pattern(kind, width, height, 4), "RGBA").save(buffer, format="WEBP", quality=90)
        return buffer.getvalue(), "webp"
    elif mode == "WEBP-ANIMATED":
        frames = [Image.fromarray(pattern(k, width, height), "RGB") for k in (kind, "noise")]
        frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=100, quality=90)
        return buffer.getvalue(), "webp"
    else:
        raise ValueError(mode)
    return buffer.getvalue(), "png"


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
    assert PIL.__version__ == "12.3.0", "capture requires Pillow 12.3.0"
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
    # Span costs 1022 to 1026 at one and two token rows, and the largest
    # square the test's pixel limit takes.
    for columns in range(1019, 1024):
        sizes.update({(42 * columns - 41, 42), (42 * columns, 42), (42 * columns + 1, 42)})
    for columns in range(509, 512):
        sizes.update({(42 * columns, 84), (42 * columns + 1, 84)})
    sizes.add((16384, 16384))
    while len(sizes) < 960:
        w = int(10 ** rng.uniform(0, 4.3))
        h = int(10 ** rng.uniform(0, 4.3))
        if max(w, h) / min(w, h) <= 30000:
            sizes.add((w, h))
    sweep = [[h, w] + planned(module, args, w, h) for (w, h) in sorted(sizes)]

    cases = []
    for case in CASES:
        name, kind, width, height = case[:4]
        mode = case[4] if len(case) > 4 else "RGB"
        data, extension = encoded(name, kind, width, height, mode)
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = module.load_image({"data": data}, args)
        # The reference's F32 values before its BF16 conversion, by the same
        # operations on the same decode.
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
        assert image.size == (width, height), (name, image.size)
        plan = planned(module, args, width, height)
        padded = ImageOps.pad(image, (plan[1], plan[0]), color=(127, 127, 127))
        x = torch.from_numpy(np.asarray(padded, dtype=np.float32)).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        rows = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p)
        assert torch.equal(rows.to(torch.bfloat16).reshape(patches.shape), patches), name
        bf16 = patches.contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
        entry = {"name": name, "pattern": kind, "width": width, "height": height}
        if extension is not None:
            entry["file"] = f"{NAME}-{name}.{extension}"
            (FIXTURES / entry["file"]).write_bytes(data)
            entry["decoded_rgb_sha256"] = sha256(np.asarray(image, dtype=np.uint8).tobytes())
        entry.update(
            {
                "plan": plan,
                "grid": [1, n_vit_h, n_vit_w],
                "tokens": n_llm_h * n_llm_w,
                "span_tokens": module.num_image_tokens(n_llm_h, n_llm_w),
                "rows_sha256": sha256(rows.contiguous().numpy().astype("<f4").tobytes()),
                "bf16_sha256": sha256(bf16),
            }
        )
        cases.append(entry)

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
