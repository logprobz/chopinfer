"""PNG decoding and the image processor's pixel rows for images that need no
resize, as the Qwen3.8 reference workloads build them
(models/qwen38/reference_workloads.py). `pixel_rows`' source is part of the
probe references' processor identity: keep it byte for byte.
"""

from pathlib import Path
import struct
import zlib

import numpy as np


def read_png(path):
    """An 8-bit RGB, non-interlaced PNG as an (H, W, 3) uint8 array."""
    data = Path(path).read_bytes()
    width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", data[16:29])
    if (depth, color, interlace) != (8, 2, 0):
        raise ValueError(f"{path}: only 8-bit RGB non-interlaced PNGs")
    position, idat = 8, b""
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        if kind == b"IDAT":
            idat += data[position + 8:position + 8 + length]
        position += 12 + length
    raw = zlib.decompress(idat)
    stride, out, previous = width * 3, [], bytearray(width * 3)
    for y in range(height):
        kind, row = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for x in range(stride):
            left = row[x - 3] if x >= 3 else 0
            up, corner = previous[x], previous[x - 3] if x >= 3 else 0
            if kind == 1:
                row[x] = (row[x] + left) & 255
            elif kind == 2:
                row[x] = (row[x] + up) & 255
            elif kind == 3:
                row[x] = (row[x] + (left + up) // 2) & 255
            elif kind == 4:
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                row[x] = (row[x] + (left if pa <= pb and pa <= pc else up if pb <= pc else corner)) & 255
        out.append(bytes(row))
        previous = row
    return np.frombuffer(b"".join(out), dtype=np.uint8).reshape(height, width, 3)


def pixel_rows(image, temporal=2, patch=16, merge=2):
    """The processor's F32 normalization, (x - 0.5 * 255) / (0.5 * 255), and its
    patch order, for an image that needs no resize."""
    height, width, _ = image.shape
    values = (image.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    planes = np.repeat(values.transpose(2, 0, 1)[None], temporal, axis=0)
    gh, gw = height // patch, width // patch
    patches = planes.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(gh * gw, 3 * temporal * patch * patch), [1, gh, gw]
