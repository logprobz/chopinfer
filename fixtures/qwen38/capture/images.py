"""Deterministic PNG probes for the Qwen3.8 image preprocessing captures.

Sizes are chosen against smart_resize (factor 32, 65,536 to 16,777,216
pixels): odd sizes force resampling, one image is below the minimum and one
above the maximum. The pattern is smooth plus a hard edge, so bilinear,
bicubic and antialiasing give different pixels.

`functional()` rebuilds the two MiMo functional images the probes also use,
blue-square.png and frames/00.png, byte for byte: 256x256 RGB, each row
filtered as libpng's default heuristic chooses (the least sum of absolute
filtered values, signed), zlib level 6.
"""

import struct
import zlib

SIZES = {"gradient-333x517": (333, 517), "small-100x60": (100, 60), "wide-1000x40": (1000, 40),
         "large-4500x4000": (4500, 4000)}


def png(width, height):
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            edge = 255 if (x * 7 + y * 3) % 97 < 48 else 0
            row += bytes(((x * 255) // max(1, width - 1), (y * 255) // max(1, height - 1), edge))
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
            + chunk(b"IEND", b""))


FUNCTIONAL = {"blue-square": "blue-square.png", "frame-00": "frames/00.png"}
WHITE, BLUE, RED = b"\xff\xff\xff", b"\x00\x00\xff", b"\xff\x00\x00"


def filtered(line, previous, kind):
    """One PNG filter (0 none, 1 sub, 2 up, 3 average, 4 Paeth) over a row of RGB bytes."""
    out = bytearray(len(line))
    for x, value in enumerate(line):
        left = line[x - 3] if x >= 3 else 0
        up, corner = previous[x], previous[x - 3] if x >= 3 else 0
        if kind == 0:
            predicted = 0
        elif kind == 1:
            predicted = left
        elif kind == 2:
            predicted = up
        elif kind == 3:
            predicted = (left + up) >> 1
        else:
            pa, pb, pc = abs(up - corner), abs(left - corner), abs(left + up - 2 * corner)
            predicted = left if pa <= pb and pa <= pc else up if pb <= pc else corner
        out[x] = (value - predicted) & 255
    return out


def adaptive_png(rows, width, height):
    """An RGB PNG of `rows` (bytes per row) with libpng's default row filters."""
    raw, previous = bytearray(), bytes(3 * width)
    for line in rows:
        options = [filtered(line, previous, kind) for kind in range(5)]
        best = min(range(5), key=lambda kind: sum(v if v < 128 else 256 - v for v in options[kind]))
        raw += bytes([best]) + options[best]
        previous = line

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def functional():
    """{name: PNG bytes} of blue-square (a blue square from 48 to 208 on white)
    and frame-00 (solid red)."""
    square = [b"".join(BLUE if 48 <= x <= 208 and 48 <= y <= 208 else WHITE for x in range(256))
              for y in range(256)]
    return {"blue-square": adaptive_png(square, 256, 256),
            "frame-00": adaptive_png([RED * 256] * 256, 256, 256)}


def write_all(directory):
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (width, height) in SIZES.items():
        path = directory / f"{name}.png"
        if not path.exists():
            path.write_bytes(png(width, height))
        paths[name] = path
    return paths
