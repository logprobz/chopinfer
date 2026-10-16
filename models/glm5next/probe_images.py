"""Deterministic PNG probes for GLM 5.3 Flash's image references and media gate.

Sizes are chosen against the HF processor's 28-pixel grid and its 16 to
8,000 image-token budget: an image below the minimum (upscaled), odd and
wide sizes (resampled, zero-padded), an exact multiple, a translucent RGBA
image whose alpha the processor's RGB conversion drops (as recipe
image.zero-padded-token-budget.v1 does) and a large one. Every probe stays
at or below 2,048 image tokens so its reference fits the f64 run. The
pattern is smooth gradients plus a hard edge, so resampling kernels differ.

The PNGs are written byte for byte the same everywhere (filter 0 rows, zlib
level 9): reference_workloads.py renders the references from them, and the
media gate (tools/qual/multimodal.py, input `media_images`) regenerates them
with `write_all` and checks them against the capture's identity.json.
"""

from pathlib import Path
import struct
import zlib

# name: (width, height, channels); RGB unless 4 (RGBA with translucent alpha).
PROBES = {"small-100x60": (100, 60, 3), "odd-333x517": (333, 517, 3), "wide-1000x40": (1000, 40, 3),
          "exact-448x448": (448, 448, 3), "translucent-300x200": (300, 200, 4),
          "large-1300x1000": (1300, 1000, 3)}


def pixel(x, y, width, height, channels):
    values = ((x * 255) // max(1, width - 1), (y * 255) // max(1, height - 1),
              255 if (x * 7 + y * 3) % 97 < 48 else 0)
    if channels == 4:
        values += (64 + (x + y) * 128 // max(1, width + height - 2),)
    return values


def png(width, height, channels):
    """The probe image as PNG bytes (color type 2 RGB or 6 RGBA, 8 bits)."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row += bytes(pixel(x, y, width, height, channels))
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
            + chunk(b"IEND", b""))


def pngs():
    """{name: PNG bytes} of every probe."""
    return {name: png(*size) for name, size in PROBES.items()}


def write_all(directory):
    """Writes every probe's PNG into `directory` (keeping existing files);
    returns {name: path}."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, size in PROBES.items():
        path = directory / f"{name}.png"
        if not path.exists():
            path.write_bytes(png(*size))
        paths[name] = path
    return paths
