"""The image decode corpus: what a reference loader makes of encoded images.

Builds about a hundred small PNG, JPEG, GIF and WebP files that cover color
types, transparency, first-frame selection and every place Pillow 12.3 reads
an orientation (EXIF and XMP in PNG, JPEG and WebP), and records the RGB a
loader decodes from each. Two loaders are recorded for runtime-media's own
decoder test:

- `load_image`: transformers 5.16.1 `image_utils.load_image` on the file's
  data URL, which runs `ImageOps.exif_transpose` and then `convert("RGB")`.
- `open_convert`: `Image.open` and then `convert("RGB")`, which keeps the
  stored orientation.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m tools.refrun.image_decode

writes the files under crates/runtime-media/tests/fixtures/image-decode/
(refusing to replace a file with other bytes) and the golden
crates/runtime-media/tests/fixtures/image-decode.json. CPU only; no model
is loaded.

A model's capture calls `model_capture` where the model's image processor
is installed: it repeats the `load_image` column over the committed files,
checks that the processor makes the same pixel_values of `load_image`'s
image as of its RGB array (so the processor adds no decode step of its
own), and checks that the model's qualification probes need no
orientation, recording each probe's mode. runtime-host's
`image_decode::check_image_decode_v1` holds a model's served decode to it.

A result is the decoded size and the SHA-256 of its RGB bytes, or the
exception class a loader raised. A case whose served decode deliberately
differs from a loader carries `runtime`: "refused", or the size and digest
of the stored pixels with alpha discarded, which the producer computes from
the source pixels; `divergence` says why.
"""

import base64
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "crates/runtime-media/tests/fixtures"
DIRECTORY = FIXTURES / "image-decode"
GOLDEN = FIXTURES / "image-decode.json"
PILLOW = "12.3.0"
TRANSFORMERS = "5.16.1"
MIME = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
# Pillow's per-chunk limit on decompressed PNG text (PngImagePlugin.MAX_TEXT_CHUNK).
TEXT_CHUNK = 1 << 20


@dataclass
class Case:
    name: str
    extension: str
    data: bytes
    # The served decode where it deliberately differs from a loader:
    # {loader: "refused" | "stored"}, with the reason.
    runtime: dict = field(default_factory=dict)
    divergence: str = ""
    # Stored pixels with alpha discarded (uint8 [height, width, 3]) for a
    # "stored" runtime outcome; None takes `open_convert`'s decode.
    stored: object = None


def pattern(kind, width, height, channels=3):
    """uint8 [height, width, channels], from integer math only."""
    import numpy as np
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
        return ((x // np.uint64(3) * np.uint64(37) + y // np.uint64(2) * np.uint64(59)
                 + c * np.uint64(85)) % np.uint64(256)).astype(np.uint8)
    raise ValueError(kind)


# PNG, written here so that every byte (depth, filter, interlace, chunk
# order) is chosen by the case rather than by an encoder.

def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def packed(samples, depth):
    """One row of samples (ints below 2^depth) packed MSB first."""
    if depth == 8:
        return bytes(int(v) for v in samples)
    if depth == 16:
        return b"".join(struct.pack(">H", int(v)) for v in samples)
    out, bits, used = bytearray(), 0, 0
    for value in samples:
        bits = (bits << depth) | int(value)
        used += depth
        if used == 8:
            out.append(bits)
            bits, used = 0, 0
    if used:
        out.append(bits << (8 - used))
    return bytes(out)


ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def png(samples, depth, color, *, before=(), after=(), plte=None, trns=None, interlace=False, actl=None):
    """A PNG of `samples` (int [height, width, channels]); `before` and
    `after` are chunks placed before the first IDAT and before IEND."""
    height, width = samples.shape[:2]
    rows = []
    passes = ADAM7 if interlace else ((0, 0, 1, 1),)
    for sx, sy, dx, dy in passes:
        sub = samples[sy::dy, sx::dx]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue
        for row in sub:
            rows.append(b"\0" + packed(row.reshape(-1), depth))
    ihdr = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 1 if interlace else 0)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
    if actl is not None:
        out += actl
    if plte is not None:
        out += chunk(b"PLTE", bytes(plte))
    if trns is not None:
        out += chunk(b"tRNS", trns)
    out += b"".join(before)
    out += chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
    return out + b"".join(after) + chunk(b"IEND", b"")


def tiff(entries, little=True, declared=None, header=b""):
    """A TIFF stream whose IFD0 holds `entries` of (tag, type, count, value
    bytes); values over four bytes follow the IFD. `declared` overrides the
    entry count, so an IFD can claim more entries than it holds."""
    e = "<" if little else ">"
    out = (b"II*\0" if little else b"MM\0*") + struct.pack(e + "I", 8)
    out += struct.pack(e + "H", len(entries) if declared is None else declared)
    extra = b""
    offset = 8 + 2 + 12 * len(entries) + 4
    for tag, typ, count, value in entries:
        if len(value) > 4:
            out += struct.pack(e + "HHII", tag, typ, count, offset + len(extra))
            extra += value
        else:
            out += struct.pack(e + "HHI", tag, typ, count) + value.ljust(4, b"\0")
    return header + out + b"\0\0\0\0" + extra


def short(value, little=True):
    return struct.pack("<H" if little else ">H", value)


def exif(orientation, little=True):
    return tiff([(0x0112, 3, 1, short(orientation, little))], little)


def xmp(orientation, element=False):
    inner = (f"<tiff:Orientation>{orientation}</tiff:Orientation>" if element
             else f'<rdf:Description tiff:Orientation="{orientation}"/>')
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:tiff="http://ns.adobe.com/tiff/1.0/">' + inner + "</rdf:RDF></x:xmpmeta>").encode()


def text(keyword, value):
    return chunk(b"tEXt", keyword + b"\0" + value)


def ztxt(keyword, value, method=0, corrupt=False):
    data = zlib.compress(value, 9)
    if corrupt:
        data = data[:2] + bytes(b ^ 0x5A for b in data[2:])
    return chunk(b"zTXt", keyword + b"\0" + bytes([method]) + data)


def itxt(keyword, value, compressed=False, corrupt=False):
    data = zlib.compress(value, 9) if compressed else value
    if corrupt:
        data = data[:2] + bytes(b ^ 0x5A for b in data[2:])
    return chunk(b"iTXt", keyword + b"\0" + bytes([1 if compressed else 0, 0]) + b"\0\0" + data)


def raw_profile(data):
    """ImageMagick's hex text form of an EXIF block, as Pillow reads it
    (the three lines before the hex are skipped)."""
    hexed = data.hex()
    lines = [hexed[i:i + 72] for i in range(0, len(hexed), 72)]
    return ("\nexif\n" + f"{len(data):8d}\n" + "\n".join(lines) + "\n").encode()


def fctl(sequence, width, height):
    return chunk(b"fcTL", struct.pack(">IIIIIHHBB", sequence, width, height, 0, 0, 1, 10, 0, 0))


def apng(first, second, *, default_outside, between=(), before_iend=()):
    """A two-frame APNG whose frame 0 is `first`. With `default_outside`
    the IDAT image is not part of the animation, so the animation's frames
    are `second` twice; `between` goes after IDAT, before the next fcTL."""
    height, width = first.shape[:2]
    frames = 2
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    out += chunk(b"acTL", struct.pack(">II", frames, 0))
    sequence = 0
    if not default_outside:
        out += fctl(sequence, width, height)
        sequence += 1
    rows = lambda image: zlib.compress(b"".join(b"\0" + packed(r.reshape(-1), 8) for r in image), 9)
    out += chunk(b"IDAT", rows(first)) + b"".join(between)
    for _ in range(frames if default_outside else frames - 1):
        out += fctl(sequence, width, height)
        out += chunk(b"fdAT", struct.pack(">I", sequence + 1) + rows(second))
        sequence += 2
    return out + b"".join(before_iend) + chunk(b"IEND", b"")


# JPEG, encoded by Pillow, with metadata segments spliced in.

def segments(data):
    """(marker, start, end) of every marker segment, scan data included in
    the SOS segment's end."""
    out, pos = [], 2
    while pos < len(data):
        assert data[pos] == 0xFF, pos
        marker = data[pos + 1]
        if marker == 0xD9:
            out.append((marker, pos, pos + 2))
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        end = pos + 2 + length
        if marker == 0xDA:
            while not (data[end] == 0xFF and data[end + 1] not in (0, *range(0xD0, 0xD8))):
                end += 1
        out.append((marker, pos, end))
        pos = end
    return out


def app1(payload):
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def jpeg(image, *, mode="RGB", subsampling=0, progressive=False, start=(), before_sos=(), after_first_scan=()):
    from PIL import Image
    buffer = io.BytesIO()
    options = {"quality": 90, "progressive": progressive}
    if mode != "L" and mode != "CMYK":
        options["subsampling"] = subsampling
    Image.fromarray(image, mode).save(buffer, format="JPEG", **options)
    data = buffer.getvalue()
    marks = segments(data)
    first_sos = next(start_ for marker, start_, _ in marks if marker == 0xDA)
    first_scan_end = next(end for marker, _, end in marks if marker == 0xDA)
    insert = {2: b"".join(start), first_sos: b"".join(before_sos), first_scan_end: b"".join(after_first_scan)}
    out, last = b"", 0
    for at in sorted(insert):
        out += data[last:at] + insert[at]
        last = at
    return out + data[last:]


def exif_app1(entries=None, *, orientation=None, little=True, raw=None):
    if raw is not None:
        return app1(b"Exif\0\0" + raw)
    if entries is None:
        entries = [(0x0112, 3, 1, short(orientation, little))]
    return app1(b"Exif\0\0" + tiff(entries, little))


def xmp_app1(orientation, element=False):
    return app1(b"http://ns.adobe.com/xap/1.0/\0" + xmp(orientation, element))


# GIF, encoded by Pillow, with the screen, frame and palette rearranged.

def gif_blocks(data):
    """Offsets of the logical screen's color table end and the first image
    descriptor."""
    pos = 13 + (3 << ((data[10] & 7) + 1) if data[10] & 0x80 else 0)
    table_end = pos
    while data[pos] == 0x21:
        pos += 2
        while data[pos]:
            pos += data[pos] + 1
        pos += 1
    assert data[pos] == 0x2C
    return table_end, pos


def palette_gif(indices, palette, **options):
    from PIL import Image
    picture = Image.fromarray(indices, "P")
    picture.putpalette(palette)
    buffer = io.BytesIO()
    picture.save(buffer, format="GIF", optimize=False, **options)
    return buffer.getvalue()


def relocated_gif(data, screen, offset):
    data = bytearray(data)
    data[6:10] = struct.pack("<HH", *screen)
    _, descriptor = gif_blocks(data)
    data[descriptor + 1:descriptor + 5] = struct.pack("<HH", *offset)
    return bytes(data)


def local_palette_gif(data, replacement):
    """Moves the global color table into frame 0 as a local table and puts
    `replacement` (same size) in the global one."""
    data = bytearray(data)
    flags = data[10]
    table = data[13:13 + (3 << ((flags & 7) + 1))]
    assert len(replacement) == len(table)
    data[13:13 + len(table)] = replacement
    _, descriptor = gif_blocks(data)
    data[descriptor + 9] = (data[descriptor + 9] & 0x40) | 0x80 | (flags & 7)
    return bytes(data[:descriptor + 10] + table + data[descriptor + 10:])


def short_table_gif(data, entries):
    """Cuts the global color table to `entries` (a power of two)."""
    data = bytearray(data)
    size = 3 << ((data[10] & 7) + 1)
    bits = entries.bit_length() - 2
    data[10] = (data[10] & ~7) | bits
    return bytes(data[:13 + 3 * entries] + data[13 + size:])


def gif_xmp(data, orientation):
    """Adds an XMP application extension before frame 0."""
    payload = xmp(orientation)
    blocks = b"".join(bytes([len(payload[i:i + 255])]) + payload[i:i + 255] for i in range(0, len(payload), 255))
    extension = b"\x21\xff\x0bXMP DataXMP" + blocks + b"\0"
    _, descriptor = gif_blocks(data)
    return data[:descriptor] + extension + data[descriptor:]


# WebP, encoded by Pillow, with its container rebuilt where a case needs it.

def riff(chunks):
    body = b"WEBP" + b"".join(k + struct.pack("<I", len(d)) + d + (b"\0" if len(d) % 2 else b"") for k, d in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def webp_chunks(data):
    out, pos = [], 12
    while pos < len(data):
        kind, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        out.append((kind, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (size & 1)
    return out


def vp8x(flags, width, height):
    return bytes([flags, 0, 0, 0]) + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")


def still_webp(image, mode="RGB", **options):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image, mode).save(buffer, format="WEBP", **options)
    return buffer.getvalue()


def bitstream(data):
    """The VP8 or VP8L chunk of a simple still file, and its ALPH chunk."""
    chunks = dict(webp_chunks(data))
    for kind in (b"VP8L", b"VP8 "):
        if kind in chunks:
            return kind, chunks[kind], chunks.get(b"ALPH")
    raise ValueError("no bitstream")


def anmf(x, y, width, height, kind, data):
    head = b"".join(v.to_bytes(3, "little") for v in (x // 2, y // 2, width - 1, height - 1, 100)) + b"\0"
    return head + kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def cases():
    import numpy as np
    from PIL import Image
    out = []

    def add(name, extension, data, **kw):
        out.append(Case(name, extension, data, **kw))

    rgb = pattern("noise", 7, 5)
    rgba = pattern("noise", 7, 5, 4)
    gray = pattern("noise", 7, 5, 1)

    # PNG color types, depths and transparency.
    for depth in (1, 2, 4, 8):
        add(f"png-gray-{depth}bit", "png", png(gray >> (8 - depth), depth, 0))
    key = int(gray[0, 0, 0])
    add("png-gray-trns", "png", png(gray, 8, 0, trns=struct.pack(">H", key)))
    four = gray >> 4
    key4 = int(four[0, 0, 0])
    add("png-gray-4bit-trns-high-key", "png", png(four, 4, 0, trns=struct.pack(">H", 0x100 | key4)))
    add("png-rgb", "png", png(rgb, 8, 2))
    add("png-rgb-trns", "png", png(rgb, 8, 2, trns=struct.pack(">HHH", *(int(v) for v in rgb[0, 0]))))
    palette = pattern("blocks", 256, 1, 3).reshape(-1)
    for depth in (1, 2, 4, 8):
        indices = gray >> (8 - depth)
        add(f"png-palette-{depth}bit", "png", png(indices, depth, 3, plte=palette[:3 << depth]))
    add("png-palette-partial-trns", "png",
        png(gray >> 4, 4, 3, plte=palette[:48], trns=bytes(i * 37 % 256 for i in range(9))))
    add("png-gray-alpha", "png", png(pattern("noise", 7, 5, 2), 8, 4))
    add("png-rgba", "png", png(rgba, 8, 6))
    add("png-rgba-transparent-black", "png", png(pattern("zero", 7, 5, 4), 8, 6))
    add("png-adam7-rgba", "png", png(pattern("noise", 11, 9, 4), 8, 6, interlace=True))
    add("png-adam7-palette", "png", png(pattern("noise", 11, 9, 1) >> 6, 2, 3, plte=palette[:12], interlace=True))
    wide = pattern("noise", 7, 5).astype(np.uint16) * 257 + 3
    add("png-rgb-16bit", "png", png(wide, 16, 2), runtime={"load_image": "refused", "open_convert": "refused"},
        divergence="runtime-media refuses PNG with 16-bit samples")
    blocks = pattern("blocks", 7, 5)
    add("png-apng-default-in-animation", "png", apng(rgb, blocks, default_outside=False))
    add("png-apng-default-outside-animation", "png", apng(rgb, blocks, default_outside=True))

    # PNG orientation.
    for orientation in range(1, 9):
        add(f"png-exif{orientation}", "png", png(rgb, 8, 2, before=[chunk(b"eXIf", exif(orientation))]))
    add("png-exif6-after-idat", "png", png(rgb, 8, 2, after=[chunk(b"eXIf", exif(6))]))
    add("png-exif6-before-exif3-after-idat", "png",
        png(rgb, 8, 2, before=[chunk(b"eXIf", exif(6))], after=[chunk(b"eXIf", exif(3))]))
    add("png-exif6-long", "png", png(rgb, 8, 2, before=[chunk(b"eXIf", tiff([(0x0112, 4, 1, struct.pack("<I", 6))]))]))
    add("png-exif6-big-endian", "png", png(rgb, 8, 2, before=[chunk(b"eXIf", exif(6, little=False))]))
    add("png-text-exif6", "png", png(rgb, 8, 2, before=[text(b"exif", exif(6))]))
    add("png-text-raw-profile6", "png", png(rgb, 8, 2, before=[text(b"Raw profile type exif", raw_profile(exif(6)))]))
    add("png-ztxt-raw-profile6", "png", png(rgb, 8, 2, after=[ztxt(b"Raw profile type exif", raw_profile(exif(6)))]))
    add("png-raw-profile-invalid-hex", "png",
        png(rgb, 8, 2, before=[text(b"Raw profile type exif", b"\nexif\n      8\nzz\n")]),
        runtime={"load_image": "stored"}, stored=rgb,
        divergence="Pillow raises on invalid raw-profile hex; runtime-media keeps the stored orientation")
    add("png-itxt-xmp6", "png", png(rgb, 8, 2, before=[itxt(b"XML:com.adobe.xmp", xmp(6))]))
    add("png-itxt-xmp6-compressed-element", "png",
        png(rgb, 8, 2, after=[itxt(b"XML:com.adobe.xmp", xmp(6, element=True), compressed=True)]))
    add("png-text-xmp6", "png", png(rgb, 8, 2, before=[text(b"XML:com.adobe.xmp", xmp(6))]))
    add("png-ztxt-xmp6-then-xmp3", "png",
        png(rgb, 8, 2, before=[ztxt(b"XML:com.adobe.xmp", xmp(6))], after=[ztxt(b"XML:com.adobe.xmp", xmp(3))]))
    add("png-ztxt-xmp6-then-corrupt", "png",
        png(rgb, 8, 2, before=[ztxt(b"XML:com.adobe.xmp", xmp(6)), ztxt(b"XML:com.adobe.xmp", xmp(3), corrupt=True)]))
    add("png-itxt-xmp6-then-corrupt", "png",
        png(rgb, 8, 2, before=[itxt(b"XML:com.adobe.xmp", xmp(6)),
                               itxt(b"XML:com.adobe.xmp", xmp(3), compressed=True, corrupt=True)]))
    truncated_xmp = zlib.compress(xmp(6), 0)[:-8]
    add("png-ztxt-xmp6-truncated", "png",
        png(rgb, 8, 2, before=[chunk(b"zTXt", b"XML:com.adobe.xmp\0\0" + truncated_xmp)]))
    add("png-itxt-xmp6-then-corrupt-ztxt", "png",
        png(rgb, 8, 2, before=[itxt(b"XML:com.adobe.xmp", xmp(6)), ztxt(b"XML:com.adobe.xmp", xmp(3), corrupt=True)]))
    add("png-exif-bad-header", "png", png(rgb, 8, 2, before=[chunk(b"eXIf", b"XX*\0\x08\0\0\0")]),
        runtime={"load_image": "stored"}, stored=rgb,
        divergence="Pillow raises on a malformed TIFF header; runtime-media keeps the stored orientation")
    add("png-ztxt-exif6-text", "png", png(rgb, 8, 2, before=[ztxt(b"exif", exif(6))]),
        runtime={"load_image": "stored"}, stored=rgb,
        divergence="Pillow raises on an exif value stored as text; runtime-media keeps the stored orientation")
    add("png-itxt-exif6-text", "png", png(rgb, 8, 2, before=[itxt(b"exif", exif(6))]),
        runtime={"load_image": "stored"}, stored=rgb,
        divergence="Pillow raises on an exif value stored as text; runtime-media keeps the stored orientation")
    add("png-exif1-xmp6", "png",
        png(rgb, 8, 2, before=[chunk(b"eXIf", exif(1)), itxt(b"XML:com.adobe.xmp", xmp(6))]))
    add("png-rgba-exif6", "png", png(rgba, 8, 6, before=[chunk(b"eXIf", exif(6))]))
    for outside in (False, True):
        form = "outside" if outside else "in"
        add(f"png-apng-default-{form}-exif6-before-fctl", "png",
            apng(rgb, blocks, default_outside=outside, between=[chunk(b"eXIf", exif(6))]))
        add(f"png-apng-default-{form}-exif6-before-iend", "png",
            apng(rgb, blocks, default_outside=outside, before_iend=[chunk(b"eXIf", exif(6))]))
    one_frame = chunk(b"acTL", struct.pack(">II", 1, 0)) + fctl(0, 7, 5)
    add("png-actl-one-frame-exif6-before-iend", "png", png(rgb, 8, 2, actl=one_frame, after=[chunk(b"eXIf", exif(6))]))
    padded = lambda orientation: xmp(orientation) + b" " * (TEXT_CHUNK - len(xmp(orientation)))
    add("png-ztxt-xmp-over-1mib", "png", png(rgb, 8, 2, before=[ztxt(b"XML:com.adobe.xmp", padded(6) + b" ")]),
        runtime={"open_convert": "stored"}, stored=rgb,
        divergence="Pillow refuses compressed text over 1 MiB; runtime-media reads no metadata for the stored orientation")
    add("png-ztxt-xmp-over-budget", "png", png(rgb, 8, 2, before=[ztxt(b"XML:com.adobe.xmp", padded(6))] * 5),
        runtime={"load_image": "refused"},
        divergence="runtime-media refuses orientation metadata that inflates past 4 MiB in total")
    add("png-ztxt-comment-over-1mib", "png", png(rgb, 8, 2, before=[ztxt(b"Comment", b" " * (TEXT_CHUNK + 1))]),
        runtime={"load_image": "stored", "open_convert": "stored"}, stored=rgb,
        divergence="Pillow refuses compressed text over 1 MiB; runtime-media reads no other text")
    add("png-ztxt-method1-xmp6", "png", png(rgb, 8, 2, before=[ztxt(b"XML:com.adobe.xmp", xmp(6), method=1)]),
        runtime={"load_image": "stored", "open_convert": "stored"}, stored=rgb,
        divergence="Pillow refuses a zTXt chunk with an unknown compression method; runtime-media ignores it")

    # JPEG.
    photo = pattern("blocks", 13, 9)
    add("jpeg-444", "jpg", jpeg(photo))
    add("jpeg-420", "jpg", jpeg(photo, subsampling=2))
    add("jpeg-gray", "jpg", jpeg(pattern("blocks", 13, 9, 1)[:, :, 0], mode="L"))
    add("jpeg-cmyk", "jpg", jpeg(pattern("blocks", 13, 9, 4), mode="CMYK"))
    add("jpeg-progressive", "jpg", jpeg(photo, progressive=True, subsampling=2))
    for orientation in range(1, 9):
        add(f"jpeg-exif{orientation}", "jpg", jpeg(photo, start=[exif_app1(orientation=orientation)]))
    add("jpeg-exif6-big-endian", "jpg", jpeg(photo, start=[exif_app1(orientation=6, little=False)]))
    for name, typ, count, value in [
        ("long", 4, 1, struct.pack("<I", 6)), ("long8", 16, 1, struct.pack("<Q", 6)),
        ("rational", 5, 1, struct.pack("<II", 12, 2)), ("float", 11, 1, struct.pack("<f", 6.0)),
        ("byte", 1, 1, b"\x06"), ("unregistered-type", 17, 1, struct.pack("<I", 6)),
        ("short-count2", 3, 2, struct.pack("<HH", 3, 6)),
    ]:
        add(f"jpeg-exif6-{name}", "jpg", jpeg(photo, start=[exif_app1([(0x0112, typ, count, value)])]))
    add("jpeg-exif0", "jpg", jpeg(photo, start=[exif_app1(orientation=0)]))
    add("jpeg-exif9", "jpg", jpeg(photo, start=[exif_app1(orientation=9)]))
    add("jpeg-exif-duplicate-6-3", "jpg",
        jpeg(photo, start=[exif_app1([(0x0112, 3, 1, short(6)), (0x0112, 3, 1, short(3))])]))
    truncated = tiff([(0x0112, 3, 1, short(6))], declared=5)[:-4]
    add("jpeg-exif6-truncated-ifd", "jpg", jpeg(photo, start=[exif_app1(raw=truncated)]))
    make = [(0x010F, 2, 4, b"Cam\0")]
    add("jpeg-app1-without-then-exif6", "jpg", jpeg(photo, start=[exif_app1(make), exif_app1(orientation=6)]))
    add("jpeg-app1-exif1-then-exif6", "jpg", jpeg(photo, start=[exif_app1(orientation=1), exif_app1(orientation=6)]))
    add("jpeg-xmp6-attribute", "jpg", jpeg(photo, start=[xmp_app1(6)]))
    add("jpeg-xmp6-element", "jpg", jpeg(photo, start=[xmp_app1(6, element=True)]))
    add("jpeg-exif1-xmp6", "jpg", jpeg(photo, start=[exif_app1(orientation=1), xmp_app1(6)]))
    add("jpeg-exif-ascii-xmp6", "jpg", jpeg(photo, start=[exif_app1([(0x0112, 2, 2, b"6\0")]), xmp_app1(6)]))
    add("jpeg-exif-bad-header", "jpg", jpeg(photo, start=[exif_app1(raw=b"XX*\0\x08\0\0\0")]))
    from PIL import Image as _Image
    buffer = io.BytesIO()
    _Image.fromarray(photo).save(buffer, format="JPEG", quality=90, subsampling=0, dpi=(72, 72))
    dpi = buffer.getvalue()
    add("jpeg-dpi-exif-bad-header", "jpg", dpi[:2] + exif_app1(raw=b"XX*\0\x08\0\0\0") + dpi[2:],
        runtime={"load_image": "stored"}, stored=None,
        divergence="Pillow raises on a malformed TIFF header in a JPEG with a JFIF density; "
                   "runtime-media keeps the stored orientation")
    add("jpeg-exif6-before-sos", "jpg", jpeg(photo, before_sos=[exif_app1(orientation=6)]))
    add("jpeg-progressive-exif6-after-first-scan", "jpg",
        jpeg(photo, progressive=True, after_first_scan=[exif_app1(orientation=6)]))

    # GIF.
    indices = pattern("noise", 9, 7, 1)[:, :, 0] >> 5
    colors = pattern("noise", 8, 1, 3).reshape(-1).tolist()
    add("gif-plain", "gif", palette_gif(indices, colors, interlace=False))
    add("gif-interlaced", "gif", palette_gif(indices, colors, interlace=True))
    add("gif-transparency", "gif", palette_gif(indices, colors, interlace=False, transparency=int(indices[0, 0])))
    small = palette_gif(indices[:4, :5], colors, interlace=False)
    add("gif-offset-frame", "gif", relocated_gif(small, (9, 7), (4, 3)))
    add("gif-frame-over-screen", "gif", relocated_gif(palette_gif(indices, colors, interlace=False), (6, 4), (2, 1)))
    second = pattern("blocks", 9, 7, 1)[:, :, 0] >> 5
    picture = Image.fromarray(indices, "P")
    picture.putpalette(colors)
    following = Image.fromarray(second, "P")
    following.putpalette(colors)
    buffer = io.BytesIO()
    picture.save(buffer, format="GIF", save_all=True, append_images=[following], duration=100, optimize=False)
    add("gif-animated", "gif", buffer.getvalue())
    plain = palette_gif(indices, colors, interlace=False)
    table = plain[13:13 + (3 << ((plain[10] & 7) + 1))]
    add("gif-local-palette", "gif", local_palette_gif(plain, bytes(255 - b for b in table)))
    gray_palette = [v for i in range(8) for v in (i, i, i)]
    add("gif-gray-identity-past-end", "gif", short_table_gif(palette_gif(indices, gray_palette, interlace=False), 4))
    add("gif-short-palette-past-end", "gif", short_table_gif(plain, 4))
    add("gif-xmp6", "gif", gif_xmp(plain, 6))

    # WebP.
    add("webp-lossy", "webp", still_webp(pattern("blocks", 23, 17), quality=90))
    add("webp-lossless", "webp", still_webp(rgb, lossless=True))
    add("webp-lossy-alpha", "webp", still_webp(pattern("blocks", 23, 17, 4), "RGBA", quality=90))
    add("webp-lossless-alpha", "webp", still_webp(rgba, "RGBA", lossless=True))
    frames = [Image.fromarray(rgb), Image.fromarray(blocks)]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=100, lossless=True)
    add("webp-animated", "webp", buffer.getvalue())
    kind, data, _ = bitstream(still_webp(pattern("noise", 4, 3), lossless=True))
    for flags, name in ((0x12, "alpha"), (0x02, "opaque")):
        add(f"webp-animated-offset-{name}", "webp", riff([
            (b"VP8X", vp8x(flags, 9, 7)), (b"ANIM", b"\0" * 6), (b"ANMF", anmf(4, 2, 4, 3, kind, data))]))
    kind, data, _ = bitstream(still_webp(rgb, lossless=True))
    for orientation in range(1, 9):
        add(f"webp-exif{orientation}", "webp",
            riff([(b"VP8X", vp8x(0x08, 7, 5)), (kind, data), (b"EXIF", exif(orientation))]))
    add("webp-exif6-with-header", "webp",
        riff([(b"VP8X", vp8x(0x08, 7, 5)), (kind, data), (b"EXIF", b"Exif\0\0" + exif(6))]))
    add("webp-exif6-unflagged", "webp", riff([(b"VP8X", vp8x(0, 7, 5)), (kind, data), (b"EXIF", exif(6))]))
    add("webp-simple-trailing-exif6", "webp", riff([(kind, data), (b"EXIF", exif(6))]))
    add("webp-exif6-then-exif3", "webp",
        riff([(b"VP8X", vp8x(0x08, 7, 5)), (kind, data), (b"EXIF", exif(6)), (b"EXIF", exif(3))]))
    add("webp-xmp6", "webp", riff([(b"VP8X", vp8x(0x04, 7, 5)), (kind, data), (b"XMP ", xmp(6))]))
    add("webp-xmp6-unflagged", "webp", riff([(b"VP8X", vp8x(0, 7, 5)), (kind, data), (b"XMP ", xmp(6))]))
    add("webp-exif1-xmp6", "webp",
        riff([(b"VP8X", vp8x(0x0C, 7, 5)), (kind, data), (b"EXIF", exif(1)), (b"XMP ", xmp(6))]))
    add("webp-exif-bad-header", "webp",
        riff([(b"VP8X", vp8x(0x08, 7, 5)), (kind, data), (b"EXIF", b"XX*\0\x08\0\0\0")]),
        runtime={"load_image": "stored"}, stored=rgb,
        divergence="Pillow raises on a malformed TIFF header; runtime-media keeps the stored orientation")
    buffer = io.BytesIO()
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=100, lossless=True,
                   exif=exif(6))
    add("webp-animated-exif6", "webp", buffer.getvalue())
    alpha_kind, alpha_data, _ = bitstream(still_webp(rgba, "RGBA", lossless=True))
    add("webp-alpha-exif6", "webp",
        riff([(b"VP8X", vp8x(0x18, 7, 5)), (alpha_kind, alpha_data), (b"EXIF", exif(6))]))
    names = [case.name for case in out]
    assert len(names) == len(set(names)), "duplicate case name"
    return out


def write_files(cases_, directory=DIRECTORY):
    """Writes each case's file, keeping one with the same bytes and refusing
    one with other bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    for case in cases_:
        path = directory / f"{case.name}.{case.extension}"
        if path.exists():
            if path.read_bytes() != case.data:
                raise SystemExit(f"{path.name} exists with other bytes; the corpus is not reproducible")
        else:
            path.write_bytes(case.data)


def data_url(case):
    return f"data:{MIME[case.extension]};base64," + base64.b64encode(case.data).decode()


def rgb_result(image):
    """The size and RGB digest of a decoded PIL image or uint8 [h, w, 3]."""
    import numpy as np
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"decoded image has shape {array.shape}")
    return {"width": int(array.shape[1]), "height": int(array.shape[0]),
            "rgb_sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def capture(load, cases_):
    """{name: result} of `load(case)` (a PIL image) over `cases_`; a raised
    exception is recorded by class."""
    results = {}
    for case in cases_:
        try:
            image = load(case)
            if image.mode != "RGB":
                raise ValueError(f"{case.name}: loader returned mode {image.mode}")
            results[case.name] = rgb_result(image)
        except Exception as error:  # noqa: BLE001 - any refusal is the result
            results[case.name] = {"refused": type(error).__name__}
    return results


def load_image(case):
    from transformers.image_utils import load_image as loader
    return loader(data_url(case))


def open_convert(case):
    from PIL import Image
    with Image.open(io.BytesIO(case.data)) as opened:
        return opened.convert("RGB")


def check_versions():
    import PIL
    import transformers
    if PIL.__version__ != PILLOW or transformers.__version__ != TRANSFORMERS:
        raise SystemExit(f"capture requires Pillow {PILLOW} and transformers {TRANSFORMERS}, "
                         f"found {PIL.__version__} and {transformers.__version__}")


def load_image_source_sha256():
    import inspect
    from transformers import image_utils
    return hashlib.sha256(inspect.getsource(image_utils.load_image).encode()).hexdigest()


def runtime_outcome(case, loader, results):
    """The served outcome a divergence case declares for `loader`: a
    refusal, or the stored pixels (from the source array, or as
    `open_convert` decoded them when the case has none)."""
    outcome = case.runtime.get(loader)
    if outcome is None:
        return None
    if outcome == "refused":
        return "refused"
    if case.stored is None:
        stored = results["open_convert"][case.name]
        if "refused" in stored:
            raise SystemExit(f"{case.name}: no stored pixels for its served outcome")
        return stored
    return rgb_result(case.stored)


# The cases a model capture also feeds to its processor: one per format,
# each oriented or transparent as far as the format allows.
PROCESSOR_CASES = ("png-rgba-exif6", "jpeg-exif6", "gif-transparency", "webp-alpha-exif6")
FORMATS = {"PNG": "png", "JPEG": "jpg", "GIF": "gif", "WEBP": "webp"}


def committed_cases():
    """The producer's cases, each checked against its committed file."""
    corpus = cases()
    for case in corpus:
        path = DIRECTORY / f"{case.name}.{case.extension}"
        if not path.is_file() or path.read_bytes() != case.data:
            raise SystemExit(f"{path.relative_to(ROOT)} is not the producer's file; run tools.refrun.image_decode")
    return corpus


def check_processor_input(processor, case):
    """Refuses a processor whose pixel_values of `load_image`'s image differ
    from those of the same RGB as a plain array image."""
    import numpy as np
    from PIL import Image
    image = load_image(case)
    rows = [np.asarray(processor(images=[source], return_tensors="np")["pixel_values"])
            for source in (image, Image.fromarray(np.asarray(image), "RGB"))]
    if rows[0].dtype != rows[1].dtype or rows[0].tobytes() != rows[1].tobytes():
        raise SystemExit(f"{case.name}: the processor changes load_image's image")


def probe(name, data):
    """A qualification probe's format and mode, refused unless `load_image`
    decodes it as stored, so no orientation applies."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as opened:
        extension, mode = FORMATS[opened.format], opened.mode
        alpha = "A" in mode or "transparency" in opened.info
        stored = opened.convert("RGB")
    if rgb_result(load_image(Case(name, extension, data))) != rgb_result(stored):
        raise SystemExit(f"probe {name}: load_image orients it")
    return {"file_sha256": hashlib.sha256(data).hexdigest(), "format": extension, "mode": mode, "alpha": alpha}


def model_capture(script, processor, probes, output):
    """Writes a model's decode capture to `output`: `load_image` over the
    committed corpus, the processor check over `PROCESSOR_CASES`, and
    `probes` ({name: encoded bytes}) as `probe` records them. `script` names
    the capture in the output."""
    import warnings
    warnings.simplefilter("ignore")
    check_versions()
    import PIL
    import numpy
    import torch
    import transformers
    corpus = committed_cases()
    named = {case.name: case for case in corpus}
    for name in PROCESSOR_CASES:
        check_processor_input(processor, named[name])
    results = capture(load_image, corpus)
    golden = {
        "source": (f"{script}: transformers {transformers.__version__} load_image under Pillow {PIL.__version__}, "
                   f"numpy {numpy.__version__}, torch {torch.__version__}, on CPU; size and SHA-256 of the "
                   "decoded RGB, or the exception class"),
        "processor": type(processor).__qualname__,
        "processor_cases": {"cases": list(PROCESSOR_CASES),
                            "check": "pixel_values of load_image's image equal those of its RGB array"},
        "load_image_source_sha256": load_image_source_sha256(),
        "probes": {name: probe(name, data) for name, data in probes.items()},
        "cases": [{"name": case.name, "file_sha256": hashlib.sha256(case.data).hexdigest(),
                   "load_image": results[case.name]} for case in corpus],
    }
    output.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {output.relative_to(ROOT)} with {len(corpus)} cases and {len(probes)} probes")


def main():
    import warnings
    warnings.simplefilter("ignore")
    check_versions()
    import PIL
    import numpy
    import transformers
    corpus = cases()
    write_files(corpus)
    loaders = {"load_image": load_image, "open_convert": open_convert}
    results = {name: capture(loader, corpus) for name, loader in loaders.items()}
    rows = []
    for case in corpus:
        row = {"name": case.name, "file": f"image-decode/{case.name}.{case.extension}",
               "file_sha256": hashlib.sha256(case.data).hexdigest()}
        for loader in loaders:
            row[loader] = results[loader][case.name]
            outcome = runtime_outcome(case, loader, results)
            if outcome is not None:
                row[loader] = {**row[loader], "runtime": outcome}
        if case.divergence:
            row["divergence"] = case.divergence
        rows.append(row)
    golden = {
        "source": (f"tools/refrun/image_decode.py: Pillow {PIL.__version__}, transformers {transformers.__version__}, "
                   f"numpy {numpy.__version__}, on CPU; size and SHA-256 of the decoded RGB, or the exception class"),
        "loaders": {
            "load_image": "transformers image_utils.load_image on the data URL: ImageOps.exif_transpose, then convert(\"RGB\")",
            "open_convert": "Image.open, then convert(\"RGB\")",
        },
        "load_image_source_sha256": load_image_source_sha256(),
        "cases": rows,
    }
    text = json.dumps(golden, indent=1)
    lines = text.split("\n")
    GOLDEN.write_text(text + "\n")
    print(f"wrote {GOLDEN.relative_to(ROOT)} with {len(rows)} cases ({len(lines)} lines)")


if __name__ == "__main__":
    main()
