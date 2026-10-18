"""Ask a server what color solid, split and square images are.

The same run pointed at Chopinfer and at vLLM separates model behavior from
engine defects. Each probe (solid images in six colors and several sizes,
PNG and JPEG, a small square on white, and two-color splits) is asked the
same question with the image before and after the text, at temperature 0
with the top 20 logprobs of every answer token. Answers are recorded, not
gated: MiMo maps any uniform image to nearly the same encoder row, so a
wrong color for a solid image is only a defect if the reference engine
answers differently.

  color_probe.py --url URL               run the probes, write a receipt
  color_probe.py --compare A.json B.json per-probe answers of two receipts
  color_probe.py --write DIR             write the probe images to DIR

Preprocessing qualification is provided by the declared media_preprocessing check.
"""

import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

from common import (answer_budget, answer_control, frontend, thinking_can_be_disabled, Client, arguments,
                    receipt)


QUESTION = "What colour is this image? Answer with one word."
PROMPTS = ("image-first", "text-first")
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
          "white": (255, 255, 255), "black": (0, 0, 0), "yellow": (255, 255, 0)}
COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "black",
               "white", "gray", "grey", "brown")
WHITE = COLORS["white"]
# (width, height) of the solid PNG probes: identity, a small image,
# a 4:3 resize, one above the maximum budget, and a thin strip
# with a narrow aspect ratio.
SOLID_SIZES = ((256, 256), (64, 64), (1000, 750), (2048, 1536), (40, 600))


def png(width, height, pixel):
    """An 8-bit RGB PNG whose pixel (x, y) is `pixel(x, y)`."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    rows = b"".join(b"\0" + b"".join(bytes(pixel(x, y)) for x in range(width))
                    for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


ZIGZAG = (0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
          41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22,
          15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55,
          62, 63)
LUMA_QUANT = (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16,
              24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109,
              103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
              72, 92, 95, 98, 112, 100, 103, 99)
CHROMA_QUANT = (17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56,
                99, 99, 99, 99, 99, 47, 66) + (99,) * 38
# Flat Huffman codes: every DC category gets a 4-bit code and every AC
# (run, size) symbol, EOB and ZRL an 8-bit one. Valid, if not compact.
DC_SYMBOLS = tuple(range(12))
AC_SYMBOLS = (0x00, 0xF0) + tuple(run << 4 | size for run in range(16) for size in range(1, 11))
DCT = tuple(tuple((math.sqrt(0.5) if u == 0 else 1.0) * 0.5
                  * math.cos((2 * x + 1) * u * math.pi / 16) for x in range(8))
            for u in range(8))


def jpeg(width, height, pixel, quality=90):
    """A baseline 4:2:0 JPEG of `pixel(x, y)`, as the libjpeg encoder builds
    one: JFIF YCbCr, 2x2 averaged chroma and the Annex K tables at `quality`."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    tables = [[min(255, max(1, (value * scale + 50) // 100)) for value in table]
              for table in (LUMA_QUANT, CHROMA_QUANT)]
    padded_w, padded_h = -(-width // 16) * 16, -(-height // 16) * 16
    planes = ([], [], [])
    for y in range(padded_h):
        rows = tuple([] for _ in planes)
        for x in range(padded_w):
            r, g, b = pixel(min(x, width - 1), min(y, height - 1))
            rows[0].append(0.299 * r + 0.587 * g + 0.114 * b)
            rows[1].append(-0.168736 * r - 0.331264 * g + 0.5 * b + 128)
            rows[2].append(0.5 * r - 0.418688 * g - 0.081312 * b + 128)
        for plane, row in zip(planes, rows):
            plane.append(row)
    chroma = [[[sum(plane[2 * y + dy][2 * x + dx] for dy in (0, 1) for dx in (0, 1)) / 4
                for x in range(padded_w // 2)] for y in range(padded_h // 2)]
              for plane in planes[1:]]
    luma = planes[0]

    def block(plane, bx, by, table):
        samples = [[min(255, max(0, round(plane[by * 8 + y][bx * 8 + x]))) - 128
                    for x in range(8)] for y in range(8)]
        if all(value == samples[0][0] for row in samples for value in row):
            coefficients = [8 * samples[0][0]] + [0] * 63
        else:
            rows = [[sum(DCT[u][x] * row[x] for x in range(8)) for u in range(8)]
                    for row in samples]
            coefficients = [sum(DCT[v][y] * rows[y][u] for y in range(8))
                            for v in range(8) for u in range(8)]
        return [round(coefficients[index] / table[index]) for index in ZIGZAG]

    bits = []

    def put(value, length):
        bits.extend((value >> shift) & 1 for shift in range(length - 1, -1, -1))

    def magnitude(value):
        size = abs(value).bit_length()
        return size, value if value >= 0 else value + (1 << size) - 1

    def encode(coefficients, previous):
        size, value = magnitude(coefficients[0] - previous)
        put(DC_SYMBOLS.index(size), 4)
        put(value, size)
        run = 0
        for coefficient in coefficients[1:]:
            if coefficient == 0:
                run += 1
                continue
            while run > 15:
                put(AC_SYMBOLS.index(0xF0), 8)
                run -= 16
            size, value = magnitude(coefficient)
            put(AC_SYMBOLS.index(run << 4 | size), 8)
            put(value, size)
            run = 0
        if run:
            put(AC_SYMBOLS.index(0x00), 8)
        return coefficients[0]

    previous = [0, 0, 0]
    for my in range(padded_h // 16):
        for mx in range(padded_w // 16):
            for dy in (0, 1):
                for dx in (0, 1):
                    previous[0] = encode(block(luma, 2 * mx + dx, 2 * my + dy, tables[0]),
                                         previous[0])
            for component, plane in enumerate(chroma, start=1):
                previous[component] = encode(block(plane, mx, my, tables[1]),
                                             previous[component])
    bits.extend([1] * (-len(bits) % 8))
    scan = bytearray()
    for start in range(0, len(bits), 8):
        byte = int("".join(map(str, bits[start:start + 8])), 2)
        scan += bytes((byte, 0)) if byte == 0xFF else bytes((byte,))

    def segment(marker, payload):
        return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload

    def huffman(kind, symbols, length):
        counts = [0] * 16
        counts[length - 1] = len(symbols)
        return bytes((kind << 4,)) + bytes(counts) + bytes(symbols)

    return (b"\xff\xd8"
            + segment(0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0")
            + segment(0xDB, b"".join(bytes((index,)) + bytes(table[i] for i in ZIGZAG)
                                     for index, table in enumerate(tables)))
            + segment(0xC0, struct.pack(">BHHB", 8, height, width, 3)
                      + bytes((1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1)))
            + segment(0xC4, huffman(0, DC_SYMBOLS, 4) + huffman(1, AC_SYMBOLS, 8))
            + segment(0xDA, bytes((3, 1, 0, 2, 0, 3, 0, 0, 63, 0)))
            + bytes(scan) + b"\xff\xd9")


def probes():
    """(id, format, width, height, pixel, expected answer or None)."""
    items = []
    for name, rgb in COLORS.items():
        for width, height in SOLID_SIZES:
            items.append((f"solid-{name}-{width}x{height}", "png", width, height,
                          lambda x, y, rgb=rgb: rgb, name))
        items.append((f"solid-{name}-256x256", "jpeg", 256, 256, lambda x, y, rgb=rgb: rgb, name))

    def square(x, y):
        """blue-square.png: a 161-pixel blue square at 48..208 on white."""
        return COLORS["blue"] if 48 <= x < 209 and 48 <= y < 209 else WHITE

    items.append(("square-blue-256x256", "png", 256, 256, square, "blue"))
    items.append(("square-blue-256x256", "jpeg", 256, 256, square, "blue"))
    red, blue = COLORS["red"], COLORS["blue"]
    items.append(("split-red-blue-left-right-256x256", "png", 256, 256,
                  lambda x, y: red if x < 128 else blue, None))
    items.append(("split-red-blue-left-right-256x256", "jpeg", 256, 256,
                  lambda x, y: red if x < 128 else blue, None))
    items.append(("split-red-blue-top-bottom-256x256", "png", 256, 256,
                  lambda x, y: red if y < 128 else blue, None))
    items.append(("split-red-blue-left-right-320x192", "png", 320, 192,
                  lambda x, y: red if x < 160 else blue, None))
    return items


def encode_probe(fmt, width, height, pixel):
    return (png if fmt == "png" else jpeg)(width, height, pixel)


def probe_files():
    """(file name, bytes, mime, expected) for every probe."""
    for name, fmt, width, height, pixel, expected in probes():
        suffix, mime = (".png", "image/png") if fmt == "png" else (".jpg", "image/jpeg")
        yield name + suffix, encode_probe(fmt, width, height, pixel), mime, expected


def messages(prompt, image, mime):
    """The question alone when `prompt` is None, else with the image first
    ("image-first") or last ("text-first")."""
    if prompt is None:
        return [{"role": "user", "content": QUESTION}]
    part = {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"}}
    text = {"type": "text", "text": QUESTION}
    return [{"role": "user", "content": [part, text] if prompt == "image-first" else [text, part]}]


def color_scores(top):
    """Best logprob per color word among one position's top tokens."""
    scores = {}
    for entry in top:
        word = (entry.get("token") or "").strip().casefold()
        if word in COLOR_WORDS:
            scores[word] = max(scores.get(word, float("-inf")), entry["logprob"])
    return scores


def answer_start(tokens):
    """Position of the first answer token among the generated `tokens`: the
    first, unless the frontend refuses disabled thinking; then the first
    nonblank token after its reasoning close."""
    if thinking_can_be_disabled():
        return 0
    close = frontend()["format"]["parser"]["reasoning_close"]
    closes = [index for index, token in enumerate(tokens) if token["token"].strip().endswith(close)]
    start = next((index for index in range(closes[-1] + 1, len(tokens)) if tokens[index]["token"].strip()),
                 None) if closes else None
    if start is None:
        raise ValueError("server returned no answer after the reasoning")
    return start


def ask(client, model, content):
    body = {"model": model, "messages": content, "temperature": 0, "max_tokens": answer_budget(8),
            "logprobs": True, "top_logprobs": 20,
            **answer_control()}
    response, _ = client.request("/v1/chat/completions", body)
    choice = response["choices"][0]
    tokens = (choice.get("logprobs") or {}).get("content") or []
    if not tokens:
        raise ValueError("server returned no answer logprobs")
    tokens = tokens[answer_start(tokens):]
    top = [[{"token": entry["token"], "logprob": entry["logprob"]}
            for entry in token.get("top_logprobs") or []] for token in tokens]
    scores = color_scores(top[0])
    return {"answer": choice["message"].get("content") or "",
            "tokens": [token["token"] for token in tokens],
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "top_logprobs": top, "color_scores": scores,
            "top_color": max(scores, key=scores.get) if scores else None}


def run(client, model):
    rows = [{"id": "no-image", "prompt": "text-only", "expected": None}]
    try:
        rows[0].update(ask(client, model, messages(None, None, None)))
    except Exception as error:
        rows[0]["error"] = str(error)
    for file, image, mime, expected in probe_files():
        for prompt in PROMPTS:
            row = {"id": file, "prompt": prompt, "expected": expected,
                   "sha256": hashlib.sha256(image).hexdigest()}
            try:
                row.update(ask(client, model, messages(prompt, image, mime)))
                row["correct"] = (None if expected is None
                                  else expected in row["answer"].casefold())
            except Exception as error:
                row["error"] = str(error)
            rows.append(row)
    errors = [row for row in rows if "error" in row]
    summary = {}
    for prompt in PROMPTS:
        graded = [row for row in rows if row["prompt"] == prompt and row.get("correct") is not None]
        summary[prompt] = {"graded": len(graded),
                           "correct": sum(row["correct"] for row in graded),
                           "answers": sorted({row["answer"].strip() for row in graded})}
    return {"passed": not errors, "record_only": True, "question": QUESTION,
            "errors": errors, "summary": summary, "rows": rows}


def compare(first, second):
    """Per-probe answers of two receipts, keyed by (id, prompt)."""
    def load(path):
        value = json.loads(Path(path).read_text())
        return value["server"], {(row["id"], row["prompt"]): row for row in value["rows"]}
    (left_server, left), (right_server, right) = load(first), load(second)
    rows = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key, {}), right.get(key, {})
        shared = a.get("color_scores", {}).keys() & b.get("color_scores", {}).keys()
        rows.append({"id": key[0], "prompt": key[1],
                     "expected": a.get("expected", b.get("expected")),
                     "answers": [a.get("answer"), b.get("answer")],
                     "top_colors": [a.get("top_color"), b.get("top_color")],
                     "same_answer": a.get("answer") == b.get("answer"),
                     "max_abs_color_dlogprob": max(
                         (abs(a["color_scores"][word] - b["color_scores"][word]) for word in shared),
                         default=None)})
    return {"servers": [left_server, right_server], "rows": rows,
            "same_answers": sum(row["same_answer"] for row in rows), "total": len(rows)}


def write(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for file, image, mime, expected in probe_files():
        (directory / file).write_bytes(image)
        manifest[file] = {"sha256": hashlib.sha256(image).hexdigest(), "mime": mime,
                          "expected": expected}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--write", type=Path, metavar="DIR", help="write the probe images")
    parser.add_argument("--compare", nargs=2, type=Path, metavar="RECEIPT")
    args = parser.parse_args()
    if args.write:
        manifest = write(args.write)
        print(json.dumps({"directory": str(args.write), "probes": len(manifest)}))
        return
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=2))
        return
    if args.mechanics_only:
        parser.error("color probes need a served model with an image encoder")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    result = run(client, server["model"])
    # No timed work, so no host load samples: the receipt's timing reads informational.
    receipt("color-probe", server, result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
