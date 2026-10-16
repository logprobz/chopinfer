"""Check that image content conditions MiMo answers, with no frozen reference.

A red, a green and a blue square on white (the geometry of blue-square.png)
each get the fixture's color question. For each, the first answer token must
rank the square's color above every other color word, which a server that
ignores image content cannot do for all three. Solid single-color images get
a neutral question too, but are only recorded: MiMo's vision tower maps any
uniform image to nearly the same large-norm row (red and blue rows have
cosine 0.989), so the model barely tells them apart in either engine. The
logprob gain over the same question without an image is recorded. An
optional photo is described with and without the image; its description
must be at least GAIN_LIMIT nats more likely with the image. The same run
against vLLM tells engine faults apart from model behavior.
"""

import base64
from pathlib import Path
import struct
import zlib

from common import answer_budget, answer_kwargs, Client, arguments, not_applicable, receipt


SQUARE = "What color is the square? Answer with one color word."
QUESTION = "What is the main color of this image? Answer with one color word."
DESCRIBE = "Describe this image in one sentence."
SOLID = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
COLOR_WORDS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "black",
               "white", "gray", "grey", "brown")
# Minimum logprob gain of a photo description from adding the image. One nat
# is far above the run-to-run noise of either engine.
GAIN_LIMIT = 1.0


def png(rgb, size=256, square=None):
    """A size x size RGB PNG: solid `rgb`, or a centered `square` of side
    `square` in `rgb` on white (blue-square.png is 161 pixels at 48..208)."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    solid = b"\0" + bytes(rgb) * size
    rows = [solid] * size
    if square is not None:
        start = (size - square + 1) // 2
        white = b"\xff\xff\xff"
        inside = b"\0" + white * start + bytes(rgb) * square + white * (size - start - square)
        rows = [inside if start <= y < start + square else b"\0" + white * size
                for y in range(size)]
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def user(prompt, image=None, mime="image/png"):
    if image is None:
        return [{"role": "user", "content": prompt}]
    url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    return [{"role": "user", "content": [{"type": "text", "text": prompt + " "},
                                         {"type": "image_url", "image_url": {"url": url}}]}]


class Probe:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def body(self, messages, **extra):
        return {"model": self.model, "messages": messages, "temperature": 0,
                "chat_template_kwargs": answer_kwargs(), **extra}

    def answer(self, messages, max_tokens):
        response, _ = self.client.request("/v1/chat/completions",
                                          self.body(messages, max_tokens=answer_budget(max_tokens)))
        return (response["choices"][0]["message"].get("content") or "",
                response["usage"]["prompt_tokens"])

    def forced(self, messages, answer):
        """Prompt logprob rows of `answer` appended as the assistant turn."""
        response, _ = self.client.request("/v1/chat/completions", self.body(
            messages + [{"role": "assistant", "content": answer}],
            continue_final_message=True, add_generation_prompt=False, max_tokens=1,
            prompt_logprobs=20, return_token_ids=True))
        ids = response.get("prompt_token_ids")
        rows = response.get("prompt_logprobs")
        if not isinstance(ids, list) or not isinstance(rows, list) or len(ids) != len(rows):
            raise ValueError("server returned no aligned prompt logprobs")
        return ids, rows


def color_scores(row):
    """Best first-token logprob per color word, ignoring case and spacing."""
    scores = {}
    for entry in row.values():
        word = (entry.get("decoded_token") or "").strip().casefold()
        if word in COLOR_WORDS:
            scores[word] = max(scores.get(word, float("-inf")), entry["logprob"])
    return scores


def answer_logprob(ids, rows, start):
    return sum(rows[position][str(ids[position])]["logprob"] for position in range(start, len(ids)))


def color_case(probe, color, baseline, square):
    """Gated for squares; recorded only for solid images (see the docstring)."""
    question = SQUARE if square else QUESTION
    messages = user(question, png(SOLID[color], square=161 if square else None))
    text, start = probe.answer(messages, 16)
    ids, rows = probe.forced(messages, text)
    scores = color_scores(rows[start])
    best = max(scores, key=scores.get) if scores else None
    gain = scores.get(color, float("-inf")) - baseline[question].get(color, float("-inf"))
    correct = best == color and color in text.casefold()
    return {"id": f"{'square' if square else 'solid'}-{color}", "answer": text,
            "color_scores": scores, "top_color": best, "gain_over_no_image": gain,
            "correct": correct, "gated": square, "passed": correct or not square}


def photo_case(probe, path):
    from refs import fixture_bytes
    data = fixture_bytes(path)
    mime = "image/jpeg" if path.suffix.casefold() in (".jpg", ".jpeg") else "image/png"
    with_image = user(DESCRIBE, data, mime)
    text, start = probe.answer(with_image, 64)
    ids, rows = probe.forced(with_image, text)
    conditioned = answer_logprob(ids, rows, start)
    plain = user(DESCRIBE)
    plain_text, plain_start = probe.answer(plain, 64)
    plain_ids, plain_rows = probe.forced(plain, text)
    unconditioned = answer_logprob(plain_ids, plain_rows, plain_start)
    gain = conditioned - unconditioned
    return {"id": f"photo-{path.name}", "answer": text, "answer_without_image": plain_text,
            "answer_logprob": conditioned, "answer_logprob_without_image": unconditioned,
            "gain_over_no_image": gain, "passed": gain >= GAIN_LIMIT}


def run(probe, photo):
    rows = []
    baseline = {}
    try:
        for question in (SQUARE, QUESTION):
            text, start = probe.answer(user(question), 16)
            _, plain = probe.forced(user(question), text)
            baseline[question] = color_scores(plain[start])
            rows.append({"id": "no-image", "question": question, "answer": text,
                         "color_scores": baseline[question], "passed": True})
    except Exception as error:
        return {"passed": False, "errors": [{"id": "no-image", "error": str(error)}],
                "rows": []}
    cases = [lambda color=color, square=square: color_case(probe, color, baseline, square)
             for square in (True, False) for color in SOLID]
    if photo is not None:
        cases.append(lambda: photo_case(probe, photo))
    for case in cases:
        try:
            rows.append(case())
        except Exception as error:
            rows.append({"passed": False, "error": str(error)})
    return {"passed": all(row["passed"] for row in rows),
            "errors": [row for row in rows if not row["passed"]], "rows": rows,
            "limits": {"gain_over_no_image": GAIN_LIMIT}}


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--photo", type=Path, help="Optional real photo (PNG or JPEG)")
    args = parser.parse_args()
    if args.mechanics_only:
        parser.error("grounding needs the served model and its image encoder")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    if reason := not_applicable(server, "image"):
        # Image grounding does not apply to a server without image input.
        receipt("grounding", server, {"passed": False, "not_applicable": [reason]})
        raise SystemExit(0)
    result = run(Probe(client, server["model"]), args.photo)
    receipt("grounding", server, result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
