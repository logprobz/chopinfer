"""Frozen functional media prompts shared by capture and qualification."""

import base64
import hashlib
import json
import os
from pathlib import Path


FIXTURES = Path(os.environ.get("CHOPINFER_MIMO_FIXTURES", "fixtures/mimo26")) / "functional"


def cases():
    image_prompt = "What color is the square? Answer with one color word."
    for name in ("blue-square.png", *(f"frames/{i:02d}.png" for i in range(8))):
        yield {"id": "image-" + name.replace("/", "-"), "kind": "image",
               "file": name, "prompt": image_prompt}
    yield {"id": "video-red-then-blue", "kind": "video",
           "file": "red-then-blue.mp4",
           "prompt": "What colors appear in the video, in order?"}
    yield {"id": "text-parts", "kind": "text",
           "prompt": "Say the two words red blue in order."}


def verified_bytes(case, fixture_dir=FIXTURES):
    path = fixture_dir / case["file"]
    expected = json.loads((fixture_dir / "manifest.json").read_text())["files"][case["file"]]
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{case['file']} sha256 mismatch: {actual}")
    return data, actual


def messages(case, fixture_dir=FIXTURES, *, media_bytes=None):
    if case["kind"] == "text":
        parts = [{"type": "text", "text": "Say the two words "},
                 {"type": "text", "text": "red blue in order."}]
    else:
        data = media_bytes
        if data is None:
            data, _ = verified_bytes(case, fixture_dir)
        mime = {"image": "image/png", "audio": "audio/wav",
                "video": "video/mp4"}[case["kind"]]
        url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        part_type = case.get("transport") or {"image": "image_url", "audio": "audio_url",
                                              "video": "video_url"}[case["kind"]]
        media_part = ({"type": "input_audio", "input_audio": {
            "data": base64.b64encode(data).decode("ascii"), "format": "wav"}}
            if part_type == "input_audio" else
            {"type": part_type, part_type: {"url": url}})
        parts = [{"type": "text", "text": case["prompt"] + " "}, media_part]
    return [{"role": "user", "content": parts}]
