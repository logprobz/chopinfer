"""MiMo mixed-media encoder workloads from upstream preprocessing.

Run in a queued reference job. Existing frozen single-item gate cases are
kept. Audio is served but not qualified in-tree, so no case holds audio.
"""

import argparse
import base64
import copy
import json
from pathlib import Path

from tools.model_sdk.diagnostic import canonical
from tools.refrun import keys, preprocessing, workload

CASES = (
    ("single-image", ("blue",)),
    ("two-images", ("blue", "red")),
    ("two-resized-images", ("small", "gradient")),
    ("image-video", ("blue", "video")),
    ("32-images", ("blue",) * 32),
)


def prepare(args):
    import numpy as np
    from PIL import Image
    from tokenizers import Tokenizer
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "media").mkdir()
    (args.output / "raw").mkdir()
    (args.output / "bodies").mkdir()
    declaration = json.loads(args.declaration.read_text())
    overrides = preprocessing.declared_overrides(declaration)
    proc = preprocessing.processor(args.checkpoint, overrides)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    def fixture(name):
        data = (args.fixtures / name).read_bytes()
        if keys.sha256(data) != manifest["files"][name]:
            raise ValueError(f"changed fixture: {name}")
        return data
    raw = {"blue": ("image", "image/png", fixture("blue-square.png")),
           "red": ("image", "image/png", fixture("frames/00.png")),
           "video": ("video", "video/mp4", fixture("red-then-blue.mp4"))}
    import io
    for name, width, height in (("small", 100, 60), ("gradient", 333, 517)):
        y, x = np.indices((height, width), dtype=np.uint32)
        rgb = np.stack((x * 255 // (width - 1), y * 255 // (height - 1),
                        np.where((x * 7 + y * 3) % 97 < 48, 255, 0)), axis=-1).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        raw[name] = ("image", "image/png", stream.getvalue())
    code = keys.sha256(canonical(keys.repo_files(("models/mimo26/media_workload.py", "tools/refrun/preprocessing.py"))))
    processor_identity = {"code_sha256": code, "config_sha256": keys.sha256(canonical(proc.to_dict()))}
    media = {}
    def vision(name, rows, grid):
        (args.output / "media" / f"{name}.f32").write_bytes(rows.numpy().astype("<f4").tobytes())
        (args.output / "media" / f"{name}.i64").write_bytes(grid.numpy().astype("<i8").tobytes())
        media[name] = {"kind": raw[name][0], "processor": processor_identity,
                       "raw_sha256": keys.sha256(raw[name][2]),
                       "tensor": {"path": f"media/{name}.f32", "dtype": "f32", "shape": list(rows.shape)},
                       "grid": {"path": f"media/{name}.i64", "dtype": "i64", "shape": list(grid.shape)}}
    for name, (kind, _, data) in raw.items():
        (args.output / "raw" / f"{name}.bin").write_bytes(data)
        if kind == "image":
            result = proc(images=Image.open(io.BytesIO(data)).convert("RGB"), return_tensors="pt")
            vision(name, result["pixel_values"], result["image_grid_thw"])
    # Use the verified decoded RGB input, not the lossless source PNGs:
    # the compressed clip's red/blue values differ from those PNGs.
    decoded_root = keys.ROOT / "fixtures/mimo26/multimodal/processor"
    decoded_path = decoded_root / "video-rgb.npy"
    decoded_manifest = json.loads((decoded_root / "processor-manifest.json").read_text())
    if keys.file_sha256(decoded_path) != decoded_manifest["files"][decoded_path.name]:
        raise ValueError("decoded video RGB fixture digest differs")
    decoded = np.load(decoded_path, allow_pickle=False)
    if decoded.dtype != np.uint8 or decoded.shape != (8, 256, 256, 3):
        raise ValueError("unexpected decoded video fixture shape or dtype")
    frames = [Image.fromarray(frame) for frame in decoded]
    if any(a.tobytes() != b.tobytes() for a, b in zip(frames[::2], frames[1::2])):
        raise ValueError("video fixture's decoded temporal pairs are no longer identical")
    video = [proc(images=frame, return_tensors="pt") for frame in frames[::2]]
    grid = video[0]["image_grid_thw"].clone()
    if any(not torch.equal(item["image_grid_thw"], grid) for item in video):
        raise ValueError("video grids differ")
    grid[0, 0] = len(video)
    vision("video", torch.cat([item["pixel_values"] for item in video]), grid)
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    prompt = "Describe the media briefly."
    sequences, requests = [], []
    for name, names in CASES:
        text = "<|im_start|>user\n" + prompt + " "
        content = [{"type": "text", "text": prompt + " "}]
        for part in names:
            kind, mime, data = raw[part]
            transport = {"image": "image_url", "video": "video_url"}[kind]
            content.append({"type": transport, transport: {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}})
            if kind == "image":
                rows = media[part]["tensor"]["shape"][0] // 4
                text += "<|vision_start|>" + "<|image_pad|>" * rows + "<|vision_end|>"
            else:
                text += "<|vision_start|><|mimo_video_start|>"
                for index, item in enumerate(video):
                    text += f"00:{index * 2:02}<|vision_start|>" + "<|video_pad|>" * (item["pixel_values"].shape[0] // 4) + "<|vision_end|>"
                text += "<|mimo_video_end|><|vision_end|>"
        ids = encode(text + "<|im_end|><|im_start|>assistant\n<think></think>")
        sequences.append({"name": name, "category": "mixed-media", "token_ids": ids, "score_from": len(ids),
                          "answer": {"policy": "forced", "ids": encode("The")},
                          "media": [copy.deepcopy(media[part]) for part in names]})
        body = {"messages": [{"role": "user", "content": content}], "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0, "max_tokens": 1}
        (args.output / "bodies" / f"{name}.json").write_bytes(canonical(body))
        requests.append(f"{name}=/v1/chat/completions={args.output / 'bodies' / (name + '.json')}")
    doc = {"schema": workload.SCHEMA, "model": "mimo26", "sequences": sequences}
    (args.output / "encoder-workload.json").write_bytes(canonical(doc))
    (args.output / "workload.json").write_bytes(canonical({**doc, "sequences": [s for s in sequences if s["name"] != "32-images"]}))
    (args.output / "requests.txt").write_text("\n".join(requests) + "\n")
    (args.output / "producer.json").write_bytes(canonical({"code": keys.repo_files(("models/mimo26/media_workload.py", "tools/refrun/preprocessing.py")),
        "processor": {"class": type(proc).__name__, "config": proc.to_dict(), "overrides": overrides},
        "decoded_video_rgb_sha256": keys.file_sha256(decoded_path),
        "sources": {name: keys.sha256(value[2]) for name, value in raw.items()}}))
    print(json.dumps({"encoder_sequences": len(sequences), "forced_sequences": len(sequences) - 1}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "fixtures", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--declaration", type=Path)
    args = parser.parse_args()
    if args.declaration is None:
        parser.error("mixed workloads need --declaration")
    prepare(args)


if __name__ == "__main__":
    main()
