"""Regenerate media prompts and tensors with HF preprocessing on fixed raw inputs.

The decoded video RGB is fixed input data, authenticated by the checked-in
manifest. No video decoder or model weights are loaded by this CPU producer.
The declared ImageNet normalization and pixel bounds remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.refrun import preprocessing
from multimodal_cases import cases, messages


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    processor_root = args.output / "processor"
    processor_root.mkdir(exist_ok=True)
    media_root = args.output / "media"
    media_root.mkdir(exist_ok=True)
    previous = ROOT / "fixtures/mimo26/multimodal/processor"
    manifest = json.loads((previous / "processor-manifest.json").read_text())
    raw_manifest = json.loads((args.fixtures / "manifest.json").read_text())
    def raw(name):
        path = args.fixtures / name
        if digest(path) != raw_manifest["files"][name]:
            raise ValueError(f"raw input changed: {name}")
        return path
    video_path = previous / "video-rgb.npy"
    if digest(video_path) != manifest["files"][video_path.name]:
        raise ValueError("fixed decoded video input changed")
    video = np.load(video_path, allow_pickle=False)
    if video.shape != (8, 256, 256, 3) or video.dtype != np.uint8 or not np.array_equal(video[::2], video[1::2]):
        raise ValueError("fixed video must contain four identical temporal pairs")
    # Preserve the full .npy byte stream, including its header and digest.
    (processor_root / video_path.name).write_bytes(video_path.read_bytes())
    overrides = preprocessing.declared_overrides(json.loads((ROOT / "models/mimo26/qual.json").read_text()))
    proc = preprocessing.processor(args.checkpoint, overrides)
    files, records, tensors = {}, {}, {}
    def save(name, value):
        value = np.ascontiguousarray(value)
        path = processor_root / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        files[path.name] = digest(path)
        return {"file": path.name, "dtype": str(value.dtype), "shape": list(value.shape)}
    for name in ("blue-square.png", *(f"frames/{i:02d}.png" for i in range(8))):
        rgb = np.asarray(Image.open(raw(name)).convert("RGB"))
        result = proc(images=Image.fromarray(rgb), return_tensors="np")
        stem = name.replace("/", "-").removesuffix(".png")
        pixels = np.asarray(result["pixel_values"], dtype=np.float32)
        grid = np.asarray(result["image_grid_thw"], dtype=np.int32)
        records[name] = {"decoded_rgb": save(stem + "-rgb", rgb), "pixel_values": save(stem + "-pixels", pixels),
                         "image_grid_thw": save(stem + "-grid", grid)}
        tensors[stem] = (pixels, grid)
    # Identical temporal pairs reduce to the HF image processor's repeated
    # frame patchification, also used by the existing mixed-media reference.
    video_rows = [proc(images=Image.fromarray(frame), return_tensors="np") for frame in video[::2]]
    pixels = np.concatenate([row["pixel_values"] for row in video_rows]).astype(np.float32)
    grid = np.asarray([[4, 16, 16]], dtype=np.int32)
    tensors["video"] = (pixels, grid)
    files[video_path.name] = digest(video_path)
    records["red-then-blue.mp4"] = {
        "decoded_rgb": {"file": video_path.name, "dtype": "uint8", "shape": list(video.shape)},
        "pixel_values_videos": save("video-pixels", pixels), "video_grid_thw": save("video-grid", grid),
        "frame_timestamps": save("video-frame-timestamps", np.arange(8, dtype=np.float32)),
        "second_per_grid_ts": save("video-seconds-per-grid", np.asarray([2.0], dtype=np.float32)),
        "video_start_times": save("video-start-times", np.asarray([0.0], dtype=np.float32))}
    record = {"schema": "mimo26-processor-cpu/v1", "processor": type(proc).__name__, "overrides": overrides,
              "fixed_video_sha256": digest(video_path), "fixture_sha256": {name: raw_manifest["files"][name] for name in records},
              "cases": records, "files": files}
    write(processor_root / "processor.json", record)
    write(processor_root / "processor-manifest.json", {"schema": "mimo26-processor-manifest/v1",
          "files": {**files, "processor.json": digest(processor_root / "processor.json")}})
    identity = {"code_sha256": digest(Path(__file__)), "config_sha256": digest(processor_root / "processor.json")}
    media = {}
    for name, (pixels, grid) in tensors.items():
        (media_root / f"{name}.f32").write_bytes(pixels.astype("<f4").tobytes())
        (media_root / f"{name}.i64").write_bytes(grid.astype("<i8").tobytes())
        media[name] = {"kind": "video" if name == "video" else "image", "processor": identity,
            "tensor": {"path": f"media/{name}.f32", "dtype": "f32", "shape": list(pixels.shape)},
            "grid": {"path": f"media/{name}.i64", "dtype": "i64", "shape": list(grid.shape)}}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    stop = json.loads((args.checkpoint / "generation_config.json").read_text())["eos_token_id"]
    if isinstance(stop, int):
        stop = [stop]
    sequences, prompts = [], []
    for case in cases():
        text = tokenizer.apply_chat_template(messages(case, args.fixtures), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        item = None
        if case["kind"] == "image":
            item = case["file"].replace("/", "-").removesuffix(".png")
            text = text.replace("<|image_pad|>", "<|image_pad|>" * (tensors[item][0].shape[0] // 4))
        elif case["kind"] == "video":
            item = "video"
            frames = "".join(f"00:{i * 2:02}<|vision_start|>" + "<|video_pad|>" * 64 + "<|vision_end|>" for i in range(4))
            text = text.replace("<|video_pad|>", "<|mimo_video_start|>" + frames + "<|mimo_video_end|>")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompts.append({**case, "prompt_token_ids": ids, "answer_token_start": len(ids)})
        sequence = {"name": case["id"], "category": case["kind"], "token_ids": ids, "score_from": len(ids),
                    "answer": {"policy": "greedy", "max_new_tokens": 256, "stop_ids": stop}}
        if item:
            sequence["media"] = [media[item]]
        sequences.append(sequence)
    write(args.output / "prompts.json", prompts)
    write(args.output / "greedy-workload.json", {"schema": "chopinfer.refrun.workload.v1", "model": "mimo26", "sequences": sequences})
    print(json.dumps({"cases": len(sequences), "processor_arrays": len(files), "fixed_video_sha256": digest(video_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
