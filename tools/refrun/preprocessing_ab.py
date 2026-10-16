"""Paired F32 preprocessing A/B on labeled, resized images.

Run through the GPU queue. Alternatives use the same probe groups, with
arm-specific grids and right padding. Forced answers and raw images are unchanged.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

from tools.model_sdk.diagnostic import canonical
from tools.refrun import keys, modes, preprocessing, stamps, workload
from tools.refrun.run import device_label, load_adapter

VARIANTS = ("bilinear-8192", "bicubic-8192", "bicubic-3136", "bilinear-3136")


def sign_test(differences):
    if not differences or any(not math.isfinite(d) for d in differences):
        raise ValueError("paired differences must be nonempty and finite")
    positive = sum(d > 0 for d in differences)
    negative = sum(d < 0 for d in differences)
    count = positive + negative
    p = min(1.0, 2 * sum(math.comb(count, i) for i in range(min(positive, negative) + 1)) / 2 ** count)
    return {"positive": positive, "negative": negative, "ties": len(differences) - count, "p": p,
            "mean_logprob_difference": sum(differences) / len(differences)}


def prepare(args):
    import numpy as np
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from torchvision.transforms.v2 import functional as tvf

    overrides = preprocessing.declared_overrides(json.loads(args.declaration.read_text()))
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    probes = json.loads(args.probes.read_text())["probes"]
    if not probes or len({p["name"] for p in probes}) != len(probes):
        raise ValueError("probes must be nonempty with unique names")
    def area(probe):
        with Image.open(args.probes.parent / probe["image"]) as image:
            return image.width * image.height
    probes.sort(key=area, reverse=True)
    specs = {variant: [] for variant in VARIANTS}
    sources, resized = [], {variant: 0 for variant in VARIANTS}
    torch.set_num_threads(1)
    for probe in probes:
        path = args.probes.parent / probe["image"]
        if keys.file_sha256(path) != probe["sha256"]:
            raise ValueError("raw probe image digest mismatch")
        image = Image.open(path).convert("RGB")
        for variant in VARIANTS:
            root = args.output / "workloads" / variant
            (root / "media").mkdir(parents=True, exist_ok=True)
            proc = preprocessing.processor(args.checkpoint, {**overrides,
                "min_pixels": int(variant.rsplit("-", 1)[1]),
                "resample": Image.Resampling.BILINEAR if variant.startswith("bilinear") else Image.Resampling.BICUBIC})
            if variant.startswith("bilinear"):
                # The old served recipe: floating bilinear, no antialiasing,
                # with its small-side upscale and the arm's pixel minimum.
                h, w = image.height, image.width
                if min(h, w) < 32:
                    h, w = (32, int(w * 32 / h)) if h < w else (int(h * 32 / w), 32)
                rh, rw = round(h / 32) * 32, round(w / 32) * 32
                if rh * rw > 1048576:
                    beta = math.sqrt(h * w / 1048576)
                    rh, rw = max(32, math.floor(h / beta / 32) * 32), max(32, math.floor(w / beta / 32) * 32)
                elif rh * rw < int(variant.rsplit("-", 1)[1]):
                    beta = math.sqrt(int(variant.rsplit("-", 1)[1]) / (h * w))
                    rh, rw = math.ceil(h * beta / 32) * 32, math.ceil(w * beta / 32) * 32
                rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
                rgb = tvf.resize(rgb, [rh, rw], interpolation=tvf.InterpolationMode.BILINEAR, antialias=False)
                result = proc(images=rgb, do_resize=False, return_tensors="pt")
            else:
                result = proc(images=image, return_tensors="pt")
            grid, pixels = result["image_grid_thw"], result["pixel_values"].float()
            resized[variant] += int(tuple((grid[0, 1:] * proc.patch_size).tolist()) != (image.height, image.width))
            stem = f"media/{probe['name']}"
            payload = pixels.numpy().astype("<f4").tobytes()
            (root / f"{stem}.bin").write_bytes(payload)
            (root / f"{stem}.grid").write_bytes(grid.numpy().astype("<i8").tobytes())
            item = {"kind": "image", "tensor": {"path": f"{stem}.bin", "dtype": "f32", "shape": list(pixels.shape)},
                    "grid": {"path": f"{stem}.grid", "dtype": "i64", "shape": list(grid.shape)},
                    "processor": {"code_sha256": keys.file_sha256(Path(__file__)), "config_sha256": keys.sha256(canonical(proc.to_dict()))},
                    "raw_sha256": keys.file_sha256(path)}
            text = "<|im_start|>user\n" + probe["question"] + " <|vision_start|>" + "<|image_pad|>" * (pixels.shape[0] // 4)
            ids = encode(text + "<|vision_end|><|im_end|><|im_start|>assistant\n<think></think>")
            answer = encode(probe["answer"])
            if not answer:
                raise ValueError("probe has no answer tokens")
            specs[variant].append({"name": probe["name"], "category": probe["category"], "token_ids": ids,
                "score_from": len(ids), "answer": {"policy": "forced", "ids": answer}, "media": [item]})
            sources.append({"case": probe["name"], "scene": probe["scene"], "variant": variant,
                            "raw_sha256": item["raw_sha256"], "tensor_sha256": keys.sha256(payload),
                            "grid": grid.tolist(), "size": list(image.size), "answer_ids": answer})
    require_effective_probes(sources, resized)
    for variant, sequences in specs.items():
        (args.output / "workloads" / variant / "workload.json").write_bytes(canonical({
            "schema": workload.SCHEMA, "model": json.loads(args.declaration.read_text())["model"], "sequences": sequences}))
    return {"sources": sources, "resized_images": resized, "overrides": overrides,
            "probe_manifest_sha256": keys.file_sha256(args.probes), "processor": type(proc).__name__}


def require_effective_probes(sources, resized):
    if not all(resized.get(v, 0) > 0 for v in VARIANTS):
        raise ValueError("A/B requires resized probes in every variant")
    rows = {(r["case"], r["variant"]): r for r in sources}
    names = {r["case"] for r in sources}
    for left, right, field in ((VARIANTS[3], VARIANTS[2], "tensor_sha256"), (VARIANTS[1], VARIANTS[2], "grid")):
        if not any(rows[n, left][field] != rows[n, right][field] for n in names):
            raise ValueError(f"A/B has no changed {field} for {left} versus {right}")


def batched(reference, sequences, policy, device, tensor, pad=151643):
    import torch
    from tools.refrun import score
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    model = reference.model
    rows = [list(s.forced_ids) for s in sequences]
    length = max(map(len, rows))
    keep = length - min(s.score_from for s in sequences) + 1
    with modes.active(policy):
        images = [m for s in sequences for m in s.media]
        if any(m.kind != "image" for m in images):
            raise ValueError("the paired probe batch supports images only")
        kwargs = {"pixel_values": torch.cat([tensor(m.tensor, device) for m in images]),
                  "image_grid_thw": torch.cat([tensor(m.grid, device) for m in images])}
        ids = torch.tensor([row + [pad] * (length - len(row)) for row in rows], device=device)
        shape = torch.empty((len(rows), length, 1), dtype=model.model.embed_tokens.weight.dtype, device=device)
        masks = {"full_attention": create_causal_mask(model.config, shape, None, None),
                 "sliding_window_attention": create_sliding_window_causal_mask(model.config, shape, None, None)}
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False, logits_to_keep=keep, **kwargs).logits
        result = []
        for b, (sequence, row) in enumerate(zip(sequences, rows)):
            first = sequence.score_from - 1 - (length - keep)
            block = logits[b, first:first + len(row) - sequence.score_from]
            result.append(score.rows(block, row[sequence.score_from:], modes.torch_dtype(policy["log_softmax"])))
        return result


def main():
    from tools.refrun import adapter as interface, place, quant
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "probes", "declaration", "output", "lock"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--lock-variant", default="mopd")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    modes.prepare_environment(cuda=True)
    prepared = prepare(args)
    sequences = {variant: workload.load(args.output / "workloads" / variant / "workload.json")[1] for variant in VARIANTS}
    module, adapter_path = load_adapter(args.adapter)
    devices = args.devices.split(",")
    lock = keys.read_lock(args.lock)[args.lock_variant]
    root = args.checkpoint
    verified = stamps.verify(root, lock, ["config.json", place.INDEX, *place.shard_files(root), *module.EXTRA_SHARDS], stamps.default_cache())
    checkpoint = place.Checkpoint(root, module.EXTRA_SHARDS)
    units = module.units(checkpoint)
    for device in devices:
        quant.check_codecs(device)
    placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units if not u.host], devices)
    store = place.Store(checkpoint, placement, units)
    policy = modes.policy("f32", "rounded")
    dtype = modes.torch_dtype(policy["dtype"])
    reference = module.build(interface.BuildConfig(checkpoint, policy, dtype, store), placement)
    interface.check_tensor_names(reference, store.tensors)
    quant.configure(reference.model, dtype, policy["dequant"])
    place.attach(reference.model, placement)
    report = {"schema": "chopinfer.refrun.preprocessing-ab.v1", **prepared, "policy": policy,
              "code": keys.repo_files(["tools/refrun/preprocessing_ab.py", "tools/refrun/preprocessing.py", adapter_path, *keys.NUMERICS]),
              "reference_code": keys.reference_code({k: keys.file_sha256(v) for k, v in module.reference_code(checkpoint).items()}),
              "checkpoint": keys.checkpoint(lock, verified), "environment": keys.environment(device_label(devices)),
              "batch_size": args.batch, "entries": {variant: [] for variant in VARIANTS}}
    for start in range(0, len(sequences[VARIANTS[0]]), args.batch):
        began = time.monotonic()
        for variant in VARIANTS:
            batch = sequences[variant][start:start + args.batch]
            scores = batched(reference, batch, policy, placement.first, module.tensor)
            if start == 0 and canonical(scores) != canonical(batched(reference, batch, policy, placement.first, module.tensor)):
                raise ValueError("paired reference failed repeatability")
            report["entries"][variant].extend({"name": s.name, "category": s.category, "scores": value}
                                              for s, value in zip(batch, scores))
        print(json.dumps({"completed": min(start + args.batch, len(sequences[VARIANTS[0]])), "seconds": time.monotonic() - began}), flush=True)
        (args.output / "scores.json").write_bytes(canonical(report))
    comparisons = {}
    for label, left, right in (("interpolation", "bilinear-3136", "bicubic-3136"),
                               ("minimum_bicubic", "bicubic-8192", "bicubic-3136"),
                               ("minimum_bilinear", "bilinear-8192", "bilinear-3136")):
        differences = [sum(b["scores"]["logprob"]) - sum(a["scores"]["logprob"])
                       for a, b in zip(report["entries"][left], report["entries"][right])]
        # Aggregate scales of one source before the sign test. Correlated
        # resizes are not independent votes for significance.
        scene_for = {r["case"]: r["scene"] for r in report["sources"]}
        grouped = {}
        for row, difference in zip(report["entries"][left], differences, strict=True):
            grouped.setdefault(scene_for[row["name"]], []).append(difference)
        scene_differences = {name: sum(values) / len(values) for name, values in grouped.items()}
        result = {**sign_test(list(scene_differences.values())), "scene_differences": scene_differences,
                  "paired_probe_differences": differences}
        result["selected"] = (right if result["positive"] > result["negative"] else left) if result["p"] < 0.01 else right
        comparisons[label] = {"left": left, "right": right, **result}
    stamps.confirm(root, verified)
    interpolation = comparisons["interpolation"]["selected"].split("-")[0]
    report["selected_recipe"] = comparisons["minimum_" + interpolation]["selected"]
    report["comparisons"] = comparisons
    (args.output / "scores.json").write_bytes(canonical(report))
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
