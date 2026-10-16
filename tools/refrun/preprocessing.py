"""Checkpoint frontend references, produced only by a queued reference job.

Images use the checkpoint's default AutoImageProcessor backend.
The separate audio environment runs torchaudio's Hann sinc and HTK mel.
No serving implementation or serving recipe is used to compute references.
"""

import argparse
from array import array
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

from tools.model_sdk.diagnostic import canonical
from tools.refrun import keys

SCHEMA = "chopinfer.refrun.preprocessing.v1"
CODE = ("tools/refrun/preprocessing.py",)
SIZES = ((256, 256), (17, 19), (100, 60), (333, 517), (1000, 40), (4500, 4000))
RATES = (16000, 22050, 24000, 44100, 48000)


def write_tensor(root, name, data, dtype, shape):
    path = root / name
    path.write_bytes(data)
    return {"file": name, "dtype": dtype, "shape": list(shape), "sha256": keys.sha256(data)}


def packed(values, dtype):
    values = array({"f32": "f", "f64": "d", "u8": "B"}[dtype], values)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def binding(checkpoint, configs, packages, requirements):
    versions = {name: importlib.metadata.version(name) for name in packages}
    return {"code": keys.repo_files(CODE), "configs": {name: keys.file_sha256(checkpoint / name) for name in configs},
            "environment": versions, "requirements": keys.repo_files((requirements,))}


def declared_overrides(declaration):
    """The check's typed constant, in a raw or parsed qualification declaration."""
    checks = declaration.get("checks", {})
    check = (checks.get("media_preprocessing", {}) if isinstance(checks, dict) else
             next((c for c in checks if c.get("module") == "media_preprocessing"), {}))
    overrides = check.get("constants", {}).get("media_preprocessing", {}).get("PROCESSOR_OVERRIDES", {}).get("value", {})
    allowed = {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225], "max_pixels": 1048576, "min_pixels": 8192}
    if overrides and overrides != allowed:
        raise ValueError("processor overrides must be exactly the documented ImageNet statistics and serving pixel bounds")
    return overrides


def processor(checkpoint, overrides=None):
    """Load the checkpoint default backend; no fast/slow substitution."""
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True, **(overrides or {}))


def image_modes(checkpoint, image, overrides=None):
    import numpy as np
    import torch
    from transformers import image_processing_backends
    from transformers.image_transforms import rescale

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        single_processor = processor(checkpoint, overrides)
        integer = single_processor(images=image, do_rescale=False, do_normalize=False, return_tensors="np")
        single = single_processor(images=image, return_tensors="np")
        repeated = single_processor(images=image, return_tensors="np")
        if single["pixel_values"].tobytes() != repeated["pixel_values"].tobytes():
            raise ValueError("nondeterministic image processor")
        torch.set_default_dtype(torch.float64)
        double_processor = processor(checkpoint, overrides)
        if type(double_processor).__name__.endswith("Pil"):
            def rescale64(*values, **kwargs):
                return rescale(*values, **kwargs, dtype=np.float64)
            original_patchify, original_asarray = double_processor.patchify, np.asarray
            def asarray64(value, dtype=None, *values, **kwargs):
                return original_asarray(value, np.float64 if dtype == np.float32 else dtype, *values, **kwargs)
            def patchify64(*values, **kwargs):
                with patch.object(np, "asarray", asarray64):
                    return original_patchify(*values, **kwargs)
            with patch.object(image_processing_backends, "np_rescale", rescale64), \
                    patch.object(double_processor, "patchify", patchify64):
                double = double_processor(images=image, return_tensors="np")
        else:
            # The integer resize stays unchanged. Upstream constructs its
            # fused statistics in the default dtype; lift the exact uint8
            # input back to F64 after its explicit F32 normalization cast.
            normalize = double_processor.normalize
            def normalize64(image, mean, std, **kwargs):
                return normalize(image.to(torch.float64), mean, std, **kwargs)
            with patch.object(double_processor, "normalize", normalize64):
                double = double_processor(images=image, return_tensors="np")
        if double["pixel_values"].dtype != np.float64:
            raise ValueError("the reference did not preserve F64 normalization")
        return integer, single, double
    finally:
        torch.set_default_dtype(previous)


def images(args):
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1)
    overrides = {}
    if args.declaration is not None:
        declared = json.loads(args.declaration.read_text())
        overrides = declared_overrides(declared)
    default = processor(args.checkpoint)
    effective = processor(args.checkpoint, overrides)
    record = binding(args.checkpoint, ("preprocessor_config.json",),
                     ("torch", "torchvision", "transformers", "pillow", "numpy"),
                     "tools/refrun/requirements-cu130.txt")
    record["processor"] = {"class": type(effective).__name__, "module": type(effective).__module__,
                           "default_config": default.to_dict(), "config": effective.to_dict(),
                           "overrides": overrides, "transformers_sha256": keys.installed_transformers()[1]}
    cases = []
    for width, height in SIZES:
        name = f"image-{width}x{height}"
        y, x = np.indices((height, width), dtype=np.uint32)
        rgb = np.stack((x * 255 // max(1, width - 1), y * 255 // max(1, height - 1),
                        np.where((x * 7 + y * 3) % 97 < 48, 255, 0)), axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb)
        source = write_tensor(args.output, name + ".rgb", rgb.tobytes(), "u8", rgb.shape)
        integer, single, double = image_modes(args.checkpoint, image, overrides)
        stages = {}
        for label, result, dtype in (("integer", integer, "u8"), ("f32", single, "f32"), ("f64", double, "f64")):
            rows = result["pixel_values"]
            encoded = rows.astype({"u8": "u1", "f32": "<f4", "f64": "<f8"}[dtype]).tobytes()
            stages[label] = write_tensor(args.output, f"{name}.{label}", encoded, dtype, rows.shape)
        cases.append({"name": name, "kind": "image", "source": source,
                      "grid": single["image_grid_thw"].tolist()[0], "stages": {"patches": stages}})
    return record, cases


def audio(args):
    import torch
    import torchaudio
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads((args.checkpoint / "audio_tokenizer/config.json").read_text())
    target = config["sampling_rate"]
    record = binding(args.checkpoint, ("audio_tokenizer/config.json",), ("torch", "torchaudio"),
                     "tools/refrun/requirements-audio-reference.txt")
    record["processor"] = {"resampling_method": "sinc_interp_hann", "lowpass_filter_width": 6,
                           "rolloff": 0.99, "mel_scale": "htk", "norm": None, "power": 1,
                           "center": True, "pad_mode": "reflect", "log_floor": 1e-7}
    if args.declaration is None:
        raise ValueError("audio preprocessing needs its declared upstream source")
    declaration = json.loads(args.declaration.read_text())
    source = declaration["inputs"]["media_audio_source"]
    record["upstream"] = json.loads((keys.ROOT / source).read_text())
    record["code"].update(keys.repo_files((source,)))
    cases = []
    for rate in RATES:
        name = f"audio-{rate}"
        # The same continuous, broadband chirp plus two tones at every rate.
        samples = torch.tensor([0.25 * math.sin(2 * math.pi * (110 * (i / rate) + 1800 * (i / rate) ** 2))
                                + 0.1 * math.sin(2 * math.pi * 997 * i / rate)
                                + 0.05 * math.cos(2 * math.pi * 3701 * i / rate)
                                for i in range(rate // 4 + 7)], dtype=torch.float32)
        source = write_tensor(args.output, name + ".samples", packed(samples.tolist(), "f32"), "f32", samples.shape)
        stages = {"resampled": {}, "mel": {}}
        for label, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            torch.set_default_dtype(dtype)
            def compute():
                resampled = torchaudio.functional.resample(samples.to(dtype), rate, target,
                                                          lowpass_filter_width=6, rolloff=0.99,
                                                          resampling_method="sinc_interp_hann")
                transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=target, n_fft=config["nfft"], win_length=config["window_size"],
                    hop_length=config["hop_length"], f_min=config["fmin"], f_max=config["fmax"],
                    n_mels=config["n_mels"], power=1, normalized=False, center=True, pad_mode="reflect",
                    norm=None, mel_scale="htk").to(dtype)
                mel = transform(resampled).clamp_min(1e-7).log().T.contiguous()
                return resampled, mel
            result = compute()
            if any(not torch.equal(a, b) for a, b in zip(result, compute())):
                raise ValueError(f"{name}/{label}: nondeterministic audio preprocessing")
            for stage, value in zip(stages, result):
                stages[stage][label] = write_tensor(args.output, f"{name}.{stage}.{label}",
                    packed(value.flatten().tolist(), label), label, value.shape)
        cases.append({"name": name, "kind": "audio", "sample_rate": rate, "source": source, "stages": stages})
    return record, cases


def verify(record, root):
    if record.get("schema") != SCHEMA or not record.get("cases"):
        raise ValueError("missing preprocessing reference cases")
    body = {key: value for key, value in record.items() if key != "sha256"}
    if keys.sha256(canonical(body)) != record.get("sha256"):
        raise ValueError("preprocessing reference digest mismatch")
    for section in ("code", "requirements"):
        for name, digest in record["binding"][section].items():
            if keys.file_sha256(keys.ROOT / name) != digest:
                raise ValueError(f"stale preprocessing reference: {name}")
    for case in record["cases"]:
        for spec in (case["source"], *(spec for stage in case["stages"].values() for spec in stage.values())):
            path = Path(spec["file"])
            if path.name != str(path) or keys.file_sha256(root / path) != spec["sha256"]:
                raise ValueError("preprocessing tensor digest mismatch")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("images", "audio"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--declaration", type=Path, help="qualification declaration with documented processor overrides")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    record, cases = {"images": images, "audio": audio}[args.kind](args)
    result = {"schema": SCHEMA, "binding": record, "cases": cases}
    result["sha256"] = keys.sha256(canonical(result))
    (args.output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()
