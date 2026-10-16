"""Reference runner workloads for Qwen3.8's teacher texts, image probes and
long-context texts.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/qwen38/reference_workloads.py \\
        --refs "$CHOPINFER_DATA/qwen38-refs" --output DIR
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/qwen38/reference_workloads.py \\
        --answers RUN/f32.json --output DIR
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/qwen38/reference_workloads.py --probe-workload

Writes, in DIR (refused if it exists):

- `teacher.json`: the texts of fixtures/qwen38/teacher-texts.json, scored
  from teacher-reference.json's positions (the last 512 of long_4k);
- `probes-greedy.json`: the 7 image probes of fixtures/qwen38/frontend/images.json
  with greedy answers (64 tokens at most, ending at an end token);
- one `<image>.f32` / `<image>-grid.i64` pair per image;
- `long-context.json`: the frozen prompt set's long texts beyond the
  teacher set's long_4k that fit the runner (`LONG_CONTEXT`), each scored
  over its last `score_last` positions, from the full prompt file under
  `--refs`.

With `--answers`, DIR already holds those files and gains `probes.json`: the
probes as the gates score them, from `probes-greedy.json` and the greedy
answers of the runner's f32 output `RUN/f32.json`. A short answer stays
greedy, since its gate holds the served answer to the reference's greedy
steps. A free-text probe forces its F32 answer as a served chat renders it:
without the end token and, for an answer that starts with whitespace, the
text without it tokenized as the chat template trims it.

With `--probe-workload`, it rebuilds the image probe reference's workload
from tracked inputs alone, on CPU, and installs it where
models/qwen38/qual.json declares it (`references.image_probes.workload`,
under $CHOPINFER_DATA; `--output DIR` writes it elsewhere): the probes of
fixtures/qwen38/image-probe-references.json in the bound order, each short
answer greedy and each free-text answer forcing the bound
`answer_token_ids`. The result must have the binding's workload digest and
score its sequences as every bound mode does; an existing file that differs
is refused, never replaced.

Pixel rows come from tracked inputs: fixtures/qwen38/capture/images.py
writes every probe PNG (each must hash to fixtures/qwen38/frontend/identity.json),
the two images that need no resize go through image_rows.py's
`pixel_rows`, and the four others through `processed_rows`, the pinned
processor's steps (fixtures/qwen38/image-processor.json's recipe: smart_resize,
the uint8 bicubic antialiased resize torchvision calls, the fused 1/255 scale
and normalization in F32, the patch order). Each image's rows must hash to
image-processor.json's `pixels_sha256`, and each probe's concatenated rows
to the reference's `inputs.pixels_sha256`. The processor identity a row
file is keyed with is fixtures/qwen38/image-processor-identity.json (the
captured processor) or `pixel_rows`' source, with the checkpoint's
preprocessor_config.json. blue-square leads each probe workload, so the
runner's repeat check costs one short sequence.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fixtures/qwen38/capture"))
from tools.model_sdk.diagnostic import canonical  # noqa: E402

SCHEMA = "chopinfer.refrun.workload.v1"
FIXTURES = ROOT / "fixtures/qwen38"
MAX_NEW_TOKENS = 64
# generation_config.json eos_token_id, as the HF image references stopped.
STOP_IDS = (248044, 248046)
FIRST = "blue-square"
# The frozen prompt set's long texts that the teacher set does not hold and
# the runner can score in every mode within one GPU job. long_128k is left
# out: HF's eager text attention holds a [heads, 2048, 131072] score tensor
# per prefill chunk, and its f64 run would take about an hour.
LONG_CONTEXT = ("long_32k",)
# long_128k has a reference of its own, without the f64 run: its limits are
# the large-set floors, which the f64 spread cannot raise.
LONG_CONTEXT_128K = ("long_128k",)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def teacher():
    """The texts of teacher-texts.json, in its order, with the prompts and
    first scored positions of teacher-reference.json (the gate's f64
    reference); each prompt must hash to teacher-texts.json's value."""
    texts = json.loads((FIXTURES / "teacher-texts.json").read_text())["texts"]
    reference = {t["name"]: t for t in json.loads((FIXTURES / "teacher-reference.json").read_text())["texts"]}
    sequences = []
    for text in texts:
        ids = reference[text["name"]]["prompt_token_ids"]
        if sha256(json.dumps(ids).encode()) != text["prompt_token_ids_sha256"]:
            raise ValueError(f"{text['name']}: teacher-reference.json's prompt does not hash to teacher-texts.json's")
        sequences.append({"name": text["name"], "category": text["category"], "token_ids": ids,
                          "score_from": reference[text["name"]]["score_from"]})
    return sequences


def long_context(refs, names=LONG_CONTEXT):
    """The `names` texts of fixtures/qwen38/prompts.json, in its order, with
    token IDs from the full prompt file it pins, each scored over its last
    `score_last` positions."""
    frozen = json.loads((FIXTURES / "prompts.json").read_text())
    path = Path(refs) / "prompts/prompts.json"
    if sha256(path.read_bytes()) != frozen["full_file"]["sha256"]:
        raise ValueError(f"{path} is not the full prompt file fixtures/qwen38/prompts.json pins")
    full = {text["name"]: text for text in json.loads(path.read_text())["long"]}
    sequences = []
    for text in frozen["long"]:
        if text["name"] not in names:
            continue
        ids = full[text["name"]]["token_ids"]
        if sha256(json.dumps(ids).encode()) != text["token_ids_sha256"] or len(ids) != text["token_count"]:
            raise ValueError(f"{text['name']}: the full prompt file's tokens do not hash to the frozen set's")
        sequences.append({"name": text["name"], "category": "long-context", "token_ids": ids,
                          "score_from": len(ids) - text["score_last"]})
    if [s["name"] for s in sequences] != list(names):
        raise ValueError(f"the frozen prompt set lacks {names}")
    return sequences


def processor_identity(checkpoint, rebuilt):
    """The processor that made the rows (the pinned fast processor, or
    image_rows.py's `pixel_rows` for images that need no resize) and the
    checkpoint's preprocessor_config.json both follow."""
    import image_rows
    config = sha256((Path(checkpoint) / "preprocessor_config.json").read_bytes())
    if rebuilt:
        code = sha256(inspect.getsource(image_rows.pixel_rows).encode())
    else:
        code = sha256(canonical(json.loads((FIXTURES / "image-processor-identity.json").read_text())["identity"]))
    return {"code_sha256": code, "config_sha256": config}


def smart_resize(height, width, factor, minimum, maximum):
    """transformers' smart_resize (qwen2_vl): the nearest multiple of `factor`
    per side, scaled to keep the pixel count within [minimum, maximum]."""
    rows, columns = round(height / factor) * factor, round(width / factor) * factor
    if rows * columns > maximum:
        beta = math.sqrt((height * width) / maximum)
        rows = max(factor, math.floor(height / beta / factor) * factor)
        columns = max(factor, math.floor(width / beta / factor) * factor)
    elif rows * columns < minimum:
        beta = math.sqrt(minimum / (height * width))
        rows, columns = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return rows, columns


def processed_rows(image, recipe):
    """The pinned fast processor's rows for an (H, W, 3) uint8 image: resize
    to smart_resize's grid with torch's uint8 bicubic antialiased kernel (what
    torchvision's resize runs for uint8 on CPU), then (x - 255 mean) / (255 std)
    in F32, then the patch order of transformers' Qwen2VLImageProcessor."""
    import numpy as np
    import torch
    if recipe["interpolation"] != "bicubic-antialias" or recipe["upscale_small_side"]:
        raise ValueError(f"unsupported image recipe {recipe}")
    height, width, _ = image.shape
    patch, merge, temporal = recipe["patch"], recipe["merge"], recipe["temporal_patch"]
    size = smart_resize(height, width, recipe["factor"], recipe["minimum_pixels"], recipe["maximum_pixels"])
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].contiguous()
    resized = torch.nn.functional.interpolate(pixels, size=size, mode="bicubic", align_corners=False, antialias=True)
    if resized.dtype != torch.uint8:
        raise ValueError("the uint8 bicubic kernel returned another dtype")
    scale = 1.0 / recipe["value_scale"]
    mean = torch.tensor(recipe["mean"]) * scale
    std = torch.tensor(recipe["std"]) * scale
    values = (resized.to(torch.float32) - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    gh, gw = size[0] // patch, size[1] // patch
    patches = values.reshape(1, 3, gh // merge, merge, patch, gw // merge, merge, patch).permute(0, 2, 5, 3, 6, 1, 4, 7)
    rows = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1).reshape(
        gh * gw, 3 * temporal * patch * patch)
    return np.ascontiguousarray(rows.numpy(), dtype=np.float32), [1, gh, gw]


def probe_pixels():
    """{image: (rows, grid)} of every probe image, from tracked inputs, each
    checked against the tracked digests of its PNG and of its rows."""
    import images
    import image_rows
    import tempfile
    identity = json.loads((FIXTURES / "frontend/identity.json").read_text())["images"]
    pinned = json.loads((FIXTURES / "image-processor.json").read_text())
    pngs = images.functional() | {name: images.png(*size) for name, size in images.SIZES.items()}
    pixels = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, data in pngs.items():
            if sha256(data) != identity[name]:
                raise ValueError(f"{name}: the generated PNG does not hash to the capture's")
            path = Path(scratch) / f"{name}.png"
            path.write_bytes(data)
            image = image_rows.read_png(path)
            pixels[name] = (image_rows.pixel_rows(image) if name in images.FUNCTIONAL
                            else processed_rows(image, pinned["recipe"]))
            expected = pinned["images"].get(name)
            if expected is not None and (sha256(pixels[name][0].tobytes()) != expected["pixels_sha256"]
                                         or pixels[name][1] != expected["grid"]):
                raise ValueError(f"{name}: rows differ from image-processor.json's")
    return pixels


def probes(checkpoint, output):
    import images
    import numpy as np
    pixels = probe_pixels()
    media = {}
    for name, (rows, grid) in pixels.items():
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        grid = np.asarray([grid], dtype=np.int64)
        (output / f"{name}.f32").write_bytes(rows.tobytes())
        (output / f"{name}-grid.i64").write_bytes(grid.tobytes())
        media[name] = {"kind": "image",
                       "tensor": {"path": f"{name}.f32", "dtype": "f32", "shape": list(rows.shape)},
                       "grid": {"path": f"{name}-grid.i64", "dtype": "i64", "shape": [1, 3]},
                       "processor": processor_identity(checkpoint, name in images.FUNCTIONAL)}
    references = json.loads((FIXTURES / "image-probe-references.json").read_text())["probes"]
    greedy = []
    for probe in sorted(json.loads((FIXTURES / "frontend/images.json").read_text()),
                        key=lambda p: p["name"] != FIRST):
        name, prompt, reference = probe["name"], probe["token_ids"], references[probe["name"]]
        rows = b"".join((output / media[image]["tensor"]["path"]).read_bytes() for image in probe["images"])
        if sha256(rows) != reference["inputs"]["pixels_sha256"]:
            raise ValueError(f"{name}: pixel rows do not hash to the fixture's")
        if sha256(json.dumps(prompt).encode()) != reference["inputs"]["token_ids_sha256"]:
            raise ValueError(f"{name}: prompt does not hash to the fixture's")
        common = {"name": name, "category": reference["kind"], "token_ids": prompt, "score_from": len(prompt),
                  "media": [media[image] for image in probe["images"]]}
        greedy.append(common | {"answer": {"policy": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
                                           "stop_ids": list(STOP_IDS)}})
    return greedy


def rendered(tokenizer, answer):
    """The answer IDs a served chat renders for a generated `answer`: the
    end token dropped and, when its text starts with whitespace, the text
    without it tokenized again, which must change only the first token."""
    scored = answer[:-1] if answer and answer[-1] in STOP_IDS else answer
    text = tokenizer.decode(scored)
    if not text[:1].isspace():
        return scored
    ids = tokenizer(text.lstrip(), add_special_tokens=False)["input_ids"]
    if len(ids) != len(scored) or ids[1:] != scored[1:]:
        raise ValueError("the trimmed answer does not tokenize as the answer's tail")
    return ids


def answered(output, run, checkpoint):
    """`probes-greedy.json` with each free-text probe forcing its rendered
    answer from the runner output `run`; short answers stay greedy."""
    import transformers
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=str(Path(checkpoint) / "tokenizer.json"))
    greedy = json.loads((output / "probes-greedy.json").read_text())["sequences"]
    entries = json.loads(Path(run).read_text())["entries"]
    answers = {entry["name"]: entry["scores"]["answer"] for entry in entries
               if entry["key"]["sequence"]["answer"]["policy"] == "greedy"}
    if sorted(answers) != sorted(s["name"] for s in greedy):
        raise ValueError(f"{run} does not answer every probe of probes-greedy.json")
    sequences = []
    for sequence in greedy:
        if sequence["category"] == "free-text":
            sequence = sequence | {"answer": {"policy": "forced",
                                              "ids": rendered(tokenizer, answers[sequence["name"]])}}
        sequences.append(sequence)
    return sequences


def bound(output, checkpoint):
    """The image probe reference's bound workload, written to `output`: the
    greedy probes, with each free-text probe forcing the reference's
    `answer_token_ids`."""
    items = json.loads((FIXTURES / "image-probe-references.json").read_text())["probes"]
    sequences = []
    for sequence in probes(checkpoint, output):
        item = items[sequence["name"]]
        if item["prompt_token_ids"] != sequence["token_ids"] or item["score_from"] != sequence["score_from"]:
            raise ValueError(f"{sequence['name']}: the reference states another prompt")
        if item["kind"] == "free-text":
            sequence = sequence | {"answer": {"policy": "forced", "ids": item["answer_token_ids"]}}
        sequences.append(sequence)
    return sequences


def install(output, checkpoint):
    """Rebuild the probe workload in a scratch directory beside `output`,
    check it against the reference's binding, then move each file into
    `output`. A file already there must hold the same bytes."""
    import shutil
    import tempfile
    from tools.refrun import cache
    reference = json.loads((FIXTURES / "image-probe-references.json").read_text())
    name = "image-probe-references.workload.json"
    output.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=output, prefix=".rebuild-"))
    try:
        write(scratch / name, bound(scratch, checkpoint))
        cache.check_workload(reference["binding"], scratch / name)
        files = sorted(path.name for path in scratch.iterdir())
        for file in files:
            target = output / file
            if target.exists() and target.read_bytes() != (scratch / file).read_bytes():
                raise SystemExit(f"{target} exists with other bytes; not replaced")
        for file in files:
            if not (output / file).exists():
                os.replace(scratch / file, output / file)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"workload": str(output / name), "workload_sha256": reference["binding"]["workload_sha256"],
            "files": files}


def write(path, sequences):
    path.write_text(json.dumps({"schema": SCHEMA, "model": "qwen38", "sequences": sequences}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refs", help="qwen38-refs directory holding prompts/ (the full prompt file)")
    parser.add_argument("--probe-workload", action="store_true",
                        help="rebuild and install the image probe reference's workload")
    parser.add_argument("--answers", type=Path, help="the runner's f32.json of probes-greedy.json; "
                                                     "writes probes.json into the existing --output")
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_QWEN38_CHECKPOINT"),
                        help="checkpoint directory holding preprocessor_config.json")
    parser.add_argument("--long-128k", action="store_true",
                        help="write only long-context-128k.json (long_128k) into the existing --output")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.long_128k:
        if not (args.refs and args.output):
            parser.error("--long-128k needs --refs and --output")
        path = args.output / "long-context-128k.json"
        if path.exists():
            raise SystemExit(f"{path} exists")
        write(path, long_context(args.refs, LONG_CONTEXT_128K))
        return
    if args.probe_workload:
        sys.path.insert(0, str(ROOT / "tools/qual"))
        from common import declared_path, read_config
        spec = read_config(str(ROOT / "models/qwen38/qual.json"))["qualification"]["references"]["image_probes"]
        output = args.output or declared_path(spec["workload"])
        if output is None:
            parser.error(f"set {spec['workload']['env']} or --output")
        print(json.dumps(install(output if args.output else output.parent, args.checkpoint), indent=1))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.answers:
        path = args.output / "probes.json"
        if path.exists():
            raise SystemExit(f"{path} exists")
        write(path, answered(args.output, args.answers, args.checkpoint))
        return
    if not args.refs:
        parser.error("--refs is required without --answers")
    args.output.mkdir(parents=True)
    write(args.output / "teacher.json", teacher())
    write(args.output / "probes-greedy.json", probes(args.checkpoint, args.output))
    write(args.output / "long-context.json", long_context(args.refs))


if __name__ == "__main__":
    main()
