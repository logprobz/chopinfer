"""Workloads and fixture skeletons of MiMo's reference-code references.

The teacher and media gate references are scored by the reference runner
(tools/refrun) with models/mimo26/reference.py, which runs the vendor's
modeling_mimo_v2.py on the GPUs, and assembled by `python3 -m tools.refrun
emit` from a skeleton written here:

- teacher: the 33 texts of teacher-exact-reference.json (names, categories,
  prompt token IDs and score_from of teacher-texts.json's frozen workload),
  each scored from score_from;
- media: one sequence per case of multimodal/reference.json, its prompt
  with the frozen answer forced from answer_token_start, and the case's
  processor output (`--payloads`: the HF F32 pixel rows in <item>.f32 and
  grids in <item>.i64).

Variants: `exact` is the F32 run (each exact dequantized product rounded
once to F32, the `rounded` rule), the gate variant; `bf16` (upstream BF16
loading and activations) and `f64` are the spread variants that set the
gate limits (tools/qual/limits.py).

    python3 fixtures/mimo26/capture/runner_references.py workloads --payloads DIR --out DIR
    python3 fixtures/mimo26/capture/runner_references.py skeletons --out DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures/mimo26"
MULTIMODAL = FIXTURES / "multimodal"
SCHEMA = "chopinfer.refrun.workload.v1"
TEACHER_SCHEMA = "chopinfer.mimo26.teacher-exact-reference.v3"
MEDIA_SCHEMA = "chopinfer.mimo26.multimodal-exact-reference.v2"
VARIANTS = {
    "exact": "reference runner f32: vendor modeling_mimo_v2.py in F32, each exact dequantized product rounded "
             "once to F32 (rounded), F32 log-softmax; the gate variant",
    "bf16": "reference runner bf16: weights rounded to BF16 as upstream loading, BF16 activations and "
            "encoders, F32 log-softmax",
    "f64": "reference runner f64: exact dequantized products, f64 activations and log-softmax",
}
# Encoder payload of each media item: kind, file, dtype, shape and grid.
PAYLOADS = {"blue-square": ("image", "blue-square.f32", "f32", [256, 1536], [1, 16, 16]),
            "frames-00": ("image", "frames-00.f32", "f32", [256, 1536], [1, 16, 16]),
            "frames-04": ("image", "frames-04.f32", "f32", [256, 1536], [1, 16, 16]),
            "video": ("video", "video.f32", "f32", [1024, 1536], [4, 16, 16])}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def items(case):
    """The payload item of a media case, or None for a text case."""
    if case["kind"] == "text":
        return None
    if case["kind"] == "image":
        name = Path(case["file"]).stem
        if name.isdigit():
            name = f"frames-{int(name) // 4 * 4:02d}"
        return name
    return {"video": "video"}[case["kind"]]


def teacher_texts():
    texts = json.loads((FIXTURES / "teacher-exact-reference.json").read_text())["texts"]
    return [{key: text[key] for key in ("name", "category", "prompt_token_ids", "score_from")} for text in texts]


def media_cases():
    return json.loads((MULTIMODAL / "reference.json").read_text())["cases"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def workloads(payloads, out):
    teacher = [{"name": t["name"], "category": t["category"], "token_ids": t["prompt_token_ids"],
                "score_from": t["score_from"]} for t in teacher_texts()]
    write_json(out / "teacher.json", {"schema": SCHEMA, "model": "mimo26", "sequences": teacher})
    processor = {"code_sha256": sha256(FIXTURES / "capture/frontend_media.py"),
                 "config_sha256": sha256(MULTIMODAL / "processor/processor.json")}
    (out / "media").mkdir(parents=True, exist_ok=True)
    media = []
    for case in media_cases():
        ids, start = case["forced_prompt_token_ids"], case["answer_token_start"]
        sequence = {"name": case["id"], "category": case["kind"], "token_ids": ids[:start], "score_from": start,
                    "answer": {"policy": "forced", "ids": ids[start:]}}
        item = items(case)
        if item is not None:
            kind, payload, dtype, shape, grid = PAYLOADS[item]
            (out / "media" / payload).write_bytes((Path(payloads) / payload).read_bytes())
            spec = {"kind": kind, "tensor": {"path": f"media/{payload}", "dtype": dtype, "shape": shape},
                    "processor": processor}
            if grid is not None:
                name = f"media/{item}-grid.i64"
                (out / name).write_bytes(b"".join(v.to_bytes(8, "little") for v in grid))
                spec["grid"] = {"path": name, "dtype": "i64", "shape": [1, 3]}
            sequence["media"] = [spec]
        media.append(sequence)
    write_json(out / "media.json", {"schema": SCHEMA, "model": "mimo26", "sequences": media})


def provenance(source):
    pin = json.loads((ROOT / "models/mimo26/qual.json").read_text())["qualification"]["checkpoint"]
    checkpoint = {key: pin[key] for key in ("pin", "repository", "revision", "served_model_name",
                                            "source_lock_sha256", "checkpoint_identity")}
    return {"source": source, "checkpoint": checkpoint, "adapter": "models/mimo26/reference.py",
            "variants": VARIANTS, "gate_variant": "exact",
            "binding": "each variant is the reference runner's scores of the bound cache entries "
                       "(tools/refrun/README.md, fixture bindings)"}


def skeletons(out):
    write_json(out / "teacher.json", {
        "schema": TEACHER_SCHEMA,
        "provenance": provenance("tools/refrun over teacher-exact-reference.workload.json: the texts of "
                                 "teacher-texts.json, each scored from score_from"),
        "texts": teacher_texts()})
    capture = MULTIMODAL / "reference.json"
    cases = [{"id": case["id"], "kind": case["kind"],
              "answer_token_ids": case["forced_prompt_token_ids"][case["answer_token_start"]:]}
             for case in media_cases()]
    write_json(out / "media.json", {
        "schema": MEDIA_SCHEMA, "capture_sha256": sha256(capture),
        "provenance": provenance("tools/refrun over exact-reference.workload.json: the frozen answers of "
                                 "multimodal/reference.json teacher-forced after each case's prompt and "
                                 "processor output"),
        "cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    built = commands.add_parser("workloads", help="the teacher and media workloads")
    built.add_argument("--payloads", required=True, type=Path, help="directory of the encoder payload files")
    built.add_argument("--out", required=True, type=Path)
    commands.add_parser("skeletons", help="the fixture skeletons for emit").add_argument(
        "--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "workloads":
        workloads(args.payloads, args.out)
    else:
        skeletons(args.out)


if __name__ == "__main__":
    main()
