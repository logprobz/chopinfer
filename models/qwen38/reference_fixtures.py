"""Skeletons of Qwen3.8's bound gate references, filled by the runner's emit.

    python3 models/qwen38/reference_fixtures.py teacher --workload W/teacher.json --output S.json
    python3 models/qwen38/reference_fixtures.py long-context --workload W/long-context.json --output S.json
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/qwen38/reference_fixtures.py probes \\
        --workload W/probes.json --greedy RUN/f32.json --output S.json

then `python3 -m tools.refrun emit --skeleton S.json ... --modes f32=exact,bf16,f64`
(tools/refrun/README.md). A skeleton holds everything a gate reads except
the scores emit copies from the runs: the gate variant `exact` is the F32
run, and `bf16` and `f64` are the spread variants whose distance from it
sets the limits (tools/qual/limits.py).

- `teacher` and `long-context`: schema `chopinfer.qwen38.teacher-reference.v2`,
  one text per workload sequence (name, category, prompt token IDs and first
  scored position), as chopinfer eval run's teacher capture reads it.
- `probes`: schema `chopinfer.qwen38.image-probe-references.v5`, one probe per
  workload sequence of reference_workloads.py's `probes.json`, with the
  answer the F32 run generated (`--greedy`, the runner's f32.json of
  `probes-greedy.json`): its text, IDs and whether it stopped. A short
  answer is bound by its greedy entry; emit adds its steps (`--greedy-fields`).
  A free-text probe forces the answer as a served chat renders it
  (`answer_token_ids`), and keeps the generated answer's steps, which no
  gate uses, under `generated` with their entry key. `token_bytes` holds
  the bytes of every token an answer or a step's top 5 names, since a
  served answer is compared by its token bytes.

The probe rule is the free-text probes' per-probe spread rule, unchanged
from the CPU references (tools/qual/multimodal.py).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
from common import checkpoint_pin, read_config  # noqa: E402
from tools.refrun import keys  # noqa: E402

GATE = ROOT / "models/qwen38/qual.json"
TEACHER_SCHEMA = "chopinfer.qwen38.teacher-reference.v2"
PROBE_SCHEMA = "chopinfer.qwen38.image-probe-references.v5"
CAPTURE = ROOT / "fixtures/qwen38/frontend/images.json"
TOP = 5
STOP_IDS = (248044, 248046)
CODE = "HF transformers 5.16.1 Qwen4ExpForConditionalGeneration (models/qwen38/reference.py)"
VARIANTS = {
    "exact": f"{CODE} in F32 on the GPU runner (tools/refrun), FP8 weights times their scales exactly "
             "(dequant exact): the gate variant",
    "bf16": f"{CODE} in BF16, weights dequantized and rounded to BF16 as HF loads them (dequant hf-bf16), "
            "F32 log-softmax: spread variant",
    "f64": f"{CODE} in f64 with HF's own F32 steps (each run's layout.upstream_f32), dequant exact: "
           "spread variant",
}
POSITIONS = ("entry i scores the token at score_from + i from the logits of the row before it; top1 is the "
             "highest logit (ties to the lower ID), margin its logit minus the second's, and the top-k columns "
             "are ranked by logit")
RULE = {"variants": ["bf16", "f64"], "mean_step": 0.05, "mean_floor": 0.15, "top1_floor": 0.08,
        "near_tie": 0.3,
        "text": "short answers: the served greedy answer equals the F32 reference's, or follows it until a token "
                "the F32 reference ranks within near_tie of its own (logits); free text: the F32 answer "
                "teacher-forced as a served chat renders it, mean |dlogprob| <= max(mean_floor, "
                "the worst variant rounded up to mean_step) and the share of positions whose top-1 differs from "
                "the F32 pass's <= max(top1_floor, the worst variant), over the BF16 and f64 runs scored against "
                "the F32 run like a server"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sequences(path):
    return json.loads(Path(path).read_text())["sequences"]


def provenance(workload, **extra):
    pin = checkpoint_pin(read_config(str(GATE)))
    return {"source": "tools/refrun emit of models/qwen38/reference.py runs (models/qwen38/reference_fixtures.py)",
            "checkpoint": {key: pin[key] for key in ("repository", "revision", "served_model_name",
                                                     "source_lock_sha256", "checkpoint_identity")},
            "workload": Path(workload).name, "gate_variant": "exact", "variants": VARIANTS,
            "positions": POSITIONS, **extra}


def texts(workload):
    """A teacher-reference skeleton over a text workload."""
    return {"schema": TEACHER_SCHEMA, "provenance": provenance(workload),
            "texts": [{"name": s["name"], "category": s["category"], "prompt_token_ids": s["token_ids"],
                       "score_from": s["score_from"]} for s in sequences(workload)]}


def byte_decoder():
    """GPT-2's byte-level BPE alphabet, inverted: token characters to bytes."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    codes, extra = [], 0
    for byte in range(256):
        if byte in printable:
            codes.append(byte)
        else:
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in enumerate(codes)}


def token_bytes(tokenizer, ids):
    """Each token's bytes: byte-level BPE characters decoded, an added
    (special) token's text as UTF-8."""
    decoder, added = byte_decoder(), set(tokenizer.get_added_vocab().values())
    result = {}
    for token in sorted(set(ids)):
        text = tokenizer.convert_ids_to_tokens(token)
        result[str(token)] = list(text.encode() if token in added else bytes(decoder[c] for c in text))
    return result


def probes(workload, greedy, checkpoint):
    """An image-probe skeleton over reference_workloads.py's probes.json and
    the F32 greedy run of probes-greedy.json."""
    import transformers
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=str(Path(checkpoint) / "tokenizer.json"))
    run = json.loads(Path(greedy).read_text())
    if run["policy"]["name"] != "f32":
        raise ValueError(f"{greedy} is not an f32 run")
    generated = {e["name"]: e for e in run["entries"] if e["key"]["sequence"]["answer"]["policy"] == "greedy"}
    images = {probe["name"]: probe["images"] for probe in json.loads(CAPTURE.read_text())}
    root, items, ids = Path(workload).parent, {}, set()
    for sequence in sequences(workload):
        name, entry = sequence["name"], generated[sequence["name"]]
        if entry["key"]["sequence"]["token_ids_sha256"] != keys.ids_sha256(sequence["token_ids"]):
            raise ValueError(f"{name}: the greedy run scored another prompt")
        answer, scores = entry["scores"]["answer"], entry["scores"]
        stopped = bool(answer) and answer[-1] in STOP_IDS
        pixels = b"".join((root / media["tensor"]["path"]).read_bytes() for media in sequence["media"])
        item = {"kind": sequence["category"],
                "inputs": {"token_ids_sha256": sha256(json.dumps(sequence["token_ids"]).encode()),
                           "pixels_sha256": sha256(pixels), "images": images[name]},
                "prompt_token_ids": sequence["token_ids"], "score_from": sequence["score_from"],
                "answer": {"text": tokenizer.decode(answer[:-1] if stopped else answer), "token_ids": answer,
                           "stopped": stopped}}
        ids.update(answer)
        steps = {field: scores[field] for field in ("logprob", "top1", "margin")}
        steps |= {field: [row[:TOP] for row in scores[field]] for field in ("topk_ids", "topk_logits",
                                                                            "topk_logprobs")}
        ids.update(i for row in steps["topk_ids"] for i in row)
        if sequence["answer"]["policy"] == "forced":
            item["answer_token_ids"] = sequence["answer"]["ids"]
            item["generated"] = {"key_sha256": entry["key_sha256"], "answer": answer, **steps}
        elif sequence["answer"]["policy"] != "greedy":
            raise ValueError(f"{name}: unknown answer policy")
        items[name] = item
    capture = sha256(CAPTURE.read_bytes())
    return {"schema": PROBE_SCHEMA,
            "provenance": provenance(workload, capture_sha256=capture,
                                     generated={"run": str(Path(greedy).parent.name), "mode": "f32"}),
            "rule": RULE, "token_bytes": token_bytes(tokenizer, ids), "probes": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=("teacher", "long-context", "probes"))
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--greedy", type=Path, help="probes: the runner's f32.json of probes-greedy.json")
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_QWEN38_CHECKPOINT"),
                        help="probes: checkpoint directory holding tokenizer.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.kind == "probes":
        if args.greedy is None:
            parser.error("probes needs --greedy")
        value = probes(args.workload, args.greedy, args.checkpoint)
    else:
        value = texts(args.workload)
    with args.output.open("x") as stream:
        json.dump(value, stream)
        stream.write("\n")


if __name__ == "__main__":
    main()
