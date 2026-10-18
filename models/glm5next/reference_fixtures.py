"""Skeletons of GLM 5.3 Flash's bound gate references, filled by the runner's emit.

    python3 models/glm5next/reference_fixtures.py teacher --workload W/teacher.json --output S.json
    python3 models/glm5next/reference_fixtures.py teacher --workload C/chat.json --output S.json
    python3 models/glm5next/reference_fixtures.py long-context --workload W/long-context.json --output S.json
    python3 models/glm5next/reference_fixtures.py teacher-texts --workload W/teacher.json \\
        --output fixtures/glm5next/gate/teacher-texts.json
    python3 models/glm5next/reference_fixtures.py nvfp4-teacher --checkpoint DIR --pin PIN.json \\
        --workload W/teacher.json --output S.json
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/glm5next/reference_fixtures.py probes \\
        --workload W/probes.json --greedy RUN/f32.json --output S.json

then `python3 -m tools.refrun emit --skeleton S.json ... --modes f32=exact,bf16,f64` (tools/refrun/README.md;
`--modes f32=exact,bf16` for long-context, which has no f64 run). A skeleton holds everything a gate reads
except the scores emit copies from the runs: the gate variant `exact` is the F32 run (dequant rounded:
the checkpoint's FP8 block scales use the full F32 mantissa), and `bf16` and `f64` are the spread variants.
W is reference_workloads.py's output directory and C its `--chat` output; the chat reference is a teacher
skeleton over chat.json (`nvfp4-teacher` over it for a checkpoint variant).

- `teacher` and `long-context`: schema `chopinfer.glm5next.teacher-reference.v1`, one text per workload
  sequence (name, category, prompt token IDs and first scored position), the layout the teacher capture
  (chopinfer eval run) reads.
- `nvfp4-teacher`: the same over another checkpoint of the model that stores projections as NVFP4,
  qualified as an external variant (tools/qual/README.md#external-variants): its provenance is the
  checkpoint pin `--pin` (what `python3 -m tools.qual variant pin` prints) and its variants are
  described from the layout of `--checkpoint`'s safetensors headers (which projections are NVFP4);
  its runs are F32 and BF16 (`--modes f32=exact,bf16`).
- `probes`: schema `chopinfer.glm5next.image-probe-references.v1`, the image-probes format
  tools/qual/multimodal.py reads: `rule` (the free-text spread rule), `provenance.capture_sha256` (the
  request capture reference_workloads.py writes in `capture/`), `token_bytes`, and per probe of
  reference_workloads.py's `probes.json` (kind `free-text`) the answer the F32 run generated (`--greedy`,
  the runner's f32.json of `probes-greedy.json`, checked to be that workload's sequences, model and
  checkpoint lock): its served text and IDs, the forced IDs a server renders (`answer_token_ids`) and the
  generated steps (top 5) under `generated`.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
sys.path.insert(0, str(ROOT / "models/glm5next"))
from common import checkpoint_pin, read_config  # noqa: E402

GATE = ROOT / "models/glm5next/qual.json"
TEACHER_SCHEMA = "chopinfer.glm5next.teacher-reference.v1"
TEXTS_SCHEMA = "chopinfer.glm5next.teacher-texts.v1"
PROBE_SCHEMA = "chopinfer.glm5next.image-probe-references.v1"
TOP = 5
# generation_config.json eos_token_id.
STOP_IDS = (154820, 154827, 154829)
CODE = "HF transformers 5.16.1 Glm5NextForConditionalGeneration (models/glm5next/reference.py)"
VARIANTS = {
    "exact": f"{CODE} in F32 on the GPU runner (tools/refrun), FP8 weights times their F32 block scales rounded "
             "once to F32 (dequant rounded): the gate variant",
    "bf16": f"{CODE} in BF16, weights dequantized and rounded to BF16 as HF loads them (dequant hf-bf16), "
            "F32 log-softmax: spread variant",
    "f64": f"{CODE} in f64 with HF's own F32 steps (each run's layout.upstream_f32), dequant exact: "
           "spread variant",
}
# The free-text probes' spread rule, as Qwen's image probes state it (tools/qual/multimodal.py).
RULE = {"variants": ["bf16", "f64"], "mean_step": 0.05, "mean_floor": 0.15, "top1_floor": 0.08,
        "near_tie": 0.3,
        "text": "free text: the F32 answer teacher-forced as a served chat renders it (reasoning, </think> and the "
                "stripped content), mean |dlogprob| <= max(mean_floor, the worst variant rounded up to mean_step) "
                "and the share of positions whose top-1 differs from the F32 pass's <= max(top1_floor, the worst "
                "variant), over the BF16 and f64 runs scored against the F32 run like a server"}
# The long-context reference runs f32 and bf16 only: f64 stays within about 4K tokens.
LONG_CONTEXT_VARIANTS = ("exact", "bf16")
# The parts of a decoder layer a checkpoint may store as NVFP4, by the tensor names under `layers.N.`, in the
# order a description names them.
NVFP4_PARTS = (("dense MLPs", re.compile(r"mlp\.(?:gate|up|down)_proj\.weight$")),
               ("shared experts", re.compile(r"mlp\.shared_experts\.")),
               ("routed experts", re.compile(r"mlp\.experts\.")),
               ("attention projections", re.compile(r"self_attn\.")))


def safetensors_headers(checkpoint):
    """Each tensor's header entry, by name, over every safetensors file of
    the checkpoint and its non-hidden immediate subdirectories."""
    root, entries = Path(checkpoint), {}
    files = [*root.glob("*.safetensors"), *(path for path in root.glob("*/*.safetensors")
                                            if not path.parent.name.startswith("."))]
    for path in sorted(files):
        with path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        entries.update({name: entry for name, entry in header.items() if name != "__metadata__"})
    return entries


def nvfp4_parts(checkpoint):
    """The parts of the text layers the reference loads (the MTP layer is
    not loaded) that `checkpoint` stores as NVFP4: packed U8 codes with an
    E4M3 `weight_scale` per 16 and an F32 `weight_scale_2`."""
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    layers = (config.get("text_config") or config)["num_hidden_layers"]
    entries, found = safetensors_headers(checkpoint), set()
    for name, entry in entries.items():
        located = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", name)
        base = name[:-len("weight")]
        if (located is None or int(located.group(1)) >= layers or not name.endswith(".weight")
                or entry.get("dtype") != "U8" or entries.get(base + "weight_scale", {}).get("dtype") != "F8_E4M3"
                or entries.get(base + "weight_scale_2", {}).get("dtype") != "F32"):
            continue
        found |= {part for part, pattern in NVFP4_PARTS if pattern.match(located.group(2))}
    return [part for part, _ in NVFP4_PARTS if part in found]


def nvfp4_variants(checkpoint):
    """The F32 gate and BF16 spread variants of a reference over an NVFP4
    checkpoint, described from its layout (`nvfp4_parts`); everything it does
    not store as NVFP4 is BF16."""
    parts = nvfp4_parts(checkpoint)
    if not parts:
        raise ValueError(f"{checkpoint} stores no projection of its text layers as NVFP4")
    named = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return {
        "exact": f"{CODE} in F32 on the GPU runner (tools/refrun) over the NVFP4 checkpoint, {named}' E2M1 codes "
                 "times their E4M3 group scales and F32 global scale rounded once to F32 (dequant rounded), "
                 "everything else BF16: the gate variant",
        "bf16": f"{CODE} in BF16 over the NVFP4 checkpoint, weights dequantized and rounded to BF16 as HF loads "
                "them (dequant hf-bf16), F32 log-softmax: spread variant",
    }


POSITIONS = ("entry i scores the token at score_from + i from the logits of the row before it; top1 is the "
             "highest logit (ties to the lower ID), margin its logit minus the second's, and the top-k columns "
             "are ranked by logit")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sequences(path):
    return json.loads(Path(path).read_text())["sequences"]


def provenance(workload, variants=tuple(VARIANTS), gate=GATE, described=VARIANTS, **extra):
    """The provenance of a reference over `workload`: the checkpoint pin of
    the declaration `gate` (or the pin itself) and the `variants` (described
    by `described`)."""
    pin = gate if isinstance(gate, dict) else checkpoint_pin(read_config(str(gate)))
    return {"source": "tools/refrun emit of models/glm5next/reference.py runs (models/glm5next/reference_fixtures.py)",
            "checkpoint": {key: pin[key] for key in ("repository", "revision", "served_model_name",
                                                     "source_lock_sha256", "checkpoint_identity")},
            "workload": Path(workload).name, "gate_variant": "exact",
            "variants": {name: described[name] for name in variants},
            "positions": POSITIONS, **extra}


def texts(workload, variants=tuple(VARIANTS), gate=GATE, described=VARIANTS):
    """A teacher-reference skeleton over a text workload, whose provenance
    names the `variants` the runs give it (long-context has no f64 run) and
    the checkpoint of the declaration `gate`."""
    return {"schema": TEACHER_SCHEMA, "provenance": provenance(workload, variants, gate, described),
            "texts": [{"name": s["name"], "category": s["category"], "prompt_token_ids": s["token_ids"],
                       "score_from": s["score_from"]} for s in sequences(workload)]}


def frozen_texts(workload):
    """The frozen teacher text set the teacher gate holds a reference to
    (chopinfer eval run reads `<fixture_root>/teacher-texts.json`): each
    text's name, category and prompt token IDs' SHA256, in workload order."""
    return {"schema": TEXTS_SCHEMA,
            "provenance": "The frozen GLM 5.3 Flash teacher texts: reference_workloads.py's teacher.json, the short "
                          "texts and long_4k, in reference order.",
            "texts": [{"name": s["name"], "category": s["category"],
                       "prompt_token_ids_sha256": sha256(json.dumps(s["token_ids"]).encode())}
                      for s in sequences(workload)]}


def byte_decoder():
    """GPT-2's byte-level BPE alphabet, inverted: token characters to bytes
    (GLM's tokenizer.json is byte-level BPE)."""
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


def probes(workload, greedy, checkpoint, proc=None):
    """An image-probe skeleton over reference_workloads.py's probes.json and
    the F32 greedy run of probes-greedy.json beside it, in the form
    tools/qual/multimodal.py reads (format image-probes): every probe
    free-text, forcing the served form of its F32 answer."""
    import reference_workloads
    root = Path(workload).parent
    proc = proc or reference_workloads.processor(checkpoint)
    records, entries = reference_workloads.greedy_entries(root, greedy)
    greedy_sequences = {s["name"]: s for s in sequences(root / "probes-greedy.json")}
    items, ids = {}, set()
    for sequence in sequences(workload):
        name = sequence["name"]
        if {k: v for k, v in sequence.items() if k != "answer"} != \
                {k: v for k, v in greedy_sequences[name].items() if k != "answer"}:
            raise ValueError(f"{name}: probes.json and probes-greedy.json state other probes")
        entry = entries[name]
        answer, scores = entry["scores"]["answer"], entry["scores"]
        text, forced = reference_workloads.continuation(proc, name, sequence["token_ids"], answer)
        if sequence["answer"] != {"policy": "forced", "ids": forced}:
            raise ValueError(f"{name}: probes.json does not force the served form of the greedy answer")
        stopped = bool(answer) and answer[-1] in STOP_IDS
        pixels = b"".join((root / media["tensor"]["path"]).read_bytes() for media in sequence["media"])
        steps = {field: scores[field] for field in ("logprob", "top1", "margin")}
        steps |= {field: [row[:TOP] for row in scores[field]]
                  for field in ("topk_ids", "topk_logits", "topk_logprobs")}
        ids.update(answer)
        ids.update(forced)
        ids.update(i for row in steps["topk_ids"] for i in row)
        items[name] = {
            "kind": sequence["category"],
            "inputs": {"token_ids_sha256": sha256(json.dumps(sequence["token_ids"]).encode()),
                       "pixels_sha256": sha256(pixels), "images": [name],
                       "processors": [media["processor"] for media in sequence["media"]]},
            "prompt_token_ids": sequence["token_ids"], "score_from": sequence["score_from"],
            "answer": {"text": text, "token_ids": answer, "stopped": stopped},
            "answer_token_ids": forced,
            "generated": {"key_sha256": entry["key_sha256"], "answer": answer, **steps}}
    capture = root / "capture/probes.json"
    return {"schema": PROBE_SCHEMA,
            "provenance": provenance(workload, capture_sha256=sha256(capture.read_bytes()),
                                     generated={"run": str(Path(greedy).parent.name), "mode": "f32"}),
            "rule": RULE, "token_bytes": token_bytes(proc.tokenizer, ids), "probes": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=("teacher", "long-context", "probes", "teacher-texts", "nvfp4-teacher"))
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--greedy", type=Path, help="probes: the runner's f32.json of probes-greedy.json")
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_GLM5NEXT_CHECKPOINT"),
                        help="probes: checkpoint directory holding tokenizer.json")
    parser.add_argument("--pin", type=Path,
                        help="nvfp4-teacher: the checkpoint pin (`python3 -m tools.qual variant pin`)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.kind == "probes":
        if args.greedy is None or args.checkpoint is None:
            parser.error("probes needs --greedy and --checkpoint")
        value = probes(args.workload, args.greedy, args.checkpoint)
    elif args.kind == "teacher-texts":
        value = frozen_texts(args.workload)
    elif args.kind == "nvfp4-teacher":
        if args.pin is None or args.checkpoint is None:
            parser.error("nvfp4-teacher needs --pin and --checkpoint")
        described = nvfp4_variants(args.checkpoint)
        value = texts(args.workload, tuple(described), json.loads(args.pin.read_text()), described)
    else:
        value = texts(args.workload, LONG_CONTEXT_VARIANTS if args.kind == "long-context" else tuple(VARIANTS))
    with args.output.open("x") as stream:
        json.dump(value, stream)
        stream.write("\n")


if __name__ == "__main__":
    main()
