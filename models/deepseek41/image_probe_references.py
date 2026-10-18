"""The skeleton of DeepSeek V4.1 Flash's image-probe reference, filled by the
runner's emit.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/deepseek41/image_probe_references.py \\
        --workload W/probes-served.json --greedy RUN/f32.json --capture W/capture/probes.json \\
        --output S.json

then `python3 -m tools.refrun emit --skeleton S.json --runs DIR... --workload W/probes-served.json
--lock models/deepseek41/checkpoint-files.json --modes f32=exact,bf16,f64 --output REF.json`
(tools/refrun/README.md). The skeleton holds everything the media gate
(`tools/qual/multimodal.py`, format `image-probes`) reads except the scores
emit copies: the gate variant `exact` is the F32-TQ run, and `bf16` and
`f64` are the spread variants whose distance from it sets each probe's
limits (`RULE`, tools/qual/limits.py).

One free-text probe per sequence of reference_workloads.py's
`probes-served.json`, which forces each answer as a served chat continues it
(the greedy answer without its end-of-sentence token, `answer_token_ids`),
with the answer the F32-TQ run generated (`--greedy`, the runner's f32.json
of `probes-greedy.json`): its text, IDs and whether it stopped, and its
steps (top 5) under `generated` with their entry key, which no gate uses.
`token_bytes` holds the bytes of every token an answer or a step's top 5
names, since a served answer is compared by its token bytes. `--capture` is
the request capture (`capture/probes.json`) the gate sends.
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

GATE = ROOT / "models/deepseek41/qual.json"
SCHEMA = "chopinfer.deepseek41.image-probe-references.v1"
TOP = 5
STOP_IDS = (1,)
CODE = ("the checkpoint's inference code (models/deepseek41/reference.py, kernels replaced as "
        "reference_kernel.LEDGER records)")
VARIANTS = {
    "exact": f"{CODE} in F32 with quantizer inputs rounded to BF16 (F32-TQ) on the GPU runner (tools/refrun): "
             "the gate variant",
    "bf16": "the same in BF16, the vendor's own numerics (BF16 default dtype, F32 head): spread variant",
    "f64": "the same in F64 with quantizer inputs rounded to BF16 (F64-TQ), the widened vendor copy except its "
           "RoPE angle lines: spread variant",
}
POSITIONS = ("entry i scores the token at score_from + i from the logits of the row before it; top1 is the "
             "highest logit (ties to the lower ID), margin its logit minus the second's, and the top-k columns "
             "are ranked by logit")
RULE = {"variants": ["bf16", "f64"], "mean_step": 0.05, "mean_floor": 0.15, "top1_floor": 0.08,
        "near_tie": 0.3,
        "text": "free text: the F32-TQ answer teacher-forced as a served chat continues it, against the BF16 and "
                "F64-TQ runs scored like a server: mean |dlogprob| <= max(mean_floor, the worst variant rounded "
                "up to mean_step). A top-1 that differs from the F32-TQ pass's is a near tie when the F32-TQ "
                "logprob gap to it, in its stored top-k, is at most near_tie nats, and decisive otherwise; over "
                "the positions, decisive flips <= max(top1_floor, the worst variant's rate), near ties <= "
                "max(0.08, the worst variant's rate rounded up to 0.01) and all flips <= max(0.10, the worst "
                "variant's rate)"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def byte_decoder():
    """The byte-level BPE alphabet, inverted: token characters to bytes."""
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
    decoder = byte_decoder()
    added = {token.content for token in tokenizer.get_added_tokens_decoder().values()}
    result = {}
    for token in sorted(set(ids)):
        text = tokenizer.id_to_token(token)
        result[str(token)] = list(text.encode() if text in added else bytes(decoder[c] for c in text))
    return result


def provenance(workload, **extra):
    pin = checkpoint_pin(read_config(str(GATE)))
    return {"source": "tools/refrun emit of models/deepseek41/reference.py runs "
                      "(models/deepseek41/image_probe_references.py)",
            "checkpoint": {key: pin[key] for key in ("repository", "revision", "served_model_name",
                                                     "source_lock_sha256", "checkpoint_identity")},
            "workload": Path(workload).name, "gate_variant": "exact", "variants": VARIANTS,
            "positions": POSITIONS, **extra}


def probes(workload, greedy, capture, checkpoint):
    """The image-probe skeleton over `workload` (probes-served.json) and the
    F32-TQ greedy run of probes-greedy.json."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    run = json.loads(Path(greedy).read_text())
    if run["policy"]["name"] != "f32":
        raise ValueError(f"{greedy} is not an f32 run")
    generated = {e["name"]: e for e in run["entries"] if e["key"]["sequence"]["answer"]["policy"] == "greedy"}
    images = {probe["name"]: probe["images"] for probe in json.loads(Path(capture).read_text())}
    root, items, ids = Path(workload).parent, {}, set()
    for sequence in json.loads(Path(workload).read_text())["sequences"]:
        name, entry = sequence["name"], generated[sequence["name"]]
        if entry["key"]["sequence"]["token_ids_sha256"] != keys.ids_sha256(sequence["token_ids"]):
            raise ValueError(f"{name}: the greedy run scored another prompt")
        if sequence["answer"]["policy"] != "forced":
            raise ValueError(f"{name}: not a forced probe")
        answer, scores = entry["scores"]["answer"], entry["scores"]
        stopped = bool(answer) and answer[-1] in STOP_IDS
        if not stopped or sequence["answer"]["ids"] != answer[:-1]:
            raise ValueError(f"{name}: the probe does not force the greedy answer as a served chat continues it")
        text = tokenizer.decode(answer[:-1], skip_special_tokens=False)
        if tokenizer.encode(text, add_special_tokens=False).ids != answer[:-1]:
            raise ValueError(f"{name}: the answer text does not encode to the answer")
        pixels = b"".join((root / media["tensor"]["path"]).read_bytes() for media in sequence["media"])
        steps = {field: scores[field] for field in ("logprob", "top1", "margin")}
        steps |= {field: [row[:TOP] for row in scores[field]]
                  for field in ("topk_ids", "topk_logits", "topk_logprobs")}
        ids.update(answer)
        ids.update(i for row in steps["topk_ids"] for i in row)
        items[name] = {
            "kind": sequence["category"],
            "inputs": {"token_ids_sha256": sha256(json.dumps(sequence["token_ids"]).encode()),
                       "pixels_sha256": sha256(pixels), "images": len(images[name])},
            "prompt_token_ids": sequence["token_ids"], "score_from": sequence["score_from"],
            "answer": {"text": text, "token_ids": answer, "stopped": stopped},
            "answer_token_ids": sequence["answer"]["ids"],
            "generated": {"key_sha256": entry["key_sha256"], "answer": answer, **steps}}
    return {"schema": SCHEMA,
            "provenance": provenance(workload, capture_sha256=sha256(Path(capture).read_bytes()),
                                     generated={"run": Path(greedy).parent.name, "mode": "f32"}),
            "rule": RULE, "token_bytes": token_bytes(tokenizer, ids), "probes": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", required=True, type=Path, help="reference_workloads.py's probes-served.json")
    parser.add_argument("--greedy", required=True, type=Path, help="the runner's f32.json of probes-greedy.json")
    parser.add_argument("--capture", required=True, type=Path, help="the request capture, capture/probes.json")
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT"),
                        help="checkpoint directory holding tokenizer.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("set CHOPINFER_DEEPSEEK_CHECKPOINT or --checkpoint")
    value = probes(args.workload, args.greedy, args.capture, args.checkpoint)
    with args.output.open("x") as stream:
        json.dump(value, stream)
        stream.write("\n")


if __name__ == "__main__":
    main()
