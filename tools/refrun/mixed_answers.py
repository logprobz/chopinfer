"""Predeclare, generate and bind full-answer mixed-media probes.

Preparation reuses the authenticated media workload, changing only the user
instruction. Run generation-workload.json with reference_answers.py in F32,
then force-workload.json with reference.py in F32/BF16/F64, through the queue.
The entire generated text is scored; the terminal stop token is recorded but
excluded from assistant text. No short answer or truncated answer is accepted.
"""
import argparse
import copy
import json
from pathlib import Path

from tools.model_sdk.diagnostic import canonical
from tools.refrun import cache, emit, keys

CASES = ("single-image", "two-images", "two-resized-images", "image-video")
MIN_POSITIONS = 32
MAX_TOKENS = 256
PROMPT = ("Write one paragraph of four sentences, using 60 to 100 words, about the supplied media. "
          "Describe its visible contents and any audible speech, noting similarities or changes when present. "
          "Do not enumerate or invent additional items.")


def write(path, value):
    path.write_bytes(canonical(value) + b"\n")


def protocol(stop_ids):
    return {"prompt": PROMPT, "minimum_scored_positions": MIN_POSITIONS,
            "generation": {"policy": "greedy", "max_new_tokens": MAX_TOKENS, "stop_ids": stop_ids},
            "mode": "f32", "dequant": "rounded", "score": "entire text excluding terminal stop token"}


def prepare(args, tokenizer):
    args.output.mkdir(parents=True, exist_ok=False)
    emit.copy_workload(args.inputs / "encoder-workload.json", args.output / "encoder-workload.json")
    source = json.loads((args.output / "encoder-workload.json").read_text())
    stop_ids = json.loads((args.checkpoint / "generation_config.json").read_text())["eos_token_id"]
    spec = protocol(stop_ids)
    sequences = []
    (args.output / "bodies").mkdir()
    for sequence in source["sequences"]:
        if sequence["name"] not in CASES:
            continue
        body = json.loads((args.inputs / "bodies" / (sequence["name"] + ".json")).read_text())
        old = body["messages"][0]["content"][0]["text"]
        text = tokenizer.decode(sequence["token_ids"], skip_special_tokens=False)
        if tokenizer.encode(text, add_special_tokens=False).ids != sequence["token_ids"] or text.count(old) != 1:
            raise ValueError("original prompt does not round-trip uniquely")
        text = text.replace(old, PROMPT + " ", 1)
        sequence["token_ids"] = tokenizer.encode(text, add_special_tokens=False).ids
        sequence["score_from"] = len(sequence["token_ids"])
        sequence["answer"] = spec["generation"]
        sequences.append(sequence)
        body["messages"][0]["content"][0]["text"] = PROMPT + " "
        write(args.output / "bodies" / (sequence["name"] + ".json"), body)
    if tuple(s["name"] for s in sequences) != CASES:
        raise ValueError("mixed generation case set or order differs")
    write(args.output / "generation-workload.json", {**source, "sequences": sequences})
    write(args.output / "protocol.json", {**spec, "producer": keys.repo_files(("tools/refrun/mixed_answers.py",)),
          "source_workload_sha256": cache.workload_sha256(args.inputs / "encoder-workload.json"),
          "generation_config_sha256": keys.file_sha256(args.checkpoint / "generation_config.json")})


def full_answer(entry, spec, tokenizer):
    ids = entry["scores"]["answer"]
    if not ids or ids[-1] not in spec["generation"]["stop_ids"]:
        raise ValueError("reference answer is truncated or has no terminal stop")
    text_ids = ids[:-1]
    if len(text_ids) < spec["minimum_scored_positions"]:
        raise ValueError("reference answer has fewer than 32 scored positions")
    if any(token in spec["generation"]["stop_ids"] for token in text_ids):
        raise ValueError("reference answer contains an earlier stop")
    text = tokenizer.decode(text_ids, skip_special_tokens=False)
    if tokenizer.encode(text, add_special_tokens=False).ids != text_ids:
        raise ValueError("full reference answer does not round-trip")
    return {"answer": text, "answer_token_ids": text_ids, "terminal_stop_id": ids[-1],
            "generation_key_sha256": entry["key_sha256"]}


def force(args, tokenizer):
    spec = json.loads((args.output / "protocol.json").read_text())
    if spec["producer"] != keys.repo_files(("tools/refrun/mixed_answers.py",)):
        raise ValueError("generation protocol producer changed")
    source = json.loads((args.output / "generation-workload.json").read_text())
    runs = emit.read_runs([args.output / "generation"])
    generated = {entry["name"]: entry for entry in runs["f32"]["entries"]
                 if entry["key"]["sequence"]["answer"]["policy"] == "greedy"}
    _, sequences = cache.workload_sequences(args.output / "generation-workload.json")
    ordered = {"f32": emit.ordered(runs["f32"], sequences, "f32")}
    skeleton = {"cases": [{"id": name} for name in CASES]}
    capture = emit.assemble(skeleton, ordered, {"f32": "exact"})
    capture = cache.bind(capture, ordered, str(args.output / "generation-workload.json"),
                         f"models/{source['model']}/checkpoint-files.json", "mopd", {"f32": "exact"},
                         cache.Store(cache.default_root()))
    write(args.output / "generation-reference.json", capture)
    answers = []
    for sequence in source["sequences"]:
        entry = generated[sequence["name"]]
        if entry["key"]["mode"]["name"] != "f32":
            raise ValueError("answers must be generated in F32")
        answer = full_answer(entry, spec, tokenizer)
        sequence["answer"] = {"policy": "forced", "ids": answer["answer_token_ids"]}
        answers.append({"id": sequence["name"], "prompt_token_ids": sequence["token_ids"],
                        "score_from": sequence["score_from"], **answer})
    write(args.output / "force-workload.json", source)
    # Keep the independently scored 32-image encoder-capacity case in the binding.
    encoder = json.loads((args.output / "encoder-workload.json").read_text())
    maximum = next(s for s in encoder["sequences"] if s["name"] == "32-images")
    write(args.output / "combined-workload.json", {**source, "sequences": [*source["sequences"], maximum]})
    write(args.output / "answers.json", {"protocol": spec, "cases": answers})
    print(json.dumps({case["id"]: len(case["answer_token_ids"]) for case in answers}))


def bind(args, tokenizer):
    del tokenizer
    recorded = json.loads((args.output / "answers.json").read_text())
    old = json.loads(args.previous.read_text())
    maximum = copy.deepcopy(next(c for c in old["cases"] if c["id"] == "32-images"))
    for variant in ("exact", "bf16", "f64"):
        maximum.pop(variant)
    source = json.loads((args.output / "combined-workload.json").read_text())
    skeleton = {"schema": old["schema"], "provenance": {**old["provenance"], "generation": recorded["protocol"],
                "generation_reference_sha256": keys.file_sha256(args.output / "generation-reference.json")},
                "cases": [*recorded["cases"], maximum]}
    source = args.output / "combined-workload.json"
    _, sequences = cache.workload_sequences(source)
    found = emit.read_runs([args.output / "forced", args.maximum_runs])
    runs = {mode: emit.ordered(found[mode], sequences, mode) for mode in ("f32", "bf16", "f64")}
    variants = {"f32": "exact", "bf16": "bf16", "f64": "f64"}
    value = emit.assemble(skeleton, runs, variants)
    value = cache.bind(value, runs, str(source), f"models/{json.loads(source.read_text())['model']}/checkpoint-files.json", "mopd", variants,
                       cache.Store(cache.default_root()))
    args.fixture.write_text(json.dumps(value) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "force", "bind"))
    for name in ("checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("inputs", "previous", "maximum-runs", "fixture"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    {"prepare": prepare, "force": force, "bind": bind}[args.action](args, tokenizer)


if __name__ == "__main__":
    main()
