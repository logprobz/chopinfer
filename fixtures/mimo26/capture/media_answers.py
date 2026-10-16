"""Build the media capture and forced workload from refrun F32 greedy answers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.refrun import cache, keys
from runner_references import provenance


def write_capture(directory, reference, fixture_hashes):
    directory.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(reference, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = {"schema": "mimo26-multimodal-manifest/v1",
                "files": {"reference.json": hashlib.sha256(payload).hexdigest()}, "fixture_sha256": fixture_hashes}
    for name, data in (("reference.json", payload), ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())):
        with (directory / name).open("xb") as handle:
            handle.write(data)


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    source = json.loads((args.inputs / "greedy-workload.json").read_text())
    prompts = json.loads((args.inputs / "prompts.json").read_text())
    run = json.loads(args.run.read_text())
    if run["policy"]["name"] != "f32":
        raise ValueError("media answers must be generated in F32")
    generated = {entry["name"]: entry for entry in run["entries"] if entry["key"]["sequence"]["answer"]["policy"] == "greedy"}
    if set(generated) != {row["id"] for row in prompts}:
        raise ValueError("generation did not cover exactly the frozen media cases")
    rows = []
    for case, sequence in zip(prompts, source["sequences"], strict=True):
        entry = generated[case["id"]]
        ids = entry["scores"]["answer"]
        if not ids or ids[-1] not in sequence["answer"]["stop_ids"]:
            raise ValueError(f"{case['id']}: answer exhausted its generation budget")
        answer_ids = ids[:-1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=False)
        if not answer or tokenizer(answer, add_special_tokens=False)["input_ids"] != answer_ids:
            raise ValueError(f"{case['id']}: empty answer or token round-trip mismatch")
        rows.append({**case, "answer": answer, "answer_finish_reason": "stop",
                     "forced_prompt_token_ids": case["prompt_token_ids"] + answer_ids,
                     "generation_key_sha256": entry["key_sha256"]})
        sequence["answer"] = {"policy": "forced", "ids": answer_ids}
    processor = json.loads((args.inputs / "processor/processor.json").read_text())
    reference = {"schema": "mimo26-multimodal-reference/v1", "source": {
        "adapter": "models/mimo26/reference_answers.py", "mode": "f32", "answer_policy": "greedy",
        "max_new_tokens": 256, "fixed_video_sha256": processor["fixed_video_sha256"]}, "cases": rows}
    write_capture(args.inputs, reference, processor["fixture_sha256"])
    (args.inputs / "forced-workload.json").write_text(json.dumps(source, indent=1) + "\n")
    skeleton = {"schema": "chopinfer.mimo26.multimodal-exact-reference.v2",
        "capture_sha256": keys.file_sha256(args.inputs / "reference.json"),
        "provenance": provenance("tools/refrun F32 greedy media answers, forced in F32, BF16 and F64 after HF preprocessing"),
        "cases": [{"id": case["id"], "kind": case["kind"],
                   "answer_token_ids": case["forced_prompt_token_ids"][case["answer_token_start"]:]} for case in rows]}
    (args.inputs / "skeleton.json").write_text(json.dumps(skeleton, indent=1) + "\n")
    # Keep the generating run bound independently from the forced scorer.
    greedy = {"cases": [{"id": row["id"], "answer_token_ids": generated[row["id"]]["scores"]["answer"]} for row in rows]}
    greedy = cache.bind(greedy, {"f32": run}, args.inputs / "greedy-workload.json",
                        "models/mimo26/checkpoint-files.json", "mopd", {"f32": "exact"}, cache.Store(cache.default_root()))
    (args.inputs / "generation-reference.json").write_text(json.dumps(greedy, indent=1) + "\n")
    print(json.dumps({"cases": len(rows), "answer_tokens": {row["id"]: len(sequence["answer"]["ids"])
                                                            for row, sequence in zip(rows, source["sequences"], strict=True)}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "inputs", "run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
