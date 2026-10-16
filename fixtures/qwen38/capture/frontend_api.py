"""Prepare HF F32 greedy API-answer workloads and expected response shapes.

The tool-result history is a fixed serving-surface input. It is never used
as an expected generated answer. Error and stream shapes are transport
contracts; successful non-stream answer shapes come only from refrun.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.refrun.frontend import messages_for_hf
from tools.refrun import cache, keys


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    inputs = json.loads(args.inputs.read_text())
    stop_ids = json.loads((args.checkpoint / "generation_config.json").read_text())["eos_token_id"]
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    sequences, prompts = [], []
    for case in inputs:
        body = case["request"]
        if "messages" in body:
            options = {**body.get("chat_template_kwargs", {}), "tools": body.get("tools"), "add_generation_prompt": True}
            if "reasoning_effort" in body:
                options["reasoning_effort"] = body["reasoning_effort"]
            ids = tokenizer.apply_chat_template(messages_for_hf(body["messages"], tool_reasoning=True), tokenize=True, **options)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
        else:
            ids = tokenizer(body["prompt"], add_special_tokens=False)["input_ids"]
        prompts.append({**case, "token_ids": ids, "prompt_tokens": len(ids)})
        sequences.append({"name": case["name"], "category": "api", "token_ids": ids, "score_from": len(ids),
            "answer": {"policy": "greedy", "max_new_tokens": body["max_tokens"], "stop_ids": stop_ids}})
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "api-prompts.json", prompts)
    write(args.output / "workload.json", {"schema": "chopinfer.refrun.workload.v1", "model": "qwen38", "sequences": sequences})
    print(f"prepared {len(sequences)} API answer cases")


def shapes(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompts = json.loads((args.output / "api-prompts.json").read_text())
    run = json.loads(args.run.read_text())
    answers = {row["name"]: row for row in run["entries"] if row["key"]["sequence"]["answer"]["policy"] == "greedy"}
    if set(answers) != {row["name"] for row in prompts}:
        raise ValueError("the F32 run does not cover exactly the API answer cases")
    result = {}
    for case in prompts:
        entry = answers[case["name"]]
        if entry["key"]["mode"]["name"] != "f32":
            raise ValueError("API shapes require F32")
        ids = entry["scores"]["answer"]
        stop = entry["key"]["sequence"]["answer"]["stop_ids"]
        ended = bool(ids) and ids[-1] in stop
        text = tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False)
        body = case["request"]
        reasoning = ""
        content = text
        if "messages" in body and body.get("chat_template_kwargs", {}).get("enable_thinking", True):
            reasoning, separator, content = text.partition("</think>")
            if not separator:
                content = ""
        calls = re.findall(r"<tool_call>\s*<function=([^>]+)>.*?</function>\s*</tool_call>", content, re.S)
        if calls:
            content = content.split("<tool_call>", 1)[0].strip()
        result[case["name"]] = {"status": 200, "finish_reason": "tool_calls" if calls else ("stop" if ended else "length"),
            "content": bool(content), "reasoning": bool(reasoning), "tool_calls": calls}
    result.update({name: {"status": 400} for name in ("chat-effort-high", "error-video", "error-five-images", "error-too-long")})
    result.update({name: {"status": 200} for name in ("tool-call-stream", "chat-stream")})
    if len(result) != 15:
        raise ValueError("expected 15 API cases")
    write(args.output / "api-shapes.json", result)
    bound = cache.bind({"cases": [{"id": name, "answer_token_ids": row["scores"]["answer"]}
                                  for name, row in answers.items()]}, {"f32": run},
                       args.output / "workload.json", "models/qwen38/checkpoint-files.json",
                       variants={"f32": "exact"}, store=cache.Store(cache.default_root()))
    write(args.output / "api-generation-reference.json", bound)
    write(args.output / "api-shapes-metadata.json", {
        "source": "tools/refrun F32 greedy; HF templates and tokenizer",
        "answer_cases": len(answers), "surface_cases": 6,
        "shapes_sha256": keys.file_sha256(args.output / "api-shapes.json"),
        "generation_reference_sha256": keys.file_sha256(args.output / "api-generation-reference.json"),
        "completion-n2-seeded": "shape from F32 greedy at the same max_tokens; sampling parameters exercised by the server only"})
    print(f"derived {len(result)} API shapes from {len(answers)} F32 answers and six surface contracts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "shapes"))
    for name in ("checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else shapes)(args)


if __name__ == "__main__":
    main()
