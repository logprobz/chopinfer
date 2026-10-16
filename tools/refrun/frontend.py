"""Regenerate frontend goldens from local HF templates, tokenizers and processors.

The input JSON supplies cases only. Expected text and IDs are always recomputed.
API reasoning and string tool arguments are translated to HF template inputs;
this preserves the declared serving transport without using server rendering.
No model weights are loaded and no network access is needed.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def messages_for_hf(messages, *, tool_reasoning=False, images=None):
    messages = copy.deepcopy(messages)
    for message in messages:
        reasoning = message.pop("reasoning", None)
        fallback = message.pop("reasoning_content", None)
        if reasoning is None and tool_reasoning and message.get("tool_calls"):
            reasoning = fallback
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part["type"] == "image_url":
                    digest = part.pop("image_url")["url"].removeprefix("sha256:")
                    part.update(type="image", image=str(images[digest]))
    return messages


def generate(args):
    import torch
    import transformers
    from jinja2 import TemplateError
    from transformers import AutoProcessor, AutoTokenizer

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    paths = [path for directory in args.images for path in directory.rglob("*.png")]
    images = {hashlib.sha256(path.read_bytes()).hexdigest(): path for path in paths}
    by_name = {path.stem: path for path in paths}
    outputs = {}
    rows = json.loads((args.inputs / "chat-renders.json").read_text())
    renders = []
    for case in rows:
        row = {key: case[key] for key in ("id", "category", "messages", "tools", "chat_template_kwargs") if key in case}
        messages = messages_for_hf(case["messages"], tool_reasoning=args.tool_reasoning, images=images)
        options = {**case.get("chat_template_kwargs", {}), "tools": case.get("tools"), "add_generation_prompt": True}
        media = any(isinstance(message.get("content"), list) and
                    any(part["type"] == "image" for part in message["content"]) for message in messages)
        try:
            if media:
                ids = processor.apply_chat_template(messages, tokenize=True, return_dict=True, **options)["input_ids"][0]
            else:
                ids = tokenizer.apply_chat_template(messages, tokenize=True, **options)
                if not isinstance(ids, list):
                    ids = ids["input_ids"]
            row.update(token_ids=ids, rendered_text=tokenizer.decode(ids, skip_special_tokens=False))
            if "status" in case:
                row["status"] = 200
        except TemplateError as error:
            if case.get("status", 200) == 200:
                raise
            row.update(status=400, error=str(error))
        renders.append(row)
    outputs["chat-renders.json"] = renders
    outputs["tokenizer-golden.json"] = [
        {"text": row["text"], "token_ids": tokenizer(row["text"], add_special_tokens=False)["input_ids"]}
        for row in json.loads((args.inputs / "tokenizer-golden.json").read_text())]
    if (args.inputs / "images.json").exists():
        probes = []
        for case in json.loads((args.inputs / "images.json").read_text()):
            parts = [{"type": "image", "image": str(by_name[name])} for name in case["images"]]
            messages = [{"role": "user", "content": [*parts, {"type": "text", "text": case["text"]}]}]
            ids = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                add_generation_prompt=True, enable_thinking=False)["input_ids"][0]
            row = {key: case[key] for key in ("name", "images", "text")}
            row.update(token_ids=ids, image_pad_tokens=ids.count(tokenizer.convert_tokens_to_ids(processor.image_token)))
            probes.append(row)
        outputs["images.json"] = probes
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
        print(f"{name}: {len(value)} cases")
    provenance = {"source": "HF transformers templates, tokenizers and processors", "transformers": transformers.__version__,
                  "api_reasoning_field": "reasoning", "tool_call_reasoning_content_fallback": args.tool_reasoning,
                  "checkpoint_files": {name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
                                       for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")},
                  "images": {path.stem: digest for digest, path in images.items()}}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "inputs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--images", type=Path, action="append", default=[])
    parser.add_argument("--tool-reasoning", action="store_true", help="the API accepts reasoning_content on tool-call turns")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
