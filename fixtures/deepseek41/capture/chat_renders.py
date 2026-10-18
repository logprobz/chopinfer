"""Regenerate chat renders on CPU with the pinned encoder and tokenizer."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/qual")]

from fixtures.mimo26.capture.conversations import conversations  # noqa: E402
from tools.qual.frontend_goldens import capture_bodies, wire_message  # noqa: E402
from tools.refrun.frontend_corpus import Model, model_reference, template_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CHOPINFER_MIMO_FIXTURES", str(ROOT / "fixtures/mimo26"))
    model = Model("deepseek41", root=ROOT)
    os.environ["CHOPINFER_QUAL_CONFIG"] = str(model.directory / "qual.json")
    reference = model_reference(model, template_source(model)[0])
    rows = []
    requests = list(conversations())
    bodies, _ = capture_bodies(model.declaration["models"][0]["id"], requests, [])
    for request, body in zip(requests, bodies):
        request = {**request, "messages": [wire_message(m) for m in request["messages"]]}
        result = reference.outcome(model.hf_input(body))
        if "refusal" in result:
            raise ValueError(f"{request['id']}: {result['refusal']}")
        rows.append({**request, **{key: result[key] for key in ("rendered_text", "token_ids")}})
    data = (json.dumps(rows, ensure_ascii=False, indent=1) + "\n").encode()
    if args.check:
        if data != (ROOT / "fixtures/deepseek41/frontend/chat-renders.json").read_bytes():
            raise ValueError("chat-renders.json differs from the pinned CPU encoder output")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    print(f"{len(rows)} chat renders: {'verified' if args.check else 'written'}")


if __name__ == "__main__":
    main()
