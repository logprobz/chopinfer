"""Build DeepSeek V4.1 Flash's frozen needle prompts (the needle check's prompt set).

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -I fixtures/deepseek41/capture/needles.py [--check]

Runs on CPU with the checkpoint's tokenizer.json (through `tokenizers`) and its prompt-format reference `encoding/encoding.py`; no weights. The method is GLM's (fixtures/glm5next/capture/needles.py): the haystack and the three needles of MiMo's functional needle prompt (`$CHOPINFER_MIMO_FIXTURES/functional/context-524288.json`), decoded with MiMo's tokenizer, rebuilt in DeepSeek's prompt format at DeepSeek token lengths, with the needles in front of the paragraphs nearest 10%, 50% and 90% of the haystack. The prompt is rendered by `encode_messages` in chat mode (`thinking_mode="chat"`), which answers without reasoning, and each row leaves 1,024 tokens to answer.

Outputs:
- `$CHOPINFER_DATA/deepseek41/needle/prompts.json` (`--output`): every needle
  row with its token IDs, the needle check's prompt set (format prompt-set).
- `fixtures/deepseek41/needles.json` (`--receipt`): the SHA256 of that file,
  and each row's name, length, answer and token-ID SHA256.

`--check` rebuilds both in memory and refuses unless they equal the files.
Each prompt decodes to the text `encode_messages` renders for the same
system and user turns (checked for every length).
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "chopinfer.deepseek41.needles.v1"
CONTEXTS = (8192, 131072, 262144, 1048576)
OUTPUT_TOKENS = 1024
THINKING_MODE = "chat"
NEEDLES = (("ALPHA", "8D71A93C"), ("BRAVO", "46B2F8E0"), ("CHARLIE", "C5920AD7"))
SYSTEM = "Read the records carefully and answer exactly as requested."
QUESTION = "Return the values for ALPHA, BRAVO, and CHARLIE, in that order, separated only by commas."
# MiMo's user-turn header and the line its needles stand on.
MIMO_USER = "<|im_start|>user\n"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encoding(checkpoint):
    """The checkpoint's prompt-format reference module, loaded from its file."""
    spec = importlib.util.spec_from_file_location("deepseek41_encoding", Path(checkpoint) / "encoding/encoding.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def haystack(mimo_tokenizer, fixtures):
    """The functional prompt's haystack paragraphs, without its needles."""
    frozen = json.loads((Path(fixtures) / "functional/context-524288.json").read_text())
    text = mimo_tokenizer.decode(frozen["prompt"], skip_special_tokens=False)
    body = text[text.index(MIMO_USER) + len(MIMO_USER):text.index("\n" + QUESTION)]
    for key, value in NEEDLES:
        body = body.replace(f"{key} is {value}.\n", "")
    return [part + "\n" for part in body.split("\n") if part]


def render(module, user):
    """The reference's text for the system and user turns, generation prompt included."""
    return module.encode_messages([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                  thinking_mode=THINKING_MODE)


def needle_rows(tokenizer, module, paragraphs):
    marker = "\x00"
    head, tail = render(module, marker).split(marker)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    prefix, suffix = encode(head), encode("\n" + QUESTION + tail)
    paragraph_ids = [encode(part) for part in paragraphs]
    needles = [encode(f"{key} is {value}.\n") for key, value in NEEDLES]
    rows = []
    for context in CONTEXTS:
        budget = context - OUTPUT_TOKENS - len(prefix) - len(suffix) - sum(map(len, needles))
        filler, starts = [], []
        while len(filler) < budget:
            filler.extend(paragraph_ids[len(starts) % len(paragraph_ids)])
            starts.append(len(filler))
        filler = filler[:budget]
        cuts = [min(starts, key=lambda s: abs(s - depth * budget)) for depth in (0.1, 0.5, 0.9)]
        prompt, last = list(prefix), 0
        for cut, needle in zip(cuts, needles):
            prompt += filler[last:cut] + needle
            last = cut
        prompt += filler[last:] + suffix
        if len(prompt) != context - OUTPUT_TOKENS:
            raise ValueError(f"needle_{context // 1024}k: {len(prompt)} prompt tokens")
        user = tokenizer.decode(prompt[len(prefix):len(prompt) - len(suffix)], skip_special_tokens=False)
        if tokenizer.decode(prompt, skip_special_tokens=False) != render(module, user + "\n" + QUESTION):
            raise ValueError(f"needle_{context // 1024}k does not decode to the reference's render")
        rows.append({"name": f"needle_{context // 1024}k", "total_context": context, "max_tokens": OUTPUT_TOKENS,
                     "expected": ",".join(value for _, value in NEEDLES), "needle_offsets": cuts,
                     "token_ids": prompt})
    return rows


def build(checkpoint, mimo_checkpoint, fixtures):
    checkpoint = Path(checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    mimo = Tokenizer.from_file(str(Path(mimo_checkpoint) / "tokenizer.json"))
    rows = needle_rows(tokenizer, encoding(checkpoint), haystack(mimo, fixtures))
    blob = (json.dumps({"schema": SCHEMA, "needles": rows}, separators=(",", ":")) + "\n").encode()
    receipt = {
        "schema": SCHEMA,
        "provenance": __doc__.split("\n\n")[2].replace("\n", " "),
        "tokenizer_json_sha256": sha256((checkpoint / "tokenizer.json").read_bytes()),
        "encoding_sha256": sha256((checkpoint / "encoding/encoding.py").read_bytes()),
        "needle_source_sha256": sha256((Path(fixtures) / "functional/context-524288.json").read_bytes()),
        "full_file": {"path": "${CHOPINFER_DATA}/deepseek41/needle/prompts.json", "sha256": sha256(blob),
                      "bytes": len(blob)},
        "needles": [{key: value for key, value in row.items() if key != "token_ids"}
                    | {"token_count": len(row["token_ids"]),
                       "token_ids_sha256": sha256(json.dumps(row["token_ids"]).encode())} for row in rows],
    }
    return blob, (json.dumps(receipt, indent=1) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT"))
    parser.add_argument("--mimo-checkpoint", default=os.environ.get("CHOPINFER_MIMO_CHECKPOINT"))
    parser.add_argument("--mimo-fixtures", default=os.environ.get("CHOPINFER_MIMO_FIXTURES"))
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("CHOPINFER_DATA", "")) / "deepseek41/needle/prompts.json")
    parser.add_argument("--receipt", type=Path, default=REPO / "fixtures/deepseek41/needles.json")
    parser.add_argument("--check", action="store_true", help="refuse unless both files are current")
    args = parser.parse_args()
    if not (args.checkpoint and args.mimo_checkpoint and args.mimo_fixtures):
        parser.error("set CHOPINFER_DEEPSEEK_CHECKPOINT, CHOPINFER_MIMO_CHECKPOINT and CHOPINFER_MIMO_FIXTURES")
    blob, receipt = build(args.checkpoint, args.mimo_checkpoint, args.mimo_fixtures)
    if args.check:
        stale = [str(path) for path, data in ((args.output, blob), (args.receipt, receipt))
                 if not path.is_file() or path.read_bytes() != data]
        if stale:
            raise SystemExit("stale: " + ", ".join(stale))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    args.receipt.write_bytes(receipt)
    print(json.dumps({"prompts": str(args.output), "sha256": sha256(blob),
                      "needles": [len(json.loads(blob)["needles"][i]["token_ids"]) for i in range(len(CONTEXTS))]}))


if __name__ == "__main__":
    main()
