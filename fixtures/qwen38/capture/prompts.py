"""Build the frozen Qwen3.8 prompt set: teacher texts, long texts and needles.

Runs on the host with the HF environment (see prompts.sh). Token IDs come from
the checkpoint's tokenizer.json through `tokenizers`, and chat texts are
rendered with the checkpoint's chat template through transformers, so the
prompts exist before any server does. The reference workload sends these
token IDs directly to the HF model adapter.

Outputs:
- `<refs>/prompts/prompts.json`: every prompt with its token IDs.
- `fixtures/qwen38/prompts.json`: the short texts in full, and the name, token
  count and SHA256 of every long prompt, plus the SHA256 of the full file.

Sources:
- Short teacher texts: `fixtures/mimo26/teacher-texts.json` without its MiMo
  chat and long-context texts, plus Qwen chat conversations rendered here.
- Long texts: cleaned wikitext-103 test paragraphs from the pinned evals cache
  (`${CHOPINFER_DATA}/evals/wikitext.json`, SHA256 in its manifest), in order
  from a seeded start. The last `score_last` tokens quote a passage from the
  first tenth of the text again, so those positions can copy across the whole
  context. Beyond 2051 visible tokens QSA attends to selected blocks, so the
  quote is scored only where attention is sparse.
- Needles: the haystack and the three needles of the MiMo functional needle
  prompts (`${CHOPINFER_MIMO_FIXTURES}/functional/context-524288.json`),
  decoded with the MiMo tokenizer and rebuilt in Qwen's chat format at Qwen
  token lengths, needles at 10%, 50% and 90% depth.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/qual"))
from evals import paragraphs  # noqa: E402

SCHEMA = "chopinfer.qwen38.prompts.v1"
SEED = 20260924
LONG_LENGTHS = (4096, 32768, 131072)
SCORE_LAST = {4096: 512, 32768: 1024, 131072: 1024}
NEEDLE_CONTEXTS = (8192, 131072, 262144)
NEEDLE_OUTPUT = 128
NEEDLES = (("ALPHA", "8D71A93C"), ("BRAVO", "46B2F8E0"), ("CHARLIE", "C5920AD7"))
NEEDLE_SYSTEM = "Read the records carefully and answer exactly as requested."
NEEDLE_QUESTION = "Return the values for ALPHA, BRAVO, and CHARLIE, in that order, separated only by commas."

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}

# Qwen chat texts, the counterparts of MiMo's chat teacher texts. Rendered
# with add_generation_prompt false; assistant reasoning uses the template's
# reasoning_content field.
CHATS = {
    "chat_system": {"messages": [
        {"role": "system", "content": "You are a concise assistant. Answer in at most two sentences."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "reasoning_content": "", "content":
            "Sunlight is scattered by air molecules, and shorter blue wavelengths scatter much more strongly "
            "than red ones. That scattered blue light reaches our eyes from every direction of the sky."}],
        "kwargs": {"enable_thinking": False}},
    "chat_thinking": {"messages": [
        {"role": "user", "content": "Is 91 a prime number?"},
        {"role": "assistant", "reasoning_content":
            "I need to check small divisors. 91 is odd, so 2 does not divide it. The digit sum is 10, so 3 "
            "does not divide it. It does not end in 0 or 5. 7 * 13 = 91, so 91 is composite.",
         "content": "No. 91 = 7 * 13, so it is not prime."}]},
    "chat_effort_low": {"messages": [
        {"role": "user", "content": "Name three prime numbers greater than 50."},
        {"role": "assistant", "reasoning_content": "53, 59 and 61 are prime.",
         "content": "53, 59 and 61."}],
        "kwargs": {"reasoning_effort": "low"}},
    "chat_multi_turn": {"messages": [
        {"role": "user", "content": "Give me a name for a gray cat."},
        {"role": "assistant", "reasoning_content": "",
         "content": "How about Pewter? It is a soft gray metal, which suits a gray cat."},
        {"role": "user", "content": "Something shorter, please."},
        {"role": "assistant", "reasoning_content": "",
         "content": "Try Ash, Smoke or Slate. All three are short and gray."}],
        "kwargs": {"enable_thinking": False}},
    "chat_zh": {"messages": [
        {"role": "user", "content": "请用一句话解释什么是机器学习。"},
        {"role": "assistant", "reasoning_content": "",
         "content": "机器学习是让计算机从数据中自动发现规律，并利用这些规律对新数据做出预测或决策的方法。"},
        {"role": "user", "content": "再举一个生活中的例子。"},
        {"role": "assistant", "reasoning_content": "",
         "content": "例如，邮箱会根据你过去标记的邮件，自动把新的垃圾邮件分到垃圾箱里。"}],
        "kwargs": {"enable_thinking": False}},
    "chat_tool_call": {"messages": [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "reasoning_content": "The user wants the current weather, so call get_weather.",
         "content": "", "tool_calls": [{"type": "function", "function": {
             "name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "{\"temperature\": 18, \"condition\": \"cloudy\"}"},
        {"role": "assistant", "reasoning_content": "The tool says 18 degrees and cloudy.",
         "content": "It is 18 degrees Celsius and cloudy in Paris."}],
        "tools": [WEATHER]},
    "chat_document_boundary": {"text":
        "First document: the harbor opens at dawn and closes at dusk.<|endoftext|>"
        "Second document: a new record begins here, and its n-gram history starts over."},
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def teacher_texts(tokenizer, chat_tokenizer):
    source = json.loads((REPO / "fixtures/mimo26/teacher-texts.json").read_text())
    rows = []
    for item in source["texts"]:
        if item["category"] in ("chat", "long-context"):
            continue
        rows.append({"name": item["name"], "category": item["category"], "text": item["text"],
                     "source": "fixtures/mimo26/teacher-texts.json"})
    for name, spec in CHATS.items():
        if "text" in spec:
            text = spec["text"]
        else:
            text = chat_tokenizer.apply_chat_template(
                spec["messages"], tools=spec.get("tools"), tokenize=False, add_generation_prompt=False,
                **spec.get("kwargs", {}))
        rows.append({"name": name, "category": "chat", "text": text, "source": "qwen38 chat template"})
    for row in rows:
        row["token_ids"] = tokenizer.encode(row["text"], add_special_tokens=False).ids
    return rows


def long_texts(tokenizer, corpus):
    rng = random.Random(f"{SEED}:qwen38-long")
    start = rng.randrange(len(corpus))
    ordered = corpus[start:] + corpus[:start]
    ids = []
    for paragraph in ordered:
        ids.extend(tokenizer.encode(paragraph + "\n\n", add_special_tokens=False).ids)
        if len(ids) >= max(LONG_LENGTHS):
            break
    if len(ids) < max(LONG_LENGTHS):
        raise ValueError("the wikitext corpus is too short for the longest text")
    rows = []
    intro = tokenizer.encode("\n\nThe following passage appeared earlier in this text:\n\n",
                             add_special_tokens=False).ids
    for length in LONG_LENGTHS:
        score_last = SCORE_LAST[length]
        body = ids[:length - score_last - len(intro)]
        # The quoted passage starts at the first paragraph break after 5% of
        # the text, so it is far behind the scored positions.
        start = next(i for i in range(len(body) // 20, len(body)) if tokenizer.decode(body[i - 1:i]).endswith("\n"))
        quote = body[start:start + score_last]
        prompt = body + intro + quote
        assert len(prompt) == length
        rows.append({"name": f"long_{length // 1024}k", "tokens": length, "score_last": score_last,
                     "quote_source_start": start, "token_ids": prompt})
    return rows


def needle_prompts(tokenizer, mimo_tokenizer, fixtures):
    frozen = json.loads((fixtures / "functional/context-524288.json").read_text())
    text = mimo_tokenizer.decode(frozen["prompt"], skip_special_tokens=False)
    head = "<|im_start|>user\n"
    body = text[text.index(head) + len(head):text.index("\n" + NEEDLE_QUESTION)]
    for key, value in NEEDLES:
        body = body.replace(f"{key} is {value}.\n", "")
    paragraph_ids = [tokenizer.encode(part + "\n", add_special_tokens=False).ids
                     for part in body.split("\n") if part]
    prefix = tokenizer.encode(f"<|im_start|>system\n{NEEDLE_SYSTEM}<|im_end|>\n<|im_start|>user\n",
                              add_special_tokens=False).ids
    suffix = tokenizer.encode(f"\n{NEEDLE_QUESTION}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                              add_special_tokens=False).ids
    needles = [tokenizer.encode(f"{key} is {value}.\n", add_special_tokens=False).ids for key, value in NEEDLES]
    rows = []
    for context in NEEDLE_CONTEXTS:
        budget = context - NEEDLE_OUTPUT - len(prefix) - len(suffix) - sum(map(len, needles))
        filler, starts = [], []
        while len(filler) < budget:
            filler.extend(paragraph_ids[len(starts) % len(paragraph_ids)])
            starts.append(len(filler))
        filler = filler[:budget]
        # Needles go in front of the paragraph starting nearest each depth.
        cuts = [min(starts, key=lambda s: abs(s - depth * budget)) for depth in (0.1, 0.5, 0.9)]
        prompt, last = list(prefix), 0
        for cut, needle in zip(cuts, needles):
            prompt += filler[last:cut] + needle
            last = cut
        prompt += filler[last:] + suffix
        assert len(prompt) == context - NEEDLE_OUTPUT
        rows.append({"name": f"needle_{context // 1024}k", "total_context": context, "max_tokens": NEEDLE_OUTPUT,
                     "expected": ",".join(value for _, value in NEEDLES), "needle_offsets": cuts,
                     "token_ids": prompt})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_QWEN38_CHECKPOINT"))
    parser.add_argument("--mimo-checkpoint", default=os.environ.get("CHOPINFER_MIMO_CHECKPOINT"))
    parser.add_argument("--mimo-fixtures", default=os.environ.get("CHOPINFER_MIMO_FIXTURES"))
    parser.add_argument("--evals", default=Path(os.environ.get("CHOPINFER_DATA", "")) / "evals")
    parser.add_argument("--output", required=True, help="refs directory; writes prompts/prompts.json")
    parser.add_argument("--receipt", default=REPO / "fixtures/qwen38/prompts.json")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    chat_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    mimo_tokenizer = Tokenizer.from_file(str(Path(args.mimo_checkpoint) / "tokenizer.json"))
    evals = Path(args.evals)
    manifest = json.loads((evals / "manifest.json").read_text())["wikitext"]
    raw = (evals / "wikitext.json").read_bytes()
    if sha256(raw) != manifest["sha256"]:
        raise ValueError("wikitext.json does not match its manifest")
    corpus = paragraphs(json.loads(raw))

    teacher = teacher_texts(tokenizer, chat_tokenizer)
    long = long_texts(tokenizer, corpus)
    needles = needle_prompts(tokenizer, mimo_tokenizer, Path(args.mimo_fixtures))
    full = {"schema": SCHEMA, "teacher": teacher, "long": long, "needles": needles}
    blob = (json.dumps(full, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    out = Path(args.output) / "prompts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.json").write_bytes(blob)

    def digest(row):
        return {k: v for k, v in row.items() if k != "token_ids"} | {
            "token_count": len(row["token_ids"]),
            "token_ids_sha256": sha256(json.dumps(row["token_ids"]).encode())}

    receipt = {
        "schema": SCHEMA,
        "provenance": (__doc__.split("\n\n", 1)[1]).strip(),
        "tokenizer_json_sha256": sha256((checkpoint / "tokenizer.json").read_bytes()),
        "chat_template_sha256": sha256((checkpoint / "chat_template.jinja").read_bytes()),
        "wikitext": {"revision": manifest["revision"], "sha256": manifest["sha256"]},
        "needle_source_sha256": sha256((Path(args.mimo_fixtures) / "functional/context-524288.json").read_bytes()),
        "full_file": {"path": "${QWEN38_REFS}/prompts/prompts.json", "sha256": sha256(blob), "bytes": len(blob)},
        "teacher": teacher,
        "long": [digest(row) for row in long],
        "needles": [digest(row) for row in needles],
    }
    Path(args.receipt).write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"prompts": str(out / "prompts.json"), "sha256": sha256(blob),
                      "teacher": len(teacher), "long": [r["tokens"] for r in long],
                      "needles": [len(r["token_ids"]) for r in needles]}))


if __name__ == "__main__":
    main()
