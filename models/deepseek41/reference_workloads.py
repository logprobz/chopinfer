"""Reference runner workloads for DeepSeek V4.1 Flash's teacher texts,
long-context texts and image probes.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/deepseek41/reference_workloads.py \\
        --output DIR [--checkpoint CK] [--evals DIR] [--long-tokens 131072]
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/deepseek41/reference_workloads.py \\
        --answers RUN/f32.json --output DIR [--lock LOCK]

Runs on CPU in the runner environment: the checkpoint's tokenizer.json
(through `tokenizers`, which adds no special tokens for this tokenizer) and
its pinned `encoding/encoding.py` (the rendering reference, V4), never the
model. Writes, in DIR (refused if it exists):

- `teacher.json`: the short texts of fixtures/mimo26/teacher-texts.json
  (not its chat and long-context texts), DeepSeek chat texts rendered by
  `encode_messages` (`CHATS`, chat and thinking modes, reasoning efforts and
  a tool call) and `long_4k`, each scored from its second token (`long_4k`
  over its last 512). Plain texts start with the BOS token, as every
  rendered conversation does. Every sequence holds at most `F64_TOKENS`,
  so the F64-TQ run covers the whole teacher set.
- `long-context.json`: `long_4k` first, so the runner's repeat check of each
  mode's first sequence is cheap, then `long_<n>k` of `--long-tokens`
  tokens (default 131,072), scored over its last 1,024. Long texts are
  cleaned wikitext-103 test paragraphs from the pinned evals cache
  (`<evals>/wikitext.json`, SHA256 in its manifest) from a seeded start; the
  scored tail quotes a passage from the first twentieth of the text, so
  those positions can copy across the whole context.
- `probes-greedy.json` and one `<probe>.bf16` / `<probe>-grid.i64` pair per
  probe: the PNGs of probe_images.py in a user turn with `PROBE_QUESTION`
  (chat mode), each image turned into the vendor's BF16 patches and
  [n_vit_h, n_vit_w] grid by the pinned `inference/image_processor.py`
  (`load_image`) and its span expanded as `prepare_vl_inputs` does; a
  free-text probe with a greedy answer of at most `MAX_NEW_TOKENS` ending at
  the end-of-sentence token.
- `capture/`: the probes' names, images, question and prompt token IDs, the
  PNG digests and a manifest, for a media gate.
- `manifest.json`: the digests of every input (tokenizer, encoding.py,
  image processor and config, wikitext, each probe PNG and its patches) and
  each sequence's length and token-ID SHA256.

With `--answers RUN/f32.json`, DIR already holds those files and gains
`probes.json`: the probes forcing the F32 run's greedy answers (checked to
be this workload's sequences, model and checkpoint lock).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
sys.path.insert(0, str(ROOT / "models/deepseek41"))

SCHEMA = "chopinfer.refrun.workload.v1"
MODEL = "deepseek41"
SEED = 20261017
F64_TOKENS = 4096
LONG_4K, LONG_4K_SCORED = 4096, 512
LONG_SCORED = 1024
MIN_LONG_TOKENS = 65536
ENCODING = "encoding/encoding.py"
ENCODING_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"
BOS = "<｜begin▁of▁sentence｜>"
IMAGE_PROCESSOR = "inference/image_processor.py"
IMAGE_PROCESSOR_SHA256 = "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272"
CONFIG = "inference/config.json"
LOCK = ROOT / "models/deepseek41/checkpoint-files.json"
PROBE_QUESTION = "Describe this image in one short paragraph."
MAX_NEW_TOKENS = 256
# <｜end▁of▁sentence｜>, the config's eos_token_id.
STOP_IDS = (1,)
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}
# DeepSeek chat texts, the counterparts of GLM's, MiMo's and Qwen's chat
# teacher texts: each conversation ends with an assistant turn.
CHATS = {
    "chat_system": {"thinking_mode": "chat", "messages": [
        {"role": "system", "content": "You are a concise assistant. Answer in at most two sentences."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant",
         "content": "Sunlight is scattered by air molecules, and shorter blue wavelengths scatter much more "
                    "strongly than red ones. That scattered blue light reaches our eyes from every direction."}]},
    "chat_thinking": {"thinking_mode": "thinking", "messages": [
        {"role": "user", "content": "Is 91 a prime number?"},
        {"role": "assistant", "reasoning_content":
            "I need to check small divisors. 91 is odd, so 2 does not divide it. The digit sum is 10, so 3 "
            "does not divide it. It does not end in 0 or 5. 7 * 13 = 91, so 91 is composite.",
         "content": "No. 91 = 7 * 13, so it is not prime."}]},
    "chat_effort_low": {"thinking_mode": "thinking", "reasoning_effort": "low", "messages": [
        {"role": "user", "content": "Name three prime numbers greater than 50."},
        {"role": "assistant", "reasoning_content": "53, 59 and 61 are prime.", "content": "53, 59 and 61."}]},
    "chat_effort_max": {"thinking_mode": "thinking", "reasoning_effort": "max", "messages": [
        {"role": "user", "content": "What is 17 squared?"},
        {"role": "assistant", "reasoning_content": "17 * 17 = 289.", "content": "289."}]},
    "chat_multi_turn": {"thinking_mode": "chat", "messages": [
        {"role": "user", "content": "Give me a name for a gray cat."},
        {"role": "assistant", "content": "How about Pewter? It is a soft gray metal, which suits a gray cat."},
        {"role": "user", "content": "Something shorter, please."},
        {"role": "assistant", "content": "Try Ash, Smoke or Slate."}]},
    "chat_zh": {"thinking_mode": "chat", "messages": [
        {"role": "user", "content": "请用一句话解释什么是机器学习。"},
        {"role": "assistant",
         "content": "机器学习是让计算机从数据中自动发现规律，并利用这些规律对新数据做出预测或决策的方法。"}]},
    "chat_tool_call": {"thinking_mode": "thinking", "messages": [
        {"role": "system", "content": "You can call tools.", "tools": [WEATHER]},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "reasoning_content": "The user wants the current weather, so call get_weather.",
         "content": "", "tool_calls": [{"type": "function", "function": {
             "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "content": "{\"temperature\": 18, \"condition\": \"cloudy\"}"},
        {"role": "assistant", "reasoning_content": "The tool says 18 degrees and cloudy.",
         "content": "It is 18 degrees Celsius and cloudy in Paris."}]},
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ids_sha256(ids):
    return sha256(json.dumps(ids).encode())


def tokenizer(checkpoint):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))


def encoding(checkpoint):
    """The checkpoint's encoding.py, executed from its pinned bytes."""
    path = Path(checkpoint) / ENCODING
    data = path.read_bytes()
    if sha256(data) != ENCODING_SHA256:
        raise ValueError(f"{ENCODING} is not the pinned {ENCODING_SHA256}")
    module = types.ModuleType("deepseek41_encoding")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def chat_rows(renderer, chats):
    """The conversations rendered by encode_messages (BOS included)."""
    return [{"name": name, "category": "chat",
             "text": renderer.encode_messages(spec["messages"], thinking_mode=spec["thinking_mode"],
                                              reasoning_effort=spec.get("reasoning_effort"))}
            for name, spec in chats.items()]


def scored(tok, rows):
    return [{"name": row["name"], "category": row["category"], "token_ids": encode(tok, row["text"]),
             "score_from": 1} for row in rows]


def short_texts(tok, renderer):
    """The model-neutral short teacher texts (after BOS) and the DeepSeek chat texts, tokenized."""
    source = json.loads((ROOT / "fixtures/mimo26/teacher-texts.json").read_text())
    rows = [{"name": t["name"], "category": t["category"], "text": BOS + t["text"]} for t in source["texts"]
            if t["category"] not in ("chat", "long-context")]
    return scored(tok, rows + chat_rows(renderer, CHATS))


def corpus(evals):
    """Cleaned wikitext-103 test paragraphs of the pinned evals cache."""
    from evals import paragraphs
    manifest = json.loads((Path(evals) / "manifest.json").read_text())["wikitext"]
    raw = (Path(evals) / "wikitext.json").read_bytes()
    if sha256(raw) != manifest["sha256"]:
        raise ValueError("wikitext.json does not match its manifest")
    return paragraphs(json.loads(raw)), {"revision": manifest["revision"], "sha256": manifest["sha256"]}


def long_text(tok, paragraphs, length, scored_tokens):
    """`length` tokens: BOS, wikitext from a seeded start, and the last
    `scored_tokens` quoting a passage that starts at a paragraph break after
    5% of the text."""
    rng = random.Random(f"{SEED}:deepseek41-long")
    start = rng.randrange(len(paragraphs))
    ordered = paragraphs[start:] + paragraphs[:start]
    bos = encode(tok, BOS)
    intro = encode(tok, "\n\nThe following passage appeared earlier in this text:\n\n")
    body = []
    while len(body) < length:
        for paragraph in ordered:
            body.extend(encode(tok, paragraph + "\n\n"))
            if len(body) >= length:
                break
    body = body[:length - scored_tokens - len(intro) - len(bos)]
    first = next(i for i in range(len(body) // 20, len(body)) if tok.decode(body[i - 1:i]).endswith("\n"))
    prompt = bos + body + intro + body[first:first + scored_tokens]
    if len(prompt) != length:
        raise ValueError(f"the long text has {len(prompt)} tokens, not {length}")
    return {"name": f"long_{length // 1024}k", "category": "long-context", "token_ids": prompt,
            "score_from": length - scored_tokens, "quote_source_start": len(bos) + first}


def image_processor(checkpoint):
    """The checkpoint's image_processor.py, executed from its pinned bytes."""
    path = Path(checkpoint) / IMAGE_PROCESSOR
    data = path.read_bytes()
    if sha256(data) != IMAGE_PROCESSOR_SHA256:
        raise ValueError(f"{IMAGE_PROCESSOR} is not the pinned {IMAGE_PROCESSOR_SHA256}")
    module = types.ModuleType("deepseek41_image_processor")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def processor_identity(checkpoint):
    return {"code_sha256": IMAGE_PROCESSOR_SHA256,
            "config_sha256": sha256((Path(checkpoint) / CONFIG).read_bytes())}


def probe_prompt(tok, renderer, processor, args, png):
    """(token IDs with the image span expanded, BF16 patches, [n_vit_h, n_vit_w]):
    encode_messages renders the turn with one image placeholder, load_image
    makes the patches, and the placeholder becomes the vendor's span
    (image_processor.prepare_vl_inputs)."""
    messages = [{"role": "user", "content": [{"type": "image", "data": png},
                                             {"type": "text", "text": PROBE_QUESTION}]}]
    prompt, media = renderer.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
    if len(media["images"]) != 1:
        raise ValueError("a probe holds one image")
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = processor.load_image(media["images"][0], args)
    span = processor.image_token_types(n_llm_h, n_llm_w).numel()
    ids = []
    for token in encode(tok, prompt):
        ids.extend([args.image_token_id] * span if token == args.image_token_id else [token])
    if ids.count(args.image_token_id) != span:
        raise ValueError("the rendered probe does not hold exactly one image placeholder")
    return ids, patches, (n_vit_h, n_vit_w)


def probes(tok, renderer, checkpoint, output):
    """The image probes with greedy answers, and the request capture (`capture/`)."""
    import probe_images
    import torch
    processor = image_processor(checkpoint)
    args = types.SimpleNamespace(**json.loads((Path(checkpoint) / CONFIG).read_text()))
    sequences, digests, capture = [], {}, []
    pngs = probe_images.pngs()
    for name, data in pngs.items():
        ids, patches, grid = probe_prompt(tok, renderer, processor, args, data)
        raw = patches.contiguous().view(torch.int16).numpy().tobytes()
        (output / f"{name}.bf16").write_bytes(raw)
        (output / f"{name}-grid.i64").write_bytes(torch.tensor(grid, dtype=torch.int64).numpy().tobytes())
        digests[name] = {"png_sha256": sha256(data), "grid": list(grid), "patches_sha256": sha256(raw)}
        media = {"kind": "image", "tensor": {"path": f"{name}.bf16", "dtype": "bf16", "shape": list(patches.shape)},
                 "grid": {"path": f"{name}-grid.i64", "dtype": "i64", "shape": [2]},
                 "processor": processor_identity(checkpoint)}
        sequences.append({"name": name, "category": "free-text", "token_ids": ids, "score_from": len(ids),
                          "media": [media],
                          "answer": {"policy": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "stop_ids": list(STOP_IDS)}})
        capture.append({"name": name, "images": [name], "text": PROBE_QUESTION, "token_ids": ids})
    folder = output / "capture"
    folder.mkdir()
    data = (json.dumps(capture) + "\n").encode()
    (folder / "probes.json").write_bytes(data)
    (folder / "identity.json").write_text(json.dumps({"images": {n: sha256(b) for n, b in pngs.items()}}) + "\n")
    (folder / "manifest.json").write_text(json.dumps({"files": {"probes.json": {"sha256": sha256(data)}}}) + "\n")
    return sequences, digests


def answered(output, run, lock=LOCK):
    """`probes-greedy.json` with each probe forcing the F32 run's greedy
    answer, checked to be this workload's sequence, model and lock."""
    from tools.model_sdk.diagnostic import canonical
    from tools.refrun import keys, workload
    model, records = workload.load(output / "probes-greedy.json")
    document = json.loads(Path(run).read_text())
    if document["policy"]["name"] != "f32":
        raise ValueError(f"{run} is not an f32 run")
    source_lock = sha256(canonical(keys.read_lock(lock)))
    entries = {e["name"]: e for e in document["entries"] if e["key"]["sequence"]["answer"]["policy"] == "greedy"}
    if sorted(entries) != sorted(r.name for r in records):
        raise ValueError(f"{run} does not answer every probe of probes-greedy.json")
    sequences = []
    for record, spec in zip(records, json.loads((output / "probes-greedy.json").read_text())["sequences"]):
        key = entries[record.name]["key"]
        if model != MODEL or key["model"] != MODEL:
            raise ValueError(f"{record.name}: the run is of model {key['model']}, not {MODEL}")
        if key["checkpoint"]["source_lock_sha256"] != source_lock:
            raise ValueError(f"{record.name}: the run read another checkpoint lock")
        if key["sequence"] != keys.sequence(record):
            raise ValueError(f"{record.name}: the run scored another sequence")
        sequences.append(spec | {"answer": {"policy": "forced", "ids": entries[record.name]["scores"]["answer"]}})
    for sequence in sequences:
        workload.sequence(sequence, output)
    return sequences


def write(path, sequences):
    with path.open("x") as stream:
        stream.write(json.dumps({"schema": SCHEMA, "model": MODEL, "sequences": sequences}) + "\n")


def build(checkpoint, evals, output, long_tokens):
    if long_tokens < MIN_LONG_TOKENS or long_tokens % 1024:
        raise ValueError(f"--long-tokens must be a multiple of 1024 of at least {MIN_LONG_TOKENS}")
    checkpoint = Path(checkpoint)
    tok, renderer = tokenizer(checkpoint), encoding(checkpoint)
    paragraphs, wikitext = corpus(evals)
    long_4k = long_text(tok, paragraphs, LONG_4K, LONG_4K_SCORED)
    long = long_text(tok, paragraphs, long_tokens, LONG_SCORED)
    teacher = short_texts(tok, renderer) + [long_4k]
    for sequence in teacher:
        if len(sequence["token_ids"]) > F64_TOKENS:
            raise ValueError(f"{sequence['name']}: {len(sequence['token_ids'])} tokens exceed the f64 limit")
    output.mkdir(parents=True)

    def plain(sequence):
        return {k: v for k, v in sequence.items() if k != "quote_source_start"}
    media, images = probes(tok, renderer, checkpoint, output)
    write(output / "teacher.json", [plain(s) for s in teacher])
    write(output / "long-context.json", [plain(long_4k), plain(long)])
    write(output / "probes-greedy.json", media)
    manifest = {
        "schema": "chopinfer.deepseek41.reference-workloads.v1",
        "generator": "models/deepseek41/reference_workloads.py",
        "tokenizer_json_sha256": sha256((checkpoint / "tokenizer.json").read_bytes()),
        "encoding_py_sha256": ENCODING_SHA256, "image_processor": processor_identity(checkpoint),
        "wikitext": wikitext, "images": images,
        "long_quote_source_start": {s["name"]: s["quote_source_start"] for s in (long_4k, long)},
        "sequences": {kind: [{"name": s["name"], "tokens": len(s["token_ids"]), "score_from": s["score_from"],
                              "token_ids_sha256": ids_sha256(s["token_ids"])} for s in sequences]
                      for kind, sequences in (("teacher", teacher), ("long-context", [long_4k, long]),
                                              ("probes", media))}}
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT"),
                        help="checkpoint directory (tokenizer.json, encoding/encoding.py)")
    parser.add_argument("--evals", type=Path, default=Path(os.environ.get("CHOPINFER_DATA", "")) / "evals",
                        help="the evals cache holding wikitext.json and manifest.json")
    parser.add_argument("--long-tokens", type=int, default=131072)
    parser.add_argument("--answers", type=Path, help="the runner's f32.json of probes-greedy.json; "
                                                     "writes probes.json into the existing --output")
    parser.add_argument("--lock", type=Path, default=LOCK, help="the checkpoint lock the run read")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.answers:
        write(args.output / "probes.json", answered(args.output, args.answers, args.lock))
        return
    if not args.checkpoint:
        parser.error("set CHOPINFER_DEEPSEEK_CHECKPOINT or --checkpoint")
    manifest = build(args.checkpoint, args.evals, args.output, args.long_tokens)
    print(json.dumps({kind: [(s["name"], s["tokens"]) for s in rows]
                      for kind, rows in manifest["sequences"].items()}))


if __name__ == "__main__":
    main()
