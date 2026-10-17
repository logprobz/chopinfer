"""Tokenizer goldens and the HF corpus for DeepSeek V4.1 Flash.

Encodes with HF `tokenizers` from the official checkpoint's tokenizer.json
(`encode(add_special_tokens=False)`, every added token recognized):

    goldens OUT.json   small handwritten texts inline, and the checkpoint's
                       own encoder test outputs by name and SHA-256
    corpus DIR         about 30,000 texts (random mixes of whitespace, digits,
                       CJK, kana, punctuation, combining marks, emoji and
                       markers, the checkpoint's own text files and long
                       whitespace runs) as DIR/corpus.json and their ids as
                       DIR/hf-ids.json; DIR belongs under $CHOPINFER_DATA

The Rust tests compare runtime-tokenizer with these: compiler/tests/
tokenizer.rs (goldens) and tokenizer_corpus.rs (CHOPINFER_DEEPSEEK_TOKENIZER_CORPUS
names DIR). Usage: python3 -I tokenizer_goldens.py CHECKPOINT goldens|corpus OUT
"""
import hashlib
import json
import os
import random
import sys

from tokenizers import Tokenizer

TEXTS = [
    "", " ", "  ", "\n", "\n\n", " \n", "\t \n \t", "hello world", "  leading", "trailing  ",
    "a  b   c    d", "123456789", "1234 56", "3.14159", "x=10000000", "价格是123元",
    "日本語のテキストとカタカナ", "한국어 텍스트", "naïve café résumé", "é", "😀👍🏽 emoji",
    "مرحبا بالعالم", "Ω≈ç√∫˜µ≤≥÷", "def f(x):\n    return x**2\n\n\n",
    "<｜User｜>hi<｜Assistant｜><think>", "｜DSML｜ calls>",
    "<｜DSML｜ invoke name=\"get\">\n</｜DSML｜ invoke>", "a\r\nb\r\n\r\n", "   \n   x", "?!...",
    "--flag", "\"quoted\"", "it's", "I'M", "path/to/file.rs:12", "　full width space", " nbsp",
    "​zero width", "﻿bom", "x̀́̂", "<｜begin▁of▁sentence｜><｜System｜>sys<｜latest_reminder｜>",
    "<｜deepseek_image｜><｜deepseek_image｜>text", "<｜fim▁hole｜>a<｜fim▁end｜>", " " * 300 + "x",
]
CHECKPOINT_TEXTS = [f"encoding/tests/test_output_{i}.txt" for i in range(1, 6)]


def goldens(checkpoint, tokenizer, out):
    rows = [{"text": text, "token_ids": tokenizer.encode(text, add_special_tokens=False).ids} for text in TEXTS]
    for name in CHECKPOINT_TEXTS:
        data = open(os.path.join(checkpoint, name), "rb").read()
        rows.append({"checkpoint_file": name, "sha256": hashlib.sha256(data).hexdigest(),
                     "token_ids": tokenizer.encode(data.decode(), add_special_tokens=False).ids})
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(rows)} goldens")


def corpus(checkpoint, tokenizer, out):
    specials = [a["content"] for a in json.load(open(os.path.join(checkpoint, "tokenizer.json")))["added_tokens"]]
    texts = list(TEXTS)
    for name in CHECKPOINT_TEXTS + ["encoding/encoding.py", "inference/model.py", "README.md"]:
        texts.append(open(os.path.join(checkpoint, name)).read())
    alpha = list(" \n\t\r") * 6 + list("abcxyzABCXYZ") * 3 + list("0123456789") * 2 + \
        list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + list("一二三价格龥ぁあゟ゠アヿ") + list("éüñßøΩжд") + \
        ["́", "̈", "😀", "👍", "‍", " ", "　", "한", "ب", " ", "\x0b", "\x0c", "\u0085"]
    rng = random.Random(1234)
    for _ in range(30000):
        texts.append("".join(rng.choice(alpha) for _ in range(rng.randint(1, 60))))
    for _ in range(500):
        texts.append("".join(rng.choice(alpha + specials[:5] + ["<think>", "｜DSML｜", "<｜User｜>"])
                             for _ in range(rng.randint(1, 30))))
    texts += [" " * 5000 + "x", "\n" * 3000 + "y", "ab " * 20000]
    ids = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    os.makedirs(out, exist_ok=True)
    json.dump(texts, open(os.path.join(out, "corpus.json"), "w"))
    json.dump(ids, open(os.path.join(out, "hf-ids.json"), "w"))
    print(f"{len(texts)} corpus texts")


def main():
    checkpoint, command, out = sys.argv[1:4]
    tokenizer = Tokenizer.from_file(os.path.join(checkpoint, "tokenizer.json"))
    {"goldens": goldens, "corpus": corpus}[command](checkpoint, tokenizer, out)


if __name__ == "__main__":
    main()
