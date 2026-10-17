"""Write the whitespace-tail tokenizer fixture and its HF token ids.

A byte-level BPE whose last split pattern ends with the alternatives
`\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`, with merges that join characters across the
places those alternatives split, so a wrong split changes the ids. Run with
the HF `tokenizers` package (0.23.2 for the committed files):

    python -I generate.py
"""
import json
import pathlib
import random

import tokenizers

HERE = pathlib.Path(__file__).resolve().parent
PATTERNS = [
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
]
SPECIALS = ["<bos>", "<eos>", "<pad>"]
MAX_LENGTH = 1048576


def byte_alphabet():
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    mapping, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + extra)
            extra += 1
    return mapping


def merges():
    alphabet = byte_alphabet()
    space, tab, newline, cr = (alphabet[ord(c)] for c in " \t\n\r")
    ideographic = "".join(alphabet[b] for b in "　".encode())
    pairs = [
        (space, space), (space * 2, space * 2), (space * 4, space * 4), (space * 8, space * 8),
        (newline, newline), (space, newline), (newline, space), (cr, newline), (tab, space),
        (space, tab), (tab, tab), (space, "x"), (space, "a"), (newline, "x"), (space, "."),
        (alphabet[0xE3], alphabet[0x80]), (alphabet[0xE3] + alphabet[0x80], alphabet[0x80]),
        (space, ideographic), (ideographic, space), (alphabet[0xC2], alphabet[0xA0]),
    ]
    return [f"{left} {right}" for left, right in pairs], ["".join(pair) for pair in pairs]


def definition():
    alphabet = byte_alphabet()
    vocab = {token: index for index, token in enumerate(SPECIALS)}
    for byte in range(256):
        vocab[alphabet[byte]] = byte + len(SPECIALS)
    records, results = merges()
    for result in results:
        assert result not in vocab, result
        vocab[result] = len(vocab)
    split = [{"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False}
             for pattern in PATTERNS]
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": index, "content": token, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True}
                         for index, token in enumerate(SPECIALS)],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": split + [
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                  "vocab": vocab, "merges": records},
    }


def configuration():
    def added(content):
        return {"__type": "AddedToken", "content": content, "lstrip": False, "normalized": True,
                "rstrip": False, "single_word": False}
    return {"add_bos_token": False, "add_eos_token": False, "bos_token": added("<bos>"),
            "clean_up_tokenization_spaces": False, "eos_token": added("<eos>"), "legacy": True,
            "model_max_length": MAX_LENGTH, "pad_token": added("<eos>"), "sp_model_kwargs": {},
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": None}


def texts():
    whitespace = [" ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", "\x85", "\xa0", " ", "　"]
    tails = ["", "x", "a", ".", "1", "一", "́", "\U0001f600", "!x", "-", "\n", " \nx"]
    shaped = []
    for head in ["", "x"]:
        for ws in whitespace:
            for count in [1, 2, 3]:
                for tail in tails:
                    shaped.append(head + ws * count + tail)
    for left in whitespace:
        for right in whitespace:
            shaped.append("x" + left + right + "x")
            shaped.append(left + " " + right)
    alphabet = whitespace * 4 + list("abxyzAB") + list("0123456789") + list("!.,-_(){}") + \
        ["一", "あ", "ア", "é", "́", "̈", "\U0001f600", "‍", "가", "ب"]
    rng = random.Random(20261017)
    generated = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24))) for _ in range(400)]
    return shaped + generated


def long_cases():
    count = 1 << 20
    return [(" ", count, ""), (" ", count - 1, "x"), (" ", count - 1, "\n"), (" ", count - 3, " \nx"),
            ("\n", count, ""), ("\n", count - 1, "x"), ("\t", count - 1, "x"), ("\r\n", count >> 1, " x"),
            (" \t", count >> 1, "x"), ("　", count // 3, "x"), (" ", count - 2, ".x")]


def runs(ids):
    out = []
    for token in ids:
        if out and out[-1][0] == token:
            out[-1][1] += 1
        else:
            out.append([token, 1])
    return out


def main():
    (HERE / "tokenizer.json").write_text(json.dumps(definition(), ensure_ascii=False, separators=(",", ":")))
    (HERE / "tokenizer_config.json").write_text(json.dumps(configuration(), separators=(",", ":")))
    tokenizer = tokenizers.Tokenizer.from_file(str(HERE / "tokenizer.json"))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def line(value):
        return json.dumps(value, ensure_ascii=False)

    cases = [line([text, " ".join(map(str, encode(text)))]) for text in texts()]
    long = [line({"repeat": repeat, "count": count, "suffix": suffix, "runs": runs(encode(repeat * count + suffix))})
            for repeat, count, suffix in long_cases()]
    (HERE / "cases.json").write_text(
        f'{{"tokenizers":{line(tokenizers.__version__)},\n"cases":[\n'
        + ",\n".join(cases) + '],\n"long":[\n' + ",\n".join(long) + "]}\n")


if __name__ == "__main__":
    main()
