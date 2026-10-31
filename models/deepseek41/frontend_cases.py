"""DeepSeek V4.1 Flash cases for the differential frontend corpus.

The checkpoint ships no chat template: its prompt-format reference is the
Python encoder `encoding/encoding.py`, which `PIN` names with its digest.
`python -m tools.refrun.frontend_corpus generate --model deepseek41` renders
these cases with that encoder (`$CHOPINFER_DEEPSEEK_CHECKPOINT`, revision
1bae0e10), through `encoder_render`, the mapping of a request onto
`encode_messages` that `models/deepseek41/contract/src/frontend.rs` serves.
The module states the pin, the transport translation, the generation endings,
how a history turn reads back, and DeepSeek's own axes and fixed cases.
"""

import bisect
import copy
import json
import math
import re

PIN = {
    "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    "source": "encoding/encoding.py",
    "source_sha256": "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2",
    "checkpoint_env": "CHOPINFER_DEEPSEEK_CHECKPOINT",
}

# What the export's OpenedV1 states: history reasoning_content on every
# assistant turn and the client's tool objects whole.
TRANSPORT = {
    "reasoning_content_fallback": True,
    "tool_call_reasoning_fallback": False,
    "reads_tool_objects": True,
    "tool_references": False,
    "extra_roles": [],
}

# The request fields the encoder renders beyond the HF input, as request
# mapping reads them.
INPUT_FIELDS = ("response_format", "tool_choice")

# What a forced tool choice's prompt ends with (the declaration's
# tools.choice forced_prefill): the opening of the calls block, as
# deepseek-recipe prefills it (v4/mod.rs lines 261-266).
FORCED_PREFILL = "\n\n<｜DSML｜ calls>\n"

# Tool-call arguments reach the encoder as the text sent: it reads them
# itself, up to twice (lines 533-544).
RAW_TOOL_ARGUMENTS = True

# The protocol efforts the encoder does not name, as deepseek-recipe 0.2.0
# renders them and the served format reads them (reasoning.rs lines 37-48,
# dsv41.rs line 24): minimal as low, medium and xhigh as high, ultra as max.
EFFORT_ALIASES = {"minimal": "low", "medium": "high", "xhigh": "high", "ultra": "max"}

EOS = "<｜end▁of▁sentence｜>"

# Every assistant turn ends with the end-of-sentence token (encoding.py
# line 60), so a history render holds it; these apply only without it.
ENDINGS = {"tool_calls": [EOS], "answer": [EOS]}

# Refused by this frontend: each runs in one row of otherwise ordinary values.
# A strict first tool beside the shared rows' non-strict clock tool is
# refused, as deepseek-recipe 0.2.0 refuses strict and non-strict tools
# together.
ISOLATED = {"tool-field:strict"}

# The markers the vendor parser refuses inside reasoning or content
# (encoding.py lines 981-983).
SPECIAL = ("<｜begin▁of▁sentence｜>", EOS, "<think>", "</think>", "｜DSML｜")


def thinking(keywords):
    """The mode the served frontend selects: the first declared thinking
    keyword the request sets, else thinking (`default_enabled`). The
    declaration switches by the top-level thinking object, which the HF
    input carries as both keywords, and drops the client's own keywords."""
    for keyword in ("enable_thinking", "thinking"):
        if isinstance(keywords.get(keyword), bool):
            return keywords[keyword]
    return True


def encoder_message(message):
    """One HF message as the encoder reads it: a media item back in OpenAI's
    `image_url` form, which `_extract_image` (lines 305-320) reads. A file
    part's data URL is an image, as deepseek-recipe maps it (convert.rs
    lines 263-279), from OpenAI's nested file object or the recipe's flat
    fields; a file ID is refused, as the recipe refuses it."""
    message = copy.deepcopy(message)
    if isinstance(message.get("content"), list):
        parts = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") in ("image", "video", "audio"):
                kind = part["type"]
                part = {"type": f"{kind}_url", f"{kind}_url": {"url": part.get(kind)}}
            elif isinstance(part, dict) and part.get("type") == "file":
                file = part.get("file") if isinstance(part.get("file"), dict) else part
                if file.get("file_id") is not None or not isinstance(file.get("file_data"), str):
                    raise ValueError("file must have file_data and no file_id")
                part = {"type": "image_url", "image_url": {"url": file["file_data"]}}
            parts.append(part)
        message["content"] = parts
    return message


def recipe_tool(tool, encoder):
    """An offered tool as deepseek-recipe maps it (convert.rs lines 178-192,
    v4/mod.rs lines 284-293): its name, its description or "", its
    parameters as sent (None when absent or null) and whether it is strict,
    every other field dropped; `recipe_function` checks the parameters. A
    tool in a namespace (on the tool, else on its function) is named and
    described as the encoder's `tools_from_openai_format` (lines 84-98)
    names it, the transport's qualified name (`namespace` in
    frontend-divergences.json), and then mapped the same way."""
    sent = tool.get("function") or {}
    function = {"name": sent.get("name"), "description": sent.get("description") or ""}
    wrapped = {"type": "function", "function": function}
    if tool.get("namespace") is not None:
        wrapped["namespace"] = tool["namespace"]
    elif sent.get("namespace") is not None:
        function["namespace"] = sent["namespace"]
    if "namespace" in wrapped or "namespace" in function:
        function = encoder["tools_from_openai_format"]([wrapped])[0]
    return {"name": function["name"], "description": function.get("description") or "",
            "parameters": sent.get("parameters"), "strict": sent.get("strict") is True}


def recipe_function(tool):
    """The tool object the encoder reads for a `recipe_tool`, its parameters
    checked as deepseek-recipe 0.2.0 checks them (validation.rs lines
    209-240): a strict tool's normalized (`strict_tool_parameters`), and a
    non-strict tool's present ones declaring type object, else `{}`."""
    parameters = tool["parameters"]
    if tool["strict"]:
        parameters = strict_tool_parameters(parameters)
    elif parameters is not None:
        kind = parameters.get("type") if isinstance(parameters, dict) else None
        if kind != "object":
            raise ValueError(f"Invalid schema for function '{tool['name']}': schema must be a JSON Schema of "
                             f"'type: \"object\"', got 'type: {compact_json(kind)}'.")
    else:
        parameters = {}
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool["description"], "parameters": parameters}}


def compact_json(value):
    """`value` as serde_json's `Display` writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Strict JSON Schema normalization, as deepseek-recipe 0.2.0 normalizes a
# strict tool's parameters and checks a strict json_schema response format
# (json_schema.rs lines 1-287, validation.rs lines 200-264), and as
# models/deepseek41/contract/src/frontend/strict_schema.rs ports it: a node
# is read as the recipe deserializes it, with serde's messages, then
# normalized, and written in the recipe's key order.

NO_NODE = "one of `type`, `anyOf`, `$ref` field is required"
SCHEMA_KINDS = ("object", "string", "number", "integer", "boolean", "array", "null")
SIMPLE_KINDS = ("string", "number", "integer", "boolean", "null")
STRING_FORMATS = ("email", "hostname", "ipv4", "ipv6", "uuid")
STRING_KEYWORDS = ("pattern", "format", "enum")
NUMERIC_KEYWORDS = ("const", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")
I64 = range(-(1 << 63), 1 << 63)
U64 = range(0, 1 << 64)


class SchemaError(ValueError):
    """A strict schema the recipe refuses."""


def float_text(value):
    """A float as the recipe's serde_json writes it in a refusal (the `zmij`
    crate's format), as strict_schema.rs writes it: the shortest digits that
    read back, in fixed notation with at least one fraction digit when the
    decimal exponent is from -5 through 15 (`0.00001`, `100.0`), else in
    exponent notation with a signed exponent (`1e-6`, `1.5e+16`)."""
    if value != value or value in (float("inf"), float("-inf")):
        return {"nan": "NaN"}.get(repr(value), repr(value))
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits, exponent = whole + fraction, int(exponent or 0) + len(whole) - 1
    while len(digits) > 1 and digits[0] == "0":
        digits, exponent = digits[1:], exponent - 1
    digits = digits.rstrip("0") or "0"
    if digits == "0":
        exponent = 0
    if not -5 <= exponent <= 15:
        fraction = f".{digits[1:]}" if len(digits) > 1 else ""
        return f"{sign}{digits[0]}{fraction}e{'-' if exponent < 0 else '+'}{abs(exponent)}"
    if exponent < 0:
        return f"{sign}0.{'0' * (-exponent - 1)}{digits}"
    if exponent + 1 >= len(digits):
        return f"{sign}{digits}{'0' * (exponent + 1 - len(digits))}.0"
    return f"{sign}{digits[:exponent + 1]}.{digits[exponent + 1:]}"


# The code points Rust's `Debug` for str writes as `\u{...}` besides its
# named escapes, as hexadecimal ranges: the printability and
# grapheme-extend tables of the pinned toolchain (rust-toolchain.toml, Rust
# 1.89.0, Unicode 16.0.0), which Python's own Unicode database, an older
# version, cannot reproduce. strict_schema.rs's tests check the table
# against the toolchain.
RUST_DEBUG_ESCAPES = """
1-8 b-c e-1f 7f-a0 ad 300-36f 378-379 380-383 38b 38d 3a2 483-489 530 557-558 58b-58c 590-5bd 5bf 5c1-5c2 5c4-5c5
5c7-5cf 5eb-5ee 5f5-605 610-61a 61c 64b-65f 670 6d6-6dd 6df-6e4 6e7-6e8 6ea-6ed 70e-70f 711 730-74c 7a6-7b0 7b2-7bf
7eb-7f3 7fb-7fd 816-819 81b-823 825-827 829-82f 83f 859-85d 85f 86b-86f 88f-89f 8ca-902 93a 93c 941-948 94d 951-957
962-963 981 984 98d-98e 991-992 9a9 9b1 9b3-9b5 9ba-9bc 9be 9c1-9c6 9c9-9ca 9cd 9cf-9db 9de 9e2-9e5 9fe-a02 a04
a0b-a0e a11-a12 a29 a31 a34 a37 a3a-a3d a41-a58 a5d a5f-a65 a70-a71 a75 a77-a82 a84 a8e a92 aa9 ab1 ab4 aba-abc
ac1-ac8 aca acd-acf ad1-adf ae2-ae5 af2-af8 afa-b01 b04 b0d-b0e b11-b12 b29 b31 b34 b3a-b3c b3e-b3f b41-b46 b49-b4a
b4d-b5b b5e b62-b65 b78-b82 b84 b8b-b8d b91 b96-b98 b9b b9d ba0-ba2 ba5-ba7 bab-bad bba-bbe bc0 bc3-bc5 bc9 bcd-bcf
bd1-be5 bfb-c00 c04 c0d c11 c29 c3a-c3c c3e-c40 c45-c57 c5b-c5c c5e-c5f c62-c65 c70-c76 c81 c8d c91 ca9 cb4 cba-cbc
cbf-cc0 cc2 cc5-cdc cdf ce2-ce5 cf0 cf4-d01 d0d d11 d3b-d3c d3e d41-d45 d49 d4d d50-d53 d57 d62-d65 d80-d81 d84
d97-d99 db2 dbc dbe-dbf dc7-dcf dd2-dd7 ddf-de5 df0-df1 df5-e00 e31 e34-e3e e47-e4e e5c-e80 e83 e85 e8b ea4 ea6 eb1
eb4-ebc ebe-ebf ec5 ec7-ecf eda-edb ee0-eff f18-f19 f35 f37 f39 f48 f6d-f7e f80-f84 f86-f87 f8d-fbd fc6 fcd fdb-fff
102d-1030 1032-1037 1039-103a 103d-103e 1058-1059 105e-1060 1071-1074 1082 1085-1086 108d 109d 10c6 10c8-10cc
10ce-10cf 1249 124e-124f 1257 1259 125e-125f 1289 128e-128f 12b1 12b6-12b7 12bf 12c1 12c6-12c7 12d7 1311 1316-1317
135b-135f 137d-137f 139a-139f 13f6-13f7 13fe-13ff 1680 169d-169f 16f9-16ff 1712-171e 1732-1734 1737-173f 1752-175f
176d 1771-177f 17b4-17b5 17b7-17bd 17c6 17c9-17d3 17dd-17df 17ea-17ef 17fa-17ff 180b-180f 181a-181f 1879-187f
1885-1886 18a9 18ab-18af 18f6-18ff 191f-1922 1927-1928 192c-192f 1932 1939-193f 1941-1943 196e-196f 1975-197f
19ac-19af 19ca-19cf 19db-19dd 1a17-1a18 1a1b-1a1d 1a56 1a58-1a60 1a62 1a65-1a6c 1a73-1a7f 1a8a-1a8f 1a9a-1a9f
1aae-1b03 1b34-1b3d 1b42-1b44 1b4d 1b6b-1b73 1b80-1b81 1ba2-1ba5 1ba8-1bad 1be6 1be8-1be9 1bed 1bef-1bfb 1c2c-1c33
1c36-1c3a 1c4a-1c4c 1c8b-1c8f 1cbb-1cbc 1cc8-1cd2 1cd4-1ce0 1ce2-1ce8 1ced 1cf4 1cf8-1cf9 1cfb-1cff 1dc0-1dff
1f16-1f17 1f1e-1f1f 1f46-1f47 1f4e-1f4f 1f58 1f5a 1f5c 1f5e 1f7e-1f7f 1fb5 1fc5 1fd4-1fd5 1fdc 1ff0-1ff1 1ff5
1fff-200f 2028-202f 205f-206f 2072-2073 208f 209d-209f 20c1-20ff 218c-218f 242a-243f 244b-245f 2b74-2b75 2b96
2cef-2cf1 2cf4-2cf8 2d26 2d28-2d2c 2d2e-2d2f 2d68-2d6e 2d71-2d7f 2d97-2d9f 2da7 2daf 2db7 2dbf 2dc7 2dcf 2dd7
2ddf-2dff 2e5e-2e7f 2e9a 2ef4-2eff 2fd6-2fef 3000 302a-302f 3040 3097-309a 3100-3104 3130 318f 31e6-31ee 321f
a48d-a48f a4c7-a4cf a62c-a63f a66f-a672 a674-a67d a69e-a69f a6f0-a6f1 a6f8-a6ff a7ce-a7cf a7d2 a7d4 a7dd-a7f1 a802
a806 a80b a825-a826 a82c-a82f a83a-a83f a878-a87f a8c4-a8cd a8da-a8f1 a8ff a926-a92d a947-a951 a953-a95e a97d-a982
a9b3 a9b6-a9b9 a9bc-a9bd a9c0 a9ce a9da-a9dd a9e5 a9ff aa29-aa2e aa31-aa32 aa35-aa3f aa43 aa4c aa4e-aa4f aa5a-aa5b
aa7c aab0 aab2-aab4 aab7-aab8 aabe-aabf aac1 aac3-aada aaec-aaed aaf6-ab00 ab07-ab08 ab0f-ab10 ab17-ab1f ab27 ab2f
ab6c-ab6f abe5 abe8 abed-abef abfa-abff d7a4-d7af d7c7-d7ca d7fc-d7ff e000-f8ff fa6e-fa6f fada-faff fb07-fb12
fb18-fb1c fb1e fb37 fb3d fb3f fb42 fb45 fbc3-fbd2 fd90-fd91 fdc8-fdce fdd0-fdef fe00-fe0f fe1a-fe2f fe53 fe67
fe6c-fe6f fe75 fefd-ff00 ff9e-ff9f ffbf-ffc1 ffc8-ffc9 ffd0-ffd1 ffd8-ffd9 ffdd-ffdf ffe7 ffef-fffb fffe-ffff 1000c
10027 1003b 1003e 1004e-1004f 1005e-1007f 100fb-100ff 10103-10106 10134-10136 1018f 1019d-1019f 101a1-101cf
101fd-1027f 1029d-1029f 102d1-102e0 102fc-102ff 10324-1032c 1034b-1034f 10376-1037f 1039e 103c4-103c7 103d6-103ff
1049e-1049f 104aa-104af 104d4-104d7 104fc-104ff 10528-1052f 10564-1056e 1057b 1058b 10593 10596 105a2 105b2 105ba
105bd-105bf 105f4-105ff 10737-1073f 10756-1075f 10768-1077f 10786 107b1 107bb-107ff 10806-10807 10809 10836
10839-1083b 1083d-1083e 10856 1089f-108a6 108b0-108df 108f3 108f6-108fa 1091c-1091e 1093a-1093e 10940-1097f
109b8-109bb 109d0-109d1 10a01-10a0f 10a14 10a18 10a36-10a3f 10a49-10a4f 10a59-10a5f 10aa0-10abf 10ae5-10aea
10af7-10aff 10b36-10b38 10b56-10b57 10b73-10b77 10b92-10b98 10b9d-10ba8 10bb0-10bff 10c49-10c7f 10cb3-10cbf
10cf3-10cf9 10d24-10d2f 10d3a-10d3f 10d66-10d6d 10d86-10d8d 10d90-10e5f 10e7f 10eaa-10eac 10eae-10eaf 10eb2-10ec1
10ec5-10eff 10f28-10f2f 10f46-10f50 10f5a-10f6f 10f82-10f85 10f8a-10faf 10fcc-10fdf 10ff7-10fff 11001 11038-11046
1104e-11051 11070 11073-11074 11076-11081 110b3-110b6 110b9-110ba 110bd 110c2-110cf 110e9-110ef 110fa-11102
11127-1112b 1112d-11135 11148-1114f 11173 11177-11181 111b6-111be 111c0 111c9-111cc 111cf 111e0 111f5-111ff 11212
1122f-11231 11234-11237 1123e 11241-1127f 11287 11289 1128e 1129e 112aa-112af 112df 112e3-112ef 112fa-11301 11304
1130d-1130e 11311-11312 11329 11331 11334 1133a-1133c 1133e 11340 11345-11346 11349-1134a 1134d-1134f 11351-1135c
11364-1137f 1138a 1138c-1138d 1138f 113b6 113b8 113bb-113c9 113cb 113ce-113d0 113d2 113d6 113d9-113ff 11438-1143f
11442-11444 11446 1145c 1145e 11462-1147f 114b0 114b3-114b8 114ba 114bd 114bf-114c0 114c2-114c3 114c8-114cf
114da-1157f 115af 115b2-115b7 115bc-115bd 115bf-115c0 115dc-115ff 11633-1163a 1163d 1163f-11640 11645-1164f
1165a-1165f 1166d-1167f 116ab 116ad 116b0-116b7 116ba-116bf 116ca-116cf 116e4-116ff 1171b-1171d 1171f 11722-11725
11727-1172f 11747-117ff 1182f-11837 11839-1183a 1183c-1189f 118f3-118fe 11907-11908 1190a-1190b 11914 11917 11930
11936 11939-1193e 11943 11947-1194f 1195a-1199f 119a8-119a9 119d4-119db 119e0 119e5-119ff 11a01-11a0a 11a33-11a38
11a3b-11a3e 11a47-11a4f 11a51-11a56 11a59-11a5b 11a8a-11a96 11a98-11a99 11aa3-11aaf 11af9-11aff 11b0a-11bbf
11be2-11bef 11bfa-11bff 11c09 11c30-11c3d 11c3f 11c46-11c4f 11c6d-11c6f 11c90-11ca8 11caa-11cb0 11cb2-11cb3
11cb5-11cff 11d07 11d0a 11d31-11d45 11d47-11d4f 11d5a-11d5f 11d66 11d69 11d8f-11d92 11d95 11d97 11d99-11d9f
11daa-11edf 11ef3-11ef4 11ef9-11f01 11f11 11f36-11f3d 11f40-11f42 11f5a-11faf 11fb1-11fbf 11ff2-11ffe 1239a-123ff
1246f 12475-1247f 12544-12f8f 12ff3-12fff 13430-13440 13447-1345f 143fb-143ff 14647-160ff 1611e-16129 1612d-1612f
1613a-167ff 16a39-16a3f 16a5f 16a6a-16a6d 16abf 16aca-16acf 16aee-16af4 16af6-16aff 16b30-16b36 16b46-16b4f 16b5a
16b62 16b78-16b7c 16b90-16d3f 16d7a-16e3f 16e9b-16eff 16f4b-16f4f 16f88-16f92 16fa0-16fdf 16fe4-16fff 187f8-187ff
18cd6-18cfe 18d09-1afef 1aff4 1affc 1afff 1b123-1b131 1b133-1b14f 1b153-1b154 1b156-1b163 1b168-1b16f 1b2fc-1bbff
1bc6b-1bc6f 1bc7d-1bc7f 1bc89-1bc8f 1bc9a-1bc9b 1bc9d-1bc9e 1bca0-1cbff 1ccfa-1ccff 1ceb4-1cf4f 1cfc4-1cfff
1d0f6-1d0ff 1d127-1d128 1d165-1d169 1d16d-1d182 1d185-1d18b 1d1aa-1d1ad 1d1eb-1d1ff 1d242-1d244 1d246-1d2bf
1d2d4-1d2df 1d2f4-1d2ff 1d357-1d35f 1d379-1d3ff 1d455 1d49d 1d4a0-1d4a1 1d4a3-1d4a4 1d4a7-1d4a8 1d4ad 1d4ba 1d4bc
1d4c4 1d506 1d50b-1d50c 1d515 1d51d 1d53a 1d53f 1d545 1d547-1d549 1d551 1d6a6-1d6a7 1d7cc-1d7cd 1da00-1da36
1da3b-1da6c 1da75 1da84 1da8c-1deff 1df1f-1df24 1df2b-1e02f 1e06e-1e0ff 1e12d-1e136 1e13e-1e13f 1e14a-1e14d
1e150-1e28f 1e2ae-1e2bf 1e2ec-1e2ef 1e2fa-1e2fe 1e300-1e4cf 1e4ec-1e4ef 1e4fa-1e5cf 1e5ee-1e5ef 1e5fb-1e5fe
1e600-1e7df 1e7e7 1e7ec 1e7ef 1e7ff 1e8c5-1e8c6 1e8d0-1e8ff 1e944-1e94a 1e94c-1e94f 1e95a-1e95d 1e960-1ec70
1ecb5-1ed00 1ed3e-1edff 1ee04 1ee20 1ee23 1ee25-1ee26 1ee28 1ee33 1ee38 1ee3a 1ee3c-1ee41 1ee43-1ee46 1ee48 1ee4a
1ee4c 1ee50 1ee53 1ee55-1ee56 1ee58 1ee5a 1ee5c 1ee5e 1ee60 1ee63 1ee65-1ee66 1ee6b 1ee73 1ee78 1ee7d 1ee7f 1ee8a
1ee9c-1eea0 1eea4 1eeaa 1eebc-1eeef 1eef2-1efff 1f02c-1f02f 1f094-1f09f 1f0af-1f0b0 1f0c0 1f0d0 1f0f6-1f0ff
1f1ae-1f1e5 1f203-1f20f 1f23c-1f23f 1f249-1f24f 1f252-1f25f 1f266-1f2ff 1f6d8-1f6db 1f6ed-1f6ef 1f6fd-1f6ff
1f777-1f77a 1f7da-1f7df 1f7ec-1f7ef 1f7f1-1f7ff 1f80c-1f80f 1f848-1f84f 1f85a-1f85f 1f888-1f88f 1f8ae-1f8af
1f8bc-1f8bf 1f8c2-1f8ff 1fa54-1fa5f 1fa6e-1fa6f 1fa7d-1fa7f 1fa8a-1fa8e 1fac7-1facd 1fadd-1fade 1faea-1faef
1faf9-1faff 1fb93 1fbfa-1ffff 2a6e0-2a6ff 2b73a-2b73f 2b81e-2b81f 2cea2-2ceaf 2ebe1-2ebef 2ee5e-2f7ff 2fa1e-2ffff
3134b-3134f 323b0-10ffff
"""


def debug_ranges(table):
    """The first and last code point of each range `table` lists."""
    ranges = []
    for text in table.split():
        first, _, last = text.partition("-")
        ranges.append((int(first, 16), int(last or first, 16)))
    return ranges


RUST_DEBUG_RANGES = debug_ranges(RUST_DEBUG_ESCAPES)


def rust_escapes(char):
    """Whether Rust's `Debug` writes `char` as a `\\u` escape; a surrogate,
    which no Rust string holds, is escaped too."""
    code = ord(char)
    index = bisect.bisect_right(RUST_DEBUG_RANGES, (code, 0x10FFFF)) - 1
    return 0xD800 <= code <= 0xDFFF or (index >= 0 and code <= RUST_DEBUG_RANGES[index][1])


def debug_text(text):
    """A string as Rust's `Debug` quotes it (`RUST_DEBUG_ESCAPES`)."""
    escapes = {"\t": "\\t", "\r": "\\r", "\n": "\\n", "\\": "\\\\", "\"": "\\\"", "\0": "\\0"}
    return '"' + "".join(escapes.get(char) or (f"\\u{{{ord(char):x}}}" if rust_escapes(char) else char)
                         for char in text) + '"'


def invalid_type(value, expected):
    """serde's `invalid type` refusal, the value as serde_json spells it."""
    if value is None:
        unexpected = "null"
    elif isinstance(value, bool):
        unexpected = f"boolean `{'true' if value else 'false'}`"
    elif isinstance(value, int) and (value in I64 or value in U64):
        unexpected = f"integer `{value}`"
    elif isinstance(value, (int, float)):
        unexpected = f"floating point `{float_text(float(value))}`"
    elif isinstance(value, str):
        unexpected = f"string {debug_text(value)}"
    elif isinstance(value, list):
        unexpected = "sequence"
    else:
        unexpected = "map"
    return SchemaError(f"invalid type: {unexpected}, expected {expected}")


def unknown_variant(name, variants):
    return SchemaError(f"unknown variant `{name}`, expected one of {', '.join(f'`{variant}`' for variant in variants)}")


def schema_variant(value, variants):
    """A unit enum value: a variant name, or a map of one variant name to
    null."""
    if isinstance(value, str):
        if value not in variants:
            raise unknown_variant(value, variants)
        return value
    if isinstance(value, dict):
        if len(value) != 1:
            raise SchemaError("invalid value: map, expected map with a single key")
        [(name, unit)] = value.items()
        name = schema_variant(name, variants)
        if unit is not None:
            raise invalid_type(unit, "unit")
        return name
    raise invalid_type(value, "string or map")


def schema_string(value):
    if not isinstance(value, str):
        raise invalid_type(value, "a string")
    return value


def schema_sequence(value, read):
    if not isinstance(value, list):
        raise invalid_type(value, "a sequence")
    return [read(item) for item in value]


def schema_nodes(value):
    if not isinstance(value, dict):
        raise invalid_type(value, "a map")
    return [(key, read_schema_node(item)) for key, item in value.items()]


def schema_number(value, kind):
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise invalid_type(value, "a JSON number")
    elif isinstance(value, int) and not isinstance(value, bool) and value not in I64 and value in U64:
        raise SchemaError(f"invalid value: integer `{value}`, expected i64")
    elif isinstance(value, bool) or not isinstance(value, int) or value not in I64:
        raise invalid_type(value, "i64")
    return value


def labeled(name, read, value):
    try:
        return read(value)
    except SchemaError as error:
        raise SchemaError(f"field `{name}`: {error}") from None


def optional(value, read):
    return None if value is None else read(value)


def read_schema_node(value):
    """`JsonSchema`: a `type` array, a single type, an `anyOf` list or a
    `$ref`, by the field present, as ("types" | "type" | "anyOf" | "ref",
    ...)."""
    if not isinstance(value, dict):
        raise SchemaError(NO_NODE)
    description = lambda: labeled("description", lambda item: optional(item, schema_string), value.get("description"))
    if isinstance(value.get("type"), list):
        types = labeled("type", lambda item: schema_sequence(item, lambda kind: schema_variant(kind, SIMPLE_KINDS)),
                        value["type"])
        return ("types", types, description())
    if "type" in value:
        return ("type", read_typed_node(value))
    if "anyOf" in value:
        return ("anyOf", labeled("anyOf", lambda item: schema_sequence(item, read_typed_node), value["anyOf"]),
                description())
    if "$ref" in value:
        return ("ref", labeled("$ref", schema_string, value["$ref"]))
    raise SchemaError(NO_NODE)


def schema_tag(tag):
    if not isinstance(tag, str):
        raise invalid_type(tag, "variant identifier")
    if tag not in SCHEMA_KINDS:
        raise unknown_variant(tag, SCHEMA_KINDS)
    return tag


def read_typed_node(value):
    """`JsonSchemaType`, tagged by `type`: the tag first, then the type's
    own fields in their arrival order, every other key dropped."""
    if isinstance(value, list):
        if not value:
            raise SchemaError("missing field `type`")
        schema_tag(value[0])
        raise invalid_type(value, "internally tagged enum JsonSchemaType")
    if not isinstance(value, dict):
        raise invalid_type(value, "internally tagged enum JsonSchemaType")
    if "type" not in value:
        raise SchemaError("missing field `type`")
    kind = schema_tag(value["type"])
    node = {"kind": kind, "description": None, "properties": None, "$def": None, "additionalProperties": None,
            "required": None, "keywords": {}, "items": None}
    items = ()
    for key, item in value.items():
        if key == "description":
            node["description"] = optional(item, schema_string)
        elif kind == "object" and key in ("properties", "$def"):
            node[key] = optional(item, schema_nodes)
        elif kind == "object" and key == "additionalProperties":
            if item is not None and not isinstance(item, bool):
                raise invalid_type(item, "a boolean")
            node[key] = item
        elif kind == "object" and key == "required":
            node[key] = optional(item, lambda names: schema_sequence(names, schema_string))
        elif kind == "array" and key == "items":
            items = (read_schema_node(item),)
        elif item is None:
            continue
        elif kind == "string" and key in STRING_KEYWORDS:
            node["keywords"][key] = (schema_string(item) if key == "pattern" else
                                     schema_variant(item, STRING_FORMATS) if key == "format" else
                                     schema_sequence(item, schema_string))
        elif kind in ("number", "integer") and key in NUMERIC_KEYWORDS:
            node["keywords"][key] = schema_number(item, kind)
    if kind == "array":
        if not items:
            raise SchemaError("missing field `items`")
        node["items"] = items[0]
    order = (*STRING_KEYWORDS, *NUMERIC_KEYWORDS)
    node["keywords"] = dict(sorted(node["keywords"].items(), key=lambda entry: order.index(entry[0])))
    return node


def normalize_schema_node(node, allow_empty_object):
    """`JsonSchema::normalize` (json_schema.rs lines 158-166)."""
    if node[0] == "type":
        normalize_typed_node(node[1], allow_empty_object)
    elif node[0] == "anyOf":
        for item in node[1]:
            normalize_typed_node(item, allow_empty_object)


def normalize_typed_node(node, allow_empty_object):
    """`JsonSchemaType::normalize` (json_schema.rs lines 188-253)."""
    if node["kind"] == "array":
        normalize_schema_node(node["items"], False)
    if node["kind"] != "object":
        return
    properties = node["properties"] or []
    if not allow_empty_object and not properties:
        raise SchemaError("An object with no properties is not allowed.")
    for _, value in properties:
        normalize_schema_node(value, False)
    for _, value in node["$def"] or []:
        normalize_schema_node(value, False)
    names = [name for name, _ in properties]
    if node["required"] is not None:
        seen = set()
        for key in node["required"]:
            if key not in names:
                raise SchemaError(f"Required property '{key}' not found in object")
            if key in seen:
                raise SchemaError(f"Duplicate required property '{key}'")
            seen.add(key)
        if len(seen) != len(names):
            raise SchemaError("Required properties must match all properties in the object")
    if node["additionalProperties"] is True:
        raise SchemaError("additionalProperties should not be true")
    node.update(properties=properties, required=names, additionalProperties=False)


def write_schema_node(node):
    """The node as the recipe serializes it."""
    if node[0] == "type":
        return write_typed_node(node[1])
    if node[0] == "ref":
        return {"$ref": node[1]}
    written = {"type": node[1]} if node[0] == "types" else {"anyOf": [write_typed_node(item) for item in node[1]]}
    if node[2] is not None:
        written["description"] = node[2]
    return written


def write_typed_node(node):
    written = {"type": node["kind"]}
    description = {} if node["description"] is None else {"description": node["description"]}
    if node["kind"] == "object":
        written["properties"] = (None if node["properties"] is None else
                                 {key: write_schema_node(value) for key, value in node["properties"]})
        written["additionalProperties"] = node["additionalProperties"]
        written["required"] = node["required"]
        written.update(description)
        if node["$def"] is not None:
            written["$def"] = {key: write_schema_node(value) for key, value in node["$def"]}
    elif node["kind"] == "array":
        written["items"] = write_schema_node(node["items"])
        written.update(description)
    else:
        written.update(description)
        written.update(node["keywords"])
    return written


def normalize_strict_schema(schema, allow_empty_object, read_error, top_level_error):
    """`normalize_strict_schema` (json_schema.rs lines 273-287)."""
    try:
        node = read_schema_node(schema)
    except SchemaError as error:
        raise SchemaError(f"{read_error}{error}") from None
    normalize_schema_node(node, allow_empty_object)
    if node[0] != "type" or node[1]["kind"] != "object":
        raise SchemaError(top_level_error)
    return write_schema_node(node)


def strict_tool_parameters(parameters):
    """A strict tool's parameters normalized, a missing schema being the
    empty object (validation.rs lines 214-225)."""
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "additionalProperties": False, "required": []}
    return normalize_strict_schema(parameters, True, "Invalid tool parameters schema : ",
                                   "The top level of the function parameters schema must be an object.")


def strict_response_schema(schema):
    """A strict json_schema's schema checked as the recipe checks it
    (validation.rs lines 249-260)."""
    return normalize_strict_schema(schema, False, "Invalid json schema: ",
                                   "The top level of the json schema must be an object.")


# The format's refusals of tools and response formats deepseek-recipe 0.2.0
# refuses (output_constraint.rs lines 34-46), and of a json_schema without
# the schema its request schema requires.
STRICT_WITH_NON_STRICT = "Cannot use strict tool with non-strict tool in the same request"
FORMAT_WITH_STRICT_TOOLS = "response_format should not be used with tools"
SCHEMA_REQUIRED = "response_format.json_schema.schema is required"


def recipe_format(response_format):
    """What deepseek-recipe 0.2.0 renders for a response format (dsv41.rs
    lines 252-256): json_object as `{"type": "json_object"}`, json_schema
    as its schema, which a strict one must normalize (`strict_response_schema`),
    and nothing for text."""
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind != "json_schema":
        return None
    fields = response_format.get("json_schema")
    fields = fields if isinstance(fields, dict) else {}
    if fields.get("schema") is None:
        raise ValueError(SCHEMA_REQUIRED)
    if fields.get("strict") is True:
        strict_response_schema(fields["schema"])
    return fields["schema"]


# The text that starts a system message deepseek-recipe strips
# (chat_completion/request/convert.rs lines 373-394), and the whitespace
# Rust's `trim_start` strips after it.
BILLING_HEADER = "x-anthropic-billing-header"
RUST_WHITESPACE = ("\t\n\x0b\x0c\r \x85\xa0\u1680" + "".join(map(chr, range(0x2000, 0x200b)))
                   + "\u2028\u2029\u202f\u205f\u3000")


def strip_billing_header(text):
    """System text without a leading billing header: the text from the first
    newline or `You are`, whichever comes first, without leading whitespace,
    or nothing without either."""
    if not text.startswith(BILLING_HEADER):
        return text
    starts = [index for index in (text.find("\n"), text.find("You are")) if index >= 0]
    return text[min(starts):].lstrip(RUST_WHITESPACE) if starts else ""


def system_text(message):
    """A system message's text, its parts joined as the encoder joins them
    (lines 387-393)."""
    content = message.get("content")
    if isinstance(content, list):
        content = "\n\n".join(part.get("text", "") for part in content
                               if isinstance(part, dict) and part.get("type") == "text")
    return content or ""


def recipe_message(message):
    """One HF message as deepseek-recipe maps it for the encoder: a developer
    message is a system message, as the recipe maps a Responses developer
    message for V4.1 (responses/request/convert.rs lines 421-446), and a
    system message's billing header is removed."""
    if message.get("role") not in ("system", "developer"):
        return message
    message = {**message, "role": "system"}
    text = system_text(message)
    if strip_billing_header(text) != text:
        message["content"] = strip_billing_header(text)
    return message


def empty_system(message):
    """Whether a message is a system message deepseek-recipe drops: its
    text, parts joined, is empty (convert.rs lines 428-440)."""
    return message.get("role") == "system" and not system_text(message)


def encoder_render(namespace, messages, tools, add_generation_prompt, continue_final_message, keywords,
                   response_format=None, tool_choice=None):
    """The prompt `encode_messages` (lines 810-844) writes for one HF input,
    with the request mapped into messages as deepseek-recipe maps it (plan
    V4; `mapping` in frontend-divergences.json) and the served format
    serves it:
    - the mode from the top-level thinking object, else the top-level
      effort (`none` disables), as deepseek-recipe resolves it (convert.rs
      lines 81-98) and the HF input carries it; the thinking keywords and
      the effort in `chat_template_kwargs` are ignored, as the recipe
      ignores that object; the effort, with a protocol alias read as the
      effort it names, applies in thinking mode only, and an enabled object
      over an effort of `none` thinks at the default effort;
    - `preserve_thinking` turns off `drop_thinking`;
    - a developer message is a system message, a system message's billing
      header is removed, and a system message left empty is dropped;
    - a tool choice of none renders without the offered tools, a named one
      with only its tool, and a required or named one ends the prompt with
      the calls block's opening and is refused in thinking mode, as the
      recipe maps it (convert.rs lines 155-215, v4/mod.rs lines 261-266);
      without offered tools any choice but none is auto (convert.rs lines
      172-174);
    - each tool left is the recipe's function object, a strict one's
      parameters normalized; strict and non-strict tools together, and a
      response format beside strict tools, are refused (recipe 0.2.0);
    - the tools and the response format go to the leading system message,
      an empty one when the request has none (README: tools are defined on
      the system message) and they render something; the response format
      renders as recipe 0.2.0 renders it (`recipe_format`), and the encoder
      renders nothing for one Python reads as false;
    - a continued final assistant turn renders without its end token
      (`wo_eos`); the encoder writes the generation header itself, so
      `add_generation_prompt` changes nothing."""
    mode = "thinking" if thinking(keywords) else "chat"
    effort = keywords.get("reasoning_effort") if mode == "thinking" else None
    effort = EFFORT_ALIASES.get(effort, effort) if isinstance(effort, str) else effort
    messages = [encoder_message(message) for message in map(recipe_message, messages) if not empty_system(message)]
    tools = [recipe_tool(tool, namespace) for tool in tools or []]
    named = tool_choice.get("function", {}).get("name") if isinstance(tool_choice, dict) else None
    forced = bool(tools) and (tool_choice == "required" or named is not None)
    if tool_choice == "none":
        tools = []
    elif named is not None:
        tools = [tool for tool in tools if tool["name"] == named]
    functions = [recipe_function(tool) for tool in tools]
    rendered_format = recipe_format(response_format)
    if any(tool["strict"] for tool in tools):
        if not all(tool["strict"] for tool in tools):
            raise ValueError(STRICT_WITH_NON_STRICT)
        if response_format is not None:
            raise ValueError(FORMAT_WITH_STRICT_TOOLS)
    if forced and mode == "thinking":
        raise ValueError("thinking mode does not support a required or named tool_choice")
    if continue_final_message:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("continue_final_message requires a final assistant message")
        messages[-1]["wo_eos"] = True
    rendered_format = rendered_format if rendered_format else None
    if functions or rendered_format is not None:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if functions:
            if messages[0].get("tools"):
                raise ValueError("tools are offered both by the request and by its system message")
            messages[0]["tools"] = functions
        if rendered_format is not None:
            messages[0]["response_format"] = rendered_format
    prompt = namespace["encode_messages"](
        messages, thinking_mode=mode, drop_thinking=not keywords.get("preserve_thinking", False),
        reasoning_effort=effort)
    return prompt + FORCED_PREFILL if forced else prompt


def turn_arguments(text):
    """A call's arguments as the encoder writes them (`encode_arguments_to_dsml`,
    lines 524-555): the text read as JSON up to twice, and an object
    whose every value is written as itself, else one string parameter
    named `arguments` holding the text."""
    value = text
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value if isinstance(value, dict) else {"arguments": text}


def expected_turn(message, context):
    """What the vendor's `parse_message_from_completion_text` (lines
    946-990) reads back from a generated assistant turn: the reasoning in
    thinking mode (the prompt leaves `<think>` open), the content, and each
    call's arguments as `decode_dsml_to_arguments` (lines 156-177) writes
    them from the DSML the encoder wrote, the object `json.dumps` writes.
    None for a turn whose text holds a marker the vendor parser refuses."""
    reasoning = message.get("reasoning_content") if context["prompt"].endswith("<think>") else None
    content = message.get("content")
    if isinstance(content, list):
        content = "\n\n".join(part.get("text", "") for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
    content = content or ""
    reasoning = reasoning or ""
    if any(marker in text for marker in SPECIAL for text in (content, reasoning)):
        return None
    calls = [{"name": call["function"]["name"],
              "arguments": json.dumps(turn_arguments(call["function"]["arguments"]), ensure_ascii=False)}
             for call in message.get("tool_calls") or []]
    return {"reasoning": reasoning, "content": content, "tool_calls": calls}


RESPONSE_FORMATS = {
    "json-schema": {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city", "days"], "additionalProperties": False}}},
    "json-object": {"type": "json_object"},
    "text": {"type": "text"},
}


def features(declaration):
    return {"response-format": ["absent", *RESPONSE_FORMATS]}


def apply_feature(request, axis, value, rng):
    """Sets the request's `response_format`."""
    if axis != "response-format" or value == "absent":
        return ()
    request["response_format"] = copy.deepcopy(RESPONSE_FORMATS[value])
    return {f"response-format:{value}"}


def user(text):
    return {"role": "user", "content": text}


def call(identifier, name, arguments):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a specific location",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string", "description": "The city name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}},
        "required": ["location"]}}}


def tool_image_cases():
    """Image results and the vendor's source order, including sorting across user parts."""
    def image(data):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}

    def text(value):
        return {"type": "text", "text": value}

    def result(identifier, parts):
        return {"role": "tool", "tool_call_id": identifier, "content": parts}

    initial = [user("Look."), {"role": "assistant", "content": "", "tool_calls": [
        call("a", "look", {}), call("b", "look", {})]}]
    for name, results, order in [
        ("tool-image-single", [result("a", [image("AA==")])], ["AA=="]),
        ("tool-images-several", [result("a", [image("AA=="), image("AQ==")])], ["AA==", "AQ=="]),
        ("tool-images-mixed-text", [result("a", [text("before"), image("AA=="),
            text("between"), image("AQ=="), text("after")])], ["AA==", "AQ=="]),
        ("tool-images-sorted", [result("b", [text("second"), image("AA==")]),
            result("a", [text("first"), image("AQ=="), image("Ag==")])], ["AQ==", "Ag==", "AA=="]),
        ("tool-images-with-user", [result("b", [text("second"), image("AQ==")]),
            user([image("Ag==")]), result("a", [text("first"), image("Aw=="), image("BA=="), text("end")])],
            ["AA==", "Aw==", "BA==", "Ag==", "AQ=="]),
    ]:
        messages = copy.deepcopy(initial)
        if name == "tool-images-with-user":
            messages[0] = user([image("AA==")])
        yield name, {"messages": messages + results, "thinking": {"type": "disabled"}}, [
            f"data:image/png;base64,{data}" for data in order]


def image_format_cases():
    """Images as GIF and WebP data URLs and under other spellings of the JPEG
    type, and file parts carrying an image data URL, in user and tool
    messages, which the encoder renders as any image; and the file parts
    this format refuses."""
    def image(mime, data="AA=="):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}

    def file(data_url, **fields):
        return {"type": "file", "file": {"file_data": data_url, **fields}}

    text = {"type": "text", "text": "Describe."}
    looked = [user("Look."), {"role": "assistant", "content": "", "tool_calls": [call("a", "look", {})]}]
    chat = {"thinking": {"type": "disabled"}}

    def result(parts):
        return {"role": "tool", "tool_call_id": "a", "content": parts}

    yield "image-gif", {"messages": [user([image("image/gif"), text])]}, {"image:gif"}
    yield "image-webp", {"messages": [user([image("image/webp"), text])]}, {"image:webp"}
    yield "image-jpg-spellings", {"messages": [user([image("image/jpg"), image("image/JPG", "AQ=="), text])]}, {
        "image:jpg"}
    yield "tool-images-gif-webp", {"messages": looked + [result([image("image/gif"), image("image/webp", "AQ==")])],
                                   **chat}, {"tool-results:images", "image:gif", "image:webp"}
    yield "file-image", {"messages": [user([file("data:image/png;base64,AA==", filename="a.png"), text])]}, {
        "file:image"}
    yield "tool-file-image", {"messages": looked + [result([file("data:image/webp;base64,AA==")])], **chat}, {
        "tool-results:images", "file:image"}
    yield "file-id", {"messages": [user([{"type": "file", "file": {"file_id": "file-1"}}])]}, {"file:id"}
    yield "file-pdf", {"messages": [user([file("data:application/pdf;base64,AA==")])]}, {"file:non-image"}
    yield "file-flat", {"messages": [user([{"type": "file", "file_data": "data:image/png;base64,AA=="}])]}, {
        "file:flat"}


def cases(declaration):
    for name, request, _ in tool_image_cases():
        yield name, request, {f"fixed:{name}", "tool-results:images"}
    for name, request, tags in image_format_cases():
        yield name, request, {f"fixed:{name}", *tags}
    # The encoder's test pair 1 (encoding/tests/test_input_1.json) as a
    # served request: tools on the leading system message, a call without
    # an id and its result, thinking at the default effort.
    yield "vendor-pair-1", {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        user("What's the weather like in Beijing?"),
        {"role": "assistant", "reasoning_content": "The user wants the weather in Beijing. I should call get_weather.",
         "content": "", "tool_calls": [call("call_0", "get_weather", {"location": "Beijing", "unit": "celsius"})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"},
        {"role": "assistant", "reasoning_content": "Got the weather data.",
         "content": "The weather in Beijing is sunny at 22°C."},
    ], "tools": [WEATHER]}, {"fixed:vendor-pair-1", "ends:assistant", "tools:wrapped"}
    # A later system message counts as a user turn: the header follows it,
    # and reasoning before it drops.
    yield "later-system-message", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "system", "content": "Answer in French from now on."},
    ]}, {"fixed:later-system-message", "system:late"}
    # Consecutive user messages and a result followed by user text join
    # into one user turn (merge_tool_messages, lines 215-226).
    yield "joined-user-turns", {"messages": [
        user("first"), user("second"),
        {"role": "assistant", "content": "", "tool_calls": [call("call_a", "get_weather", {"location": "Paris"})]},
        {"role": "tool", "tool_call_id": "call_a", "content": "18"}, user("And tomorrow?"),
    ], "tools": [WEATHER]}, {"fixed:joined-user-turns", "tools:wrapped"}
    # Reasoning kept by preserve_thinking, and an effort the encoder maps.
    yield "preserved-reasoning-at-max", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, user("q2"),
    ], "chat_template_kwargs": {"preserve_thinking": True}, "reasoning_effort": "max"}, {
        "fixed:preserved-reasoning-at-max", "preserve_thinking:true"}
    # The response format on a request without a system message, and with
    # tools: both go to an inserted empty system message.
    yield "response-format-with-tools", {"messages": [user("Weather in Oslo as JSON.")],
                                         "tools": [WEATHER], "response_format": RESPONSE_FORMATS["json-schema"]}, {
        "fixed:response-format-with-tools", "response-format:json-schema", "tools:wrapped"}
    # Arguments as the encoder writes them: strings raw, other values as
    # Python's JSON, keys in their order.
    yield "dsml-argument-values", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [call("call_v", "f", {
            "s": "a \"quoted\" <tag>", "n": 1.5, "i": -0, "b": True, "z": None, "l": [1, "x"], "o": {"k": "v"}})]},
        {"role": "tool", "tool_call_id": "call_v", "content": "ok"},
    ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, {
        "fixed:dsml-argument-values", "tools:wrapped"}
    # The image placeholder in client text, which the encoder refuses in
    # string content and reasoning (lines 355-367) and in a text block
    # (lines 342-348).
    placeholder = "<｜deepseek_image｜>"
    yield "image-placeholder-in-content", {"messages": [user(f"see {placeholder}")]}, {
        "fixed:image-placeholder-in-content"}
    yield "image-placeholder-in-reasoning", {"messages": [
        user("q"), {"role": "assistant", "content": "a", "reasoning_content": f"r {placeholder}"}, user("q2"),
    ]}, {"fixed:image-placeholder-in-reasoning"}
    yield "image-placeholder-in-text-part", {"messages": [
        {"role": "user", "content": [{"type": "text", "text": f"see {placeholder}"}]}]}, {
        "fixed:image-placeholder-in-text-part", "user:parts"}
    # Arguments that are JSON text of JSON text, which the encoder reads
    # twice (lines 533-544).
    yield "double-encoded-arguments", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_d", "type": "function",
            "function": {"name": "f", "arguments": json.dumps(json.dumps({"x": 1, "y": "z"}))}}]},
        {"role": "tool", "tool_call_id": "call_d", "content": "ok"},
    ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, {
        "fixed:double-encoded-arguments", "tools:wrapped"}
    # List content on a system and an assistant turn: its text parts joined
    # (lines 380-393).
    text = lambda value: {"type": "text", "text": value}
    yield "list-content-on-system-and-assistant", {"messages": [
        {"role": "system", "content": [text("Be brief."), text("Be kind.")]}, user("q1"),
        {"role": "assistant", "content": [text("a"), text("b")]}, user("q2"),
    ]}, {"fixed:list-content-on-system-and-assistant"}
    # A developer message is a system message (`developer-as-system` in
    # frontend-divergences.json): before the last user turn in thinking
    # mode, which the encoder would drop unread as a developer message
    # (lines 756-761); leading, carrying the offered tools and the response
    # format; later, followed by the generation header; and empty, dropped.
    yield "developer-before-last-user", {"messages": [
        {"role": "developer", "content": "Be terse."}, user("q1"),
        {"role": "assistant", "content": "a1"}, user("q2"),
    ]}, {"fixed:developer-before-last-user", "developer:present"}
    yield "developer-message", {"messages": [{"role": "developer", "content": "Answer in JSON."}, user("Oslo?")],
                                "tools": [WEATHER], "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:developer-message", "developer:present", "tools:wrapped", "response-format:json-object"}
    yield "later-developer-message", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1"},
        {"role": "developer", "content": "Answer in French from now on."},
    ]}, {"fixed:later-developer-message", "developer:present"}
    yield "empty-developer-message", {"messages": [{"role": "developer", "content": ""}, user("q")]}, {
        "fixed:empty-developer-message", "developer:present"}
    # Protocol efforts the encoder does not name.
    yield "medium-effort", {"messages": [user("q")], "reasoning_effort": "medium"}, {"fixed:medium-effort"}
    yield "minimal-effort", {"messages": [user("q")], "reasoning_effort": "minimal"}, {"fixed:minimal-effort"}
    yield "xhigh-effort", {"messages": [user("q")], "reasoning_effort": "xhigh"}, {"fixed:xhigh-effort"}
    yield "ultra-effort", {"messages": [user("q")], "reasoning_effort": "ultra"}, {"fixed:ultra-effort"}
    # Argument text the encoder reads itself: spacing kept in a value it
    # reads once, a null, three JSON string layers, and a repeated key,
    # whose last value Python keeps.
    raw = lambda identifier, arguments: {"id": identifier, "type": "function",
                                         "function": {"name": "f", "arguments": arguments}}
    yield "raw-argument-text", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [
            raw("call_l", "[1,  2]"), raw("call_n", "null"),
            raw("call_t", json.dumps(json.dumps(json.dumps({"x": 1})))), raw("call_r", "{\"x\":1,\"y\":3,\"x\":2}")]},
        {"role": "tool", "tool_call_id": "call_l", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_n", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_t", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_r", "content": "ok"},
    ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, {
        "fixed:raw-argument-text", "tools:wrapped"}
    # Mapping rules (`mapping` in frontend-divergences.json). R1: reasoning
    # before the last user turn drops, and a turn without any writes
    # </think> alone, as the encoder renders it.
    yield "earlier-reasoning-dropped", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        user("q2"), {"role": "assistant", "content": "a2"}, user("q3"),
    ]}, {"fixed:earlier-reasoning-dropped"}
    # R2: each offered tool is the recipe's function object: name,
    # description or "", parameters or {}, every other field dropped.
    yield "recipe-tool-schema", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"parameters": {"type": "object"}, "strict": False, "name": "a",
                                          "defer_loading": True, "description": "A."}},
        {"type": "function", "function": {"name": "b", "description": None, "parameters": None}},
        {"type": "function", "function": {"name": "c"}},
        {"type": "function", "function": {"name": "d", "function": {"name": "g"}}}]}, {
        "fixed:recipe-tool-schema", "tools:wrapped"}
    # R2: a strict tool's parameters are the recipe's normalized schema:
    # every property required, additionalProperties false, keywords outside
    # its type set dropped, and no parameters the empty object.
    lookup = {"type": "function", "function": {"name": "lookup", "description": "Look a record up.", "strict": True,
                                               "parameters": {
        "type": "object", "title": "Lookup", "$defs": {"unused": {"type": "string"}},
        "properties": {"id": {"type": "integer", "minimum": 1, "description": "The id."},
                       "tags": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 3},
                       "filter": {"type": "object", "properties": {
                           "kind": {"enum": ["a", "b"], "type": "string"},
                           "since": {"anyOf": [{"type": "string", "format": "uuid"}, {"type": "null"}]}}}}}}}
    clock = {"type": "function", "function": {"name": "now", "strict": True}}
    yield "strict-tool-schemas", {"messages": [user("Look up record 7.")], "tools": [lookup, clock]}, {
        "fixed:strict-tool-schemas", "tools:wrapped"}
    # R2: a strict schema the recipe refuses, a non-strict tool's parameters
    # without type object, strict and non-strict tools together, and a
    # response format beside strict tools are refused; a named choice
    # leaves only its tool, so the strict tool it drops no longer counts.
    partial = {"type": "function", "function": {"name": "f", "strict": True, "parameters": {
        "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a"]}}}
    yield "strict-tool-partial-required", {"messages": [user("q")], "tools": [partial]}, {
        "fixed:strict-tool-partial-required", "tools:wrapped"}
    yield "non-object-tool-parameters", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "parameters": {"type": "string"}}}]}, {
        "fixed:non-object-tool-parameters", "tools:wrapped"}
    yield "strict-and-non-strict-tools", {"messages": [user("q")], "tools": [clock, WEATHER]}, {
        "fixed:strict-and-non-strict-tools", "tools:wrapped"}
    yield "strict-tool-with-response-format", {"messages": [user("q")], "tools": [clock],
                                               "response_format": RESPONSE_FORMATS["text"]}, {
        "fixed:strict-tool-with-response-format", "tools:wrapped", "response-format:text"}
    yield "strict-mix-named-choice", {"messages": [user("Weather in Oslo?")], "tools": [clock, WEATHER],
                                      "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                                      "thinking": {"type": "disabled"}}, {
        "fixed:strict-mix-named-choice", "tools:wrapped"}
    # R3: a response format renders as the recipe renders it: text nothing
    # (and needs no system message), json_object its type, json_schema its
    # schema, a strict one normalizing as a strict tool's does. A schema the
    # encoder reads as false renders nothing, and a missing one is refused.
    yield "response-format-text", {"messages": [user("q")], "response_format": RESPONSE_FORMATS["text"]}, {
        "fixed:response-format-text", "response-format:text"}
    yield "response-format-json-object", {"messages": [user("Answer in JSON.")],
                                          "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:response-format-json-object", "response-format:json-object"}
    yield "response-format-json-schema", {"messages": [user("Weather in Oslo as JSON.")],
                                          "response_format": RESPONSE_FORMATS["json-schema"]}, {
        "fixed:response-format-json-schema", "response-format:json-schema"}
    for name, schema in (("empty", {}), ("false", False)):
        yield f"response-format-{name}-schema", {"messages": [user("q")], "response_format": {
            "type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}}, {
            f"fixed:response-format-{name}-schema"}
    yield "response-format-without-schema", {"messages": [user("q")], "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer"}}}, {"fixed:response-format-without-schema"}
    yield "strict-response-format-partial-required", {"messages": [user("q")], "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"]}}}}, {"fixed:strict-response-format-partial-required"}
    # R4: system messages alone render as the encoder renders them: a
    # later system message is followed by the generation header (lines
    # 713, 731-739), so two are served, and an empty one is dropped (R5).
    yield "system-messages-only", {"messages": [
        {"role": "system", "content": "Be brief."}, {"role": "system", "content": "Answer in French."},
    ]}, {"fixed:system-messages-only", "system:late"}
    yield "system-messages-with-empty", {"messages": [
        {"role": "system", "content": "Be brief."}, {"role": "system", "content": "Answer in French."},
        {"role": "system", "content": ""},
    ]}, {"fixed:system-messages-with-empty", "system:late", "system:empty"}
    # R4: a final assistant turn ends with the end-of-sentence token and no
    # header follows, so it is refused unless continued.
    yield "final-assistant-turn", {"messages": [user("q"), {"role": "assistant", "content": "a"}]}, {
        "fixed:final-assistant-turn", "ends:assistant"}
    # A turn whose content holds a special token, which the vendor parser
    # refuses (lines 981-983), so it reads no turn back
    # (`special-token-in-turn`).
    yield "special-token-in-assistant-turn", {"messages": [
        user("q"), {"role": "assistant", "content": "<think>inline plan</think> It is mild."}]}, {
        "fixed:special-token-in-assistant-turn", "ends:assistant", "assistant-content:inline-think"}
    yield "continued-final-assistant-turn", {"messages": [user("q"), {"role": "assistant", "content": "a"}],
                                             "continue_final_message": True, "add_generation_prompt": False}, {
        "fixed:continued-final-assistant-turn", "ends:continue"}
    # R5: an empty system message is dropped, leading, later or as empty
    # parts; the one carrying the offered tools is still inserted.
    yield "empty-system-messages-dropped", {"messages": [
        {"role": "system", "content": ""}, user("q1"), {"role": "assistant", "content": "a1"},
        {"role": "system", "content": [{"type": "text", "text": ""}]}, user("q2"),
    ]}, {"fixed:empty-system-messages-dropped", "system:empty"}
    yield "empty-system-message-with-tools", {"messages": [{"role": "system", "content": ""}, user("q")],
                                              "tools": [WEATHER]}, {
        "fixed:empty-system-message-with-tools", "system:empty", "tools:wrapped"}
    # R5: a leading billing header is removed first, up to the first newline
    # or "You are", and the leading whitespace after it; a system message of
    # the header alone is empty and dropped.
    header = "x-anthropic-billing-header: cc_version=2.1.3; cc_entrypoint=cli;"
    yield "billing-header-stripped", {"messages": [
        {"role": "system", "content": f"{header}\n  You are a coding agent.\nBe brief."}, user("q")]}, {
        "fixed:billing-header-stripped"}
    yield "billing-header-before-you-are", {"messages": [
        {"role": "system", "content": f"{header}You are a coding agent.\nBe brief."}, user("q")]}, {
        "fixed:billing-header-before-you-are"}
    yield "billing-header-in-parts", {"messages": [
        {"role": "system", "content": [text(header), text("You are a coding agent.")]}, user("q")]}, {
        "fixed:billing-header-in-parts"}
    yield "billing-header-alone", {"messages": [{"role": "system", "content": header}, user("q")]}, {
        "fixed:billing-header-alone"}
    # R6: a tool choice of none renders no tools; required and a named
    # choice end the prompt with the calls block's opening (a named one
    # renders only its tool), and are refused in thinking mode.
    chat = {"thinking": {"type": "disabled"}}
    second = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
    yield "tool-choice-none", {"messages": [user("q")], "tools": [WEATHER], "tool_choice": "none"}, {
        "fixed:tool-choice-none", "tools:wrapped"}
    yield "tool-choice-required", {"messages": [user("Weather in Oslo?")], "tools": [WEATHER, second],
                                   "tool_choice": "required", **chat}, {
        "fixed:tool-choice-required", "tools:wrapped"}
    yield "tool-choice-named", {"messages": [user("Weather in Oslo?")], "tools": [WEATHER, second],
                                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, **chat}, {
        "fixed:tool-choice-named", "tools:wrapped"}
    yield "tool-choice-required-thinking", {"messages": [user("Weather in Oslo?")], "tools": [WEATHER],
                                            "tool_choice": "required"}, {
        "fixed:tool-choice-required-thinking", "tools:wrapped"}
    # Without offered tools, required and a named choice are auto, in
    # thinking mode too.
    yield "tool-choice-required-without-tools", {"messages": [user("q")], "tools": [], "tool_choice": "required"}, {
        "fixed:tool-choice-required-without-tools"}
    yield "tool-choice-named-without-tools", {"messages": [user("q")], "tool_choice": {
        "type": "function", "function": {"name": "get_weather"}}}, {"fixed:tool-choice-named-without-tools"}
    # The thinking switch: the top-level object, which overrides the
    # effort, then the effort; the template keywords are ignored.
    yield "thinking-object-disabled", {"messages": [user("q")], "thinking": {"type": "disabled"},
                                       "chat_template_kwargs": {"enable_thinking": True, "thinking": True}}, {
        "fixed:thinking-object-disabled", "thinking:object=disabled"}
    yield "thinking-object-adaptive-over-none", {"messages": [user("q")], "thinking": {"type": "adaptive"},
                                                 "reasoning_effort": "none"}, {
        "fixed:thinking-object-adaptive-over-none", "thinking:object=adaptive"}
    yield "thinking-object-enabled-at-max", {"messages": [user("q")],
                                             "thinking": {"type": "enabled", "budget_tokens": 1024},
                                             "reasoning_effort": "max"}, {
        "fixed:thinking-object-enabled-at-max", "thinking:object=enabled"}
    yield "template-thinking-ignored", {"messages": [user("q")], "chat_template_kwargs": {
        "enable_thinking": False, "thinking": False, "reasoning_effort": "low"}}, {"fixed:template-thinking-ignored"}
    yield "effort-none-disables", {"messages": [user("q")], "reasoning_effort": "none"}, {
        "fixed:effort-none-disables"}
    # Namespaces (`namespace` in frontend-divergences.json): on a function
    # object, as an object with a description or as a name; on the tool
    # object; spelled in the name; a call by its qualified name in history;
    # and a named choice of a qualified tool.
    namespaced = {"type": "function", "function": {
        "name": "f", "namespace": {"name": "ns", "description": "Tools of ns."},
        "description": "F.", "parameters": {"type": "object"}}}
    yield "namespaced-function-object", {"messages": [user("q")], "tools": [namespaced]}, {
        "fixed:namespaced-function-object", "tools:wrapped"}
    yield "namespace-name-on-function", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "namespace": "ns", "parameters": {"type": "object"}}}]}, {
        "fixed:namespace-name-on-function", "tools:wrapped"}
    yield "namespace-on-tool-object", {"messages": [user("q")], "tools": [
        {"type": "function", "namespace": {"name": "ns", "description": "Tools of ns."},
         "function": {"name": "f", "description": "F.", "parameters": {"type": "object"}}}]}, {
        "fixed:namespace-on-tool-object", "tools:wrapped"}
    yield "qualified-tool-name", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "ns::f", "parameters": {"type": "object"}}}]}, {
        "fixed:qualified-tool-name", "tools:wrapped"}
    yield "qualified-call-in-history", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [call("call_q", "ns::f", {"x": "v"})]},
        {"role": "tool", "tool_call_id": "call_q", "content": "ok"}, user("Again."),
    ], "tools": [namespaced]}, {"fixed:qualified-call-in-history", "tools:wrapped"}
    # Refused by the encoder's assertions and here by name: a namespace
    # other than the one the name spells, and a separator inside a part.
    yield "conflicting-namespace", {"messages": [user("q")], "tools": [
        {"type": "function", "namespace": "b", "function": {"name": "a::f", "parameters": {"type": "object"}}}]}, {
        "fixed:conflicting-namespace", "tools:wrapped"}
    yield "separator-in-namespace", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "namespace": "a::b", "parameters": {"type": "object"}}}]}, {
        "fixed:separator-in-namespace", "tools:wrapped"}
    yield "named-choice-of-qualified-tool", {"messages": [user("q")], "tools": [namespaced, second],
                                             "tool_choice": {"type": "function", "function": {"name": "ns::f"}},
                                             **chat}, {"fixed:named-choice-of-qualified-tool", "tools:wrapped"}
    # A system message's own tools, which the recipe's request messages do
    # not carry and this format refuses.
    yield "system-message-tools", {"messages": [
        {"role": "system", "content": "s", "tools": [WEATHER]}, user("q"),
    ]}, {"fixed:system-message-tools", "message-tools:present"}
