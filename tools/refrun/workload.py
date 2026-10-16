"""Reference workloads: the sequences one reference run scores.

A workload file is JSON:

    {"schema": "chopinfer.refrun.workload.v1", "model": "qwen38",
     "sequences": [{"name": "...", "category": "...", "token_ids": [...], "score_from": 1,
                    "answer": {"policy": "forced", "ids": [...]},
                    "media": [{"kind": "image",
                               "tensor": {"path": "a.bin", "dtype": "f32", "shape": [...]},
                               "grid": {"path": "a-grid.bin", "dtype": "i64", "shape": [1, 3]},
                               "processor": {"code_sha256": "...", "config_sha256": "..."},
                               "raw_sha256": "..."}]}]}

`token_ids` is the prompt. A forced answer is appended to it and scored with
it; a greedy answer is generated from the prompt (`max_new_tokens`, ending at
any of `stop_ids`) and then forced in every mode. Position p >= score_from is
scored by the logits at p - 1. Media tensors are raw little-endian files
relative to the workload file, already produced by the model's processor;
the processor itself never runs here. `raw_sha256` (the source image or
audio bytes) is provenance only.

This module does not import torch.
"""

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "chopinfer.refrun.workload.v1"
WIDTH = {"f64": 8, "f32": 4, "bf16": 2, "f16": 2, "i64": 8, "i32": 4, "u8": 1}
POLICIES = ("forced", "greedy")


@dataclass(frozen=True)
class Tensor:
    """A processor output: raw little-endian bytes of `shape` in `dtype`."""
    dtype: str
    shape: tuple
    data: bytes = field(repr=False)

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class Media:
    kind: str
    tensor: Tensor
    processor: dict
    grid: Tensor = None
    raw_sha256: str = None


@dataclass(frozen=True)
class Answer:
    policy: str
    ids: tuple = ()
    max_new_tokens: int = None
    stop_ids: tuple = ()


@dataclass(frozen=True)
class Sequence:
    name: str
    category: str
    token_ids: tuple
    score_from: int
    answer: Answer = Answer("forced")
    media: tuple = ()

    @property
    def forced_ids(self):
        """Prompt plus forced answer: the IDs one teacher-forced pass reads."""
        if self.answer.policy != "forced":
            raise ValueError(f"{self.name}: a greedy answer must be generated before forcing")
        return self.token_ids + self.answer.ids

    def forced(self, answer_ids):
        """This sequence with a generated answer forced, as every mode then scores it."""
        return replace(self, answer=Answer("forced", tuple(answer_ids)))


def ids(value, label):
    if not isinstance(value, list) or any(type(i) is not int or i < 0 for i in value):
        raise ValueError(f"{label} must be a list of non-negative token IDs")
    return tuple(value)


def tensor(spec, root, label):
    if set(spec) != {"path", "dtype", "shape"} or spec["dtype"] not in WIDTH:
        raise ValueError(f"{label}: tensor needs path, dtype in {sorted(WIDTH)} and shape")
    shape = spec["shape"]
    if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f"{label}: invalid shape")
    path = Path(spec["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}: tensor path must be relative to the workload")
    data = (root / path).read_bytes()
    if len(data) != math.prod(shape) * WIDTH[spec["dtype"]]:
        raise ValueError(f"{label}: {len(data)} bytes do not hold {spec['dtype']} {shape}")
    return Tensor(spec["dtype"], tuple(shape), data)


def media(spec, root, label):
    allowed = {"kind", "tensor", "grid", "processor", "raw_sha256"}
    if not isinstance(spec, dict) or not {"kind", "tensor", "processor"} <= set(spec) <= allowed:
        raise ValueError(f"{label}: media needs kind, tensor and processor")
    processor = spec["processor"]
    if set(processor) != {"code_sha256", "config_sha256"}:
        raise ValueError(f"{label}: processor identity needs code_sha256 and config_sha256")
    grid = tensor(spec["grid"], root, label + " grid") if "grid" in spec else None
    return Media(spec["kind"], tensor(spec["tensor"], root, label), dict(processor), grid,
                 spec.get("raw_sha256"))


def answer(spec, label):
    policy = spec.get("policy")
    if policy == "forced" and set(spec) == {"policy", "ids"}:
        return Answer("forced", ids(spec["ids"], label + " answer"))
    if policy == "greedy" and set(spec) == {"policy", "max_new_tokens", "stop_ids"}:
        count = spec["max_new_tokens"]
        if type(count) is not int or count <= 0:
            raise ValueError(f"{label}: max_new_tokens must be positive")
        return Answer("greedy", (), count, ids(spec["stop_ids"], label + " stop_ids"))
    raise ValueError(f"{label}: answer must be forced {{ids}} or greedy {{max_new_tokens, stop_ids}}")


def sequence(spec, root):
    allowed = {"name", "category", "token_ids", "score_from", "answer", "media"}
    if not isinstance(spec, dict) or not {"name", "category", "token_ids", "score_from"} <= set(spec) \
            or not set(spec) <= allowed:
        raise ValueError(f"sequence fields must be within {sorted(allowed)}")
    name = spec["name"]
    if not isinstance(name, str) or not name or not isinstance(spec["category"], str):
        raise ValueError("sequence name and category must be strings")
    prompt = ids(spec["token_ids"], name)
    reply = answer(spec.get("answer", {"policy": "forced", "ids": []}), name)
    score_from = spec["score_from"]
    end = len(prompt) + (len(reply.ids) if reply.policy == "forced" else 1)
    if type(score_from) is not int or not 1 <= score_from < end:
        raise ValueError(f"{name}: score_from must leave at least one scored position")
    items = tuple(media(m, root, f"{name} media {i}") for i, m in enumerate(spec.get("media", [])))
    return Sequence(name, spec["category"], prompt, score_from, reply, items)


def load(path):
    """The workload's model name and its sequences, in file order."""
    path = Path(path)
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema") != SCHEMA \
            or set(document) != {"schema", "model", "sequences"}:
        raise ValueError(f"workload must be {SCHEMA} with model and sequences")
    sequences = [sequence(spec, path.parent) for spec in document["sequences"]]
    if not sequences or len({s.name for s in sequences}) != len(sequences):
        raise ValueError("workload needs at least one sequence and unique names")
    return document["model"], sequences
