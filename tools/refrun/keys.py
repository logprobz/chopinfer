"""Reference cache keys: one per (sequence, mode, output spec).

key_sha256 = sha256(canonical(key document)), with `canonical` from
tools/model_sdk/diagnostic.py. The document holds only semantic inputs:
the checkpoint source lock and the files read, the sequence (token IDs,
score_from, answer policy, media tensors and processor identity), the
upstream reference code (the files an adapter names, plus a digest of every
Python file of the installed transformers, which covers the cache, masking,
RoPE, activation and attention helpers the modeling code imports), the
adapter and the runner files that execute a reference, the mode policy, the
environment and the outputs. Raw media bytes, GPU serials, driver version,
timings, git HEAD, the device each unit was placed on, and runner files
that do not take part in computing a score (stamps, which only refuse
changed files; taps, whose hooks only copy values; keys; calibration) are
provenance and never keyed.

This module does not import torch; `environment()` imports it when called.
"""

import hashlib
import importlib.metadata
from pathlib import Path

from tools.model_sdk.diagnostic import canonical, decode_json

SCHEMA = "chopinfer.refcache.key.v1"
ROOT = Path(__file__).resolve().parents[2]
# Runner files whose content can change a score. adapter.py holds the
# default forward and decode calls; place.py assembles every stored weight
# (Store) and gathers host table rows (HostRows); run.py configures each
# mode's dequantization and attaches the device hops; workload.py assembles
# the IDs a forced pass reads (Sequence.forced_ids).
NUMERICS = ("tools/refrun/adapter.py", "tools/refrun/modes.py", "tools/refrun/ops.py",
            "tools/refrun/place.py", "tools/refrun/quant.py", "tools/refrun/run.py", "tools/refrun/score.py",
            "tools/refrun/workload.py")
# The files that compute a generic encoder reference (encoder.py): its own
# modules, the mode flags and the loader. Encoder keys hold them with the
# declaration and a model's variant file. They are not in `NUMERICS`: no
# text reference executes them, and adding them there would change every
# text key and stale the fixtures bound to those keys.
ENCODER_NUMERICS = ("tools/refrun/encoder.py", "tools/refrun/encoder_ops.py", "tools/refrun/modes.py",
                    "tools/refrun/place.py")
ENCODER_SCHEMA = "chopinfer.refcache.encoder-key.v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    return sha256(canonical(list(ids)))


def checkpoint(lock, files_used):
    """`lock` is the parsed models/<m>/checkpoint-files.json mapping of the
    checkpoint read (for a lock with variants, the variant's mapping)."""
    return {"source_lock_sha256": sha256(canonical(lock)), "files_used": sorted(set(files_used))}


def read_lock(path):
    return decode_json(Path(path).read_bytes())


def tensor(item):
    return {"sha256": item.sha256, "dtype": item.dtype, "shape": list(item.shape)}


def sequence(record):
    """The semantic identity of a workload sequence (workload.Sequence)."""
    answer = record.answer
    forced = answer.policy == "forced"
    return {
        "token_ids_sha256": ids_sha256(record.token_ids),
        "score_from": record.score_from,
        "answer": {"policy": answer.policy,
                   "ids_sha256": ids_sha256(answer.ids) if forced else None,
                   "max_new_tokens": None if forced else answer.max_new_tokens,
                   "stop_ids": None if forced else sorted(answer.stop_ids)},
        "media": [{"kind": item.kind, "tensor": tensor(item.tensor),
                   "grid": tensor(item.grid) if item.grid is not None else None,
                   "processor": {"code_sha256": item.processor["code_sha256"],
                                 "config_sha256": item.processor["config_sha256"]}}
                  for item in record.media],
    }


def package_sha256(root):
    """The content of a Python package: sha256 of the sorted (relative path,
    sha256) pairs of every .py file under `root`, bytecode caches left out."""
    root = Path(root)
    files = sorted((path.relative_to(root).as_posix(), file_sha256(path)) for path in root.rglob("*.py")
                   if "__pycache__" not in path.parts)
    return sha256(canonical([list(item) for item in files]))


def installed_transformers():
    """(version, package_sha256) of the installed transformers."""
    import importlib.util
    spec = importlib.util.find_spec("transformers")
    return importlib.metadata.version("transformers"), package_sha256(Path(spec.origin).parent)


def reference_code(files, transformers=None, transformers_sha256=None):
    """`files` maps a label (checkpoint- or package-relative path) to its
    sha256. The installed transformers' version and content digest are
    looked up unless given (tests)."""
    if transformers is None or transformers_sha256 is None:
        transformers, transformers_sha256 = installed_transformers()
    return {"files": dict(sorted(files.items())), "transformers": transformers,
            "transformers_sha256": transformers_sha256}


def adapter(files):
    """`files` maps repo-relative paths to sha256; the runner numerics files are required."""
    missing = sorted(set(NUMERICS) - set(files))
    if missing:
        raise ValueError(f"adapter section lacks runner numerics files {missing}")
    return {"files": dict(sorted(files.items()))}


def repo_files(paths):
    return {path: file_sha256(ROOT / path) for path in paths}


def mode(document):
    return {"name": document["name"], "policy_sha256": sha256(canonical(document))}


def freeze_sha256():
    """The installed distributions, as sorted name==version lines."""
    lines = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in importlib.metadata.distributions()})
    return sha256(canonical(lines))


def environment(device):
    """Versions a score depends on; `device` is "sm_<cc>" or "cpu"."""
    import torch
    try:
        cublas = importlib.metadata.version("nvidia-cublas")
    except importlib.metadata.PackageNotFoundError:
        cublas = None
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cublas": cublas,
            "device": device, "env_freeze_sha256": freeze_sha256()}


def outputs(topk, taps=None):
    return {"topk": topk, "taps": taps}


def document(model, checkpoint, sequence, reference_code, adapter, mode, environment, outputs):
    return {"schema": SCHEMA, "model": model, "checkpoint": checkpoint, "sequence": sequence,
            "reference_code": reference_code, "adapter": adapter, "mode": mode,
            "environment": environment, "outputs": outputs}


def encoder_document(model, checkpoint, declaration, media, code, mode, environment, outputs):
    """The key of one sequence's media through one encoder declaration in one
    mode: `declaration` its repo path and sha256, `media` the sequence's
    items it takes (`sequence(...)["media"]` entries), `code` the
    `ENCODER_NUMERICS` files and the model's variant file, `outputs` the
    taps written."""
    missing = sorted(set(ENCODER_NUMERICS) - set(code["files"]))
    if missing:
        raise ValueError(f"encoder key lacks its numerics files {missing}")
    return {"schema": ENCODER_SCHEMA, "model": model, "checkpoint": checkpoint, "declaration": declaration,
            "media": media, "code": code, "mode": mode, "environment": environment, "outputs": outputs}


def key_sha256(doc):
    return sha256(canonical(doc))
