"""Freeze real checkpoint operands and seed rows from an authenticated HC capture."""
import argparse
import hashlib
import json
from pathlib import Path


def read(root, name, size=None, sha=None):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("operand escaped corpus root")
    path = root / relative
    if any(p.is_symlink() for p in path.parents) or path.is_symlink():
        raise ValueError("operand symlink refused")
    data = path.read_bytes()
    if size is not None and len(data) != size:
        raise ValueError("operand size differs")
    if sha is not None and hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("operand digest differs")
    return data


def stage(source, expected_sha, output):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("source manifest digest differs")
    manifest = json.loads(data)
    if manifest["schema"] != "chopinfer.flash-hc-corpus.v1" or not manifest["complete"]:
        raise ValueError("complete authenticated HC capture required")
    capture = next(c for c in manifest["captures"] if c["phase"] == "decode-8")
    if capture["request_origin"] != "real":
        raise ValueError("warmup is not evidence")
    eligible = [c for c in capture["cases"] if c["eligible_post_mix"]]
    if len(eligible) != 77:
        raise ValueError("expected 77 eligible boundaries")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    sizes = {"fn": 24 * 20480 * 4, "base": 24 * 4, "scale": 3 * 4,
             "norm_weight": 5120 * 2, "streams": 8 * 20480 * 2,
             "sum": 8 * 5120 * 4, "pre_in": 8 * 4 * 4,
             "post_in": 8 * 4 * 4, "comb_in": 8 * 16 * 4}
    for ordinal, record in enumerate([eligible[0], eligible[38], eligible[76]]):
        tensors = {}
        for role, size in sizes.items():
            operand = record["inputs"][role]
            value = read(source.parent / capture["group"] / "operands",
                         operand["file"], operand["bytes"], operand["sha256"])
            if len(value) != size:
                raise ValueError(f"unexpected {role} shape")
            name = f"site{ordinal}-{role}.bin"
            (output / name).write_bytes(value)
            tensors[role] = {"file": name, "sha256": hashlib.sha256(value).hexdigest(),
                             "bytes": size, "dtype": operand["dtype"]}
        cases.append({"id": f"site{ordinal}", "site": record["site"], "inputs": tensors})
    result = {"schema": "chopinfer.hc-prefill-fused-corpus.v1",
              "source_sha256": expected_sha,
              "checkpoint_identity": manifest["provenance"]["checkpoint_identity"],
              "seed_rows": 8, "cases": cases,
              "scope": "Real checkpoint weights; seeded prefill streams use captured decode magnitudes."}
    (output / "corpus.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.source, args.sha256, args.output)
    print("HC_PREFILL_FUSED_CORPUS sites=3 seed_rows=8 PASS")
