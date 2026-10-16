"""Prepare bounded expert operands on an assigned GPU, using explicit recipes.

This loads selected checkpoint matrices and dumped routed rows only. It does
not instantiate a model. Down inputs are frozen later by the lab's baseline
probe, so every down candidate consumes identical BF16 values.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


def choose_experts(routes, low, high, count):
    counts = Counter(x for row in routes for x in row if low <= x < high)
    # Hot and rare experts both contribute; do not synthesize missing routes.
    hot = sorted(counts, key=lambda x: (-counts[x], x))
    rare = sorted(counts, key=lambda x: (counts[x], x))
    selected = hot[:max(1, count // 2)]
    for expert in rare:
        if expert not in selected:
            selected.append(expert)
        if len(selected) >= count:
            break
    return selected[:count]


def owned(rank, ranks, intermediate, experts, expert_parallel):
    if ranks not in (2, 4) or not 0 <= rank < ranks:
        raise ValueError("unsupported rank topology")
    if expert_parallel:
        return (0, intermediate), (rank * (experts // ranks), (rank + 1) * (experts // ranks))
    return (rank * (intermediate // ranks), (rank + 1) * (intermediate // ranks)), (0, experts)


def source_digest(paths):
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("expert operand preparation requires an assigned CUDA device")
    import torch
    from safetensors import safe_open
    from tools.refrun import keys, stamps
    from capture_non_dense import Capture

    recipe = json.loads(args.recipe.read_text())
    if recipe["schema"] != "chopinfer.grouped-capture.v1":
        raise ValueError("unknown recipe schema")
    if not 1 <= recipe["expert_count"] <= 4 or not 1 <= recipe["max_rows"] <= 128:
        raise ValueError("unbounded expert/row selection")
    checkpoint = Path(os.environ[recipe["checkpoint_env"]])
    lock_path = Path(recipe["checkpoint_lock"])
    lock = keys.read_lock(lock_path)
    if recipe.get("lock_variant"):
        lock = lock[recipe["lock_variant"]]
    index_file = "model.safetensors.index.json"
    verified = stamps.verify(checkpoint, lock, ["config.json", index_file], stamps.default_cache())
    index = json.loads((checkpoint / index_file).read_text())["weight_map"]
    device = torch.device(args.device)
    i = recipe["intermediate"]
    local, expert_range = owned(recipe["rank"], recipe["ranks"], i, recipe["experts"], recipe["expert_parallel"])
    site = recipe["input"]
    input_path = Path(os.environ[site["root_env"]]) / site["file"]
    routes_path = Path(os.environ[site["root_env"]]) / site["routes_file"]
    if site.get("tensor"):
        with safe_open(input_path, framework="pt", device=str(device)) as file:
            x = file.get_tensor(site["tensor"])
            if x.dtype != torch.bfloat16:
                raise ValueError("expert dump must contain BF16 operands")
            ids = file.get_tensor(site["routes_tensor"]).cpu().tolist()
    else:
        data = input_path.read_bytes()
        x = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(-1, recipe["hidden"]).to(device)
        route_data = routes_path.read_bytes()
        flat = [v[0] for v in struct.iter_unpack("<i", route_data)]
        ids = [flat[start:start + recipe["top_k"]] for start in range(0, len(flat), recipe["top_k"])]
    if x.ndim != 2 or x.shape[1] != recipe["hidden"] or x.shape[0] != len(ids):
        raise ValueError("dump rows/routes do not match the recipe")
    if any(len(row) != recipe["top_k"] or len(set(row)) != len(row) or any(type(x) is not int or not 0 <= x < recipe["experts"] for x in row) for row in ids):
        raise ValueError("malformed captured top-k routing")

    def tensor(name, rows=None, columns=None):
        file = index[name]
        if file not in verified:
            verified.update(stamps.verify(checkpoint, lock, [file], stamps.default_cache()))
        with safe_open(checkpoint / file, framework="pt", device=str(device)) as source:
            if rows is None and columns is None:
                return source.get_tensor(name)
            sliced = source.get_slice(name)
            return sliced[slice(*rows) if rows is not None else slice(None),
                          slice(*columns) if columns is not None else slice(None)]

    if site.get("norm_tensor"):
        norm = tensor(site["norm_tensor"]).float()
        wide = x.float()
        normalized = wide * torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + site["eps"])
        x = (normalized.to(torch.bfloat16).float() * norm).to(torch.bfloat16)
    if not torch.isfinite(x.float()).all():
        raise ValueError("nonfinite expert input")
    selected = choose_experts(ids, *expert_range, recipe["expert_count"])
    if not selected:
        raise ValueError("no routed experts selected")
    source = source_digest([Path(__file__), Path(__file__).with_name("capture_non_dense.py"), args.recipe, lock_path, input_path, routes_path])
    capture = Capture(args.output, source, 256 << 20)
    metadata = []

    def bank(expert, projection, prefix):
        name = recipe["weight_prefix"].format(expert=expert) + projection + ".weight"
        scale_name = name + recipe["scale_suffix"]
        rows = local if projection != "down_proj" and not recipe["expert_parallel"] else None
        start, end = local if projection == "down_proj" and not recipe["expert_parallel"] else (0, recipe["hidden"] if projection != "down_proj" else i)
        if recipe["format"] == "mxfp4":
            code = tensor(name, rows, (start // 2, end // 2)).to(torch.uint8)
            scale = tensor(scale_name, rows, (start // 32, end // 32)).to(torch.uint8)
            codes = torch.stack((code & 15, code >> 4), dim=-1).flatten(-2)
            table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64, device=device)
            powers = torch.ldexp(torch.ones_like(scale, dtype=torch.float64), scale.long() - 127)
            exact = table[codes.long()] * powers.repeat_interleave(32, dim=1)
            decoded = exact.to(torch.bfloat16)
            if not torch.equal(exact, decoded.double()):
                raise ValueError("MX operand decode is not exact in BF16")
            scales = None
        elif recipe["format"] == "fp8_post":
            code = tensor(name)
            scale = tensor(scale_name).float()
            decoded = code.to(torch.bfloat16)
            if not torch.equal(code.float(), decoded.float()):
                raise ValueError("FP8 code conversion is not exact BF16")
            scales = scale.repeat_interleave(128, dim=0)[:decoded.shape[0]]
            if scales.shape != (decoded.shape[0], decoded.shape[1] // 128):
                raise ValueError("FP8 scale grid mismatch")
        else:
            raise ValueError("unsupported expert operand scale order")
        if not torch.isfinite(decoded.float()).all() or (scales is not None and not torch.isfinite(scales).all()):
            raise ValueError("nonfinite expert operand")
        n, k = decoded.shape
        result = {"n": n, "k": k, "weight": capture.file(prefix + ".bf16", decoded),
                  "scales": capture.file(prefix + "-scales.f32", scales) if scales is not None else None, "global": 1.}
        metadata.append({"projection": projection, "name": name, "scale_name": scale_name,
                         "stored_code_sha256": hashlib.sha256(code.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                         "stored_scale_sha256": hashlib.sha256(scale.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()})
        return result

    for expert in selected:
        positions = [p for p, row in enumerate(ids) if expert in row][:recipe["max_rows"]]
        chosen = x[torch.tensor(positions, device=device)]
        prefix = f"expert-{expert:03}"
        case = {"id": prefix, "origin": "checkpoint weights and dumped routed rows; policy norm where declared",
                "source_sha256": source, "ranks": recipe["ranks"], "rank": recipe["rank"], "expert": expert, "rows": len(positions),
                "input": capture.file(prefix + "-input.bf16", chosen),
                "gate": bank(expert, "gate_proj", prefix + "-gate"), "up": bank(expert, "up_proj", prefix + "-up"),
                "down": bank(expert, "down_proj", prefix + "-down"), "schedules": recipe["schedules"]}
        capture.cases.append(case)
        metadata.append({"id": prefix, "route_rows": positions, "down_input": "frozen later by declared baseline GPU probe"})
    stamps.confirm(checkpoint, verified)
    (args.output / "manifest.json").write_text(json.dumps({"schema": "chopinfer.grouped-accumulation.v1", "cases": capture.cases}, indent=2) + "\n")
    (args.output / "capture.json").write_text(json.dumps({"source_sha256": source, "bytes": capture.used,
        "format": recipe["format"], "ranks": recipe["ranks"], "rank": recipe["rank"], "selected_experts": selected,
        "scope": "isolated expert operands; no whole model or serving equivalence", "sites": metadata}, indent=2) + "\n")
    print(f"captured {len(capture.cases)} routed experts, {capture.used} bytes")


if __name__ == "__main__":
    main()
