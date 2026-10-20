"""Freeze only the authenticated query operands needed by the pair job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def read_regular(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError(f"not a regular file: {path}")
        return source.read()


def freeze(ready_path, output):
    ready = json.loads(read_regular(ready_path))
    source = Path(ready["directory"])
    manifest = read_regular(source / "dataset.json")
    if hashlib.sha256(manifest).hexdigest() != ready["manifest_sha256"]:
        raise ValueError("captured dataset differs from authenticated READY")
    dataset = json.loads(manifest)
    if (dataset["schema"] != "chopinfer.prefill-glue-dataset.v1"
            or dataset["rows"] != 8192 or dataset["epsilon_bits"] != 507307272):
        raise ValueError("capture row or epsilon contract differs")
    if dataset["producer"]["capture_job_sha256"] != dataset["capture_job_sha256"]:
        raise ValueError("capture producer identity differs")
    output.mkdir(parents=True, exist_ok=True)
    tensors = []
    for layer in (0, 39):
        for name, role, dtype, shape, size in (
            ("query-input.f32", "input", "f32", [8192, 1280], 8192*1280*4),
            ("query-weight.bf16", "weight", "bf16", [1280], 1280*2),
        ):
            selected = [t for t in dataset["tensors"] if t["layer"] == layer
                        and t["physical_rank"] == 0 and t["name"] == name]
            if len(selected) != 1:
                raise ValueError("missing or duplicate real query operand")
            tensor = selected[0]
            relative = Path(tensor["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe capture path")
            for parent in (relative, *relative.parents):
                if (source / parent).is_symlink():
                    raise ValueError("capture symlink refused")
            if tensor["shape"] != shape or tensor["dtype"] != dtype or tensor["bytes"] != size:
                raise ValueError("real query operand declaration differs")
            data = read_regular(source / relative)
            if len(data) != size or hashlib.sha256(data).hexdigest() != tensor["sha256"]:
                raise ValueError("real query operand digest differs")
            filename = f"L{layer:02}-{name}"
            destination = output / filename
            if destination.exists() and read_regular(destination) != data:
                raise ValueError("refuse to replace a different frozen operand")
            destination.write_bytes(data)
            tensors.append(dict(file=filename, role=role, layer=layer,
                                sha256=tensor["sha256"], bytes=size, shape=shape))
    (output / "source-dataset.json").write_bytes(manifest)
    result = dict(schema="chopinfer.prefill-query-corpus.v1", rows=8192,
                  source_manifest_sha256=ready["manifest_sha256"],
                  capture_id=dataset["capture_id"], capture_job_sha256=dataset["capture_job_sha256"],
                  producer=dataset["producer"], tensors=tensors)
    (output / "corpus.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.ready, args.output)
    print(f"QUERY_PREFILL_CORPUS tensors={len(result['tensors'])} rows=8192 PASS")


if __name__ == "__main__":
    main()
