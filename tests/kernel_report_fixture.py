"""Small producer-shaped reports with real temporary sources and artifacts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import kernel_reports as kr


HEADER = "crates/kernels-sm120/cuda/common/ptx.cuh"
# Reports are current only for the workspace version that built them.
VERSION = tomllib.loads((Path(__file__).resolve().parents[1] / "Cargo.toml").read_text())["workspace"]["package"]["version"]


def make_report(directory, package=f"bundle-qwen38@{VERSION}", *, source="chopinfer",
                paths=(HEADER,), owner="chopinfer", content=b"header", declarations=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = directory / "source"
    files, members = {}, []
    for index, path in enumerate(paths):
        location = root / path
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_bytes(content)
        file = {"identity": {"kind": "Project", "owner": owner, "path": path},
                "sha256": kr.digest(content), "bytes": len(content), "location": str(location)}
        key = kr.file_id(file)
        files[key] = file
        members.append({"closure_key": f"source/{index}/{path}", "file": key})
    generated = directory / "entry.cu"
    generated.write_bytes(b"generated unit")
    file = {"identity": {"kind": "Generated", "owner": None, "path": "entry.cu"},
            "sha256": kr.digest(generated.read_bytes()), "bytes": generated.stat().st_size,
            "location": str(generated)}
    key = kr.file_id(file)
    files[key] = file
    members.append({"closure_key": "generated/entry.cu", "file": key})
    compiler = directory / "nvcc"
    compiler.write_bytes(b"compiler")
    binary = directory / "fixture_v1.cubin"
    binary.write_bytes(b"cubin")
    closure = hashlib.sha256()
    for member in sorted(members, key=lambda m: m["closure_key"]):
        key = member["closure_key"].encode()
        raw = Path(files[member["file"]]["location"]).read_bytes()
        closure.update(len(key).to_bytes(8, "little") + key + len(raw).to_bytes(8, "little") + raw)
    recipe = {"toolchain_version": "fixture", "nvcc_sha256": kr.digest(compiler.read_bytes()),
              "flags": ["-O3", "-arch=sm_120f"], "defines": [["B", "2"], ["A", "1"]],
              "headers": list(paths), "include_dirs": [str(root)],
              "generated_sha256": {"entry.cu": file["sha256"]}, "source_closure_sha256": closure.hexdigest()}
    identity = {"name": "fixture_v1", "line_info": False,
                **{k: deepcopy(recipe[k]) for k in ("toolchain_version", "nvcc_sha256", "flags", "defines", "source_closure_sha256")},
                "cubin_sha256": kr.digest(binary.read_bytes()), "entries": [], "resources": []}
    sidecar = directory / "fixture_v1.identity.json"
    sidecar.write_text(json.dumps(identity))
    lock = directory / "kernels.lock"
    lock.write_text(json.dumps([identity]))
    report = {"schema": kr.SCHEMA, "context": {"consumer": {"source": source, "package": package},
              "selection": "fixture.v1", "sources": [{"id": owner, "path": str(root)}]},
              "source_membership_complete": True, "source_ownership_complete": True,
              "entry_declarations_complete": declarations, "lock_status": "Verified",
              "lock_path": str(lock), "lock_sha256": kr.digest(lock.read_bytes()), "compiler_path": str(compiler),
              "files": files, "cubins": [{"name": identity["name"], "path": str(binary),
              "identity_path": str(sidecar), "identity": identity, "recipe": recipe,
              "recipe_sha256": kr.digest(kr.recipe_bytes(recipe)), "membership": members, "declarations": []}]}
    path = directory / "kernel-build-report.json"
    write_report(path, report)
    return path, report, {owner: root}


def write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=2))


def to_directory(path, report, family="fixture"):
    """Rewrite a fixture report into the producer's directory-lock shape."""
    directory = Path(report["lock_path"]).parent / "kernels.lock.d"
    directory.mkdir(exist_ok=True)
    lock, provenance = directory / (family + ".lock"), directory / (family + ".provenance.json")
    lock.write_text(json.dumps([kr.family_record(c["identity"]) for c in report["cubins"]], indent=2))
    provenance.write_text(json.dumps([{"name": c["name"], "source_closure_sha256": c["identity"]["source_closure_sha256"]}
                                      for c in report["cubins"]]))
    report.update(lock_path=str(directory), lock_sha256=None, lock_families=[{
        "family": family, "lock_path": str(lock), "lock_sha256": kr.digest(lock.read_bytes()),
        "provenance_path": str(provenance), "cubins": [c["name"] for c in report["cubins"]],
        "provenance_changed": []}])
    write_report(path, report)
    return lock, provenance
