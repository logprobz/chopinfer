"""Compare served preprocessing with adopted checkpoint-processor references.

Integer resize and patch rows must match exactly. Floating rows must differ
from reference F32 by at most twice the reference's own maximum F32/F64
spread on the same case and stage (mel: magnitude relative to each frame peak). The factor two is the triangle bound for
two implementations each within that measured spread of F64. A zero spread
requires exact equality; no absolute tolerance is added.
"""

import argparse
from array import array
import json
import math
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.refrun import preprocessing
from common import ROOT, qualification_config, resolve_input
from refs import read_member, verify_input


SPREAD_FACTOR = 2.0
PROCESSOR_OVERRIDES = {}  # Declared metadata; validated and read by declared_overrides.


def values(path, dtype):
    result = array({"u8": "B", "f32": "f", "f64": "d"}[dtype])
    result.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        result.byteswap()
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"nonfinite tensor: {path.name}")
    return result


def floating(candidate, single, double, bins=None):
    if not single or len(candidate) != len(single) or len(single) != len(double):
        raise ValueError("empty or mismatched floating tensors")
    if not all(math.isfinite(x) for data in (candidate, single, double) for x in data):
        raise ValueError("nonfinite tensor")
    if bins is None:
        spread = max(abs(a - b) for a, b in zip(single, double))
        error = max(abs(a - b) for a, b in zip(candidate, single))
    else:
        if bins <= 0 or len(single) % bins:
            raise ValueError("invalid mel frame shape")
        spread, error = 0.0, 0.0
        for start in range(0, len(single), bins):
            reference = double[start:start + bins]
            # Subtract the frame's peak log magnitude before exp to avoid
            # overflow and compare magnitude error relative to that peak.
            peak = max(reference)
            for a, b, c in zip(candidate[start:start + bins], single[start:start + bins], reference):
                spread = max(spread, abs(math.exp(b - peak) - math.exp(c - peak)))
                error = max(error, abs(math.exp(a - peak) - math.exp(b - peak)))
    bound = SPREAD_FACTOR * spread
    return {"metric": "frame-peak-relative-magnitude" if bins else "absolute",
            "reference_f32_f64_spread": spread, "maximum_error": error,
            "bound": bound, "passed": error <= bound}


def compare(reference, reference_root, candidate, candidate_root):
    expected = {case["name"]: case for case in reference["cases"]}
    actual = {case["name"]: case for case in candidate["cases"]}
    if len(expected) != len(reference["cases"]) or len(actual) != len(candidate["cases"]) or expected.keys() != actual.keys():
        raise ValueError("duplicate, missing or extra preprocessing cases")
    results = []
    for name, case in expected.items():
        got = actual[name]
        if case.get("kind") == "image" and "integer" not in case["stages"].get("patches", {}):
            raise ValueError(f"{name}: image reference requires integer evidence")
        if got["stages"].keys() != case["stages"].keys():
            raise ValueError(f"{name}: missing or extra stages")
        for stage, target in case["stages"].items():
            observed = got["stages"][stage]
            shape_matches = observed["shape"] == target["f32"]["shape"]
            grid_matches = got.get("grid") == case.get("grid")
            row = {"case": name, "stage": stage, "shape_matches": shape_matches, "grid_matches": grid_matches}
            if not shape_matches or not grid_matches:
                results.append({**row, "passed": False})
                continue
            single = values(reference_root / target["f32"]["file"], "f32")
            double = values(reference_root / target["f64"]["file"], "f64")
            ours = values(candidate_root / observed["file"], "f32")
            row.update(floating(ours, single, double, observed["shape"][-1] if stage == "mel" else None))
            if "integer" in target:
                integer = values(reference_root / target["integer"]["file"], "u8")
                raw = values(candidate_root / observed["integer"], "f32")
                row["integer_equal"] = len(raw) == len(integer) and all(a == b for a, b in zip(raw, integer))
                row["passed"] &= row["integer_equal"]
            results.append(row)
    return {"passed": bool(results) and all(row["passed"] for row in results), "rows": results}


def receipt_reference(config):
    """Authenticate the existing reference set for nested receipt admission."""
    root = resolve_input("media_preprocessing_reference", config=config)
    frontend = resolve_input("media_preprocessing_frontend", config=config)
    verify_input(config, "media_preprocessing_reference", root)
    verify_input(config, "media_preprocessing_frontend", frontend)
    references = []
    for manifest in sorted(root.glob("*/reference.json")):
        reference = preprocessing.verify(json.loads(read_member(config, "media_preprocessing_reference", manifest)),
                                         manifest.parent)
        if reference["cases"][0]["kind"] == "image" and reference["binding"]["processor"].get("overrides", {}) \
                != preprocessing.declared_overrides(config):
            raise ValueError("image reference overrides differ from the declaration")
        references.append(reference)
    if not references:
        raise ValueError("no preprocessing references")
    return {"references": references, "frontend_sha256": preprocessing.keys.file_sha256(frontend)}


def receipt_metrics(value, reference):
    """Judge recorded tensor errors after exact reference/case/stage admission.

    Raw candidate tensors are not in this legacy stdout receipt. Error,
    shape and integer observations are recorded evidence, not recomputed
    tensor comparisons, even when the derived verdict equals the old one.
    """
    from kinds import require

    def indexed(rows, key, label):
        require(isinstance(rows, list) and bool(rows), f"{label}: no rows")
        require(all(isinstance(row, dict) for row in rows), f"{label}: malformed rows")
        result = {key(row): row for row in rows}
        require(len(result) == len(rows), f"{label}: duplicate rows")
        return result

    def boolean(value, label):
        require(type(value) is bool, f"{label}: missing boolean")
        return value

    require(value.get("reference_bound") is True and value.get("exporter_build_verified") is True,
            "preprocessing reference or exporter is unverified")
    digest = value.get("exporter_sha256")
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "preprocessing exporter digest missing")
    require(value.get("frontend_sha256") == reference["frontend_sha256"], "preprocessing frontend differs")
    expected = indexed(reference["references"], lambda row: row["sha256"], "reference set")
    reports = indexed(value.get("references"), lambda row: row["reference_sha256"], "preprocessing reports")
    require(reports.keys() == expected.keys(), "missing or extra preprocessing reference reports")
    scored = []
    for digest, report in reports.items():
        target = expected[digest]
        require(report.get("binding") == target["binding"], "preprocessing reference binding differs")
        cases = indexed(target["cases"], lambda row: row["name"], "reference cases")
        stages = {(name, stage): spec for name, case in cases.items() for stage, spec in case["stages"].items()}
        require(stages, "reference has no stages")
        rows = indexed(report.get("rows"), lambda row: (row["case"], row["stage"]), "preprocessing rows")
        require(rows.keys() == stages.keys(), "missing or extra preprocessing cases or stages")
        judged = []
        for key, row in rows.items():
            shape = boolean(row.get("shape_matches"), "shape_matches")
            grid = boolean(row.get("grid_matches"), "grid_matches")
            passed = shape and grid
            if passed:
                metric = "frame-peak-relative-magnitude" if key[1] == "mel" else "absolute"
                require(row.get("metric") == metric, "preprocessing metric differs")
                for field in ("reference_f32_f64_spread", "maximum_error", "bound"):
                    number = row.get(field)
                    require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                            f"preprocessing {field}: invalid number")
                require(row["bound"] == SPREAD_FACTOR * row["reference_f32_f64_spread"],
                        "preprocessing spread bound differs")
                passed = row["maximum_error"] <= row["bound"]
                if "integer" in stages[key]:
                    integer = boolean(row.get("integer_equal"), "integer_equal")
                    passed = passed and integer
            judged.append({**row, "passed": passed})
        scored.append({**report, "rows": judged, "passed": all(row["passed"] for row in judged)})
    return {"passed": all(report["passed"] for report in scored), "references": scored,
            **{key: value[key] for key in ("reference_bound", "exporter_sha256", "exporter_build_verified", "frontend_sha256")}}


def build_exporter(requested=None):
    """Resolve a fresh Cargo artifact, never an unchecked target/debug binary."""
    command = ["cargo", "build", "-p", "runtime-host", "--example", "media-preprocessing",
               "--locked", "--offline", "--message-format=json"]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    artifacts = [row for line in result.stdout.splitlines() if line.startswith("{")
                 if (row := json.loads(line)).get("reason") == "compiler-artifact"
                 and row.get("target", {}).get("name") == "media-preprocessing" and row.get("executable")]
    if len(artifacts) != 1:
        raise ValueError("fresh preprocessing exporter artifact missing or ambiguous")
    built = Path(artifacts[0]["executable"])
    digest = preprocessing.keys.file_sha256(built)
    if requested is not None and preprocessing.keys.file_sha256(requested) != digest:
        raise ValueError("requested exporter differs from the fresh Cargo build")
    return built, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, help="must match the fresh Cargo build")
    parser.add_argument("--reference", type=Path, help="diagnostic override; the declared check uses its adopted input")
    parser.add_argument("--frontend", type=Path, help="diagnostic override")
    args = parser.parse_args()
    global PROCESSOR_OVERRIDES
    config = None if args.reference else qualification_config()
    PROCESSOR_OVERRIDES = {} if config is None else preprocessing.declared_overrides(config)
    root = args.reference or resolve_input("media_preprocessing_reference", config=config)
    frontend = args.frontend or resolve_input("media_preprocessing_frontend", config=config)
    if config is not None:
        verify_input(config, "media_preprocessing_reference", root)
        verify_input(config, "media_preprocessing_frontend", frontend)
    manifests = sorted(root.glob("*/reference.json"))
    if not manifests:
        raise ValueError("no preprocessing references")
    exporter, exporter_digest = build_exporter(args.exporter)
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        # Cargo may replace its shared target artifact during a parallel CI
        # build. Execute an immutable private copy of the verified bytes.
        copied = Path(tmp) / "media-preprocessing"
        shutil.copyfile(exporter, copied)
        copied.chmod(0o700)
        if preprocessing.keys.file_sha256(copied) != exporter_digest:
            raise ValueError("exporter changed while copying the fresh build")
        exporter = copied
        for manifest in manifests:
            data = manifest.read_bytes() if config is None else read_member(config, "media_preprocessing_reference", manifest)
            reference = preprocessing.verify(json.loads(data), manifest.parent)
            if config is not None and reference["cases"][0]["kind"] == "image":
                expected_overrides = PROCESSOR_OVERRIDES
                if reference["binding"]["processor"].get("overrides", {}) != expected_overrides:
                    raise ValueError("image reference overrides differ from the declaration")
            output = Path(tmp) / manifest.parent.name
            subprocess.run([str(exporter), str(frontend), str(manifest.parent), str(output)], check=True)
            candidate = json.loads((output / "candidate.json").read_bytes())
            report = compare(reference, manifest.parent, candidate, output)
            reports.append({"reference_sha256": reference["sha256"], "binding": reference["binding"], **report})
        exporter_verified = preprocessing.keys.file_sha256(exporter) == exporter_digest
    result = {"passed": all(report["passed"] for report in reports), "references": reports,
              "reference_bound": config is not None,
              "exporter_sha256": exporter_digest,
              "exporter_build_verified": exporter_verified,
              "frontend_sha256": preprocessing.keys.file_sha256(frontend)}
    if not result["exporter_build_verified"]:
        raise ValueError("exporter changed during comparison")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
