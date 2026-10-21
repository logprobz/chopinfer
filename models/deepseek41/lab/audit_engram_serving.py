"""Audit full-row Engram admission against complete captured host journals."""
import argparse
import json
from pathlib import Path

BASE = "chop_deepseek41_mxfp8_fold_m8192_n25600_k6144_g1_f32_v2"
OLD = BASE + "_prefill_tma_v1"
NEW = BASE + "_tma_scales_s2_v1"


def resource_delta(old, new, tp4):
    functions = {line.split()[1] for line in new
                 if line.startswith("resolve ") and line.split()[2] == NEW}
    extra = {f"configure {handle} 68608" for handle in functions}
    assert len(extra) == (4 if tp4 else 0)
    assert not extra.intersection(old)
    assert sum(line in extra for line in new) == len(extra)
    assert old == [line for line in new if line not in extra]
    return len(extra)


def substitutions(old, new, expected):
    assert len(old) == len(new)
    admits = computes = 0
    for a, b in zip(old, new):
        if a == b:
            continue
        assert a.startswith("launch ") and a.split()[2] in (OLD, OLD + "_admit_v1")
        assert a.replace(OLD, NEW).replace(
            "shared_memory_bytes: 66560,", "shared_memory_bytes: 68608,"
        ) == b
        if a.split()[2].endswith("_admit_v1"):
            admits += 1
        else:
            computes += 1
    assert admits == computes == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for kind, count in [("lifecycle", 10), ("verify", 6)]:
        old_files = sorted((args.before / kind).glob("*.journal"))
        assert len(old_files) == count
        assert {p.name for p in old_files} == {
            p.name for p in (args.after / kind).glob("*.journal")
        }
        for old in old_files:
            new = args.after / kind / old.name
            assert Path(str(old) + ".handles.json").read_bytes() == Path(
                str(new) + ".handles.json"
            ).read_bytes()
            full = ".full" if kind == "verify" else ""
            configs = resource_delta(
                Path(str(old) + full).read_text().splitlines(),
                Path(str(new) + full).read_text().splitlines(),
                "tp4" in old.name,
            )
            if kind == "verify":
                assert old.read_bytes() == new.read_bytes()
            results.append(dict(file=f"{kind}/{old.name}", added_configures=configs))
    old_files = sorted((args.before / "scales").glob("*.journal"))
    assert len(old_files) == 18
    for old in old_files:
        new = args.after / "scales" / old.name
        expected = 8 if old.name.startswith("rows8192-") and "referencefalse" in old.name else 0
        a, b = old.read_text().splitlines(), new.read_text().splitlines()
        substitutions(a, b, expected)
        if expected:
            bad = b.copy()
            at = next(i for i, line in enumerate(b) if line.startswith("launch ") and line.split()[2] == NEW)
            bad[at] = bad[at].replace("offset: ", "offset: 1", 1)
            try:
                substitutions(a, bad, expected)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupted operand escaped the audit")
        results.append(dict(file=f"scales/{old.name}", changed_computes=expected))
    isolated = sorted((args.after / "scales").glob("engram-*.journal"))
    assert len(isolated) == 12
    for p in isolated:
        original = p.name.removeprefix("engram-").replace("referencetrue", "referencefalse")
        root = args.before if "referencetrue" in p.name else args.after
        assert p.read_bytes() == (root / "scales" / original).read_bytes()
    results.append(dict(isolated_controls_equal_recorded_base=6, isolated_candidates_equal_default=6))
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print("ENGRAM_JOURNAL_AUDIT", len(results), "negative controls passed")


if __name__ == "__main__":
    main()
