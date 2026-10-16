"""Trace gate modules offline, with real fixture reads and a transport stub.

Run once with --source pointing at tools exported from the baseline revision,
then with the candidate. The output records attempted reads, receipt names and
thinking settings. This is input parity evidence, not an accuracy gate run.
Unavailable external evidence and deliberately refused HTTP requests are listed
as stops. --root supplies the unchanged fixtures for both source revisions.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import contextlib
import importlib
import io
import json
import os
import re
import subprocess
import tempfile
from unittest.mock import patch


def gate_file(source, name):
    """The model's gate in a tool tree; older trees kept it in tools/qual/gates."""
    legacy = source / "tools/qual/gates" / (name + ".json")
    return legacy if legacy.is_file() else source / "models" / name / "qual.json"


def worker(args):
    source = Path(args.source).resolve()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(source / "tools/qual"))
    import common
    common.ROOT = root
    import declaration
    gate = declaration.load(gate_file(source, args.gate))
    check = (next(check for check in gate["checks"] if check["id"] == args.check)
             if args.check != "policy-probes" else {"module": "prefix_exact", "args": []})
    reads, thinking, receipts = set(), set(), set()
    original_open = Path.open
    original_read_text = Path.read_text
    data = os.environ.get("CHOPINFER_DATA", "")
    fixtures = os.environ.get("CHOPINFER_MIMO_FIXTURES", "")

    def normalized(path):
        value = str(Path(path).absolute())
        if value.startswith(str(source / "fixtures") + "/"):
            value = str(root / "fixtures") + value[len(str(source / "fixtures")):]
        for base, label in [(args.scratch, "$SCRATCH"), (str(root), "$ROOT"), (str(source), "$SOURCE"),
                            (fixtures, "$FIXTURES"), (data, "$DATA")]:
            if base and (value == base or value.startswith(base + "/")):
                return label + value[len(base):]
        return value

    def opened(path, mode="r", *values, **kwargs):
        label = normalized(path)
        if "r" in mode and not label.endswith(".py") and not label.endswith(".pyc"):
            if "/gates/" not in label and not label.endswith(("/qual.json", "frontend.json")):
                reads.add(label)
        return original_open(path, mode, *values, **kwargs)

    class Client:
        def __init__(self, url, timeout=900, **kwargs):
            self.url, self.timeout = url, timeout

        def identity(self, model=None):
            return {"model": model or "trace", "url": self.url}

        def request(self, endpoint, body=None, **kwargs):
            if body and "chat_template_kwargs" in body:
                thinking.add(json.dumps(body["chat_template_kwargs"], sort_keys=True))
            raise RuntimeError("offline transport stub")

    def receipt(check, *values, **kwargs):
        receipts.add(check)
        return Path(args.scratch) / (check + ".json")

    # Every input a check names reads a file of that name in the scratch directory.
    values = {key: str(Path(args.scratch) / (key + ".json"))
              for key in re.findall(r"\{(\w+)\}", json.dumps(check["args"]))}
    values.update(url="http://offline", reference_url="http://offline", model="trace",
                  run_id="trace", run=args.scratch)
    argv = [value.format(**values) for value in check["args"]]
    if check["module"] == "soak":
        argv += ["--duration", "0.03", "--concurrency", "1"]
    if check["module"] == "evals" and argv[0] == "run":
        argv += ["--limit", "1", "--long-lengths", "32k", "--concurrency", "1"]
    stop = None
    with patch.object(common, "Client", Client), patch.object(common, "receipt", receipt), \
         patch.object(Path, "open", opened), patch("urllib.request.urlopen", side_effect=RuntimeError("offline")), \
         patch.object(sys, "argv", [check["module"], *argv]), contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        module = importlib.import_module(check["module"])
        if args.check == "policy-probes":
            from test_prefix_exact import qwen_reference, aligned_rows
            import verify_exact
            import spec_bench
            import spec_verify
            import evals
            ids = list(range(20, 30))
            exact = qwen_reference(ids, -1.0)
            if args.gate == "mimo26":
                exact["schema"] = json.loads((root / "fixtures/mimo26/teacher-exact-reference.json").read_text())["schema"]
            record = {"family": args.gate, "positions": aligned_rows(ids, -1.10, -1.0, -1.05),
                      "prompt_end": [], "detokenized": {}}
            scored = module.evaluate(record, exact)
            spec_bench.gsm8k_questions(1)
            spec_verify.prompt_sets()
            body = spec_bench.workload("trace", "trace", "gsm8k", 1, 0, "Question", "Solve", 64)
            sizing = []
            with patch.object(evals, "haystack", side_effect=lambda *args: sizing.append(args[2]) or ["text"]):
                evals.long_item(["text"], "multikey", 100, 0)
            return {"reads": sorted(reads), "thinking": body["chat_template_kwargs"],
                    "prefix_checks": scored["checks"], "anchors": list(scored["scores"]["by_anchor"]),
                    "draft_counts": [verify_exact.draft_count(k, 20) for k in range(1, 10)],
                    "long_prompt_characters": sizing}
        try:
            module.main()
        except (Exception, SystemExit) as error:
            stop = type(error).__name__
        if check["module"] == "soak":
            module.media_requests(list(module.MEDIA_CASES))
    return {"reads": sorted(reads), "thinking": [json.loads(v) for v in sorted(thinking)],
            "receipts": sorted(receipts), "stop": stop}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", choices=("mimo26", "qwen38"), required=True)
    parser.add_argument("--check")
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(worker(args), sort_keys=True))
        return
    import declaration
    gate_path = gate_file(args.source.resolve(), args.gate)
    gate = declaration.load(gate_path)
    output = {}
    for check in [*gate["checks"], {"id": "policy-probes"}]:
        with tempfile.TemporaryDirectory(dir=args.scratch) as scratch:
            env = dict(os.environ, CHOPINFER_QUAL_CONFIG=str(gate_path),
                       CHOPINFER_QUAL_RECEIPTS=scratch)
            result = subprocess.run([sys.executable, "-B", __file__, "--source", str(args.source),
                "--root", str(args.root), "--gate", args.gate, "--check", check["id"],
                "--scratch", scratch], env=env, text=True, capture_output=True, timeout=90)
            if result.returncode:
                raise RuntimeError(f"{check['id']}: {result.stderr}")
            output[check["id"]] = json.loads(result.stdout)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
