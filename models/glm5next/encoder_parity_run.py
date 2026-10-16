"""Prepare GLM 5.3 Flash's encoder_parity run directory (plan step 39, images).

    python3 models/glm5next/encoder_parity_run.py requests --dir DIR
    python3 models/glm5next/encoder_parity_run.py assemble --run RUN \\
        [--reference-exact DIR] [--reference-policy DIR]
    python3 models/glm5next/encoder_parity_run.py record --url URL \\
        --server-log LOG --taps-dir TAPS --run RUN [--reference-exact DIR] [--reference-policy DIR]

The encoder_parity check reads a run directory (tools/qual/README.md,
encoder parity): the engine's encoder taps, captured into it by
tools/qual/tap_binding.py from a diagnostic-taps server, and the encoder
reference runs (tools/refrun/encoder.py) of the same images.

`requests` writes DIR/<probe>.json, one chat request per image probe of the
gate's media capture (qualification.media), as the multimodal check sends
it (the probe's images and question, its thinking switch) with one output
token, so the engine encodes exactly the reference workload's images, and
prints the capture command. The capture creates RUN; run it, and the check,
while the diagnostic-taps server (CHOPINFER_TAPS_DIR,
CHOPINFER_REQUEST_TRACE=1, prefix caching off, nothing else in flight) still
serves: the check admits the capture only against the process that wrote
it.

The check is qual.taps.json's (`--model glm5next.taps`), gated against the
diagnostic-taps build. `assemble` completes RUN: RUN/reference/vision, one reference directory in
the layout the scorer reads (tools/qual/encoder_parity.py: run.json and
policy.json naming the precision policy, and taps/<mode> for exact, policy
and reference-bf16), linked from the exact run and the policy and
reference-bf16 run (exact.json, exact-run.json and taps/exact from the
first, the rest from the second); a copy of their workload
(RUN/workload.json and the pixel and grid files it names); and
RUN/parity.json (chopinfer.encoder-parity-run.v1) naming them and the
capture's engine steps. `record` does all three against the live server, as
the correctness runner's producer of the taps configuration's
encoder_parity_run input. It refuses a RUN without a capture, captured
requests other than the workload's sequences, and reference runs of
another declaration or workload, or without their modes.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CHOPINFER_DATA", Path.home() / "chopinfer-data"))
REFERENCE = DATA / "glm5next/V2/encoder-reference"
SCHEMA = "chopinfer.encoder-parity-run.v1"


def tools():
    sys.path.insert(0, str(ROOT / "tools/qual"))
    os.environ.setdefault("CHOPINFER_QUAL_CONFIG", str(ROOT / "models/glm5next/qual.json"))
    import common
    import multimodal
    return common, multimodal


def workload_of(reference):
    """The workload a reference run scored (its run.json's --workload)."""
    argv = json.loads((reference / "run.json").read_text())["argv"]
    return Path(argv[argv.index("--workload") + 1])


def requests(images_dir):
    """{probe name: chat request body} for the gate's image probes."""
    common, multimodal = tools()
    probes, hashes, _ = multimodal.load_probes()
    images = multimodal.probe_images(images_dir, None, hashes)
    thinking = multimodal.media_policy().get("thinking", False)
    return {probe["name"]: {"messages": multimodal.probe_messages(probe, images), "max_tokens": 1,
                            "temperature": 0, "chat_template_kwargs": common.thinking_kwargs(thinking)}
            for probe in probes}


def write_requests(directory, images_dir):
    """Write the probes' request bodies into `directory`; their names."""
    bodies = requests(images_dir)
    directory.mkdir(parents=True, exist_ok=False)
    for name, body in bodies.items():
        (directory / f"{name}.json").write_text(json.dumps(body) + "\n")
    return sorted(bodies)


# Where each file of the combined reference directory comes from: the exact
# run (exact) or the policy and reference-bf16 run (policy).
LAYOUT = {"run.json": ("policy", "run.json"), "policy.json": ("policy", "policy.json"),
          "reference-bf16.json": ("policy", "reference-bf16.json"), "exact.json": ("exact", "exact.json"),
          "exact-run.json": ("exact", "run.json"), "taps/exact": ("exact", "taps/exact"),
          "taps/policy": ("policy", "taps/policy"), "taps/reference-bf16": ("policy", "taps/reference-bf16")}


def combined(directory, runs):
    """Link the two reference runs into one reference directory."""
    declarations = {label: json.loads((run / "run.json").read_text())["declaration"] for label, run in runs.items()}
    if declarations["exact"] != declarations["policy"]:
        raise ValueError("the exact and policy reference runs ran different encoder declarations")
    for name, (label, source) in LAYOUT.items():
        target = runs[label].resolve() / source
        if not target.exists():
            raise ValueError(f"{runs[label]} holds no {source}")
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).symlink_to(target, target_is_directory=target.is_dir())


def assemble(run, exact, policy):
    """Complete the captured run directory `run`; returns its parity.json value."""
    workload = workload_of(exact)
    if workload_of(policy) != workload:
        raise ValueError("the exact and policy reference runs scored different workloads")
    capture = run / "tap-capture.json"
    if not capture.is_file() or not (run / "engine").is_dir():
        raise ValueError(f"{run} holds no tap capture: run tap_binding.py capture --out {run} first")
    captured = sorted(record["name"] for record in json.loads(capture.read_text())["requests"])
    sequences = sorted(sequence["name"] for sequence in json.loads(workload.read_text())["sequences"])
    if captured != sequences:
        raise ValueError(f"the captured requests {captured} are not the workload's sequences {sequences}")
    combined(run / "reference" / "vision", {"exact": exact, "policy": policy})
    shutil.copyfile(workload, run / "workload.json")
    for sequence in json.loads(workload.read_text())["sequences"]:
        for item in sequence.get("media", []):
            for key in ("tensor", "grid"):
                shutil.copyfile(workload.parent / item[key]["path"], run / item[key]["path"])
    source = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout.strip()
    value = {"schema": SCHEMA, "model": "glm5next", "engine": "engine", "workload": "workload.json",
             "references": ["reference/vision"], "source": source}
    (run / "parity.json").write_text(json.dumps(value, indent=1) + "\n")
    return value


def record(args):
    """requests, the tap capture and assemble against the live server."""
    bodies = args.run.parent / f"{args.run.name}-requests"
    with tempfile.TemporaryDirectory() as images:
        names = write_requests(bodies, Path(images))
    command = [sys.executable, "-B", str(ROOT / "tools/qual/tap_binding.py"), "capture", "--url", args.url,
               "--server-log", str(args.server_log), "--taps-dir", str(args.taps_dir), "--out", str(args.run)]
    for name in names:
        command += ["--request", f"{name}=/v1/chat/completions={bodies / f'{name}.json'}"]
    done = subprocess.run(command, cwd=ROOT, check=False)
    if done.returncode:
        raise ValueError(f"tap_binding.py capture exited {done.returncode}")
    assemble(args.run, args.reference_exact, args.reference_policy)
    print(json.dumps({"encoder_parity_run": str(args.run)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("requests", help="write the probes' request bodies")
    write.add_argument("--dir", type=Path, required=True, help="a new directory for the bodies")
    write.add_argument("--run", type=Path, default=Path("RUN"), help="the run directory the printed capture makes")
    complete = commands.add_parser("assemble", help="complete a captured run directory")
    complete.add_argument("--run", type=Path, required=True)
    complete.add_argument("--reference-exact", type=Path, default=REFERENCE / "vision-exact")
    complete.add_argument("--reference-policy", type=Path, default=REFERENCE / "vision-policy")
    live = commands.add_parser("record", help="requests, capture and assemble against a live taps server")
    live.add_argument("--url", required=True)
    live.add_argument("--server-log", type=Path, required=True)
    live.add_argument("--taps-dir", type=Path, required=True)
    live.add_argument("--run", type=Path, required=True)
    live.add_argument("--reference-exact", type=Path, default=REFERENCE / "vision-exact")
    live.add_argument("--reference-policy", type=Path, default=REFERENCE / "vision-policy")
    args = parser.parse_args()
    try:
        if args.command == "record":
            record(args)
        elif args.command == "requests":
            with tempfile.TemporaryDirectory() as images:
                names = write_requests(args.dir, Path(images))
            bodies = " ".join(f"--request {name}=/v1/chat/completions={args.dir}/{name}.json" for name in names)
            print(f"python3 -B tools/qual/tap_binding.py capture --url URL --server-log LOG --taps-dir TAPS "
                  f"--out {args.run} {bodies}")
        else:
            assemble(args.run, args.reference_exact, args.reference_policy)
            print(f"python3 -B -m tools.qual check encoder_parity --model glm5next.taps --url URL --inputs INPUTS"
                  f"  # INPUTS holds {{\"encoder_parity_run\": \"{args.run}\"}}")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
