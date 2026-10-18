"""Encoder equality across ranks, repeated warm encode timings, and
multimodal smoke.

Set CHOPINFER_QUAL_CONFIG to the model's qualification config and pass each
--declaration in serving encoder-index order. The declarations select the
media workloads, supported parallel mode and large-record geometry checks.
Serve the checkpoint directly at the requested tensor parallel size.
--host-bin must sit in a private install directory beside the family helper
chopinfer-model-<model> (and its bundle beside it or in ../lib), none
of it group or world writable. The helper prepares the package on CPU before
the first server starts. Invoke only on as many idle GPUs as ranks, named
by physical index, until a UTC end time. This runner never acquires a device reservation or stops
an existing process.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import base64
import io
import math
import statistics
import struct
import wave
import zlib
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import socket
import subprocess
import time

from common import thinking_control, input_path, Client, ROOT, qualification_config
import encoder_declarations
from kvo_resume import prepare_direct
from multimodal import cases, messages, load_reference


def chat_body(model, prompt):
    return {"model": model, "messages": prompt, **thinking_control(False),
            "temperature": 0, "max_tokens": 32, "return_token_ids": True}


def compare_arenas(left, right, expected_records=None, ranks=4):
    def captures(root):
        paths = list(root.rglob("*.bin"))
        found = {path.name: path for path in paths}
        if len(paths) != len(found):
            raise ValueError("duplicate arena capture identities across runs")
        return found
    before, after = captures(left), captures(right)
    if not before or before.keys() != after.keys():
        raise ValueError("empty or different encoder arena capture sets")
    groups = {}
    digests = {}
    for name, path in before.items():
        match = re.fullmatch(r"(request-\d+-record-\d+)-rank-(\d+)\.bin", name)
        if not match:
            raise ValueError(f"unexpected arena capture: {name}")
        data = path.read_bytes()
        if not data or data != after[name].read_bytes():
            raise ValueError(f"serial/parallel encoder arena mismatch: {name}")
        digest = hashlib.sha256(data).hexdigest()
        digests[name] = digest
        groups.setdefault(match[1], {})[int(match[2])] = digest
    if (expected_records is not None and len(groups) != expected_records) or any(
            set(g) != set(range(ranks)) or len(set(g.values())) != 1 for g in groups.values()):
        raise ValueError(f"expected complete records with identical output on all {ranks} ranks")
    return digests


def compare_tokens(left, right):
    if left.keys() != right.keys() or not left:
        raise ValueError("empty or different request sets")
    for key, value in left.items():
        if not value["tokens"] or value["tokens"] != right[key]["tokens"]:
            raise ValueError(f"greedy token mismatch: {key}")
        if value["text"] != right[key]["text"]:
            raise ValueError(f"greedy text mismatch: {key}")


def record_fields(log):
    for line in log.splitlines():
        if " encoder " in line:
            fields = dict(re.findall(r"(\w+)=([^ ]+)", line.split(" encoder ", 1)[1]))
            if "encoder" in fields and "rows" in fields:
                yield fields


def parse_timings(log, expected_encoder):
    rows = []
    for fields in record_fields(log):
        if fields["encoder"] != str(expected_encoder):
            raise ValueError(f"unexpected encoder timing: {fields}")
        value = float(fields["elapsed_ms"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid encoder timing")
        rows.append(value)
    if not rows:
        raise ValueError(f"missing uncached encoder {expected_encoder} timing")
    return sum(rows)


def parse_phases(log):
    phases = {name: 0.0 for name in ("prepare_ms", "submit_ms", "drain_ms")}
    records = 0
    for fields in record_fields(log):
        if not all(key in fields for key in phases):
            raise ValueError("missing encoder phase timings")
        for key in phases:
            value = float(fields[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid encoder phase timing")
            phases[key] += value
        records += 1
    if not records:
        raise ValueError("missing encoder record timings")
    return phases


def png_variant(index):
    """Distinct decoded colors ensure each timing sample misses the media cache."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    width, height = 512, 256
    pixels = bytes((20 + index * 7, 40 + index * 5, 180)) * width
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + pixels) * height)) + chunk(b"IEND", b""))


def wav_variant(index, seconds=16):
    """PCM fixtures are generated on CPU, with distinct nonzero waveforms."""
    rate = 24000
    samples = bytearray()
    for i in range(rate * seconds):
        samples.extend(struct.pack("<h", int(3000 * math.sin(2 * math.pi * (220 + 31 * index) * i / rate))))
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(samples)
    return output.getvalue()


def media_message(parts):
    return [{"role": "user", "content": [{"type": "text", "text": "Describe the media briefly."}, *parts]}]


def media_part(kind, data):
    mime = {"image": "image/png", "audio": "audio/wav", "video": "video/mp4"}[kind]
    return {"type": kind + "_url", kind + "_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}


def declared_encoders(paths, model):
    values = [encoder_declarations.read(path) for path in paths]
    expected = {value["path"].resolve() for value in encoder_declarations.for_model(model)}
    if {value["path"].resolve() for value in values} != expected or len(values) != len(expected):
        raise ValueError("pass every model encoder declaration once, in serving encoder-index order")
    if any(value["model"] != model or "record-parallel" not in value["parallel"] for value in values):
        raise ValueError("declarations must belong to the model and support record-parallel")
    if any(value["sequences"]["kind"] not in ("temporal-grid", "segments") for value in values):
        raise ValueError("large-record workloads require temporal-grid or segmented encoders")
    return values


def workloads(video, samples, encoders):
    kinds = {kind for value in encoders for kind in value["kinds"]}
    media = [kind for kind in ("image", "audio") if kind in kinds]
    frozen = [next(case for case in cases() if case["kind"] == kind) for kind in media]
    capture = [(case["kind"], messages(case, input_path("media_fixtures")), 1) for case in frozen]
    if "video" in kinds:
        if not video:
            raise ValueError("video declaration requires --video-fixture spanning at least two encoder chunks")
        capture.append(("video-two-chunks", media_message([media_part("video", video)]), 1))
    if "audio" in kinds:
        capture.append(("audio-two-segments", media_message([media_part("audio", wav_variant(9, 65))]), 1))
    generated = {"image": png_variant, "audio": wav_variant}
    if len(media) > 1:
        capture.append(("two-records", media_message([media_part(kind, generated[kind](8)) for kind in media]),
                        len(media)))
    timing = []
    for index in range(samples + 1):
        for kind in media:
            encoder = next(i for i, value in enumerate(encoders) if kind in value["kinds"])
            timing.append((f"{kind}-{index}", media_message([media_part(kind, generated[kind](index))]),
                           encoder, index == 0))
    return capture, timing


def require_large_geometry(log, encoders):
    records = list(record_fields(log))
    for index, value in enumerate(encoders):
        held = [row for row in records if row["encoder"] == str(index)]
        sequences = value["sequences"]
        if sequences["kind"] == "segments":
            if not any(int(row.get("patches", 0)) > sequences["input_rows"] for row in held):
                raise ValueError(f"no multi-segment {value['name']} record was encoded")
        elif sequences["kind"] == "temporal-grid":
            if not any(int(row.get("runs", 0)) > 1 and
                       int(row.get("patches", 0)) > value["input"]["capacity_rows"] for row in held):
                raise ValueError(f"no multi-chunk {value['name']} record was encoded")
        else:
            raise ValueError(f"unsupported large-record geometry: {sequences}")


def query(kind, fields):
    result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    return [tuple(part.strip() for part in row.split(","))
            for row in result.stdout.splitlines() if row.strip()]


def require_empty(uuids):
    rows = query("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    busy = [row for row in rows if row[0] in uuids]
    if busy:
        raise RuntimeError(f"reserved GPUs are not empty: {busy}")
    return query("gpu", "index,uuid,memory.used")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-bin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, action="append", required=True,
                        help="encoder JSON, repeated in serving encoder-index order")
    parser.add_argument("--model-cache-dir", type=Path,
                        help="model cache for the direct package (default: the host's, $CHOPINFER_DATA/model-cache)")
    parser.add_argument("--go-devices", required=True,
                        help="One idle physical nvidia-smi index per rank")
    parser.add_argument("--tensor-parallel-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--go-until", required=True, help="run end, ISO UTC timestamp")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("CHOPINFER_DATA", Path.home() / "chopinfer-data")) / "encoder-par")
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--video-fixture", type=Path,
                        help="CPU-prepared video with at least 32 frames at 512x512")
    parser.add_argument("--samples", type=int, default=5, help="Warm uncached samples per encoder and mode")
    args, unknown = parser.parse_known_args()
    if "--install-root" in unknown:
        parser.error("--install-root is gone: the runner serves --checkpoint DIR directly (serve --model)")
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.checkpoint is None or not args.checkpoint.is_dir():
        parser.error("--checkpoint must name the checkpoint directory")
    if not 3 <= args.samples <= 8:
        parser.error("--samples must be between 3 and 8")
    config = qualification_config()
    encoders = declared_encoders(args.declaration, config["model"])
    video = args.video_fixture.read_bytes() if args.video_fixture else b""
    capture_work, timing_work = workloads(video, args.samples, encoders)
    end = datetime.fromisoformat(args.go_until.replace("Z", "+00:00"))
    if end.tzinfo is None:
        parser.error("--go-until requires a timezone")
    remaining = (end - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        parser.error("run window has expired")
    deadline = time.monotonic() + remaining - 30
    selected = args.go_devices.split(",")
    ranks = args.tensor_parallel_size
    if len(selected) != ranks or len(set(selected)) != ranks:
        parser.error(f"--go-devices must name {ranks} distinct devices")
    # Every declared encoder is exercised, so the package serves every modality.
    serve = ["serve", "--model", str(args.checkpoint.resolve()), "--tensor-parallel-size", str(ranks),
             "--modalities", "all"]
    if args.model_cache_dir is not None:
        serve += ["--model-cache-dir", str(args.model_cache_dir.resolve())]
    serve += ["--listen", f"127.0.0.1:{args.port}", "--devices", ",".join(map(str, range(ranks))),
              "--speculative-tokens", "0",
              "--max-sequences", "32", "--max-batch-tokens", "8192", "--prefill-chunk-tokens", "2048",
              "--kv-memory-fraction", "0.85"]
    # The package the server will select; the helper prepares it on CPU.
    package = prepare_direct(serve, args.host_bin.resolve().parent / f"chopinfer-model-{config['model']}")
    manifest = Path(package["bundle"]) / "manifest.json"
    provenance = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "host_sha256": hashlib.sha256(args.host_bin.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "package_identity": package.get("package_identity"),
        "checkpoint_identity": package.get("checkpoint_identity"),
        "checkpoint": str(args.checkpoint.resolve()),
        "encoders": [{"name": value["name"], "sha256": hashlib.sha256(value["path"].read_bytes()).hexdigest()}
                     for value in encoders],
        "reference_sha256": load_reference(input_path("media_capture"))[1],
        "go_until": end.isoformat(),
        "video_sha256": hashlib.sha256(video).hexdigest(),
        "samples": args.samples,
        "tensor_parallel_size": ranks,
    }
    mapping = dict(query("gpu", "index,uuid"))
    uuids = [mapping[index] for index in selected]
    root = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True, exist_ok=False)
    (root / "devices-before.json").write_text(json.dumps(require_empty(uuids)))
    records = {}
    timings = {}
    def budget():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("run window ended; stopping owned server")
        return value
    try:
        for capture in (True, False):
            for mode in ("serial", "parallel"):
                name = f"{mode}-{'capture' if capture else 'timing'}"
                run = root / name
                run.mkdir()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids),
                           CHOPINFER_ENCODER_SERIAL=str(int(mode == "serial")),
                           CHOPINFER_QUAL_RECEIPTS=str(run / "receipts"))
                for key in ("CHOPINFER_ENCODER_CAPTURE_DIR", "CHOPINFER_ENCODER_CAPTURE_ALL_LAYERS",
                            "CHOPINFER_ENCODER_ARENA_DIR"):
                    env.pop(key, None)
                if capture:
                    env["CHOPINFER_ENCODER_ARENA_DIR"] = str(run / "arenas")
                command = [str(args.host_bin.resolve()), *serve]
                budget()
                # Refuse an occupied port before starting or sending requests.
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind(("127.0.0.1", args.port))
                with (run / "server.log").open("x") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        client = Client(f"http://127.0.0.1:{args.port}", timeout=1, retries=0)
                        while True:
                            budget()
                            if process.poll() is not None:
                                raise RuntimeError(f"server exited {process.returncode}; see {run / 'server.log'}")
                            try:
                                client.request("/health")
                                models, _ = client.request("/v1/models")
                                model = models["data"][0]["id"]
                                break
                            except Exception:
                                time.sleep(0.5)
                        served = re.findall(r"^startup direct_package: (.*)$", (run / "server.log").read_text(), re.M)
                        if len(served) != 1 or json.loads(served[0]).get("package_identity") != package.get("package_identity"):
                            raise RuntimeError(f"server did not load the prepared package; see {run / 'server.log'}")
                        answers = {}
                        samples_by_encoder = {index: [] for index in range(len(encoders))}
                        workload = capture_work if capture else timing_work
                        for entry in workload:
                            key, prompt = entry[:2]
                            body = chat_body(model, prompt)
                            before = (run / "server.log").stat().st_size
                            previous_files = set((run / "arenas").rglob("*.bin")) if capture else set()
                            client.timeout = min(180, budget())
                            reply, elapsed = client.request("/v1/chat/completions", body)
                            choice = reply["choices"][0]
                            tokens = choice.get("token_ids")
                            if not tokens:
                                raise ValueError("server did not return generated token IDs")
                            answers[key] = {"tokens": tokens, "text": choice["message"].get("content"), "seconds": elapsed}
                            log_delta = (run / "server.log").read_bytes()[before:].decode()
                            if capture:
                                added = set((run / "arenas").rglob("*.bin")) - previous_files
                                if len(added) != entry[2] * ranks:
                                    raise ValueError(f"{key}: expected {entry[2]} new record captures on {ranks} ranks, got {len(added)} files")
                                answers[key]["arena_files"] = sorted(path.name for path in added)
                            else:
                                measured = parse_timings(log_delta, entry[2])
                                answers[key]["encode_ms"] = measured
                                answers[key]["phases"] = parse_phases(log_delta)
                                answers[key]["warmup"] = entry[3]
                                if not entry[3]:
                                    samples_by_encoder[entry[2]].append(measured)
                        if capture:
                            require_large_geometry((run / "server.log").read_text(), encoders)
                        else:
                            timings[mode] = {encoders[kind]["tower"]: {"samples_ms": values, "median_ms": statistics.median(values)}
                                             for kind, values in samples_by_encoder.items()}
                        records[name] = answers
                        (run / "answers.json").write_text(json.dumps(answers, indent=2))
                        if not capture and mode == "parallel":
                            with (run / "smoke.log").open("x") as smoke:
                                subprocess.run([sys.executable, "-B", str(ROOT / "tools/qual/multimodal.py"),
                                                "--url", client.url, "--model", model],
                                               env=env, stdout=smoke, stderr=subprocess.STDOUT,
                                               check=True, timeout=budget())
                    finally:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                release_deadline = time.monotonic() + 20
                while True:
                    try:
                        require_empty(uuids)
                        break
                    except RuntimeError:
                        if time.monotonic() >= release_deadline:
                            raise
                        time.sleep(0.5)
        arenas = compare_arenas(root / "serial-capture/arenas", root / "parallel-capture/arenas",
                                expected_records=sum(entry[2] for entry in capture_work), ranks=ranks)
        for suffix in ("capture", "timing"):
            compare_tokens(records[f"serial-{suffix}"], records[f"parallel-{suffix}"])
        result = {"provenance": provenance, "arenas_sha256": arenas, "encode_ms": timings, "greedy_tokens_equal": True,
                  "multimodal_smoke": "passed", "devices": uuids,
                  "speedup": {kind: timings["serial"][kind]["median_ms"] / timings["parallel"][kind]["median_ms"]
                              for kind in timings["serial"]}}
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        (root / "devices-after.json").write_text(json.dumps(require_empty(uuids)))
        print(f"Selected GPUs are empty. Logs: {root}")


if __name__ == "__main__":
    main()
