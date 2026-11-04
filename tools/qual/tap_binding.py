"""Bind diagnostic tap steps to the serving process and request that wrote them.

A diagnostic-taps bundle's step names its producer in its own metadata:
capture.json `producer.startup_nonce` (the serving process's startup
nonce, which the host passes at create when CHOPINFER_TAPS_DIR is set) and
items.json each item's engine request ID. The server must run with
CHOPINFER_REQUEST_TRACE=1, so each finished request writes a
`chopinfer.request.v1` record mapping its X-Client-Request-Id to that engine ID.

`capture` binds the live server at --url to the gate's checkpoint pin and
its startup log (common.bind_checkpoint), then sends each request once
with a fresh X-Client-Request-Id challenge and cache salt, requires every
response to name the bound process (common.confirm_live) and no cached
prompt tokens, reads the challenge's engine request ID from the bound log,
and admits only the new steps whose producer nonce is the live one and
whose items are all that request's. A prefix hit (--prefix-hit) is the
continuation of a named primer, an earlier cold request: it is sent under
the primer's salt, so the pair shares one fresh salt, and must report
0 < cached_tokens <= the primer's prompt tokens, as determinism.py's
prefix-hit rule requires; its steps must name the process and its own
request like any other. It copies the steps, numbered in order,
into OUT/engine and writes OUT/tap-capture.json: the server's identity
(IDENTITY) and, beside it, its complete model option map and whether the
host marks it qualified (OPTIONS, as /health reports them; a server from
before the report has neither), each request's challenge, engine request ID,
reported prompt and cached tokens, cache salt, primer and steps, and the
digests of every copied file. `check_requests` re-checks a manifest's
cache rules.

The server's options must be its qualified defaults
(common.bind_checkpoint). With --experimental and a sealed pin
(experimental_determinism.py: the complete option map, checkpoint identity
and startup nonce, as determinism.py --experimental takes one) it captures
a server serving exactly the pinned diagnostic options instead, and the
manifest records the pin and `qualified: false`: the capture is
experimental, and no gate admits it.

`admit` (used by the tap gates, qwen38_layers and encoder_parity) re-checks
a capture against the live candidate: every IDENTITY field must equal the
candidate's /health serving identity, startup nonce included, so the
candidate is the process that wrote the steps; each step's files must
hold their recorded digests (read without following links) and its own
producer nonce and engine request IDs must be the recorded ones. An
experimental capture is refused. A missing or differing binding raises
Unbound: the evidence is insufficient.

Usage:
  python3 tools/qual/tap_binding.py capture --url URL --server-log LOG \\
      --taps-dir DIR --out OUT --request NAME=PATH=BODY.json [...] \\
      [--prefix-hit NAME=PRIMER=PATH=BODY.json ...] \\
      [--experimental --experimental-pin PIN --experimental-pin-sha256 SHA256]
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
import common  # noqa: E402
import experimental_determinism  # noqa: E402
from tools.model_sdk import diagnostic as wire  # noqa: E402

SCHEMA = "chopinfer.tap-capture.v1"
MANIFEST = "tap-capture.json"
STEPS = "engine"
TRACE = "chopinfer.request.v1"
# The serving identity a capture and its candidate must share.
IDENTITY = ("bundle_id", "package_identity", "checkpoint_identity", "metadata_identity", "startup_nonce",
            "profile", "version", "source_commit")
# The model options a capture records beside IDENTITY: the complete option
# map /health reports and whether the host marks it qualified.
OPTIONS = ("model_options", "model_options_qualified")
HEX = {"package_identity": 64, "metadata_identity": 64, "checkpoint_identity": 64, "startup_nonce": 32,
       "source_commit": 40}
# How long a finished request's trace record may take to reach the log.
TRACE_WAIT_SECONDS = 10.0


class Unbound(ValueError):
    """A binding is missing or differs: the evidence is insufficient."""


def identity_of(serving):
    """The IDENTITY fields of a /health serving identity, each required."""
    identity = {key: serving.get(key) for key in IDENTITY}
    for key, width in HEX.items():
        if not isinstance(identity[key], str) or not re.fullmatch(f"[0-9a-f]{{{width}}}", identity[key]):
            raise Unbound(f"the server reports no {key}")
    if not isinstance(identity["profile"], dict) or not identity["profile"].get("tensor_parallel_size"):
        raise Unbound("the server reports no serving profile")
    return identity


def options_of(serving):
    """The OPTIONS fields a /health serving identity reports."""
    return {key: serving[key] for key in OPTIONS if key in serving}


def live(url, config, log=None, experimental_pin=None):
    """(client, server, identity) of the live server at `url`, bound to the
    gate's checkpoint pin (and to its startup `log` when given): its model
    options are its qualified defaults, or exactly `experimental_pin`'s."""
    client = common.Client(url, retries=0)
    server = client.identity(common.checkpoint_pin(config)["served_model_name"])
    common.bind_checkpoint(server, log, url, config, experimental_pin=experimental_pin)
    return client, server, identity_of(common.live_identity(server))


def step_files(step):
    """{name: {bytes, sha256}} of a tap step's files, read without following
    links; a step holding anything but regular files is refused."""
    step = wire.external(Path(step).absolute())
    files = {}
    for entry in sorted(os.scandir(step), key=lambda entry: entry.name):
        if not entry.is_file(follow_symlinks=False):
            raise Unbound(f"{step}/{entry.name} is not a regular file")
        files[wire.component(entry.name)] = wire.identity(step / entry.name)
    return files


def producer(step):
    """(startup nonce, engine request IDs) a step's own metadata names."""
    capture = wire.read_json(step / "capture.json")
    items = wire.read_json(step / "items.json")["items"]
    nonce = (capture.get("producer") or {}).get("startup_nonce")
    return nonce, {item["request"] for item in items}


def verify_step(step, nonce, requests):
    """Refuses a step another process, or another request, wrote."""
    written, ids = producer(step)
    if written != nonce:
        raise Unbound(f"{step} was written by process {written}, not {nonce}")
    if not ids or not ids <= set(requests):
        raise Unbound(f"{step} holds engine requests {sorted(ids)}, not {sorted(requests)}")


def engine_requests(log, url, challenge, nonce):
    """The engine request IDs the live server's startup log traces for the
    X-Client-Request-Id `challenge`; the log must be the process with `nonce`."""
    deadline = time.monotonic() + TRACE_WAIT_SECONDS
    while True:
        logged = common.served_checkpoint(log, url)
        if logged["startup_nonce"] != nonce:
            raise Unbound(f"{log} is not the log of the process with startup nonce {nonce}")
        found = set()
        for line in Path(log).read_text(errors="replace").splitlines():
            if f'"schema":"{TRACE}"' not in line.replace(" ", ""):
                continue
            record = json.loads(line[line.index("{"):])
            if record.get("schema") == TRACE and record.get("request_id") == challenge:
                found.add(record["engine_request_id"])
        if found or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    if len(found) != 1:
        raise Unbound(f"{log} traces {len(found)} engine requests for {challenge}; serve with "
                      "CHOPINFER_REQUEST_TRACE=1")
    return found


def primed(name, primer, earlier):
    """The record of `primer` among the `earlier` requests, which the prefix
    hit `name` continues: a cold request that primes no other."""
    cold = [record for record in earlier if record.get("name") == primer and record.get("primer") is None]
    if len(cold) != 1:
        raise Unbound(f"{name}: names no single earlier cold request {primer} as its primer")
    if other := next((record["name"] for record in earlier if record.get("primer") == primer), None):
        raise Unbound(f"{name}: {primer} already primes {other}")
    return cold[0]


def check_request(record, earlier):
    """Refuses a request that breaks the cache rules given the `earlier`
    requests of its capture: a cold request carries its own cache salt and
    reports 0 cached prompt tokens; a prefix hit reuses its primer's salt and
    reports 0 < cached_tokens <= the primer's prompt tokens."""
    name, salt, primer, cached = (record.get(key) for key in ("name", "cache_salt", "primer", "cached_tokens"))
    if not isinstance(salt, str) or not salt:
        raise Unbound(f"{name}: records no cache salt")
    if primer is None:
        if any(other.get("cache_salt") == salt for other in earlier):
            raise Unbound(f"{name}: shares its cache salt with an earlier request")
        if type(cached) is not int or cached != 0:
            raise Unbound(f"{name}: the server reused {cached} cached prompt tokens")
        return
    first = primed(name, primer, earlier)
    if salt != first.get("cache_salt"):
        raise Unbound(f"{name}: does not reuse the cache salt of its primer {primer}")
    length = first.get("prompt_tokens")
    if not (type(cached) is int and type(length) is int and 0 < cached <= length):
        raise Unbound(f"{name}: a prefix hit requires reported 0 < cached_tokens <= {length} (the prompt tokens "
                      f"of its primer {primer}), got {cached}")


def check_requests(records):
    """Refuses a capture's request records that break the cache rules."""
    for at, record in enumerate(records):
        check_request(record, records[:at])


def capture(url, log, taps, requests, out, config, subject=None, experimental_pin=None):
    """Sends `requests` ((name, path, body) each, or (name, path, body,
    primer) for a prefix hit continuing the earlier cold request `primer`)
    once to the bound server at `url` and writes the capture of their new
    steps in `out`. `subject` is recorded beside them (what the steps were
    captured against). With `experimental_pin` the server serves the pinned
    options, and the capture is experimental."""
    client, server, identity = live(url, config, log, experimental_pin)
    taps = wire.external(Path(taps).absolute())
    out = Path(out).absolute()
    out.mkdir(parents=True)
    wire.external(out)
    (out / STEPS).mkdir()
    records, files, count = [], {}, 0
    for name, path, body, *hit in requests:
        primer = hit[0] if hit else None
        salt = primed(name, primer, records)["cache_salt"] if primer else f"tap-capture/{uuid.uuid4().hex}"
        before = {step.name for step in taps.glob("step*")}
        challenge = f"tap-capture-{uuid.uuid4().hex}"
        body = {"model": server["model"], **body, "cache_salt": salt}
        response, _ = client.request(path, body, headers={"X-Client-Request-Id": challenge})
        common.confirm_live(client, server, experimental_pin=experimental_pin)
        usage = response.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        reported = {"prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached, "cache_salt": salt,
                    "primer": primer}
        check_request({"name": name, **reported}, records)
        engine = engine_requests(log, url, challenge, identity["startup_nonce"])
        new = sorted(step for step in taps.glob("step*") if step.name not in before)
        if not new:
            raise Unbound(f"{name}: {taps} gained no step")
        steps = []
        for step in new:
            verify_step(step, identity["startup_nonce"], engine)
            target = out / STEPS / f"step{count:04d}"
            count += 1
            target.mkdir()
            for file in step_files(step):
                with wire.opened(step / file) as (stream, _), (target / file).open("xb") as copy:
                    shutil.copyfileobj(stream, copy)
            files[target.name] = step_files(target)
            steps.append(target.name)
        records.append({"name": name, "path": path, "challenge": challenge, "engine_request_id": engine.pop(),
                        "response_id": response.get("id"), **reported, "steps": steps})
    manifest = {"schema": SCHEMA, "server": identity | options_of(common.live_identity(server)),
                "server_log_sha256": server["checkpoint"]["server_log_sha256"], "url": url,
                "requests": records, "files": files, "subject": subject}
    if experimental_pin is not None:
        manifest |= {experimental_determinism.MARKER: experimental_pin.marker(), "qualified": False}
    wire.write_new(out / MANIFEST, manifest)
    return manifest


def admit(directory, url, config):
    """The capture in `directory`, reverified against the live candidate at
    `url`: its manifest, with every step bound to its recorded process and
    request."""
    directory = wire.external(Path(directory).absolute())
    manifest = wire.read_json(directory / MANIFEST)
    if manifest.get("schema") != SCHEMA:
        raise Unbound(f"{directory}/{MANIFEST} is not {SCHEMA}")
    if experimental_determinism.MARKER in manifest or manifest.get("qualified", True) is not True:
        raise Unbound(f"{directory} is an experimental capture of pinned diagnostic options, which no gate admits")
    recorded = identity_of(manifest["server"])
    if url is None:
        raise Unbound("no live candidate to bind the capture to (--url)")
    _, _, candidate = live(url, config)
    for key in IDENTITY:
        if candidate[key] != recorded[key]:
            raise Unbound(f"the candidate's {key} differs from the capture's")
    steps = sorted(step.name for step in (directory / STEPS).glob("step*"))
    named = [step for record in manifest["requests"] for step in record["steps"]]
    if steps != sorted(named) or sorted(manifest["files"]) != sorted(named):
        raise Unbound(f"{directory}/{STEPS} holds other steps than its manifest records")
    for record in manifest["requests"]:
        for name in record["steps"]:
            step = directory / STEPS / name
            if step_files(step) != manifest["files"][name]:
                raise Unbound(f"{step} differs from the files its manifest records")
            verify_step(step, recorded["startup_nonce"], {record["engine_request_id"]})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("capture", help="send requests and capture their bound steps")
    make.add_argument("--url", required=True)
    make.add_argument("--server-log", required=True)
    make.add_argument("--taps-dir", required=True)
    make.add_argument("--out", required=True)
    make.add_argument("--request", action="append", dest="sends", required=True, metavar="NAME=PATH=BODY.json",
                      type=lambda text: (text, False), help="a cold request")
    make.add_argument("--prefix-hit", action="append", dest="sends", metavar="NAME=PRIMER=PATH=BODY.json",
                      type=lambda text: (text, True),
                      help="a prefix hit: the continuation of PRIMER, an earlier --request, under its cache salt")
    make.add_argument("--experimental", action="store_true",
                      help="capture a server serving the sealed pin's diagnostic options; no gate admits it")
    make.add_argument("--experimental-pin", type=Path,
                      help="the sealed complete option map, checkpoint identity and startup nonce")
    make.add_argument("--experimental-pin-sha256", help="SHA256 of the sealed pin file")
    args = parser.parse_args()
    requests = []
    for text, hit in args.sends:
        name, *primer, path, body = text.split("=", 3 if hit else 2)
        requests.append((name, path, json.loads(Path(body).read_text()), *primer))
    config = common.qualification_config()
    try:
        pin = experimental_determinism.selected(args)
        manifest = capture(args.url, args.server_log, args.taps_dir, requests, args.out, config,
                           experimental_pin=pin)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"capture refused: {error}", file=sys.stderr)
        return 3
    print(json.dumps({record["name"]: record["engine_request_id"] for record in manifest["requests"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
