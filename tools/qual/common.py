"""Small OpenAI-compatible qualification client with timed retries and receipts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import shlex
import sys
from functools import lru_cache
from pathlib import Path
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import declaration

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ENV = next((path for path in (ROOT / "local.env", Path(os.environ.get(
    "CHOPINFER_LOCAL_ENV",
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "chopinfer/local.env")))
    if path.is_file()), None)
for line in LOCAL_ENV.read_text().splitlines() if LOCAL_ENV else []:
    if line.startswith("export ") and "=" in line:
        key, value = shlex.split(line[7:])[0].split("=", 1)
        if key.startswith("CHOPINFER_"):
            os.environ.setdefault(key, os.path.expandvars(value))
# CHOPINFER_QUAL_RECEIPTS selects a separate directory for receipts.
RECEIPTS = None


@lru_cache(maxsize=None)
def read_config(path):
    """A JSON file; a model's qual.json reads as the gate config it declares."""
    return declaration.load(path)


def qualification_config():
    path = os.environ.get("CHOPINFER_QUAL_CONFIG")
    if not path:
        raise ValueError("set CHOPINFER_QUAL_CONFIG to a model's qual.json or a gate config")
    declaration.require_rooted(path, reference_root())
    return read_config(str(Path(path).resolve()))


def resolve_input(name, optional=False, config=None, root=None):
    config = qualification_config() if config is None else config
    value = config["receipts"] if name == "receipts" else config["inputs"].get(name)
    if value is None:
        if optional:
            return None
        raise ValueError(f"gate config must declare input {name}")
    path = declared_path(value, reference_root(root))
    if path is None and not optional:
        raise ValueError(f"input {name} needs {value['env']}")
    return path


def input_path(name, optional=False, config=None, root=None):
    config = qualification_config() if config is None else config
    path = resolve_input(name, optional, config, root)
    if path is not None and name != "receipts":
        from refs import verify_input
        if not optional or path.exists():
            verify_input(config, name, path, reference_root(root))
    return path


def input_bytes(name, config=None, root=None):
    config = qualification_config() if config is None else config
    path = resolve_input(name, config=config, root=root)
    from refs import load_lock, verify_input
    data = verify_input(config, name, path, reference_root(root))
    if data is None:
        if name not in ("receipts",) and load_lock(config, root) is not None:
            raise ValueError(f"lock entry {name}: file loader requires authenticated bytes")
        return path.read_bytes()
    return data


def declared_path(value, root=ROOT):
    """A declared path: a string relative to `root` (or absolute), or
    {"env", "default", "path"} under the environment variable `env` (else
    `default`); None when neither is set."""
    if isinstance(value, dict):
        base = os.environ.get(value["env"]) or value.get("default")
        if base is None:
            return None
        value = str(Path(base) / value.get("path", ""))
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def receipt_root():
    if RECEIPTS is not None:
        return Path(RECEIPTS)
    if override := os.environ.get("CHOPINFER_QUAL_RECEIPTS"):
        return Path(override)
    if os.environ.get("CHOPINFER_QUAL_CONFIG"):
        return input_path("receipts")
    data = os.environ.get("CHOPINFER_DATA")
    if not data:
        raise ValueError("set CHOPINFER_DATA or CHOPINFER_QUAL_RECEIPTS for receipts")
    return Path(data) / "qual"


def startup(model_facts=False, directory=None):
    """Resolve declarations and verify receipt writes before doing any work."""
    if model_facts or os.environ.get("CHOPINFER_QUAL_CONFIG"):
        config = qualification_config()
        if not isinstance(config.get("inputs"), dict) or not isinstance(config.get("qualification"), dict):
            raise ValueError("gate config needs inputs and qualification")
    if model_facts:
        thinking_kwargs(False)
    directory = receipt_root() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=directory) as probe:
        probe.write(b"receipt preflight")
        probe.flush()


class QualificationParser(argparse.ArgumentParser):
    def __init__(self, *args, model_facts=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_facts = model_facts

    def parse_args(self, *args, **kwargs):
        parsed = super().parse_args(*args, **kwargs)
        startup(self.model_facts(parsed) if callable(self.model_facts) else self.model_facts)
        return parsed


def frontend():
    """The gate's frontend declaration (input `frontend`)."""
    return declaration.parse(json.loads(input_bytes("frontend")),
                             declared=resolve_input("frontend").name == declaration.NAME)


def frontend_reasoning():
    return frontend()["reasoning"]


def thinking_kwargs(enabled):
    keywords = frontend_reasoning()["thinking_keywords"]
    if not keywords:
        raise ValueError("frontend has no thinking switch keyword")
    return {keywords[0]: enabled}


def thinking_can_be_disabled():
    """Whether the frontend admits a request that disables thinking
    (`reasoning.allow_disabled`, true when absent); one that does not
    refuses it."""
    return frontend_reasoning().get("allow_disabled", True) is not False


def answer_kwargs():
    """chat_template_kwargs for a request that wants an answer rather than
    reasoning: thinking disabled, or nothing when the frontend refuses that,
    so the request reasons as the frontend's default does. Such a request
    needs a budget for the reasoning before its answer (REASONING_TOKENS)."""
    return thinking_kwargs(False) if thinking_can_be_disabled() else {}


# The output budget of an answer that may follow reasoning (answer_kwargs).
REASONING_TOKENS = 4096


def answer_budget(tokens):
    """The output budget of an answer_kwargs request whose answer needs
    `tokens`: at least REASONING_TOKENS when the frontend refuses disabled
    thinking, so the answer can follow the reasoning."""
    return tokens if thinking_can_be_disabled() else max(tokens, REASONING_TOKENS)



def arguments(description, model_facts=False):
    parser = QualificationParser(description=description, model_facts=model_facts)
    parser.add_argument("--url", default="http://127.0.0.1:8101")
    parser.add_argument("--model", help="Defaults to the first model from /v1/models")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--mechanics-only", action="store_true")
    return parser


# The response headers a server with a serving identity names itself with.
NONCE_HEADER, CHECKPOINT_HEADER = "x-chopinfer-startup-nonce", "x-chopinfer-checkpoint-identity"


class Client:
    def __init__(self, url, timeout=900, retries=3):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        # The (checkpoint identity, startup nonce) every response named.
        self.responders = set()

    def responded(self, headers):
        self.responders.add((headers.get(CHECKPOINT_HEADER), headers.get(NONCE_HEADER)))

    def request(self, path, body=None, stream=False, headers=None):
        payload = None if body is None else json.dumps(body).encode()
        headers = {**({"Content-Type": "application/json"} if payload is not None else {}), **(headers or {})}
        if key := os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        last_error = None
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                request = urllib.request.Request(self.url + path, data=payload, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self.responded(response.headers)
                    if stream:
                        events = []
                        for line in response:
                            if not line.startswith(b"data: "):
                                continue
                            at = time.monotonic()
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                break
                            events.append((at, json.loads(data)))
                        return events, time.monotonic() - started
                    return json.load(response), time.monotonic() - started
            except urllib.error.HTTPError as error:
                self.responded(error.headers)
                detail = error.read().decode(errors="replace")[:1000]
                last_error = f"HTTP {error.code}: {detail}"
                if error.code not in (429, 503):
                    break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                last_error = f"{type(error).__name__}: {error}"
            if attempt < self.retries:
                time.sleep(min(0.5 * 2**attempt, 4))
        raise RuntimeError(last_error)

    def identity(self, model=None):
        models, _ = self.request("/v1/models")
        available = [row["id"] for row in models["data"]]
        if not available or (model is not None and model not in available):
            raise ValueError(f"model {model!r} is unavailable: {available}")
        try:
            health, _ = self.request("/health")
        except (RuntimeError, ValueError):
            health = None
        return {"url": self.url, "model": model or available[0], "models": available,
                "health": health}


def checkpoint_pin(config=None):
    """The gate's checkpoint pin (`qualification.checkpoint`), checked against
    the repository's checkpoint registry and source lock it names. A model
    with one checkpoint has no registry: its lock is that checkpoint's file
    list and the pin itself names the repository and revision."""
    config = qualification_config() if config is None else config
    pin = config["qualification"]["checkpoint"]
    files = json.loads((ROOT / pin["lock"]).read_text())
    if "registry" in pin:
        registry = json.loads((ROOT / pin["registry"]).read_text())
        files = files[pin["pin"]]
        entry = registry["checkpoints"][pin["pin"]]
        if registry["default"] != pin["pin"] or any(
                entry[key] != pin[key] for key in ("repository", "revision", "served_model_name")):
            raise ValueError(f"the gate's checkpoint pin differs from {pin['registry']}")
    lock = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if lock != pin["source_lock_sha256"]:
        raise ValueError(f"the gate's checkpoint pin differs from {pin['lock']}")
    return pin


def workload_sha256(texts):
    """The digest a reference records over its scored workload."""
    workload = [[text["name"], text["score_from"], text["prompt_token_ids"]] for text in texts]
    return hashlib.sha256(json.dumps(workload, separators=(",", ":")).encode()).hexdigest()


def reference_root(root=None):
    return Path(root or os.environ.get("CHOPINFER_QUAL_MODEL_ROOT", ROOT))


def reference_path(kind, config=None, root=None):
    """Resolve a pinned reference through the same path contract as inputs."""
    config = qualification_config() if config is None else config
    spec = config["qualification"]["references"][kind]["path"]
    path = declared_path(spec, reference_root(root))
    if path is None:
        raise ValueError(f"the {kind} reference needs {spec['env']}")
    return path


def pinned_kind(path, config=None, root=None):
    """The kind of the gate's pinned reference stored at `path`, or None when
    the gate pins no reference there."""
    config = qualification_config() if config is None else config
    resolved = Path(path).resolve()
    for kind, spec in config.get("qualification", {}).get("references", {}).items():
        declared = declared_path(spec["path"], reference_root(root))
        if declared is not None and declared.resolve() == resolved:
            return kind
    return None


def pinned_bytes(kind, path, config=None):
    config = qualification_config() if config is None else config
    from refs import verify_input
    data = verify_input(config, "reference:" + kind, Path(path), reference_root())
    return Path(path).read_bytes() if data is None else data


def pinned_reference(kind, path, config=None):
    """The gate's pinned exact reference `kind` read from `path`. Its bytes,
    schema, workload and variants must be the pinned ones and its provenance
    must name the pinned checkpoint, so a stale or foreign reference fails.
    A reference older than the checkpoint block names only the revision
    (`checkpoint_keys`), as a string or `checkpoint_revision`, and carries
    its variants on every text; a pin without `variants` (image probes) has
    none to check. A reference cache binding must name the pinned checkpoint
    (`pinned_binding`)."""
    config = qualification_config() if config is None else config
    spec = config["qualification"]["references"][kind]
    data = pinned_bytes(kind, path, config)
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"{path} is not the pinned {kind} reference")
    value = json.loads(data)
    pin = checkpoint_pin(config)
    provenance = value.get("provenance") or {}
    stated = provenance.get("checkpoint") or {"revision": provenance.get("checkpoint_revision")}
    if isinstance(stated, str):
        stated = {"revision": stated}
    if value.get("schema") != spec["schema"] or any(
            stated.get(key) != pin[key] for key in spec.get("checkpoint_keys", PINNED_CHECKPOINT_KEYS)):
        raise ValueError(f"{path}: schema or checkpoint provenance differs from the pin")
    if "variants" in spec:
        declared = list(provenance["variants"]) if "variants" in provenance else [
            variant for variant in spec["variants"]
            if all(isinstance(text.get(variant), dict) for text in value["texts"])]
        if declared != spec["variants"]:
            raise ValueError(f"{path}: variants differ from the pinned policy")
    if "workload_sha256" in spec and workload_sha256(value["texts"]) != spec["workload_sha256"]:
        raise ValueError(f"{path}: workload differs from the pinned one")
    if "capture_sha256" in spec and value.get("capture_sha256") != spec["capture_sha256"]:
        raise ValueError(f"{path}: scores another capture")
    pinned_binding(kind, value, config)
    return value


def pinned_binding(kind, value, config=None, checkpoint=None):
    """The reference cache binding the gate's pinned reference `kind`
    records. Its bytes are pinned by SHA-256 (`pinned_reference`), so the
    binding is read, not verified against the reference cache again on each
    read (`tools.refrun verify-binding` re-verifies one; adoption verifies it
    with `verified_binding`). It must name the gate's checkpoint pin (lock
    file, variant and digest). A reference pinned without a spread variant
    (variants `["exact"]`) is still verified on each read: its floor-only
    gate has no spread, so its binding is its producer authentication."""
    config = qualification_config() if config is None else config
    spec = config["qualification"]["references"][kind]
    binding = value.get("binding")
    if binding is None:
        return refcache().unbound()
    binding_names_pin(kind, binding, config)
    if spec.get("variants") == ["exact"]:
        return verified_binding(kind, value, config, checkpoint)
    return recorded_binding(value)


def binding_names_pin(kind, binding, config):
    """Refuse a binding of reference `kind` that does not name the gate's
    checkpoint pin (lock file, variant and digest)."""
    pin = checkpoint_pin(config)
    stated = (binding.get("lock"), binding.get("digests", {}).get("lock"))
    if stated != ({"path": pin["lock"], "variant": pin.get("pin")}, pin["source_lock_sha256"]):
        raise refcache().StaleReference("lock", f"the {kind} reference is bound to another checkpoint lock than the pin")


def recorded_binding(value):
    """A reference's binding as its fixture records it, unverified: unbound
    without a binding block, else bound with its refset digest."""
    cache = refcache()
    binding = value.get("binding")
    if binding is None:
        return cache.unbound()
    if not isinstance(binding, dict) or binding.get("schema") != cache.BINDING:
        raise cache.StaleReference("schema", f"binding schema is not {cache.BINDING}")
    return {"status": "bound", "refset_sha256": binding["refset_sha256"], "cache": None}


def verified_binding(kind, value, config=None, checkpoint=None, *, authenticate_workload=True):
    """The reference cache binding of reference `kind`, verified
    (`reference_binding`) against the workload it scored, recomputed from
    the repository: a fixture cannot vouch for its own workload. The
    workload is `references.KIND.workload` when the declaration names one (a
    repo path, or an environment-relative path for a workload whose media
    are too large for the repository), else `reference_workload` of the
    reference."""
    config = qualification_config() if config is None else config
    spec = config["qualification"]["references"][kind]
    if value.get("binding") is None:
        return reference_binding(value, checkpoint)
    cache = refcache()
    workload = declared_path(spec["workload"]) if "workload" in spec else reference_workload(reference_path(kind, config, ROOT))
    if workload is None:
        raise cache.StaleReference("workload", f"the bound {kind} reference's workload needs "
                                               f"{spec['workload']['env']}")
    if not workload.is_file():
        raise cache.StaleReference("workload", f"the bound {kind} reference's workload {workload} is missing")
    from refs import verify_input
    if authenticate_workload:
        verify_input(config, "workload:" + kind, workload, reference_root())
    return reference_binding(value, checkpoint, workload)


def reference_workload(path):
    """The canonical workload of a bound reference: the file beside it named
    `<stem>.workload.json`, where the runner's fixture emission writes it."""
    path = Path(path)
    return path.with_name(path.stem + ".workload.json")


def refcache():
    """tools/refrun/cache.py, shared with the reference runner."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from tools.refrun import cache
    return cache


def reference_binding(value, checkpoint=None, workload=None):
    """The reference cache binding of a reference fixture. A fixture without
    a binding block (every tracked fixture until the runner re-emits it)
    keeps its other checks and is reported {"status": "unbound"}. A binding
    is verified against the working tree, the checkpoint directory when
    given (upstream reference code), the workload file when given, and the
    cache under
    $CHOPINFER_DATA/refcache/v1 when present (scores); a difference raises
    StaleReference, a ValueError whose message starts with stale_reference
    and the differing field."""
    cache = refcache()
    root = cache.default_root()
    store = cache.Store(root) if root is not None and root.is_dir() else None
    return cache.verify(value, ROOT, checkpoint, store, workload)


PINNED_CHECKPOINT_KEYS = ("repository", "revision", "served_model_name", "source_lock_sha256",
                          "checkpoint_identity")
# Startup records that name the checkpoint a server loaded.
CHECKPOINT_RECORDS = re.compile(
    r"^startup (direct_package|direct_load|weight_plan|serving_identity): (\{.*\})$", re.M)


# A startup nonce as the host draws it: 128 random bits in hex.
STARTUP_NONCE = re.compile(r"[0-9a-f]{32}")


def startup_nonce(value):
    return isinstance(value, str) and STARTUP_NONCE.fullmatch(value) is not None


MODALITIES = ("text", "image", "video", "audio")


def served_modalities(server):
    """The modalities the server's `/health` reports it serves (`chopinfer
    serve --modalities`), or None when it reports none: a server from before
    the report serves every modality its package declares, so its media
    checks run and fail on any it refuses."""
    health = (server or {}).get("health")
    value = health.get("modalities") if isinstance(health, dict) else None
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        return None
    return frozenset(value)


def not_applicable(server, modality):
    """Why a check of `modality` does not apply to the server: the reason it
    does not serve that modality, or None when it does (or does not say).
    Such a check is not applicable, never passed."""
    served = served_modalities(server)
    if served is None or modality in served:
        return None
    listed = ",".join(name for name in MODALITIES if name in served)
    return f"the server does not serve {modality} input (--modalities {listed})"


def require_qualified_model_options(server):
    """Refuse advertised options unless the host marks its defaults qualified.
    Legacy servers and receipts omit both fields."""
    serving = ((server.get("health") or {}).get("serving")
               or server.get("serving") or server)
    if "model_options" not in serving and "model_options_qualified" not in serving:
        return
    options = serving.get("model_options")
    if not isinstance(options, dict) or not options or serving.get("model_options_qualified") is not True:
        raise ValueError("server model options are non-default, unqualified, or incompletely reported")


def live_identity(server):
    """The serving identity the live server reported from /health when
    `server` was taken: the checkpoint it loaded and its startup nonce."""
    serving = (server.get("health") or {}).get("serving")
    if not isinstance(serving, dict) or not serving.get("checkpoint_identity") or not startup_nonce(
            serving.get("startup_nonce")):
        raise ValueError(f"the server at {server.get('url')} reports no checkpoint identity and startup nonce "
                         "in /health")
    return serving


def served_checkpoint(log, url):
    """The checkpoint identity and startup nonce the server listening at
    `url` reported in its startup log: every direct package, weight plan,
    direct load and serving identity record must name the same checkpoint,
    and exactly one serving identity record names the nonce."""
    text = Path(log).read_text(errors="replace")
    address = urllib.parse.urlsplit(url).netloc
    listening = [json.loads(line) for line in text.splitlines()
                 if line.startswith("{") and '"status":"listening"' in line]
    if not listening or listening[-1].get("address") != address:
        raise ValueError(f"{log} is not the log of the server at {address}")
    records = [(kind, json.loads(record)) for kind, record in CHECKPOINT_RECORDS.findall(text)]
    identities = {record.get("checkpoint_identity") for _, record in records}
    if len(identities) != 1 or None in identities:
        raise ValueError(f"{log} names no single checkpoint identity: {sorted(map(str, identities))}")
    nonces = [record.get("startup_nonce") for kind, record in records if kind == "serving_identity"]
    if len(nonces) != 1 or not startup_nonce(nonces[0]):
        raise ValueError(f"{log} names no single startup nonce")
    return {"checkpoint_identity": identities.pop(), "startup_nonce": nonces[0], "server_log": str(log),
            "server_log_sha256": hashlib.sha256(Path(log).read_bytes()).hexdigest()}


def bind_checkpoint(server, log, url, config=None):
    """Record in `server` the checkpoint the live server reports from
    /health, refusing any but the gate's pin. A startup log, when given, must
    be the live process's: the same checkpoint and startup nonce, so a log
    another process left at the same address vouches for nothing. A display
    name alone does not identify a checkpoint."""
    pin = checkpoint_pin(config)
    live = live_identity(server)
    require_qualified_model_options(server)
    if live["checkpoint_identity"] != pin["checkpoint_identity"]:
        raise ValueError(f"the server loaded checkpoint {live['checkpoint_identity']}, "
                         f"not the pinned {pin['checkpoint_identity']}")
    if server.get("model") != pin["served_model_name"]:
        raise ValueError(f"the server serves {server.get('model')}, not {pin['served_model_name']}")
    evidence = {key: live.get(key) for key in ("checkpoint_identity", "package_identity", "metadata_identity",
                                               "startup_nonce")}
    if log is not None:
        logged = served_checkpoint(log, url)
        if (logged["checkpoint_identity"], logged["startup_nonce"]) != (live["checkpoint_identity"],
                                                                        live["startup_nonce"]):
            raise ValueError(f"{log} is not the log of the live server at {url}: its checkpoint or startup "
                             "nonce differs from /health")
        evidence.update(server_log=logged["server_log"], server_log_sha256=logged["server_log_sha256"])
    server["checkpoint"] = evidence
    return server


def serving_profile(server):
    """The topology and speculation the live server reported in /health when
    `server` was taken (tensor parallel size, speculation mode, tokens and
    drafter, KV dtype, prefix caching, sequence cap, profile name)."""
    return live_identity(server).get("profile")


def bound_live(server, config=None):
    """Whether a recorded `server` was bound live to the gate's pin: it names
    the pinned checkpoint and a startup nonce, and was not rescored offline."""
    checkpoint = server.get("checkpoint") or {}
    try:
        require_qualified_model_options(server)
    except ValueError:
        return False
    return (checkpoint.get("checkpoint_identity") == checkpoint_pin(config)["checkpoint_identity"]
            and startup_nonce(checkpoint.get("startup_nonce")) and "rescored_from" not in server)


def confirm_live(client, server):
    """Refuse results scored on a bound server unless the bound process wrote
    every response: each one's identity headers, and /health read again after
    scoring, must name the bound checkpoint and startup nonce. A response
    without them, or from another process (after an address reuse, or behind
    a proxy or shared listener), refuses the run. An unbound `server` has
    nothing to confirm."""
    bound = server.get("checkpoint")
    if not bound or "startup_nonce" not in bound:
        return
    try:
        identity = client.identity(server["model"])
        require_qualified_model_options(identity)
        live = live_identity(identity)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"refused: the server at {server.get('url')} no longer reports its identity: {error}")
    expected = (bound["checkpoint_identity"], bound["startup_nonce"])
    if (live["checkpoint_identity"], live["startup_nonce"]) != expected:
        raise SystemExit(f"refused: the server at {server.get('url')} changed while scoring: it now reports "
                         f"checkpoint {live['checkpoint_identity']} and startup nonce {live['startup_nonce']}")
    if any(nonce is None for _, nonce in client.responders):
        raise SystemExit(f"refused: responses from the server at {server.get('url')} lack the {NONCE_HEADER} and "
                         f"{CHECKPOINT_HEADER} headers: the server predates per-response identity headers; "
                         "rebuild it from the commit \"Name the serving process on every HTTP response\" or later")
    if others := client.responders - {expected}:
        raise SystemExit(f"refused: responses from the server at {server.get('url')} named another process "
                         f"(checkpoint, startup nonce): {sorted(others, key=str)}")


# Host load labels a receipt's throughput and latency figures; it never
# changes a verdict. The host is idle for timing evidence when, over every
# interval between samples, the CPU idle fraction (idle plus iowait over all
# CPU time, so niced work counts as busy) is at least IDLE_CPU_FRACTION and
# the shares of elapsed time in which some task stalled waiting for a CPU and
# for I/O (the kernel's pressure stall information: the "some" totals of
# /proc/pressure/cpu and /proc/pressure/io) are each at most
# IDLE_PRESSURE_SHARE. Otherwise, or without two samples or pressure stall
# information, the figures are informational.
#
# Every judged quantity is a cumulative counter differenced between samples,
# so only what ran in the measured window counts. The 1-minute load average
# is recorded but not judged: it is an exponential average with a 60 s time
# constant, so it carries minutes of work from before the window. On
# 2026-10-17 a 1 s sampler classed every thread of the 48-CPU host through a
# Qwen3.8 spec soak with the shared CPU launcher paused. When the check
# started the load average was 10.35, nearly all of it the server's own
# startup (17 threads blocked on weight reads; I/O pressure 0.57 over its
# 45 s). During the soak no server thread was in uninterruptible sleep, the
# server kept one CPU busy, the load average decayed to 1.05, and the
# pressure shares were 0.0007 (CPU) and 0.0010 (I/O).
#
# The limits say what an idle host leaves to others. The run takes a few
# cores (one in that soak), so at least 90% idle leaves under a tenth of the
# CPUs to other work competing for cores and memory bandwidth. With idle
# CPUs the run's threads do not wait for one, and after startup its work
# does not wait on storage; incidental reads and log writes stall for
# milliseconds. A stall in a tenth of the window is therefore other work
# keeping CPUs or storage busy. iowait counts as idle CPU because the I/O
# pressure judges I/O.
#
# Every interval must be valid: the CPU tick totals advance with the idle
# ticks between none and all of them, the elapsed time advances, and the
# pressure stall totals do not go back. An invalid interval makes the timing
# informational; it is never skipped, since it could hide the busy one.
# Elapsed time is the host's uptime (/proc/uptime: monotonic, 10 ms steps),
# not wall time, which can step. The counters are integers below 2**64,
# differenced before any conversion to a float, as host_load.rs does.
#
# A receipt is judged over the samples its check takes around its timed
# work, the first at its start and the last at its end. Without them the
# timing is informational: a window from the check's start would average
# warm-up and setup into it and dilute contention during the timed work.
IDLE_CPU_FRACTION = 0.9
IDLE_PRESSURE_SHARE = 0.1
# crates/runtime-eval/src/capture/host_load.rs applies the same rule to the
# receipts its captures write.
HOST_LOAD_RULE = ("idle when, over every interval between samples, the CPU idle fraction is at least 0.9 and "
                  "the CPU and I/O pressure stall shares are at most 0.1; otherwise throughput and latency are "
                  "informational")
PRESSURE_KINDS = (("cpu", "CPU"), ("io", "I/O"))
UNBOUNDED_TIMING = "no host load samples bound the timed work"


def pressure_some_us():
    """The cumulative microseconds in which some task stalled waiting for a
    CPU and for I/O ({"cpu", "io"}), or None without /proc/pressure."""
    value = {}
    try:
        for kind, _ in PRESSURE_KINDS:
            fields = Path(f"/proc/pressure/{kind}").read_text().splitlines()[0].split()
            total = next((field[6:] for field in fields[1:] if field.startswith("total=")), "")
            if fields[0] != "some" or not re.fullmatch(r"[0-9]+", total):
                return None
            value[kind] = int(total)
    except (OSError, IndexError):
        return None
    return value


def uptime_seconds():
    """The host's uptime in seconds (/proc/uptime), or None without it."""
    try:
        field = Path("/proc/uptime").read_text().split()[0]
        return float(field) if re.fullmatch(r"[0-9.]+", field) else None
    except (OSError, ValueError, IndexError):
        return None


def host_load_sample():
    """The wall time, uptime, 1-minute, 5-minute and 15-minute load
    averages, the cumulative idle and total CPU ticks, the online CPUs and
    the cumulative pressure stall microseconds (`pressure_some_us`) of this
    host, the uptime and stalls None without them, or None without /proc."""
    try:
        load = [float(field) for field in Path("/proc/loadavg").read_text().split()[:3]]
        lines = Path("/proc/stat").read_text().splitlines()
        # user nice system idle iowait irq softirq steal; guest time is in user.
        ticks = [int(field) for field in lines[0].split()[1:9]]
    except (OSError, ValueError, IndexError):
        return None
    if len(load) != 3 or len(ticks) != 8:
        return None
    return {"at": time.time(), "uptime_seconds": uptime_seconds(), "load_average": load,
            "cpu_idle_ticks": ticks[3] + ticks[4], "cpu_ticks": sum(ticks),
            "cpus": sum(bool(re.match(r"cpu\d", line)) for line in lines), "pressure_some_us": pressure_some_us()}


def counter(value):
    """`value` if it is a cumulative counter (an integer from 0 below
    2**64), else None."""
    return value if type(value) is int and 0 <= value < 2**64 else None


def seconds(value):
    """`value` as a float if it is a finite JSON number, else None."""
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def judged_sample(row):
    """Whether `row` has what the CPU part of the rule reads."""
    return (isinstance(row, dict) and isinstance(row.get("load_average"), list) and bool(row["load_average"])
            and seconds(row["load_average"][0]) is not None
            and counter(row.get("cpu_idle_ticks")) is not None and counter(row.get("cpu_ticks")) is not None)


def timed_sample(row):
    """Whether `row` also has the uptime and stalls the pressure part reads."""
    stalls = row.get("pressure_some_us")
    return (seconds(row.get("uptime_seconds")) is not None and isinstance(stalls, dict)
            and all(counter(stalls.get(kind)) is not None for kind, _ in PRESSURE_KINDS))


def host_load(samples, cpus=None):
    """`samples` (oldest first) judged by HOST_LOAD_RULE, recording `cpus`
    online CPUs (default: the latest sample's count)."""
    cpus = cpus or next((row["cpus"] for row in reversed(samples)
                         if isinstance(row, dict) and row.get("cpus")), None) or os.cpu_count()
    result = {"rule": HOST_LOAD_RULE, "scope": "check_host", "cpus": cpus, "samples": samples,
              "max_load_average_1m": None, "cpu_idle_fraction": None, "min_cpu_idle_fraction": None,
              "max_cpu_pressure_share": None, "max_io_pressure_share": None}
    reasons = []
    pairs = list(zip(samples, samples[1:]))
    if len(samples) < 2:
        reasons.append("fewer than two host load samples")
    elif not all(map(judged_sample, samples)):
        reasons.append("a host load sample is missing or incomplete")
    else:
        # Recorded, not judged (see above).
        result["max_load_average_1m"] = max(seconds(row["load_average"][0]) for row in samples)
        ticks = [(b["cpu_idle_ticks"] - a["cpu_idle_ticks"], b["cpu_ticks"] - a["cpu_ticks"]) for a, b in pairs]
        if not all(total > 0 and 0 <= idle <= total for idle, total in ticks):
            reasons.append("CPU ticks do not advance between host load samples")
        else:
            first, last = samples[0], samples[-1]
            result["cpu_idle_fraction"] = (float(last["cpu_idle_ticks"] - first["cpu_idle_ticks"])
                                           / float(last["cpu_ticks"] - first["cpu_ticks"]))
            result["min_cpu_idle_fraction"] = low = min(float(idle) / float(total) for idle, total in ticks)
            if low < IDLE_CPU_FRACTION:
                reasons.append(f"CPU idle fell to {low:.3f}")
            if not all(map(timed_sample, samples)):
                reasons.append("no uptime or pressure stall information")
            else:
                elapsed = [seconds(b["uptime_seconds"]) - seconds(a["uptime_seconds"]) for a, b in pairs]
                stalls = {kind: [b["pressure_some_us"][kind] - a["pressure_some_us"][kind] for a, b in pairs]
                          for kind, _ in PRESSURE_KINDS}
                if not (all(span > 0 for span in elapsed)
                        and all(stall >= 0 for values in stalls.values() for stall in values)):
                    reasons.append("uptime or pressure stalls do not advance between host load samples")
                else:
                    for kind, name in PRESSURE_KINDS:
                        result[f"max_{kind}_pressure_share"] = high = max(
                            float(stall) / 1e6 / span for stall, span in zip(stalls[kind], elapsed))
                        if high > IDLE_PRESSURE_SHARE:
                            reasons.append(f"{name} pressure stall share reached {high:.3f}")
    return {**result, "idle": not reasons, "reasons": reasons,
            "timing": "informational" if reasons else "evidence"}


def unbounded_host_load():
    """The host load of a receipt whose check took no samples around its
    timed work: informational, since what it would judge could include
    warm-up or setup that dilutes contention in the timed work."""
    return {**host_load([]), "reasons": [UNBOUNDED_TIMING]}


def host_load_windows(windows):
    """Separate timed windows, each a list of host load samples (oldest
    first) that bound it, judged one by one: idle only when every window is.
    The gaps between windows, such as a warm-up, are never judged."""
    judged = [host_load(samples) for samples in windows]
    if not judged:
        return unbounded_host_load()
    reasons = [f"window {index}: {reason}" for index, load in enumerate(judged, 1) for reason in load["reasons"]]
    return {"rule": HOST_LOAD_RULE, "scope": "check_host_windows", "windows": judged, "idle": not reasons,
            "reasons": reasons, "timing": "informational" if reasons else "evidence"}


def compared_timing(loads):
    """The label of throughput or latency compared across runs, from each
    run's recorded host load ({name: host_load or None}): evidence only when
    every run was idle."""
    reasons = [f"{name}: " + ("; ".join(load["reasons"]) if load else "no host load recorded")
               for name, load in loads.items() if not (load and load.get("idle") is True)]
    return {"rule": HOST_LOAD_RULE, "timing": "informational" if reasons else "evidence", "reasons": reasons}


def receipt(check, identity, result, mechanics_only=False, load_samples=None, load_windows=None):
    """Writes a receipt with the host load of `load_samples`, the host load
    samples that bound the check's timed work (oldest first, the first at
    its start and the last at its end), or of `load_windows`, one such list
    per separate timed window (`host_load_windows`). Without them the
    receipt's timing is informational (`unbounded_host_load`)."""
    load = (host_load_windows(load_windows) if load_windows else
            host_load(load_samples) if load_samples else unbounded_host_load())
    directory = receipt_root()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}-{check}.json"
    value = {"schema": "chopinfer-qualification/v1", "check": check, "at": stamp,
             "server": identity, "mechanics_only": mechanics_only, "host_load": load, **result}
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"receipt": str(path), "passed": value.get("passed"),
                      "skipped": value.get("skipped"), "errors": value.get("errors", [])}))
    return path


def served_entries(row):
    """A served prompt-logprob row ({token: {logprob, rank}}) as the
    receipts keep it: [token, logprob, rank] entries, rank None when the
    server gave none, in top-1 order (`entry_order`)."""
    return sorted(([int(token), value["logprob"], value.get("rank")] for token, value in row.items()),
                  key=entry_order)


def entry_order(entry):
    """The served top-1 is the entry with the lowest rank, then the highest
    logprob, then the lowest token ID."""
    token, logprob, rank = entry
    return (10**9 if rank is None else rank, -logprob, token)


def entries_top1(entries):
    return min(entries, key=entry_order)[0]


def rows_digest(rows):
    """SHA-256 of served rows (lists of `served_entries`) as compact JSON."""
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def response_text(events, chat=False):
    parts = []
    for _, event in events:
        for choice in event.get("choices", []):
            part = choice.get("delta", {}).get("content") if chat else choice.get("text")
            if part:
                parts.append(part)
    return "".join(parts)
