"""Read-only evidence applicability, separate from recorded result and qualification.

Adapters read fields produced by existing runners, never an asserted `observed`
wrapper. Normalized Record instances are internal, not a new capture schema.
Diagnostic producers own measured identity and completeness; these readers own
which dimensions a validation claim needs. Unknown producers remain missing.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import release_gate

MAX_BYTES = 8_000_000
SHA256 = re.compile(r"[0-9a-f]{64}")
MODEL = {"checkpoint", "prepared_weights", "frontend", "package", "native_module", "kernel",
         "host", "topology", "devices", "settings", "workload", "reference", "tolerances", "implementation", "source"}
REQUIRED = {
    "scoped_oracle_diagnostic": {"checkpoint", "prepared_weights", "workload", "reference", "tolerances", "implementation", "source"},
    "cpu_contract": {"source", "implementation", "test_scope"},
    "fake_host": {"source", "implementation", "test_scope", "workload"},
    "frontend_cargo_goldens": {"checkpoint", "frontend", "source", "implementation", "workload", "reference"},
    "artifact_identity": {"kernel", "native_module", "source", "implementation"},
    "gpu_correctness": MODEL,
    "fixed_batch_exactness": MODEL,
    "model_quality": MODEL | {"policy"},
    "service_performance": MODEL | {"policy", "exclusive_device"},
}
KERNEL = {"kernel", "native_module", "host", "topology", "devices", "settings", "workload", "reference", "tolerances", "implementation", "source"}


def requirements(obligation):
    dimensions = KERNEL if obligation.get("scope") == "kernel_operation" else REQUIRED.get(obligation["kind"], MODEL)
    return {"dimensions": sorted(dimensions),
            "arms": ["candidate", "baseline"] if obligation["kind"] == "fixed_batch_exactness" else ["candidate"],
            "coverage": "Observed complete coverage must match the expected nonzero comparison count and named cases"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@contextmanager
def opened(path):
    """Regular-file read with no symlink traversal, including parents."""
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(file, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("evidence must be a regular file")
            yield stream
            def stamp(value):
                return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
            if (stamp(info) != stamp(os.fstat(stream.fileno())) or
                    stamp(info) != stamp(os.stat(path.name, dir_fd=descriptor, follow_symlinks=False))):
                raise ValueError("evidence changed during read")
    finally:
        os.close(descriptor)


def read_bytes(path, *, max_bytes=None):
    """Bounded read; callers may declare a larger bound for complete captures."""
    bound = MAX_BYTES if max_bytes is None else max_bytes
    with opened(path) as stream:
        if os.fstat(stream.fileno()).st_size > bound:
            raise ValueError("evidence must be a bounded regular file")
        value = stream.read(bound + 1)
        if len(value) > bound:
            raise ValueError("evidence exceeds the read bound")
        return value


def read_json(path, *, max_bytes=None):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(read_bytes(path, max_bytes=max_bytes), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    canonical(value)
    return value


def digest_file(path, expected):
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise ValueError("malformed SHA-256")
    if hashlib.sha256(read_bytes(path)).hexdigest() != expected:
        raise ValueError("referenced file digest mismatch")


@dataclass
class Record:
    kind: str
    claim: str | None = None
    result: str | None = None
    execution: str = "executed"
    observed: dict = field(default_factory=lambda: {"candidate": {}, "baseline": {}})
    coverage: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    stale: list = field(default_factory=list)
    raw: dict | None = None
    origin: str = ""

    def bind(self, name, value, pointer, arm="candidate"):
        if value is not None:
            self.observed[arm][name] = {"value": value, "field": pointer}


def result_of(value):
    if value.get("skipped") or value.get("status") in ("informational", "skipped"):
        return "skipped"
    if value.get("passed") is True or value.get("correct") is True:
        return "passed"
    if value.get("passed") is False or value.get("correct") is False or value.get("errors") or value.get("error"):
        return "failed"
    return None


def _load(path):
    path = Path(path).absolute()
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("receipt must be a JSON object")
    except (OSError, ValueError) as error:
        return [Record("unknown", execution="not_run", issues=[str(error)], origin=str(path))]
    records = []
    if value.get("schema") == "chopinfer-release-receipt/v1":
        config = value.get("config", {})
        policy_digest = hashlib.sha256((json.dumps(config, indent=2) + "\n").encode()).hexdigest()
        # A checkpoint variant's run claims its own gate (MODEL.VARIANT), as
        # validation reports name its obligations.
        variant = config.get("variant") if isinstance(config, dict) else None
        gate = f"{value.get('model')}.{variant}" if variant else value.get("model")
        for row in value.get("checks", [])[:100]:
            check_id = row.get("id")
            check = next((c for c in config.get("checks", []) if c.get("id") == check_id), {})
            kind = "service_performance" if check.get("module") == "soak" else "model_quality"
            if value.get("mechanics_only"):
                kind = "fake_host"
            record = Record(kind, f"{gate}:gate:{check_id}", result_of(row), origin=str(path))
            record.bind("policy", value.get("config_sha256"), "/config_sha256")
            if policy_digest != value.get("config_sha256"):
                record.stale.append("stored gate policy digest mismatch")
            if row.get("receipt"):
                file = Path(row["receipt"])
                if not file.is_absolute():
                    file = path.parent / file
                try:
                    if not file.absolute().is_relative_to(path.parent) or ".." in file.parts:
                        raise ValueError("referenced receipt is outside the supplied receipt directory")
                    digest_file(file, row.get("sha256"))
                    record.raw = read_json(file)
                    if not isinstance(record.raw, dict):
                        raise ValueError("referenced receipt is not an object")
                    record.coverage = {"comparisons": len(check.get("rules", [])), "complete": True}
                    if record.raw.get("skipped") or record.raw.get("mechanics_only") or record.raw.get("errors"):
                        record.issues.append("raw evidence is skipped, mechanics-only or failed")
                except (OSError, ValueError) as error:
                    record.stale.append(str(error))
            else:
                record.issues.append("no authenticated raw receipt")
            # Revision and API aliases identify the runner, not the serving host,
            # checkpoint or loaded module. They are deliberately not bindings.
            record.issues.append("legacy HTTP receipt lacks observed deployment and coverage bindings")
            records.append(record)
        for name in value.get("not_run", []):
            records.append(Record("model_quality", f"{gate}:gate:{name}", execution="not_run",
                                  issues=["runner did not execute this check"], origin=str(path)))
    elif isinstance(value.get("identity"), dict) and value["identity"].get("format") in ("fixed-batch-logits-v1", "fixed-batch-packages-v4"):
        identity = value["identity"]
        record = Record("fixed_batch_exactness", result=result_of(value), origin=str(path))
        record.bind("host", identity.get("source"), "/identity/source")
        record.bind("topology", {k: identity[k] for k in ("topology", "ranks", "fold") if k in identity}, "/identity")
        record.bind("workload", hashlib.sha256(canonical(identity.get("trace"))).hexdigest(), "/identity/trace (digest)")
        record.bind("settings", {k: identity[k] for k in ("max_sequences", "maximum_verify_tokens") if k in identity}, "/identity")
        manifest = value.get("manifest", identity.get("manifest"))
        if manifest:
            record.bind("package", hashlib.sha256(canonical(manifest)).hexdigest(), "/manifest (canonical digest)")
        record.coverage = {"comparisons": value.get("rows", 0), "complete": value.get("steps") == 13 and value.get("bytes_recorded_or_compared", 0) > 0}
        if not value.get("reference"):
            record.issues.append("recording without comparison reference is not exactness evidence")
        record.issues.append("legacy fixed-batch receipt lacks full observed bindings for both arms")
        records.append(record)
    elif "measurement_label" in value and any(key in value for key in ("correct", "worst_element", "cubin_sha256")):
        record = Record("gpu_correctness", result=result_of(value), origin=str(path))
        kernel = {k: value[k] for k in ("cubin_sha256", "source_closure_sha256", "nvcc_sha256") if k in value}
        if kernel:
            record.bind("kernel", kernel, "/ (kernel identity fields)")
        record.bind("devices", {k: value[k] for k in ("device_name", "uuid", "cuda_driver_version", "rank_devices", "folded") if k in value}, "/ (device fields)")
        record.bind("exclusive_device", value["measurement_label"] == "exclusive-device", "/measurement_label")
        record.coverage = {"comparisons": value.get("comparisons", 0), "complete": False}
        record.issues.append("kernel receipt has no complete claim/shape/coverage binding; filenames do not establish it")
        records.append(record)
    elif value.get("format") == "chopinfer.diagnostic.v1" and value.get("mode") == "oracle":
        record=Record("scoped_oracle_diagnostic", origin=str(path))
        try:
            from tools.qual.oracle_diagnostics import verify_report
            checked=verify_report(value)
            record.result={"match":"passed","diverged":"failed"}.get(checked['status'])
            record.raw=checked
            identity=checked['identity']['candidate']
            from tools.model_sdk import diagnostic as wire
            candidate=wire.load(wire.external(checked['attachments']['candidate']['root']))
            description=candidate['description']
            scope=checked['scope']
            producer=description['producer']
            for dimension in ('checkpoint','prepared_weights','package','native_module','kernel','frontend'):
                record.bind(dimension,identity.get(dimension),'/identity/candidate/'+dimension)
            record.bind('workload',scope['inputs'],'/scope/inputs')
            record.bind('topology',scope.get('topology'),'/scope/topology')
            record.bind('tolerances',{'contract':scope['contract'],'policy_sha256':scope['policy_sha256']},'/scope')
            record.bind('reference',checked['attachments']['reference'],'/attachments/reference')
            record.bind('implementation',producer['implementation_sha256'],'/attachments/candidate/description/producer/implementation_sha256')
            record.bind('source',producer['source_sha256'],'/attachments/candidate/description/producer/source_sha256')
            for dimension in ('host','settings','topology'):
                record.bind(dimension,identity.get('parent_observations',{}).get(dimension),'/identity/candidate/parent_observations/'+dimension)
            record.coverage={'comparisons':checked['coverage']['gated'],'complete':checked['coverage']['complete'],
                             'cases':[probe['name'] for probe in scope['probes']]}
            if checked['status'] not in ('match','diverged'):
                record.issues.append(checked.get('reason','insufficient scoped oracle evidence'))
        except (ValueError,OSError,KeyError,TypeError,IndexError) as error:
            record.issues.append(str(error))
        records.append(record)
    elif value.get("format") == "chopinfer.diagnostic.v1":
        # Keep the diagnostic producer's field paths. Configured switches and
        # calculated buckets do not become observed graph/capacity behavior.
        mode = value.get("mode")
        kind = "fixed_batch_exactness" if mode == "fixed-schedule" else "fake_host" if mode == "journal" else "unknown"
        result = {"match": "passed", "diverged": "failed"}.get(value.get("status"))
        record = Record(kind, result=result, origin=str(path))
        description = value.get("description", {})
        invariants = description.get("invariants", {})
        for dimension, key in (("host", "host"), ("workload", "trace")):
            record.bind(dimension, invariants.get(key), "/description/invariants/" + key)
        record.bind("checkpoint", description.get("source_checkpoint"), "/description/source_checkpoint")
        record.bind("package", description.get("context", {}).get("manifest_sha256"), "/description/context/manifest_sha256")
        record.bind("topology", {k: invariants[k] for k in ("topology", "ranks", "fold") if k in invariants}, "/description/invariants")
        if mode == "journal":
            record.bind("workload", value.get("identity"), "/identity")
        record.coverage = {"comparisons": value.get("rows", value.get("events", 0)),
                           "complete": value.get("completion") == "full" and value.get("completed_steps") == description.get("expected_steps")}
        allowed = {"frames.jsonl", "rows.bin", "receipt.json", "mismatch.json", "mismatch.bin", "journal.txt"}
        lifecycle = mode == "journal" and value.get("extension") == "qwen38.lifecycle.v1"
        if lifecycle:
            allowed |= {"input.json", "observations.jsonl"}
            record.raw = {"device": value.get("device")}
            record.result = value.get("execution_result")
            record.bind("implementation", value.get("producer"), "/producer")
            record.bind("test_scope", {key: value.get(key) for key in ("extension", "backend")}, "/")
            record.coverage = {"comparisons": value.get("completed_actions", 0),
                               "complete": value.get("completion") == "full"
                               and value.get("completed_actions") == value.get("expected_actions")
                               and set(value.get("artifacts", {})) == {"input.json", "journal.txt", "observations.jsonl"}}
            record.issues.append("CPU lifecycle mechanics only; idle-device lifecycle evidence is missing")
        for name, artifact in value.get("artifacts", {}).items():
            try:
                if name not in allowed:
                    raise ValueError("unknown diagnostic artifact path")
                data = read_bytes(path.parent / name)
                if len(data) != artifact["bytes"] or hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                    raise ValueError("diagnostic artifact digest/length mismatch")
            except (OSError, ValueError, KeyError) as error:
                record.issues.append(("unverified lifecycle artifact " + name + ": " if lifecycle else "") + str(error))
                if lifecycle:
                    record.coverage["complete"] = False
        record.issues.append("bounded report does not establish complete diagnostic reference/control or observed device behavior; use the diagnostic verifier")
        records.append(record)
    else:
        records.append(Record("unknown", execution="not_run", issues=["unsupported receipt producer/version"], origin=str(path)))
    return records or [Record("unknown", execution="not_run", issues=["empty receipt has zero checks"], origin=str(path))]


def load(path):
    try:
        return _load(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
        return [Record("unknown", execution="not_run", issues=["malformed receipt: " + str(error)], origin=str(path))]


def invalid_digest(value):
    if isinstance(value, dict):
        return any((key.endswith("sha256") and (not isinstance(item, str) or not SHA256.fullmatch(item)))
                   or invalid_digest(item) for key, item in value.items())
    if isinstance(value, list):
        return any(invalid_digest(item) for item in value)
    return False


def assess(record, obligation, expected):
    """Match measured producer fields to candidate expectations, never vice versa."""
    status = {"selection": "selected", "execution": record.execution, "result": record.result,
              "applicability": "missing", "reasons": list(record.issues), "receipt": record.origin,
              "observed_fields": {}, "satisfied": False}
    missing, stale = status["reasons"], list(record.stale)
    if record.claim != obligation["id"]:
        missing.append("receipt does not bind this exact claim")
    if record.kind != obligation["kind"]:
        missing.append("evidence kind mismatch: " + record.kind + " cannot establish " + obligation["kind"])
    required = requirements(obligation)
    needed, arms = required["dimensions"], required["arms"]
    for arm in arms:
        desired = expected.get(arm, {})
        observed = record.observed.get(arm, {})
        for dimension in sorted(needed):
            measurement = observed.get(dimension)
            if desired.get(dimension) in (None, {}, "") or not measurement or measurement.get("value") in (None, {}, ""):
                missing.append(arm + ": missing observed/expected " + dimension)
                continue
            value = measurement["value"]
            status["observed_fields"][arm + "." + dimension] = measurement["field"]
            if invalid_digest(value) or invalid_digest(desired[dimension]):
                stale.append(arm + ": malformed digest in " + dimension)
            elif canonical(value) != canonical(desired[dimension]):
                stale.append(arm + ": changed " + dimension)
    coverage = record.coverage
    count = coverage.get("comparisons")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or coverage.get("complete") is not True:
        missing.append("zero or incomplete comparison coverage")
    required_coverage = expected.get("coverage", {})
    if not isinstance(required_coverage.get("comparisons"), int) or isinstance(required_coverage.get("comparisons"), bool) or required_coverage["comparisons"] <= 0:
        missing.append("missing required comparison coverage")
    elif count != required_coverage["comparisons"]:
        missing.append("comparison count differs from the required workload")
    for key, value in required_coverage.items():
        if key != "comparisons" and coverage.get(key) != value:
            missing.append("missing required coverage: " + key)
    if any(coverage.get(key) for key in ("skipped", "partial_goldens", "truncated", "subprocess_failed")):
        missing.append("skipped, partial, truncated or failed coverage")
    if record.execution != "executed" or record.result in (None, "skipped"):
        missing.append("check was not executed completely")
    if obligation["kind"] == "service_performance" and record.observed["candidate"].get("exclusive_device", {}).get("value") is not True:
        missing.append("exclusive-device performance conditions were not observed")
    if record.raw is not None and obligation.get("gate"):
        try:
            passed = all(release_gate.judge(record.raw, rule) for rule in obligation["gate"]["rules"])
        except (KeyError, TypeError, ValueError, IndexError):
            passed = False
        status["predicate_result"] = "passed" if passed else "failed"
        if not passed:
            missing.append("current gate predicates are not satisfied")
    status["reasons"] = sorted(set(missing + stale))
    status["applicability"] = "stale" if stale else "missing" if missing else "current"
    status["satisfied"] = status["applicability"] == "current" and status["result"] == "passed"
    return status


def attach(report, context, receipts):
    if not isinstance(context, dict):
        raise ValueError("context must contain candidate/baseline expectations, not asserted observations")
    if len(receipts) > 100:
        raise ValueError("too many receipts for bounded report")
    records = [record for path in receipts for record in load(path)]
    report["supplied_receipts"] = [{"path": record.origin, "claim": record.claim, "kind": record.kind,
                                    "result": record.result, "issues": record.issues + record.stale} for record in records]
    if context.get("cuda_membership"):
        report["gaps"].append("Supplied CUDA membership is advisory; authoritative build provenance and completeness remain unproven")
    for obligation in report["obligations"]:
        if obligation["evidence"]["selection"] == "not_applicable":
            continue
        expected = json.loads(json.dumps(context.get("claims", {}).get(obligation["id"], context)))
        # Never let an old context override the policy and implementation actually
        # selected by this report. Workspace binding includes dirty input content.
        candidate = expected.setdefault("candidate", {})
        candidate["source"] = ({**report["source_identity"], "change_identity_sha256": report["change_identity_sha256"]}
                               if report.get("source_identity") else None)
        if obligation["kind"] in ("cpu_contract", "fake_host"):
            candidate["test_scope"] = {"targets": obligation["targets"], "features": obligation["observation"]["features"],
                                       "platform": obligation["observation"]["platform"]}
        if obligation.get("gate"):
            candidate["policy"] = obligation["gate"]["policy_sha256"]
            candidate["implementation"] = obligation["gate"]["implementation_sha256"]
        matches = [record for record in records if record.claim == obligation["id"] or
                   (record.claim is None and record.kind == obligation["kind"])]
        assessments = [assess(record, obligation, expected) for record in matches]
        profile_dependent = (obligation["kind"] in ("model_quality", "service_performance", "fixed_batch_exactness")
                             or obligation["kind"] == "gpu_correctness" and obligation.get("scope") != "kernel_operation")
        model = obligation["id"].split(":", 1)[0].split(".", 1)[0]
        current_profiles = {row["profile"] for row in report.get("profiles", [])
                            if row["model"] == model and not row["removed"]}
        if profile_dependent and len(current_profiles) > 1:
            reason = "per-profile identity and result aggregation is unavailable for multiple current serving profiles"
            for status in assessments or [obligation["evidence"]]:
                status.setdefault("reasons", []).append(reason)
                if status.get("applicability") != "stale":
                    status["applicability"] = "missing"
                status["satisfied"] = False
        if assessments:
            # Keep contradictory runs visible, even when one complete match exists.
            assessments.sort(key=lambda row: (not row["satisfied"], row["applicability"] != "current", row["receipt"]))
            obligation["evidence"] = assessments[0]
            obligation["evidence_runs"] = assessments
