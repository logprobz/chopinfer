"""The check kinds of the qualification suite and their registry.

A model declaration (declaration.py) lists checks; each names the module
that runs it, its arguments, its receipt and the predicates that judge it.
The check's kind is the one registered for its module (`of`): it states how
the check captures from a server, which receipt fields and coverage it
requires, how the receipt scores and how the score is judged. The kinds
marked shared (teacher_forced, media_forced, greedy_pin, determinism, api,
structured) are the contract `chopinfer eval` implements; docs/check-kinds.md
specifies each, and tests/qual-vectors holds their conformance vectors.

capture(command, environment, log, timeout) -> exit code
    Live. Runs the check against a server: its module (release_gate.py
    --worker), which writes a `chopinfer-qualification/v1` object
    (common.receipt) named `<UTC stamp>-<receipt id>.json`, or JSON on stdout
    or in a declared file; or, for a kind captured by eval
    (`captured_by_eval`) of a model's declaration, `chopinfer eval run
    --check ID`, whose report holds the same record (release_gate.py).
reference(check, config, revision=None) -> reference or None
    The pinned reference the kind verifies and recomputes against, read
    under the declaration in effect (`policy`), from the working tree or
    from a source revision.
verify(receipt, reference, check) -> recorded fields
    Offline. Raises Insufficient unless the receipt holds every field and
    item the kind requires, with counts that agree with each other and with
    the reference. Returns the metric fields read as recorded: what this
    code cannot recompute from the receipt and the reference.
score(receipt, reference=None, check=None, mechanics=False) -> metrics
    Offline. The receipt's metrics: the kind's metric fields and every field
    a declared predicate reads, nested as in the receipt, so a predicate
    reads the same value in both. What the kind derives is recomputed by
    this code, from the reference when one is given.
judge(metrics, check) -> (status, predicates)
    `pass` when every declared predicate holds, else `fail`, with each
    predicate and its result. A predicate whose path the metrics lack raises
    KeyError, as it did on the receipt.

The shared kinds' and frontend_goldens' verify, score and judge run in
`chopinfer eval rescore` (tools/qual/README.md#release-gates), the one scorer implementation: `call`
sends each step to the binary, found as `CHOPINFER_EVAL` or the checkout's
target directory. This module keeps what eval cannot do from Python's side:
reading a pinned reference from the working tree or a source revision, and
the live capture.

The gate (release_gate.py) scores and judges the receipt its own module
just wrote. `rescore` (below; checks.py) re-judges a stored receipt: admit,
verify, score, judge. A check whose declaration has an
`informational_reason` is not run or judged: its status is `informational`.
"""

import atexit
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STATUSES = ("pass", "fail", "informational", "insufficient", "not_run", "not_applicable", "waived")
# api.v1: the cases api.py runs, in order, for a complete (non-mechanics) run.
API_CASES = (
    "choice_single_False", "multiple_choices_False_False", "multiple_choices_False_True",
    "choice_single_True", "multiple_choices_True_False", "multiple_choices_True_True",
    "generated_token_ids_False_False", "generated_token_ids_False_True",
    "generated_token_ids_True_False", "generated_token_ids_True_True",
    "tokenize", "batch_single", "batch_False_False", "batch_False_True",
    "batch_single", "batch_True_False", "batch_True_True",
    "add_special_tokens_False_False", "add_special_tokens_True_False",
    "add_special_tokens_False_True", "add_special_tokens_True_True",
    "detokenize", "stop_token_ids", "generation_prompt_control", "hidden_reasoning", "ignore_eos",
    "nullable_flags", "unknown_extension", "sampling_extensions", "chat_token_limit_precedence",
    "continuous_usage", "stream_usage", "stop", "completion_logprobs", "reasoning_True", "reasoning_False",
    "stream_tool_arguments", "tool_call", "tool_result", "metrics_compatibility")
# structured.v1: the cases structured.py defines, each run `repeat` times
# and, with `stream`, once more streamed as `<case>+stream`.
STRUCTURED_CASES = (
    "chat_json_object", "chat_json_schema_strict", "chat_json_schema_loose", "chat_so_json", "chat_so_choice",
    "chat_so_regex", "chat_so_json_object", "completion_so_json", "completion_json_schema", "tool_required",
    "tool_named", "reasoning_json_schema")


class Insufficient(ValueError):
    """What is stored cannot be judged: a required field or item is
    missing, counts disagree, or the receipt scored another reference."""


class MissingMetric(KeyError):
    """A metric a predicate reads is missing; prints as Python's KeyError
    prints the key."""

    def __str__(self):
        return self.args[0]


def eval_binary():
    """The chopinfer binary whose `eval rescore` scores the shared kinds."""
    if path := os.environ.get("CHOPINFER_EVAL"):
        return Path(path)
    target = Path(os.environ.get("CARGO_TARGET_DIR") or ROOT / "target")
    for profile in ("debug", "release"):
        if (target / profile / "chopinfer").is_file():
            return target / profile / "chopinfer"
    raise RuntimeError("the shared kinds score in chopinfer eval: build it with "
                       "`cargo build -p runtime-host --bin chopinfer` or set CHOPINFER_EVAL to the binary")


def finite_json(value):
    """`value` as JSON, or None when it holds a non-finite number, which
    makes a receipt or reference insufficient (docs/check-kinds.md)."""
    try:
        return json.dumps(value, allow_nan=False)
    except ValueError:
        return None


def evaluated(calls, run=subprocess.run):
    """The results of `chopinfer eval rescore` calls, in order. `run` is
    bound at import: the gate's tests stand in for its workers by patching
    subprocess.run, which must not replace the scorer."""
    request = finite_json({"schema": "chopinfer-eval-rescore/v1", "calls": calls})
    if request is None:
        raise Insufficient("a receipt or reference holds a non-finite number")
    done = run([str(eval_binary()), "eval", "rescore"], input=request, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"chopinfer eval rescore exited {done.returncode}: {done.stderr.strip()}")
    try:
        response = json.loads(done.stdout)
        if not isinstance(response, dict) or response.get("schema") != "chopinfer-eval-rescore-result/v1":
            raise ValueError("incompatible response schema")
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(calls):
            raise ValueError("result count does not match calls")
        if finite_json(response) is None:
            raise ValueError("non-finite result")
        for request, result in zip(calls, results):
            validate_result(request["op"], result)
        return results
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid chopinfer eval rescore response: {error}") from error


def validate_result(op, result):
    """Validate the v1 operation's result before trusting any verdict."""
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    if "error" in result:
        error = result["error"]
        if op == "rescore":
            if not isinstance(error, str):
                raise ValueError("rescore error must be text")
        else:
            if (not isinstance(error, dict) or error.get("type") not in
                    ("insufficient", "environment", "invalid", "missing_metric") or
                    not isinstance(error.get("message"), str)):
                raise ValueError("invalid operation error")
            return
    if op in ("rescore", "judge"):
        if result.get("status") not in STATUSES:
            raise ValueError("invalid status")
        if result["status"] == "not_applicable" and not isinstance(result.get("reason"), str):
            raise ValueError("not_applicable needs a reason")
        predicates = result["predicates"]
        if predicates is None and op == "rescore" and result["status"] in ("fail", "insufficient", "not_applicable"):
            pass
        elif not isinstance(predicates, list) or any(
                not isinstance(rule, dict) or not isinstance(rule.get("path"), list) or
                rule.get("op") not in ("eq", "le", "ge", "count_ge", "true_count_ge") or
                "value" not in rule or not isinstance(rule.get("source"), str) or
                type(rule.get("passed")) is not bool for rule in predicates):
            raise ValueError("invalid predicates")
    if op in ("rescore", "score", "score_capture"):
        metrics = result["metrics"]
        if not isinstance(metrics, dict) and not (
                metrics is None and op == "rescore" and result["status"] in ("fail", "insufficient")):
            raise ValueError("metrics must be an object")
    if op in ("rescore", "verify"):
        fields = result["recorded_fields"]
        if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
            raise ValueError("recorded_fields must be a list of strings")


def call(op, kind, value=None, reference=None, check=None, mechanics=False, metrics=None):
    """One eval call of `kind`; an error result raises as the Python step
    would: Insufficient, ValueError or a missing metric's KeyError."""
    request = {"op": op, "kind": kind.name, "receipt": value, "reference": reference, "check": check or {},
               "mechanics": mechanics}
    if (isinstance(value, dict) and value.get("not_applicable")
            and value.get("schema") == "chopinfer-qualification/v1"
            and "checkpoint_identity" not in request["check"]):
        import common
        request["check"] = {**request["check"], "checkpoint_identity": common.checkpoint_pin()["checkpoint_identity"]}
    if metrics is not None:
        request["metrics"] = metrics
    [result] = evaluated([request])
    error = result.get("error") if op != "rescore" else None
    if error:
        raise {"insufficient": Insufficient, "missing_metric": MissingMetric}.get(error["type"], ValueError)(
            error["message"])
    return result


def validated_waivers(check):
    """A check's declared waivers once eval's waiver validator accepts them
    (declaration::validate_waivers: a nonempty reason and decision, a
    declared predicate with a looser ceiling, or a teacher category), else
    ValueError; [] for a check that declares none. The one waiver validator:
    every judge and every gate config applies it before a waiver counts."""
    if not isinstance(check, dict) or "waivers" not in check:
        return []
    named = {key: check[key] for key in ("module", "rules", "waivers") if key in check}
    [result] = evaluated([{"op": "waivers", "check": named}])
    if result.get("error"):
        raise ValueError(f"check {check.get('id') or check.get('module')}: {result['error']['message']}")
    return check["waivers"]


def natural(value):
    return type(value) is int and value >= 0


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def select(value, path):
    """The values at a predicate path; `*` takes every child."""
    if not path:
        return [value]
    key, *tail = path
    if key == "*":
        children = value.values() if isinstance(value, dict) else value
        return [item for child in children for item in select(child, tail)]
    return select(value[key], tail)


def predicate(value, rule):
    """Whether a declared predicate holds over `value` (a receipt or its metrics)."""
    values = select(value, rule["path"])
    if not values:
        return False
    op, expected = rule["op"], rule["value"]
    if op == "true_count_ge":
        return sum(item is True for item in values) >= max(1, expected)
    for item in values:
        if op in ("le", "ge"):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                return False
            if not (item <= expected if op == "le" else item >= expected):
                return False
        elif op == "eq":
            if type(item) is not type(expected) or item != expected:
                return False
        elif op == "count_ge":
            if len(item) < max(1, expected):
                return False
        else:
            raise ValueError(f"unknown predicate {op}")
    return True


def admit(value, mechanics=False):
    """Raise ValueError for a receipt the gate fails before any predicate:
    not an object, mechanics-only or skipped coverage outside a mechanics
    run, or errors."""
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object receipt")
    if "experimental_determinism" in value:
        raise ValueError("experimental determinism is not release qualification")
    if isinstance(value.get("server"), dict):
        import common
        common.require_qualified_model_options(value["server"])
    if not mechanics and (value.get("mechanics_only") or value.get("skipped")):
        raise ValueError("quality check reported mechanics-only or skipped coverage")
    if value.get("errors"):
        raise ValueError("check reported errors")


_SOURCES = []


def source(path, revision=None, *, external=False):
    """`path` in the working tree, or its bytes at a source `revision`
    written to a temporary file of the same name. External data stays at its
    declared location; the caller authenticates it against the revision's pin."""
    path = Path(path)
    if revision is None or external:
        return path
    if not _SOURCES:
        _SOURCES.append(tempfile.mkdtemp(prefix="chopinfer-kinds-"))
        atexit.register(shutil.rmtree, _SOURCES[0], True)
    relative = path.resolve().relative_to(ROOT)
    target = Path(_SOURCES[0]) / revision / relative
    if not target.is_file():
        data = subprocess.run(["git", "-C", str(ROOT), "show", f"{revision}:{relative}"], capture_output=True,
                              check=False)
        if data.returncode:
            raise OSError(f"{relative} is not in revision {revision}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.stdout)
    return target


@contextmanager
def policy(config_path, check):
    """Module state as release_gate.py's worker sets it for `check`: the
    declaration it reads (CHOPINFER_QUAL_CONFIG) and the module constants
    the check overrides. Restored on exit."""
    saved = os.environ.get("CHOPINFER_QUAL_CONFIG")
    os.environ["CHOPINFER_QUAL_CONFIG"] = str(Path(config_path).resolve())
    restore = []
    try:
        for name, constants in check.get("constants", {}).items():
            target = importlib.import_module(name)
            for key, entry in constants.items():
                if not hasattr(target, key):
                    raise ValueError(f"unknown threshold {name}.{key}")
                restore.append((target, key, getattr(target, key)))
                setattr(target, key, entry["value"])
        yield
    finally:
        for target, key, value in reversed(restore):
            setattr(target, key, value)
        if saved is None:
            os.environ.pop("CHOPINFER_QUAL_CONFIG", None)
        else:
            os.environ["CHOPINFER_QUAL_CONFIG"] = saved


class Kind:
    """A kind whose metrics are its receipt's fields; the declared
    predicates are its verdict. Subclasses name their metric fields, what
    they require and what they recompute."""

    name = "receipt"
    modules = ()
    shared = False
    # Whether `chopinfer eval run --check ID` captures a model declaration's
    # check of this kind; the Python module keeps only the mechanics capture.
    captured_by_eval = False
    version = None
    inputs = "the module's declared arguments and inputs"
    receipt = "the module's JSON receipt"
    metrics = ("passed",)
    verdict = "every declared predicate holds"

    def fields(self, value, check):
        """The metric fields of a receipt: the kind's, then those a declared
        predicate reads. A predicate over the whole receipt or each of its
        fields reads every field."""
        paths = [rule["path"] for rule in (check or {}).get("rules", [])]
        if any(not path or path[0] == "*" for path in paths):
            return list(value)
        return [key for key in dict.fromkeys([*self.metrics, *(path[0] for path in paths)]) if key in value]

    def recorded(self, value, check):
        """The receipt's own metric values, as it stores them."""
        return {key: value[key] for key in self.fields(value, check)}

    def reference(self, check, config, revision=None):
        return None

    def verify(self, value, reference=None, check=None):
        """Nothing is required beyond the declared predicates; every metric is as recorded."""
        return self.fields(value, check)

    def verify_score(self, value, reference=None, check=None, mechanics=False):
        """Verify and score together when a kind can share their work."""
        fields = [] if value.get("mechanics_only") else self.verify(value, reference, check)
        return fields, self.score(value, reference, check, mechanics)

    def score(self, value, reference=None, check=None, mechanics=False):
        admit(value, mechanics)
        if not mechanics and not ((check or {}).get("module") == "grounding"
                                   and inapplicable_reason(value, check)):
            evidence = [value[key] for key in ("rows", "records", "cases", "requests", "items", "texts",
                        "positions", "scored", "completed", "count") if key in value]
            if self.name == "receipt" and not evidence:
                for rule in (check or {}).get("rules", []):
                    if rule["op"] == "count_ge" or (rule["path"] and rule["path"][-1] in
                            ("requests", "items", "positions", "scored", "completed", "count")):
                        evidence.extend(select(value, rule["path"]))
                    elif rule["op"] == "true_count_ge":
                        evidence.append([item for item in select(value, rule["path"]) if item is True])
            require(all((isinstance(item, (list, dict)) and bool(item))
                        or (natural(item) and item > 0) for item in evidence), "no scored items")
            if self.name == "receipt":
                require(bool(evidence), "no item coverage in generic receipt")
        return self.recorded(value, check)

    def judge(self, metrics, check):
        """The declared predicates; a failing one a declared waiver names
        (`waivers`, its path and op) is covered when it holds with the
        waiver's ceiling in place of its value, as eval's judge applies it
        (kind::waive). Category waivers belong to the shared teacher kind.
        Waivers the validator refuses raise ValueError (validated_waivers)."""
        waivers = validated_waivers(check)
        predicates = [{**rule, "passed": predicate(metrics, rule)} for rule in check["rules"]]
        if metrics.get("passed") is not False and all(rule["passed"] for rule in predicates):
            return "pass", predicates
        for rule in predicates:
            waiver = next((waiver for waiver in waivers
                           if waiver.get("predicate") == {"path": rule["path"], "op": rule["op"]}), None)
            if not rule["passed"] and waiver and predicate(metrics, {**rule, "value": waiver["ceiling"]}):
                rule["waiver"] = {"predicate": waiver["predicate"], "measured": select(metrics, rule["path"]),
                                  "ceiling": waiver["ceiling"], "reason": waiver["reason"],
                                  "decision": waiver["decision"]}
        covered = metrics.get("passed") is not False and all(rule["passed"] or "waiver" in rule for rule in predicates)
        return ("waived" if covered else "fail"), predicates

    def capture(self, command, environment, log, timeout):
        with Path(log).open("w") as output:
            return subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode


class MediaPreprocessing(Kind):
    """Nested reference/case/stage coverage from the CPU exporter check.

    Tensor errors and integer equality remain recorded fields. This kind
    validates coverage and recomputes predicates, not deleted exporter data.
    """

    name = "media_preprocessing"
    version = "media_preprocessing.v1"
    modules = ("media_preprocessing",)
    inputs = "the adopted preprocessing references and frontend configuration"
    receipt = "one bound report per reference, with every declared case and stage"
    metrics = ("passed", "references", "reference_bound", "exporter_sha256",
               "exporter_build_verified", "frontend_sha256")
    verdict = "complete bound reports, exact shapes and integer rows, and the declared spread bound"

    def reference(self, check, config, revision=None):
        import media_preprocessing
        return media_preprocessing.receipt_reference(config)

    def score(self, value, reference=None, check=None, mechanics=False):
        import common
        import media_preprocessing
        admit(value, mechanics)
        if mechanics:
            return self.recorded(value, check)
        if reference is None:
            reference = self.reference(check, common.qualification_config())
        return media_preprocessing.receipt_metrics(value, reference)


class Shared(Kind):
    """A kind `chopinfer eval` scores (docs/check-kinds.md): its verify,
    score and judge run in eval. The shared kinds are eval's portable
    subset; frontend_goldens is scored there but captured offline."""

    shared = True

    def verify(self, value, reference=None, check=None):
        return call("verify", self, value, reference, check)["recorded_fields"]

    def score(self, value, reference=None, check=None, mechanics=False):
        return call("score", self, value, reference, check, mechanics)["metrics"]

    def judge(self, metrics, check):
        validated_waivers(check)
        result = call("judge", self, check=check, metrics=metrics)
        return result["status"], result["predicates"]


class TeacherForced(Shared):
    """Teacher-forced prompt logprobs against a pinned reference's gate
    variant, with limits from the reference's own variant spread
    (limits.py, variant-spread.v2) for the pooled positions and each text
    category. Metrics `vs_<variant>` score the server against each spread
    variant."""

    name = "teacher_forced"
    modules = ("teacher",)
    captured_by_eval = True
    version = "teacher_forced.v1"
    inputs = ("the pinned `teacher` reference (per text: name, category, prompt token IDs, score_from, and per "
              "variant the logprob and top-1 of each scored position)")
    receipt = ("chopinfer-qualification/v1 `teacher`: one row per reference text (name, prompt_tokens, "
               "scored), pooled and per-category scores, limits, server identity and reference binding")
    metrics = ("passed", "scored", "mean_abs_dlogprob", "max_abs_dlogprob", "top1_disagreements",
               "top1_disagreement", "top1_near_ties", "top1_near_tie", "top1_decisive_flips", "top1_decisive",
               "top1_flips", "near_tie_nats", "limits", "reference_spread", "by_category", "failed_categories",
               "reference_sha256", "reference_binding", "server", "diagnostic_passed", "informational_reason")
    verdict = ("pooled mean |dlogprob|, decisive and near-tie top-1 rates within the spread limits, every "
               "category within its own, and the declared predicates")

    def fields(self, value, check):
        variants = [key for key in value if key.startswith("vs_")]
        return list(dict.fromkeys([*super().fields(value, check), *variants]))

    def reference(self, check, config, revision=None):
        """The pinned reference of an exact-reference gate, read and checked
        by common.pinned_reference, with the variants the gate pins: the gate
        variant and the spread variants. It is the teacher reference unless
        the check names another with `--reference-name` (Qwen3.8's
        long-context texts)."""
        import common
        import teacher
        if config["qualification"].get("teacher_reference_kind") != "exact":
            return None
        args = (check or {}).get("args", [])
        name = args[args.index("--reference-name") + 1] if "--reference-name" in args else "teacher"
        spec = config["qualification"]["references"][name]
        path = source(common.input_path("teacher_reference", config=config) if name == "teacher"
                      else common.reference_path(name, config), revision,
                      external=isinstance(spec["path"], dict))
        value = common.pinned_reference(name, path, config)
        result = {"sha256": hashlib.sha256(common.pinned_bytes(name, path, config)).hexdigest(), "gate": teacher.GATE_VARIANT,
                  "variants": list(spec["variants"]), "texts": value["texts"]}
        if (check or {}).get("constants", {}).get("teacher", {}).get("ALLOW_FLOOR_ONLY", {}).get("value") is True:
            result["binding"] = value.get("binding")
        return result


class PairedTeacher(Kind):
    """Non-regression against a stored accepted capture, in addition to the
    absolute gates."""

    name = "paired_teacher"
    modules = ("paired_teacher",)
    version = "paired_teacher.v4"
    inputs = "complete candidate and stored accepted teacher captures, and the declared margins"
    receipt = "both raw teacher arms and the margins digest"
    metrics = ("passed", "pooled", "by_category", "failed_categories", "affected_positions",
               "absolute", "absolute_failed_arms")
    verdict = "the declared paired margins pooled and per category, plus both existing absolute gates"

    def reference(self, check, config, revision=None):
        import common
        import paired_teacher
        absolute = paired_teacher.teacher_check(config, check)
        reference = KINDS["teacher_forced"].reference(absolute, config, revision)
        return {**reference, "checkpoint": common.checkpoint_pin(config), "absolute_check": absolute} if reference else None

    def verify(self, value, reference=None, check=None):
        import common
        import paired_teacher
        if reference is None:
            reference = self.reference(check, common.qualification_config())
        paired_teacher.verify_arms(value, reference, check)
        return ["arms", "policy_sha256"]

    def score(self, value, reference=None, check=None, mechanics=False):
        import common
        import paired_teacher
        admit(value, mechanics)
        require(not mechanics, "paired teacher needs complete live captures")
        if reference is None:
            reference = self.reference(check, common.qualification_config())
        metrics = paired_teacher.compare(value, reference, check)
        if any(rule["path"] and rule["path"][0] == "server" for rule in (check or {}).get("rules", [])):
            server = value["arms"]["candidate"]["server"]
            require(value.get("server") == server, "paired topology identity differs from the candidate capture")
            live = common.live_identity(server)
            require(live == common.live_identity(value.get("qualified_server", {})),
                    "paired topology identity differs from the qualified process")
            if (check or {}).get("_qualified_server") is not None:
                require(live == common.live_identity(check["_qualified_server"]),
                        "paired topology identity differs from the live parent")
            metrics["server"] = server
        return metrics

    def verify_score(self, value, reference=None, check=None, mechanics=False):
        # score verifies both arms before comparing them.
        return ["arms", "policy_sha256"], self.score(value, reference, check, mechanics)


class MediaForced(Shared):
    """Media probes: each reference item's answer teacher-forced against the
    pinned media reference's gate variant within the item's own spread
    limits, its greedy answer walked along the reference answer, its
    content and prompt token count checked."""

    name = "media_forced"
    modules = ("multimodal",)
    version = "media_forced.v1"
    inputs = ("the pinned media reference (per item: id, kind, answer token IDs and per variant the logprob and "
              "top-1 of each answer position), its prompt capture and the media fixtures")
    receipt = ("chopinfer-qualification/v1 `multimodal`: one row per reference item in order (id, kind, logprobs, "
               "greedy, fixture_expectation_match, token_count_match, passed), exact_reference_sha256")
    metrics = ("passed", "rows", "limits", "reference", "reference_sha256", "exact_reference_sha256",
               "capture_sha256", "reference_binding", "server", "mixed")
    verdict = "every item within its limits, greedy-consistent, matching its expectation and prompt count"

    def reference(self, check, config, revision=None):
        """The pinned media reference's cases, as multimodal.py loads and
        checks them, each with its variants under `exact`."""
        import common
        import multimodal
        config = multimodal.media_config(config)
        policy = config["qualification"]["media"]
        name = policy["reference"]
        spec = config["qualification"]["references"][name]
        exact = source(common.reference_path(name, config), revision, external=isinstance(spec["path"], dict))
        declared = common.input_path("media_capture", config=config)
        capture = source(declared, revision)
        inputs = [declared, declared.parent / "manifest.json"]
        if policy["format"] == "image-probes":
            inputs.append(declared.parent / "identity.json")
        for original in inputs:
            copied = source(original, revision)
            if revision is not None:
                from refs import fixture_bytes
                if copied.read_bytes() != fixture_bytes(original, config=config):
                    raise ValueError(f"locked media input differs at revision: {original.name}")
        common.pinned_reference(name, exact, config)
        if policy["format"] == "image-probes":
            captured, _, digest = multimodal.load_probes(capture)
            probes, rule, _ = multimodal.probe_references(digest, exact)
            return {"sha256": hashlib.sha256(common.pinned_bytes(name, exact, config)).hexdigest(), "format": "image-probes",
                    "probes": probes, "rule": rule, "order": [probe["name"] for probe in captured]}
        loaded, _, _ = multimodal.load_reference(capture, exact)
        variants = list(spec["variants"])
        cases = [{"id": case["id"], "kind": case["kind"],
                  "answer_token_ids": case["forced_prompt_token_ids"][case["answer_token_start"]:],
                  "exact": {variant: case["exact"][variant] for variant in variants}} for case in loaded["cases"]]
        result = {"sha256": hashlib.sha256(common.pinned_bytes(name, exact, config)).hexdigest(), "gate": multimodal.GATE_VARIANT,
                "variants": variants, "cases": cases, "audio_near_tie": policy.get("audio_near_tie")}
        if any(rule["path"] and rule["path"][0] == "mixed" for rule in check["rules"]):
            result["mixed"] = multimodal.mixed_reference(config, revision=revision)
        return result


class GreedyPin(Shared):
    """C1 greedy tokens of every request bit-equal to a baseline record:
    the pinned approved capture, or the same run's AR arm."""

    name = "greedy_pin"
    modules = ("greedy_equality",)
    version = "greedy_pin.v1"
    inputs = "the pinned exact reference's texts (prompt prefix tokens) and the baseline greedy record"
    receipt = ("chopinfer-qualification/v1 `greedy-equality-<arm>`: requests (name -> token IDs), "
               "finish_reasons, prompt_tokens, max_tokens, concurrency, k, reference_sha256")
    metrics = ("equal", "first_differences", "finish_differences")
    verdict = "rule exact: every request and finish reason equal; rule informational: reported only"


class Determinism(Shared):
    """Bit-identical C1 answers of two server starts (arms), of the two
    cold long-prompt sends within each, and of each prefix-hit send and its
    cold token-ID send."""

    name = "determinism"
    modules = ("determinism",)
    version = "determinism.v1"
    inputs = "none beyond the module's fixed request set (an image request when the package serves media)"
    receipt = ("chopinfer-qualification/v1 `determinism-<arm>`: requests (name -> tokens, logprobs, top "
               "logprobs, finish, cached tokens), image inclusion, run id")
    metrics = ("passed", "left_arm", "right_arm", "requests", "tokens", "equal", "first_difference",
               "max_abs_difference", "cold_repeat", "prefix_hit")
    verdict = "every compared answer bit-equal: across the arms, cold repeats and prefix hits"


class Api(Shared):
    """OpenAI-compatible API behavior: completions, chat, streaming, usage,
    token IDs, logprobs, tool calls and metrics."""

    name = "api"
    modules = ("api",)
    captured_by_eval = True
    version = "api.v1"
    inputs = "the declared chat_renders fixture and the frontend declaration"
    receipt = "chopinfer-qualification/v1 `api`: one row per api.v1 case (name, passed, details), errors, stream_text"
    metrics = ("passed", "rows")
    verdict = "every api.v1 case present and passed, and the declared predicates"


class Structured(Shared):
    """Structured output: JSON schema, choice, regex and tool cases, each
    output validated locally."""

    name = "structured"
    modules = ("structured",)
    captured_by_eval = True
    version = "structured.v1"
    inputs = "the structured.v1 case set and settings (repeat, stream, max tokens, temperature, seed)"
    receipt = ("chopinfer-qualification/v1 `structured`: one row per request (name, status), summary, "
               "settings")
    metrics = ("passed", "summary", "settings")
    verdict = "every judged output valid (validity_rate 1.0), none refused or truncated, and the declared predicates"


class FrontendGoldens(Shared):
    """Exact HF chat-render and tokenizer IDs, needles and HF API answer shapes."""

    name = "frontend_goldens"
    shared = False
    modules = ("frontend_goldens",)
    version = "frontend_goldens.v1"
    inputs = "declared frontend_capture/chat-renders.json and tokenizer_goldens; a served capture directory"
    receipt = "capture report: chat-renders.json and tokenizer-golden.json counts, mismatches and agrees"
    metrics = ("chat-renders.json", "tokenizer-golden.json", "needles.json", "api-samples.json", "coverage", "status", "reference")
    verdict = "both golden files have compared rows and exact token IDs; every declared predicate holds"

    def capture(self, command, environment, log, timeout):
        # Qualification requires the producer proof without changing the
        # declared numerical policy or the standalone diagnostic interface.
        environment = dict(environment, CHOPINFER_FRONTEND_REQUIRE_BOUND="1")
        return super().capture(command, environment, log, timeout)

    def reference(self, check, config, revision=None):
        import common
        paths = {"chat-renders.json": common.resolve_input("frontend_capture", config=config) / "chat-renders.json",
                 "tokenizer-golden.json": common.resolve_input("tokenizer_goldens", config=config)}
        shapes = common.resolve_input("frontend_capture", config=config) / "api-shapes.json"
        if source(shapes, revision).exists():
            paths["api-shapes.json"] = shapes
        files = {}
        for name, path in paths.items():
            resolved = source(path, revision)
            from refs import load_lock, read_member
            key = "frontend_capture" if name != "tokenizer-golden.json" else "tokenizer_goldens"
            if revision is None:
                data = read_member(config, key, resolved)
            else:
                data = resolved.read_bytes()
                lock = load_lock(config)
                member = name if key == "frontend_capture" else "."
                if lock is not None and hashlib.sha256(data).hexdigest() != lock["entries"][key]["members"][member]:
                    raise ValueError(f"lock entry {key}: sha256 differs for {member}")
            files[name] = {"input": "frontend_capture" if name != "tokenizer-golden.json" else "tokenizer_goldens",
                           "file": name if name != "tokenizer-golden.json" else "",
                           "sha256": hashlib.sha256(data).hexdigest(), "rows": len(json.loads(data))}
        return {"files": files}



def check_arguments(check):
    check = check or {}
    cohort = check.get("_cohort", "full")
    return check.get("args", []) + check.get("profile_args", check.get("cohort_args", {})).get(cohort, [])


def argument_number(args, flag, default):
    values = [args[index + 1] for index, value in enumerate(args[:-1]) if value == flag]
    try:
        value = float(values[-1]) if values else default
    except (TypeError, ValueError):
        raise Insufficient(f"invalid declared {flag}") from None
    require(finite(value) and value > 0, f"invalid declared {flag}")
    return value


def speculative_activity(value):
    return isinstance(value, dict) and all(natural(value.get(key)) and value[key] > 0
                                          for key in ("draft_tokens", "verify_items"))


class PrefixExact(Kind):
    name = "prefix_exact"
    modules = ("prefix_exact",)
    metrics = ("passed", "checks", "scores", "skipped", "inputs", "serving_identity")

    def verify(self, value, reference=None, check=None):
        self.score(value, reference, check)
        return list(self.metrics)

    def score(self, value, reference=None, check=None, mechanics=False):
        metrics = super().score(value, reference, check, mechanics)
        main = value.get("main", value)
        positions = (main.get("scores") or {}).get("all", {}).get("positions")
        require(natural(positions) and positions > 0 and main.get("skipped") == [],
                "prefix_exact: incomplete position coverage")
        flags = dict(main.get("checks") or {})
        if value.get("not_applicable"):
            inapplicable_reason(value, check)
            require(flags.get("media") is None, "prefix_exact: unserved image marked passed")
            flags.pop("media")
        require(all(flags.get(name) is True for name in
                    ("cache", "hit_equals_cold", "chunked_equals_cold", "hit_vs_cold")),
                "prefix_exact: missing or failed bitwise checks")
        metrics["passed"] = bool(flags) and all(flag is True for flag in flags.values()) \
            and all(row.get("passed") is True for row in value.get("splits", []))
        return metrics


class SpecVerify(Kind):
    name = "spec_verify"
    modules = ("spec_verify",)
    # `bound`: the checkpoint binding both arms share, which a declaration
    # pinned to another checkpoint holds to its identity.
    metrics = ("passed", "sets", "arms", "ar_speculation", "gates", "bound")

    def verify(self, value, reference=None, check=None):
        self.score(value, reference, check)
        return list(self.metrics)

    def score(self, value, reference=None, check=None, mechanics=False):
        metrics = super().score(value, reference, check, mechanics)
        require(value.get("arms") == {"ar": "chopinfer-ar", "spec": "chopinfer-spec"},
                "spec_verify: not distinct speculative and AR arms")
        control = value.get("ar_speculation")
        require(isinstance(control, dict) and not control.get("draft_tokens", 0)
                and not control.get("verify_items", 0), "spec_verify: AR control drafted")
        groups = value.get("sets")
        require(isinstance(groups, dict) and bool(groups) and "copy" in groups, "spec_verify: no copy set")
        for group in groups.values():
            require(all(natural(group.get(key)) and group[key] > 0 for key in ("requests", "scored_positions")),
                    "spec_verify: empty request or position set")
            # Reports before the coverage block have none; a report with one
            # must not claim positions or completeness its requests lack.
            if "coverage" in group:
                require(spec_verify_coverage(group), "spec_verify: inconsistent coverage")
        copy = groups["copy"]
        require(speculative_activity(copy.get("speculation")), "spec_verify: no drafting activity")
        from spec_verify import MAXIMUM_ABS_DLOGPROB, MAXIMUM_TOP1_DISAGREEMENT
        metrics["passed"] = all(finite(copy.get(key)) and 0 <= copy[key] <= bound for key, bound in (
            ("mean_abs_dlogprob", MAXIMUM_ABS_DLOGPROB), ("top1_disagreement", MAXIMUM_TOP1_DISAGREEMENT)))
        return metrics


class SpecContinuation(Kind):
    """spec_continuation's receipt (spec-continuation.v1): the verdict its
    module computed, held to its own counts. An insufficient verdict stays
    insufficient; a pass needs complete coverage in both arms and a GPU
    execution record of every reference entry it scored."""
    name = "spec_continuation"
    modules = ("spec_continuation",)
    metrics = ("passed", "status", "rule", "failed_sets", "insufficient", "high_entropy_positions", "bound")

    def verify(self, value, reference=None, check=None):
        self.score(value, reference, check)
        return list(self.metrics)

    def score(self, value, reference=None, check=None, mechanics=False):
        metrics = super().score(value, reference, check, mechanics)
        from spec_continuation import EXECUTION_SCHEMA, MODE_VARIANTS, RULE
        require(value.get("rule") == RULE, f"spec_continuation: not a {RULE} verdict")
        status, failed, short = value.get("status"), value.get("failed_sets"), value.get("insufficient")
        require(status in ("pass", "fail", "insufficient") and value.get("passed") is (status == "pass")
                and isinstance(failed, list) and isinstance(short, list)
                and (status == "fail") == bool(failed) and (status == "pass") == (not failed and not short),
                "spec_continuation: inconsistent verdict")
        execution = value.get("execution")
        require(isinstance(execution, dict) and isinstance(execution.get("cpu_scoring"), dict)
                and isinstance(execution.get("gpu"), dict), "spec_continuation: no execution record")
        for arm in ("spec", "ar"):
            report = value.get(arm)
            require(isinstance(report, dict) and isinstance(report.get("sets"), dict) and report["sets"],
                    f"spec_continuation: no {arm} report")
            groups = [*report["sets"].values(), report.get("greedy_pooled") or {}]
            if status == "pass":
                require(all((group.get("coverage") or {}).get("complete") is True for group in groups
                            if group.get("coverage", {}).get("requests")),
                        f"spec_continuation: a pass with incomplete {arm} coverage")
            requests = sum(group["coverage"]["requests"] for group in report["sets"].values())
            runs = execution["gpu"].get(arm)
            require(isinstance(runs, list) and runs and all(
                isinstance(run, dict) and run.get("schema") == EXECUTION_SCHEMA and run.get("exit") == 0
                and isinstance(run.get("gpus"), list) and run["gpus"] and isinstance(run.get("entries"), dict)
                for run in runs),
                f"spec_continuation: no GPU execution record of the {arm} reference")
            for mode in MODE_VARIANTS:
                counted = sum(run["entries"].get(mode, {}).get("computed", 0)
                              + run["entries"].get(mode, {}).get("cache_hits", 0) for run in runs)
                require(counted == requests,
                        f"spec_continuation: the {arm} {mode} execution covers {counted} of {requests} requests")
        if status == "insufficient":
            raise Insufficient("spec_continuation: " + "; ".join(map(str, short)))
        metrics["passed"] = status == "pass"
        return metrics


def spec_verify_coverage(group):
    """Whether a spec_verify set's coverage agrees with its own requests and
    scored positions: every count adds up, and `complete` holds exactly when
    no request of either arm leaves a position unscored."""
    coverage = group["coverage"]
    if not isinstance(coverage, dict) or not isinstance(coverage.get("requests"), list):
        return False
    rows = coverage["requests"]
    keys = ("emitted", "ar_emitted", "common_prefix", "scored", "unscored")
    if len(rows) != group["requests"] or not all(
            isinstance(row, dict) and all(natural(row.get(key)) for key in keys)
            and row["scored"] == min(row["common_prefix"] + 1, row["emitted"], row["ar_emitted"])
            and row["scored"] + row["unscored"] == row["emitted"] for row in rows):
        return False
    emitted, scored = sum(row["emitted"] for row in rows), sum(row["scored"] for row in rows)
    ar_unscored = sum(row["ar_emitted"] - row["scored"] for row in rows)
    complete = sum(row["unscored"] == 0 and row["ar_emitted"] == row["scored"] for row in rows)
    return (coverage.get("emitted_positions") == emitted and coverage.get("scored_positions") == scored
            and group["scored_positions"] == scored and coverage.get("unscored_positions") == emitted - scored
            and coverage.get("ar_unscored_positions") == ar_unscored
            and coverage.get("complete_requests") == complete
            and coverage.get("complete") is (emitted > 0 and scored == emitted and ar_unscored == 0))


class VerifyExact(Kind):
    name = "verify_exact"
    modules = ("verify_exact",)
    metrics = ("records", "ar_k", "ar_arm", "ar_speculation")

    def verify(self, value, reference=None, check=None):
        self.score(value, reference, check)
        return list(self.metrics)

    def score(self, value, reference=None, check=None, mechanics=False):
        metrics = super().score(value, reference, check, mechanics)
        require(value.get("ar_k") == 0 and isinstance(value.get("ar_arm"), str), "verify_exact: no AR control")
        control = value.get("ar_speculation")
        require(isinstance(control, dict) and not control.get("draft_tokens", 0)
                and not control.get("verify_items", 0), "verify_exact: AR control drafted")
        from verify_exact import row_capacity_coverage
        args = check_arguments(check)

        def capacities(flag):
            requested = []
            for index, word in enumerate(args):
                if word == flag:
                    try:
                        capacity = int(args[index + 1])
                    except (IndexError, TypeError, ValueError):
                        raise Insufficient(f"verify_exact: invalid declared {flag}") from None
                    require(capacity > 0, f"verify_exact: invalid declared {flag}")
                    requested.append(capacity)
            return requested

        ar_capacities, candidate_capacities = capacities("--ar-row-capacity"), capacities("--row-capacity")
        required_shapes = "--require-row-shapes" in args
        if ar_capacities or required_shapes:
            require(row_capacity_coverage(control, ar_capacities or None, False)["passed"],
                    "verify_exact: incomplete AR capacity coverage")
        records = value.get("records")
        require(isinstance(records, dict) and bool(records), "verify_exact: no records")
        for row in records.values():
            require(isinstance(row, dict) and natural(row.get("k")) and row["k"] > 0
                    and isinstance(row.get("arm"), str) and row["arm"] != value["ar_arm"],
                    "verify_exact: candidate is not a different speculative arm")
            require(speculative_activity(row.get("speculation")), "verify_exact: no drafting activity")
            if candidate_capacities or required_shapes:
                require(row_capacity_coverage(row.get("speculation"), candidate_capacities or None, True)["passed"],
                        "verify_exact: incomplete speculative capacity coverage")
            coverage = row.get("coverage") or {}
            if coverage.get("rule") == "every forced decode position":
                require(natural(coverage.get("expected")) and coverage["expected"] > 0
                        and coverage.get("scored") == coverage["expected"]
                        and coverage.get("missing") == [] and coverage.get("unexpected") == [],
                        "verify_exact: incomplete forced position coverage")
            else:
                require(coverage.get("rule") == "host-drafted" and natural(coverage.get("minimum"))
                        and coverage["minimum"] > 0 and natural(coverage.get("positions"))
                        and coverage["positions"] >= coverage["minimum"], "verify_exact: no verified positions")
            require(coverage.get("passed") is True, "verify_exact: incomplete coverage")
            require((row.get("exact") or {}).get("all", {}).get("gate_against_ar", {}).get("passed") is True,
                    "verify_exact: failed numerical comparison")
        return metrics


class Soak(Kind):
    name = "soak"
    modules = ("soak",)
    metrics = ("passed", "completed", "duration_seconds", "measured_duration_seconds", "concurrency",
               "memory_samples", "memory_scope", "memory_growth_mib", "host_memory_growth_mib",
               "steady_memory_growth_mib", "steady_host_memory_growth_mib", "mixed_requests", "prefix_mix")
    verdict = ("complete measured interval, bounded host and device memory growth (after any declared warm-up, "
               "and over the whole soak within any declared cache allowance), and exercised mixes")

    def verify(self, value, reference=None, check=None):
        self.score(value, reference, check)
        return list(self.metrics)

    def score(self, value, reference=None, check=None, mechanics=False):
        metrics = super().score(value, reference, check, mechanics)
        if mechanics and value.get("mechanics_only"):
            return metrics
        args = check_arguments(check)
        duration = argument_number(args, "--duration", 1800)
        concurrency = argument_number(args, "--concurrency", 32)
        require(natural(value.get("completed")) and value["completed"] > 0, "soak: no completed requests")
        require(finite(value.get("duration_seconds")) and value["duration_seconds"] >= duration
                and finite(value.get("measured_duration_seconds"))
                and value["measured_duration_seconds"] >= duration, "soak: shorter than the declared duration")
        require(natural(value.get("concurrency")) and value["concurrency"] >= concurrency,
                "soak: below the declared concurrency")
        require("--remote-only" not in args and value.get("memory_scope") == "serving_process_tree",
                "soak: no serving-process memory measurements")
        samples = value.get("memory_samples")
        require(isinstance(samples, list) and len(samples) >= 2, "soak: need memory samples over time")
        require(all(isinstance(row, dict) and all(finite(row.get(key)) and row[key] >= 0
                    for key in ("elapsed_seconds", "mib", "host_rss_mib")) for row in samples),
                "soak: incomplete device or host memory samples")
        times = [row["elapsed_seconds"] for row in samples]
        require(times[0] <= 1 and times[-1] >= duration and all(0 < b - a <= 60
                for a, b in zip(times, times[1:])), "soak: samples do not span the measurement interval")
        require(times[-1] <= value["measured_duration_seconds"] + 1,
                "soak: sample lies outside the measured interval")
        # The declared warm-up and cache allowance (soak.memory_rule); the
        # defaults judge the growth from the first sample alone.
        import soak
        declared = (check or {}).get("constants", {}).get("soak", {})
        try:
            warmup, allowance = soak.memory_policy(declared.get("WARMUP_SECONDS", {}).get("value", 0),
                                                   declared.get("CACHE_ALLOWANCE_MIB", {}).get("value",
                                                                                                {"host": 0, "device": 0}))
        except ValueError as error:
            raise Insufficient(f"soak: {error}") from None
        require(value.get("warmup_seconds", 0) == warmup
                and value.get("cache_allowance_mib", {"host": 0, "device": 0}) == allowance,
                "soak: the receipt was judged under another warm-up or cache allowance")
        rule = soak.memory_rule(samples, duration, {
            "device": min(512, argument_number(args, "--memory-growth-mib", 512)),
            "host": min(512, argument_number(args, "--host-memory-growth-mib", 512))}, warmup, allowance)
        growth = {key: rule[key] for key in ("memory_growth_mib", "host_memory_growth_mib",
                                              "steady_memory_growth_mib", "steady_host_memory_growth_mib")}
        passed = rule["passed"]
        mixed = value.get("mixed_requests") or {}
        for flag, key in (("--media-mix", "media"), ("--structured-mix", "structured")):
            enabled = flag in args
            if enabled and key == "media" and value.get("not_applicable"):
                inapplicable_reason(value, check)
                import common
                enabled = any(not common.not_applicable(value["server"], kind) for kind in soak_media_kinds(check))
            if enabled:
                require(natural(mixed.get(key)) and mixed[key] > 0, f"soak: no {key} requests")
        if "--structured-mix" in args:
            passed = passed and mixed.get("structured_truncated") == 0
        if "--prefix-mix" in args:
            prefix = value.get("prefix_mix") or {}
            passed = passed and prefix.get("canary_passed") is True and prefix.get("hits_total", 0) > 0 \
                and prefix.get("idle_blocks_used") == 0 and prefix.get("idle_active_sequences") == 0
        metrics.update(growth, passed=passed)
        return metrics


class KvOffload(Kind):
    """Recompute the saved paired capture through the Rust limit rule."""

    name = "kv_offload"
    modules = ("kv_offload",)
    version = "kv_offload.v1"
    metrics = ("passed", "count", "gates", "c16_equality")

    def reference(self, check, config, revision=None):
        return {"facts": config["kv_offload"],
                "speculative": config["preview"]["speculation"]["tokens"] > 0}

    def score(self, value, reference=None, check=None, mechanics=False):
        admit(value, mechanics)
        import limits
        policy = value["policy"] if reference is None else reference
        return limits.one("kv_offload", **value["records"], **policy)


RECEIPT = Kind()
KINDS = {kind.name: kind for kind in (TeacherForced(), PairedTeacher(), MediaForced(), GreedyPin(), Determinism(), Api(),
                                      Structured(), FrontendGoldens(), MediaPreprocessing(), Soak(), SpecVerify(), SpecContinuation(),
                                      VerifyExact(), PrefixExact(), KvOffload(), RECEIPT)}
MODULES = {module: kind for kind in KINDS.values() for module in kind.modules}


def of(module):
    """The kind of a check run by `module`; a module no kind names is judged
    by its declared predicates over its receipt (`receipt`)."""
    return MODULES.get(module, RECEIPT)


def require(condition, message):
    if not condition:
        raise Insufficient(message)


def soak_media_kinds(check):
    args = check_arguments(check)
    if "--media-mix" not in args:
        return set()
    if "--media-cases" in args:
        return {case.split("-", 1)[0] for case in args[args.index("--media-cases") + 1].split(",")}
    from soak import MEDIA_CASES
    return {case.split("-", 1)[0] for case in MEDIA_CASES}


def inapplicable_reason(value, check):
    """Validate a partial receipt's modality reasons and recorded server identity."""
    import common
    reasons = value.get("not_applicable")
    if not reasons:
        return None
    require(isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons),
            "invalid not_applicable reasons")
    require(value.get("schema") == "chopinfer-qualification/v1" and not value.get("mechanics_only"),
            "not_applicable needs a qualification receipt")
    server = value.get("server")
    require(isinstance(server, dict), "no server for not_applicable evidence")
    served = common.served_modalities(server)
    require(served is not None and "text" in served and served <= set(common.MODALITIES),
            "no valid served modalities for not_applicable evidence")
    try:
        identity = common.live_identity(server)
        require(identity["checkpoint_identity"] == common.checkpoint_pin()["checkpoint_identity"],
                "not_applicable evidence names another checkpoint")
    except (ValueError, KeyError, TypeError) as error:
        raise Insufficient(str(error)) from None
    module = (check or {}).get("module")
    require(module in ("multimodal", "grounding", "soak", "prefix_exact"), "this check has no optional media")
    if module == "multimodal":
        rows = value.get("rows")
        require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), "no media rows")
        expected = sorted({row.get("reason") for row in rows if row.get("status") == "not_applicable"
                           and isinstance(row.get("reason"), str)})
    else:
        kinds = ("image",) if module in ("grounding", "prefix_exact") else soak_media_kinds(check)
        expected = sorted({reason for kind in kinds if (reason := common.not_applicable(server, kind))})
    require(bool(expected) and reasons == expected, "not_applicable reasons do not match the media evidence")
    return "; ".join(reasons)


def admit_pair(value, reference, check, mechanics=False):
    """Raise ValueError unless the record a check's reference_file names
    is admitted as the receipt is (admit) and the two hold the same,
    present value at each of the check's `reference_equal` paths (one run,
    checkpoint and build, say, for two arms that differ in their serving
    options)."""
    admit(reference, mechanics)
    differing = []
    for path in check.get("reference_equal", []):
        try:
            left, right = select(value, path), select(reference, path)
        except (KeyError, IndexError, TypeError):
            left, right = None, None
        if not left or left != right or any(item is None for item in left):
            differing.append("/".join(map(str, path)))
    if differing:
        raise ValueError(f"the reference record is missing or differs at {', '.join(differing)}")


def rescore(kind, value, reference, check, mechanics=False):
    """The offline verdict of one stored receipt: admit it as the gate
    does, verify its fields and coverage (skipped for a mechanics-only
    receipt, which never qualifies), score and judge. Returns status,
    metrics, predicates, the fields read as recorded and the error."""
    result = {"status": "fail", "metrics": None, "predicates": None, "recorded_fields": []}
    if isinstance(kind, Shared):
        try:
            admit(value, mechanics)
            return call("rescore", kind, value, reference, check, mechanics)
        except Insufficient as error:
            return {**result, "status": "insufficient", "error": str(error)}
        except ValueError as error:
            return {**result, "error": str(error)}
    try:
        admit(value, mechanics)
        reason = inapplicable_reason(value, check)
        result["recorded_fields"], result["metrics"] = kind.verify_score(value, reference, check, mechanics)
        if reason and check.get("module") == "grounding":
            require(not value.get("rows"), "unserved grounding receipt contains results")
            return {**result, "status": "not_applicable", "reason": reason}
    except Insufficient as error:
        return {**result, "status": "insufficient", "error": str(error)}
    except (ValueError, KeyError, TypeError) as error:
        return {**result, "error": str(error)}
    try:
        status, result["predicates"] = kind.judge(result["metrics"], check)
    except KeyError as error:
        return {**result, "error": f"missing metric {error}"}
    if status == "fail":
        result["error"] = "threshold predicate failed"
    if status == "pass" and reason:
        return {**result, "status": "not_applicable", "reason": reason}
    return {**result, "status": status}


def describe():
    """Each kind's interface as JSON-ready rows."""
    return [{"kind": kind.name, "version": kind.version, "shared": kind.shared, "modules": list(kind.modules),
             "inputs": kind.inputs, "receipt": kind.receipt, "metrics": list(kind.metrics),
             "verdict": kind.verdict} for kind in KINDS.values()]
