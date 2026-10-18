"""Run versioned release checks against already-running HTTP servers.

No server lifecycle or device operations belong here. A model's
`models/<m>/qual.json` (declaration.py) selects the qualification modules,
their arguments, limits and receipt predicates. A check of a kind captured by
eval (kinds.py `captured_by_eval`: teacher_forced, api, structured) runs as
`chopinfer eval run --check ID` against the declaration it came from; every
other check, and those kinds' mechanics checks in a release-gates/v1 config
(which eval does not read), runs its Python module as a worker.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from string import Formatter
import subprocess
import sys
import time
import uuid

from common import require_qualified_model_options, Client, ROOT, input_path
import declaration
import kinds


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The predicate rules live with the check kinds; these names stay for callers.
select = kinds.select
judge = kinds.predicate


class CheckRaw(dict):
    """{check ID: its raw directory}, which `{check_raw[ID]}` in a check's
    arguments reads: the directory check ID wrote its files to, once it has
    passed earlier in this run. A gate run's checks share one directory; a
    checks.py run takes the directory with `--with-raw ID=DIR`."""

    def __missing__(self, key):
        raise ValueError(f"no raw directory of check {key}: it has not passed earlier in this run "
                         f"(a checks.py run takes it with --with-raw {key}=DIR)")


def placeholder_fields(arguments):
    """The names `arguments` read: a placeholder's field, or for
    `{check_raw[ID]}` the earlier check ID."""
    fields = {field for arg in arguments for _, field, _, _ in Formatter().parse(arg) if field}
    return {field[len("check_raw["):-1] if field.startswith("check_raw[") and field.endswith("]") else field
            for field in fields}


def validate(config, *, external=False):
    if config["schema"] != "chopinfer-release-gates/v1":
        raise ValueError("unsupported gate config schema")
    qualification = config.get("qualification", {})
    if not isinstance(qualification, dict):
        raise ValueError("qualification must be an object")
    if external and "mechanics_only" not in qualification:
        raise ValueError("external qualification must declare mechanics_only")
    if type(qualification.get("mechanics_only", False)) is not bool:
        raise ValueError("qualification.mechanics_only must be boolean")
    ids = [check["id"] for check in config["checks"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("checks must have unique, nonempty IDs")
    for check in config["checks"]:
        if not check["module"].isidentifier() or not (Path(__file__).parent / (check["module"] + ".py")).is_file():
            raise ValueError("check must name an existing qualification module")
        if not check["rules"] or check["timeout_seconds"] <= 0:
            raise ValueError("each check needs predicates and a positive timeout")
        if "informational_reason" in check and not check["informational_reason"].strip():
            raise ValueError("informational checks need a reason")
        # Declared waivers pass eval's one waiver validator before any judgment.
        kinds.validated_waivers(check)
        for rule in check["rules"] + [r for rules in check.get("input_rules", {}).values() for r in rules]:
            if not rule["source"] or rule["op"] not in ("eq", "le", "ge", "count_ge", "true_count_ge"):
                raise ValueError("each predicate needs a source and supported operator")
        for constants in check.get("constants", {}).values():
            if any(not entry["source"] for entry in constants.values()):
                raise ValueError("each limit needs a source")
    for profile in config["profiles"].values():
        if not profile["checks"] or any(name not in ids for name in profile["checks"]):
            raise ValueError("profile must select known checks")


def describe(config, profile):
    """Describe the existing cohort without importing workers or resolving inputs.

    Serving profiles and quick/full cohorts are different dimensions. This is
    guidance only; run() retains its prerequisite order and execution policy.
    """
    if config.get("schema") != "chopinfer-release-gates/v1":
        raise ValueError("unsupported gate config schema")
    policy = config["profiles"][profile]
    checks = {check["id"]: check for check in config["checks"]}
    if not checks or len(checks) != len(config["checks"]) or not policy["checks"]:
        raise ValueError("empty or duplicate checks")
    rows = []
    earlier = set()
    for name in policy["checks"]:
        check = checks[name]
        arguments = check["args"] + check.get("profile_args", {}).get(profile, [])
        placeholders = placeholder_fields(arguments)
        comparisons = sorted(placeholders & earlier)
        prerequisites = sorted(placeholders - earlier - {"raw", "run", "run_id"})
        prerequisites += ["file: " + key for key in check.get("files", [])]
        prerequisites += ["earlier receipt: " + key for key in comparisons]
        rows.append({"id": name, "module": check["module"], "arguments": arguments,
                     "prerequisites": prerequisites, "input_rules": check.get("input_rules", {}),
                     "rules": check["rules"], "constants": check.get("constants", {}),
                     "timeout_seconds": check["timeout_seconds"],
                     "informational_reason": check.get("informational_reason"),
                     "receipt": {key: check[key] for key in ("receipt", "stdout_json", "json_file", "reference_file")
                                 if key in check}})
        earlier.add(name)
    return {"schema": "chopinfer-release-plan-v1", "mode": "plan-only", "model": config["model"],
            "cohort": profile,
            "policy_sha256": hashlib.sha256((json.dumps(config, indent=2) + "\n").encode()).hexdigest(),
            "selected_checks": list(policy["checks"]), "checks": rows,
            "inputs": config.get("inputs", {}), "limitations": config.get("limitations", []),
            "qualified": False,
            "execution": "not_run", "preflight": "Resolve all prerequisites before live checks; no resolution performed here"}


def worker(config_path, check_id, arguments):
    try:
        config = json.loads(Path(config_path).read_text())
        check = next(check for check in config["checks"] if check["id"] == check_id)
        os.environ["CHOPINFER_QUAL_CONFIG"] = str(Path(config_path).resolve())
        from refs import verify_config
        verify_config(config, policy_only=True)
        module = importlib.import_module(check["module"])
        if not hasattr(module, "main"):
            raise ValueError(f"{check['module']} has no live capture here: chopinfer eval run captures "
                             "it from a model's qual.json")
        for name, constants in check.get("constants", {}).items():
            target = importlib.import_module(name)
            for key, entry in constants.items():
                if not hasattr(target, key):
                    raise ValueError(f"unknown threshold {name}.{key}")
                setattr(target, key, entry["value"])
        sys.argv = [check["module"] + ".py", *arguments]
        return module.main()
    except (OSError, ValueError) as error:
        argparse.ArgumentParser(prog=check_id).error(str(error))


def eval_source(config, check, model_root=None):
    """The model directory and configuration `chopinfer eval run` reads
    `check` from: the qual.json of `config`'s model under the model root and
    the configuration `config` expands (None for the model's own). None when
    `config` is not a model declaration eval reads (the mechanics fixture,
    another release-gates/v1 config). Raises ValueError when that
    declaration states another check than `config` does."""
    model, variant = config.get("model"), config.get("variant")
    root = Path(model_root) if model_root is not None else ROOT
    directory = root / "models" / str(model)
    if (not declaration.plain(model) or (variant is not None and not declaration.plain(variant))
            or not (directory / declaration.NAME).is_file()):
        return None
    variants = declaration.require_variant_root(config) if declaration.external(config) else None
    if variant is None:
        source = directory / declaration.NAME
    elif variants is not None or variant in declaration.configurations(model, root):
        source = declaration.materialized(model, variant, root)
    else:
        source = declaration.path(model, root, variant)
    declared = next((entry for entry in declaration.load(source)["checks"] if entry["id"] == check["id"]), None)
    if declared != check:
        raise ValueError(f"{source} declares another {check['id']} check than this run's policy; "
                         "chopinfer eval run would capture that one")
    return directory, variant, variants


def eval_command(check, profile, context, source, report):
    """`chopinfer eval run` of one declared check against the server at
    `context`'s URL, bound to its startup log when the run has one, writing
    its report to `report`. An external variant is read from its root."""
    directory, variant, variants = source
    return [str(kinds.eval_binary()), "eval", "run", "--model", str(directory),
            *(["--variant", variant] if variant else []),
            *(["--variant-dir", str(variants)] if variants else []), "--served-model", context["model"],
            "--check", check["id"], "--cohort", profile, "--url", context["url"],
            *(["--server-log", str(context["server_log"])] if context.get("server_log") else []),
            "--report", str(report)]


def eval_record(check, arguments, report, copy=True):
    """The record `chopinfer eval run` wrote for `check` under `report`,
    with its compact capture copied to the declared `--capture-out` path
    (unless not `copy`)."""
    path = report / "records" / f"{check['id']}.json"
    if not path.is_file():
        raise ValueError(f"chopinfer eval run wrote no record of {check['id']}")
    if copy and "--capture-out" in arguments:
        capture = report / "captures" / f"{check['id']}.json"
        if not capture.is_file():
            raise ValueError(f"chopinfer eval run wrote no capture of {check['id']}")
        Path(arguments[arguments.index("--capture-out") + 1]).write_bytes(capture.read_bytes())
    return path


def eval_judged(report, name, rules=None, metrics=None):
    """Whether the `chopinfer eval run` report at `report` is a completed
    verdict that failed only on `name`'s declared predicates, which eval
    exits 1 for. The report must be complete as eval writes one: exit the
    integer 1 and a null run failure; `name` its one check, status fail,
    with its record the file of the digest the report names, an object of
    metrics, and as its error only eval's own "threshold predicate failed";
    and a nonempty list of whole predicates (path, op, value, source and a
    boolean passed), at least one failed, each passed as it holds over those
    metrics. With `rules`, the predicates are those declared rules; with
    `metrics`, the gate's own scoring of the record, each predicate measures
    there what it measures in the report. A process, capture or run failure,
    which eval may also exit 1 for, or a partial report, is not."""
    try:
        value = json.loads(Path(report).read_text())
        entries = value["checks"]
        if not (value["schema"] == "chopinfer.qual-report.v1" and type(value["exit"]) is int
                and value["exit"] == 1 and value["failure"] is None
                and isinstance(entries, list) and len(entries) == 1 and isinstance(entries[0], dict)):
            return False
        entry = entries[0]
        record = Path(report).parent / "records" / f"{name}.json"
        if not (entry["id"] == name and entry["status"] == "fail" and entry["record"] == f"records/{name}.json"
                and record.is_file() and entry["record_sha256"] == digest(record)
                and entry["error"] == "threshold predicate failed"):
            return False
        measured, predicates = entry["metrics"], entry["predicates"]
        if not isinstance(measured, dict) or not measured or not isinstance(predicates, list) or not predicates:
            return False
        exact = lambda item: json.dumps(item, sort_keys=True)
        for predicate in predicates:
            if not (isinstance(predicate, dict) and set(predicate) == {"path", "op", "value", "source", "passed"}
                    and isinstance(predicate["path"], list) and predicate["path"]
                    and predicate["op"] in ("eq", "le", "ge", "count_ge", "true_count_ge")
                    and isinstance(predicate["source"], str) and predicate["source"]
                    and type(predicate["passed"]) is bool
                    and kinds.select(measured, predicate["path"])
                    and kinds.predicate(measured, predicate) is predicate["passed"]):
                return False
            if metrics is not None and exact(kinds.select(metrics, predicate["path"])) \
                    != exact(kinds.select(measured, predicate["path"])):
                return False
        if rules is not None and [exact({key: rule[key] for key in ("path", "op", "value", "source")})
                                  for rule in rules] != \
                [exact({key: predicate[key] for key in ("path", "op", "value", "source")})
                 for predicate in predicates]:
            return False
        return any(predicate["passed"] is False for predicate in predicates)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def source_revision(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        # Exported SDK sources use a content-checked pin instead of Git metadata.
        return None


def prerequisites(config, names, profile, context, result, model_root=None):
    """Resolve the arguments, files and input predicates of the checks
    `names` (other than informational ones) before any live check runs,
    recording the SHA-256 of each input file in `result`."""
    for check in config["checks"]:
        if check["id"] not in names or check.get("informational_reason"):
            continue
        earlier = {c["id"]: "<earlier receipt>" for c in config["checks"]}
        for arg in check["args"] + check.get("profile_args", {}).get(profile, []):
            try:
                arg.format_map({**earlier, **context, "check_raw": CheckRaw({
                    **{name: "<earlier raw directory>" for name in earlier}, **context.get("check_raw", {})})})
            except KeyError as error:
                name = error.args[0]
                raise ValueError(f"{check['id']} needs {name}: an input, or the receipt of an earlier check that "
                                 f"passed in this run (a checks.py run takes it with --with {name}=RECEIPT)") from None
        for key in check.get("files", []):
            path = Path(context[key])
            if not path.exists():
                raise ValueError(f"missing input {key}: {path}")
            from refs import verify_input
            if key == "layer_reference":
                verify_input(config, key, path, model_root)
            if path.is_file():
                result.setdefault("input_sha256", {})[key] = digest(path)
        for key, rules in check.get("input_rules", {}).items():
            judge_input(key, json.loads(Path(context[key]).read_text()), rules, context)


def judge_input(key, value, rules, context):
    """Raise ValueError unless the input record `value` is complete quality
    evidence that holds every one of its `rules`, their string values
    formatted from `context` (KeyError for a name it lacks). The live
    preflight and rescore judge inputs alike."""
    if not isinstance(value, dict):
        raise ValueError(f"{key}: expected a JSON object")
    if value.get("mechanics_only") or value.get("skipped") or value.get("errors"):
        raise ValueError(f"{key}: incomplete or failed quality evidence")
    for rule in rules:
        resolved = dict(rule)
        if isinstance(resolved["value"], str):
            resolved["value"] = resolved["value"].format_map(context)
        if not judge(value, resolved):
            raise ValueError(f"{key}: input predicate failed: {rule['path']}")


def run_check(check, rows, profile, context, directory, config_path, deadline, mechanics, model_root):
    """Run one declared check through its kind and append its row to
    `rows`: capture by its module or by `chopinfer eval run`, score its
    receipt and judge the metrics. A passing check's receipt path joins
    `context` under its id for later checks. Returns the row."""
    name = check["id"]
    raw = directory / "checks"
    if name in context.get("check_raw", {}):
        # Only a passing run publishes a check's receipt and raw directory; a
        # second run of it withdraws the first's until it passes.
        del context["check_raw"][name]
        context.pop(name, None)
    kind = kinds.of(check["module"])
    row = {"id": name, "passed": False, "rules": check["rules"]}
    rows.append(row)
    if check.get("informational_reason"):
        row.update(passed=None, status="informational", evaluated=False,
                   reason=check["informational_reason"], seconds=0)
        print(f"{name}: INFORMATIONAL (not evaluated): {row['reason']}", flush=True)
        return row
    step_started = time.monotonic()
    try:
        remaining = deadline - step_started
        if remaining <= 0:
            raise TimeoutError("the check's deadline has passed")
        arguments = [arg.format_map(context) for arg in
                     check["args"] + check.get("profile_args", {}).get(profile, [])]
        source = None
        if kind.captured_by_eval:
            source = eval_source(json.loads(Path(config_path).read_text()), check, model_root)
        if source is not None:
            if mechanics:
                raise ValueError("chopinfer eval run captures a model declaration's checks; "
                                 "a mechanics run reads the mechanics fixture")
            report = raw / f"eval-{name}"
            command = eval_command(check, profile, context, source, report)
            row["eval_report"] = str(report / "report.json")
        else:
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--worker", str(config_path), name, *arguments]
        row["arguments"] = arguments
        log = directory / f"{name}.log"
        row["log"] = str(log)
        before = set(raw.glob(check.get("receipt", "*.json")))
        environment = {**os.environ, "CHOPINFER_QUAL_RECEIPTS": str(raw)}
        environment.pop("CHOPINFER_QUAL_MODEL_ROOT", None)
        if model_root is not None:
            environment["CHOPINFER_QUAL_MODEL_ROOT"] = str(Path(model_root).resolve())
        returncode = kind.capture(command, environment, log, min(remaining, check["timeout_seconds"]))
        row["exit_code"] = returncode
        row["log"] = str(log)
        # Eval exits 1 for a completed verdict that failed a declared
        # predicate as for a failed run; only that verdict's complete report
        # admits its record, scored as a worker's receipt is, and it cannot
        # pass.
        judged = (source is not None and returncode == 1
                  and eval_judged(report / "report.json", name, check["rules"]))
        if returncode and not judged:
            raise ValueError(f"check exited {returncode}")
        unlinked = dict(row)
        if source is not None:
            # A judged verdict's capture is published once its record is.
            path = eval_record(check, arguments, report, copy=not judged)
        elif "json_file" in check:
            path = Path(check["json_file"].format_map(context))
        elif check.get("stdout_json"):
            path = log
        else:
            created = set(raw.glob(check["receipt"])) - before
            if len(created) != 1:
                raise ValueError(f"expected one new receipt, found {len(created)}")
            path = created.pop()
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object receipt")
        row["receipt"] = str(path)
        row["sha256"] = digest(path)
        # A kind that scores one record against another (determinism's
        # other arm) reads the declared reference file as its reference.
        named = None
        if "reference_file" in check:
            reference_path = Path(check["reference_file"].format_map(context))
            named = json.loads(reference_path.read_text())
            row["reference_file"] = str(reference_path)
            row["reference_sha256"] = digest(reference_path)
            kinds.admit_pair(value, named, check, mechanics)
        selected = {**check, "_cohort": profile}
        if kind.name == "paired_teacher":
            selected["_qualified_server"] = context.get("_qualified_server")
        if value.get("not_applicable"):
            with kinds.policy(config_path, selected):
                reference = named if named is not None else kind.reference(selected, declaration.load(config_path))
                scored = kinds.rescore(kind, value, reference, selected, mechanics)
            metrics = scored.get("metrics")
            row.update(status=scored["status"], predicates=scored["predicates"],
                       passed=None if scored["status"] == "not_applicable" else scored["status"] in ("pass", "waived"))
            if scored["status"] == "not_applicable":
                row["evaluated"] = False
            if scored.get("reason"):
                row["reason"] = scored["reason"]
            if scored.get("error"):
                row["error"] = scored["error"]
        else:
            with kinds.policy(config_path, selected):
                reference = named if named is not None else kind.reference(selected, declaration.load(config_path))
                metrics = kind.score(value, reference, check=selected, mechanics=mechanics)
                status, row["predicates"] = kind.judge(metrics, check)
            row["passed"] = status in ("pass", "waived")
            if status == "waived":
                # Passed only under declared waivers, listed on its predicates.
                row["status"] = status
        if judged:
            # The verdict's measured values must be its record's, or the
            # exit is a failed run that links no record.
            if not eval_judged(report / "report.json", name, check["rules"], metrics or {}):
                row.clear()
                row.update(unlinked)
                raise ValueError(f"check exited {returncode}")
            eval_record(check, arguments, report)
        if judged and row["passed"] is not False:
            for key in ("status", "evaluated", "reason"):
                row.pop(key, None)
            row.update(passed=False, error=f"chopinfer eval run failed {name}'s declared predicates, "
                                           "which this gate's judge does not")
        if row["passed"] is True:
            context[name] = str(path)
            context.setdefault("check_raw", CheckRaw())[name] = str(raw)
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
        row["error"] = str(error)
    if row.get("receipt") and row["passed"] is False and "error" not in row:
        row["error"] = "threshold predicate failed"
    row["seconds"] = round(time.monotonic() - step_started, 3)
    if row.get("status") == "not_applicable":
        print(f"{name}: NOT APPLICABLE: {row['reason']}", flush=True)
    else:
        print(f"{name}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    return row


def run(config, config_path, profile, url, model, inputs, receipts, mechanics=False, *, model_root=None):
    validate(config, external=model_root is not None)
    mechanics = mechanics or config.get("qualification", {}).get("mechanics_only", False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = stamp + "-" + uuid.uuid4().hex[:8]
    directory = receipts / f"{run_id}-release-{config['model']}-{profile}"
    directory.mkdir(parents=True, exist_ok=False)
    source_config = str(config_path)
    config_path = directory / "policy.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    raw = directory / "checks"
    raw.mkdir()
    context = {**inputs, "url": url, "model": model or "", "run_id": run_id,
               "raw": str(raw), "run": str(directory), "check_raw": CheckRaw()}
    policy = config["profiles"][profile]
    result = {"schema": "chopinfer-release-receipt/v1", "at": stamp,
              "model": config["model"], "profile": profile, "mechanics_only": mechanics,
              "config_sha256": digest(config_path), "config_source": source_config, "config": config,
              "variant_source": None,
              "inputs": inputs, "checks": [], "limitations": config.get("limitations", []),
              "passed": False, "qualified": False}
    started = time.monotonic()
    # Each check is bounded by its own timeout_seconds; a cohort has no overall deadline.
    deadline = math.inf
    try:
        from refs import verify_config, variant_source
        # An external variant's overlay, lock and checkpoint lock, by digest.
        result["variant_source"] = variant_source(config)
        verify_config(config, model_root, policy_only=True)
        result["revision"] = source_revision(ROOT)
        if model_root is not None:
            pin = Path(model_root) / "sdk-pin.json"
            result["provenance"] = {"model_revision": source_revision(model_root),
                                    "sdk_revision": result["revision"],
                                    "sdk_pin": json.loads(pin.read_text()) if pin.is_file() else None}
        result["server"] = Client(url, timeout=10, retries=0).identity(model)
        require_qualified_model_options(result["server"])
        context["model"] = result["server"]["model"]
        if any(check["module"] == "paired_teacher" and check["id"] in policy["checks"] for check in config["checks"]):
            context["_qualified_server"] = result["server"]
        # Resolve every prerequisite before spending time on live checks.
        prerequisites(config, policy["checks"], profile, context, result, model_root)
        for name in policy["checks"]:
            check = next(check for check in config["checks"] if check["id"] == name)
            row = run_check(check, result["checks"], profile, context, directory, config_path, deadline,
                            mechanics, model_root)
            # Quick runs every check so each verdict is visible; full stops,
            # since its later checks build on earlier receipts and take hours.
            if row["passed"] is False and profile != "quick":
                break
        unjudged = ("informational", "not_applicable")
        result["passed"] = (len(result["checks"]) == len(policy["checks"])
                            and all(row["passed"] or row.get("status") in unjudged for row in result["checks"])
                            and any(row["passed"] for row in result["checks"]))
        result["qualified"] = result["passed"] and profile == "full" and not mechanics and not any(row.get("status") in unjudged for row in result["checks"])
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    result["not_run"] = [name for name in policy["checks"] if name not in {r["id"] for r in result["checks"]}]
    result["seconds"] = round(time.monotonic() - started, 3)
    summary = directory / "summary.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [f"# Release gate: {config['model']} / {profile}", "",
             f"Passed: {result['passed']}. Qualified: {result['qualified']}. Mechanics only: {mechanics}.", "",
             "| Check | Result | Seconds |", "|---|---|---:|"]
    labels = {"informational": "INFORMATIONAL (not evaluated)", "not_applicable": "NOT APPLICABLE",
              "waived": "WAIVED (declared)",
              "no_accepted_arm": "no accepted arm (declared)"}
    lines += [f"| {r['id']} | {labels.get(r.get('status')) or ('PASS' if r['passed'] else 'FAIL')} | {r['seconds']} |" for r in result["checks"]]
    if inapplicable := [f"{r['id']}: not applicable: {r['reason']}" for r in result["checks"]
                        if r.get("status") == "not_applicable"]:
        lines += ["", *inapplicable]
    lines += [f"| {name} | NOT RUN | |" for name in result["not_run"]]
    if missing := [f"{r['id']}: no accepted arm (declared): {r['reason']} Next arm: {r['accepted_arm_rule']}"
                   for r in result["checks"] if r.get("status") == "no_accepted_arm"]:
        lines += ["", *missing]
    lines += ["", *[f"{r['id']}: {r['error']}" for r in result["checks"] if r.get("error")], ""]
    if result.get("error"):
        lines += ["", result["error"]]
    lines += ["", *config.get("limitations", []), "", "See summary.json for thresholds, sources, logs and evidence hashes."]
    summary.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"receipt": str(summary), "passed": result["passed"], "qualified": result["qualified"]}))
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", help="model whose models/<m>/qual.json to read, e.g. mimo26 or qwen38")
    selection.add_argument("--config", type=Path, help="external gate declaration")
    parser.add_argument("--model-root", type=Path, help="base for external declared input paths")
    parser.add_argument("--variant-dir", type=Path,
                        help=f"root of external variants, MODEL/VARIANT.json (default ${declaration.VARIANTS_ENV})")
    parser.add_argument("--served-model", help="API model alias; defaults to /v1/models")
    parser.add_argument("--url")
    parser.add_argument("--list-inputs", action="store_true", help="Resolve configured inputs without HTTP or receipts")
    parser.add_argument("--plan", action="store_true", help="Describe the cohort without HTTP, execution or prerequisite preparation")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--inputs", type=Path, help="JSON object of external reference URLs and evidence paths")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--mechanics-only", "--self-test", action="store_true", help="CPU fixture self-test; never qualifies a model")
    parser.add_argument("--timeout", type=float, default=10, help="compatibility with CPU fixture driver")
    args = parser.parse_args()
    if bool(args.config) != bool(args.model_root):
        parser.error("--config and --model-root must be supplied together")
    declaration.select_variant_root(args.variant_dir)
    model_root = args.model_root.resolve() if args.model_root else None
    if args.config:
        config_path = args.config.resolve()
    else:
        name = "mechanics" if args.mechanics_only else args.model
        # A MODEL or MODEL.VARIANT declaration name: no path separators,
        # and nothing an option could be mistaken for.
        try:
            declaration.split(name or "")
            if not name[0].isalnum():
                raise ValueError("--model must name a model declaration, as MODEL or MODEL.VARIANT")
        except ValueError as error:
            parser.error(str(error))
    try:
        if not args.config:
            config_path = declaration.named(name)
        declaration.require_rooted(config_path, model_root or ROOT)
        config = declaration.load(config_path)
        validate(config, external=model_root is not None)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.plan:
        print(json.dumps(describe(config, args.profile), indent=2))
        return 0
    if args.list_inputs:
        from refs import verify_config
        try:
            verify_config(config, model_root, optional=True)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        os.environ["CHOPINFER_QUAL_CONFIG"] = str(config_path.resolve())
        print(json.dumps({"paths": {key: str(path) if (path := input_path(key, optional=True, config=config, root=model_root)) else None
                                   for key in config["inputs"]},
                          "qualification": config["qualification"],
                          "profiles": config["profiles"], "checks": config["checks"]}, indent=2))
        return 0
    if not args.url:
        parser.error("--url is required unless --list-inputs is used")
    from refs import verify_config
    try:
        verify_config(config, model_root, policy_only=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    inputs = json.loads(args.inputs.read_text()) if args.inputs else {}
    receipts = args.receipts or (Path(os.environ["CHOPINFER_QUAL_RECEIPTS"]) if os.environ.get("CHOPINFER_QUAL_RECEIPTS")
                                 else input_path("receipts", config=config, root=model_root))
    return run(config, config_path, args.profile, args.url, args.served_model,
               inputs, receipts.resolve(), args.mechanics_only, model_root=model_root)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.exit(worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4:]))
    sys.exit(main())
