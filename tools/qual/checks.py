"""Run one declared check, or re-judge stored receipts, through the check kinds.

    python3 tools/qual/checks.py list [--model M] [--json]
    python3 tools/qual/checks.py check ID --model M (--url URL | --offline) [--cohort quick|full]
        [--served-model NAME] [--inputs FILE] [--with ID=RECEIPT]... [--with-raw ID=DIR]...
        [--receipts DIR]
    python3 tools/qual/checks.py rescore --model M [--limits current|recorded]
        [--check ID] [--json FILE] [--require-recorded] PATH...

Each command takes `--variant-dir DIR` (default $CHOPINFER_QUAL_VARIANTS)
to read `--model MODEL.VARIANT` as an external variant, DIR/MODEL/VARIANT.json.

`check` runs the check as release_gate.py runs it in a cohort: the same
arguments, receipt, metrics and predicates (kinds.py). It writes the run
under the receipts directory, with the policy, the check's receipt and
`check.json`, its verdict. `--offline` runs a check whose module judges
recorded, server-bound inputs without a server (RUNNER_CHECK_ARM "offline");
its run names no live server.

`rescore` re-judges stored receipts offline, without a server. A PATH is a
release gate run (its directory or summary.json) or a check receipt. Each
receipt is scored by its kind with this code, recomputing what the kind
derives from the receipt and its pinned reference, and judged by the
current declaration's predicates and limits, or with `--limits recorded`
by the policy the gate run recorded. Each verdict is shown next to the
recorded one, with the metric fields that differ. Exit status: 0 when every
verdict passes, 1 when one fails, 3 when one has insufficient evidence and
none fails, 2 for a usage error; with --require-recorded, 1 when any verdict
or metric differs from the recorded one.
"""

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from common import require_qualified_model_options, Client, input_path
import declaration
import kinds
import release_gate

PASSING = ("pass", "informational", "not_applicable", "waived")


def load(model):
    """A model's (or the mechanics fixture's) gate config and its file."""
    declaration.split(model or "")
    path = declaration.named(model)
    config = declaration.load(path)
    release_gate.validate(config)
    return config, path


def list_command(args):
    rows = kinds.describe()
    if args.model:
        config, _ = load(args.model)
        rows = {"kinds": rows, "checks": [{"id": check["id"], "module": check["module"],
                                           "kind": kinds.of(check["module"]).name,
                                           "informational": "informational_reason" in check}
                                          for check in config["checks"]]}
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows["kinds"] if args.model else rows:
        print(f"{row['kind']}{' (shared)' if row['shared'] else ''}: modules {', '.join(row['modules']) or 'any other'}")
        for key in ("inputs", "receipt", "verdict"):
            print(f"  {key}: {row[key]}")
        print(f"  metrics: {', '.join(row['metrics'])}")
    for row in rows["checks"] if args.model else []:
        print(f"{row['id']}: {row['kind']} ({row['module']}){' informational' if row['informational'] else ''}")
    return 0


def check_command(args):
    config, config_path = load(args.model)
    checks = {check["id"]: check for check in config["checks"]}
    if args.id not in checks:
        raise ValueError(f"{args.model} declares no check {args.id}; it declares {sorted(checks)}")
    check = checks[args.id]
    offline = getattr(args, "offline", False)
    if offline and offline_module(check["module"]) is not True:
        raise ValueError(f"{args.id} needs a live server: its module {check['module']} does not judge offline")
    if not offline and not args.url:
        raise ValueError("check needs --url, or --offline for a check judged without a server")
    inputs = json.loads(args.inputs.read_text()) if args.inputs else {}
    earlier = {}
    for value in args.with_receipt:
        name, separator, path = value.partition("=")
        if not separator or name not in checks or not Path(path).is_file():
            raise ValueError(f"--with needs ID=RECEIPT naming a declared check and a file: {value}")
        earlier[name] = str(Path(path).resolve())
    earlier_raw = release_gate.CheckRaw()
    for value in getattr(args, "with_raw", []):
        name, separator, path = value.partition("=")
        if not separator or name not in earlier or not Path(path).is_dir():
            raise ValueError(f"--with-raw needs ID=DIR naming a check given with --with and a directory: {value}")
        earlier_raw[name] = str(Path(path).resolve())
    receipts = args.receipts or (Path(os.environ["CHOPINFER_QUAL_RECEIPTS"])
                                 if os.environ.get("CHOPINFER_QUAL_RECEIPTS") else input_path("receipts", config=config))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = stamp + "-" + uuid.uuid4().hex[:8]
    directory = receipts.resolve() / f"{run_id}-check-{config['model']}-{args.id}"
    (directory / "checks").mkdir(parents=True)
    policy = directory / "policy.json"
    policy.write_text(json.dumps(config, indent=2) + "\n")
    mechanics = config.get("qualification", {}).get("mechanics_only", False)
    context = {**inputs, **earlier, "url": args.url or "", "model": args.served_model or "", "run_id": run_id,
               "raw": str(directory / "checks"), "run": str(directory), "check_raw": earlier_raw}
    result = {"schema": "chopinfer-check-run/v1", "at": stamp, "model": config["model"], "check": args.id,
              "kind": kinds.of(check["module"]).name, "cohort": args.cohort, "mechanics_only": mechanics,
              "config_sha256": release_gate.digest(policy), "config_source": str(config_path), "inputs": inputs,
              "earlier": earlier, "earlier_raw": dict(earlier_raw), "variant_source": None, "status": "fail"}
    rows = []
    try:
        from refs import variant_source
        result["variant_source"] = variant_source(config)
        result["revision"] = release_gate.source_revision(release_gate.ROOT)
        if offline:
            result["server"] = None
        else:
            result["server"] = Client(args.url, timeout=10, retries=0).identity(args.served_model)
            require_qualified_model_options(result["server"])
            context["model"] = result["server"]["model"]
        if kinds.of(check["module"]).name == "paired_teacher":
            context["_qualified_server"] = result["server"]
        release_gate.prerequisites({**config, "checks": [check]}, [args.id], args.cohort, context, result)
        row = release_gate.run_check(check, rows, args.cohort, context, directory, policy,
                                     time.monotonic() + check["timeout_seconds"], mechanics, None)
        result["status"] = row.get("status") or ("pass" if row["passed"] else "fail")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result["error"] = str(error)
    result["row"] = rows[0] if rows else None
    path = directory / "check.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(path), "check": args.id, "status": result["status"]}))
    return 0 if result["status"] in PASSING else 1


def offline_module(module):
    """Whether a check module judges without a server (RUNNER_CHECK_ARM
    "offline"), read from its source without importing it."""
    import module_metadata
    return module_metadata.module_constants(module).get("RUNNER_CHECK_ARM") == ("offline",)


def gate_runs(paths):
    """Each PATH as (file, gate summary), the summary None for a check receipt."""
    for path in map(Path, paths):
        if path.is_dir():
            path = path / "summary.json"
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"{path}: not a JSON object")
        if value.get("schema") == "chopinfer-release-receipt/v1":
            yield path, value
        elif value.get("schema") == "chopinfer-qualification/v1":
            yield path, None
        else:
            raise ValueError(f"{path}: neither a release gate summary nor a check receipt")


def stored(summary, row, key="receipt"):
    """A gate row's receipt (or the file its `key` names), where the run is
    now. Its recorded path is under the run directory its log names."""
    recorded = Path(row[key])
    if recorded.is_file():
        return recorded
    return summary.parent / recorded.relative_to(Path(row["log"]).parent)


def reference_of(summary, row):
    """A gate row's reference file, where it is now: the recorded path, or,
    for a file inside the run directory, the same place in the moved run."""
    recorded = Path(row.get("reference_file") or "")
    run = Path(row["log"]).parent if row.get("log") else None
    if row.get("reference_file") and not recorded.is_file() and run is not None and recorded.is_relative_to(run):
        return summary.parent / recorded.relative_to(run)
    return recorded


def unrecorded(summary, row, check):
    """The receipt a check wrote before it exited nonzero, which its gate
    row does not record: the record in its `chopinfer eval run` report, or
    the one file in the run's checks directory its receipt pattern
    matches."""
    if not row.get("exit_code"):
        return None
    if row.get("eval_report"):
        try:
            record = stored(summary, row, "eval_report").parent / "records" / f"{row['id']}.json"
        except (KeyError, ValueError):
            return None
        return record if record.is_file() else None
    if "receipt" not in check:
        return None
    matches = sorted((summary.parent / "checks").glob(check["receipt"]))
    return matches[0] if len(matches) == 1 else None


def judged(summary, row, metrics=None):
    """Whether a gate row's nonzero exit is eval's exit 1 for a completed
    verdict that failed a declared predicate (release_gate.eval_judged, by
    the rules the row records and, when given, the `metrics` its record
    scores now), its linked receipt that verdict's record, rather than a
    failed run. A row the gate failed for another reason, such as its own
    judge passing what eval failed, is not."""
    if (row.get("exit_code") != 1 or not row.get("eval_report") or not row.get("receipt")
            or row.get("passed") is not False or row.get("error") != "threshold predicate failed"
            or not isinstance(row.get("rules"), list)):
        return False
    try:
        report = stored(summary, row, "eval_report")
        receipt = stored(summary, row)
    except (KeyError, ValueError):
        return False
    return (receipt == report.parent / "records" / f"{row['id']}.json"
            and release_gate.eval_judged(report, row["id"], row["rules"], metrics))


def rescore_one(value, check, config, config_path, mechanics, revision=None, named=None):
    """The rescored verdict of one receipt (kinds.rescore): status, metrics,
    predicates, the fields read as recorded and the error, against the
    references in the working tree or at a source `revision`, or against
    `named`, the record a check's reference_file named. The receipt's own
    metric values are `recorded_metrics`."""
    kind = kinds.of(check["module"])
    result = {"kind": kind.name,
              "recorded_metrics": kind.recorded(value, check) if isinstance(value, dict) else None}
    if named is not None:
        try:
            kinds.admit_pair(value, named, check, mechanics)
        except ValueError as error:
            return {**result, "status": "fail", "error": str(error)}
    with kinds.policy(config_path, check):
        try:
            reference = named if named is not None else kind.reference(check, config, revision)
        except (OSError, ValueError, KeyError) as error:
            return {**result, "status": "insufficient", "error": f"reference: {error}"}
        return {**result, **kinds.rescore(kind, value, reference, check, mechanics)}


def input_verdict(row, check, value, named, summary):
    """None when the check's input records hold its declared input_rules as
    the live preflight judges them (release_gate.judge_input), else the
    rescored status and error: fail for a record the rules refuse, so a
    stored run cannot pass rules its preflight would refuse now, and
    insufficient for an input the run did not record, or that changed since
    it did. An input the check reads as its json_file or reference_file is
    the receipt `value` or the reference record `named`; any other is the
    run's recorded input file (including an explicitly recorded earlier
    receipt), which must keep the digest it recorded."""
    rules = check.get("input_rules") or {}
    if not rules:
        return None
    if not isinstance(summary, dict):
        return {"status": "insufficient", "error": "input rules need the gate run's summary"}
    inputs, earlier, digests = (summary.get(key, {}) for key in ("inputs", "earlier", "input_sha256"))
    if any(not isinstance(values, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                  for key, value in values.items())
           for values in (inputs, earlier, digests)):
        return {"status": "insufficient", "error": "recorded input, earlier and digest maps must hold strings"}
    conflicts = sorted(key for key in inputs.keys() & earlier.keys() if inputs[key] != earlier[key])
    if conflicts:
        return {"status": "insufficient", "error": "conflicting recorded input paths: " + ", ".join(conflicts)}
    inputs = {**inputs, **earlier}
    context = dict(inputs)
    if (summary.get("server") or {}).get("model"):
        context["model"] = summary["server"]["model"]
    for key, key_rules in rules.items():
        if check.get("json_file") == f"{{{key}}}":
            record = value
        elif check.get("reference_file") == f"{{{key}}}":
            record = named
        else:
            path = Path(inputs.get(key) or "")
            if not inputs.get(key) or not path.is_file() or digests.get(key) is None \
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digests[key]:
                return {"status": "insufficient",
                        "error": f"{key}: the input the gate judged is missing, unrecorded or changed"}
            record = json.loads(path.read_text())
        try:
            release_gate.judge_input(key, record, key_rules, context)
        except KeyError as error:
            return {"status": "insufficient", "error": f"{key}: an input rule names {error}, which the run did "
                                                       "not record"}
        except ValueError as error:
            return {"status": "fail", "error": str(error)}
    return None


def recorded_status(row):
    if row.get("status") in ("informational", "not_applicable", "waived"):
        return row["status"]
    return "pass" if row.get("passed") is True else "fail"


def rescore_row(path, row, check, config, config_path, mechanics, revision, summary=None):
    """One gate row rescored. A worker that exited nonzero failed the check
    whatever its receipt holds, linked on the row or left in the run; that
    receipt's own verdict is `receipt_status`. Eval's exit 1 for a judged
    verdict (judged) is not such a failure: its receipt is judged again as
    one a worker wrote before exiting 0. The check's input_rules are judged
    first (input_verdict), against the run's `summary`."""
    kind = kinds.of(check["module"]).name
    if check.get("informational_reason"):
        return {"kind": kind, "status": "informational"}
    named = None
    if "reference_file" in check and row.get("receipt"):
        # The record the gate scored the receipt against, as it read it.
        reference = reference_of(path, row)
        if not row.get("reference_file") or not reference.is_file() \
                or hashlib.sha256(reference.read_bytes()).hexdigest() != row.get("reference_sha256"):
            return {"kind": kind, "status": "insufficient",
                    "error": "the reference file the gate scored against is missing or changed"}
        named = json.loads(reference.read_text())
    if row.get("receipt"):
        receipt = stored(path, row)
        if hashlib.sha256(receipt.read_bytes()).hexdigest() != row.get("sha256"):
            result = {"kind": kind, "receipt": str(receipt), "status": "insufficient",
                      "error": "receipt changed after the gate ran"}
        else:
            value = json.loads(receipt.read_text())
            refused = input_verdict(row, check, value, named, summary)
            result = {"receipt": str(receipt), **({"kind": kind, **refused} if refused else rescore_one(
                value, check, config, config_path, mechanics, revision, named))}
    elif row.get("exit_code") and (receipt := unrecorded(path, row, check)) is not None:
        value = json.loads(receipt.read_text())
        refused = input_verdict(row, check, value, named, summary)
        result = {"receipt": str(receipt), "unrecorded": True, **({"kind": kind, **refused} if refused else
                                                                  rescore_one(value, check, config, config_path,
                                                                              mechanics, revision, named))}
    else:
        # The check failed before writing its receipt; one recorded as
        # informational was never run and has no evidence to judge.
        status = "insufficient" if recorded_status(row) == "informational" else "fail"
        return {"kind": kind, "status": status, "error": row.get("error") or "no receipt"}
    if row.get("exit_code") and not judged(path, row, result.get("metrics")):
        result.update(receipt_status=result["status"], receipt_error=result.get("error"), status="fail",
                      error=f"check exited {row['exit_code']}")
    return result


def rescore_gate(path, summary, model, limits):
    """A gate run rescored against the current declaration of `model`
    (`limits` current) or the run's recorded policy and source revision:
    one row per check the declaration's cohort selects, in its order, then
    the run's other rows, which do not count toward the gate. A selected
    check without a row is `not_run` when the run recorded it so, else
    `insufficient`."""
    if limits == "recorded":
        config_path = path.parent / "policy.json"
        config = declaration.load(config_path)
        renamed, revision = {}, summary.get("revision")
    else:
        config, config_path = load(model)
        renamed, revision = declaration.renamed(model), None
    checks = {check["id"]: check for check in config["checks"]}
    profile = summary.get("profile")
    if profile not in config["profiles"]:
        raise ValueError(f"{path}: the declaration has no cohort {profile}")
    selected = config["profiles"][profile]["checks"]
    recorded = {renamed.get(row["id"], row["id"]): row for row in summary["checks"]}
    not_run = {renamed.get(name, name) for name in summary.get("not_run", [])}
    mechanics = summary.get("mechanics_only", False)
    rows = []
    for name in [*selected, *(name for name in recorded if name not in selected)]:
        row = recorded.get(name)
        entry = {"source": str(path), "id": name, "recorded_id": row["id"] if row else None,
                 "recorded": recorded_status(row) if row else "not_run" if name in not_run else "absent",
                 "selected": name in selected}
        if name not in checks:
            rows.append({**entry, "status": "insufficient", "error": "the declaration has no such check"})
        elif row is None:
            rows.append({**entry, "kind": kinds.of(checks[name]["module"]).name,
                         "status": "not_run" if name in not_run else "insufficient",
                         "error": "the run did not reach this check" if name in not_run
                         else "the run has no row for this check"})
        else:
            rows.append({**entry, **rescore_row(path, row, {**checks[name], "_cohort": profile}, config, config_path,
                                                mechanics, revision, summary)})
    return rows


def rescore_receipt(path, value, model, limits, check_id=None):
    """A check receipt outside a gate run, rescored against the current
    declaration: the check its id names, else the check whose record
    `chopinfer eval run` wrote at `records/<ID>.json` (a teacher capture's
    id is its receipt name, such as long-context), else the one whose
    receipt pattern its file name matches. `check_id` (--check), the check a
    gate row recorded the receipt for, decides between checks whose receipt
    patterns both match (a module's receipt names neither, as evals compare's
    `evals-compare` does), or names the check of a capture nothing else
    identifies. It must be one of the checks the receipt's own identity
    names, in that order, when it names any; a check with a receipt pattern
    must also have its file name match."""
    if limits == "recorded":
        raise ValueError(f"{path}: --limits recorded needs a release gate run, not a single receipt")
    config, config_path = load(model)
    renamed = declaration.renamed(model)
    checks = {check["id"]: check for check in config["checks"]}
    name = renamed.get(value.get("check"), value.get("check"))
    if name not in checks and path.parent.name == "records":
        name = renamed.get(path.stem, path.stem)
    known = [name] if name in checks else [check["id"] for check in config["checks"]
                                           if fnmatch.fnmatch(path.name, check.get("receipt", ""))]
    if check_id is not None:
        wanted = renamed.get(check_id, check_id)
        if wanted not in checks:
            raise ValueError(f"{path}: --check {check_id}: the declaration has no such check")
        if known and wanted not in known:
            raise ValueError(f"{path}: the receipt of {' or '.join(known)}, not {wanted}")
        if "receipt" in checks[wanted] and wanted not in known:
            raise ValueError(f"{path}: not a receipt {wanted}'s pattern {checks[wanted]['receipt']} names")
        known = [wanted]
    if len(known) != 1:
        raise ValueError(f"{path}: names no declared check and matches {len(known)} receipt patterns")
    name = known[0]
    entry = {"source": str(path), "id": name, "recorded_id": value.get("check"), "recorded": None,
             "receipt": str(path)}
    if checks[name].get("informational_reason"):
        return {**entry, "kind": kinds.of(checks[name]["module"]).name, "status": "informational"}
    return {**entry, **rescore_one(value, checks[name], config, config_path, value.get("mechanics_only") is True)}


def differing(row):
    """The metric fields whose rescored value differs from the receipt's."""
    old, new = row.get("recorded_metrics"), row.get("metrics")
    if old is None or new is None:
        return []
    return sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))


def gate_passed(statuses):
    """Whether a stored gate's selected verdicts pass: none fails, and at
    least one passes or is waived (an all-informational or not-applicable
    gate judged nothing, so it does not pass)."""
    return all(status in PASSING for status in statuses) and any(status in ("pass", "waived") for status in statuses)


def rescore_command(args):
    config = load(args.model)[0]
    model = config["model"]
    rows, gates = [], []
    for path, summary in gate_runs(args.paths):
        if summary is None:
            rows.append(rescore_receipt(path, json.loads(path.read_text()), args.model, args.limits, args.check))
            continue
        if args.check is not None:
            raise ValueError(f"{path}: --check names the check of a single receipt, not of a gate run")
        ran = (summary.get("model"), (summary.get("config") or {}).get("variant"))
        if ran != (model, config.get("variant")):
            name = ran[0] if ran[1] is None else f"{ran[0]}.{ran[1]}"
            raise ValueError(f"{path}: a gate run of {name}, not {args.model}")
        gate = rescore_gate(path, summary, args.model, args.limits)
        statuses = [row["status"] for row in gate if row["selected"]]
        passed = gate_passed(statuses)
        gates.append({"source": str(path), "recorded_passed": summary.get("passed"), "passed": passed})
        rows += gate
    for row in rows:
        row["differing_metrics"] = differing(row)
    same = [row for row in rows if row["recorded"] in (None, row["status"]) and not row["differing_metrics"]]
    report = {"schema": "chopinfer-rescore/v1", "model": args.model, "limits": args.limits,
              "gates": gates, "checks": rows}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("| Receipt | Check | Kind | Recorded | Rescored | Metrics | As recorded | Error |",
          "|---|---|---|---|---|---|---|---|", sep="\n")
    for row in rows:
        metrics = ("differ: " + ", ".join(row["differing_metrics"]) if row["differing_metrics"]
                   else "equal" if row.get("metrics") is not None and row.get("recorded_metrics") is not None else "")
        print(f"| {row.get('receipt') or row['source']} | {row['id']}{'' if row.get('selected', True) else ' (not selected)'} | "
              f"{row.get('kind', '')} | {row['recorded'] or ''} | {row['status']} | {metrics} | "
              f"{', '.join(row.get('recorded_fields') or [])} | {row.get('error', '')} |")
    for gate in gates:
        print(f"gate {gate['source']}: recorded passed {gate['recorded_passed']}, rescored passed {gate['passed']}")
    print(f"{len(rows)} verdicts rescored with {args.limits} limits; {len(same)} equal the recorded verdict "
          f"and metrics; {sum(gate['passed'] == gate['recorded_passed'] for gate in gates)} of {len(gates)} "
          "gate results equal")
    if args.require_recorded:
        return 0 if len(same) == len(rows) and all(g["passed"] == g["recorded_passed"] for g in gates) else 1
    statuses = {row["status"] for row in rows if row.get("selected", True)}
    return 1 if statuses & {"fail", "not_run"} else 3 if "insufficient" in statuses else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="the check kinds and, with --model, the kind of each declared check")
    listing.add_argument("--model")
    listing.add_argument("--json", action="store_true")
    check = commands.add_parser("check", help="run one declared check against a live server")
    check.add_argument("id")
    check.add_argument("--model", required=True)
    where = check.add_mutually_exclusive_group(required=True)
    where.add_argument("--url")
    where.add_argument("--offline", action="store_true",
                       help="judge a check that reads only recorded, server-bound inputs, without a server")
    check.add_argument("--served-model", help="API model alias; defaults to /v1/models")
    check.add_argument("--cohort", choices=("quick", "full"), default="quick",
                       help="the cohort whose extra arguments the check takes")
    check.add_argument("--inputs", type=Path, help="JSON object of external reference URLs and evidence paths")
    check.add_argument("--with", dest="with_receipt", action="append", default=[], metavar="ID=RECEIPT",
                       help="an earlier check's receipt the check's arguments name")
    check.add_argument("--with-raw", dest="with_raw", action="append", default=[], metavar="ID=DIR",
                       help="the raw directory of an earlier check given with --with, which "
                            "{check_raw[ID]} in the check's arguments names")
    check.add_argument("--receipts", type=Path)
    rescore = commands.add_parser("rescore", help="re-judge stored receipts offline")
    rescore.add_argument("paths", nargs="+", metavar="PATH")
    rescore.add_argument("--model", required=True)
    rescore.add_argument("--limits", choices=("current", "recorded"), default="current",
                         help="judge by the current declaration (default) or by a gate run's recorded policy")
    rescore.add_argument("--check", metavar="ID",
                         help="the declared check a single receipt is judged as (a gate row's check)")
    rescore.add_argument("--json", type=Path, help="write the rescore report here")
    rescore.add_argument("--require-recorded", action="store_true",
                         help="exit 1 unless every verdict and metric equals the recorded one")
    for command in (listing, check, rescore):
        command.add_argument("--variant-dir", type=Path,
                             help=f"root of external variants, MODEL/VARIANT.json (default ${declaration.VARIANTS_ENV})")
    args = parser.parse_args()
    declaration.select_variant_root(args.variant_dir)
    try:
        command = {"list": list_command, "check": check_command, "rescore": rescore_command}[args.command]
        return command(args)
    except (OSError, ValueError, declaration.DeclarationError) as error:
        print(f"{parser.prog} {args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
