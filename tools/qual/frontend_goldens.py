"""Compare declared frontend captures and exact token-ID goldens.

Offline, CAPTURE is a directory a capture script wrote for the server. With
--url instead, this captures the served frontend itself: each row of the
declared chat-renders.json (messages, tools, chat_template_kwargs) and each
text of the declared tokenizer goldens goes to the server's /tokenize, in
order, and the served token IDs are written to --capture-out and compared.
"""

import argparse
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

from common import template_control, ROOT, resolve_input, qualification_config

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from refs import load_lock, read_member
from kinds import predicate
FILES = ('needles.json', 'api-samples.json', 'chat-renders.json', 'tokenizer-golden.json')
RUNNER_RECORDS = {'capture': ('spec', 'frontend_goldens.py', [
    '--url', '{url}', '--model', '{name}', '--capture-out', '{run_dir}', '--server-log', '{server_log}'])}
RUNNER_DIRECTORY_INPUTS = ['capture']
CAPTURE_SCHEMA = 'chopinfer.frontend-capture.v1'


def load(directory, name, *, config=None, reference=False):
    path = Path(directory) / name
    if not path.exists():
        return None
    if reference and config is None:
        config = qualification_config()
    data = reference_bytes(config, "frontend_capture", path) if config is not None else path.read_bytes()
    return json.loads(data)


def reference_bytes(config, key, path):
    """Consume adopted bytes once; undeclared overrides remain diagnostics without a lock."""
    expected = resolve_input(key, config=config)
    expected = expected / path.name if key == "frontend_capture" else expected
    if load_lock(config) is not None and path.resolve() != expected.resolve():
        raise ValueError(f"lock entry {key}: undeclared frontend reference")
    return read_member(config, key, expected if path.resolve() == expected.resolve() else path)


def wire_message(message):
    """A captured message as an OpenAI client sends it: tool-call arguments as their JSON text."""
    def text(arguments):
        return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    calls = message.get("tool_calls")
    if not calls:
        return message
    return {**message, "tool_calls": [{**call, "function": {**call["function"],
                                                            "arguments": text(call["function"].get("arguments"))}}
                                      for call in calls]}


def tokenize(client, body):
    """The served token IDs of one /tokenize request, or the refusal."""
    try:
        response, _ = client.request("/tokenize", body)
    except RuntimeError as error:
        status, _, detail = str(error).removeprefix("HTTP ").partition(": ")
        try:
            detail = json.loads(detail)
        except ValueError:
            pass
        return {"status": int(status) if status.isdigit() else None,
                "error": detail if isinstance(detail, dict) else {"error": {"message": str(error)}}}
    return {"status": 200, "token_ids": response.get("tokens")}


def capture_bodies(model, renders, texts):
    """The existing ordered frontend workload, including all template options."""
    chats = []
    for row in renders:
        body = {"model": model, "messages": [wire_message(m) for m in row["messages"]],
                "add_generation_prompt": True}
        if row.get("tools"):
            body["tools"] = row["tools"]
        if row.get("chat_template_kwargs"):
            body.update(template_control(row["chat_template_kwargs"]))
        chats.append(body)
    tokens = [{"model": model, "prompt": row["text"], "add_special_tokens": False} for row in texts]
    return chats, tokens


def body_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def extension(config, renders, model):
    """An owning model can supply the remaining declared frontend requests."""
    import declaration
    name = config.get("model", "")
    if not declaration.plain(name):
        raise ValueError("invalid frontend model")
    path = ROOT / "models" / name / "frontend_capture.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("owned_frontend_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare(config, renders, model)


def extra_requests(plan):
    return {group: [{"name": row["name"], "endpoint": row["endpoint"], "sha256": body_hash(row["body"])}
                    for row in plan[group]] for group in ("api", "needles")}


def observation(client, row):
    """Keep successful bodies and expected HTTP refusals with their responder proof."""
    try:
        body, _ = client.request(row["endpoint"], row["body"], stream=bool(row["body"].get("stream")))
    except RuntimeError as error:
        status, separator, detail = str(error).removeprefix("HTTP ").partition(": ")
        if not separator or not status.isdigit():
            raise
        return {"status": int(status), "body": json.loads(detail)}
    if row["body"].get("stream"):
        return [event for _, event in body]
    return {"status": 200, "body": body}


def serve_capture(url, model, renders, texts, output, timeout, *, server_log=None, config=None, references=None,
                  require_bound=False, plan=None):
    """Capture the unchanged workload, optionally bound to its live process."""
    from common import Client, bind_checkpoint, confirm_live
    from tools.model_sdk import diagnostic as wire
    if output.exists():
        raise ValueError("frontend capture output already exists")
    bound = server_log is not None or require_bound or os.environ.get("CHOPINFER_FRONTEND_REQUIRE_BOUND") == "1"
    client = Client(url, timeout, retries=1)
    server = client.identity(model)
    if bound:
        bind_checkpoint(server, server_log, url, config)
    bodies = capture_bodies(server["model"], plan["renders"] if plan else renders, texts)
    chats = [{key: row[key] for key in ("id", "name") if key in row} | tokenize(client, body)
             for row, body in zip(renders, bodies[0])]
    tokens = [{"text": row["text"]} | tokenize(client, body) for row, body in zip(texts, bodies[1])]
    outputs = {"chat-renders.json": chats, "tokenizer-golden.json": tokens}
    if plan is not None:
        outputs["api-samples.json"] = {row["name"]: observation(client, row) for row in plan["api"]}
        outputs["needles.json"] = [{"name": row["name"], "total_context": row["total_context"],
                                    "expected": row["expected"], "response": observation(client, row)["body"]}
                                   for row in plan["needles"]]
    metadata = {"server": server}
    if bound:
        confirm_live(client, server)
        from release_preview import source
        metadata.update(schema=CAPTURE_SCHEMA, producer_source=source(), references=references,
                        request_sha256=[[body_hash(body) for body in group] for group in bodies],
                        responders=sorted([list(identity) for identity in client.responders]))
    output.mkdir(parents=True)
    for name, value in outputs.items():
        wire.write_new(output / name, value)
    if bound:
        metadata["files"] = {name: wire.identity(output / name)["sha256"] for name in outputs}
        if plan is not None:
            metadata["extension"] = {"identity": plan["identity"], "requests": extra_requests(plan)}
    wire.write_new(output / "capture-meta.json", metadata)


def admit_capture(directory, config, renders, texts, references, plan=None):
    """Validate a new producer's saved bytes and identity before comparison.

    Legacy diagnostic directories keep their existing comparison interface.
    They carry no bound producer proof and cannot supply a recovery supplement.
    """
    from common import bound_live, live_identity
    from tools.model_sdk import diagnostic as wire
    path = Path(directory) / "capture-meta.json"
    if not path.exists():
        return None
    value = wire.read_json(path)
    if "schema" not in value:
        return None
    if value.get("schema") != CAPTURE_SCHEMA or not bound_live(value.get("server", {}), config):
        raise ValueError("frontend capture has no bound producer identity")
    server = value["server"]
    live, saved = live_identity(server), server["checkpoint"]
    keys = ("checkpoint_identity", "startup_nonce", "package_identity", "metadata_identity")
    if any(not live.get(key) or live[key] != saved.get(key) for key in keys):
        raise ValueError("frontend live and bound identities differ")
    expected = [[live["checkpoint_identity"], live["startup_nonce"]]]
    if value.get("responders") != expected:
        raise ValueError("frontend responses name another process")
    producer = value.get("producer_source") or {}
    if not producer.get("commit") or type(producer.get("dirty")) is not bool:
        raise ValueError("frontend capture records no tool source")
    bodies = capture_bodies(server["model"], plan["renders"] if plan else renders, texts)
    if value.get("request_sha256") != [[body_hash(body) for body in group] for group in bodies]:
        raise ValueError("frontend request workload differs")
    if value.get("references") != references:
        raise ValueError("frontend capture reference binding differs")
    expected_extension = {"identity": plan["identity"], "requests": extra_requests(plan)} if plan else None
    if value.get("extension") != expected_extension:
        raise ValueError("frontend extra request workload differs")
    names = FILES if plan else ("chat-renders.json", "tokenizer-golden.json")
    files = {name: wire.identity(Path(directory) / name)["sha256"] for name in names}
    if value.get("files") != files:
        raise ValueError("frontend captured bytes changed")
    return {"metadata": str(path), "sha256": wire.identity(path)["sha256"],
            "server": server, "producer_source": producer, "files": files}


def normalized(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())


def needles(ours):
    rows = []
    for row in ours:
        expected = normalized(row["expected"])
        text = (row["response"] or {}).get("choices", [{}])[0].get("text", "")
        rows.append({"name": row["name"], "passed": expected in normalized(text),
                     "text": text[:120]})
    return {"passed": all(row["passed"] for row in rows), "rows": rows}


def shape(sample):
    body = sample.get("body") or {}
    if sample.get("status") != 200 or not isinstance(body, dict) or "choices" not in body:
        return {"status": sample.get("status")}
    choice = body["choices"][0]
    message = choice.get("message") or {}
    return {
        "status": 200,
        "finish_reason": choice.get("finish_reason"),
        "content": bool(message.get("content") or choice.get("text")),
        "reasoning": bool(message.get("reasoning") or message.get("reasoning_content")),
        "tool_calls": [call["function"]["name"] for call in message.get("tool_calls") or []],
    }


def api(ours, theirs):
    rows = []
    for name, sample in ours.items():
        if name not in theirs:
            continue
        if isinstance(sample, list):
            # Streams: compare the chunk count only loosely (both non-empty).
            rows.append({"name": name, "agrees": bool(sample) and theirs[name] == {"status": 200}})
            continue
        a, b = shape(sample), theirs[name]
        agrees = a == b
        if name == "completion-n2-seeded":
            choices = (sample.get("body") or {}).get("choices", [])
            agrees = (agrees and len(choices) == 2 and {choice.get("index") for choice in choices} == {0, 1}
                      and all(shape({"status": 200, "body": {"choices": [choice]}}) == b for choice in choices))
        rows.append({"name": name, "ours": a, "reference": b, "agrees": agrees})
    return {"agrees": all(row["agrees"] for row in rows), "rows": rows}


REFUSED_MEDIA = "this bundle does not support media inputs"
REFUSED_PROMPT_SPECIAL = "prompt contains a special token the model does not admit in prompts: "


def refusal_message(row):
    error = row.get("error")
    error = error.get("error") if isinstance(error, dict) else None
    return error.get("message") if isinstance(error, dict) else None


def media_refused(row):
    return row.get("status") == 400 and refusal_message(row) == REFUSED_MEDIA


def identical_ids(ours, theirs, key, expected_refusals=()):
    declared = {}
    if expected_refusals and key != "tokenizer-golden.json":
        raise ValueError("expected tokenizer refusals apply only to tokenizer-golden.json")
    for refusal in expected_refusals:
        sha = refusal["sha256"]
        if sha in declared:
            raise ValueError("duplicate expected tokenizer refusal")
        matches = [row for row in theirs if hashlib.sha256(row["text"].encode()).hexdigest() == sha]
        if len(matches) != 1 or refusal["special_token"] not in matches[0]["text"]:
            raise ValueError("expected tokenizer refusal must name one golden text containing its special token")
        declared[sha] = refusal
    mismatches, expected, matched = [], [], 0
    for index, (a, b) in enumerate(zip(ours, theirs)):
        sha = hashlib.sha256(b["text"].encode()).hexdigest() if declared else None
        if sha in declared:
            refusal = declared[sha]
            message = refusal_message(a)
            agrees = (a.get("text") == b["text"] and a.get("status") == 400
                      and message == REFUSED_PROMPT_SPECIAL + refusal["special_token"])
            expected.append({**refusal, "row": index, "status": a.get("status"),
                             "message": message, "agrees": agrees})
        else:
            agrees = a.get("status", 200) == 200 and a.get("token_ids") == b.get("token_ids")
            matched += int(agrees)
        if not agrees:
            mismatches.append(index)
    refused = [index for index in mismatches if media_refused(ours[index])]
    return {"rows": len(ours), "mismatches": mismatches[:20], "media_refused": refused,
            "matched_rows": matched, "expected_refusals": expected,
            "agrees": not mismatches and len(ours) == len(theirs) and len(expected) == len(declared),
            "key": key, "reference_rows": len(theirs)}


def compare():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", nargs="?", type=Path, help="directory of served frontend observations")
    parser.add_argument("--url", help="capture the served frontend at this URL instead of reading CAPTURE")
    parser.add_argument("--model", help="with --url: the served model name (default: the server's first)")
    parser.add_argument("--capture-out", type=Path, help="with --url: the new directory the served capture is written to")
    parser.add_argument("--require-bound", action="store_true", help="require the producer metadata and live identity proof")
    parser.add_argument("--server-log", type=Path, help="with --url: bind the producer to the live startup log")
    parser.add_argument("--timeout", type=float, default=120, help="with --url: seconds per request")
    parser.add_argument("--reference", type=Path, help="reference capture directory (default: declared frontend_capture)")
    parser.add_argument("--json", help="write the report here")
    parser.add_argument('--required-file', action='append', default=[], choices=FILES,
                        help='fail if this requested file has no compared rows; repeat for a partial scope')
    args = parser.parse_args()
    if (args.capture is None) == (args.url is None) or bool(args.url) != bool(args.capture_out):
        parser.error("give either CAPTURE, or --url with --capture-out")
    config = qualification_config()
    check = next((check for check in config["checks"] if check["module"] == "frontend_goldens"), None)
    if check is None:
        parser.error("the declaration has no frontend_goldens check")
    reference = args.reference or resolve_input("frontend_capture")
    if args.server_log is not None and not args.url:
        parser.error("--server-log requires --url")
    captures = {}
    for name in ("chat-renders.json", "tokenizer-golden.json"):
        path = reference / name if args.reference or name == "chat-renders.json" else resolve_input("tokenizer_goldens")
        if path.exists():
            key = "frontend_capture" if name == "chat-renders.json" else "tokenizer_goldens"
            captures[name] = reference_bytes(config, key, path)
    references = {name: hashlib.sha256(data).hexdigest() for name, data in captures.items()}
    model = args.model or config.get('qualification', {}).get('checkpoint', {}).get('served_model_name')
    plan = extension(config, json.loads(captures.get("chat-renders.json", b"[]")), model)
    if args.url:
        renders, texts = (json.loads(captures[name]) for name in ("chat-renders.json", "tokenizer-golden.json"))
        serve_capture(args.url, args.model, renders, texts, args.capture_out, args.timeout,
                      server_log=args.server_log, config=config, references=references, require_bound=args.require_bound, plan=plan)
        args.capture = args.capture_out
    provenance = admit_capture(args.capture, config, json.loads(captures.get("chat-renders.json", b"[]")),
                               json.loads(captures.get("tokenizer-golden.json", b"[]")), references, plan=plan)
    if (args.require_bound or os.environ.get("CHOPINFER_FRONTEND_REQUIRE_BOUND") == "1") and provenance is None:
        raise ValueError("frontend qualification requires bound capture metadata")
    active_rules = [rule for rule in check["rules"] if not args.required_file or rule["path"][0] in args.required_file]
    report, passed = {}, True
    bindings = {}
    counts, omitted = {}, []
    captured_needles = load(args.capture, "needles.json")
    if captured_needles:
        report["needles.json"] = needles(captured_needles)
        counts["needles.json"] = len(report["needles.json"]["rows"])
    else:
        omitted.append("needles.json")
    samples = load(args.capture, "api-samples.json")
    shape_path = reference / "api-shapes.json"
    shape_bytes = reference_bytes(config, "frontend_capture", shape_path) if shape_path.exists() else None
    expected_shapes = json.loads(shape_bytes) if shape_bytes is not None else None
    if expected_shapes is not None:
        bindings["api-shapes.json"] = {"input": "frontend_capture", "file": "api-shapes.json",
                                      "sha256": hashlib.sha256(shape_bytes).hexdigest(), "rows": len(expected_shapes)}
    if samples and expected_shapes:
        report["api-samples.json"] = api(samples, expected_shapes)
        counts["api-samples.json"] = len(report["api-samples.json"]["rows"])
    else:
        omitted.append("api-samples.json")

    for name in ("chat-renders.json", "tokenizer-golden.json"):
        path = (reference / name if args.reference or name == "chat-renders.json" else resolve_input("tokenizer_goldens"))
        ours = load(args.capture, name)
        key = "frontend_capture" if name == "chat-renders.json" else "tokenizer_goldens"
        data = reference_bytes(config, key, path) if path.exists() else None
        theirs = json.loads(data) if data is not None else None
        if ours and theirs:
            report[name] = identical_ids(ours, theirs, name, check.get("tokenizer_refusals", [])
                                         if name == "tokenizer-golden.json" else [])
            bindings[name] = {"input": "frontend_capture" if name == "chat-renders.json" else "tokenizer_goldens",
                              "file": name if name == "chat-renders.json" else "",
                              "sha256": hashlib.sha256(data).hexdigest(), "rows": len(theirs)}
            counts[name] = min(len(ours), len(theirs))
        else:
            omitted.append(name)
    missing = [name for name in args.required_file if not counts.get(name)]
    gated_names = set(args.required_file) | {rule['path'][0] for rule in active_rules}
    gated = sum(count for name, count in counts.items() if name in gated_names)
    report['coverage'] = {'compared': sum(counts.values()), 'gated': gated,
                          'files': counts, 'required': args.required_file,
                          'missing': missing, 'omitted': omitted}
    report['reference'] = {'files': bindings}
    if provenance is not None:
        report['capture'] = provenance
    try:
        passed = all(predicate(report, rule) for rule in active_rules)
    except (KeyError, TypeError, IndexError):
        if not missing:
            missing.append('declared predicate evidence')
    report['status'] = 'insufficient' if not gated or missing else ('match' if passed else 'diverged')
    print(json.dumps(report, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return {'match': 0, 'diverged': 1, 'insufficient': 3}[report['status']]


def main():
    try:
        return compare()
    except ValueError as error:
        raise SystemExit(f"refused: {error}") from None
    except RuntimeError as error:
        raise SystemExit(f"capture failed: {error}") from None


if __name__ == "__main__":
    sys.exit(main())
