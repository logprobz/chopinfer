"""Compare verify-row outputs with autoregressive outputs, and record acceptance.

`record` sends the same greedy completions to one server and stores each
emitted token with its logprob and top-5 alternatives, plus the speculation
counters around each prompt set. A scored record binds the server to the
gate's checkpoint pin (and its startup log, when given), confirms every
response came from that process, and keeps the topology and speculation it
reported. `compare` reads a speculative and an autoregressive record of the
same run and scores the speculative tokens against the autoregressive ones
over each request's common prefix and first divergent position; bound arms
must have served the same checkpoint under the same name and topology, the
speculative arm drafting and the control not. Positions after a divergence
condition on different histories and are not scored: each set's `coverage`
counts them per request and in aggregate, and is complete only when no
request of either arm leaves a position unscored. The gate reads the copy
set's means over the scored positions; coverage is reported, not gated.
Each set's `ms_per_token` is labelled by `host_load`: timing evidence only
when both records ran on an idle host (common.compared_timing); the label
changes no verdict.

The "copy" set makes nearly every decode row a verify row: each prompt is a
text followed by its own first half, so prompt-lookup drafts the text's second
half and greedy decoding largely copies it. Over the frozen teacher texts and
GSM8K questions, its scored positions are verify rows on identical histories,
up to each request's first divergence. The "natural"
set is GSM8K questions answered step by step, where lookup drafts only where
the answer repeats its context.
"""

import hashlib
import json
from pathlib import Path
import os
import re
import statistics
import sys

from common import (input_bytes, input_path, Client, arguments, bind_checkpoint, compared_timing, confirm_live,
                    host_load_sample, receipt, serving_profile, startup_nonce)
from spec_metrics import difference, metrics_text, snapshot

ROOT = Path(__file__).resolve().parents[2]


MAXIMUM_ABS_DLOGPROB = 0.15
MAXIMUM_TOP1_DISAGREEMENT = 0.08


def prompt_sets(include_chat=False):
    from refs import fixture_bytes
    document = json.loads(input_bytes("spec_teacher"))
    texts = document["texts"] if isinstance(document, dict) else document
    teacher = [row["text"] for row in texts
               if "name" not in row or row["name"] in ("frozen0", "frozen1", "frozen2")]
    questions = [row["question"] for row in json.loads(fixture_bytes(input_path("gsm8k") / "examples.json"))]
    copies = [text + "\n\n" + text[: len(text) // 2] for text in teacher + questions[: 16 - len(teacher)]]
    natural = [f"Question: {question}\nAnswer step by step.\n" for question in questions[:16]]
    sets = {"copy": copies, "natural": natural}
    if include_chat:
        from spec_bench import CHAT_PROMPTS
        sets["chat"] = list(CHAT_PROMPTS)
    return sets


def engine_counters(text):
    values = {}
    for name in ("chop_generation_tokens_total", "chop_batches_total"):
        match = re.search(rf"^{name} (\d+)$", text, re.MULTILINE)
        values[name] = int(match[1]) if match else 0
    return values


def completion(client, model, prompt, output_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": output_tokens,
            "temperature": 0, "top_p": 1, "logprobs": 5}
    response, seconds = client.request("/v1/completions", body)
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    return {"tokens": logprobs["tokens"], "logprobs": logprobs["token_logprobs"],
            "top": logprobs["top_logprobs"], "finish_reason": choice["finish_reason"],
            "seconds": seconds}


def record(args):
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    try:
        bind_checkpoint(server, args.server_log, args.url)
        serving = serving_profile(server)
    except ValueError as error:
        raise SystemExit(f"refused: {error}")
    sets = {}
    load = [host_load_sample()]
    for name, prompts in prompt_sets(args.include_chat).items():
        text = metrics_text(client)
        before, engine_before = snapshot(text, "chopinfer", server["model"]), engine_counters(text)
        rows = [completion(client, server["model"], prompt, args.output_tokens) for prompt in prompts]
        text = metrics_text(client)
        after, engine_after = snapshot(text, "chopinfer", server["model"]), engine_counters(text)
        generated = engine_after["chop_generation_tokens_total"] - engine_before["chop_generation_tokens_total"]
        steps = engine_after["chop_batches_total"] - engine_before["chop_batches_total"]
        sets[name] = {"prompts_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
                      "requests": rows, "speculation": difference(before, after),
                      "generated_tokens": generated, "engine_steps": steps,
                      # Requests run one at a time, so this is C1 latency,
                      # prefill included.
                      "ms_per_token": 1000 * sum(row["seconds"] for row in rows)
                      / sum(len(row["tokens"]) for row in rows),
                      # Prefill steps are included, so this slightly
                      # understates decode tokens per step.
                      "tokens_per_engine_step": generated / steps if steps else None}
    load.append(host_load_sample())
    confirm_live(client, server)
    receipt(f"spec-verify-{args.arm}", server, {
        "passed": True, "arm": args.arm, "run_id": args.run_id, "serving": serving,
        "output_tokens": args.output_tokens, "sampling": {"temperature": 0, "top_p": 1, "logprobs": 5}, "sets": sets}, False,
        load_samples=load)


def top_one(top):
    return max(top, key=lambda token: top[token]) if top else None


def bound_server(record):
    """The checkpoint identity, startup nonce and served name a record's
    server was bound to, with the topology and speculation it served, or None
    for a record taken without a checkpoint binding. A record that claims a
    binding must carry all of them, well typed."""
    server = record.get("server") or {}
    checkpoint = server.get("checkpoint")
    if not checkpoint:
        return None
    identity, nonce, model = checkpoint.get("checkpoint_identity"), checkpoint.get("startup_nonce"), server.get("model")
    if not (isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity)):
        raise ValueError(f"a bound record names no checkpoint identity: {identity!r}")
    if not startup_nonce(nonce):
        raise ValueError(f"a bound record names no startup nonce: {nonce!r}")
    if not (isinstance(model, str) and model):
        raise ValueError(f"a bound record names no served model: {model!r}")
    serving = record.get("serving")
    if not isinstance(serving, dict):
        raise ValueError("a bound record lacks the serving topology and speculation")
    size, kv = serving.get("tensor_parallel_size"), serving.get("kv_cache_dtype")
    speculation = serving.get("speculation")
    if type(size) is not int or size < 1:
        raise ValueError(f"a bound record names no tensor parallel size: {size!r}")
    if kv not in ("bf16", "fp8"):
        raise ValueError(f"a bound record names no KV cache dtype: {kv!r}")
    if not (isinstance(speculation, dict) and speculation.get("mode") in ("off", "fixed", "adaptive")
            and type(speculation.get("tokens")) is int and speculation["tokens"] >= 0
            and (speculation["mode"] == "off") == (speculation["tokens"] == 0)):
        raise ValueError(f"a bound record names no consistent speculation: {speculation!r}")
    return {"checkpoint_identity": identity, "model": model, "startup_nonce": nonce, "serving": serving}


def compare_servers(spec, ar):
    """Both arms completely bound to one checkpoint and served name, by two
    processes with one topology; the speculative arm drafting and the control
    not. Records taken without a binding (before it existed) compare only
    with each other."""
    servers = {"spec": bound_server(spec), "ar": bound_server(ar)}
    if (servers["spec"] is None) != (servers["ar"] is None):
        raise ValueError("only one arm is bound to a checkpoint")
    if servers["spec"] is None:
        return None
    key = {arm: (value["checkpoint_identity"], value["model"]) for arm, value in servers.items()}
    if key["spec"] != key["ar"]:
        raise ValueError(f"the arms served different checkpoints or names: {key}")
    if servers["spec"]["startup_nonce"] == servers["ar"]["startup_nonce"]:
        raise ValueError("both arms name one server process; a drafting and a control arm need two")
    serving = {arm: value["serving"] for arm, value in servers.items()}
    if serving["spec"]["tensor_parallel_size"] != serving["ar"]["tensor_parallel_size"] \
            or serving["spec"]["kv_cache_dtype"] != serving["ar"]["kv_cache_dtype"]:
        raise ValueError("the arms served different topologies or KV dtypes")
    drafting = {arm: value["speculation"]["tokens"] for arm, value in serving.items()}
    if drafting["spec"] <= 0 or drafting["ar"] != 0:
        raise ValueError(f"the speculative arm must draft and the control must not: {drafting}")
    return {"checkpoint_identity": key["spec"][0], "model": key["spec"][1], "serving": serving}


def compare_records(spec, ar, maximum_abs_dlogprob=MAXIMUM_ABS_DLOGPROB,
                    maximum_top1_disagreement=MAXIMUM_TOP1_DISAGREEMENT):
    if spec.get("arm") != "chopinfer-spec" or ar.get("arm") != "chopinfer-ar":
        raise ValueError("comparison requires distinct speculative and AR arms")
    bound = compare_servers(spec, ar)
    for name, group in spec["sets"].items():
        if not group.get("requests") or not ar["sets"].get(name, {}).get("requests"):
            raise ValueError("comparison needs nonempty requests in both arms")
        control = ar["sets"][name].get("speculation", {})
        if control.get("draft_tokens", 0) or control.get("verify_items", 0):
            raise ValueError("AR control reports drafting activity")
    activity = spec["sets"].get("copy", {}).get("speculation", {})
    if activity.get("draft_tokens", 0) <= 0 or activity.get("verify_items", 0) <= 0:
        raise ValueError("speculative arm has no drafting activity")
    if spec["run_id"] != ar["run_id"] or spec["output_tokens"] != ar["output_tokens"]:
        raise SystemExit("records come from different runs")
    if spec.get("sampling") != ar.get("sampling"):
        raise SystemExit("records use different sampling settings")
    if set(spec["sets"]) != set(ar["sets"]):
        raise SystemExit("records contain different prompt sets")
    report = {}
    for name in spec["sets"]:
        left, right = spec["sets"][name], ar["sets"][name]
        if left["prompts_sha256"] != right["prompts_sha256"]:
            raise SystemExit(f"{name} prompts differ")
        if not left["requests"] or len(left["requests"]) != len(right["requests"]):
            raise SystemExit(f"{name} request counts differ or are empty")
        differences, disagreements, exact, prefixes, requests = [], 0, 0, [], []
        for spec_row, ar_row in zip(left["requests"], right["requests"]):
            prefix = 0
            for spec_token, ar_token in zip(spec_row["tokens"], ar_row["tokens"]):
                if spec_token != ar_token:
                    break
                prefix += 1
            prefixes.append(prefix)
            exact += spec_row["tokens"] == ar_row["tokens"]
            # Every position of the common prefix, plus the first divergent
            # position, conditions on identical tokens in both runs. Later
            # positions condition on different histories and stay unscored.
            compared = min(prefix + 1, len(spec_row["tokens"]), len(ar_row["tokens"]))
            for position in range(compared):
                if position < prefix:
                    differences.append(abs(spec_row["logprobs"][position] - ar_row["logprobs"][position]))
                disagreements += top_one(spec_row["top"][position]) != top_one(ar_row["top"][position])
            requests.append({"emitted": len(spec_row["tokens"]), "ar_emitted": len(ar_row["tokens"]),
                             "common_prefix": prefix, "scored": compared,
                             "unscored": len(spec_row["tokens"]) - compared})
        scored = sum(row["scored"] for row in requests)
        emitted = sum(row["emitted"] for row in requests)
        ar_unscored = sum(row["ar_emitted"] - row["scored"] for row in requests)
        report[name] = {
            "requests": len(prefixes), "exact_requests": exact,
            "mean_common_prefix": statistics.mean(prefixes),
            "scored_positions": scored,
            # Coverage of the speculative arm's positions. Complete only when
            # no request of either arm leaves a position unscored.
            "coverage": {"rule": "common prefix and first divergence",
                         "emitted_positions": emitted, "scored_positions": scored,
                         "unscored_positions": emitted - scored, "ar_unscored_positions": ar_unscored,
                         "fraction": scored / emitted if emitted else None,
                         "complete_requests": sum(row["unscored"] == 0 and row["ar_emitted"] == row["scored"]
                                                  for row in requests),
                         "complete": emitted > 0 and scored == emitted and ar_unscored == 0,
                         "requests": requests},
            "mean_abs_dlogprob": statistics.mean(differences) if differences else None,
            "top1_disagreement": disagreements / scored if scored else None,
            "speculation": left["speculation"],
            "tokens_per_engine_step": {"spec": left["tokens_per_engine_step"],
                                       "ar": right["tokens_per_engine_step"]},
            "ms_per_token": {"spec": left.get("ms_per_token"), "ar": right.get("ms_per_token")}}
    gate = report["copy"]
    passed = (gate["mean_abs_dlogprob"] is not None
              and gate["mean_abs_dlogprob"] <= maximum_abs_dlogprob
              and gate["top1_disagreement"] <= maximum_top1_disagreement)
    return {"passed": passed, "run_id": spec["run_id"], "bound": bound,
            "arms": {"ar": ar["arm"], "spec": spec["arm"]},
            "ar_speculation": ar["sets"]["copy"].get("speculation", {}),
            "gates": {"set": "copy", "maximum_abs_dlogprob": maximum_abs_dlogprob,
                      "maximum_top1_disagreement": maximum_top1_disagreement},
            "host_load": compared_timing({"spec": spec.get("host_load"), "ar": ar.get("host_load")}),
            "sets": report}


def compare(args):
    result = compare_records(json.loads(Path(args.spec).read_text()),
                             json.loads(Path(args.ar).read_text()),
                             args.maximum_abs_dlogprob, args.maximum_top1_disagreement)
    json.dump(result, sys.stdout, indent=1)
    print()
    raise SystemExit(0 if result['passed'] else 1)


def main():
    parser = arguments(__doc__, model_facts=lambda args: args.command == "record")
    parser.add_argument("command", choices=("record", "compare"))
    parser.add_argument("--arm", choices=("chopinfer-ar", "chopinfer-spec"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--server-log", type=Path,
                        help="record: the server's startup log; when given, it must be the live server's")
    parser.add_argument("--spec", help="compare: the speculative record")
    parser.add_argument("--ar", help="compare: the autoregressive record")
    parser.add_argument("--include-chat", action="store_true",
                        help="record: also test the benchmark's chat prompts")
    parser.add_argument("--maximum-abs-dlogprob", type=float, default=MAXIMUM_ABS_DLOGPROB,
                        help="compare: maximum copy-set mean absolute logprob difference")
    parser.add_argument("--maximum-top1-disagreement", type=float, default=MAXIMUM_TOP1_DISAGREEMENT,
                        help="compare: maximum copy-set top-1 disagreement fraction")
    args = parser.parse_args()
    if not 0 <= args.maximum_abs_dlogprob < float("inf") or not 0 <= args.maximum_top1_disagreement <= 1:
        parser.error("comparison bounds must be finite and nonnegative; disagreement must be at most 1")
    if args.command == "record":
        if not args.arm or not args.run_id:
            parser.error("record needs --arm and --run-id")
        record(args)
    else:
        if not args.spec or not args.ar:
            parser.error("compare needs --spec and --ar")
        compare(args)


if __name__ == "__main__":
    main()
