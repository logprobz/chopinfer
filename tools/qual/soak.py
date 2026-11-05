"""Run a mixed C32 chat soak and watch device memory for growth.

With --prefix-mix, a third of the workers hold multi-turn conversations and a
third share long system prompts, so the prefix cache hits, captures and evicts
under load. A greedy canary checks restores (prefix-cache contract, gate 6). Its
system prompt is one the workers share, so it runs under its own cache salt:
only anchors the canary captured can serve it. Before the load, a primer
shares the canary's tokens up to its last block boundary and then diverges
(prefix_exact.primer: it runs a block on, or past the next chunk, so that
neither its prompt-end anchor, even rounded down to a prefill granule, nor
its final-chunk anchor lies at or before the divergence), so the cold canary
that follows ends a prefill chunk at that anchor and captures there. The
soak retains this chunked baseline; a C1 cold one-shot prefill is also
numerically valid under the bitwise prefix-cache contract. After the load
the canary runs twice more; each either hits the anchor
or, if it was evicted, is cut there again, and the last one must hit. An
answer that differs under load follows the separate near-tie rule: at the first
differing token, each run ranks the other's token second within
NEAR_TIE_MARGIN and the common prefix stayed within PREFIX_DLOGPROB; any
other difference counts as a restore error. The soak fails on a restore
error, on a canary that did not hit, on live pages once idle, or when the
cache served no hit.

--media-mix sends every eighth request of each worker with one frozen image,
audio or video part (a server packaged with encoders); the blue-square image
must be answered blue. --structured-mix sends another eighth with the strict
JSON schema of structured.py at default sampling; an output that stops must
parse and satisfy the schema, and a length stop is counted as truncated. The
receipt keeps the first KEEP_STRUCTURED truncated or invalid outputs
(`structured_kept`: request body, response text, finish reason, completion
tokens and the server's response id) and their total.

Each memory sample also records the serving tree's RssAnon, RssFile and
RssShmem and, when /metrics serves them, the host's mapped-view gauges. The
memory rule judges device memory and the host memory of the declared
HOST_MEMORY_SCOPE: VmRSS by default, or VmRSS less the Rss of the declared
read-only mapped views the serving tree holds at the first sample, read per
mapping from /proc/PID/smaps.
"""

from concurrent.futures import ThreadPoolExecutor
import errno
import math
import os
import re
import stat
import statistics
import subprocess
import threading
import time
from urllib.parse import urlsplit

import json
from pathlib import Path
import importlib.util
import uuid

from common import (REASONING_TOKENS, answer_control, host_load_sample, input_path, thinking_can_be_disabled,
                    Client, arguments, not_applicable, receipt)
from prefix_exact import BLOCK_TOKENS, primer, tokenize
from spec_metrics import metrics_text
from structured import PERSON, PERSON_PROMPT, schema_errors

# A model declares these in its soak check's constants (checks.soak.constants
# .soak) when its server fills a bounded cache during warm-up, such as pinned
# prefill graphs. The growth after WARMUP_SECONDS, measured from the first
# sample at or after it, must stay within the memory limits; the growth over
# the whole soak, from the first sample, within the limits plus
# CACHE_ALLOWANCE_MIB (host and device, the cache's declared bound). The
# warm-up is capped at a quarter of the duration. With the defaults both
# conditions are the plain rule: growth from the first sample within the
# limits.
WARMUP_SECONDS = 0
CACHE_ALLOWANCE_MIB = {"host": 0, "device": 0}
# The host memory the rule judges, declared in the same constants: "rss",
# the default, judges VmRSS. "rss_less_mapped_views", for a server that maps
# large read-only checkpoint files, judges VmRSS less the Rss of the shared
# read-only mappings of MAPPED_VIEW_FILES (file names relative to the served
# checkpoint directory) that the serving tree holds at the first sample,
# attributed per mapping from /proc/PID/smaps, while each stays present,
# shared, read-only and clean (no Anonymous, Private_Dirty, Shared_Dirty or
# Swap pages) with its first sample's permissions in every sample. A mapping
# is identified by its process (PID and start time), address range, device,
# inode and offset, compared as integers; a sample with two mappings that
# share an address within a process has no evidence. Every other mapping
# counts: one a sample saw absent or otherwise, from then on, even under the
# same identity, one first seen later, a private, writable or dirty one, and
# every other file-backed mapping, such as the driver's pinned host buffers
# mapped through the GPU device nodes. One case goes unseen: a mapping
# unmapped and replaced within one sampling interval (30 seconds) by one with
# the identical identity that is also shared, read-only with the same
# permissions, and clean. That is the same file's clean shared page cache at
# the same address, so it cannot hide anonymous or dirty memory.
HOST_MEMORY_SCOPE = "rss"
MAPPED_VIEW_FILES = ()


def memory_policy(warmup=None, allowance=None):
    """The declared (warm-up seconds, allowance) pair, validated."""
    warmup = WARMUP_SECONDS if warmup is None else warmup
    allowance = CACHE_ALLOWANCE_MIB if allowance is None else allowance
    if type(warmup) is not int or warmup < 0:
        raise ValueError("soak warm-up must be a non-negative integer number of seconds")
    if not isinstance(allowance, dict) or set(allowance) != {"host", "device"} or any(
            type(value) is not int or value < 0 for value in allowance.values()):
        raise ValueError("soak cache allowance must give non-negative integer host and device MiB")
    return warmup, dict(allowance)


def host_memory_scope(scope=None):
    """The declared host memory scope, validated."""
    scope = HOST_MEMORY_SCOPE if scope is None else scope
    if scope not in ("rss", "rss_less_mapped_views"):
        raise ValueError(f"soak host memory scope must be rss or rss_less_mapped_views, not {scope!r}")
    return scope


def mapped_view_files(files=None, scope=None):
    """The declared mapped-view files, validated for the host memory scope:
    distinct file names relative to the checkpoint directory, none ending as
    smaps marks a deleted file, at least one under rss_less_mapped_views and
    none under rss."""
    files = MAPPED_VIEW_FILES if files is None else files
    scope = host_memory_scope(scope)
    if not isinstance(files, (list, tuple)) or not all(
            isinstance(name, str) and all(part not in ("", ".", "..") for part in name.split("/"))
            and not name.endswith(" (deleted)") for name in files) or len(set(files)) != len(files):
        raise ValueError("soak mapped-view files must be distinct file names relative to the checkpoint directory")
    if bool(files) != (scope == "rss_less_mapped_views"):
        raise ValueError("soak mapped-view files are declared with, and only with, the rss_less_mapped_views scope")
    return list(files)


def natural(value):
    """Whether `value` is a non-negative integer (and not a boolean)."""
    return type(value) is int and value >= 0


def exempt_identity(entry):
    """The identity of an exempt mapping a sample lists (PID, start time,
    start and end address, device major and minor, inode, offset, each an
    integer), or None unless the entry is a shared read-only mapping without
    Anonymous, Private_Dirty, Shared_Dirty or Swap pages whose Size, more
    than 0, is its address range and holds its Rss."""
    keys = ("pid", "start_time", "range", "device", "inode", "offset", "perms", "size_kib", "rss_kib",
            "anonymous_kib", "private_dirty_kib", "shared_dirty_kib", "swap_kib")
    if not isinstance(entry, dict) or set(entry) != set(keys):
        return None
    pair = lambda key: type(entry[key]) is list and len(entry[key]) == 2 and all(map(natural, entry[key]))
    if not (pair("range") and pair("device") and natural(entry["pid"]) and entry["pid"] > 0
            and all(natural(entry[key]) for key in ("start_time", "inode", "offset", "size_kib", "rss_kib"))
            and isinstance(entry["perms"], str) and re.fullmatch(r"[r-]-[x-]s", entry["perms"])
            and all(entry[key] == 0 and natural(entry[key]) for key in keys[9:])):
        return None
    (start, end), size = entry["range"], entry["size_kib"]
    if size == 0 or size * 1024 != end - start or entry["rss_kib"] > size:
        return None
    return entry["pid"], entry["start_time"], start, end, *entry["device"], entry["inode"], entry["offset"]


def overlapping(spans):
    """Whether any two of the address ranges `spans` ([start, end)) share an
    address."""
    spans = sorted(spans)
    return any(start < end for (_, end), (start, _) in zip(spans, spans[1:]))


def exempt_problem(samples):
    """Why the rss_less_mapped_views scope cannot judge `samples`, or None.
    Each sample lists its exempt mappings (`host_exempt_mappings`), each a
    distinct clean shared read-only mapping whose Size is its address range
    and holds its Rss (exempt_identity), none overlapping another of its
    process, which has one start time, and every one listed, with the same
    permissions, by the first sample and by each sample since. Its VmRSS,
    exempt Rss, bound and judged host memory are finite non-negative
    numbers: the exempt Rss and bound the listed mappings' Rss and Size, in
    MiB, the exempt Rss within VmRSS, and the judged value VmRSS less it."""
    keys = ("host_rss_mib", "host_exempt_rss_mib", "host_exempt_bound_mib", "host_judged_mib")
    held = None
    for row in samples:
        at = f"memory sample at {row.get('elapsed_seconds')} s"
        if missing := [key for key in keys if type(row.get(key)) not in (int, float)
                       or not math.isfinite(row[key]) or row[key] < 0]:
            return f"{at} has no {', '.join(missing)}" + (f" ({row['host_exempt_error']})"
                                                          if row.get("host_exempt_error") else "")
        entries = row.get("host_exempt_mappings")
        identities = [exempt_identity(entry) for entry in entries] if isinstance(entries, list) else [None]
        if None in identities or len(set(identities)) != len(identities):
            return (f"{at} lists no exempt mappings, or one that is not a distinct clean shared read-only mapping "
                    "within its Size")
        processes = {}
        for entry in entries:
            processes.setdefault(entry["pid"], []).append((entry["start_time"], entry["range"]))
        if any(len({born for born, _ in listed}) > 1 or overlapping(span for _, span in listed)
               for listed in processes.values()):
            return f"{at} lists exempt mappings that overlap within a process, or a process with two start times"
        listed = {identity: entry["perms"] for identity, entry in zip(identities, entries)}
        if held is not None and any(held.get(identity) != perms for identity, perms in listed.items()):
            return (f"{at} exempts a mapping that the first sample and every sample since did not hold with the "
                    "same permissions")
        held = listed
        if (row["host_exempt_rss_mib"] != sum(entry["rss_kib"] for entry in entries) / 1024
                or row["host_exempt_bound_mib"] != sum(entry["size_kib"] for entry in entries) / 1024):
            return f"{at} holds an exempt Rss or bound other than its listed mappings'"
        if (row["host_exempt_rss_mib"] > row["host_rss_mib"]
                or row["host_judged_mib"] != row["host_rss_mib"] - row["host_exempt_rss_mib"]):
            return f"{at} holds an exempt Rss beyond VmRSS, or a judged value other than VmRSS less it"
    return None


def exempt_receipt_problem(value, files):
    """Why the rss_less_mapped_views scope cannot judge a soak receipt
    declaring the mapped-view `files`, or None: the receipt names each
    file's path, device and inode in the serving process's namespace
    (`mapped_view_file_identities`), its samples hold consistent evidence
    (exempt_problem), and every exempt mapping they list is of one of those
    files."""
    identities = value.get("mapped_view_file_identities")
    if not (isinstance(identities, list) and all(
            isinstance(entry, dict) and set(entry) == {"file", "path", "device", "inode"}
            and isinstance(entry["path"], str) and type(entry["device"]) is list and len(entry["device"]) == 2
            and all(map(natural, entry["device"])) and natural(entry["inode"]) for entry in identities)
            and [entry["file"] for entry in identities] == files):
        return "the receipt names no device and inode of each declared mapped-view file"
    if problem := exempt_problem(value.get("memory_samples") or []):
        return problem
    declared = {(*entry["device"], entry["inode"]) for entry in identities}
    for row in value["memory_samples"]:
        if any((*entry["device"], entry["inode"]) not in declared for entry in row["host_exempt_mappings"]):
            return f"memory sample at {row['elapsed_seconds']} s exempts a mapping of no declared file"
    return None


def memory_rule(samples, duration, limits, warmup=None, allowance=None, scope=None):
    """Growth of device (`mib`) and host memory (`host_rss_mib`, or under the
    rss_less_mapped_views scope VmRSS less the exempt mappings' Rss,
    `host_judged_mib`) over the whole soak and after the warm-up, judged
    against `limits` ({"device", "host"} MiB) and the declared allowance.
    Under that scope every sample must carry its exempt mappings and
    consistent judged, exempt and bound values (exempt_problem), and the
    judged series is recomputed from VmRSS and the exempt Rss."""
    warmup, allowance = memory_policy(warmup, allowance)
    scope = host_memory_scope(scope)
    if scope == "rss_less_mapped_views":
        if problem := exempt_problem(samples):
            raise ValueError(problem)
        host = lambda row: row["host_rss_mib"] - row["host_exempt_rss_mib"]
    else:
        host = lambda row: row["host_rss_mib"]
    effective = min(warmup, duration / 4)
    steady = [row for row in samples if row["elapsed_seconds"] >= effective]
    if not steady:
        raise ValueError("no memory sample after the soak warm-up")
    result = {"warmup_seconds": warmup, "effective_warmup_seconds": effective,
              "steady_baseline_seconds": steady[0]["elapsed_seconds"], "cache_allowance_mib": allowance}
    passed = True
    for kind, value, prefix in (("device", lambda row: row["mib"], "memory"), ("host", host, "host_memory")):
        whole = max(map(value, samples)) - value(samples[0])
        after = max(map(value, steady)) - value(steady[0])
        result[prefix + "_growth_mib"] = whole
        result["steady_" + prefix + "_growth_mib"] = after
        passed = passed and after <= limits[kind] and whole <= limits[kind] + allowance[kind]
    if scope == "rss_less_mapped_views":
        result.update(host_memory_scope=scope,
                      host_exempt_rss_mib=max(row["host_exempt_rss_mib"] for row in samples),
                      host_exempt_bound_mib=max(row["host_exempt_bound_mib"] for row in samples))
    result["passed"] = passed
    return result


def media_module():
    path = input_path("media_cases")
    spec = importlib.util.spec_from_file_location("qualification_media_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPTS = ("Say hello.", "Summarize why a cache can improve repeated reads in one sentence.",
           "List three ways to check a numerical result. " * 60)


SYSTEMS = tuple(f"You are assistant {index}. " + " ".join(
    f"Rule {rule}: answer with care about topic {index * 31 + rule}." for rule in range(160))
    for index in range(4))
CANARY = [{"role": "system", "content": SYSTEMS[0]},
          {"role": "user", "content": "State rule 7 in your own words."}]


# The chunked-prefill noise floor reached 0.413 max |dlogprob| between two
# chunk sizes without caching.
NEAR_TIE_MARGIN = 0.5
PREFIX_DLOGPROB = 0.5


def classify_canary(cold, other):
    """Whether `other` emits the cold canary's tokens, flips a near tie, or not."""
    tokens = cold["tokens"], other["tokens"]
    at = next((index for index, (x, y) in enumerate(zip(*tokens)) if x != y),
              None if len(tokens[0]) == len(tokens[1]) else min(map(len, tokens)))
    if at is None:
        return {"verdict": "exact"}
    drift = max((abs(x - y) for x, y in zip(cold["logprobs"][:at], other["logprobs"][:at])),
                default=0.0)
    result = {"verdict": "restore_error", "first_difference": at, "prefix_max_abs_dlogprob": drift,
              "cold_margin": None, "other_margin": None}
    if at >= len(tokens[0]) or at >= len(tokens[1]):
        # One run stopped where the other went on; the stop has no top-2 entry.
        return result

    def margin(run, chosen, runner_up):
        top = {entry["token"]: entry["logprob"] for entry in run["top_logprobs"][at]}
        return top[chosen] - top[runner_up] if chosen in top and runner_up in top else None

    a, b = tokens[0][at], tokens[1][at]
    result["cold_margin"] = margin(cold, a, b)
    result["other_margin"] = margin(other, b, a)
    if (result["cold_margin"] is not None and result["other_margin"] is not None
            and max(result["cold_margin"], result["other_margin"]) <= NEAR_TIE_MARGIN
            and drift <= PREFIX_DLOGPROB):
        result["verdict"] = "near_tie"
    return result


def completion_canary(value):
    """A canary answer from a /v1/completions response with top-2 logprobs."""
    choice = value["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {"content": choice.get("text"), "usage": value["usage"],
            "tokens": logprobs.get("tokens") or [],
            "logprobs": logprobs.get("token_logprobs") or [],
            "top_logprobs": [[{"token": token, "logprob": logprob}
                              for token, logprob in (top or {}).items()]
                             for top in logprobs.get("top_logprobs") or []]}


def cached(run):
    return (run["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def check_canaries(canaries, anchor):
    """Compares every later canary with the baseline cut at `anchor`."""
    baseline, later = canaries[0], canaries[1:]
    verdicts = [classify_canary(baseline, other) for other in later]
    return {"canary_anchor": anchor, "canary_verdicts": verdicts,
            "canary_cached_tokens": [cached(run) for run in canaries],
            "canary_passed": (cached(baseline) == 0
                              and all(cached(run) in (0, anchor) for run in later)
                              and bool(later) and cached(later[-1]) == anchor
                              and all(verdict["verdict"] != "restore_error"
                                      for verdict in verdicts))}


def metric(text, name):
    for line in text.splitlines():
        if line.startswith(name + " "):
            return int(float(line.split()[1]))
    return None


def prefix_messages(worker_id, turn, conversation):
    """The next request of a --prefix-mix worker."""
    kind = worker_id % 3
    if kind == 0:
        if not conversation or len(conversation) > 12:
            conversation[:] = [{"role": "user", "content": f"Worker {worker_id} turn {turn}: "
                                + PROMPTS[turn % len(PROMPTS)]}]
        return conversation
    if kind == 1:
        return [{"role": "system", "content": SYSTEMS[(worker_id + turn) % len(SYSTEMS)]},
                {"role": "user", "content": f"Worker {worker_id} asks question {turn}."}]
    return [{"role": "user", "content": f"Worker {worker_id} turn {turn}: "
             + PROMPTS[(worker_id + turn) % len(PROMPTS)]}]


MEDIA_CASES = ("image-blue-square.png", "video-red-then-blue")


def media_fixtures():
    """The media mix's fixture directory: the declaration's
    soak_media_fixtures input when it states one (a model whose
    media_fixtures holds other media, such as image probes), else
    media_fixtures."""
    from common import qualification_config
    declared = qualification_config()["inputs"]
    return input_path("soak_media_fixtures" if "soak_media_fixtures" in declared else "media_fixtures")


def media_requests(case_ids=MEDIA_CASES, fixture_dir=None):
    """(case id, messages) for the media mix, each fixture read and verified once."""
    multimodal_cases = media_module()
    fixture_dir = fixture_dir or media_fixtures()
    by_id = {case["id"]: case for case in multimodal_cases.cases()}
    return [(case_id, multimodal_cases.messages(by_id[case_id], fixture_dir)) for case_id in case_ids]


def check_media(case_id, choice):
    content = (choice["message"].get("content") or "").strip()
    if not content:
        raise ValueError(f"{case_id}: empty answer")
    if case_id == "image-blue-square.png" and "blue" not in content.lower():
        raise ValueError(f"{case_id}: answered {content[:40]!r}, expected blue")


STRUCTURED_FORMAT = {"type": "json_schema",
                     "json_schema": {"name": "person", "schema": PERSON, "strict": True}}


# The truncated or invalid structured outputs a receipt keeps in full.
KEEP_STRUCTURED = 16


def kept_structured(worker_id, turn, elapsed, latency, body, value, reason):
    """A structured request worth inspecting: what was sent and returned."""
    choice = value["choices"][0]
    answer = choice.get("message") or {}
    return {"worker": worker_id, "turn": turn, "elapsed_seconds": elapsed, "latency_seconds": latency,
            "reason": reason, "request": body, "response_id": value.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (value.get("usage") or {}).get("completion_tokens"),
            "content": answer.get("content"),
            "reasoning_content": answer.get("reasoning_content") or answer.get("reasoning")}


def check_structured(choice):
    """True when the output was truncated; raises on a grammar violation."""
    if choice.get("finish_reason") == "length":
        return True
    value = json.loads(choice["message"].get("content") or "")
    errors = schema_errors(value, PERSON)
    if errors:
        raise ValueError("schema: " + "; ".join(errors[:3]))
    return False


def serving_pid(url):
    endpoint = urlsplit(url)
    if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("GPU memory sampling requires a local server URL or --server-pid")
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    output = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"],
                            check=True, capture_output=True, text=True).stdout
    pids = {int(pid) for pid in re.findall(r"\bpid=(\d+)\b", output)}
    if len(pids) != 1:
        raise RuntimeError(f"expected one serving PID on port {port}, found {sorted(pids)}")
    return pids.pop()


def process_tree(root_pid):
    output = subprocess.run(["ps", "-eo", "pid=,ppid="],
                            check=True, capture_output=True, text=True).stdout
    parents = {}
    for line in output.splitlines():
        pid, parent = line.split()
        parents[int(pid)] = int(parent)
    if root_pid not in parents:
        raise RuntimeError(f"serving PID {root_pid} exited during GPU memory sampling")
    family = {root_pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in family}
        if children <= family:
            return family
        family.update(children)


def device_memory(root_pid):
    family = process_tree(root_pid)
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True).stdout
    matched = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, mib = (field.strip() for field in line.split(",", 1))
        if int(pid) in family:
            matched.append(int(mib))
    if not matched:
        raise RuntimeError(f"no GPU allocation found for serving PID {root_pid} or its children")
    return sum(matched)


def smaps(pid):
    """The mappings of /proc/PID/smaps, each with its identity across samples
    (start and end address, device major and minor, inode and offset, as
    integers), permissions, path, and Size, Rss, Anonymous, Private_Dirty,
    Shared_Dirty and Swap in kB. Only the header lines and those fields are
    parsed. Raises ValueError unless the address ranges are non-empty and
    disjoint, so no two mappings share an identity or an address."""
    fields = (b"Size:", b"Rss:", b"Anonymous:", b"Private_Dirty:", b"Shared_Dirty:", b"Swap:")
    mappings = []
    for line in Path(f"/proc/{pid}/smaps").read_bytes().split(b"\n"):
        if line and line[0] in b"0123456789abcdef":
            header = re.fullmatch(rb"([0-9a-f]+)-([0-9a-f]+) +(\S{4}) +([0-9a-f]+) +([0-9a-f]+):([0-9a-f]+) +([0-9]+)"
                                  rb"(?: +(.*))?", line)
            if not header:
                raise ValueError(f"/proc/{pid}/smaps has a malformed mapping line")
            start, end, perms, offset, major, minor, inode, path = header.groups()
            mappings.append({"key": (int(start, 16), int(end, 16), int(major, 16), int(minor, 16), int(inode),
                                     int(offset, 16)),
                             "perms": perms.decode(), "path": os.fsdecode(path or b"")})
        elif line.startswith(fields):
            if not mappings:
                raise ValueError(f"/proc/{pid}/smaps has a field before its first mapping")
            name, value = line.split()[:2]
            mappings[-1][name[:-1].decode()] = int(value)
    if any(field[:-1].decode() not in mapping for mapping in mappings for field in fields):
        raise ValueError(f"/proc/{pid}/smaps has a mapping without Size, Rss, Anonymous, Private_Dirty, "
                         "Shared_Dirty or Swap")
    spans = [mapping["key"][:2] for mapping in mappings]
    if any(start >= end for start, end in spans) or overlapping(spans):
        raise ValueError(f"/proc/{pid}/smaps has an empty or overlapping address range")
    return mappings


def process_root(pid):
    """The root directory of a process's mount namespace, through procfs."""
    return f"/proc/{pid}/root"


def resolve_within(root, path):
    """`path`, absolute in the namespace whose root directory is `root`, with
    every symbolic link resolved within that root (an absolute link target
    starts again at `root`): the path as that namespace names it."""
    pending, resolved, links = [part for part in path.split("/") if part], [], 0
    while pending:
        part = pending.pop(0)
        if part == ".":
            continue
        if part == "..":
            resolved = resolved[:-1]
            continue
        if not stat.S_ISLNK(os.lstat(os.path.join(root, *resolved, part)).st_mode):
            resolved.append(part)
            continue
        links += 1
        if links > 40:
            raise OSError(errno.ELOOP, "too many levels of symbolic links", path)
        target = os.readlink(os.path.join(root, *resolved, part))
        resolved = [] if target.startswith("/") else resolved
        pending[:0] = [part for part in target.split("/") if part]
    return "/" + "/".join(resolved)


def mapped_view_identities(pids, files):
    """The declared files as the serving tree's `serve` process sees them,
    each as (name, path, (device major, minor), inode): its checkpoint
    directory is the value of --model (or, beside --bundle, --checkpoint)
    against its working directory, each file's path has every symbolic link
    resolved within the process's root, and the file there must be a regular
    one. Every `serve` process of the tree must see the same files."""
    found = set()
    for pid in pids:
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if argv[1:2] != [b"serve"]:
            continue
        named = [value for flag, value in zip(argv, argv[1:]) if flag in (b"--model", b"--checkpoint")]
        if len(named) != 1 or not named[0]:
            raise ValueError(f"serving process {pid} names no single checkpoint directory")
        directory, root = os.path.join(os.readlink(f"/proc/{pid}/cwd"), os.fsdecode(named[0])), process_root(pid)
        identities = []
        for name in files:
            path = resolve_within(root, os.path.join(directory, name))
            status = os.stat(os.path.join(root, path.lstrip("/")), follow_symlinks=False)
            if not stat.S_ISREG(status.st_mode):
                raise ValueError(f"mapped-view file {name} of serving process {pid} is not a regular file ({path})")
            identities.append((name, path, (os.major(status.st_dev), os.minor(status.st_dev)), status.st_ino))
        found.add(tuple(identities))
    if len(found) != 1:
        raise ValueError(f"the serving tree names no single checkpoint directory and its files ({len(found)} found)")
    return list(found.pop())


def start_time(pid):
    """A process's start time, in clock ticks after boot (/proc/PID/stat
    field 22), which tells a restarted process or a recycled PID apart."""
    # The command name, field 2, is in parentheses and may hold either.
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


class MappedViews:
    """The mappings the rss_less_mapped_views scope exempts: the read-only
    mappings of the declared files under the served checkpoint directory
    (whose device and inode are those of the regular file the serving
    process resolves, mapped_view_identities, and whose path smaps does not
    mark deleted) that the serving tree holds at the first sample, each
    identified by its process (PID and start time), address range, device,
    inode and offset. Only a shared read-only mapping without Anonymous,
    Private_Dirty, Shared_Dirty or Swap pages, holding no more Rss than its
    Size, is eligible. An exempt mapping must stay present and eligible, with
    its first sample's permissions, in every sample: one a sample lacks or
    holds otherwise is retired for the rest of the soak, so a later mapping
    with the same identity counts. A first sample that cannot identify them
    leaves every later sample without the evidence."""

    def __init__(self, files):
        self.files = files
        self.exempt = None
        self.error = None
        # The declared files' (name, path, device, inode), for the receipt.
        self.identities = None

    @staticmethod
    def eligible(mapping):
        perms = mapping["perms"]
        return (perms[1] == "-" and perms[3] == "s" and not mapping["path"].endswith(" (deleted)")
                and mapping["Rss"] <= mapping["Size"]
                and not any(mapping[key] for key in ("Anonymous", "Private_Dirty", "Shared_Dirty", "Swap")))

    def establish(self, tree):
        identities = mapped_view_identities([pid for pid, _ in tree], self.files)
        files = {(*device, inode) for _, _, device, inode in identities}
        # Each exempt identity with its first sample's permissions.
        exempt = {}
        for (pid, start), mappings in tree.items():
            for mapping in mappings:
                if self.eligible(mapping) and mapping["key"][2:5] in files:
                    exempt[pid, start, *mapping["key"]] = mapping["perms"]
        self.identities, self.exempt = identities, exempt

    def judge(self, tree, rss, unread=None):
        """A sample's judged host memory (VmRSS `rss` less the exempt Rss),
        the exempt Rss and their bound (the exempt mappings' Size), in MiB,
        and the exempt mappings themselves (identity, permissions, and Size,
        Rss, Anonymous, Private_Dirty, Shared_Dirty and Swap in kB), from the
        tree's mappings ({(pid, start time): smaps}).
        None beside `host_exempt_error` when a process's mappings were
        `unread`, or the first sample could not identify the exempt ones; an
        unread process's exempt mappings are retired."""
        if self.exempt is None and self.error is None:
            try:
                if unread:
                    raise ValueError(f"no mappings of {unread}")
                self.establish(tree)
            except (OSError, ValueError) as error:
                self.error = f"the first sample identified no exempt mappings: {error}"
        present = {(pid, start, *mapping["key"]): mapping for (pid, start), mappings in tree.items()
                   for mapping in mappings if self.eligible(mapping)}
        if self.exempt is not None:
            self.exempt = {identity: perms for identity, perms in self.exempt.items()
                           if identity in present and present[identity]["perms"] == perms}
        if error := self.error or (unread and f"no mappings of {unread}"):
            return {"host_judged_mib": None, "host_exempt_rss_mib": None, "host_exempt_bound_mib": None,
                    "host_exempt_mappings": None, "host_exempt_error": error}
        listed = [{"pid": pid, "start_time": born, "range": [start, end], "device": [major, minor], "inode": inode,
                   "offset": offset, "perms": mapping["perms"], "size_kib": mapping["Size"], "rss_kib": mapping["Rss"],
                   "anonymous_kib": mapping["Anonymous"], "private_dirty_kib": mapping["Private_Dirty"],
                   "shared_dirty_kib": mapping["Shared_Dirty"], "swap_kib": mapping["Swap"]}
                  for identity in sorted(self.exempt) for mapping in [present[identity]]
                  for pid, born, start, end, major, minor, inode, offset in [identity]]
        exempt = sum(entry["rss_kib"] for entry in listed) / 1024
        return {"host_judged_mib": rss - exempt, "host_exempt_rss_mib": exempt,
                "host_exempt_bound_mib": sum(entry["size_kib"] for entry in listed) / 1024,
                "host_exempt_mappings": listed}


def process_memory(pid, mapped):
    """A process's /proc/PID/status and, when `mapped`, its (start time,
    mappings), or None beside why they cannot be read. Raises
    FileNotFoundError or ProcessLookupError when the process is gone."""
    status = Path(f"/proc/{pid}/status").read_text()
    if not mapped:
        return status, None, None
    try:
        return status, (start_time(pid), smaps(pid)), None
    except (FileNotFoundError, ProcessLookupError):
        raise
    except (OSError, ValueError) as error:
        return status, None, f"process {pid} ({type(error).__name__}: {error})"


def host_memory(root_pid, views=None):
    """Resident memory of the serving process and its current children, in
    MiB: VmRSS (`host_rss_mib`), which the memory rule judges, and its
    anonymous, file-backed and shared-memory parts (RssAnon, RssFile,
    RssShmem), which are recorded only. A part some process does not report
    is None. Under the
    rss_less_mapped_views scope (`views`, the soak's MappedViews) every
    process's smaps is read too, and the sample adds what that scope judges
    (MappedViews.judge); a process whose smaps cannot be read leaves those
    values None."""
    fields = {"VmRSS": "host_rss_mib", "RssAnon": "host_rss_anon_mib", "RssFile": "host_rss_file_mib",
              "RssShmem": "host_rss_shmem_mib"}
    totals = dict.fromkeys(fields.values(), 0)
    tree, unread = {}, None
    for pid in process_tree(root_pid):
        try:
            status, listed, error = process_memory(pid, views is not None)
        except (FileNotFoundError, ProcessLookupError):
            if pid == root_pid:
                raise RuntimeError("serving process exited during memory sampling") from None
            continue
        start, mappings = listed or (None, [])
        tree[pid, start] = mappings
        unread = unread or error
        for field, key in fields.items():
            match = re.search(rf"^{field}:\s+(\d+) kB$", status, re.MULTILINE)
            if match is None and field == "VmRSS":
                raise RuntimeError(f"no resident memory sample for serving process {pid}")
            totals[key] = None if match is None or totals[key] is None else totals[key] + int(match[1])
    sample = {key: None if total is None else total / 1024 for key, total in totals.items()}
    if views is not None:
        sample.update(views.judge(tree, sample["host_rss_mib"], unread))
    return sample


def mapped_views(client):
    """The host's mapped-view gauges when its /metrics serves them: the bytes
    its mapped views cover and, while residency is visible, the bytes of them
    resident. They are recorded only: a scrape that fails in transport,
    decoding or parsing, or a gauge that is not a finite whole number of
    bytes, records `mapped_view_error` beside the memory sample and never
    fails the soak."""
    try:
        text = metrics_text(client)
        views = {}
        for name in ("mapped_view_bytes", "mapped_view_resident_bytes"):
            raw = next((line.split()[1] for line in text.splitlines() if line.startswith(f"chop_{name} ")), None)
            if raw is None:
                continue
            value = float(raw)
            if not (math.isfinite(value) and value >= 0 and value.is_integer()):
                raise ValueError(f"chop_{name} is not a finite whole number of bytes: {raw}")
            views[name] = int(value)
        return views
    except Exception as error:
        return {"mapped_view_error": f"{type(error).__name__}: {error}"}


def memory_sample(root_pid, started, client, views=None):
    return {"at": time.time(), "elapsed_seconds": time.monotonic() - started,
            "mib": device_memory(root_pid), **host_memory(root_pid, views), **mapped_views(client)}


def main():
    parser = arguments(__doc__, model_facts=lambda args: args.media_mix or args.structured_mix)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--memory-growth-mib", type=int, default=512)
    parser.add_argument("--host-memory-growth-mib", type=int, default=512)
    parser.add_argument("--prefix-mix", action="store_true",
                        help="mix prefix-cache hits and misses and check them")
    parser.add_argument("--media-mix", action="store_true",
                        help="send an eighth of requests with image, audio or video parts")
    parser.add_argument("--media-cases", type=lambda value: value.split(","), default=list(MEDIA_CASES),
                        help=f"comma-separated media mix cases, default {','.join(MEDIA_CASES)}")
    parser.add_argument("--structured-mix", action="store_true",
                        help="send an eighth of requests with a strict JSON schema")
    parser.add_argument("--remote-only", action="store_true",
                        help="HTTP load only; do not inspect processes or GPU memory")
    parser.add_argument("--server-pid", type=int,
                        help="local serving process PID (auto-detected from --url by default)")
    parser.add_argument("--warmup-seconds", type=int,
                        help="seconds before the steady memory baseline (default: the declared WARMUP_SECONDS)")
    parser.add_argument("--cache-allowance-mib", type=lambda value: dict(zip(("host", "device"), map(int, value.split(",")))),
                        help="HOST,DEVICE MiB of declared bounded cache growth (default: CACHE_ALLOWANCE_MIB)")
    parser.add_argument("--host-memory-scope",
                        help="rss or rss_less_mapped_views: the host memory judged (default: the declared "
                             "HOST_MEMORY_SCOPE)")
    parser.add_argument("--mapped-view-file", action="append", dest="mapped_view_files", metavar="NAME",
                        help="a checkpoint-relative file whose read-only mappings at the first sample "
                             "rss_less_mapped_views exempts; repeatable (default: the declared MAPPED_VIEW_FILES)")
    args = parser.parse_args()
    if args.duration <= 0 or args.concurrency <= 0:
        parser.error("duration and concurrency must be positive")
    if args.server_pid is not None and args.server_pid <= 0:
        parser.error("server-pid must be positive")
    try:
        warmup, allowance = memory_policy(args.warmup_seconds, args.cache_allowance_mib)
        scope = host_memory_scope(args.host_memory_scope)
        files = mapped_view_files(args.mapped_view_files, scope)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    views = MappedViews(files) if scope == "rss_less_mapped_views" else None
    # Media and structured answers come without reasoning, or after it when
    # the frontend refuses disabled thinking.
    mixed_reasoning = (args.media_mix or args.structured_mix) and not thinking_can_be_disabled()
    no_thinking = answer_control() if args.media_mix or args.structured_mix else {}
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    root_pid = None if args.mechanics_only or args.remote_only else (args.server_pid or serving_pid(args.url))
    duration = min(args.duration, 4) if args.mechanics_only else args.duration
    concurrency = 1 if args.mechanics_only else args.concurrency
    lock = threading.Lock()
    latencies = []
    errors = []
    completed = 0
    memory = []
    load = []
    if unknown := set(args.media_cases) - set(MEDIA_CASES):
        parser.error(f"unknown media cases {sorted(unknown)}")
    # The mix sends only media of modalities the server serves; the others
    # make the soak not applicable as a media check, never passed.
    kinds = {case_id: case_id.split("-", 1)[0] for case_id in args.media_cases}
    inapplicable = sorted({reason for kind in kinds.values()
                           if args.media_mix and (reason := not_applicable(server, kind))})
    media = media_requests([case_id for case_id, kind in kinds.items()
                            if not not_applicable(server, kind)]) if args.media_mix else []
    mixed = {"media": 0, "structured": 0, "structured_truncated": 0}
    kept = {"limit": KEEP_STRUCTURED, "total": 0, "requests": []}

    def keep(*record):
        with lock:
            kept["total"] += 1
            if len(kept["requests"]) < KEEP_STRUCTURED:
                kept["requests"].append(kept_structured(*record))

    canary_salt = f"soak-canary/{uuid.uuid4().hex}"
    canary_ids = tokenize(client, server["model"], CANARY) if args.prefix_mix else []
    canary_anchor = (len(canary_ids) - 1) // BLOCK_TOKENS * BLOCK_TOKENS

    def canary(prompt=None, max_tokens=48):
        body = {"model": server["model"], "prompt": prompt or canary_ids, "temperature": 0,
                "max_tokens": max_tokens, "logprobs": 2, "cache_salt": canary_salt}
        return completion_canary(client.request("/v1/completions", body)[0])

    canaries = []
    if args.prefix_mix:
        canary(primer(canary_ids, canary_anchor), 1)
        canaries.append(canary())

    def worker(worker_id):
        nonlocal completed
        turn = 0
        conversation = []
        while time.monotonic() < deadline:
            prompt = "Hi." if args.mechanics_only else PROMPTS[(worker_id + turn) % len(PROMPTS)]
            messages = [{"role": "user", "content": prompt}]
            if args.prefix_mix:
                messages = prefix_messages(worker_id, turn, conversation)
            body = {"model": server["model"], "messages": messages,
                    "temperature": 0, "max_tokens": 1 if args.mechanics_only else (8, 32, 96)[turn % 3]}
            kind = None
            if media and (worker_id + turn) % 8 == 0:
                kind = "media"
                case_id, messages = media[(worker_id + turn) // 8 % len(media)]
                body = {"model": server["model"], "messages": messages, "temperature": 0,
                        "max_tokens": REASONING_TOKENS if mixed_reasoning else 48,
                        **no_thinking}
            elif args.structured_mix and (worker_id + turn) % 8 == 4:
                kind = "structured"
                body = {"model": server["model"], "max_tokens": REASONING_TOKENS if mixed_reasoning else 512,
                        "messages": [{"role": "user", "content": PERSON_PROMPT}],
                        **no_thinking,
                        "response_format": STRUCTURED_FORMAT}
            try:
                value, latency = client.request("/v1/chat/completions", body)
                if not value.get("choices"):
                    raise ValueError("missing choices")
                truncated = False
                if kind == "media":
                    check_media(case_id, value["choices"][0])
                elif kind == "structured":
                    record = worker_id, turn, time.monotonic() - started, latency, body, value
                    try:
                        truncated = check_structured(value["choices"][0])
                    except ValueError as error:
                        keep(*record, str(error) if str(error).startswith("schema") else f"parse: {error}")
                        raise
                    if truncated:
                        keep(*record, "truncated")
                if kind:
                    with lock:
                        mixed[kind] += 1
                        mixed["structured_truncated"] += truncated
                elif args.prefix_mix and worker_id % 3 == 0:
                    conversation.append({"role": "assistant",
                                         "content": value["choices"][0]["message"].get("content") or ""})
                    conversation.append({"role": "user", "content": f"Continue, turn {turn + 1}."})
                with lock:
                    completed += 1
                    latencies.append(latency)
            except Exception as error:
                with lock:
                    errors.append({"worker": worker_id, "error": str(error)})
            turn += 1

    started = time.monotonic()
    deadline = started + duration

    def sample():
        load.append(host_load_sample())
        if not args.mechanics_only and not args.remote_only:
            try:
                memory.append(memory_sample(root_pid, started, client, views))
            except Exception as error:
                errors.append({"memory": str(error)})

    sample()
    with ThreadPoolExecutor(max_workers=concurrency + 1) as pool:
        futures = [pool.submit(worker, index) for index in range(concurrency)]
        while time.monotonic() < deadline:
            time.sleep(min(30, max(0, deadline - time.monotonic())))
            sample()
        measured_duration = time.monotonic() - started
        for future in futures:
            future.result()
    # The timed work ends with the last request in flight at the deadline.
    load.append(host_load_sample())
    growth = host_growth = None
    rule = {}
    if memory:
        # A sample the declared scope cannot judge fails the soak but keeps
        # its receipt.
        try:
            rule = memory_rule(memory, duration,
                               {"device": args.memory_growth_mib, "host": args.host_memory_growth_mib},
                               warmup, allowance, scope)
            growth, host_growth = rule["memory_growth_mib"], rule["host_memory_growth_mib"]
        except ValueError as error:
            errors.append({"memory": str(error)})
    prefix = None
    if args.prefix_mix:
        canaries.extend([canary(), canary()])
        time.sleep(1)
        text = metrics_text(client)
        prefix = {"canaries": canaries, "canary_salt": canary_salt,
                  **check_canaries(canaries, canary_anchor),
                  "canaries_match": len({canary["content"] for canary in canaries}) == 1,
                  "idle_blocks_used": metric(text, "chop_kv_blocks_used"),
                  "idle_active_sequences": metric(text, "chop_kv_active_sequences"),
                  **{name: metric(text, f"chop_prefix_cache_{name}")
                     for name in ("hits_total", "reused_tokens_total", "captures_total",
                                  "capture_failures_total", "restore_failures_total",
                                  "evictions_total", "anchors", "blocks",
                                  "capture_microseconds_total", "restore_microseconds_total")}}
        if not prefix["canary_passed"]:
            errors.append({"prefix": "canary restore error or no canary hit"})
        if not (prefix["idle_blocks_used"] == 0 and prefix["idle_active_sequences"] == 0
                and prefix["hits_total"]):
            errors.append({"prefix": "live pages after idle, or no hits"})
    passed = completed > 0 and not errors and (args.mechanics_only or args.remote_only or (len(memory) >= 2 and rule["passed"]))
    receipt("soak", server, {"passed": passed, "skipped": ["C32 and device memory sampling"] if args.mechanics_only else [],
                            "duration_seconds": duration, "measured_duration_seconds": measured_duration,
                            "concurrency": concurrency,
                            "completed": completed, "errors": errors,
                            "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
                            "latency_max_seconds": max(latencies) if latencies else None,
                            "memory_samples": memory, "memory_growth_mib": growth,
                            "host_memory_growth_mib": host_growth,
                            "host_memory_growth_limit_mib": args.host_memory_growth_mib,
                            "memory_scope": "not_sampled_remote" if args.remote_only else "serving_process_tree", "server_pid": root_pid,
                            "memory_growth_limit_mib": args.memory_growth_mib,
                            **({key: rule[key] for key in ("warmup_seconds", "effective_warmup_seconds",
                                                           "steady_baseline_seconds", "cache_allowance_mib",
                                                           "steady_memory_growth_mib", "steady_host_memory_growth_mib")}
                               if rule else {}),
                            # Recorded only under a scope other than the default.
                            **({"host_memory_scope": scope} if scope != "rss" else {}),
                            **({"mapped_view_files": files,
                                "mapped_view_file_identities": None if views.identities is None else [
                                    {"file": name, "path": path, "device": list(device), "inode": inode}
                                    for name, path, device, inode in views.identities]}
                               if views is not None else {}),
                            **({key: rule[key] for key in ("host_exempt_rss_mib", "host_exempt_bound_mib")
                                if key in rule}),
                            "prefix_mix": prefix,
                            "mixed_requests": mixed if media or args.structured_mix else None,
                            "structured_kept": kept if args.structured_mix else None,
                            "media_cases": [case_id for case_id, _ in media] if media else None,
                            **({"not_applicable": inapplicable} if inapplicable else {})},
            args.mechanics_only, load_samples=load)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
