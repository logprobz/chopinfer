"""The reference cache: sealed entries, refsets and fixture bindings.

Layout under $CHOPINFER_DATA/refcache/v1:

    entries/<k[0:2]>/<key_sha256>/
        key.json      canonical key document; its sha256 is key_sha256
        scores.json   canonical scores of the run entry
        entry.json    schema, key_sha256, byte length and sha256 of each file
                      above, and provenance (not keyed)
    refsets/<refset_sha256>.json
    stamps/           checkpoint stamp cache (stamps.py)
    tmp/              entries and refsets are written here, then renamed
    keep              optional: one git ref per line whose fixtures gc keeps

Writes are create-only: an entry appears whole by one rename or not at all,
and an existing entry is never overwritten. A second producer of a key must
produce the same scores.json bytes or its write fails (`Nondeterministic`),
which makes every repeated run a determinism check.

A refset names the entries one assembled fixture was scored from: per mode,
the ordered (sequence name, key_sha256) pairs, with the model, the workload's
sha256 and the fixture's sha256 without its binding block.

A fixture's `binding` block (`bind`) carries the refset digest, every
entry's key, the shared key sections with their sub-digests (lock,
reference code, adapter, mode policy, environment), and the sequence
sections. Reference code labelled `models/<model>/<file>` is repository
code the adapter executes, checked against the working tree like the adapter
files. `verify` recomputes what the working tree allows and raises
`StaleReference` naming the first differing field; a fixture without a
binding is reported as unbound. With a cache present it also checks the
fixture's scores against the entries.

This module does not import torch.
"""

import errno
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from tools.model_sdk.diagnostic import canonical, decode_json
from tools.refrun import keys, modes, workload

ENTRY = "chopinfer.refcache.entry.v1"
REFSET = "chopinfer.refcache.refset.v1"
BINDING = "chopinfer.refcache.binding.v1"
FILES = ("key.json", "scores.json")
AGE_FLOOR = 14 * 86400
DIGEST = re.compile(r"[0-9a-f]{64}")
# Key sections every entry of one binding shares.
SHARED = ("checkpoint", "reference_code", "adapter", "environment", "outputs")
# Reference-code labels naming repository files, checked against the working tree.
REPOSITORY_CODE = "models/"
# `mode_policies()`, for verifiers that cannot run modes.py (chopinfer eval).
MODE_POLICIES = Path(__file__).with_name("mode-policies.json")


class StaleReference(ValueError):
    """A bound fixture no longer matches what produced it; `field` names the
    first differing part."""

    def __init__(self, field, detail):
        super().__init__(f"stale_reference: {field}: {detail}")
        self.field = field


class Nondeterministic(RuntimeError):
    """A second producer of a key wrote different scores."""


def default_root():
    data = os.environ.get("CHOPINFER_DATA")
    return Path(data) / "refcache/v1" if data else None


def mode_policies():
    """Each mode's default dequant rule and the policy digest modes.py gives
    under every rule the mode accepts: what `verify` recomputes for a bound
    mode. It lives here, not in modes.py, whose bytes every key holds."""
    result = {}
    for name, mode in modes.MODES.items():
        digests = {}
        for rule in modes.DEQUANT:
            try:
                digests[rule] = keys.mode(modes.policy(name, rule))["policy_sha256"]
            except ValueError:
                continue
        result[name] = {"dequant": mode["dequant"], "policy_sha256": digests}
    return result


def encoded_mode_policies():
    return json.dumps(mode_policies(), indent=2) + "\n"


def digest(value):
    return keys.sha256(canonical(value))


def checked(key):
    if not isinstance(key, str) or not DIGEST.fullmatch(key):
        raise ValueError(f"not a sha256 digest: {key!r}")
    return key


def seal(data):
    return {"bytes": len(data), "sha256": keys.sha256(data)}


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def entry_path(self, key):
        return self.root / "entries" / checked(key)[:2] / key

    def refset_path(self, refset_sha256):
        return self.root / "refsets" / f"{checked(refset_sha256)}.json"

    def staging(self):
        (self.root / "tmp").mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(dir=self.root / "tmp"))

    def read(self, key):
        """{file: bytes} and the parsed entry.json of a sealed entry, or None
        when absent. Files that differ from the seal are refused."""
        path = self.entry_path(key)
        if not path.exists():
            return None
        entry = decode_json((path / "entry.json").read_bytes())
        data = {name: (path / name).read_bytes() for name in FILES}
        if entry.get("schema") != ENTRY or entry.get("key_sha256") != key \
                or entry.get("files") != {name: seal(raw) for name, raw in data.items()} \
                or keys.sha256(data["key.json"]) != key:
            raise ValueError(f"cache entry {path} differs from its seal")
        return data, entry

    def get(self, key):
        """The entry's key document, scores and entry.json, or None."""
        found = self.read(key)
        if found is None:
            return None
        data, entry = found
        return {"key": decode_json(data["key.json"]), "scores": decode_json(data["scores.json"]),
                "entry": entry}

    def put(self, document, scores, provenance=None):
        """Store one entry and return its key. An entry already present must
        hold the same scores bytes."""
        key = keys.key_sha256(document)
        data = {"key.json": canonical(document), "scores.json": canonical(scores)}
        found = self.read(key)
        if found is None:
            staging = self.staging()
            entry = {"schema": ENTRY, "key_sha256": key, "files": {n: seal(raw) for n, raw in data.items()},
                     "provenance": provenance or {}}
            target = self.entry_path(key)
            try:
                for name, raw in [*data.items(), ("entry.json", canonical(entry))]:
                    write_synced(staging / name, raw)
                os.chmod(staging, 0o755)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(staging, target)
                return key
            except OSError as error:
                shutil.rmtree(staging)
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            found = self.read(key)
        if found[0]["scores.json"] != data["scores.json"]:
            raise Nondeterministic(f"cache entry {key}: this run's scores.json differs from the stored one")
        return key

    def put_refset(self, document):
        """Store a refset create-only and return its digest."""
        refset_sha256 = digest(document)
        target = self.refset_path(refset_sha256)
        if not target.exists():
            staging = self.staging()
            write_synced(staging / "refset.json", canonical(document))
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staging / "refset.json", target)
            except FileExistsError:
                pass
            finally:
                shutil.rmtree(staging)
        if self.refset(refset_sha256) != document:
            raise ValueError(f"refset {target} differs from its digest")
        return refset_sha256

    def refset(self, refset_sha256):
        path = self.refset_path(refset_sha256)
        if not path.exists():
            return None
        data = path.read_bytes()
        if keys.sha256(data) != refset_sha256:
            raise ValueError(f"refset {path} differs from its digest")
        return decode_json(data)


def write_synced(path, data):
    with open(path, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def fixture_sha256(value):
    """The digest of a fixture without its binding block."""
    return digest({name: item for name, item in value.items() if name != "binding"})


def refset(model, workload_sha256, entries, assembled_fixture_sha256):
    """`entries` maps mode to ordered [name, key_sha256] pairs."""
    return {"schema": REFSET, "model": model, "workload_sha256": workload_sha256,
            "modes": {mode: [list(pair) for pair in pairs] for mode, pairs in entries.items()},
            "assembled_fixture_sha256": assembled_fixture_sha256}


def workload_sequences(path):
    """A workload file's model and its ordered [name, sequence section] pairs."""
    model, records = workload.load(path)
    return model, [[record.name, keys.sequence(record)] for record in records]


def workload_sha256(path):
    """The digest of a workload's model and ordered sequence sections: what
    the keys depend on, not the file's bytes."""
    model, sequences = workload_sequences(path)
    return digest({"model": model, "sequences": sequences})


def check_workload(binding, path):
    """The binding scores exactly the workload at `path`: its digest, and in
    every mode the forced entries of the workload's sequences in order, each
    generated answer's greedy entry first."""
    model, sequences = workload_sequences(path)
    if digest({"model": model, "sequences": sequences}) != binding["workload_sha256"] or model != binding["model"]:
        raise StaleReference("workload", f"{path} is not the bound workload")
    greedy = [[name, section] for name, section in sequences if section["answer"]["policy"] == "greedy"]
    for mode, entries in binding["modes"].items():
        listed = [[name, binding["sequences"][index]] for name, index, _ in entries["entries"]]
        generated = [pair for pair in listed if pair[1]["answer"]["policy"] == "greedy"]
        forced = [pair for pair in listed if pair[1]["answer"]["policy"] == "forced"]
        if generated not in ([], greedy) or listed != generated + forced or len(forced) != len(sequences):
            raise StaleReference("workload", f"mode {mode} does not score the sequences of {path} in order")
        for (name, bound), (expected, section) in zip(forced, sequences):
            if section["answer"]["policy"] == "greedy":
                bound = {**bound, "answer": section["answer"]}
            if name != expected or bound != section:
                raise StaleReference("workload", f"mode {mode}: {name} is not {expected} of {path}")
    if greedy and not any(entries["entries"] and binding["sequences"][entries["entries"][0][1]]["answer"]["policy"]
                          == "greedy" for entries in binding["modes"].values()):
        raise StaleReference("workload", f"no mode generates the answers {path} leaves to the model")


def bind(value, runs, workload_path, lock_path, lock_variant=None, variants=None, store=None):
    """`value` with a binding block for the run entries it was assembled from.

    `runs` maps mode to that mode's run document (the runner's <mode>.json),
    `workload_path` is the workload the runs scored, `lock_path` is the repo
    path of the checkpoint lock and `variants` maps a mode to the fixture
    variant holding its scores (default: the mode name). The refset is
    written to `store` when one is given."""
    workload_digest = workload_sha256(workload_path)
    variants = variants or {}
    sections, sequences, bound = None, [], {}
    for mode, run in runs.items():
        listed = []
        for entry in run["entries"]:
            document = entry["key"]
            if keys.key_sha256(document) != entry["key_sha256"]:
                raise ValueError(f"{mode}/{entry['name']}: key_sha256 is not the key document's digest")
            shared = {name: document[name] for name in ("model", *SHARED)}
            if sections is None:
                sections = shared
            elif shared != sections:
                raise ValueError(f"{mode}/{entry['name']}: entries of one binding differ outside sequence and mode")
            if document["mode"] != keys.mode(run["policy"]):
                raise ValueError(f"{mode}/{entry['name']}: key mode is not the run's policy")
            if document["sequence"] not in sequences:
                sequences.append(document["sequence"])
            listed.append([entry["name"], sequences.index(document["sequence"]), entry["key_sha256"]])
        bound[mode] = {"mode": run["policy"]["name"], "variant": variants.get(mode, mode),
                       "dequant": run["policy"]["dequant"],
                       "policy_sha256": keys.mode(run["policy"])["policy_sha256"], "entries": listed}
    if sections is None:
        raise ValueError("a binding needs at least one run entry")
    model = sections.pop("model")
    assembled = fixture_sha256(value)
    document = refset(model, workload_digest, pairs(bound), assembled)
    probe = {"model": model, "modes": bound, "sequences": sequences, "workload_sha256": workload_digest}
    check_workload(probe, workload_path)
    check_sequences(value, probe)
    run_scores = {entry["key_sha256"]: entry["scores"] for run in runs.values() for entry in run["entries"]}
    differing = scores_difference(value, probe, run_scores.get)
    if differing is not None:
        raise ValueError(f"{differing[0]}: the fixture's {differing[1]} differs from the run's")
    refset_sha256 = store.put_refset(document) if store is not None else digest(document)
    binding = {
        "schema": BINDING, "model": model, "refset_sha256": refset_sha256,
        "assembled_fixture_sha256": assembled, "workload_sha256": workload_digest,
        "lock": {"path": lock_path, "variant": lock_variant},
        "digests": {"lock": sections["checkpoint"]["source_lock_sha256"],
                    "reference_code": digest(sections["reference_code"]), "adapter": digest(sections["adapter"]),
                    "mode_policy": {m: b["policy_sha256"] for m, b in bound.items()},
                    "environment": digest(sections["environment"])},
        "key": sections, "sequences": sequences, "modes": bound,
    }
    return {**value, "binding": binding}


def items(value):
    """A fixture's scored items by name: `texts` by name, `cases` by id or
    the `probes` mapping."""
    if isinstance(value.get("texts"), list):
        return {item["name"]: item for item in value["texts"]}
    if isinstance(value.get("cases"), list):
        return {item["id"]: item for item in value["cases"]}
    if isinstance(value.get("probes"), dict):
        return value["probes"]
    return {}


TOPK = ("topk_ids", "topk_logits", "topk_logprobs")


def score_difference(fixture, scores):
    """The first field the fixture copies from scores.json with another value,
    or None. Top-k rows may be cut to a prefix."""
    for field, stated in fixture.items():
        if field not in scores:
            continue
        actual = scores[field]
        if field in TOPK and isinstance(stated, list) and len(stated) == len(actual):
            actual = [row[:len(mine)] for row, mine in zip(actual, stated)]
        if canonical(stated) != canonical(actual):
            return field
    return None


def unbound():
    return {"status": "unbound", "refset_sha256": None}


def verify(value, root=keys.ROOT, checkpoint=None, store=None, workload=None, lock_root=None):
    """Check a fixture's binding against itself, the working tree under
    `root`, the checkpoint directory (reference code, when given), the
    workload file it scored (when given) and the cache `store` (scores, when
    given). The checkpoint lock it names is read under `lock_root` (default
    `root`): an external variant keeps its lock outside the tree. Returns
    the binding's status:
    "unbound" for a fixture without one, else "bound" with the refset digest
    and the cache comparison. The most specific difference is named first:
    the binding itself, then the working tree, the sequences, the cached
    scores and last the fixture's own digest."""
    binding = value.get("binding")
    if binding is None:
        return unbound()
    if not isinstance(binding, dict) or binding.get("schema") != BINDING:
        raise StaleReference("schema", f"binding schema is not {BINDING}")
    sections, digests, bound = binding["key"], binding["digests"], binding["modes"]

    # The binding agrees with itself.
    stated = {"lock": sections["checkpoint"]["source_lock_sha256"],
              "reference_code": digest(sections["reference_code"]), "adapter": digest(sections["adapter"]),
              "mode_policy": {m: b["policy_sha256"] for m, b in bound.items()},
              "environment": digest(sections["environment"])}
    for field, expected in stated.items():
        if digests.get(field) != expected:
            raise StaleReference(field, "the binding's sub-digest differs from its key section")
    for mode, entries in bound.items():
        for name, index, key in entries["entries"]:
            rebuilt = keys.document(model=binding["model"], sequence=binding["sequences"][index],
                                    mode={"name": entries["mode"], "policy_sha256": entries["policy_sha256"]},
                                    **sections)
            if keys.key_sha256(rebuilt) != key:
                raise StaleReference("key_sha256", f"{mode}/{name} is not its key document's digest")
    document = refset(binding["model"], binding["workload_sha256"], pairs(bound),
                      binding["assembled_fixture_sha256"])
    if digest(document) != binding["refset_sha256"]:
        raise StaleReference("refset_sha256", "the bound entries do not form the bound refset")

    # The working tree still produces the bound inputs.
    lock = keys.read_lock(Path(root if lock_root is None else lock_root) / binding["lock"]["path"])
    if binding["lock"]["variant"] is not None:
        lock = lock[binding["lock"]["variant"]]
    if keys.checkpoint(lock, [])["source_lock_sha256"] != digests["lock"]:
        raise StaleReference("lock", f"{binding['lock']['path']} differs from the bound checkpoint lock")
    files = sections["adapter"]["files"]
    for path in sorted(set(files) | set(keys.NUMERICS)):
        actual = keys.file_sha256(Path(root) / path) if (Path(root) / path).exists() else None
        if files.get(path) != actual:
            raise StaleReference("adapter", f"{path} differs from the bound adapter and numerics files")
    # Repository code an adapter executes beyond its own file (another
    # model file it loads), labelled by its repository path under models/.
    for label, expected in sections["reference_code"]["files"].items():
        if label.startswith(REPOSITORY_CODE):
            path = Path(root) / label
            if not path.is_file() or keys.file_sha256(path) != expected:
                raise StaleReference("reference_code", f"{label} differs from the bound repository code")
    for mode, entries in bound.items():
        if keys.mode(modes.policy(entries["mode"], entries["dequant"]))["policy_sha256"] != entries["policy_sha256"]:
            raise StaleReference("mode_policy", f"modes.py no longer gives the bound {mode} policy")
    if checkpoint is not None:
        for label, expected in sections["reference_code"]["files"].items():
            owner, _, file = label.partition("/")
            path = Path(checkpoint) / file
            # Labels of package files (transformers/...) are checked by key only.
            if owner == binding["model"] and (not path.exists() or keys.file_sha256(path) != expected):
                raise StaleReference("reference_code", f"{path} differs from the bound reference code")

    if workload is not None:
        check_workload(binding, workload)
    check_sequences(value, binding)
    result = {"status": "bound", "refset_sha256": binding["refset_sha256"], "cache": None}
    if store is not None:
        result["cache"] = compare(value, binding, store)
    assembled = fixture_sha256(value)
    if assembled != binding["assembled_fixture_sha256"]:
        raise StaleReference("assembled_fixture", f"the fixture's content has sha256 {assembled}, "
                             f"not the bound {binding['assembled_fixture_sha256']}")
    return result


def pairs(bound):
    return {mode: [[name, key] for name, _, key in entries["entries"]] for mode, entries in bound.items()}


def check_sequences(value, binding):
    """Every bound sequence is a fixture item stating the bound token IDs,
    score_from and forced answer (each where the item states it)."""
    stated = items(value)
    for entries in binding["modes"].values():
        for name, index, _ in entries["entries"]:
            sequence = binding["sequences"][index]
            item = stated.get(name)
            if item is None:
                raise StaleReference("sequence", f"{name} is bound but not in the fixture")
            answer = sequence["answer"]
            if answer["policy"] == "greedy" and "answer_token_ids" not in item:
                raise StaleReference("sequence", f"{name}: a generated answer is bound but the fixture states none")
            if "prompt_token_ids" in item and keys.ids_sha256(item["prompt_token_ids"]) \
                    != sequence["token_ids_sha256"]:
                raise StaleReference("sequence", f"{name}: token IDs differ from the bound sequence")
            if "score_from" in item and item["score_from"] != sequence["score_from"]:
                raise StaleReference("sequence", f"{name}: score_from differs from the bound sequence")
            if "answer_token_ids" in item and answer["policy"] == "forced" \
                    and keys.ids_sha256(item["answer_token_ids"]) != answer["ids_sha256"]:
                raise StaleReference("sequence", f"{name}: answer token IDs differ from the bound sequence")


def scores_difference(value, binding, scores):
    """(mode/name, field, key) of the first copied score that differs from
    `scores(key)`, skipping keys it returns None for, or None. A forced
    entry's scores are copied under the mode's variant; a greedy entry's
    answer is the item's `answer_token_ids` and its step scores are copied
    under `greedy`."""
    by_name = items(value)
    for mode, entries in binding["modes"].items():
        for name, index, key in entries["entries"]:
            actual = scores(key)
            if actual is None:
                continue
            item = by_name[name]
            if binding["sequences"][index]["answer"]["policy"] == "greedy":
                if item.get("answer_token_ids") != actual["answer"]:
                    return f"{mode}/{name}", "answer", key
                fixture = item.get("greedy")
            else:
                fixture = item.get(entries["variant"])
            if isinstance(fixture, dict):
                field = score_difference(fixture, actual)
                if field is not None:
                    return f"{mode}/{name}", field, key
    return None


def compare(value, binding, store):
    """The fixture's scores against the cache entries: each scores field the
    fixture copies must equal the entry's. Entries absent from this cache are
    counted, not refused."""
    found = {}
    for entries in binding["modes"].values():
        for _, _, key in entries["entries"]:
            entry = store.get(key)
            found[key] = None if entry is None else entry["scores"]
    differing = scores_difference(value, binding, found.get)
    if differing is not None:
        raise StaleReference("scores", f"{differing[0]}: {differing[1]} differs from cache entry {differing[2]}")
    return {"root": str(store.root), "refset_present": store.refset(binding["refset_sha256"]) is not None,
            "entries": len(found), "missing": sum(scores is None for scores in found.values())}


def bound_keys(binding, where):
    """The entry keys a tracked binding reaches. A binding gc cannot read is
    refused, never skipped: gc would then delete what it binds."""
    try:
        keys_reached = {key for entries in binding["modes"].values() for _, _, key in entries["entries"]}
        if binding["schema"] != BINDING or not all(map(DIGEST.fullmatch, [binding["refset_sha256"], *keys_reached])):
            raise ValueError("not a binding")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{where}: cannot read its reference binding ({error!r}); refusing to collect") from None
    return keys_reached


def bound_refsets(repo, refs):
    """{refset_sha256: entry keys} of every JSON fixture under fixtures/ at
    each git ref that names the binding schema. Such a fixture must decode
    and hold a readable top-level binding, or gc refuses to run."""
    found = {}
    for ref in refs:
        listed = subprocess.run(["git", "grep", "-l", "--fixed-strings", BINDING, ref, "--",
                                 ":(glob)fixtures/**/*.json"], cwd=repo, capture_output=True, text=True)
        if listed.returncode not in (0, 1):
            raise RuntimeError(f"git grep at {ref}: {listed.stderr.strip()}")
        for line in listed.stdout.splitlines():
            shown = subprocess.run(["git", "show", line], cwd=repo, capture_output=True, check=True)
            try:
                value = decode_json(shown.stdout)
            except ValueError as error:
                raise ValueError(f"{line}: not JSON ({error}); refusing to collect") from None
            binding = value.get("binding") if isinstance(value, dict) else None
            reached = bound_keys(binding, line)
            found.setdefault(binding["refset_sha256"], set()).update(reached)
    return found


def keep_refs(store, extra=()):
    """main, the keep file's refs and `extra`, each once. Named refs only add
    to the refs whose fixtures keep their entries."""
    path = store.root / "keep"
    listed = path.read_text().split() if path.exists() else []
    return list(dict.fromkeys(["main", *listed, *extra]))


def gc(store, repo=keys.ROOT, refs=(), now=None, delete=False):
    """Entries and refsets reachable from no refset bound by a fixture at
    main, a ref in the keep file or one of `refs`, and older than AGE_FLOOR.
    Removed only with `delete`; stamps are never touched."""
    refs = keep_refs(store, refs)
    now = time.time() if now is None else now
    bindings = bound_refsets(repo, refs)
    reachable = set().union(*bindings.values())

    def old(path):
        return now - path.stat().st_mtime >= AGE_FLOOR

    removed = {"entries": [], "refsets": [], "tmp": []}
    kept = {"entries": 0, "refsets": 0}
    for path in sorted(store.root.glob("entries/*/*")):
        # entry.json is written last, so its mtime is the entry's creation.
        if path.name in reachable or not old(path / "entry.json"):
            kept["entries"] += 1
        else:
            removed["entries"].append(path)
    for path in sorted(store.root.glob("refsets/*.json")):
        if path.stem in bindings or not old(path):
            kept["refsets"] += 1
        else:
            removed["refsets"].append(path)
    removed["tmp"] = [path for path in sorted(store.root.glob("tmp/*")) if old(path)]
    if delete:
        for path in [*removed["entries"], *removed["refsets"], *removed["tmp"]]:
            if path.is_dir():
                # Out of entries/ first, so a reader never sees half an entry.
                trash = store.staging()
                os.rename(path, trash / path.name)
                shutil.rmtree(trash)
            else:
                path.unlink()
    return {"refs": refs, "bound_refsets": sorted(bindings), "reachable_entries": len(reachable),
            "kept": kept, "removed": {kind: [str(p) for p in paths] for kind, paths in removed.items()},
            "deleted": delete}


def score_delta(a, b):
    """How the scores of two entries of one sequence differ."""
    if a.get("token") != b.get("token") or a.get("score_from") != b.get("score_from"):
        return {"bitwise_equal": False, "tokens_differ": True}
    delta = {"bitwise_equal": canonical(a) == canonical(b), "positions": len(a["token"]),
             "top1_changed": sum(x != y for x, y in zip(a["top1"], b["top1"])),
             "logprob_max_abs": max((abs(x - y) for x, y in zip(a["logprob"], b["logprob"])), default=0.0),
             "margin_max_abs": max((abs(x - y) for x, y in zip(a["margin"], b["margin"])), default=0.0)}
    if "answer" in a:
        delta["answer_equal"] = a["answer"] == b.get("answer")
    return delta


def labelled(listed):
    """Refset pairs by name; a name's later entries (a greedy sequence's
    forced pass) are labelled name#2, name#3."""
    result = {}
    for name, key in listed:
        label, count = name, 1
        while label in result:
            count += 1
            label = f"{name}#{count}"
        result[label] = key
    return result


def diff_entries(store, a, b):
    """The key sections and scores two entries differ in, or, for two
    refsets, every pair of entries matched by mode and sequence name."""
    if store.refset(a) is not None and store.refset(b) is not None:
        left, right = store.refset(a), store.refset(b)
        pairs = {}
        for mode in sorted(set(left["modes"]) | set(right["modes"])):
            mine, theirs = labelled(left["modes"].get(mode, [])), labelled(right["modes"].get(mode, []))
            for name in [*mine, *[n for n in theirs if n not in mine]]:
                if name in mine and name in theirs:
                    pairs[f"{mode}/{name}"] = diff_entries(store, mine[name], theirs[name])
                else:
                    pairs[f"{mode}/{name}"] = {"only_in": a if name in mine else b}
        return {"refsets": [a, b], "entries": pairs}
    found = [store.get(key) for key in (a, b)]
    for key, entry in zip((a, b), found):
        if entry is None:
            raise ValueError(f"{key} is neither a cached entry nor (with the other) a refset")
    left, right = (entry["key"] for entry in found)
    sections = sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
    return {"keys": [a, b], "key_sections_differ": sections,
            "scores": score_delta(found[0]["scores"], found[1]["scores"])}


if __name__ == "__main__":
    MODE_POLICIES.write_text(encoded_mode_policies())
