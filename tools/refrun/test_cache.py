"""The reference cache store, refsets, gc, diff-entries and fixture bindings
on synthetic entries. Runs without torch."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.model_sdk.diagnostic import canonical
from tools.refrun import cache, keys, modes, workload

ENVIRONMENT = {"torch": "2.14.0+cu130", "cuda_runtime": "13.0", "cublas": "13.1.1.3",
               "device": "sm_120", "env_freeze_sha256": "e" * 64}
LOCK = {"config.json": {"byte_length": 10, "sha256": "c" * 64}}
ADAPTER = "models/demo/reference.py"
TEXTS = {"alpha": [1, 5, 9, 33], "beta": [3, 4, 5, 6, 7]}
# beta's greedy answer, where a test generates it.
ANSWER = [8, 9]


def scores(ids, score_from, offset=0.0):
    count = len(ids) - score_from
    return {"score_from": score_from, "token": ids[score_from:], "logit": [1.0] * count,
            "logprob": [-0.5 - offset - i for i in range(count)], "top1": ids[score_from:],
            "margin": [0.25] * count, "topk_ids": [[t, 0] for t in ids[score_from:]],
            "topk_logits": [[1.0, 0.75]] * count, "topk_logprobs": [[-0.5, -0.75]] * count}


def sequence(ids, score_from=1, answer=(), greedy=False):
    answer = ({"policy": "greedy", "ids_sha256": None, "max_new_tokens": len(ANSWER), "stop_ids": []} if greedy
              else {"policy": "forced", "ids_sha256": keys.ids_sha256(list(answer)), "max_new_tokens": None,
                    "stop_ids": None})
    return {"token_ids_sha256": keys.ids_sha256(ids), "score_from": score_from, "answer": answer, "media": []}


class Tree:
    """A repository root with the files a binding is checked against."""

    def __init__(self, root):
        self.root = Path(root)
        for path in keys.NUMERICS:
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(keys.ROOT / path, self.root / path)
        self.write(ADAPTER, "# demo adapter\n")
        self.write("models/demo/checkpoint-files.json", json.dumps({"base": LOCK}))
        self.checkpoint = self.root / "checkpoint"
        self.write("checkpoint/modeling_demo.py", "# upstream\n")
        # Repository code the demo adapter executes beyond its own file.
        self.write("models/demo/inherited.py", "# inherited\n")

    def write(self, path, text):
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_text(text)

    def sections(self):
        return {"checkpoint": keys.checkpoint(LOCK, ["config.json"]),
                "reference_code": keys.reference_code(
                    {"demo/modeling_demo.py": keys.file_sha256(self.checkpoint / "modeling_demo.py"),
                     "models/demo/inherited.py": keys.file_sha256(self.root / "models/demo/inherited.py"),
                     "transformers/modeling_utils.py": "f" * 64}, transformers="5.16.1",
                    transformers_sha256="9" * 64),
                "adapter": keys.adapter({path: keys.file_sha256(self.root / path)
                                         for path in [ADAPTER, *keys.NUMERICS]}),
                "environment": ENVIRONMENT, "outputs": keys.outputs(20)}

    def workload(self, texts=TEXTS, greedy=False):
        """A workload file of `texts`; with `greedy`, beta's answer is generated."""
        sequences = [{"name": name, "category": "prose", "token_ids": ids, "score_from": 1}
                     for name, ids in texts.items()]
        if greedy:
            sequences[1]["answer"] = {"policy": "greedy", "max_new_tokens": len(ANSWER), "stop_ids": []}
        self.write("workloads/demo.json", json.dumps({"schema": workload.SCHEMA, "model": "demo",
                                                      "sequences": sequences}))
        return self.root / "workloads/demo.json"

    def entry(self, policy, name, section, entry_scores):
        document = keys.document(model="demo", sequence=section, mode=keys.mode(policy), **self.sections())
        return {"name": name, "category": "prose", "key_sha256": keys.key_sha256(document), "key": document,
                "scores": entry_scores}

    def run(self, mode, offset=0.0, greedy=None, dequant=None):
        """A runner <mode>.json document over TEXTS. `greedy` is beta's
        generated answer: its greedy entry comes first, as in the runner's
        first mode, when `greedy` is a dict {"answer": ids, "offset": x}."""
        policy = modes.policy(mode, dequant)
        entries = []
        if greedy is not None and greedy.get("entry", True):
            steps = scores([0] + greedy["answer"], 1, greedy.get("offset", 0.0))
            entries.append(self.entry(policy, "beta", sequence(TEXTS["beta"], greedy=True),
                                      {"answer": greedy["answer"], **steps}))
        for name, ids in TEXTS.items():
            answer = greedy["answer"] if greedy is not None and name == "beta" else []
            entries.append(self.entry(policy, name, sequence(ids, answer=answer), scores(ids + answer, 1, offset)))
        return {"policy": policy, "entries": entries}


def fixture(runs):
    """A fixture holding each mode's logprob and top1 under the mode's name,
    and a generated answer with its greedy step scores under `greedy`."""
    texts = []
    for name, ids in TEXTS.items():
        text = {"name": name, "category": "prose", "prompt_token_ids": ids, "score_from": 1}
        for mode, run in runs.items():
            for entry in run["entries"]:
                if entry["name"] != name:
                    continue
                if entry["key"]["sequence"]["answer"]["policy"] == "greedy":
                    text["answer_token_ids"] = entry["scores"]["answer"]
                    text["greedy"] = {"answer": entry["scores"]["answer"], "logprob": entry["scores"]["logprob"]}
                else:
                    text[mode] = {"logprob": entry["scores"]["logprob"], "top1": entry["scores"]["top1"],
                                  "topk_ids": [row[:1] for row in entry["scores"]["topk_ids"]]}
        texts.append(text)
    return json.loads(json.dumps({"schema": "demo.teacher-reference.v1", "texts": texts}))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.directory.name) / "repo")
        self.store = cache.Store(Path(self.directory.name) / "cache")
        self.entry = self.tree.run("f32")["entries"][0]

    def tearDown(self):
        self.directory.cleanup()

    def test_an_entry_is_sealed_and_read_back(self):
        key = self.store.put(self.entry["key"], self.entry["scores"], {"name": "alpha"})
        self.assertEqual(key, self.entry["key_sha256"])
        path = self.store.entry_path(key)
        self.assertEqual(path, self.store.root / "entries" / key[:2] / key)
        self.assertEqual(keys.sha256((path / "key.json").read_bytes()), key)
        found = self.store.get(key)
        self.assertEqual((found["key"], found["scores"]), (self.entry["key"], self.entry["scores"]))
        self.assertEqual(found["entry"]["provenance"], {"name": "alpha"})
        self.assertEqual(list((self.store.root / "tmp").iterdir()), [])
        self.assertIsNone(self.store.get("0" * 64))

    def test_a_failed_write_leaves_no_entry(self):
        real = cache.write_synced

        def fail(path, data):
            if path.name == "entry.json":
                raise OSError("disk full")
            real(path, data)
        with mock.patch.object(cache, "write_synced", side_effect=fail), self.assertRaises(OSError):
            self.store.put(self.entry["key"], self.entry["scores"])
        self.assertFalse(self.store.entry_path(self.entry["key_sha256"]).exists())
        self.assertIsNone(self.store.get(self.entry["key_sha256"]))
        self.assertEqual(list((self.store.root / "tmp").iterdir()), [])

    def test_entries_are_create_only_and_a_second_producer_must_match_bytes(self):
        key = self.store.put(self.entry["key"], self.entry["scores"], {"run": 1})
        before = (self.store.entry_path(key) / "entry.json").read_bytes()
        self.assertEqual(self.store.put(self.entry["key"], self.entry["scores"], {"run": 2}), key)
        self.assertEqual((self.store.entry_path(key) / "entry.json").read_bytes(), before)
        drifted = json.loads(json.dumps(self.entry["scores"]))
        drifted["logprob"][0] += 2.0 ** -40
        with self.assertRaisesRegex(cache.Nondeterministic, "differs from the stored one"):
            self.store.put(self.entry["key"], drifted)
        self.assertEqual(self.store.get(key)["scores"], self.entry["scores"])

    def test_a_producer_that_loses_the_rename_race_is_compared(self):
        key = self.entry["key_sha256"]
        real = cache.Store.read
        calls = []

        def racing(store, wanted):
            # The other producer finishes between this one's lookup and rename.
            if not calls:
                calls.append(wanted)
                other = cache.Store(store.root)
                with mock.patch.object(cache.Store, "read", real):
                    other.put(self.entry["key"], drifted)
                return None
            return real(store, wanted)
        drifted = json.loads(json.dumps(self.entry["scores"]))
        drifted["margin"][0] = 0.5
        with mock.patch.object(cache.Store, "read", racing), self.assertRaises(cache.Nondeterministic):
            self.store.put(self.entry["key"], self.entry["scores"])
        self.assertEqual(self.store.get(key)["scores"], drifted)
        self.assertEqual(list((self.store.root / "tmp").iterdir()), [])

    def test_a_tampered_entry_is_refused(self):
        key = self.store.put(self.entry["key"], self.entry["scores"])
        path = self.store.entry_path(key) / "scores.json"
        path.write_bytes(path.read_bytes().replace(b"-0.5", b"-0.6", 1))
        with self.assertRaisesRegex(ValueError, "differs from its seal"):
            self.store.get(key)

    def test_refsets_are_content_addressed(self):
        document = cache.refset("demo", "a" * 64, {"f32": [["alpha", self.entry["key_sha256"]]]}, "b" * 64)
        digest = self.store.put_refset(document)
        self.assertEqual(digest, keys.sha256(canonical(document)))
        self.assertEqual(self.store.put_refset(document), digest)
        self.assertEqual(self.store.refset(digest), document)
        self.store.refset_path(digest).write_bytes(canonical({**document, "model": "other"}))
        with self.assertRaisesRegex(ValueError, "differs from its digest"):
            self.store.refset(digest)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.directory.name) / "repo")
        self.store = cache.Store(Path(self.directory.name) / "cache")
        self.runs = {"f32": self.tree.run("f32"), "bf16": self.tree.run("bf16", offset=0.01)}
        for run in self.runs.values():
            for entry in run["entries"]:
                self.store.put(entry["key"], entry["scores"])
        self.value = cache.bind(fixture(self.runs), self.runs, str(self.tree.workload()),
                                "models/demo/checkpoint-files.json", "base", store=self.store)

    def tearDown(self):
        self.directory.cleanup()

    def verify(self, value=None, store=None, checkpoint=None):
        return cache.verify(self.value if value is None else value, self.tree.root, checkpoint, store)

    def stale(self, field, value=None, **kwargs):
        with self.assertRaises(cache.StaleReference) as raised:
            self.verify(value, **kwargs)
        self.assertEqual(raised.exception.field, field, str(raised.exception))
        self.assertIn(f"stale_reference: {field}:", str(raised.exception))

    def edited(self, edit):
        value = json.loads(json.dumps(self.value))
        edit(value)
        return value

    def test_a_fresh_binding_verifies_with_and_without_the_cache(self):
        binding = self.value["binding"]
        self.assertEqual(binding["digests"]["lock"], keys.checkpoint(LOCK, [])["source_lock_sha256"])
        self.assertEqual(set(binding["digests"]), {"lock", "reference_code", "adapter", "mode_policy", "environment"})
        self.assertEqual([name for name, _, _ in binding["modes"]["f32"]["entries"]], list(TEXTS))
        self.assertEqual(self.verify(checkpoint=self.tree.checkpoint),
                         {"status": "bound", "refset_sha256": binding["refset_sha256"], "cache": None})
        result = self.verify(store=self.store)
        self.assertEqual(result["cache"], {"root": str(self.store.root), "refset_present": True, "entries": 4,
                                           "missing": 0})
        self.assertEqual(self.store.refset(binding["refset_sha256"])["assembled_fixture_sha256"],
                         cache.fixture_sha256(self.value))

    def test_a_mode_run_under_another_dequant_rule_is_bound_by_its_label(self):
        # run.py labels `--modes f32:hf-bf16` f32-hf-bf16; its key and policy are f32's.
        runs = {"f32": self.runs["f32"], "f32-hf-bf16": self.tree.run("f32", offset=0.02, dequant="hf-bf16")}
        value = cache.bind(fixture(runs), runs, str(self.tree.workload()), "models/demo/checkpoint-files.json", "base")
        self.assertEqual(value["binding"]["modes"]["f32-hf-bf16"]["mode"], "f32")
        self.verify(value)
        with mock.patch.dict(modes.MODES["f32"], {"log_softmax": "float64"}):
            self.stale("mode_policy", value)

    def test_an_unbound_fixture_is_reported_unbound(self):
        self.assertEqual(self.verify(fixture(self.runs)), {"status": "unbound", "refset_sha256": None})

    def test_bind_refuses_a_fixture_that_is_not_its_runs(self):
        value = fixture(self.runs)
        value["texts"][1]["f32"]["logprob"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "f32/beta: the fixture's logprob differs"):
            cache.bind(value, self.runs, str(self.tree.workload()), "models/demo/checkpoint-files.json", "base")
        value = fixture(self.runs)
        value["texts"][0]["prompt_token_ids"] = [1, 5, 9]
        with self.assertRaisesRegex(cache.StaleReference, "alpha: token IDs differ"):
            cache.bind(value, self.runs, str(self.tree.workload()), "models/demo/checkpoint-files.json", "base")

    def test_each_binding_field_is_named_when_it_disagrees_with_itself(self):
        self.stale("schema", self.edited(lambda v: v["binding"].update(schema="obsolete")))
        for field in ("lock", "reference_code", "adapter", "environment"):
            with self.subTest(field):
                self.stale(field, self.edited(lambda v: v["binding"]["digests"].update({field: "0" * 64})))
        self.stale("mode_policy", self.edited(lambda v: v["binding"]["digests"]["mode_policy"].update(f32="0" * 64)))
        self.stale("key_sha256", self.edited(lambda v: v["binding"]["modes"]["bf16"]["entries"][1].__setitem__(
            2, "0" * 64)))
        self.stale("refset_sha256", self.edited(lambda v: v["binding"].update(refset_sha256="0" * 64)))
        self.stale("refset_sha256", self.edited(lambda v: v["binding"].update(workload_sha256="0" * 64)))

    def test_working_tree_changes_are_named(self):
        self.tree.write("models/demo/checkpoint-files.json", json.dumps({"base": {**LOCK, "x.bin": LOCK["config.json"]}}))
        self.stale("lock")
        self.tree.write("models/demo/checkpoint-files.json", json.dumps({"base": LOCK}))
        self.verify()
        self.tree.write(ADAPTER, "# demo adapter, changed\n")
        self.stale("adapter")
        self.tree.write(ADAPTER, "# demo adapter\n")
        score = self.tree.root / "tools/refrun/score.py"
        score.write_text(score.read_text() + "\n")
        self.stale("adapter")
        score.write_bytes((keys.ROOT / "tools/refrun/score.py").read_bytes())
        self.tree.write("tools/refrun/taps.py", "# a new numerics file\n")
        with mock.patch.object(keys, "NUMERICS", (*keys.NUMERICS, "tools/refrun/taps.py")):
            self.stale("adapter")
        with mock.patch.dict(modes.FLAGS, {"cudnn_benchmark": True}):
            self.stale("mode_policy")
        self.verify(checkpoint=self.tree.checkpoint)
        self.tree.write("checkpoint/modeling_demo.py", "# upstream, changed\n")
        self.stale("reference_code", checkpoint=self.tree.checkpoint)
        self.verify()

    def test_inherited_repository_code_is_checked_without_a_checkpoint(self):
        self.verify()
        self.tree.write("models/demo/inherited.py", "# inherited, changed\n")
        self.stale("reference_code")
        (self.tree.root / "models/demo/inherited.py").unlink()
        self.stale("reference_code")
        self.tree.write("models/demo/inherited.py", "# inherited\n")
        self.verify()

    def test_fixture_changes_are_named(self):
        self.stale("sequence", self.edited(lambda v: v["texts"][0].update(prompt_token_ids=[1, 5, 9, 34])))
        self.stale("sequence", self.edited(lambda v: v["texts"][1].update(score_from=2)))
        self.stale("sequence", self.edited(lambda v: v["texts"].pop()))
        changed = self.edited(lambda v: v["texts"][1]["bf16"]["logprob"].__setitem__(0, -9.0))
        self.stale("assembled_fixture", changed)
        self.stale("scores", changed, store=self.store)
        changed = self.edited(lambda v: v["texts"][0]["f32"]["topk_ids"][0].__setitem__(0, 99))
        self.stale("scores", changed, store=self.store)
        self.stale("assembled_fixture", self.edited(lambda v: v.update(schema="demo.teacher-reference.v2")))

    def test_entries_missing_from_the_cache_are_counted(self):
        key = self.value["binding"]["modes"]["bf16"]["entries"][0][2]
        shutil.rmtree(self.store.entry_path(key))
        os.unlink(self.store.refset_path(self.value["binding"]["refset_sha256"]))
        self.assertEqual(self.verify(store=self.store)["cache"],
                         {"root": str(self.store.root), "refset_present": False, "entries": 4, "missing": 1})


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


class GcTests(unittest.TestCase):
    def setUp(self):
        # Isolate fixture writes and Git calls made by the code under test.
        environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
        isolated = mock.patch.dict(os.environ, environment, clear=True)
        isolated.start()
        self.addCleanup(isolated.stop)
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.tree = Tree(root / "repo")
        self.store = cache.Store(root / "cache")
        self.now = 1_900_000_000
        self.runs = {"f32": self.tree.run("f32"), "f64": self.tree.run("f64"), "bf16": self.tree.run("bf16")}
        for run in self.runs.values():
            for entry in run["entries"]:
                self.store.put(entry["key"], entry["scores"])
        main = cache.bind(fixture({"f32": self.runs["f32"]}), {"f32": self.runs["f32"]}, str(self.tree.workload()),
                          "models/demo/checkpoint-files.json", "base", store=self.store)
        kept = cache.bind(fixture({"f64": self.runs["f64"]}), {"f64": self.runs["f64"]}, str(self.tree.workload()),
                          "models/demo/checkpoint-files.json", "base", store=self.store)
        self.loose = self.store.put_refset(cache.refset("demo", "a" * 64, {}, "c" * 64))
        self.refsets = {"main": main["binding"]["refset_sha256"], "kept": kept["binding"]["refset_sha256"]}
        repo = self.tree.root
        git(repo, "init", "-q", "-b", "main")
        self.tree.write("fixtures/demo/teacher.json", json.dumps(main))
        git(repo, "add", "fixtures")
        git(repo, "commit", "-q", "-m", "main fixture")
        git(repo, "checkout", "-q", "-b", "candidate")
        self.tree.write("fixtures/demo/teacher.json", json.dumps(kept))
        git(repo, "commit", "-q", "-am", "candidate fixture")
        git(repo, "checkout", "-q", "main")
        old = self.now - cache.AGE_FLOOR - 60
        for path in [*self.store.root.glob("entries/*/*/entry.json"), *self.store.root.glob("refsets/*")]:
            os.utime(path, (old, old))
        (self.store.root / "tmp" / "partial").mkdir(parents=True)
        os.utime(self.store.root / "tmp" / "partial", (old, old))

    def tearDown(self):
        self.directory.cleanup()

    def keys_of(self, mode):
        return {entry["key_sha256"] for entry in self.runs[mode]["entries"]}

    def removed(self, report):
        return {Path(path).name for kind in ("entries", "refsets") for path in report["removed"][kind]}

    def test_a_dry_run_lists_what_no_bound_fixture_reaches(self):
        report = cache.gc(self.store, self.tree.root, now=self.now)
        self.assertEqual(report["refs"], ["main"])
        self.assertEqual(report["bound_refsets"], [self.refsets["main"]])
        expected = self.keys_of("f64") | self.keys_of("bf16") | {f"{self.refsets['kept']}.json", f"{self.loose}.json"}
        self.assertEqual(self.removed(report), expected)
        self.assertEqual([Path(p).name for p in report["removed"]["tmp"]], ["partial"])
        self.assertFalse(report["deleted"])
        for key in self.keys_of("f64"):
            self.assertIsNotNone(self.store.get(key))

    def test_an_unreadable_tracked_binding_stops_gc(self):
        # The confirm review's case: reachability cannot be established, so nothing may go.
        repo = self.tree.root
        broken = {
            "truncated JSON": '{"binding": {"schema": "%s", "refset_sha256": ' % cache.BINDING,
            "a binding without entries": json.dumps({"binding": {"schema": cache.BINDING, "refset_sha256": "0" * 64}}),
            "the schema outside a binding": json.dumps({"notes": cache.BINDING}),
        }
        for label, text in broken.items():
            with self.subTest(label):
                self.tree.write("fixtures/demo/broken.json", text)
                git(repo, "add", "fixtures")
                git(repo, "commit", "-q", "-m", label)
                with self.assertRaisesRegex(ValueError, "main:fixtures/demo/broken.json"):
                    cache.gc(self.store, repo, refs=["candidate"], now=self.now, delete=True)
                for key in self.keys_of("f32") | self.keys_of("f64") | self.keys_of("bf16"):
                    self.assertIsNotNone(self.store.get(key))
                self.assertIsNotNone(self.store.refset(self.refsets["main"]))
        # A README may name the schema; only JSON fixtures hold bindings.
        git(repo, "rm", "-q", "fixtures/demo/broken.json")
        self.tree.write("fixtures/demo/README.md", f"Bound fixtures carry {cache.BINDING}.\n")
        git(repo, "add", "fixtures")
        git(repo, "commit", "-q", "-m", "readme")
        report = cache.gc(self.store, repo, refs=["candidate"], now=self.now)
        self.assertEqual(report["bound_refsets"], sorted(self.refsets.values()))

    def test_the_keep_file_and_the_age_floor_keep_entries(self):
        (self.store.root / "keep").write_text("candidate\n")
        young = self.now - cache.AGE_FLOOR + 60
        for key in self.keys_of("bf16"):
            os.utime(self.store.entry_path(key) / "entry.json", (young, young))
        report = cache.gc(self.store, self.tree.root, now=self.now, delete=True)
        self.assertEqual(report["refs"], ["main", "candidate"])
        self.assertEqual(self.removed(report), {f"{self.loose}.json"})
        (self.store.root / "keep").unlink()
        report = cache.gc(self.store, self.tree.root, now=self.now, delete=True)
        self.assertEqual(self.removed(report), self.keys_of("f64") | {f"{self.refsets['kept']}.json"})
        for key in self.keys_of("f64"):
            self.assertIsNone(self.store.get(key))
        for key in self.keys_of("f32") | self.keys_of("bf16"):
            self.assertIsNotNone(self.store.get(key))
        self.assertIsNotNone(self.store.refset(self.refsets["main"]))
        self.assertEqual(list((self.store.root / "tmp").iterdir()), [])


    def test_named_refs_add_to_main_and_the_keep_file(self):
        # The review's case: a ref named on the command line must not drop main.
        report = cache.gc(self.store, self.tree.root, refs=["candidate"], now=self.now, delete=True)
        for key in self.keys_of("f32") | self.keys_of("f64"):
            self.assertIsNotNone(self.store.get(key))
        self.assertIsNotNone(self.store.refset(self.refsets["main"]))
        self.assertEqual(report["refs"], ["main", "candidate"])
        self.assertEqual(self.removed(report), self.keys_of("bf16") | {f"{self.loose}.json"})
        git(self.tree.root, "branch", "other")
        (self.store.root / "keep").write_text("candidate\nmain\n")
        report = cache.gc(self.store, self.tree.root, refs=["other", "candidate"], now=self.now, delete=True)
        self.assertEqual(report["refs"], ["main", "candidate", "other"])
        self.assertEqual(self.removed(report), set())


class GreedyTests(unittest.TestCase):
    """A generated answer is bound like a forced one: its greedy entry's
    answer and copied step scores must match the fixture."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.directory.name) / "repo")
        self.workload = self.tree.workload(greedy=True)
        self.runs = {"f32": self.tree.run("f32", greedy={"answer": ANSWER}),
                     "bf16": self.tree.run("bf16", offset=0.01, greedy={"answer": ANSWER, "entry": False})}
        self.store = self.cached({"answer": ANSWER})

    def tearDown(self):
        self.directory.cleanup()

    def cached(self, greedy, name="cache"):
        """A cache holding the runs' entries, with beta's greedy entry made from `greedy`."""
        store = cache.Store(Path(self.directory.name) / name)
        cached = self.tree.run("f32", greedy=greedy)["entries"][0]
        store.put(cached["key"], cached["scores"])
        for run in self.runs.values():
            for entry in run["entries"]:
                if entry["key_sha256"] != cached["key_sha256"]:
                    store.put(entry["key"], entry["scores"])
        return store

    def bind(self, value=None, store=None):
        return cache.bind(fixture(self.runs) if value is None else value, self.runs, str(self.workload),
                          "models/demo/checkpoint-files.json", "base", store=store)

    def test_a_greedy_binding_verifies_against_its_entries(self):
        value = self.bind(store=self.store)
        self.assertEqual(value["texts"][1]["answer_token_ids"], ANSWER)
        result = cache.verify(value, self.tree.root, store=self.store)
        self.assertEqual(result["cache"]["entries"], 5)

    def test_bind_refuses_another_greedy_answer_or_step_score(self):
        for label, edit in (("answer", lambda text: text["greedy"]["answer"].__setitem__(1, 10)),
                            ("logprob", lambda text: text["greedy"]["logprob"].__setitem__(0, 0.0))):
            with self.subTest(label):
                value = fixture(self.runs)
                edit(value["texts"][1])
                with self.assertRaisesRegex(ValueError, f"f32/beta: the fixture's {label} differs"):
                    self.bind(value)

    def test_a_cached_greedy_entry_unlike_the_fixture_is_stale(self):
        value = self.bind()
        for label, greedy in (("answer", {"answer": [8, 10]}), ("logprob", {"answer": ANSWER, "offset": 0.5})):
            with self.subTest(label):
                store = self.cached(greedy, f"cache-{label}")
                with self.assertRaises(cache.StaleReference) as raised:
                    cache.verify(value, self.tree.root, store=store)
                self.assertEqual(raised.exception.field, "scores")
                self.assertIn(f"f32/beta: {label} differs", str(raised.exception))


class WorkloadTests(unittest.TestCase):
    """The binding records the workload's semantic sequences; a workload
    changed after the fixture was emitted is stale."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.directory.name) / "repo")
        self.workload = self.tree.workload()
        runs = {"f32": self.tree.run("f32")}
        self.value = cache.bind(fixture(runs), runs, str(self.workload), "models/demo/checkpoint-files.json", "base")

    def tearDown(self):
        self.directory.cleanup()

    def test_the_bound_workload_verifies(self):
        self.assertEqual(self.value["binding"]["workload_sha256"], cache.workload_sha256(self.workload))
        cache.verify(self.value, self.tree.root, workload=self.workload)

    def test_a_changed_workload_is_stale(self):
        original = json.loads(self.workload.read_text())
        changes = {
            "token IDs": lambda sequences: sequences[0].update(token_ids=[1, 5, 9, 34]),
            "score_from": lambda sequences: sequences[1].update(score_from=2),
            "answer policy": lambda sequences: sequences[1].update(
                answer={"policy": "greedy", "max_new_tokens": 2, "stop_ids": []}),
            "order": lambda sequences: sequences.reverse(),
            "a new sequence": lambda sequences: sequences.append(
                {"name": "gamma", "category": "prose", "token_ids": [2, 3], "score_from": 1}),
        }
        for label, change in changes.items():
            with self.subTest(label):
                edited = json.loads(json.dumps(original))
                change(edited["sequences"])
                self.workload.write_text(json.dumps(edited))
                with self.assertRaises(cache.StaleReference) as raised:
                    cache.verify(self.value, self.tree.root, workload=self.workload)
                self.assertEqual(raised.exception.field, "workload", str(raised.exception))
        self.workload.write_text(json.dumps(original))
        cache.verify(self.value, self.tree.root, workload=self.workload)


class ModePolicyTests(unittest.TestCase):
    def test_the_shipped_mode_policies_are_what_modes_gives(self):
        self.assertEqual(cache.MODE_POLICIES.read_text(), cache.encoded_mode_policies(),
                         "regenerate it: python3 -B -m tools.refrun.cache")

    def test_every_accepted_dequant_rule_is_listed(self):
        policies = cache.mode_policies()
        for name, mode in modes.MODES.items():
            self.assertEqual(policies[name]["policy_sha256"][mode["dequant"]],
                             modes.policy_sha256(modes.policy(name)))
        self.assertEqual(list(policies["bf16"]["policy_sha256"]), ["hf-bf16"])
        self.assertEqual(list(policies["f32"]["policy_sha256"]), list(modes.DEQUANT))


class DiffTests(unittest.TestCase):
    def test_entries_and_refsets_are_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = Tree(Path(directory) / "repo")
            store = cache.Store(Path(directory) / "cache")
            before, after = tree.run("f32"), tree.run("f32", offset=0.125)
            tree.write(ADAPTER, "# demo adapter, changed\n")
            after = {**after, "entries": tree.run("f32", offset=0.125)["entries"]}
            for entry in [*before["entries"], *after["entries"]]:
                store.put(entry["key"], entry["scores"])
            a, b = before["entries"][0]["key_sha256"], after["entries"][0]["key_sha256"]
            delta = cache.diff_entries(store, a, b)
            self.assertEqual(delta["key_sections_differ"], ["adapter"])
            self.assertEqual(delta["scores"], {"bitwise_equal": False, "positions": 3, "top1_changed": 0,
                                               "logprob_max_abs": 0.125, "margin_max_abs": 0.0})
            refsets = [store.put_refset(cache.refset("demo", "a" * 64, {"f32": [[e["name"], e["key_sha256"]]
                                                                                for e in run["entries"]]}, "b" * 64))
                       for run in (before, after)]
            report = cache.diff_entries(store, *refsets)
            self.assertEqual(sorted(report["entries"]), ["f32/alpha", "f32/beta"])
            self.assertEqual(report["entries"]["f32/beta"]["scores"]["positions"], 4)
            self.assertTrue(cache.diff_entries(store, a, a)["scores"]["bitwise_equal"])
            with self.assertRaisesRegex(ValueError, "neither a cached entry"):
                cache.diff_entries(store, a, "0" * 64)


if __name__ == "__main__":
    unittest.main()
