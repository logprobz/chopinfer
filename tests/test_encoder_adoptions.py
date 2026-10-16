"""CPU tests of tests/encoder-adoptions.py in throwaway git repositories:
appending passes; rewriting an approved adoption (the encoder adoption review's
probe: changed cubins, refreshed lock and a rewritten adoption under
unchanged declarations), dropping one or editing one before the last fails."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tests/encoder-adoptions.py"
spec = importlib.util.spec_from_file_location("encoder_adoptions", CHECK)
adoptions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adoptions)
LEDGER = "models/example/bundle/encoder-adoptions.json"


def adoption(declarations, closure, cubin):
    return {"previous": "", "declarations": declarations, "nvcc_sha256": "nvcc",
            "cubins": [{"name": "rows", "source_closure_sha256": closure, "cubin_sha256": cubin}]}


def chained(*entries):
    previous = adoptions.GENESIS
    ledger = []
    for entry in entries:
        entry = dict(entry, previous=previous)
        ledger.append(entry)
        previous = adoptions.digest(entry)
    return ledger


def text(ledger):
    return json.dumps(ledger, indent=2) + "\n"


class Repository:
    def __init__(self, root, main=True):
        self.root = root
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@example.invalid",
                        GIT_COMMITTER_NAME="test", GIT_COMMITTER_EMAIL="test@example.invalid")
        self.run("init", "-q", "-b", "main" if main else "work")
        (root / "README").write_text("base\n")

    def run(self, *args):
        subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True, env=self.env)

    def commit(self, ledger=None):
        if ledger is not None:
            self.write(ledger)
        self.run("add", "--", "README", "models")
        self.run("commit", "-q", "-m", "commit")

    def write(self, ledger):
        path = self.root / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ledger if isinstance(ledger, str) else text(ledger))

    def check(self):
        return subprocess.run([sys.executable, "-B", str(CHECK), str(self.root)],
                              capture_output=True, text=True, env=self.env)


A = adoption(["d1"], "c1", "k1")
B = adoption(["d2"], "c2", "k2")


class EncoderAdoptionsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repository = Repository(Path(self.directory.name))
        self.repository.commit(chained(A))
        self.repository.run("checkout", "-q", "-b", "branch")

    def tearDown(self):
        self.directory.cleanup()

    def assertPasses(self):
        result = self.repository.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assertFails(self, reason):
        result = self.repository.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(reason, result.stdout)

    def test_the_digest_matches_the_stability_check(self):
        # kernels-sm120 `stability::tests::the_digest_matches_the_ci_check`.
        entry = dict(adoption(["d1", "d2"], "c1", "k1"), previous=adoptions.GENESIS)
        self.assertEqual(adoptions.digest(entry),
                         "476a1e5eb6fe93b6800edf0240350918279a5db218ac9e82ee5f7d0e9e12ca88")

    def test_unchanged_and_appended_ledgers_pass(self):
        self.assertPasses()
        self.repository.write(chained(A, B))
        self.assertPasses()

    def test_a_rewritten_approved_adoption_fails(self):
        # The review's probe: a valid chain of its own, but not the approved
        # base's entries.
        self.repository.write(chained(adoption(["d1"], "c9", "k9")))
        self.assertFails("adoption 0 differs from the approved base")

    def test_a_dropped_adoption_fails(self):
        self.repository.write([])
        self.assertFails("adoption 0 of the approved base is missing")

    def test_an_edit_before_the_last_adoption_breaks_the_chain(self):
        ledger = chained(A, B)
        ledger[0]["cubins"][0]["cubin_sha256"] = "k3"
        self.repository.write(ledger)
        self.assertFails("adoption 1 breaks the chain")

    def test_a_ledger_must_be_canonical(self):
        self.repository.write(json.dumps(chained(A)))
        self.assertFails("canonical form")

    def test_a_new_ledger_and_a_base_before_the_chain_pass(self):
        unchained = chained(A)
        del unchained[0]["previous"]
        self.repository.run("checkout", "-q", "main")
        self.repository.commit(unchained)
        self.repository.run("checkout", "-q", "branch")
        self.repository.write(chained(A, B))
        self.assertPasses()
        (self.repository.root / LEDGER).unlink()
        new = self.repository.root / "models/other/bundle/encoder-adoptions.json"
        new.parent.mkdir(parents=True)
        new.write_text(text(chained(B)))
        self.assertPasses()

    def test_no_approved_base_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Repository(Path(directory), main=False)
            repository.commit(chained(A))
            result = repository.check()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no approved base", result.stdout + result.stderr)

    def test_the_committed_ledgers_pass(self):
        result = subprocess.run([sys.executable, "-B", str(CHECK)], capture_output=True, text=True,
                                env=self.repository.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class GitFixtureIsolationTest(unittest.TestCase):
    def test_inherited_git_paths_cannot_change_another_repository(self):
        clean = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            child = Path(directory) / "child"
            child.mkdir()
            subprocess.run(["git", "init", "-q", str(parent)], env=clean, check=True)
            config = parent / ".git/config"
            before = config.read_bytes()
            inherited = dict(clean, GIT_DIR=str(parent / ".git"), GIT_WORK_TREE=str(parent),
                             GIT_COMMON_DIR=str(parent / ".git"),
                             GIT_INDEX_FILE=str(parent / ".git/index"),
                             GIT_CONFIG=str(config), GIT_CONFIG_COUNT="1",
                             GIT_CONFIG_KEY_0="user.name", GIT_CONFIG_VALUE_0="inherited")
            with patch.dict(os.environ, inherited, clear=True):
                repository = Repository(child)
                self.assertEqual(config.read_bytes(), before)
                repository.commit(chained(A))
                result = repository.check()
            self.assertEqual(config.read_bytes(), before)
            self.assertTrue((child / ".git").is_dir())
            self.assertFalse((parent / ".git/index").exists())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
