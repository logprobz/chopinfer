"""Checkpoint verification against the lock, with and without the stamp cache."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.refrun import keys, stamps

SECOND = 1_000_000_000


class CoarseFilesystem:
    """One file on a simulated filesystem with 1-second timestamp ticks and a
    clock that moves only on sleep or when a test moves it."""

    def __init__(self, path, data, now):
        self.path, self.now, self.real_stamp = path, now, stamps.stamp
        self.write(data)

    def write(self, data):
        self.path.write_bytes(data)
        self.written = self.now

    def stamp(self, path):
        value = self.real_stamp(path)
        value["mtime_ns"] = value["ctime_ns"] = self.written // SECOND * SECOND
        return value

    def sleep(self, seconds):
        self.now += round(seconds * SECOND)

    def patched(self):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(stamps, "stamp", side_effect=self.stamp))
        stack.enter_context(mock.patch.object(stamps.time, "time_ns", side_effect=lambda: self.now))
        stack.enter_context(mock.patch.object(stamps.time, "sleep", side_effect=self.sleep))
        return stack


class StampTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(self.directory.name)
        self.root, self.cache = root / "checkpoint", root / "stamps"
        self.root.mkdir()
        files = {"config.json": b'{"rope_theta": 10000.0}', "model.safetensors": bytes(range(256)) * 4}
        self.lock = {}
        for name, data in files.items():
            (self.root / name).write_bytes(data)
            self.lock[name] = {"byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.files = sorted(files)
        # Local filesystems tick in a few ms; a short window keeps the tests fast.
        self.racy, stamps.RACY_NS = stamps.RACY_NS, 100_000_000

    def tearDown(self):
        stamps.RACY_NS = self.racy
        self.directory.cleanup()

    def same_size_edit(self, name, old, new):
        path = self.root / name
        data = path.read_bytes()
        self.assertEqual(len(old), len(new))
        path.write_bytes(data.replace(old, new))
        self.assertEqual(path.stat().st_size, len(data))

    def test_matching_files_pass_and_fill_the_cache(self):
        verified = stamps.verify(self.root, self.lock, self.files, self.cache)
        self.assertEqual(sorted(verified), self.files)
        records = [json.loads(p.read_text()) for p in self.cache.iterdir()]
        self.assertEqual(sorted(r["sha256"] for r in records), sorted(e["sha256"] for e in self.lock.values()))
        with mock.patch.object(keys, "file_sha256", side_effect=AssertionError("rehashed")):
            self.assertEqual(stamps.verify(self.root, self.lock, self.files, self.cache), verified)

    def test_files_are_read_only_once_settled(self):
        real = keys.file_sha256

        def hash_settled(path):
            info = os.stat(path)
            self.assertGreaterEqual(stamps.time.time_ns() - max(info.st_mtime_ns, info.st_ctime_ns),
                                    stamps.RACY_NS)
            return real(path)
        with mock.patch.object(keys, "file_sha256", side_effect=hash_settled) as hashed:
            stamps.verify(self.root, self.lock, self.files, self.cache)
        self.assertEqual(hashed.call_count, 2)

    def test_same_size_changes_are_refused(self):
        for cache in (None, self.cache):
            stamps.verify(self.root, self.lock, self.files, cache)
        self.same_size_edit("config.json", b"10000.0", b"1000000")
        for cache in (None, self.cache):
            with self.assertRaisesRegex(ValueError, "config.json has sha256"):
                stamps.verify(self.root, self.lock, self.files, cache)
        self.same_size_edit("config.json", b"1000000", b"10000.0")
        self.same_size_edit("model.safetensors", b"\x07\x08", b"\x08\x07")
        with self.assertRaisesRegex(ValueError, "model.safetensors has sha256"):
            stamps.verify(self.root, self.lock, self.files, self.cache)

    def test_a_restored_mtime_still_rehashes(self):
        stamps.verify(self.root, self.lock, self.files, self.cache)
        path = self.root / "config.json"
        info = path.stat()
        self.same_size_edit("config.json", b"10000.0", b"1000000")
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "config.json has sha256"):
            stamps.verify(self.root, self.lock, self.files, self.cache)

    def test_a_write_during_a_slow_hash_is_refused_and_not_cached(self):
        # The confirm review's schedule: written at t=100.1, verification
        # starts at t=100.2, the first block is rewritten after it was read,
        # and the hash ends at t=103. The hash now starts only once the
        # stamp has settled, so the rewrite lands in a later tick.
        path, old, new = self.root / "weights", b"0123456789", b"9123456789"
        fs = CoarseFilesystem(path, old, 100_100_000_000)
        lock = {"weights": {"byte_length": len(old), "sha256": hashlib.sha256(old).hexdigest()}}
        stamps.RACY_NS = 2 * SECOND
        fs.now = 100_200_000_000

        def slow_hash(file):
            value = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            fs.write(new)
            fs.now = 103_000_000_000
            return value
        with fs.patched():
            with mock.patch.object(keys, "file_sha256", side_effect=slow_hash):
                with self.assertRaisesRegex(RuntimeError, "changed while it was hashed"):
                    stamps.verify(self.root, lock, ["weights"], self.cache)
            self.assertFalse(self.cache.exists())
            with self.assertRaisesRegex(ValueError, "weights has sha256"):
                stamps.verify(self.root, lock, ["weights"], self.cache)

    def test_a_write_in_the_initial_tick_is_hashed_as_written(self):
        # A rewrite at t=100.3 keeps the t=100.1 stamp, so it must happen
        # before the bytes are read: the hash then covers the new bytes.
        path, old, new = self.root / "weights", b"0123456789", b"9123456789"
        fs = CoarseFilesystem(path, old, 100_100_000_000)
        lock = {"weights": {"byte_length": len(old), "sha256": hashlib.sha256(old).hexdigest()}}
        stamps.RACY_NS = 2 * SECOND
        fs.now = 100_200_000_000
        sleep = fs.sleep

        def rewrite_then_sleep(seconds):
            fs.now = 100_300_000_000
            fs.write(new)
            sleep(seconds)
        with fs.patched(), mock.patch.object(stamps.time, "sleep", side_effect=rewrite_then_sleep):
            with self.assertRaisesRegex(ValueError, "weights has sha256"):
                stamps.verify(self.root, lock, ["weights"], self.cache)
        self.assertEqual(fs.now, 102_100_000_000)

    def test_confirm_sees_any_write_after_verification(self):
        verified = stamps.verify(self.root, self.lock, self.files, self.cache)
        stamps.confirm(self.root, verified)
        path = self.root / "config.json"
        path.write_bytes(path.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "config.json changed after it was verified"):
            stamps.confirm(self.root, verified)

    def test_confirm_sees_a_repointed_symlink(self):
        blobs = self.root.parent / "blobs"
        blobs.mkdir()
        for name in ("a", "b"):
            (blobs / name).write_bytes((self.root / "config.json").read_bytes())
        (self.root / "config.json").unlink()
        (self.root / "config.json").symlink_to(blobs / "a")
        verified = stamps.verify(self.root, self.lock, ["config.json"], self.cache)
        for target in ("b", "a"):
            (self.root / "config.json").unlink()
            (self.root / "config.json").symlink_to(blobs / target)
            with self.assertRaisesRegex(RuntimeError, "config.json changed after it was verified"):
                stamps.confirm(self.root, verified)

    def test_timestamps_ahead_of_the_clock_are_refused(self):
        path = self.root / "config.json"
        ahead = path.stat().st_mtime_ns + 10 * stamps.RACY_NS
        os.utime(path, ns=(ahead, ahead))
        with self.assertRaisesRegex(ValueError, "ahead of this host's clock"):
            stamps.verify(self.root, self.lock, ["config.json"], self.cache)

    def test_sizes_and_unlocked_files_are_refused(self):
        (self.root / "config.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, r"config.json \(2 bytes\)"):
            stamps.verify(self.root, self.lock, self.files, self.cache)
        (self.root / "extra.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "extra.json"):
            stamps.verify(self.root, self.lock, ["extra.json"], self.cache)

    def test_cli_verifies_every_locked_file(self):
        lock = self.root.parent / "lock.json"
        lock.write_text(json.dumps({"v": self.lock}))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            stamps.main(["--checkpoint", str(self.root), "--lock", str(lock), "--lock-variant", "v",
                         "--stamps", str(self.cache)])
        self.assertIn("2 files match", out.getvalue())
        self.assertEqual(len(list(self.cache.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
