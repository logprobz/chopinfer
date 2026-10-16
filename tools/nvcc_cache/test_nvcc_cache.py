"""Check the cubin cache wrapper against a fake compiler, without nvcc."""

import concurrent.futures
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WRAPPER = Path(__file__).resolve().parent / "nvcc_cache.py"
sys.path.insert(0, str(WRAPPER.parent))
import nvcc_cache  # noqa: E402

# Writes a digest of its input file and its arguments other than include
# directories and the output (the cubin does not record them), and counts its
# calls.
FAKE_COMPILER = """\
import hashlib, os, sys
args = sys.argv[1:]
output = args[args.index("-o") + 1]
with open(os.environ["FAKE_COMPILER_CALLS"], "a") as calls:
    calls.write("call\\n")
if os.environ.get("FAKE_COMPILER_FAIL"):
    sys.exit(3)
rest = [arg for index, arg in enumerate(args) if arg != "-o" and (index == 0 or args[index - 1] != "-o")]
source = open(rest[-1], "rb").read()
rest = [arg for arg in rest if not arg.startswith("-I")]
with open(output, "wb") as file:
    file.write(hashlib.sha256(repr(rest).encode() + source).digest() * int(os.environ.get("FAKE_COMPILER_REPEAT", "64")))
"""


class NvccCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.compiler = self.root / "bin" / "nvcc"
        self.compiler.parent.mkdir()
        self.compiler.write_text(f"#!{sys.executable}\n{FAKE_COMPILER}")
        self.compiler.chmod(0o755)
        self.calls = self.root / "calls"
        self.cache = self.root / "cache"
        self.env = {
            "PATH": os.environ.get("PATH", ""),
            "FAKE_COMPILER_CALLS": str(self.calls),
            "CHOPINFER_NVCC_CACHE_DIR": str(self.cache),
            "CHOPINFER_NVCC_CACHE_CLOSURE": "a" * 64,
            "CHOPINFER_NVCC_CACHE_TOOLCHAIN": "b" * 64,
            "CHOPINFER_NVCC_CACHE_LOOKUP": "1",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def worktree(self, name):
        """A stage directory and output directory, as kernel-build lays them out."""
        stage = self.root / name / "out" / "k.stage"
        (stage / "include").mkdir(parents=True, exist_ok=True)
        (stage / "k.cu").write_text("__global__ void k() {}\n")
        return stage

    def compile(self, stage, *flags, check=True, **variables):
        output = stage.parent / "k.cubin"
        argv = [sys.executable, str(WRAPPER), str(self.compiler), "-arch=sm_120f", "-cubin", "-I.",
                f"-I{stage}/include", *flags, "k.cu", "-o", str(output)]
        result = subprocess.run(argv, cwd=stage, env={**self.env, **variables}, capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
        return output.read_bytes() if output.exists() else None, result

    def call_count(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def events(self):
        log = self.cache / "v1" / "events.log"
        return [line.split()[1] for line in log.read_text().splitlines()] if log.exists() else []

    def entries(self):
        return sorted(path for path in (self.cache / "v1").glob("??/*") if not path.name.startswith("."))

    def test_a_hit_returns_identical_bytes_without_compiling(self):
        stage = self.worktree("one")
        cold, _ = self.compile(stage)
        (stage.parent / "k.cubin").unlink()
        warm, _ = self.compile(stage)
        self.assertEqual(warm, cold)
        self.assertEqual(self.call_count(), 1)
        self.assertEqual(self.events(), ["miss", "hit"])

    def test_a_changed_key_input_misses(self):
        stage = self.worktree("one")
        baseline, _ = self.compile(stage, "-DTILE=64")
        changes = [
            (("-DTILE=128",), {}),
            (("-DTILE=64", "-lineinfo"), {}),
            (("-DTILE=64",), {"CHOPINFER_NVCC_CACHE_CLOSURE": "c" * 64}),
            (("-DTILE=64",), {"CHOPINFER_NVCC_CACHE_TOOLCHAIN": "d" * 64}),
            (("-DTILE=64",), {"NVCC_APPEND_FLAGS": "--fmad=false"}),
        ]
        for count, (flags, variables) in enumerate(changes, start=2):
            self.compile(stage, *flags, **variables)
            self.assertEqual(self.call_count(), count, (flags, variables))
        self.assertEqual(self.events(), ["miss"] * 6)
        self.assertEqual(len(self.entries()), 6)
        again, _ = self.compile(stage, "-DTILE=64")
        self.assertEqual(again, baseline)
        self.assertEqual(self.call_count(), 6)

    def test_another_worktree_path_hits(self):
        first, _ = self.compile(self.worktree("one"))
        second, _ = self.compile(self.worktree("elsewhere/two"))
        self.assertEqual(second, first)
        self.assertEqual(self.call_count(), 1)
        self.assertEqual(self.events(), ["miss", "hit"])

    def test_a_corrupt_entry_is_a_miss_and_is_replaced(self):
        stage = self.worktree("one")
        cold, _ = self.compile(stage)
        (entry,) = self.entries()
        for damage in (lambda data: data[:-1] + bytes([data[-1] ^ 1]), lambda data: data[: len(data) // 2],
                       lambda data: b"", lambda data: b"garbage\n" + data):
            entry.write_bytes(damage(entry.read_bytes()))
            calls = self.call_count()
            again, _ = self.compile(stage)
            self.assertEqual(again, cold)
            self.assertEqual(self.call_count(), calls + 1)
            self.assertEqual(self.events()[-2:], ["corrupt", "miss"])
            hit, _ = self.compile(stage)
            self.assertEqual(hit, cold)
            self.assertEqual(self.events()[-1], "hit")

    def test_eviction_keeps_the_bound_and_drops_the_least_recently_used(self):
        # Each entry is a 64 KiB payload plus its header line.
        bound = 4 * 65536 + 1024
        stage = self.worktree("one")
        for index in range(4):
            self.compile(stage, f"-DN={index}", FAKE_COMPILER_REPEAT="2048",
                         CHOPINFER_NVCC_CACHE_MAX_BYTES=str(bound))
            os.utime(self.entries_for(f"-DN={index}")[0], (1000 + index, 1000 + index))
        self.compile(stage, "-DN=0", FAKE_COMPILER_REPEAT="2048", CHOPINFER_NVCC_CACHE_MAX_BYTES=str(bound))
        self.assertEqual(self.events()[-1], "hit")
        self.compile(stage, "-DN=4", FAKE_COMPILER_REPEAT="2048", CHOPINFER_NVCC_CACHE_MAX_BYTES=str(bound))
        self.assertLessEqual(sum(path.stat().st_size for path in self.entries()), bound)
        calls = self.call_count()
        self.compile(stage, "-DN=0", FAKE_COMPILER_REPEAT="2048", CHOPINFER_NVCC_CACHE_MAX_BYTES=str(bound))
        self.assertEqual(self.call_count(), calls, "the recently used entry survives")
        self.compile(stage, "-DN=1", FAKE_COMPILER_REPEAT="2048", CHOPINFER_NVCC_CACHE_MAX_BYTES=str(bound))
        self.assertEqual(self.call_count(), calls + 1, "the least recently used entry was evicted")

    def entries_for(self, define):
        stage = self.worktree("one")
        args = ["-arch=sm_120f", "-cubin", "-I.", f"-I{stage}/include", define, "k.cu"]
        key = nvcc_cache.cache_key(args, str(stage.resolve()), self.env)
        return [self.cache / "v1" / key[:2] / key]

    def test_missing_key_variables_fall_through(self):
        stage = self.worktree("one")
        keyed = self.env
        for missing in ("CHOPINFER_NVCC_CACHE_CLOSURE", "CHOPINFER_NVCC_CACHE_TOOLCHAIN",
                        "CHOPINFER_NVCC_CACHE_DIR"):
            self.env = {name: value for name, value in keyed.items() if name != missing}
            self.compile(stage)
            self.compile(stage)
        self.assertEqual(self.call_count(), 6)
        self.assertFalse(self.cache.exists())

    def test_inputs_the_key_cannot_describe_bypass_the_cache(self):
        stage = self.worktree("one")
        options = self.root / "options"
        options.write_text("--fmad=true\n")
        uncovered = [
            ((), {"NVCC_APPEND_FLAGS": f"-optf {options}"}),
            ((), {"NVCC_PREPEND_FLAGS": f"--options-file={options}"}),
            (("-optf", str(options)), {}),
            ((f"--options-file={options}",), {}),
            ((f"-Xptxas=-O3,-optf={options}",), {}),
            (("-Xcompiler", "-include,extra.h"), {}),
            (("-ccbin=/usr/bin/g++-12",), {}),
        ]
        for flags, variables in uncovered:
            calls = self.call_count()
            self.compile(stage, *flags, **variables)
            self.compile(stage, *flags, **variables)
            self.assertEqual(self.call_count(), calls + 2, (flags, variables))
        self.assertEqual(self.events(), ["bypass"] * 2 * len(uncovered))
        self.assertEqual(self.entries() if self.cache.exists() else [], [])

    def test_preprocessing_environment_changes_the_key(self):
        stage = self.worktree("one")
        self.compile(stage, SOURCE_DATE_EPOCH="0")
        for count, variables in enumerate([{"SOURCE_DATE_EPOCH": "864000"}, {"CPATH": "include"},
                                           {"CPLUS_INCLUDE_PATH": "include"}, {"NVCC_CCBIN": "g++-12"}],
                                          start=2):
            self.compile(stage, **variables)
            self.assertEqual(self.call_count(), count, variables)
        self.compile(stage, SOURCE_DATE_EPOCH="0")
        self.assertEqual(self.events(), ["miss"] * 5 + ["hit"])

    def test_hits_alone_rotate_the_log(self):
        stage = self.worktree("one")
        argv = [str(self.compiler), "-arch=sm_120f", "-cubin", "-I.", "k.cu", "-o", str(stage.parent / "k.cubin")]
        directory = os.getcwd()
        threshold = nvcc_cache.LOG_ROTATE_BYTES
        try:
            os.chdir(stage)
            nvcc_cache.LOG_ROTATE_BYTES = 256
            for _ in range(20):
                self.assertEqual(nvcc_cache.main(argv, self.env), 0)
        finally:
            nvcc_cache.LOG_ROTATE_BYTES = threshold
            os.chdir(directory)
        log = self.cache / "v1" / "events.log"
        self.assertEqual(self.call_count(), 1)
        self.assertTrue(log.with_name("events.log.1").exists())
        self.assertLess(log.stat().st_size, 256 + 200)

    def test_lookup_off_compiles_and_refreshes(self):
        stage = self.worktree("one")
        cold, _ = self.compile(stage)
        refreshed, _ = self.compile(stage, CHOPINFER_NVCC_CACHE_LOOKUP="0")
        self.assertEqual(refreshed, cold)
        self.assertEqual(self.call_count(), 2)
        self.assertEqual(self.events(), ["miss", "refresh"])

    def test_a_failed_compile_stores_nothing(self):
        stage = self.worktree("one")
        _, result = self.compile(stage, check=False, FAKE_COMPILER_FAIL="1")
        self.assertEqual(result.returncode, 3)
        self.assertEqual(self.entries() if self.cache.exists() else [], [])

    def test_concurrent_builders_agree(self):
        stages = [self.worktree(f"tree{index}") for index in range(8)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            outputs = list(pool.map(lambda stage: self.compile(stage)[0], stages))
        self.assertEqual(len(set(outputs)), 1)
        (entry,) = self.entries()
        header, _, payload = entry.read_bytes().partition(b"\n")
        self.assertEqual(header.split()[2].decode(), hashlib.sha256(payload).hexdigest())
        self.assertEqual(payload, outputs[0])

    def test_stats_count_hits_and_misses(self):
        stage = self.worktree("one")
        self.compile(stage)
        self.compile(self.worktree("two"))
        result = subprocess.run([sys.executable, str(WRAPPER), "--stats"], env=self.env, capture_output=True,
                                text=True, check=True)
        self.assertIn("hits 1 misses 1 ", result.stdout)

    def test_size_limits_parse(self):
        self.assertEqual(nvcc_cache.parse_bytes("2G"), 2 << 30)
        self.assertEqual(nvcc_cache.parse_bytes("512M"), 512 << 20)
        self.assertEqual(nvcc_cache.parse_bytes("4096"), 4096)
        self.assertEqual(nvcc_cache.parse_bytes(None), nvcc_cache.DEFAULT_MAX_BYTES)
        self.assertEqual(nvcc_cache.parse_bytes("lots"), nvcc_cache.DEFAULT_MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
