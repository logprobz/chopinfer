#!/usr/bin/env python3
"""A content-addressed cubin cache for CHOPINFER_NVCC_WRAPPER.

kernel-build runs `<wrapper> <nvcc> <args...>` in a per-cubin stage directory
and describes the compilation in the environment:

  CHOPINFER_NVCC_CACHE_CLOSURE    source-closure SHA-256 (every input's bytes)
  CHOPINFER_NVCC_CACHE_TOOLCHAIN  digest of nvcc, nvcc.profile and the programs
                                  and libdevice nvcc runs
  CHOPINFER_NVCC_CACHE_LOOKUP     1 when kernel-build compares the result with
                                  its lock; 0 compiles and refreshes the entry

The key is those two digests, the arguments (paths below the working directory
made relative, the output path left out) and the environment that changes
nvcc's command line or the host preprocessor (NVCC_*, SOURCE_DATE_EPOCH and the
include path variables), so a cubin built in one worktree is a hit in another.
Only arguments whose meaning the key holds are cached: an options file or any
other argument outside SAFE_ARGUMENT, in the arguments or in NVCC_PREPEND_FLAGS
and NVCC_APPEND_FLAGS, compiles without the cache and logs `bypass`. Without
CHOPINFER_NVCC_CACHE_DIR, the key variables or a single `-o` output, the
wrapper runs the compiler unchanged.

Configuration:
  CHOPINFER_NVCC_CACHE_DIR        cache directory (unset: no cache)
  CHOPINFER_NVCC_CACHE_MAX_BYTES  size bound, bytes or with a K/M/G suffix
                                  (default 10G); least recently used entries go
  CHOPINFER_NVCC_CACHE_LOG        event log (default <dir>/v1/events.log)

`nvcc_cache.py --stats [LOG]` summarizes hits, misses and compile seconds.
"""
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

FORMAT = 1
MAGIC = b"chopinfer-nvcc-cache"
DEFAULT_MAX_BYTES = 10 << 30
# Eviction leaves this fraction of the bound, so a full cache does not scan on
# every store.
LOW_WATERMARK = 0.9
STALE_TEMP_SECONDS = 3600
LOG_ROTATE_BYTES = 64 << 20
# Besides every NVCC_* variable: the host preprocessor's date and include paths.
KEY_VARIABLES = ("SOURCE_DATE_EPOCH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "GCC_EXEC_PREFIX",
                 "COMPILER_PATH")
# Arguments that read no file outside the source closure and select no other
# program. Include directories may point anywhere: nvcc -M reports what they
# supply, so the closure digest holds it.
SAFE_ARGUMENT = re.compile(
    r"-cubin|-lineinfo|-O[0-3]|-std=c\+\+[0-9]{2}|-arch=[A-Za-z0-9_]+|--fmad=(true|false)|--use_fast_math"
    r"|-D[A-Za-z_][A-Za-z0-9_]*(=.*)?|-I.+")
SAFE_PTXAS_OPTION = re.compile(r"-(O[0-3]|warn-[a-z-]+|v|lineinfo|fmad=(true|false)|maxrregcount=[0-9]+)")
EVENTS = ("hit", "miss", "corrupt", "refresh", "bypass", "store-error")


def parse_bytes(text):
    match = re.fullmatch(r"\s*(\d+)\s*([KMG]?)i?B?\s*", text or "", re.I)
    if not match:
        return DEFAULT_MAX_BYTES
    return int(match[1]) << {"": 0, "K": 10, "M": 20, "G": 30}[match[2].upper()]


def split_output(args):
    """(arguments without the output, output path) or None unless one -o."""
    rest, outputs = [], []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-o", "--output-file") and index + 1 < len(args):
            outputs.append(args[index + 1])
            index += 2
            continue
        if arg.startswith("--output-file="):
            outputs.append(arg.split("=", 1)[1])
        else:
            rest.append(arg)
        index += 1
    return (rest, outputs[0]) if len(outputs) == 1 else None


def covered(args, env):
    """Whether the key describes every input and option of this compilation."""
    try:
        extra = [token for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
                 for token in shlex.split(env.get(name, ""))]
    except ValueError:
        return False
    sources = []
    for arg in args + extra:
        if arg.startswith("-Xptxas="):
            if not all(SAFE_PTXAS_OPTION.fullmatch(option) for option in arg[len("-Xptxas="):].split(",")):
                return False
        elif not arg.startswith("-") and arg in args:
            sources.append(arg)
        elif not SAFE_ARGUMENT.fullmatch(arg):
            return False
    return (len(sources) == 1 and sources[0].endswith(".cu") and not os.path.isabs(sources[0])
            and ".." not in sources[0].split("/"))


def cache_key(args, cwd, env):
    stage = re.compile(re.escape(cwd.rstrip("/")) + r"(?=/|$)")
    material = {
        "format": FORMAT,
        "closure": env["CHOPINFER_NVCC_CACHE_CLOSURE"],
        "toolchain": env["CHOPINFER_NVCC_CACHE_TOOLCHAIN"],
        "args": [stage.sub(".", arg) for arg in args],
        "environment": {name: value for name, value in env.items()
                        if name.startswith("NVCC_") or name in KEY_VARIABLES},
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class Cache:
    def __init__(self, root, env):
        self.root = os.path.join(os.path.abspath(root), f"v{FORMAT}")
        self.max_bytes = parse_bytes(env.get("CHOPINFER_NVCC_CACHE_MAX_BYTES"))
        self.default_log = os.path.join(self.root, "events.log")
        self.log_path = env.get("CHOPINFER_NVCC_CACHE_LOG") or self.default_log

    def entry(self, key):
        return os.path.join(self.root, key[:2], key)

    def temporary(self, directory, name):
        return os.path.join(directory, f".{name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")

    def read(self, key):
        """(payload, compile seconds) of a verified entry, or None."""
        path = self.entry(key)
        try:
            with open(path, "rb") as file:
                data = file.read()
        except OSError:
            return None
        header, newline, payload = data.partition(b"\n")
        fields = header.split(b" ")
        try:
            if not newline or len(fields) != 5 or fields[0] != MAGIC or int(fields[1]) != FORMAT:
                raise ValueError
            size, seconds = int(fields[3]), float(fields[4])
        except ValueError:
            return "corrupt"
        if size != len(payload) or hashlib.sha256(payload).hexdigest().encode() != fields[2]:
            return "corrupt"
        try:
            os.utime(path)
        except OSError:
            pass
        return payload, seconds

    def write(self, key, payload, seconds):
        """Publish an entry atomically, then keep the cache within its bound."""
        directory = os.path.dirname(self.entry(key))
        os.makedirs(directory, exist_ok=True)
        header = b"%s %d %s %d %.3f\n" % (MAGIC, FORMAT, hashlib.sha256(payload).hexdigest().encode(),
                                          len(payload), seconds)
        temporary = self.temporary(directory, key)
        try:
            with open(temporary, "wb") as file:
                file.write(header + payload)
            os.replace(temporary, self.entry(key))
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self.account(len(header) + len(payload))

    def account(self, added):
        """Track the cache size under a lock; scan and evict above the bound.

        The running total over-counts replaced entries, which only brings the
        next scan forward. Readers that opened an entry keep reading it after
        eviction unlinks it."""
        with open(os.path.join(self.root, "lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            total_path = os.path.join(self.root, "bytes")
            try:
                with open(total_path) as file:
                    total = int(file.read() or 0)
            except (OSError, ValueError):
                total = None
            total = self.evict() if total is None or total + added > self.max_bytes else total + added
            temporary = self.temporary(self.root, "bytes")
            with open(temporary, "w") as file:
                file.write(f"{total}\n")
            os.replace(temporary, total_path)

    def evict(self):
        """Remove least recently used entries down to the low watermark."""
        entries, now = [], time.time()
        for shard in os.scandir(self.root):
            if not shard.is_dir(follow_symlinks=False):
                continue
            for item in os.scandir(shard.path):
                try:
                    status = item.stat(follow_symlinks=False)
                except OSError:
                    continue
                if item.name.startswith("."):
                    if now - status.st_mtime > STALE_TEMP_SECONDS:
                        unlink(item.path)
                    continue
                entries.append((status.st_mtime, status.st_size, item.path))
        total = sum(size for _, size, _ in entries)
        if total > self.max_bytes:
            target = int(self.max_bytes * LOW_WATERMARK)
            for _, size, path in sorted(entries):
                if total <= target:
                    break
                unlink(path)
                total -= size
        return total

    def rotate(self):
        """Keep the default log under its bound. Under the cache lock one
        writer renames it; appends that opened the old file land in `.1`."""
        try:
            if os.path.getsize(self.default_log) <= LOG_ROTATE_BYTES:
                return
            with open(os.path.join(self.root, "lock"), "a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if os.path.getsize(self.default_log) > LOG_ROTATE_BYTES:
                    os.replace(self.default_log, self.default_log + ".1")
        except OSError:
            pass

    def log(self, event, key, seconds, name):
        line = f"{time.time():.3f} {event} {key} {seconds:.3f} {name}\n"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            if self.log_path == self.default_log:
                self.rotate()
            descriptor = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(descriptor, line.encode())
            finally:
                os.close(descriptor)
        except OSError as error:
            print(f"nvcc cache: cannot write {self.log_path}: {error}", file=sys.stderr)


def unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def publish(path, payload):
    """Write the cubin where nvcc would have, through a rename beside it."""
    directory, name = os.path.split(os.path.abspath(path))
    temporary = os.path.join(directory, f".{name}.{os.getpid()}.nvcc-cache.tmp")
    with open(temporary, "wb") as file:
        file.write(payload)
    os.replace(temporary, path)


def compile_command(argv, env):
    return subprocess.run(argv, env=env).returncode


def run(argv, env):
    """Compile argv[0] with argv[1:], through the cache when it applies."""
    root = env.get("CHOPINFER_NVCC_CACHE_DIR")
    keyed = all(env.get(name) for name in ("CHOPINFER_NVCC_CACHE_CLOSURE", "CHOPINFER_NVCC_CACHE_TOOLCHAIN"))
    split = split_output(argv[1:])
    if not root or not keyed or split is None:
        return compile_command(argv, env)
    args, output = split
    cache = Cache(root, env)
    key = cache_key(args, os.path.realpath(os.getcwd()), env)
    name = os.path.basename(output)
    if not covered(args, env):
        start = time.monotonic()
        status = compile_command(argv, env)
        cache.log("bypass", key, time.monotonic() - start, name)
        return status
    lookup = env.get("CHOPINFER_NVCC_CACHE_LOOKUP", "1") != "0"
    if lookup:
        found = cache.read(key)
        if isinstance(found, tuple):
            try:
                publish(output, found[0])
            except OSError as error:
                print(f"nvcc cache: cannot publish a hit: {error}", file=sys.stderr)
            else:
                cache.log("hit", key, found[1], name)
                return 0
        elif found == "corrupt":
            cache.log("corrupt", key, 0.0, name)
    start = time.monotonic()
    status = compile_command(argv, env)
    seconds = time.monotonic() - start
    if status != 0:
        return status
    try:
        with open(output, "rb") as file:
            payload = file.read()
        cache.write(key, payload, seconds)
        cache.log("miss" if lookup else "refresh", key, seconds, name)
    except OSError as error:
        print(f"nvcc cache: not stored: {error}", file=sys.stderr)
        cache.log("store-error", key, seconds, name)
    return 0


def stats(log_path):
    counts = {event: 0 for event in EVENTS}
    seconds = {event: 0.0 for event in EVENTS}
    with open(log_path) as file:
        for line in file:
            fields = line.split()
            if len(fields) >= 4 and fields[1] in counts:
                counts[fields[1]] += 1
                seconds[fields[1]] += float(fields[3])
    lookups = counts["hit"] + counts["miss"] + counts["corrupt"]
    rate = counts["hit"] / lookups if lookups else 0.0
    print(f"hits {counts['hit']} misses {counts['miss']} corrupt {counts['corrupt']} "
          f"refreshes {counts['refresh']} bypasses {counts['bypass']} store errors {counts['store-error']} "
          f"hit rate {rate:.3f}")
    print(f"compile seconds {seconds['miss'] + seconds['refresh'] + seconds['bypass']:.1f} "
          f"saved compile seconds {seconds['hit']:.1f}")
    return 0


def main(argv, env):
    if argv[:1] == ["--stats"]:
        if len(argv) > 1:
            return stats(argv[1])
        if not env.get("CHOPINFER_NVCC_CACHE_DIR") and not env.get("CHOPINFER_NVCC_CACHE_LOG"):
            print("usage: nvcc_cache.py --stats [LOG], or set CHOPINFER_NVCC_CACHE_DIR", file=sys.stderr)
            return 2
        return stats(Cache(env.get("CHOPINFER_NVCC_CACHE_DIR", "."), env).log_path)
    if not argv:
        print("usage: nvcc_cache.py <nvcc> <args...>", file=sys.stderr)
        return 2
    return run(argv, env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
