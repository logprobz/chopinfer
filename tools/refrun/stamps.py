"""Checkpoint files verified against their source lock by content.

Every file a run reads must have the lock's byte length and sha256. Hashing a
whole checkpoint takes minutes, so a stamp cache keeps each file's digest
under its stamp (resolved path, size, mtime_ns, ctime_ns, inode) and a file
is rehashed only when its stamp changes.

A stamp identifies content only once it has settled: timestamps advance in
coarse ticks, so two writes within one tick leave the same stamp. A file is
therefore read only after its newest timestamp is RACY_NS older than the
clock, waiting if needed (git's racy index entries). Any later write lands in
a later tick and changes the ctime, which no caller can set back. So:

- a digest computed from a settled stamp that is unchanged after hashing is
  the digest of the bytes read, and is cached under that stamp;
- a file whose stamp still equals its verified stamp after a read held the
  verified bytes during that read (`confirm`).

The runner verifies every file before it reads any of them and confirms the
stamps before it writes each output. This assumes file timestamps come from
this host's clock in ticks of at most RACY_NS (local filesystems: at most a
few ms), that the clock is not stepped back by more than RACY_NS during a
run, and that no directory on the resolved checkpoint path is renamed during
a run. Symlinked files (HF cache snapshots) are resolved per file, and the
link itself is part of what `confirm` compares.

The cache holds one JSON record per path in $CHOPINFER_DATA/refcache/v1/stamps;
without a cache directory every file is hashed.

    python3 -m tools.refrun.stamps --checkpoint DIR --lock LOCK [--lock-variant V]

verifies every locked file and fills the cache, on CPU, before a GPU job.

This module does not import torch.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from tools.refrun import cache, keys

THREADS = 8
RACY_NS = 2_000_000_000


def default_cache():
    root = cache.default_root()
    return root / "stamps" if root is not None else None


def stamp(path):
    info = os.stat(path)
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns, "inode": info.st_ino}


def settled(path):
    """The file's stamp once its newest timestamp is RACY_NS older than the
    clock, read before the stamp so a write between the two only waits longer."""
    while True:
        now = time.time_ns()
        current = stamp(path)
        wait = max(current["mtime_ns"], current["ctime_ns"]) + RACY_NS - now
        if wait <= 0:
            return current
        if wait > 2 * RACY_NS:
            raise ValueError(f"{path} has a timestamp ahead of this host's clock")
        time.sleep(wait / 1e9)


def digest(path, cache=None):
    """(sha256, stamp) of a file with a settled stamp, from the cache when the
    stamp is cached. A file that changes while it is hashed is refused."""
    path = Path(path).resolve()
    before = settled(path)
    record = None if cache is None else Path(cache) / f"{keys.sha256(str(path).encode())}.json"
    if record is not None and record.exists():
        cached = json.loads(record.read_bytes())
        if cached["stamp"] == before:
            return cached["sha256"], before
    value = keys.file_sha256(path)
    if stamp(path) != before:
        raise RuntimeError(f"{path} changed while it was hashed")
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=record.parent, suffix=".tmp", delete=False) as stream:
            json.dump({"stamp": before, "sha256": value}, stream, sort_keys=True)
        os.replace(stream.name, record)
    return value, before


def link(path):
    """The directory entry itself: replacing a symlink gives a new inode or ctime."""
    info = os.lstat(path)
    return {"inode": info.st_ino, "ctime_ns": info.st_ctime_ns}


def verify(root, lock, files, cache=None):
    """Each file under `root` is in `lock` with its locked size and sha256.
    Returns {file: verified stamps} for `confirm`."""
    root = Path(root)
    for file in files:
        entry = lock.get(file)
        size = (root / file).stat().st_size
        if entry is None or entry["byte_length"] != size:
            raise ValueError(f"{file} ({size} bytes) does not match the checkpoint lock")
    links = {file: link(root / file) for file in files}
    with ThreadPoolExecutor(min(THREADS, max(len(files), 1))) as pool:
        digests = list(pool.map(lambda file: digest(root / file, cache), files))
    verified = {}
    for file, (value, target) in zip(files, digests):
        if value != lock[file]["sha256"]:
            raise ValueError(f"{file} has sha256 {value}, not the checkpoint lock's {lock[file]['sha256']}")
        verified[file] = {"link": links[file], "target": target}
    confirm(root, verified)
    return verified


def confirm(root, verified):
    """Every verified file still has its verified stamps, so every read since
    `verify` returned saw the verified bytes."""
    root = Path(root)
    for file, expected in verified.items():
        path = root / file
        if link(path) != expected["link"] or stamp(path.resolve()) != expected["target"]:
            raise RuntimeError(f"{file} changed after it was verified")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--lock-variant")
    parser.add_argument("--stamps", type=Path, default=default_cache(),
                        help="stamp cache directory (default $CHOPINFER_DATA/refcache/v1/stamps)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    lock = keys.read_lock(args.lock)
    if args.lock_variant is not None:
        lock = lock[args.lock_variant]
    verify(args.checkpoint, lock, sorted(lock), args.stamps)
    print(f"{len(lock)} files match {args.lock}")


if __name__ == "__main__":
    main()
