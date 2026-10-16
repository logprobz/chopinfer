"""Shared steps of the data-only recipes (tests/model_geometry_recipe.py and
tests/model_export_recipe.py): preflight, a detached worktree of the
committed HEAD with its own Cargo target, and checks that what a recipe
changed in that worktree is data whose existing records keep their bytes."""
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(*args, cwd=None, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE)
    if result.returncode:
        print(result.stdout, flush=True)
        result.check_returncode()
    return result.stdout


def assert_tests_ran(output, suites):
    results = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", output)
    assert len(results) == suites, output
    assert all(int(passed) > 0 and failed == "0" and ignored == "0"
               for passed, failed, ignored in results), output


def raw_records(text, key="name"):
    """Retain the exact bytes of each JSON object, including its whitespace."""
    decoder = json.JSONDecoder()
    position = text.index("[") + 1
    records = []
    while True:
        while text[position].isspace() or text[position] == ",":
            position += 1
        if text[position] == "]":
            return records
        value, end = decoder.raw_decode(text, position)
        records.append((value[key], text[position:end]))
        position = end


def kept_lock_records(before, after):
    """(existing, added) names of a kernel lock rewritten from `before` to
    `after`: every existing record keeps its bytes and order."""
    old = raw_records(before)
    new = raw_records(after)
    old_names = {name for name, _ in old}
    assert len({name for name, _ in new}) == len(new), "duplicate lock records"
    assert [(name, raw) for name, raw in new if name in old_names] == old, \
        "pre-existing lock record bytes or order changed"
    return old, [name for name, _ in new if name not in old_names]


def appended_records(before, after, key, label):
    """(existing, added) records of an append-only ledger rewritten from
    `before` to `after`, which must keep `before` as its byte-identical
    prefix."""
    old = raw_records(before, key)
    new = raw_records(after, key)
    assert new[:len(old)] == old, f"pre-existing {label} bytes or order changed"
    return old, new[len(old):]


def changed_paths(tree):
    """Tracked files that differ from HEAD and untracked files, ignored ones
    included (an ignored path can still hold source a build reads), sorted."""
    tracked = run("git", "diff", "HEAD", "--name-only", cwd=tree).splitlines()
    untracked = run("git", "ls-files", "--others", "--exclude-standard", cwd=tree).splitlines()
    ignored = run("git", "ls-files", "--others", "--ignored", "--exclude-standard", cwd=tree).splitlines()
    return sorted(set(tracked + untracked + ignored))


def probe(helper, checkpoint, ranks, env):
    request = {"version": 2, "request": {"command": "probe", "path": str(checkpoint),
                                         "tensor_parallel_size": ranks}}
    result = subprocess.run([str(helper), "--protocol"], input=json.dumps(request) + "\n",
                            text=True, capture_output=True, env=env, check=True)
    return json.loads(result.stdout)


def tokenizer_dir(stem):
    """The pinned tokenizer directory of the model whose variables use `stem`."""
    tokenizer = (os.environ.get(f"CHOPINFER_{stem}_TOKENIZER_DIR")
                 or os.environ.get(f"CHOPINFER_{stem}_CHECKPOINT"))
    if not tokenizer:
        raise SystemExit(f"set CHOPINFER_{stem}_TOKENIZER_DIR or CHOPINFER_{stem}_CHECKPOINT to the pinned tokenizer files")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (Path(tokenizer) / name).is_file():
            raise SystemExit(f"missing pinned tokenizer file: {name}")
    return tokenizer


def require_build_tools():
    for tool in ("git", "cargo", "nvcc"):
        if tool == "nvcc":
            path = Path(os.environ.get("CHOPINFER_CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
            if not path.is_file():
                raise SystemExit("missing CUDA compiler")
        elif not shutil.which(tool):
            raise SystemExit(f"missing required tool: {tool}")


def require_clean(root, recipe):
    if run("git", "status", "--porcelain", cwd=root).strip():
        raise SystemExit(f"{recipe} requires a clean committed HEAD")


@contextlib.contextmanager
def detached_worktree(root, prefix, target_dir=None):
    """(scratch, tree, env): a temporary directory holding a detached
    worktree of `root`'s HEAD, and the environment that builds it into
    `target_dir` (default: a target in the scratch directory) without
    incremental state, extra geometry directories or lock updates. The
    worktree and the scratch directory are removed on exit, also on
    SIGTERM once the caller turns it into SystemExit."""
    with tempfile.TemporaryDirectory(prefix=prefix, dir=os.environ.get("TMPDIR")) as temp:
        scratch = Path(temp)
        tree = scratch / "tree"
        try:
            run("git", "worktree", "add", "--detach", str(tree), "HEAD", cwd=root)
            env = os.environ.copy()
            env.update(CARGO_TARGET_DIR=str(target_dir or scratch / "target"), CARGO_INCREMENTAL="0")
            env.pop("CHOPINFER_QWEN38_GEOMETRY_DIRS", None)
            env.pop("CHOPINFER_MIMO26_GEOMETRY_DIRS", None)
            env.pop("KERNELS_LOCK_UPDATE", None)
            yield scratch, tree, env
        finally:
            if tree.exists():
                run("git", "worktree", "remove", "--force", str(tree), cwd=root)
