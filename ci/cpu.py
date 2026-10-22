"""The full CPU checks as a dependency graph of supervised steps.

Each Cargo feature variant replaces the binaries in target/debug, so steps
that build into or read the shared target directory follow an order:
1. the profile CLI build, whose binaries are copied for their readers
   (qualification, model preparation, the external SDK smoke, serve
   profiles, Python scorer consumers and the reference checks), which then
   run beside everything that follows;
2. the portable checks and workspace tests (xtask check-portable), whose
   tests rebuild some target/debug binaries;
3. the fixture feature variants, then the kernel report integration and the
   feature variants that build only test executables;
4. the per-package variants that rebuild target/debug binaries, one at a
   time, each with its readers;
5. with --exhaustive, the exhaustive comparisons the gate leaves #[ignore]d.
Steps that use their own target directory or no Cargo build run alongside
from the start. ci/cpu.sh sets up the environment and runs this file."""
import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from xml.etree import ElementTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_checks
import preflight
import runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ["python3", "-B"]
LOCKED = ["--locked", "--offline"]
# Nextest writes each profile's JUnit report under the workspace's target
# directory, whatever CARGO_TARGET_DIR says.
NEXTEST_REPORTS = [ROOT / "target/nextest" / profile / "junit.xml" for profile in ("ci", "exhaustive")]
# A warm gate that takes longer than this gets a warning (docs/contributing.md#checks).
WARM_GATE_SECONDS = 15 * 60
# How often a run checks that HEAD is still the commit it started on.
HEAD_POLL_SECONDS = 10
# Independent steps that build Rust in fresh targets of their own. ci/quick.sh
# runs the other independent steps a change affects and leaves these to the gate.
FRESH_BUILDS = ("geometry-recipe", "export-recipe", "geometry-recipe-contract", "release-gate-clean",
                "cuda-toolchain")
# Steps that only compile Python. A file they alone cover still lacks a check
# that runs it.
COMPILE_ONLY = ("qual-python-compile", "refrun-python-compile")
# The qualification run (one cell per tools/qual module and gate config).
# ci/quick.sh runs it for the modules a change affects.
QUALIFICATION = "release-gate"
# Steps that check the setup of the steps that run after them. ci/quick.sh and
# `--only` run one whenever they run such a step.
PREREQUISITES = ("reference-prerequisites",)
# Steps that prepare what later steps read. ci/quick.sh runs the ones its
# selected checks run after.
SETUP = ("profile-cli-build", "preparation-fixture-build", "profile-copy") + PREREQUISITES
# Steps that read what other steps leave in the shared target or the work
# directory, beyond SETUP. `--only` refuses such a step without its producers,
# or adds them with --with-producers (docs/contributing.md#checks).
PRODUCERS = {
    # The workspace tests' kernel build reports and the external SDK smoke's.
    "kernel-report-integration": ("portable", "external-model-smoke"),
    # The runtime-host binaries the lifecycle test builds with the fixture services.
    "deploy-entrypoint": ("process-lifecycle",),
}
# The reference checks are required: each must execute tests and may skip only
# cases whose reason names the opt-in input that enables them, a checkpoint or
# a model's extra checkpoints (CHOPINFER_CI_CHECKPOINT_TESTS=1) or a CUDA round
# trip, and names no missing dependency (torch, transformers, numpy or the
# reference runner).
NO_DEPENDENCY = r"(?!(?i:.*(?:torch|transformers|numpy|refrun)))"
REFERENCE_SKIPS = (NO_DEPENDENCY + r".*\bCHOPINFER_\w+_(?:CHECKPOINTS?|TOKENIZER_DIR)\b.*",
                   NO_DEPENDENCY + r".*\bCHOPINFER_\w+_CUDA=1\b.*")

PROFILE_COPY = r"""
import os, shutil, sys
source, destination, *names = sys.argv[1:]
for name in names:
    copy = os.path.join(destination, name)
    os.makedirs(os.path.dirname(copy), exist_ok=True)
    shutil.copy2(os.path.join(source, name), copy)
"""


def steps(target, work, jobs, test_threads, exhaustive=False):
    """(shared-target steps, independent steps), each in priority order."""
    debug = target / "debug"
    # The readers of the profile build use a copy of its binaries, which the
    # workspace tests and later variant builds cannot replace while they run.
    profile = work / "profile-target"
    smoke = work / "external-sdk"
    # A step's weight is about the cores it keeps busy on average; its own
    # limits allow bursts above that. The Rust tests run test_threads at once.
    # No step counts for more than half the budget: a step weighed at the
    # whole budget starts only once no other step runs, so the workspace
    # tests at --test-threads equal to --jobs waited for every long
    # independent step (docs/contributing.md#checks).
    # Qualification runs this many modules at once, and the geometry lanes
    # share the whole budget for their Cargo and nvcc jobs, whose builds are
    # short bursts between single-threaded checks. A small budget (ci/quick.sh
    # in an 8-CPU lane) still runs up to 4 modules at once: one at a time, the
    # qualification run was the longest check of a broad change.
    qualification_jobs = max(jobs // 6, min(jobs // 2, 4), 1)
    env = os.environ.copy()

    def step(name, argv, after=(), weight=1, skips=None, **variables):
        return runner.Step(name, argv, env={**env, **variables} if variables else None,
                           cwd=str(ROOT), weight=min(weight, max(jobs // 2, 1)), after=after, skips=skips)

    # The reference runner's tests need torch and tapdiff's need numpy, from
    # CHOPINFER_REFRUN_PYTHON (the runner's venv): its check runs first and
    # refuses a missing or incomplete one. CPU only. Each starts a thread per
    # core, so they run one at a time. Some score through `chopinfer eval`;
    # in the gate they read the profile copy, which no later build replaces.
    python = os.environ.get("CHOPINFER_REFRUN_PYTHON") or "python3"
    scoring = {"CHOPINFER_EVAL": str(profile / "debug/chopinfer")}
    readers = ("profile-copy",)
    # Model reference tests, compile inputs and profile targets come from the
    # model inventories (ci/model_checks.py), which refuse an empty plan.
    planned = model_checks.plan()
    references, previous = [step("reference-prerequisites", PY + ["ci/model_checks.py", "prerequisites"])], ()
    for name, argv in [
            ("refrun", [python, "-B", "-m", "unittest", "discover", "-s", "tools/refrun", "-t", ".", "-v"]),
            *planned["references"],
            ("tapdiff", [python, "-B", "-m", "unittest", "tools.qual.test_tapdiff",
                         "tools.qual.test_encoder_parity_gate", "-v"])]:
        variables = dict(scoring) if name == "tapdiff" else {"CUDA_VISIBLE_DEVICES": "", **scoring}
        # Each names the copy, so any selection of them waits for it.
        references.append(step(name, argv, after=PREREQUISITES + readers + previous, skips=REFERENCE_SKIPS,
                               **variables))
        previous = (name,)

    independent = [
        # Each leg builds in its own temporary worktree and target.
        step("geometry-recipe", ["bash", "tests/model-geometry-recipe.sh", "--ci"], weight=max(jobs // 6, 2),
             CHOPINFER_GEOMETRY_JOBS=str(jobs)),
        # Builds in its own temporary worktree and target, with the share of
        # the budget a geometry lane gets for Cargo jobs and nvcc workers.
        step("export-recipe", ["bash", "tests/model-export-recipe.sh", "--ci"], weight=max(jobs // 6, 2),
             CARGO_BUILD_JOBS=str(max(jobs // 2, 1)), CHOPINFER_KERNEL_BUILD_JOBS=str(max(jobs // 2, 1))),
        # Contract-stage legs (models whose bundle cannot yet serve an added geometry), in their
        # own worktree and temporary target.
        step("geometry-recipe-contract", ["bash", "tests/model-geometry-recipe.sh", "--contract"], weight=2),
        # Builds its scorer in a private copy of the checkout and target.
        step("release-gate-clean", ["bash", "tests/qual-clean.sh"], weight=max(qualification_jobs // 2, 1),
             CHOPINFER_CI_JOBS=str(qualification_jobs)),
        # Its toolkit wrapper reruns runtime-kernels' build script, so it
        # builds in a private target instead of invalidating the shared one.
        step("cuda-toolchain", ["bash", "tests/cuda-toolchain.sh"], weight=2,
             CARGO_TARGET_DIR=str(work / "cuda-toolchain-target")),
        # kernel-core's NaN-order test needs an unoptimized build, in a target
        # below the shared one.
        step("kernel-core-unoptimized", ["bash", "tests/kernel-core-unoptimized.sh"]),
        step("concurrency-control", PY + ["tests/test_concurrency_control.py"]),
        step("heavy-launcher", PY + ["tests/test_heavy.py"]),
        step("validation-report", PY + ["tests/test_validation.py"], readers, **scoring),
        step("leak-guard", PY + ["tests/leak_guard.py"]),
        step("leak-guard-push", PY + ["tests/test_leak_guard.py"]),
        step("portable-policy", PY + ["-m", "unittest", "tests/test_portable_runtime_policy.py"]),
        step("quickstart", PY + ["tests/test_quickstart.py"]),
        step("checkpoint-environment", PY + ["tests/test_cpu_environment.py"]),
        step("encoder-membership", ["bash", "tests/encoder-membership.sh"]),
        step("geometry-recipe-guards", PY + ["tests/test_model_geometry_recipe.py"]),
        step("export-recipe-guards", PY + ["tests/test_model_export_recipe.py"]),
        step("quick-selection", PY + ["tests/test_quick.py"]),
        step("ci-runner", PY + ["tests/test_ci_runner.py"]),
        step("kernel-report-reader", PY + ["tests/test_kernel_reports.py"]),
        step("benchmark-loop", PY + ["tests/test_benchmark_loop.py"]),
        step("startup-timing", PY + ["tools/dev/test_startup_timing.py", "-v"]),
        step("model-inventory", PY + ["tests/test_model_inventory.py"]),
        step("kernel-lab-inventory", PY + ["ci/kernel_lab_inventory.py"]),
        step("model-check-plans", PY + ["tests/test_model_checks.py"]),
        step("external-model-policy", PY + ["tests/test_external_model_policy.py"]),
        step("external-model-sdk", PY + ["tests/test_model_sdk.py"]),
        step("project-layout", ["bash", "tests/project-layout.sh"]),
        step("bundle-boundary", ["bash", "tests/bundle-boundary.sh"]),
        step("model-neutral-host", ["bash", "tests/model-neutral-host.sh"]),
        step("release-notices", PY + ["tools/release/notices.py", str(work / "third-party-notices")]),
        step("release-policy", PY + ["tests/test_release_policy.py"]),
        step("core-guards", PY + ["tests/test_core_guards.py"]),
        step("mimo-redeploy-lock", PY + ["tests/test_mimo_redeploy_lock.py"]),
        step("redeploy-wait", PY + ["tests/test_redeploy_wait.py"]),
        step("fixed-batch-logits", PY + ["tests/test_fixed_batch_logits.py"]),
        step("fixed-state-replay", PY + ["tests/test_fixed_state_replay.py"]),
        step("lifecycle-diagnostics", PY + ["tests/test_lifecycle_diagnostics.py"]),
        step("encoder-adoptions", PY + ["tests/encoder-adoptions.py"]),
        step("encoder-adoptions-check", PY + ["tests/test_encoder_adoptions.py"]),
        step("projection-adoptions", PY + ["tests/projection-adoptions.py"]),
        step("projection-adoptions-check", PY + ["tests/test_projection_adoptions.py"]),
        step("projection-schedules", PY + ["tests/projection-schedules.py"]),
        step("projection-schedules-check", PY + ["tests/test_projection_schedules.py"]),
        step("qwen38-topology-lab", PY + ["tests/test_qwen38_topology_lab.py"]),
        *references,
        # Declaration validation and limit constants can score during module
        # import, before setUpModule. Pass the fresh copy to the interpreter.
        step("spec-bench", PY + ["tests/test_spec_bench.py"], readers, **scoring),
        step("spec-verify", PY + ["tests/test_spec_verify.py"]),
        step("profile-tools", PY + ["-m", "unittest", "discover", "-s", "tools/profile", "-p", "test_*.py", "-v"],
             readers, **scoring),
        step("nvcc-cache", PY + ["-m", "unittest", "discover", "-s", "tools/nvcc_cache", "-p", "test_*.py", "-v"]),
        step("journal-proof", PY + ["-m", "unittest", "discover", "-s", "tools/journal_proof", "-p", "test_*.py", "-v"]),
        step("deploy-artifacts", PY + ["tests/test_deploy_artifacts.py"], readers, **scoring),
        step("deploy-policy-equality", PY + ["tests/test_deploy_policy_equality.py"]),
        step("sass-same", ["bash", "crates/kernel-lab/tools/test_sass_same.sh"]),
        step("ab-serving", PY + ["-m", "unittest", "discover", "-s", "tools/ab/serving", "-p", "test_*.py", "-v"]),
        step("smoke-tools", PY + ["-m", "unittest", "discover", "-s", "tools/smoke", "-p", "test_*.py", "-v"]),
        step("qual-python-compile", ["bash", "-c", "python3 -m py_compile tools/qual/*.py tools/qual/dev/*.py"],
             PYTHONPYCACHEPREFIX=str(work / "qual-pycache")),
        step("refrun-python-compile", planned["compile"], PYTHONPYCACHEPREFIX=str(work / "refrun-pycache")),
        step("cuda-host-abi", ["bash", "-c", CUDA_HOST_ABI, "cuda-host-abi", str(work / "cuda-host-abi")]),
    ]
    profile_build = planned["profile"]
    built = [profile_build[index + 1] for index, argument in enumerate(profile_build) if argument == "--bin"]
    built += ["lib" + profile_build[index - 1].replace("-", "_") + ".so"
              for index, argument in enumerate(profile_build) if argument == "--lib"]
    built.append("examples/chopinfer-model-fixture")
    fixtures = ("recurrent-fixture", "tp4-fixture")
    test_variants = ("runtime-metrics", "cuda-fake-contracts", "cuda-launch-arguments", "bundle-abi-contract",
                     "host-cuda-fake-driver")
    shared = [
        step("profile-cli-build", profile_build, weight=4),
        step("preparation-fixture-build", ["cargo", "build", *LOCKED, "-p", "bundle-tp4-fixture", "--example",
                                           "chopinfer-model-fixture"], ("profile-cli-build",), weight=4),
        step("profile-copy", PY + ["-c", PROFILE_COPY, str(debug), str(profile / "debug"), *built],
             ("preparation-fixture-build",)),
        step("portable", ["cargo", "run", "-p", "xtask", *LOCKED, "--", "check-portable"],
             readers, weight=test_threads, CHOPINFER_CI_TEST_THREADS=str(test_threads),
             CHOPINFER_RUST_TEST_RUNNER=env.get("CHOPINFER_RUST_TEST_RUNNER") or preflight.GATE_RUNNER),
        step("release-gate", ["bash", "ci/qualification.sh", "--jobs", str(qualification_jobs)], readers,
             weight=max(qualification_jobs // 2, 1), CARGO_TARGET_DIR=str(profile), **scoring),
        step("model-preparation", PY + ["tests/test_model_preparation.py"], readers, weight=2,
             CARGO_TARGET_DIR=str(profile)),
        step("external-model-smoke", PY + ["-m", "tools.model_sdk.smoke", "--source", str(ROOT), "--output",
                                           str(smoke / "smoke"), "--host", str(profile / "debug/chopinfer"),
                                           "--conformance-host", str(profile / "debug/chopinfer-host")],
             readers, weight=2),
        step("serve-profiles", PY + ["tests/test_serve_profiles.py"], readers, CARGO_TARGET_DIR=str(profile)),
        step("recurrent-fixture", ["cargo", "test", "-p", "bundle-recurrent-fixture", "--features",
                                   "fake-host-services", *LOCKED], ("portable",), weight=4),
        step("tp4-fixture", ["cargo", "test", "-p", "bundle-tp4-fixture", "--features", "fake-host-services",
                             *LOCKED], ("portable",), weight=4),
        # Reads the kernel build reports in the shared target directory.
        step("kernel-report-integration", PY + ["tests/test_kernel_report_integration.py"],
             fixtures + ("external-model-smoke",), CHOPINFER_KERNEL_REPORT_TARGET=str(target),
             CHOPINFER_SDK_REPORT_TARGET=str(smoke / "smoke/target"),
             CHOPINFER_SDK_REPORT_WORKSPACE=str(smoke / "smoke/model")),
        # Feature variants of packages without binaries build only hashed test
        # executables, never target/debug binaries. The fake driver has shared
        # state, so its selected tests run serially; one test executable
        # serves the existing filters.
        step("runtime-metrics", ["bash", "tests/runtime-metrics.sh"], fixtures, weight=2),
        step("cuda-fake-contracts", ["cargo", "test", "-p", "runtime-cuda", "--lib", "--no-default-features",
                                     "--features", "fake-driver-tests", *LOCKED, "--",
                                     "tests::tensor_maps_of_every_rank_validate_before_encoding",
                                     "tests::every_tensor_map_element_reaches_the_encoder",
                                     "tests::invalid_tensor_maps_cost_no_driver_encode",
                                     "nvdec::tests::a_host_without_the_decoder_library_is_refused_cleanly",
                                     "tests::mapped_host_buffers_free_only_after_a_successful_fence",
                                     "tests::lent_memory_is_shared_through_its_descriptor_and_released_by_its_last_owner",
                                     "tests::a_failed_graph_release_keeps_the_graph_live_for_a_retry",
                                     "page_copy::tests", "--test-threads=1"], fixtures, weight=2),
        step("cuda-launch-arguments", ["cargo", "test", "-p", "runtime-cuda", "--lib", *LOCKED,
                                       "tests::each_launch_argument_marshals_at_the_width_it_names"], fixtures),
        step("bundle-abi-contract", ["cargo", "test", "-p", "runtime-bundle-abi", "--test", "abi_contract", *LOCKED],
             fixtures, weight=2),
        step("host-cuda-fake-driver", ["cargo", "test", "-p", "runtime-host", "--features", "cuda-fake-driver-tests",
                                       "--lib", *LOCKED], fixtures, weight=2),
        # These rebuild runtime-host's binaries for runtime-host alone, one at a
        # time. The lifecycle test's build adds the fixture services, and the
        # deploy entrypoint test reads that build.
        step("process-lifecycle", PY + ["tests/test_process_lifecycle.py"],
             ("kernel-report-integration",) + test_variants, weight=2),
        step("deploy-entrypoint", PY + ["tests/test_deploy_entrypoint.py"], ("process-lifecycle",)),
        step("host-launch-arguments", ["cargo", "test", "-p", "runtime-host", "--test", "launch_argument_contract",
                                       *LOCKED], ("deploy-entrypoint",), weight=2),
    ]
    if exhaustive:
        # The exhaustive profile in .config/nextest.toml names the tests and
        # keeps the ci time budget. It reuses the workspace test build.
        shared.append(step("exhaustive-comparisons", [
            "cargo", "nextest", "run", "--workspace", "--profile", "exhaustive", "--run-ignored", "only",
            "--no-tests", "fail", "--test-threads", str(test_threads), *LOCKED],
            ("host-launch-arguments",), weight=test_threads))
    return shared, independent


CUDA_HOST_ABI = r"""
set -Eeuo pipefail
test=$1
/usr/bin/gcc -std=c11 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${CHOPINFER_CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/include" \
  -Icrates/runtime-cuda/csrc \
  crates/runtime-cuda/csrc/cuda_host_abi_fake_driver_test.c \
  crates/runtime-cuda/csrc/tensor_map.c \
  -o "$test"
"$test"
"""


def order(shared, independent):
    """All steps in priority order: the two longest independent steps, the
    shared-target chain, then the rest."""
    return independent[:2] + shared + independent[2:]


def select(planned, names):
    """The named steps and their transitive setup. Each waits for the
    selected steps it follows, directly or through unselected ones."""
    chosen = set(names)
    steps = {item.name: item for item in planned}
    pending = list(chosen)
    while pending:
        for name in steps[pending.pop()].after:
            if name in SETUP and name not in chosen:
                chosen.add(name)
                pending.append(name)
    nearest = {}

    def follows(name):
        if name not in nearest:
            found = []
            for before in steps[name].after:
                for step in (before,) if before in chosen else follows(before):
                    if step not in found:
                        found.append(step)
            nearest[name] = tuple(found)
        return nearest[name]

    after = {name: follows(name) for name in chosen}
    planned = [item for item in planned if item.name in chosen]
    for item in planned:
        item.after = after[item.name]
    return planned


def with_producers(names):
    """`names` and the steps whose outputs they read (PRODUCERS), transitively."""
    chosen, pending = list(names), list(names)
    while pending:
        for producer in PRODUCERS.get(pending.pop(), ()):
            if producer not in chosen:
                chosen.append(producer)
                pending.append(producer)
    return chosen


def missing_producers(names):
    """{step: its unselected producers} for the steps among `names` that read another's outputs."""
    return {name: [producer for producer in PRODUCERS[name] if producer not in names]
            for name in names if any(producer not in names for producer in PRODUCERS.get(name, ()))}


class HeadWatch(threading.Thread):
    """Stops this run with SIGTERM once the checkout at `root` reads as
    another commit than `head`, until stopped; an unreadable HEAD is checked
    again at the next poll and at the end."""

    def __init__(self, root, head, seconds=HEAD_POLL_SECONDS):
        super().__init__(daemon=True)
        self.root, self.head, self.seconds = root, head, seconds
        self.moved = None
        self.done = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        while not self.done.wait(self.seconds):
            now = preflight.head(self.root)
            if now is not None and now != self.head:
                with self.lock:
                    if not self.done.is_set():
                        self.moved = now
                        os.kill(os.getpid(), signal.SIGTERM)
                return

    def stop(self):
        """No signal is sent after this returns."""
        with self.lock:
            self.done.set()


def interrupted(signum, frame):
    raise runner.Interrupted(signum)


def head_moved(head, now, out):
    print(f"cpu-ci failed: HEAD moved during the run from {head} to {now or 'an unreadable HEAD'}; its steps "
          "did not all test one commit. Commit, rebase or check out elsewhere, then run the gate again.", file=out)


def slowest_tests(reports, since, count=10):
    """The `count` longest test cases, as (seconds, binary, name), in the
    JUnit reports written since `since` (a time.time() value)."""
    cases = []
    for path in reports:
        try:
            if path.stat().st_mtime < since:
                continue
            root = ElementTree.parse(path).getroot()
        except (OSError, ElementTree.ParseError):
            continue
        cases += [(float(case.get("time") or 0), case.get("classname") or "", case.get("name") or "")
                  for case in root.iter("testcase")]
    return sorted(cases, key=lambda case: (-case[0], case[1], case[2]))[:count]


def report_times(elapsed, slowest, full_gate, out):
    if slowest:
        print("cpu-ci slowest tests:", file=out)
        for seconds, binary, name in slowest:
            print(f"  {seconds:9.3f}s {binary} {name}", file=out)
    if full_gate and elapsed > WARM_GATE_SECONDS:
        print(f"cpu-ci warning: the gate took {elapsed / 60:.1f} minutes; a warm gate should finish within "
              f"{WARM_GATE_SECONDS // 60}. Unless the target was cold or the host busy, check the slowest tests "
              "and steps (docs/contributing.md#checks).", file=out)


def remove_tree(path):
    # The external SDK smoke writes read-only source copies.
    subprocess.run(["chmod", "-R", "u+w", "--", str(path)], stderr=subprocess.DEVNULL, check=False)
    shutil.rmtree(path, ignore_errors=True)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("CHOPINFER_CI_JOBS", os.cpu_count() or 8)),
                        help="Worker budget shared by all steps (default: the CPU count, or CHOPINFER_CI_JOBS)")
    parser.add_argument("--test-threads", type=int,
                        default=int(os.environ.get("CHOPINFER_CI_TEST_THREADS", "16")),
                        help="Concurrent Rust tests under nextest (default 16, or CHOPINFER_CI_TEST_THREADS)")
    parser.add_argument("--serial", action="store_true",
                        help="Run one step at a time; each step keeps its share of the budget")
    parser.add_argument("--keep-going", action="store_true", help="Run every step the failures do not block")
    parser.add_argument("--exhaustive", action="store_true",
                        default=os.environ.get("CHOPINFER_CI_EXHAUSTIVE") == "1",
                        help="Also run the exhaustive comparisons the gate ignores (or CHOPINFER_CI_EXHAUSTIVE=1)")
    parser.add_argument("--only", action="append", default=[], help="Run only these steps (repeatable)")
    parser.add_argument("--with-producers", action="store_true",
                        help="With --only, also run the steps whose outputs the selected steps read")
    parser.add_argument("--work", type=pathlib.Path, help="Keep step logs in this directory")
    parser.add_argument("--summary", type=pathlib.Path, default=os.environ.get("CHOPINFER_CI_SUMMARY"),
                        help="Write each step's status, time, CPU time and peak memory as JSON")
    parser.add_argument("--graph", action="store_true",
                        help="Print every step's prerequisites (name -> steps it runs after) as JSON and run nothing")
    parser.add_argument("--preflight", action="store_true",
                        help="Check the environment the selected steps need (TMPDIR, reference variables) and run "
                             "nothing; every run checks it first")
    args = parser.parse_args(arguments)
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR") or ROOT / "target")
    # The steps only name paths under the work directory; nothing is created.
    shared, independent = steps(target, args.work or pathlib.Path(tempfile.gettempdir()) / "chopinfer-cpu-ci.graph",
                                args.jobs, args.test_threads, args.exhaustive)
    if args.graph:
        print(json.dumps({item.name: list(item.after) for item in order(shared, independent)}, indent=1))
        return 0
    names = [item.name for item in order(shared, independent)]
    only = with_producers(args.only) if args.with_producers else args.only
    if args.only:
        unknown = set(args.only) - set(names)
        if unknown:
            parser.error("unknown steps: " + ", ".join(sorted(unknown)))
        missing = missing_producers(only)
        if missing:
            parser.error("; ".join(f"{name} reads the outputs of {', '.join(producers)}"
                                   for name, producers in missing.items())
                         + ": add them with --only, or pass --with-producers")
        names = [item.name for item in select(order(shared, independent), only)]
    problems = preflight.problems(names)
    for problem in problems:
        print("cpu-ci preflight: " + problem, file=sys.stderr)
    if problems:
        print("cpu-ci preflight failed: nothing ran (see docs/contributing.md#setup)", file=sys.stderr)
        return 2
    if args.preflight:
        print("cpu-ci preflight: ok", file=sys.stderr)
        return 0
    work = args.work or pathlib.Path(tempfile.mkdtemp(prefix="chopinfer-cpu-ci.", dir=os.environ.get("TMPDIR")))
    work.mkdir(parents=True, exist_ok=True)
    (work / "external-sdk").mkdir(exist_ok=True)
    shared, independent = steps(target, work, args.jobs, args.test_threads, args.exhaustive)
    planned = order(shared, independent)
    if args.only:
        planned = select(planned, only)
    started = time.monotonic()
    began = time.time()
    status = 1
    head = preflight.head()
    watch = HeadWatch(ROOT, head) if head else None
    # The runner restores this handler when it returns, so a SIGTERM the
    # watch sent just before it stopped still ends here, not the process.
    terminate = signal.signal(signal.SIGTERM, interrupted)
    try:
        if watch:
            watch.start()
        # A stopped geometry lane or export recipe removes its worktree and
        # temporary target.
        results = runner.run(planned, 1 if args.serial else max(args.jobs, 1), work / "logs",
                             keep_going=args.keep_going, out=sys.stderr, label="cpu-ci", grace=120)
        if watch:
            watch.stop()
        failed = runner.failed(results)
        status = 1 if failed else 0
        if args.summary:
            runner.write_summary(args.summary, results, jobs=args.jobs, test_threads=args.test_threads,
                                 seconds=time.monotonic() - started)
        if failed:
            print("cpu-ci failed: " + ", ".join(failed), file=sys.stderr)
        now = preflight.head() if watch else None
        if watch and now != head:
            head_moved(head, now, sys.stderr)
            status = 1
    except runner.Interrupted as interruption:
        status = 128 + interruption.signum
        if watch and watch.moved:
            head_moved(head, watch.moved, sys.stderr)
            status = 1
    finally:
        if watch:
            watch.stop()
        signal.signal(signal.SIGTERM, terminate)
        elapsed = time.monotonic() - started
        report_times(elapsed, slowest_tests(NEXTEST_REPORTS, began), not args.only and not args.exhaustive, sys.stderr)
        print(f"cpu-ci total: {elapsed:.3f}s (exit {status})", file=sys.stderr)
        # Failed steps' logs are already printed; keep the directory only on request.
        if args.work:
            for path in (work / "external-sdk", work / "cuda-host-abi", work / "cuda-toolchain-target",
                         work / "profile-target"):
                if path.is_dir():
                    remove_tree(path)
                elif path.exists():
                    path.unlink()
            print(f"cpu-ci logs: {work / 'logs'}", file=sys.stderr)
        else:
            remove_tree(work)
    return status


if __name__ == "__main__":
    sys.exit(main())
