"""The CI step runner and the qualification runner keep every failure,
bound their workers, order artifact readers before incompatible writers and
leave no process behind."""
import io
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import unittest
import unittest.mock
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci"))
import cpu
import qualification
import runner
import rust_tests
import route_sweeps


def shell(script):
    return ["bash", "-c", script]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (Z) has exited and only its parent's wait remains; a reaped
    # process can show as dead (X) for a moment before it disappears.
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(")")[-1].split()[0] not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError):
        return False


def gone(pid, timeout=30.0):
    """Whether `pid` exits within `timeout`; a killed process can take a
    moment to leave uninterruptible sleep on a busy host. Once seen exited it
    stays gone, so the first such sighting answers."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


class Runner(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)

    def run_steps(self, steps, jobs=4, keep_going=False):
        return {result.name: result for result in
                runner.run(steps, jobs, self.directory / "logs", keep_going=keep_going, out=io.StringIO())}

    def test_a_failed_step_fails_the_run_and_skips_what_follows(self):
        results = self.run_steps([runner.Step("fails", shell("echo broken; exit 3")),
                                  runner.Step("later", shell("true"), after=("fails",))], jobs=1)
        self.assertEqual((results["fails"].status, results["fails"].exit), ("failed", 3))
        self.assertEqual(results["later"].status, "skipped")
        self.assertEqual(runner.failed(results.values()), ["fails", "later"])
        self.assertIn("broken", pathlib.Path(results["fails"].log).read_text())

    def test_keep_going_runs_independent_steps_but_not_dependents(self):
        results = self.run_steps([runner.Step("fails", shell("exit 1")),
                                  runner.Step("independent", shell("true")),
                                  runner.Step("dependent", shell("true"), after=("fails",))], jobs=1, keep_going=True)
        self.assertEqual([results[name].status for name in ("fails", "independent", "dependent")],
                         ["failed", "ok", "skipped"])

    def test_a_signal_death_is_a_failure(self):
        results = self.run_steps([runner.Step("killed", shell("kill -9 $$"))])
        self.assertEqual(results["killed"].status, "failed")
        self.assertEqual(results["killed"].exit, -signal.SIGKILL)

    def test_workers_stay_within_the_budget(self):
        events = self.directory / "events"
        step = 'echo "start $(date +%s%N)" >> {0}; sleep 0.3; echo "end $(date +%s%N)" >> {0}'
        self.run_steps([runner.Step(f"s{index}", shell(step.format(events))) for index in range(5)], jobs=2)
        running = peak = 0
        for _, kind in sorted((int(line.split()[1]), line.split()[0]) for line in events.read_text().splitlines()):
            running += 1 if kind == "start" else -1
            peak = max(peak, running)
        self.assertEqual(peak, 2)

    def test_an_incompatible_build_waits_for_every_reader_of_the_previous_one(self):
        events = self.directory / "events"
        record = 'echo "{1} {2} $(date +%s%N)" >> {0}'
        def timed(name, seconds):
            return shell(f"{record.format(events, name, 'start')}; sleep {seconds}; {record.format(events, name, 'end')}")
        self.run_steps([runner.Step("build", timed("build", 0.1)),
                        runner.Step("reader-a", timed("reader-a", 0.3), after=("build",)),
                        runner.Step("reader-b", timed("reader-b", 0.1), after=("build",)),
                        runner.Step("variant-build", timed("variant-build", 0.1), after=("reader-a", "reader-b"))])
        times = {(line.split()[0], line.split()[1]): int(line.split()[2]) for line in events.read_text().splitlines()}
        for reader in ("reader-a", "reader-b"):
            self.assertGreaterEqual(times[reader, "start"], times["build", "end"])
            self.assertGreaterEqual(times["variant-build", "start"], times[reader, "end"])

    def test_descendants_left_behind_are_stopped_when_their_step_ends(self):
        pid_file = self.directory / "pid"
        results = self.run_steps([runner.Step("leaves", shell(f"sleep 60 & echo $! > {pid_file}"))])
        self.assertEqual(results["leaves"].status, "ok")
        self.assertTrue(gone(int(pid_file.read_text())))

    def test_an_interrupted_run_stops_every_step_and_its_descendants(self):
        pid_file = self.directory / "pid"
        program = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT / 'ci')!r})
            import runner
            step = runner.Step("stubborn", ["bash", "-c", "trap '' TERM; sleep 60 & echo $! > {pid_file}; wait"])
            try:
                runner.run([step, runner.Step("quick", ["true"])], 2, {str(self.directory / 'logs')!r})
            except runner.Interrupted as interruption:
                sys.exit(128 + interruption.signum)
            """)
        child = subprocess.Popen([sys.executable, "-c", program], stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 10
        while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.05)
        grandchild = int(pid_file.read_text())
        child.send_signal(signal.SIGTERM)
        _, stderr = child.communicate(timeout=30)
        self.assertEqual(child.returncode, 128 + signal.SIGTERM, stderr)
        self.assertIn("interrupted by signal", stderr)
        self.assertTrue(gone(grandchild))

    def test_a_second_signal_during_the_stop_still_kills_stubborn_steps(self):
        # A terminal's Ctrl-C can reach a run twice, for example through a
        # wrapper that forwards it; the second must not skip the SIGKILL.
        pid_file = self.directory / "pid"
        program = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT / 'ci')!r})
            import runner
            step = runner.Step("stubborn", ["bash", "-c", "trap '' TERM; sleep 60 & echo $! > {pid_file}; wait"])
            try:
                runner.run([step], 1, {str(self.directory / 'logs')!r}, grace=3.0)
            except runner.Interrupted as interruption:
                sys.exit(128 + interruption.signum)
            """)
        child = subprocess.Popen([sys.executable, "-c", program], stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 10
        while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.05)
        grandchild = int(pid_file.read_text())
        child.send_signal(signal.SIGINT)
        time.sleep(0.5)
        child.send_signal(signal.SIGINT)
        _, stderr = child.communicate(timeout=30)
        self.assertEqual(child.returncode, 128 + signal.SIGINT, stderr)
        self.assertIn("stopped 1 running step(s)", stderr)
        self.assertTrue(gone(grandchild))

    def test_the_first_interruption_holds_later_signals_until_the_stop_does(self):
        # Between the first signal's raise and the stop's own hold, a second
        # signal must not raise out of the cleanup.
        try:
            with self.assertRaises(runner.Interrupted):
                runner._interrupt(signal.SIGTERM, None)
            runner._interrupt(signal.SIGINT, None)
            self.assertEqual(runner._held, [signal.SIGINT])
        finally:
            runner._held = None

    def test_stop_keeps_exited_leaders_unreaped_until_their_groups_are_killed(self):
        # A reaped leader's ID can be reused, and the final SIGKILL to its
        # group could then reach an unrelated group.
        leader = subprocess.Popen(["sleep", "60"], start_new_session=True)
        states = []
        signal_group = runner._signal_group

        def record(pid, signum):
            if signum == signal.SIGKILL:
                try:
                    with open(f"/proc/{pid}/stat") as stat:
                        states.append(stat.read().rsplit(")", 1)[1].split()[0])
                except FileNotFoundError:
                    states.append("reaped")
            signal_group(pid, signum)

        with patch.object(runner, "_signal_group", record):
            runner.stop({leader.pid}, grace=5.0)
        leader.wait(timeout=5)
        self.assertEqual(states, ["Z"])
        self.assertTrue(gone(leader.pid))

    def test_a_signal_as_a_step_finishes_leaves_it_out_of_the_stop(self):
        # A finished step's leader must stay unreaped until it leaves
        # `running`, or a signal in between hands stop() a reaped, reusable ID.
        signal_group, stop = runner._signal_group, runner.stop
        finished, stopped = [], []

        def interrupt_on_cleanup(pid, signum):
            if signum == signal.SIGKILL and not finished and not stopped:
                finished.append(pid)
                os.kill(os.getpid(), signal.SIGTERM)
            signal_group(pid, signum)

        def record(groups, grace=10.0):
            stopped.append(set(groups))
            stop(groups, grace)

        steps = [runner.Step("quick", ["true"]), runner.Step("long", ["sleep", "60"])]
        with patch.object(runner, "_signal_group", interrupt_on_cleanup), patch.object(runner, "stop", record):
            with self.assertRaises(runner.Interrupted):
                runner.run(steps, 2, self.directory / "logs", out=io.StringIO(), grace=0.5)
        self.assertEqual(len(finished), 1)
        self.assertEqual(len(stopped), 1)
        self.assertNotIn(finished[0], stopped[0])
        self.assertEqual(len(stopped[0]), 1, "the long step is stopped")

    def test_children_of_other_threads_are_theirs_to_reap(self):
        # ci/cpu.py polls HEAD with git from a thread while the steps run.
        import threading
        done, codes = threading.Event(), []

        def poll():
            while not done.is_set():
                codes.append(subprocess.run(["bash", "-c", "exit 7"]).returncode)

        thread = threading.Thread(target=poll)
        thread.start()
        try:
            results = self.run_steps([runner.Step(f"s{index}", shell("sleep 0.2")) for index in range(4)], jobs=2)
        finally:
            done.set()
            thread.join()
        self.assertEqual([result.status for result in results.values()], ["ok"] * 4)
        self.assertGreater(len(codes), 0)
        self.assertEqual(set(codes), {7}, "the runner reaped another thread's child")

    def test_an_interruption_while_a_step_starts_still_stops_that_step(self):
        pid_file = self.directory / "pid"
        start = subprocess.Popen

        def interrupted_start(*args, **kwargs):
            # The signal lands after the fork, before the runner records the child.
            child = start(*args, **kwargs)
            deadline = time.monotonic() + 10
            while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
                time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        step = runner.Step("stubborn", shell(f"trap '' TERM; sleep 60 & echo $! > {pid_file}; wait"))
        with patch.object(runner.subprocess, "Popen", interrupted_start):
            with self.assertRaises(runner.Interrupted):
                runner.run([step], 1, self.directory / "logs", out=io.StringIO(), grace=0.5)
        self.assertTrue(gone(int(pid_file.read_text()), timeout=10.0))

    def test_unknown_dependencies_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown dependencies: missing"):
            self.run_steps([runner.Step("a", ["true"], after=("missing",))])


class RequiredCoverage(unittest.TestCase):
    """A required unittest step passes on executed tests and allowed skips only, never on exit 0 alone."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)

    def run_module(self, body, skips=cpu.REFERENCE_SKIPS, verbose=True):
        (self.directory / "test_case.py").write_text("import unittest\n\nclass Cases(unittest.TestCase):\n" + body)
        step = runner.Step("reference", [sys.executable, "-B", "-m", "unittest", "test_case"] + (["-v"] if verbose else []),
                           cwd=str(self.directory), skips=skips)
        result = runner.run([step], 1, self.directory / "logs", out=io.StringIO())[0]
        return result, pathlib.Path(result.log).read_text()

    def test_executed_tests_and_opt_in_skips_pass(self):
        result, log = self.run_module(
            "    def test_a(self):\n        pass\n\n"
            "    @unittest.skip('needs CHOPINFER_GLM5NEXT_CHECKPOINT (tokenizer)')\n"
            "    def test_b(self):\n        pass\n\n"
            "    @unittest.skip('the CUDA round trip runs with CHOPINFER_LATENT_KV_CUDA=1')\n"
            "    def test_c(self):\n        pass\n\n"
            "    @unittest.skip('needs an NVFP4 checkpoint in CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS')\n"
            "    def test_d(self):\n        pass\n")
        # unittest's "Ran 4 tests" includes the three skipped cases.
        self.assertEqual((result.status, result.tests, result.skipped), ("ok", 1, 3))
        self.assertIn("reference: 1 tests executed, 3 skipped", log)

    def test_a_suite_of_allowed_skips_executes_nothing_and_fails(self):
        result, log = self.run_module("    @unittest.skip('needs CHOPINFER_REVIEW_CHECKPOINT')\n"
                                      "    def test_a(self):\n        pass\n")
        self.assertEqual((result.status, result.exit, result.tests, result.skipped), ("failed", 0, 0, 1))
        self.assertIn("reference: no test executed", log)

    def test_missing_dependency_skips_fail_with_their_cause(self):
        for reason in ("needs torch (set CHOPINFER_REFRUN_PYTHON)", "needs numpy", "needs torch",
                       "needs torch and CHOPINFER_REVIEW_CHECKPOINT",
                       "needs the refrun environment and CHOPINFER_GLM5NEXT_CHECKPOINT",
                       "needs transformers (CHOPINFER_X_CUDA=1)",
                       "the digests are of glibc 2.39's F32 functions; run the reference checks on a "
                       "glibc 2.39 host (Ubuntu 24.04)"):
            with self.subTest(reason=reason):
                result, log = self.run_module(f"    def test_a(self):\n        pass\n\n"
                                              f"    @unittest.skip({reason!r})\n    def test_b(self):\n        pass\n")
                self.assertEqual((result.status, result.exit), ("failed", 0))
                self.assertIn(f"reference: unexpected skip (1): {reason}", log)
                self.assertEqual(runner.failed([result]), ["reference"])

    def test_zero_tests_and_unreadable_skips_fail(self):
        result, log = self.run_module("    pass\n")
        self.assertEqual(result.status, "failed")
        self.assertIn("reference: no test executed", log)
        result, log = self.run_module("    @unittest.skip('the CUDA round trip runs with CHOPINFER_X_CUDA=1')\n"
                                      "    def test_a(self):\n        pass\n", verbose=False)
        self.assertEqual(result.status, "failed")
        self.assertIn("1 skips reported, 0 with a reason", log)

    def test_steps_without_declared_skips_keep_the_exit_status(self):
        result, _ = self.run_module("    @unittest.skip('needs torch')\n    def test_a(self):\n        pass\n", skips=None)
        self.assertEqual((result.status, result.tests), ("ok", None))

    def test_every_reference_step_is_required_after_its_prerequisites(self):
        with patch.dict(os.environ, {"CHOPINFER_REFRUN_PYTHON": "/fixture/python"}):
            _, independent = cpu.steps(self.directory / "target", self.directory, 8, 8)
        steps = {step.name: step for step in independent}
        references = [name for name in steps if name.startswith("refrun") and name != "refrun-python-compile"]
        self.assertGreater(len(references), 3)
        for name in references + ["tapdiff"]:
            with self.subTest(step=name):
                self.assertEqual(steps[name].skips, cpu.REFERENCE_SKIPS)
                self.assertIn("reference-prerequisites", steps[name].after)
                self.assertEqual(steps[name].argv[0], "/fixture/python")
        self.assertEqual(steps["reference-prerequisites"].argv[-2:], ["ci/model_checks.py", "prerequisites"])

    def test_reference_checks_score_with_the_profile_copy_after_it_is_made(self):
        """No reference check reads target/debug/chopinfer, which the profile build and later
        runtime-host variants rewrite while the gate runs."""
        copy = str(self.directory / "profile-target/debug/chopinfer")
        shared, independent = cpu.steps(self.directory / "target", self.directory, 8, 8)
        steps = {step.name: step for step in shared + independent}
        self.assertIn("chopinfer", steps["profile-copy"].argv)
        references = [name for name, step in steps.items() if step.skips == cpu.REFERENCE_SKIPS]
        self.assertIn("tapdiff", references)
        for name in references:
            with self.subTest(step=name):
                self.assertEqual(steps[name].env["CHOPINFER_EVAL"], copy)
                # Directly, so a selection of one step still waits for the copy.
                self.assertIn("profile-copy", steps[name].after)


class ScorerDependencies(unittest.TestCase):
    READERS = ("validation-report", "spec-bench", "deploy-artifacts", "profile-tools", "release-gate",
               "model-inventory")

    def planned(self, work):
        # An inherited evaluator can implement an older CLI than this source.
        with patch.dict(os.environ, CHOPINFER_EVAL="/stale/chopinfer"):
            return cpu.order(*cpu.steps(work / "target", work, 4, 4))

    def test_readers_receive_the_profile_copy_before_python_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            steps = {step.name: step for step in self.planned(work)}
            names = [*self.READERS, *(name for name, step in steps.items() if step.skips == cpu.REFERENCE_SKIPS)]
            for name in names:
                with self.subTest(step=name):
                    # Step.env reaches the interpreter before module-level code runs.
                    self.assertEqual((steps[name].env or {}).get("CHOPINFER_EVAL"),
                                     str(work / "profile-target/debug/chopinfer"))
                    self.assertIn("profile-copy", steps[name].after)

    def test_only_readers_keep_their_build_and_copy_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (*self.READERS, "tapdiff"):
                with self.subTest(step=name):
                    selected = cpu.select(self.planned(pathlib.Path(directory)), [name])
                    steps = {step.name: step for step in selected}
                    expected = {"profile-cli-build", "preparation-fixture-build", "profile-copy", name}
                    if name == "tapdiff":
                        expected.add("reference-prerequisites")
                    self.assertEqual(set(steps), expected)
                    self.assertEqual(steps["profile-copy"].after, ("preparation-fixture-build",))
                    self.assertEqual(steps["preparation-fixture-build"].after, ("profile-cli-build",))
                    self.assertIn("profile-copy", steps[name].after)

    def test_the_profile_copy_precedes_later_shared_target_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            steps = {step.name: step for step in self.planned(pathlib.Path(directory))}
            self.assertIn("profile-copy", steps["portable"].after)

    def test_qualification_runs_modules_at_once_even_on_a_small_budget(self):
        """ci/quick.sh in an 8-CPU lane gets jobs=8; one module at a time made the qualification
        run its longest check. The full gate keeps a sixth of its budget."""
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            for jobs, modules in ((1, 1), (2, 1), (8, 4), (16, 4), (32, 5), (48, 8)):
                with self.subTest(jobs=jobs):
                    steps = {step.name: step for step in cpu.order(*cpu.steps(work / "target", work, jobs, 8))}
                    argv = steps["release-gate"].argv
                    self.assertEqual(argv[argv.index("--jobs") + 1], str(modules))
                    self.assertEqual(steps["release-gate-clean"].env["CHOPINFER_CI_JOBS"], str(modules))

    def test_no_step_counts_for_more_than_half_the_budget(self):
        """With as many test threads as the budget, the workspace tests start beside the long
        independent steps instead of waiting until no other step runs."""
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            for jobs, threads, tests in ((32, 32, 16), (48, 16, 16), (8, 8, 4), (1, 16, 1)):
                with self.subTest(jobs=jobs, threads=threads):
                    steps = {step.name: step for step in cpu.order(*cpu.steps(work / "target", work, jobs,
                                                                              threads, exhaustive=True))}
                    self.assertEqual(steps["portable"].weight, tests)
                    self.assertEqual(steps["exhaustive-comparisons"].weight, tests)
                    self.assertLessEqual(max(step.weight for step in steps.values()), max(jobs // 2, 1))
                    self.assertEqual(steps["portable"].env["CHOPINFER_CI_TEST_THREADS"], str(threads))
                    self.assertIn(f"--test-threads {threads}", " ".join(steps["exhaustive-comparisons"].argv))


class ReferencePrerequisites(unittest.TestCase):
    def python(self, script):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = pathlib.Path(temporary.name) / "python"
        path.write_text("#!/bin/sh\n" + script)
        path.chmod(0o755)
        return str(path)

    def test_an_unset_or_incomplete_reference_python_is_refused_with_its_remedy(self):
        import model_checks
        with self.assertRaisesRegex(model_checks.PrerequisiteError, "need CHOPINFER_REFRUN_PYTHON: set CHOPINFER_REFRUN_PYTHON in local.env"):
            model_checks.reference_python({})
        with self.assertRaisesRegex(model_checks.PrerequisiteError, "does not run"):
            model_checks.reference_python({"CHOPINFER_REFRUN_PYTHON": "/nonexistent/python"})
        for missing in ("torch", "transformers", "numpy", "torch numpy"):
            with self.subTest(missing=missing):
                lacking = self.python(f"echo {missing}\n")
                with self.assertRaisesRegex(model_checks.PrerequisiteError,
                                            "lacks " + missing.replace(" ", ", ") + "; set "):
                    model_checks.reference_python({"CHOPINFER_REFRUN_PYTHON": lacking})
        broken = self.python("echo broken >&2; exit 3\n")
        with self.assertRaisesRegex(model_checks.PrerequisiteError, "failed: broken"):
            model_checks.reference_python({"CHOPINFER_REFRUN_PYTHON": broken})
        complete = self.python("exit 0\n")
        self.assertEqual(model_checks.reference_python({"CHOPINFER_REFRUN_PYTHON": complete}), complete)

    def test_a_selected_reference_step_brings_its_prerequisite_check(self):
        with tempfile.TemporaryDirectory() as directory:
            shared, independent = cpu.steps(pathlib.Path(directory) / "target", pathlib.Path(directory), 8, 8)
        planned = cpu.select(cpu.order(shared, independent), ["tapdiff"])
        self.assertEqual([item.name for item in planned], ["profile-cli-build", "preparation-fixture-build",
                                                         "profile-copy", "reference-prerequisites", "tapdiff"])
        self.assertEqual(planned[-1].after, ("reference-prerequisites", "profile-copy"))
        planned = cpu.select(cpu.order(shared, independent), ["leak-guard"])
        self.assertEqual([item.name for item in planned], ["leak-guard"])

    def test_a_step_selected_without_its_producers_is_refused_naming_them(self):
        stderr = io.StringIO()
        with patch("sys.stderr", stderr), self.assertRaises(SystemExit) as raised:
            cpu.main(["--only", "kernel-report-integration", "--only", "deploy-entrypoint",
                      "--only", "process-lifecycle"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("kernel-report-integration reads the outputs of portable, external-model-smoke: add them "
                      "with --only, or pass --with-producers", stderr.getvalue())
        self.assertNotIn("deploy-entrypoint reads", stderr.getvalue())

    def test_with_producers_a_reader_runs_after_them(self):
        with tempfile.TemporaryDirectory() as directory:
            shared, independent = cpu.steps(pathlib.Path(directory) / "target", pathlib.Path(directory), 8, 8)
        names = cpu.with_producers(["kernel-report-integration"])
        self.assertEqual(cpu.missing_producers(names), {})
        steps = {item.name: item for item in cpu.select(cpu.order(shared, independent), names)}
        self.assertEqual(set(steps), {"kernel-report-integration", "portable", "external-model-smoke",
                                      *cpu.SETUP} - set(cpu.PREREQUISITES))
        # The reader followed portable through the unselected fixture builds.
        self.assertEqual(set(steps["kernel-report-integration"].after), {"portable", "external-model-smoke"})

    def test_every_producer_precedes_its_reader_in_the_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            shared, independent = cpu.steps(pathlib.Path(directory) / "target", pathlib.Path(directory), 8, 8)
        steps = {item.name: item for item in cpu.order(shared, independent)}

        def before(name):
            return {earlier for step in steps[name].after for earlier in (step, *before(step))}

        for reader, producers in cpu.PRODUCERS.items():
            with self.subTest(reader=reader):
                self.assertLessEqual(set(producers), before(reader))

    def test_the_gate_step_reports_the_missing_prerequisite(self):
        env = {key: value for key, value in os.environ.items() if key != "CHOPINFER_REFRUN_PYTHON"}
        result = subprocess.run([sys.executable, "-B", "ci/model_checks.py", "prerequisites"], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("the reference checks need CHOPINFER_REFRUN_PYTHON", result.stderr)


PINNED = "from qual_test_config import setUpModule  # noqa: F401"
MODULE = """import unittest
{header}

class Cases(unittest.TestCase):
{body}
"""


class Qualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        (self.root / "ci").mkdir()
        for model in ("alpha", "beta"):
            (self.root / f"models/{model}").mkdir(parents=True)
            (self.root / f"models/{model}/qual.json").write_text("{}")
        mechanics = self.root / "tools/qual/gates/mechanics.json"
        mechanics.parent.mkdir(parents=True)
        mechanics.write_text("{}")
        (self.root / "tools/qual/qual_test_config.py").write_text(textwrap.dedent("""
            import os

            def configure(name="alpha"):
                os.environ["CHOPINFER_QUAL_CONFIG"] = f"models/{name}/qual.json"

            def setUpModule():
                configure()
            """))
        self.module("test_pinned", PINNED + '\nQUALIFICATION_RUNS_ONCE = "audited"',
                    "    def test_pinned(self):\n        self.assertIn('alpha', __import__('os').environ['CHOPINFER_QUAL_CONFIG'])")
        self.module("test_sensitive", "import os",
                    "    def test_outer(self):\n        print('config', os.environ['CHOPINFER_QUAL_CONFIG'])")
        # Keep the scheduling cache out of the user's cache directory.
        cache = patch.dict(os.environ, XDG_CACHE_HOME=str(self.root / "cache"))
        cache.start()
        self.addCleanup(cache.stop)

    def module(self, name, header, body):
        (self.root / f"tools/qual/{name}.py").write_text(MODULE.format(header=header, body=body))

    def run_main(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        log = self.root / "all.log"
        with unittest.mock.patch.object(sys, "stdout", stdout), unittest.mock.patch.object(sys, "stderr", stderr):
            try:
                status = qualification.main(["--root", str(self.root), "--jobs", "2", "--log", str(log), *arguments])
            except SystemExit as exit:
                status = exit.code
        return status, stdout.getvalue(), stderr.getvalue(), log.read_text() if log.exists() else ""

    def test_reviewed_modules_run_once_and_the_rest_under_every_config(self):
        status, stdout, stderr, log = self.run_main()
        self.assertEqual(status, 0, stdout + stderr)
        self.assertIn("qualification alpha: 2 cells, 2 tests", stdout)
        self.assertIn("qualification beta: 1 cells, 1 tests", stdout)
        self.assertIn("qualification mechanics: 1 cells, 1 tests", stdout)
        for config in ("models/alpha/qual.json", "models/beta/qual.json", "tools/qual/gates/mechanics.json"):
            self.assertIn(f"config {config}", log)

    def test_the_full_matrix_repeats_reviewed_modules(self):
        status, stdout, stderr, _ = self.run_main("--matrix", "full")
        self.assertEqual(status, 0, stdout + stderr)
        self.assertIn("qualification beta: 2 cells, 2 tests", stdout)

    def test_a_pinned_module_without_the_declaration_keeps_the_full_matrix(self):
        self.module("test_new", PINNED, "    def test_new(self):\n        pass")
        planned = qualification.cells(qualification.configs(self.root), qualification.modules(self.root),
                                      qualification.load_inventory(self.root))
        self.assertEqual([label for label, _, module in planned if module == "test_new"],
                         ["alpha", "beta", "mechanics"])
        self.assertEqual([label for label, _, module in planned if module == "test_pinned"], ["alpha"])

    def test_a_declared_module_that_loses_its_pin_is_refused(self):
        self.module("test_pinned", 'QUALIFICATION_RUNS_ONCE = "audited"', "    def test_pinned(self):\n        pass")
        status, _, stderr, _ = self.run_main()
        self.assertIn("tools/qual/test_pinned.py: declares QUALIFICATION_RUNS_ONCE but no longer pins", str(status))

    def test_a_declaration_needs_its_reason(self):
        self.module("test_pinned", PINNED + '\nQUALIFICATION_RUNS_ONCE = ""', "    def test_pinned(self):\n        pass")
        with self.assertRaisesRegex(ValueError, "needs the audited reason"):
            qualification.load_inventory(self.root)

    def test_recorded_durations_order_cells_longest_first(self):
        planned = qualification.cells([("a", "a.json"), ("b", "b.json")], ["test_x", "test_y", "test_z"],
                                      {"test_z"}, {"test_y": 9.0, "test_z": 3.0})
        self.assertEqual(planned, [("a", "a.json", "test_y"), ("b", "b.json", "test_y"),
                                   ("a", "a.json", "test_z"), ("a", "a.json", "test_x"), ("b", "b.json", "test_x")])
        status, stdout, stderr, _ = self.run_main()
        self.assertEqual(status, 0, stdout + stderr)
        self.assertEqual(set(qualification.load_durations()), {"test_pinned", "test_sensitive"})

    def test_every_failure_is_reported(self):
        self.module("test_broken", "", "    def test_a(self):\n        self.fail('first')\n\n"
                                       "    def test_b(self):\n        raise RuntimeError('second')")
        status, stdout, stderr, _ = self.run_main()
        self.assertEqual(status, 1)
        for config in ("alpha", "beta", "mechanics"):
            self.assertIn(f"qualification failed: {config}/test_broken", stderr)
        self.assertIn("AssertionError: first", stderr)
        self.assertIn("RuntimeError: second", stderr)

    def test_a_module_that_runs_no_tests_fails(self):
        (self.root / "tools/qual/test_empty.py").write_text("import unittest\n")
        status, _, stderr, _ = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("alpha/test_empty", stderr)

    def test_a_module_that_cannot_import_fails(self):
        (self.root / "tools/qual/test_import.py").write_text("raise ImportError('broken import')\n")
        status, _, stderr, _ = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("broken import", stderr)

    def test_without_a_config_every_module_runs_once_without_the_variable(self):
        status, stdout, stderr, log = self.run_main("--no-config")
        self.assertEqual(status, 1, stdout + stderr)
        # test_sensitive reads the variable, so it fails without one.
        self.assertIn("none/test_sensitive", stderr)
        self.assertIn("qualification none: 2 cells", stdout)

    def test_a_missing_config_is_refused(self):
        (self.root / "tools/qual/gates/mechanics.json").unlink()
        status, _, _, _ = self.run_main()
        self.assertIn("Missing qualification configuration", str(status))

    def test_cells_use_their_own_temporary_and_receipt_directories(self):
        self.module("test_sensitive", "import os",
                    "    def test_outer(self):\n        print('dirs', os.environ['TMPDIR'], "
                    "os.environ['CHOPINFER_QUAL_RECEIPTS'])")
        status, stdout, stderr, log = self.run_main()
        self.assertEqual(status, 0, stdout + stderr)
        directories = [line.split()[1:] for line in log.splitlines() if line.startswith("dirs ")]
        self.assertEqual(len(directories), 3)
        self.assertEqual(len({path for pair in directories for path in pair}), 6)


class RouteSweeps(unittest.TestCase):
    def test_new_model_is_discovered_from_its_required_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "route-allowlist.toml").write_text("")
            packages = [{"name": "future-bundle", "manifest_path": str(root / "Cargo.toml")},
                        {"name": "shared", "manifest_path": str(root / "shared/Cargo.toml")}]
            self.assertEqual(route_sweeps.libraries(["--workspace"], packages), ["future-bundle"])
            self.assertEqual(route_sweeps.libraries(["-p", "shared"], packages), [])

    def test_budget_warns_loudly_and_missing_execution_fails(self):
        output = "12 cases, 0.1s\npeak_rss_bytes=1024\ntest result: ok. 1 passed; 0 failed; 0 ignored;"
        with patch.object(sys, "stderr", new_callable=io.StringIO) as error:
            self.assertEqual(route_sweeps.check("future", output, 0, 20.01, "quick"), 1)
            self.assertIn("WARNING: ROUTE SWEEP BUDGET EXCEEDED", error.getvalue())
        for changed, code in [(output, 101), (output.replace("1 passed", "0 passed"), 0),
                              (output.replace("cases,", "missing"), 0),
                              (output.replace("peak_rss_bytes=", "missing="), 0)]:
            with self.assertRaises(RuntimeError):
                route_sweeps.check("future", changed, code, 1, "quick")

    def test_quick_models_start_in_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binaries = {}
            for name in ("first", "second"):
                binary = root / name
                binary.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                    import pathlib, time
                    root = pathlib.Path(__file__).parent
                    pathlib.Path(__file__ + ".ready").touch()
                    end = time.monotonic() + 5
                    while len(list(root.glob("*.ready"))) < 2:
                        if time.monotonic() > end:
                            raise SystemExit(99)
                        time.sleep(0.01)
                    print("12 cases, 0.1s")
                    print("peak_rss_bytes=1024")
                    print("test result: ok. 1 passed; 0 failed; 0 ignored;")
                    '''))
                binary.chmod(0o755)
                binaries[name] = (str(binary), str(root), dict(os.environ))
            with patch.object(sys, "stdout", new_callable=io.StringIO) as output:
                self.assertEqual(route_sweeps.run(binaries, "quick"), 0)
                self.assertIn("parallel_models=2", output.getvalue())
                self.assertIn("added_wall_seconds=", output.getvalue())

    def test_cpu_runs_advance_per_model_and_pinned_reproductions_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"CHOPINFER_DATA": directory}
            self.assertEqual(route_sweeps.rotation_seed("first", "cpu", env), 0)
            self.assertEqual(route_sweeps.rotation_seed("first", "cpu", env), 1)
            self.assertEqual(route_sweeps.rotation_seed("second", "cpu", env), 0)
            self.assertEqual(route_sweeps.rotation_seed("first", "cpu", {**env, "CHOPINFER_ROUTE_SEED": "41"}), 41)
            self.assertEqual(route_sweeps.rotation_seed("first", "cpu", env), 2)
            with route_sweeps.concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(lambda _: route_sweeps.rotation_seed("parallel", "cpu", env), range(8)))
            self.assertEqual(sorted(values), list(range(8)))
            root = pathlib.Path(directory) / "route-guard" / "rotation"
            key = route_sweeps.hashlib.sha256(b"linked").hexdigest()
            (root / (key + ".lock")).symlink_to(root / (key + ".json"))
            with self.assertRaises(OSError):
                route_sweeps.rotation_seed("linked", "cpu", env)
        with self.assertRaises(ValueError):
            route_sweeps.rotation_seed("first", "cpu", {})

    def test_separate_sweeps_are_excluded_from_both_normal_runners(self):
        cargo = rust_tests.commands(["--workspace"], "cargo", 8, skip_routes=True)[0]
        self.assertEqual(cargo[-3:], ["--", "--skip", "route_sweep::"])
        nextest = rust_tests.commands(["--workspace"], "nextest", 8, skip_routes=True)[0]
        self.assertEqual(nextest[-2:], ["-E", "not (test(/route_sweep::/))"])


class RustTests(unittest.TestCase):
    def test_serial_libraries_finish_before_nextest_and_keep_failures(self):
        for library_exit in (0, 7):
            with self.subTest(library_exit=library_exit), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                packages, binaries = [], {}
                for name in ("first", "second"):
                    binary = root / name
                    binary.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                        import fcntl, pathlib, sys, time
                        root = pathlib.Path(__file__).parent
                        with (root / "library.lock").open("w") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            time.sleep(0.05)
                            (root / (pathlib.Path(__file__).name + ".finished")).touch()
                        sys.exit(int(sys.argv[0].endswith("first")) * EXIT)
                        ''').replace("EXIT", str(library_exit)))
                    binary.chmod(0o755)
                    packages.append({"name": name, "manifest_path": str(root / "Cargo.toml"),
                                     "metadata": {"chopinfer": {"lib-tests-in-one-process": True}}})
                    binaries[name] = (str(binary), str(root), dict(os.environ))
                calls = []

                def invoke(command):
                    calls.append(command)
                    self.assertTrue(all((root / (name + ".finished")).is_file() for name in binaries),
                                    "nextest started before every library finished")
                    return subprocess.CompletedProcess(command, 0)

                output = io.BytesIO()
                with io.TextIOWrapper(output, encoding="utf-8") as stdout, \
                        patch.dict(os.environ, CHOPINFER_CI_SERIAL_LIBRARIES="1"), \
                        patch.object(rust_tests, "workspace_packages", return_value=packages), \
                        patch.object(rust_tests, "library_binaries", return_value=binaries), \
                        patch.object(rust_tests.subprocess, "run", side_effect=invoke), \
                        patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", new_callable=io.StringIO):
                    self.assertEqual(rust_tests.run_nextest(["--workspace"], 2), int(library_exit != 0))
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][:3], ["cargo", "nextest", "run"])
                self.assertIn("--binaries-metadata", calls[0])
                self.assertEqual(calls[1], ["cargo", "test", "--workspace", "--doc", *rust_tests.LOCKED])

    def test_a_parallel_library_can_spawn_itself_after_nextest_starts(self):
        # Cargo may replace an artifact during a second build. A live library
        # test must still be able to start its executable in a child process.
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "library-test"
            ready, release = root / "ready", root / "release"
            binary.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                import pathlib, subprocess, sys, time
                if len(sys.argv) > 1:
                    sys.exit(0)
                root = pathlib.Path(__file__).parent
                (root / "ready").touch()
                deadline = time.monotonic() + 10
                while not (root / "release").exists():
                    if time.monotonic() > deadline:
                        sys.exit(99)
                    time.sleep(0.01)
                sys.exit(subprocess.run([sys.executable, __file__, "child"]).returncode)
                '''))
            binary.chmod(0o755)
            packages = [{"name": "heavy", "manifest_path": str(root / "Cargo.toml"),
                         "metadata": {"chopinfer": {"lib-tests-in-one-process": True}}}]
            listed = {"rust-build-meta": {"target-directory": str(root)}, "rust-binaries": {
                "heavy": {"binary-id": "heavy", "kind": "lib", "binary-path": str(binary),
                          "package-id": "heavy"},
                "light": {"binary-id": "light", "kind": "lib", "binary-path": str(root / "light"),
                          "package-id": "light"}}}
            calls = []

            def invoke(command):
                calls.append(command)
                if command[:3] == ["cargo", "nextest", "run"]:
                    deadline = time.monotonic() + 10
                    while not ready.exists():
                        if time.monotonic() > deadline:
                            self.fail("the library test did not start")
                        time.sleep(0.01)
                    if "--binaries-metadata" in command:
                        self.assertFalse({"--workspace", "-p", *rust_tests.LOCKED} & set(command))
                        metadata = pathlib.Path(command[command.index("--binaries-metadata") + 1])
                        self.assertEqual(json.loads(metadata.read_text()), listed)
                    else:
                        # Model Cargo reclaiming the running executable for relinking.
                        binary.unlink()
                    release.touch()
                else:
                    self.assertEqual(command, ["cargo", "test", "--workspace", "--doc", *rust_tests.LOCKED])
                return subprocess.CompletedProcess(command, 0)

            output = io.BytesIO()
            with io.TextIOWrapper(output, encoding="utf-8") as stdout, \
                    patch.dict(os.environ, CHOPINFER_CI_SERIAL_LIBRARIES="0"), \
                    patch.object(rust_tests, "workspace_packages", return_value=packages), \
                    patch.object(rust_tests.subprocess, "check_output", return_value=json.dumps(listed).encode()) as build, \
                    patch.object(rust_tests.subprocess, "run", side_effect=invoke), \
                    patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", new_callable=io.StringIO):
                status = rust_tests.run_nextest(["--workspace"], 2)
                stdout.flush()
                self.assertEqual(status, 0, output.getvalue().decode())
            build.assert_called_once_with([
                "cargo", "nextest", "list", "--workspace", "--list-type", "binaries-only",
                "--message-format", "json", *rust_tests.LOCKED])
            self.assertEqual(len(calls), 2)

    def test_metadata_reuse_keeps_build_options_out_and_doctest_selection_intact(self):
        for selection in (["--workspace"], ["-p", "heavy", "-p", "light"]):
            with self.subTest(selection=selection), \
                    patch.object(rust_tests, "library_packages", return_value=["heavy", "light"]):
                run, doc = rust_tests.commands(selection, "nextest", 4, one_process=["heavy"],
                                               binaries_metadata="binaries.json")
                self.assertEqual(run, [
                    "cargo", "nextest", "run", "--test-threads", "4", "--profile", "ci",
                    "--no-tests", "warn", "--binaries-metadata", "binaries.json",
                    "-E", "not (binary_id(=heavy))"])
                self.assertEqual(doc, ["cargo", "test", *selection, "--doc", *rust_tests.LOCKED])

    def test_nextest_runs_the_selection_then_its_doctests(self):
        self.assertEqual(rust_tests.commands(["--workspace"], "nextest", 12), [
            ["cargo", "nextest", "run", "--test-threads", "12", "--workspace", "--profile", "ci",
             "--no-tests", "warn", "--locked", "--offline"],
            ["cargo", "test", "--workspace", "--doc", "--locked", "--offline"]])

    def test_doctests_run_only_for_selected_packages_with_a_library(self):
        with unittest.mock.patch.object(rust_tests, "library_packages", return_value=["core"]) as libraries:
            run, doc = rust_tests.commands(["-p", "core", "-p", "tool"], "nextest", 4)
        libraries.assert_called_once_with(["core", "tool"])
        self.assertEqual(run[5:7], ["-p", "core"])
        self.assertEqual(doc, ["cargo", "test", "-p", "core", "--doc", "--locked", "--offline"])
        with unittest.mock.patch.object(rust_tests, "library_packages", return_value=[]):
            self.assertEqual(len(rust_tests.commands(["-p", "tool"], "nextest", 4)), 1)

    def test_one_process_libraries_leave_nextest_and_follow_the_selection(self):
        packages = [{"name": "heavy", "metadata": {"chopinfer": {"lib-tests-in-one-process": True}}},
                    {"name": "light", "metadata": None}, {"name": "other", "metadata": {"chopinfer": {}}}]
        self.assertEqual(rust_tests.one_process_libraries(["--workspace"], packages), ["heavy"])
        self.assertEqual(rust_tests.one_process_libraries(["-p", "light"], packages), [])
        self.assertEqual(rust_tests.one_process_libraries(["-p", "heavy", "-p", "light"], packages), ["heavy"])
        run = rust_tests.commands(["--workspace"], "nextest", 8, one_process=["heavy", "more"])[0]
        self.assertEqual(run[-2:], ["-E", "not (binary_id(=heavy) | binary_id(=more))"])

    def test_the_declared_one_process_libraries_exist(self):
        self.assertEqual(rust_tests.one_process_libraries(["--workspace"]), ["bundle-glm5next", "bundle-mimo26"])

    def test_cargo_test_runs_the_same_selection_with_its_doctests(self):
        self.assertEqual(rust_tests.commands(["-p", "core"], "cargo", 4),
                         [["cargo", "test", "-p", "core", "--locked", "--offline"]])

    def test_a_missing_nextest_falls_back_to_cargo_and_says_so(self):
        with unittest.mock.patch.object(rust_tests, "nextest_version", return_value=None):
            runner, note = rust_tests.choose("auto")
            self.assertEqual(runner, "cargo")
            self.assertIn("slower", note)
            with self.assertRaisesRegex(SystemExit, "required but not installed"):
                rust_tests.choose("nextest")
        with unittest.mock.patch.object(rust_tests, "nextest_version", return_value="0.9.1"):
            self.assertEqual(rust_tests.choose("auto"), ("nextest", f"rust tests: cargo-nextest 0.9.1 (CI pins {rust_tests.PINNED})"))
        self.assertEqual(rust_tests.choose("cargo")[0], "cargo")

    def test_the_pinned_version_matches_the_nextest_config(self):
        config = tomllib.loads((ROOT / ".config/nextest.toml").read_text())
        self.assertEqual(config["nextest-version"]["required"], rust_tests.PINNED)

    def test_the_thread_count_changes_only_how_many_tests_run_at_once(self):
        old, new = (rust_tests.commands(["--workspace"], "nextest", threads, one_process=["bundle-mimo26"])
                    for threads in (8, 16))
        self.assertEqual(old[1:], new[1:])
        self.assertEqual(old[0][3:5], ["--test-threads", "8"])
        self.assertEqual(new[0][3:5], ["--test-threads", "16"])
        self.assertEqual(old[0][:4] + old[0][5:], new[0][:4] + new[0][5:])


JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="4" skipped="0" failures="1" errors="0" time="640.5">
    <testsuite name="lab" tests="3" skipped="0" errors="0" failures="1">
        <testcase name="sleeps_past_the_budget" classname="lab" time="600.004">
            <failure type="test timeout"/>
        </testcase>
        <testcase name="tie_b" classname="lab" time="2.000"/>
        <testcase name="tie_a" classname="lab" time="2.000"/>
    </testsuite>
    <testsuite name="core::api" tests="1" skipped="0" errors="0" failures="0">
        <testcase name="quick" classname="core::api" time="0.010"/>
    </testsuite>
</testsuites>
"""


class GateTimes(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)

    def report(self, name, cases, modified=None):
        path = self.directory / name
        path.write_text('<testsuites><testsuite name="s">' + "".join(
            f'<testcase name="{test}" classname="{binary}" time="{seconds}"/>' for seconds, binary, test in cases)
            + "</testsuite></testsuites>")
        if modified is not None:
            os.utime(path, (modified, modified))
        return path

    def test_the_slowest_tests_come_from_this_runs_junit_reports(self):
        recorded = self.directory / "junit.xml"
        recorded.write_text(JUNIT)
        self.assertEqual(cpu.slowest_tests([recorded], 0, 3), [
            (600.004, "lab", "sleeps_past_the_budget"), (2.0, "lab", "tie_a"), (2.0, "lab", "tie_b")])
        now = time.time()
        many = self.report("ci.xml", [(index, "bin", f"t{index}") for index in range(12)])
        stale = self.report("old.xml", [(999, "bin", "previous_run")], modified=now - 3600)
        slowest = cpu.slowest_tests([many, stale, self.directory / "absent.xml"], now - 60)
        self.assertEqual([case[2] for case in slowest], [f"t{index}" for index in range(11, 1, -1)])
        broken = self.directory / "broken.xml"
        broken.write_text("<testsuites>")
        self.assertEqual(cpu.slowest_tests([broken], 0), [])

    def test_a_full_gate_over_fifteen_minutes_warns(self):
        def printed(elapsed, full_gate, slowest=()):
            out = io.StringIO()
            cpu.report_times(elapsed, list(slowest), full_gate, out)
            return out.getvalue()

        self.assertIn("warning: the gate took 15.1 minutes", printed(906, True))
        self.assertEqual(printed(900, True), "")
        self.assertEqual(printed(3000, False), "")
        listing = printed(60, True, [(600.004, "lab", "sleeps_past_the_budget")])
        self.assertEqual(listing, "cpu-ci slowest tests:\n    600.004s lab sleeps_past_the_budget\n")

    def test_exhaustive_comparisons_run_only_on_request_and_last(self):
        names = lambda shared: [item.name for item in shared]
        shared, _ = cpu.steps(self.directory / "target", self.directory, 8, 8)
        self.assertNotIn("exhaustive-comparisons", names(shared))
        shared, _ = cpu.steps(self.directory / "target", self.directory, 8, 8, exhaustive=True)
        exhaustive = shared[-1]
        self.assertEqual(exhaustive.name, "exhaustive-comparisons")
        self.assertEqual(exhaustive.after, ("host-launch-arguments",))
        self.assertEqual(exhaustive.argv[:9], ["cargo", "nextest", "run", "--workspace", "--profile", "exhaustive",
                                               "--run-ignored", "only", "--no-tests"])


class NextestProfiles(unittest.TestCase):
    def test_the_ci_profile_bounds_each_test_and_the_exhaustive_tests_exist_ignored(self):
        config = tomllib.loads((ROOT / ".config/nextest.toml").read_text())
        ci = config["profile"]["ci"]
        self.assertEqual(ci["slow-timeout"], {"period": "60s", "terminate-after": 10})
        exhaustive = config["profile"]["exhaustive"]
        self.assertEqual(exhaustive["inherits"], "ci")
        clause = r"binary_id\(([\w-]+)\) & test\(=([\w:]+)\)"
        named = re.findall(clause, exhaustive["default-filter"])
        self.assertEqual(len(named), exhaustive["default-filter"].count("binary_id("))
        self.assertGreaterEqual(len(named), 6)

        def limits(profile):
            return {tuple(test): override["slow-timeout"] for override in profile.get("overrides", [])
                    if "slow-timeout" in override for test in re.findall(clause, override["filter"])}

        # A longer limit for an exhaustive test belongs to the exhaustive
        # profile only, so running it under ci keeps the 10-minute budget.
        self.assertFalse(limits(ci).keys() & set(named), limits(ci))
        for profile in (ci, exhaustive):
            for limit in limits(profile).values():
                self.assertEqual(limit["period"], "60s")
                self.assertGreater(limit["terminate-after"], 10)
        self.assertLessEqual(limits(exhaustive).keys(), set(named))
        self.assertEqual(limits(exhaustive)[("model-qwen38-contract", "projections::variants::tests::graph::"
                                             "complete_served_graphs_match_native_variant_specs_records_and_stages")],
                         {"period": "60s", "terminate-after": 60})
        packages = {tomllib.loads(path.read_text())["package"]["name"]: path.parent
                    for path in [*ROOT.glob("crates/*/Cargo.toml"), *ROOT.glob("models/*/*/Cargo.toml")]}
        for package, test in named:
            function = test.rsplit("::", 1)[-1]
            sources = [path.read_text() for path in packages[package].rglob("*.rs")]
            declared = [source for source in sources if re.search(rf"\bfn {function}\(", source)]
            self.assertTrue(declared, (package, test))
            self.assertTrue(any(re.search(rf"#\[ignore[^\]]*\]\s*fn {function}\(", source) for source in declared),
                            (package, test))

    def test_every_test_an_override_names_exists(self):
        """An override naming a renamed or split test matches nothing, and its limit or group
        silently stops applying."""
        config = tomllib.loads((ROOT / ".config/nextest.toml").read_text())
        packages = {tomllib.loads(path.read_text())["package"]["name"]: path.parent
                    for path in [*ROOT.glob("crates/*/Cargo.toml"), *ROOT.glob("models/*/*/Cargo.toml")]}
        clause = r"binary_id\(=?([\w-]+)(?:::[\w-]+)?\) & test\(=([\w:]+)\)"
        named = [found for profile in config["profile"].values() for override in profile.get("overrides", [])
                 for found in re.findall(clause, override["filter"])]
        self.assertGreaterEqual(len(named), 3)
        for package, test in named:
            with self.subTest(package=package, test=test):
                function = test.rsplit("::", 1)[-1]
                self.assertTrue(any(re.search(rf"\bfn {function}\(", path.read_text())
                                    for path in packages[package].rglob("*.rs")))


class Inventory(unittest.TestCase):
    def test_every_declared_module_still_pins_its_gate_config(self):
        run_once = qualification.load_inventory(ROOT)
        # B5 removed the two vLLM modules and renamed the frontend comparison.
        # Pin membership, so a lost declaration cannot hide behind a new one.
        self.assertEqual(run_once, {
            "test_agentic_markup", "test_answer_requests", "test_checkpoint_binding", "test_color_probe",
            "test_frontend_comparison", "test_frontend_extension", "test_grounding", "test_gsm8k_branches",
            "test_kinds", "test_limits",
            "test_logprobs_parity", "test_multimodal", "test_needle", "test_prefix_exact", "test_qwen_exact_pin",
            "test_qwen_teacher_gate", "test_reference_binding", "test_release_preview", "test_router_bias",
            "test_seeded_equality", "test_soak", "test_spec_metrics", "test_teacher", "test_variant",
            "test_verify_exact",
        })
        self.assertTrue(run_once < set(qualification.modules(ROOT)))

    def test_pins_are_read_from_module_level_setup_only(self):
        self.assertEqual(qualification.pin("from qual_test_config import setUpModule  # noqa: F401\n", "d"), "d")
        self.assertEqual(qualification.pin("def setUpModule():\n    configure('qwen38')\n", "d"), "qwen38")
        self.assertEqual(qualification.pin("def setUpModule():\n    configure()\n", "d"), "d")
        self.assertIsNone(qualification.pin("def test_x():\n    configure('qwen38')\n", "d"))
        self.assertIsNone(qualification.pin("from qual_test_config import setUpModule as other\n", "d"))
        self.assertIsNone(qualification.pin("def setUpModule():\n    configure(name)\n", "d"))


if __name__ == "__main__":
    unittest.main()
