"""Route qualification commands through their existing script entry points."""

import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
# Command name to the tools/qual script it runs.
COMMANDS = {
    "gate": "release_gate", "preview": "release_preview",
    "check": "checks", "rescore": "checks", "list": "checks", "refs": "refs", "variant": "variant",
}
COMMANDS.update({
    name.replace("_", "-"): name for name in (
        "agentic_compare", "agentic_mixed", "api", "bench", "color_probe",
        "determinism", "encoder_parity", "evals", "fill_batch", "frontend_goldens", "greedy_equality",
        "grounding", "gsm8k", "kv_offload", "kvo_resume", "logprobs_parity", "multimodal",
        "needle", "preempt_compare", "prefix_cache", "prefix_exact",
        "qwen38_layers", "soak",
        "spec_bench", "spec_continuation", "spec_verify", "step_overlap_ab", "structured", "tapdiff",
        "teacher_arms", "paired_teacher", "verify_exact", "vectors",
        "seeded_equality",
    )
})


DESCRIPTIONS = {
    "gate": "run or inspect a release gate",
    "preview": "prepare, run or summarize a release preview",
    "check": "run one declared check against a server",
    "rescore": "re-judge stored check receipts offline",
    "list": "list check kinds and declared checks",
    "refs": "adopt and verify locked qualification inputs",
    "variant": "lock, pin and adopt an external variant (another checkpoint, outside the tree)",
}


def main(argv=None):
    """Parse the command, then let its existing parser own all remaining flags."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python3 -m tools.qual", description="Developer qualification checks and release orchestration.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, module in COMMANDS.items():
        commands.add_parser(command, add_help=False, help=DESCRIPTIONS.get(command, f"run {module.replace('_', ' ')}"))
    selected = parser.parse_args(argv[:1]).command
    module = COMMANDS[selected]
    arguments = argv[1:]
    if selected in ("check", "rescore", "list"):
        arguments.insert(0, selected)
    if selected == "preview":
        # Replace this process so the wrapper owns its signals and exit status.
        os.execvp("bash", ["bash", str(ROOT / "release_preview.sh"), *arguments])
    # Running the existing guard also preserves signal handling, worker mode,
    # error reporting and scripts that have more than one main function.
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.path.insert(0, str(ROOT))
        # run_path sets argv[0] to the script path.
        sys.argv = [sys.argv[0], *arguments]
        runpy.run_path(str(ROOT / f"{module}.py"), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
    return 0
