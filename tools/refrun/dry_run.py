"""Check a reference run's command line on CPU, without loading weights.

    python -m tools.refrun.dry_run ARGUMENTS...

takes exactly the arguments a job passes to `python -m tools.refrun.run`,
the `run` subcommand first, and parses them with the runner's own parser.
For `run` it then takes every step the runner takes before the first weight
is read: it loads the adapter and the workload and matches their model,
resolves the modes and their rules, verifies the checkpoint files the run
reads against the source lock (through the stamp cache, which this may
fill), lists the adapter's units and places them over the devices. It needs
no GPU and initializes no CUDA, creates no output, and prints what the run
would do as JSON; any refusal exits nonzero with the runner's message. The
device list is checked against CUDA_VISIBLE_DEVICES rather than the driver.

It lives outside run.py, a keyed numerics file (keys.NUMERICS), so it can
change without changing any cache key.
"""

import json
import os
from pathlib import Path
import re
import sys

from tools.refrun import keys, modes, place, run as runner, stamps, workload


def visible_devices():
    """How many CUDA devices the process may use, from CUDA_VISIBLE_DEVICES;
    None when it is unset (the driver decides)."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return None if value is None else len([item for item in value.split(",") if item.strip()])


def check(args):
    """The runner's steps before the weights load; a dict describing the run."""
    devices = args.devices.split(",")
    for device in devices:
        if device != "cpu" and not re.fullmatch(r"cuda:\d+", device):
            raise ValueError(f"--devices: {device} is neither cpu nor cuda:N")
    cuda = sorted({int(device[5:]) for device in devices if device.startswith("cuda:")})
    visible = visible_devices()
    if cuda and visible is not None and cuda[-1] >= visible:
        raise ValueError(f"--devices names cuda:{cuda[-1]}, but CUDA_VISIBLE_DEVICES makes {visible} visible")
    modes.prepare_environment(cuda=False)
    reference_adapter, adapter_path = runner.load_adapter(args.adapter)
    model_name, sequences = workload.load(args.workload)
    if model_name != reference_adapter.MODEL:
        raise ValueError(f"workload is for {model_name}, adapter for {reference_adapter.MODEL}")
    policies = runner.mode_policies(args.modes, args.dequant)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"--output {output} exists; the runner creates it")
    parent = next(path for path in (output.absolute(), *output.absolute().parents) if path.exists())
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"--output {output}: {parent} is not writable")
    lock = keys.read_lock(args.lock)
    if args.lock_variant is not None:
        lock = lock[args.lock_variant]
    root = args.checkpoint
    first = ["config.json", place.INDEX] if (root / place.INDEX).exists() else ["config.json"]
    verified = stamps.verify(root, lock, first, args.stamps)
    extra = getattr(reference_adapter, "EXTRA_SHARDS", {})
    verified.update(stamps.verify(root, lock, [*place.shard_files(root), *extra], args.stamps))
    checkpoint = place.Checkpoint(root, extra)
    unverified = sorted(set(checkpoint.read) - set(verified))
    if unverified:
        raise RuntimeError(f"checkpoint files changed during verification: {unverified}")
    units = reference_adapter.units(checkpoint)
    files = sorted({"config.json", *checkpoint.files([n for u in units for n in u.tensors])})
    keys.checkpoint(lock, files)
    placed = [u for u in units if not u.host]
    placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in placed], devices)
    load = {}
    for unit, (_, device) in zip(placed, placement.units):
        load[device] = load.get(device, 0) + checkpoint.stored_bytes(unit.tensors)
    return {"runner": str(Path(runner.__file__).resolve()), "adapter": adapter_path, "model": model_name,
            "sequences": len(sequences), "modes": {label: policy["dequant"] for label, policy in policies},
            "checkpoint": str(root), "files_verified": len(verified), "files_read": len(files),
            "units": len(units), "host_units": len(units) - len(placed),
            "stored_gib": {device: round(size / 2**30, 2) for device, size in load.items()},
            "devices": devices, "output": str(output)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # The runner parses and dispatches; every command it could dispatch to is
    # replaced, so nothing runs, and only `run` is checked further.
    parsed, commands = [], ("run", "collect", "diff", "verify_binding")
    real = {name: getattr(runner, name) for name in commands}
    for name in commands:
        setattr(runner, name, lambda args, name=name: parsed.append((name, args)))
    try:
        runner.main(argv)
    finally:
        for name, function in real.items():
            setattr(runner, name, function)
    if parsed[0][0] != "run":
        raise SystemExit(f"dry run checks the run subcommand, not {argv[0]}")
    try:
        description = check(parsed[0][1])
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"dry run refused: {type(error).__name__}: {error}")
    print(json.dumps({"dry_run": "ok", **description}, indent=2))


if __name__ == "__main__":
    main()
