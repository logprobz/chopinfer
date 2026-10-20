"""Warm the complete sampled workload before starting the one Nsight collection."""
import time


def prepare(runner, width):
    assert not runner.session.started
    receipts = []
    for concurrency in (8, 16):
        key = str(concurrency)
        if not runner.dry:
            assert key in runner.reference, "full warmup requires the pinned K3 output reference"
        for repeat in range(2):
            path = runner.directory / f"prepare-C{concurrency}-{repeat}"
            path.mkdir()
            before = width.faults(runner.server.pid)
            started = time.monotonic()
            requests = runner.wave(concurrency, width.TOKENS)
            after = runner.scrape()
            time.sleep(.02 if runner.dry else .2)
            assert after == runner.scrape(), "preparation counters did not settle"
            faults = width.fault_delta(before, width.faults(runner.server.pid))
            if key not in runner.reference:
                runner.reference[key] = requests
            parity = runner.exact.compare({key: runner.reference[key]}, {key: requests})
            assert parity["bitwise"], parity
            width.write(path / "outputs.json", requests)
            receipt = dict(concurrency=concurrency, repeat=repeat,
                           tokens_per_request=width.TOKENS, faults=faults, bitwise=parity,
                           seconds=time.monotonic()-started, collected_by_nsight=False)
            width.write(path / "summary.json", receipt)
            receipts.append(receipt)
    assert len(receipts) == 4
    return receipts
