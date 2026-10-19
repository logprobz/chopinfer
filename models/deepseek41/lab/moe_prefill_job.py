#!/usr/bin/env python3
"""Report exact MoE prefill pairs against same-run arithmetic and traffic ceilings."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / 'bin/moe-prefill-tests'
TESTS = ['moe_active::prefill::prefill_backbone_gpu', 'moe_active::prefill::prefill_dspark_gpu']
PARTS = ['input_quant', 'router_projection', 'routing', 'gateup', 'intermediate_quant', 'down', 'shared', 'combine', 'full']


def expected_cases():
    return {f'e{e}k{k}-Prefill-{r}-{routing}' for e, k in [(384, 6), (128, 3)]
            for r in [512, 2048, 4096] for routing in ['All', 'Seeded', 'Concentrated']}


def roofline(record):
    """Per-rank logical traffic, explicitly separate from cache-amplified loads."""
    rows, top_k, experts = record['rows'], record['top_k'], record['experts']
    counts = record['counts']
    hidden, inter = 5120, 576
    active = sum(n > 0 for n in counts)
    tiles = sum((n + 63) // 64 for n in counts)
    wide = sum(n // 64 + (n % 64 > 32) for n in counts)
    padded = tiles * 64
    narrow = sum((n + 7) // 8 * 8 for n in counts)
    routes = rows * top_k
    fp4, fp8 = 17 / 32, 33 / 32
    gate_weight = 2 * hidden * inter * fp4
    down_weight = hidden * inter * fp4
    logical = {'gateup': 4 * routes * hidden * inter, 'down': 2 * routes * hidden * inter,
               'shared': 6 * rows * hidden * inter, 'router_projection': 2 * rows * experts * hidden}
    issued = {'old': {'gateup': 4 * padded * hidden * inter, 'down': 2 * padded * hidden * 640},
              'grouped': {'gateup': 4 * narrow * hidden * inter, 'down': 2 * narrow * hidden * 640}}
    traffic = {
        'input_quant': rows * hidden * (2 + fp8),
        'router_projection': rows * hidden * 4 + experts * hidden * 2 + rows * experts * 4,
        'routing': rows * experts * 8 + experts * 8 + routes * 8 + padded * 4 + (3 * experts + 4) * 4,
        'gateup': active * gate_weight + rows * hidden * fp8 + routes * 4 + routes * inter * 2,
        'intermediate_quant': routes * inter * (2 + fp8),
        'down': active * down_weight + routes * inter * fp8 + routes * hidden * 4,
        'shared': 3 * hidden * inter * fp8 + rows * hidden * fp8 + rows * inter * 16 + rows * inter * fp8 * 2 + rows * hidden * 4,
        'combine': routes * hidden * 4 + rows * hidden * 8 + routes * 4,
    }
    traffic['full'] = sum(traffic.values())
    logical['full'] = sum(logical.values())
    calibration = record['calibration']
    bf16 = logical['gateup'] + logical['down']
    compute_floor = bf16 / (calibration['bf16_tflops'] * 1e9)
    floors = {}
    for part in ['gateup', 'down', 'full']:
        compute = (bf16 if part == 'full' else logical[part]) / (calibration['bf16_tflops'] * 1e9)
        floors[part] = {'bf16_compute_ms': compute,
                        'cold_dram_ms': max(compute, traffic[part] / (calibration['dram_gb_s'] * 1e6)),
                        'optimistic_l2_ms': max(compute, traffic[part] / (calibration['l2_gb_s'] * 1e6))}
    return {'logical_flops': logical, 'issued_routed_bf16_flops': issued, 'compulsory_bytes': traffic,
            'unique_routed_weight_bytes': active * (gate_weight + down_weight),
            'scheduled_routed_weight_bytes': tiles * (gate_weight + down_weight),
            'grouped_weight_load_instruction_bytes': (tiles + wide) * (gate_weight + down_weight),
            'padded_rows': padded, 'grouped_issued_rows': narrow, 'old_row_utilization': routes / padded,
            'bf16_routed_floor_ms': compute_floor, 'floors': floors,
            'tp4_logical_flops': {k: v * 4 for k, v in logical.items()},
            'tp4_compulsory_bytes': {k: v * 4 for k, v in traffic.items()}}


def parse_records(log):
    return [json.loads(line.partition('PREFILL_RESULT ')[2]) for line in log.splitlines() if 'PREFILL_RESULT ' in line]


def check_records(records):
    errors, seen = [], set()
    for record in records:
        case = record['case']
        if case in seen:
            errors.append(f'duplicate {case}')
        seen.add(case)
        rows, top_k = record['rows'], record['top_k']
        counts = record['counts']
        if len(counts) != record['experts'] or sum(counts) != rows * top_k or any(not 0 <= n <= rows for n in counts):
            errors.append(f'{case}: invalid route counts')
        if sum(n > 0 for n in counts) != record['active_experts']:
            errors.append(f'{case}: active count differs')
        if record['routing'] == 'All' and record['active_experts'] != record['experts']:
            errors.append(f'{case}: all-active coverage missing')
        if record['routing'] == 'Concentrated' and record['active_experts'] != top_k:
            errors.append(f'{case}: concentrated coverage missing')
        if (record['bitwise'] is not True or record['negative_detected'] is not True
                or record['compared_outputs'] != 15 or record['live_replays'] != [rows, rows//2+1, 0, rows]):
            errors.append(f'{case}: incomplete parity or negative-control evidence')
        if record['selection'] != 'old prefill' or record['ranks'] != 4:
            errors.append(f'{case}: unexpected serving selection or topology')
        for name in ['bf16_tflops', 'dram_gb_s', 'l2_gb_s']:
            value = record['calibration'][name]
            if not math.isfinite(value) or value <= 0:
                errors.append(f'{case}: invalid calibration {name}')
        if record['calibration']['required_power_limit_w'] != 250:
            errors.append(f'{case}: wrong calibration power cap')
        if set(record['timing']) != set(PARTS):
            errors.append(f'{case}: missing stage breakdown')
        for part, timing in record['timing'].items():
            if len(timing['ms']) != 2 or any(not math.isfinite(t) or t <= 0 for t in timing['ms']):
                errors.append(f'{case} {part}: invalid paired medians')
                continue
            rounds = timing['round_samples_ms']
            if (len(rounds) != 5 or any(len(r) != 2 or any(len(s) != 5 for s in r) for r in rounds)
                    or timing['replays_per_sample'] != 2):
                errors.append(f'{case} {part}: insufficient interleaved samples')
                continue
            if any(not math.isfinite(t) or t <= 0 for r in rounds for s in r for t in s):
                errors.append(f'{case} {part}: invalid samples')
            expected = [sorted(sorted(r[arm])[2] for r in rounds)[2] for arm in range(2)]
            if any(not math.isclose(t, e, rel_tol=1e-6) for t, e in zip(timing['ms'], expected)):
                errors.append(f'{case} {part}: median of medians differs')
    if seen != expected_cases():
        errors.append(f'case coverage differs: missing={sorted(expected_cases()-seen)}, extra={sorted(seen-expected_cases())}')
    return errors


def speed_table(records):
    lines = [
        'TP4 prefill, one rank per GPU. Candidate is experimental; serving remains old.',
        'Stored weights MXFP4/UE8M0-32; activations MXFP8/UE8M0-32; routed arithmetic decoded BF16, fixed 128-K chains.',
        'Ceilings are measured on each assigned GPU at 250 W. Router F32 and shared MXFP8 FLOPs are reported separately from the routed BF16 ceiling.',
        'Timing is warm graph replay. Cold-DRAM roof assumes every unique operand is fetched from DRAM; it is a reference for cache-resident cases, not a bound on warm replay.',
        'Optimistic L2 roof assumes all logical traffic hits L2. Bytes count logical stage reads/writes once; real traffic can be higher.',
        'Full floor omits the router, quantizer, unpack, shared MMA and synchronization instruction costs. Stage times need not sum to full graph time.',
        'A BF16 identity prefix and dense common suffix produce balanced, seeded and concentrated routes through the actual full router shape.',
        'case | active | logical GFLOP (all/routed BF16) | logical GB | padded/issued grouped rows | BF16 TF/s | DRAM/L2 GB/s | old/grouped ms by stage | grouped/old | old/grouped over BF16 floor | old/grouped over cold roof | old/grouped over L2 roof',
    ]
    for record in sorted(records, key=lambda r: (r['experts'], r['rows'], r['routing'])):
        model = roofline(record)
        old, new = record['timing']['full']['ms']
        c = record['calibration']
        f = model['floors']['full']
        stages = '; '.join(p + '=' + '/'.join(f'{v:.4f}' for v in record['timing'][p]['ms']) for p in PARTS)
        bf16 = model['logical_flops']['gateup'] + model['logical_flops']['down']
        ratio = lambda floor: f'{old/floor:.2f}/{new/floor:.2f}'
        lines.append(f"{record['case']} | {record['active_experts']} | {model['logical_flops']['full']/1e9:.3f}/{bf16/1e9:.3f} | "
                     f"{model['compulsory_bytes']['full']/1e9:.3f} | {model['padded_rows']}/{model['grouped_issued_rows']} | "
                     f"{c['bf16_tflops']:.2f} | {c['dram_gb_s']:.1f}/{c['l2_gb_s']:.1f} | {stages} | {new/old:.4f} | "
                     f"{ratio(f['bf16_compute_ms'])} | {ratio(f['cold_dram_ms'])} | {ratio(f['optimistic_l2_ms'])}")
    return '\n'.join(lines) + '\n'


def checked(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout


def preflight():
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    dry_test = 'moe_active::prefill::prefill_cpu_dry_run'
    listing = checked([str(BINARY), 'moe_active::prefill::', '--list'])
    for name in TESTS + [dry_test]:
        assert name + ': test' in listing, name
    assert 'libcuda.so.1' in checked(['ldd', str(BINARY)])
    env = os.environ.copy()
    for name in ['CHOPINFER_KERNEL_GPU', 'CHOPINFER_KERNEL_GPUS', 'CHOPINFER_KERNEL_REQUIRE']:
        env.pop(name, None)
    dry = checked([str(BINARY), dry_test, '--exact', '--nocapture'], env=env)
    assert '1 passed; 0 failed; 0 ignored' in dry and 'skipping' not in dry
    (ROOT/'dry-run.log').write_text(dry)
    checked(['journalctl', '-k', '-n', '0', '--no-pager'])
    return {'cpu_dry_run': 'passed', 'tests': TESTS, 'sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest()}


def main():
    record = preflight()
    (ROOT / "dry-run.json").write_text(json.dumps(record, indent=2) + "\n")
    if sys.argv[1:] == ["--dry-run"]:
        print(json.dumps(record))
        return
    assert sys.argv[1:] == ["--run"]
    pair = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert re.fullmatch(r"[0-9]+,[0-9]+", pair), "runner must name two idle GPU ordinals"
    devices = pair.split(",")
    assert devices[0] != devices[1]
    start = int(time.time())
    out = ROOT / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()))
    out.mkdir(parents=True)
    processes = []
    logs = []
    failed = False
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        power = checked(["nvidia-smi", "-i", pair, "--query-gpu=index,name,power.limit,power.draw,clocks.current.sm,clocks.current.memory", "--format=csv,noheader,nounits"])
        (out / "conditions.csv").write_text(power)
        rows = [line.split(",") for line in power.strip().splitlines()]
        assert len(rows) == 2 and all(abs(float(row[2]) - 250) < 0.1 for row in rows), "expected 250 W cap"
        for device, selector in zip(devices, TESTS):
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("CHOPINFER_KERNEL_GPUS", None)
            env["CHOPINFER_KERNEL_GPU"] = device
            env["CHOPINFER_KERNEL_REQUIRE"] = "1"
            env["CHOPINFER_KERNEL_TIMEOUT_SECS"] = "1000"
            env["CHOPINFER_KERNEL_RECEIPTS"] = str(out / ("receipts-" + device))
            log = open(out / ("gpu-" + device + ".log"), "w")
            logs.append(log)
            processes.append(subprocess.Popen([str(BINARY), selector, "--exact", "--nocapture", "--test-threads=1"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        codes = [process.wait(timeout=max(1, 1100 - (time.time() - start))) for process in processes]
        for log in logs:
            log.flush()
        (out / "exit-codes.json").write_text(json.dumps(codes) + "\n")
        records = []
        for device in devices:
            output = (out / ("gpu-" + device + ".log")).read_text()
            records.extend(parse_records(output))
        for record in records:
            record["roofline"] = roofline(record)
        table = speed_table(records)
        (out / "speed-table.txt").write_text(table)
        (out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
        print(table, flush=True)
        errors = check_records(records)
        (out / "performance-errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        assert codes == [0, 0], codes
        assert not errors, errors
        for device in devices:
            output = (out / ("gpu-" + device + ".log")).read_text()
            assert "1 passed; 0 failed; 0 ignored" in output and "skipping" not in output
        power_after = checked(["nvidia-smi", "-i", pair, "--query-gpu=index,name,power.limit,power.draw,clocks.current.sm,clocks.current.memory", "--format=csv,noheader,nounits"])
        (out / "conditions-after.csv").write_text(power_after)
        rows_after = [line.split(",") for line in power_after.strip().splitlines()]
        assert len(rows_after) == 2 and all(abs(float(row[2]) - 250) < 0.1 for row in rows_after), "power cap changed"
        (out / "profiling.json").write_text(json.dumps({"status":"disabled", "gates_verdict":False,
            "reason":"Profile separately; performance-counter access is not required for qualification."}) + "\n")
    except BaseException:
        failed = True
        raise
    finally:
        for process in processes:
            # A worker parent can exit while its child still
            # lives. Terminate the whole group even after the parent exits.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
        journal = checked(["journalctl", "-k", "--since", "@" + str(start), "--no-pager"])
        (out / "kernel-journal.log").write_text(journal)
        xids = [line for line in journal.splitlines() if re.search(r"NVRM:.*Xid", line, re.I)]
        (out / "xids.json").write_text(json.dumps(xids) + "\n")
        (out / "verdict.json").write_text(json.dumps({"passed": not failed and not xids, "power_limit_w": 250, "xids": len(xids), "candidate_speed_gating": False}) + "\n")
        assert not xids, xids
    print(str(out))

if __name__ == "__main__":
    main()
