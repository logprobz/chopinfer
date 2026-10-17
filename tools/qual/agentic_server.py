"""Observed server identity and prefill-budget evidence, never derived from arm labels."""
import hashlib
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from agentic_metrics import parse_metrics


def normalize_mode(value):
    return {'fixed': 'fixed', 'tbt': 'tbt', 'decode-aware': 'tbt'}.get(value)


def observe_startup(url):
    """Read only the process owning this local listener; do not persist full argv.

    Mirrors the listener lookup in soak.py:186. Remote endpoints need a metric
    reporting mode. No claim is made from a missing flag or guessed default.
    """
    endpoint = urlsplit(url)
    if endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
        return {'mode': None, 'unavailable': 'remote endpoint; startup args unavailable'}
    port = endpoint.port or (443 if endpoint.scheme == 'https' else 80)
    try:
        result = subprocess.run(['ss', '-H', '-ltnp', f'sport = :{port}'],
                                check=True, capture_output=True, text=True, timeout=3)
        pids = {int(pid) for pid in re.findall(r'\bpid=(\d+)\b', result.stdout)}
        if len(pids) != 1:
            raise ValueError('listener PID unavailable or ambiguous')
        pid = pids.pop()
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
        argv = raw.decode().rstrip('\0').split('\0')
        modes = []
        for i, arg in enumerate(argv):
            if arg == '--prefill-budget' and i + 1 < len(argv):
                modes.append(normalize_mode(argv[i + 1]))
            elif arg.startswith('--prefill-budget='):
                modes.append(normalize_mode(arg.partition('=')[2]))
        # proc stat's comm field may contain spaces or parentheses.
        start_ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        return dict(pid=pid, start_ticks=start_ticks, argv_sha256=hashlib.sha256(raw).hexdigest(),
                    mode=modes[0] if len(modes) == 1 else None,
                    source='verified listener /proc startup args',
                    unavailable=None if len(modes) == 1 and modes[0] else 'explicit mode flag unavailable')
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        return dict(mode=None, unavailable=type(error).__name__)


def mode_evidence(before, after, model, startup_before=None, startup_after=None):
    snapshots = []
    claims = []
    errors = []
    for point, snapshot in [('before', before), ('after', after)]:
        try:
            rows = parse_metrics(snapshot['text']) if not snapshot.get('error') else {}
        except (ValueError, TypeError):
            rows = {}
        rows = {key: value for key, value in rows.items()
                if dict(key[1]).get('model_name', dict(key[1]).get('model', model)) == model}
        snapshots.append(rows)
        for (name, labels), value in rows.items():
            labels = dict(labels)
            if value <= 0:
                continue
            if 'prefill_budget' in name or name.endswith('scheduler_info'):
                mode = next((normalize_mode(labels[k]) for k in ('mode', 'prefill_budget_mode', 'prefill_budget')
                             if k in labels and normalize_mode(labels[k])), None)
                if mode:
                    claims.append(dict(mode=mode, source='metrics mode label', point=point, metric=name))
    old, new = snapshots
    # This histogram is intentionally absent before its first observation in
    # compatibility.rs:285. Only this known lazy counter treats absence as zero.
    selected = 0
    prefixes = ('chop_', 'vllm:')
    selected_prefix = next((prefix for prefix in prefixes
                            if any(key[0] == prefix + 'prefill_budget_selected_tokens_count' for key in new)), None)
    for key, value in new.items():
        if selected_prefix and key[0] == selected_prefix + 'prefill_budget_selected_tokens_count':
            change = value - old.get(key, 0)
            if change < 0:
                errors.append('selected-budget counter reset')
            selected += max(0, change)
    if selected:
        claims.append(dict(mode='tbt', source='selected-budget observations (inferred)', observations=selected))
    for point, startup in [('before', startup_before), ('after', startup_after)]:
        if startup and startup.get('mode'):
            claims.append(dict(mode=startup['mode'], source='verified listener startup args', point=point))
    if startup_before and startup_after and startup_before.get('pid') and startup_after.get('pid'):
        if any(startup_before.get(k) != startup_after.get(k) for k in ('pid', 'start_ticks', 'argv_sha256')):
            errors.append('listener process identity changed')
    modes = {claim['mode'] for claim in claims}
    if len(modes) > 1:
        errors.append('conflicting mode evidence')
    return dict(mode=next(iter(modes)) if len(modes) == 1 and not errors else None,
                claims=claims, selected_budget_observations=selected, errors=errors,
                unavailable='mode not established' if not modes else None,
                startup_before=startup_before, startup_after=startup_after)


def arm_mode(arm):
    """The prefill-budget mode an arm label claims: `fixed` or `tbt`, alone
    or before a `-` suffix naming the arm's other server settings
    (`fixed-mbt4096`), else None."""
    mode = arm.split('-', 1)[0]
    return mode if mode in ('fixed', 'tbt') else None


def verify_arm(arm, evidence):
    mode = arm_mode(arm)
    if mode is None:
        return dict(passed=True, reason='unlabeled scheduling arm; mode not gated')
    reason = None
    if evidence['errors']:
        reason = '; '.join(evidence['errors'])
    elif evidence['mode'] != mode:
        reason = f"arm {arm} disagrees with observed mode {evidence['mode'] or 'unknown'}"
    warnings = []
    if mode == 'tbt' and evidence['mode'] == 'tbt' and not evidence['selected_budget_observations']:
        warnings.append('verified tbt mode but no selected-budget observations in the measured window')
    return dict(passed=reason is None, reason=reason, warnings=warnings,
                policy_exercised=bool(evidence['selected_budget_observations']))
