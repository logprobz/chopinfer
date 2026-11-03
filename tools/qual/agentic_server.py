"""Observed server identity and prefill-policy evidence, never derived from arm labels."""
import hashlib
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from agentic_metrics import parse_metrics


# Public serving policies. TBT metrics are shared by tbt and adaptive.
MODES = ('fixed', 'tbt', 'adaptive')
RETIRED_METRICS = ('chop_prefill_budget_decode_estimate_', 'chop_prefill_budget_prefill_token_')
RETIRED_FLAGS = ('--prefill-budget', '--prefill-timing-cuts', '--prefill-itl-target-ms',
                 '--prefill-decode-steps', '--mixed-min-prefill-tokens', '--mixed-decode-steps')


def normalize_mode(value):
    return value if value in MODES else None


def observe_startup(url):
    """Read only the process owning this local listener; do not persist full argv.

    Mirrors the listener lookup in soak.py:186. Remote endpoints need a metric
    reporting mode. No claim is made from a missing flag or guessed default.
    """
    endpoint = urlsplit(url)
    if endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
        return {'policy': None, 'unavailable': 'remote endpoint; startup args unavailable'}
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
            if arg == '--prefill-policy' and i + 1 < len(argv):
                modes.append(normalize_mode(argv[i + 1]))
            elif arg.startswith('--prefill-policy='):
                modes.append(normalize_mode(arg.partition('=')[2]))
        # proc stat's comm field may contain spaces or parentheses.
        start_ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        return dict(pid=pid, start_ticks=start_ticks, argv_sha256=hashlib.sha256(raw).hexdigest(),
                    policy=modes[0] if len(modes) == 1 else None,
                    retired=any(arg.partition('=')[0] in RETIRED_FLAGS for arg in argv),
                    source='verified listener /proc startup args',
                    unavailable=None if len(modes) == 1 and modes[0] else 'explicit mode flag unavailable')
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        return dict(policy=None, unavailable=type(error).__name__)


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
            if name.startswith(RETIRED_METRICS):
                errors.append('retired decode-aware metric evidence')
            if name == 'chop_prefill_policy_info':
                policy = normalize_mode(labels.get('policy'))
                if value != 1 or policy is None or 'engine' not in labels or labels.get('model_name') != model:
                    errors.append('invalid canonical policy metric')
                else:
                    claims.append(dict(mode=policy, source='canonical policy metric', point=point, metric=name,
                                       engine=labels['engine']))
            elif 'prefill_budget' in name or name.endswith('scheduler_info'):
                if any(labels.get(k, '').startswith('decode-aware') for k in ('mode', 'prefill_budget_mode', 'prefill_budget')):
                    errors.append('retired decode-aware label evidence')
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
    for point, startup in [('before', startup_before), ('after', startup_after)]:
        if startup:
            if startup.get('retired') or str(startup.get('mode', '')).startswith('decode-aware'):
                errors.append('retired prefill startup evidence')
            if startup.get('policy') is not None:
                policy = normalize_mode(startup['policy'])
                if policy is None:
                    errors.append('invalid canonical startup policy')
                else:
                    claims.append(dict(mode=policy, source='verified listener startup args', point=point))
    if startup_before and startup_after and startup_before.get('pid') and startup_after.get('pid'):
        if any(startup_before.get(k) != startup_after.get(k) for k in ('pid', 'start_ticks', 'argv_sha256')):
            errors.append('listener process identity changed')
    modes = {claim['mode'] for claim in claims}
    if len(modes) > 1:
        errors.append('conflicting mode evidence')
    if selected and modes == {'fixed'}:
        errors.append('selected-budget observations under the fixed budget')
    return dict(identity='prefill-policy', mode=next(iter(modes)) if len(modes) == 1 and not errors else None,
                claims=claims, selected_budget_observations=selected, errors=errors,
                unavailable='mode not established' if not modes else None,
                startup_before=startup_before, startup_after=startup_after)


def arm_mode(arm):
    """The prefill policy an arm label claims: a mode in MODES, alone
    or before a `-` suffix naming the arm's other server settings
    (`fixed-mbt4096`, `tbt-150ms`), else None."""
    return next((mode for mode in sorted(MODES, key=len, reverse=True)
                 if arm == mode or arm.startswith(mode + '-')), None)


def verify_arm(arm, evidence):
    if arm.startswith('decode-aware'):
        return dict(passed=False, reason='decode-aware is retired; use fixed, tbt or adaptive')
    mode = arm_mode(arm)
    if evidence['errors'] or evidence.get('mode') not in MODES:
        return dict(passed=False, reason='; '.join(evidence['errors']) or 'canonical prefill policy not established')
    if mode is None:
        return dict(passed=True, reason='unlabeled scheduling arm; canonical policy verified')
    reason = None
    if evidence['mode'] != mode:
        reason = f"arm {arm} disagrees with observed mode {evidence['mode'] or 'unknown'}"
    warnings = []
    if mode in ('tbt', 'adaptive') and evidence['mode'] == mode and not evidence['selected_budget_observations']:
        warnings.append(f'verified {mode} policy but no selected-budget observations in the measured window')
    return dict(passed=reason is None, reason=reason, warnings=warnings,
                policy_exercised=bool(evidence['selected_budget_observations']))
