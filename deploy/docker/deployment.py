"""Validate the container and systemd deployment environment before startup.

The shared variables are declared here. A model adds its own in a deployment
policy (`chopinfer-deployment-policy-v1`, checked as model-compiler-sdk's
deployment_policy.rs checks it): each installed model helper reports its
policy (`chopinfer-model-<m> deployment-policy`), so every installed model's
variables are known before preparation, and the prepared package carries the
policy of the bundle it serves.
"""
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import stat
import sys
from pathlib import Path
import subprocess

PATHS = set('BINARY MODEL MODEL_CACHE VERIFY_MANIFEST SAMPLING_CAPTURE '
            'DEBUG_REFERENCE_TOKENS'.split())
# Per-layer taps run only in a bundle built with its diagnostic-taps feature;
# the image builds serving bundles, which refuse to start with them set. A
# model's policy names its own tap variables.
TAPS = set('TAPS_DIR TAPS_LAYERS TAPS_SITES TAPS_FROM TAPS_BYTES TAPS_SCOPE_JSON'.split())
BOOLEANS = set('VERIFY_MODEL REQUEST_TRACE VIDEO_FRAME_CHECKSUMS'.split())
PRESENCE = {'SAMPLING_TIMING'}
NUMBERS = {
    **{name: (1, 2**64 - 1) for name in ('MAX_SEQUENCES', 'MAX_BATCH_TOKENS',
       'PREFILL_CHUNK_TOKENS', 'MAX_CONTEXT_TOKENS', 'STEP_PROFILE_CAPACITY', 'SAMPLING_CAPTURE_ROWS')},
    **{name: (0, 2**64 - 1) for name in ('DRAIN_TIMEOUT_MS', 'STEP_PROFILE_EVERY', 'QUEUE_DEPTH')},
    'SPECULATIVE_TOKENS': (0, 2**32 - 1),
    'BATCH_REFUSAL_MAX_CONSECUTIVE': (0, 10000),
    'BATCH_REFUSAL_MAX_IN_WINDOW': (0, 10000),
    'BATCH_REFUSAL_WINDOW_SECONDS': (1, 86400),
}
CHOICES = {'SPECULATION': {'fixed', 'adaptive'}, 'DRAFT_COST': {'per-token', 'per-block'},
           'KV_CACHE_DTYPE': {'auto', 'bf16', 'fp8'},
           'TENSOR_PARALLEL_SIZE': {'2', '4', '8'}}
# Variables of the package deployment, which loaded rank images from a mounted
# package into an install root. The checkpoint now loads directly.
GONE = {
    'PACKAGE': 'mount the checkpoint and set CHOPINFER_MODEL',
    'INSTALL_ROOT': 'set CHOPINFER_MODEL_CACHE to a writable model cache directory',
    'BUNDLE_VERIFY_CACHE_DIR': 'set CHOPINFER_VERIFY_MODEL=1 to check the checkpoint files',
    'BUNDLE_FORCE_FULL_VERIFY': 'set CHOPINFER_VERIFY_MODEL=1 to check the checkpoint files',
    'MODULE': 'the model helper packages the module installed beside it',
    'VIDEO_INPUT': 'set CHOPINFER_MODALITIES=all, or a list naming video, to serve video',
}
KNOWN = {'MODEL_OPTIONS'} | PATHS | BOOLEANS | PRESENCE | NUMBERS.keys() | CHOICES.keys() | {
    'API_KEY', 'LISTEN', 'KV_MEMORY_FRACTION', 'SERVE_ARGS', 'EXPECTED_BUNDLE_ID', 'MODALITIES'}
# Every shared name. A model policy may neither declare one nor own a
# prefix one starts with, so no policy changes a shared constraint; the
# compiler's reader (deployment_policy.rs SHARED_VARIABLES) holds the same
# list.
SHARED = KNOWN | TAPS | GONE.keys()
MODALITIES = ('text', 'image', 'video', 'audio')
# Speculation variables apply only to a package that declares a drafter.
SPECULATIVE = {'SPECULATIVE_TOKENS', 'SPECULATION', 'DRAFT_COST'}
BUNDLE_ID = r'[a-z0-9][a-z0-9.-]*'
DRAFTER_KIND = r'[a-z0-9][a-z0-9_-]*'

POLICY_SCHEMA = 'chopinfer-deployment-policy-v1'
POLICY_ROLE = 'chop.deployment-policy.v1'
POLICY_MAX_BYTES = 256 * 1024


def policy(data, source):
    """One model's deployment policy from the bytes `data`, read at `source`,
    with its optional fields filled in; refused unless it is valid."""
    def invalid(reason):
        raise ValueError(f'{source}: invalid deployment policy: {reason}')

    def unique_keys(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            invalid('a key repeats')
        return dict(pairs)

    def strings(value, pattern):
        return (isinstance(value, list) and all(isinstance(item, str) and re.fullmatch(pattern, item)
                                                for item in value) and len(set(value)) == len(value))

    def integer(value, low=0):
        return type(value) is int and low <= value <= 2**64 - 1

    def unsigned(text):
        # No member takes a negative integer; the compiler refuses -0 too.
        if text.startswith('-'):
            invalid(f'negative integer {text}')
        return int(text)

    if len(data) > POLICY_MAX_BYTES:
        invalid('it exceeds its bound')
    try:
        # Strict UTF-8 without a byte order mark, as the compiler reads it.
        value = json.loads(data.decode('utf-8'), object_pairs_hook=unique_keys, parse_int=unsigned)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        invalid(str(error))

    def texts(item):
        if isinstance(item, str):
            yield item
        elif isinstance(item, list):
            for inner in item:
                yield from texts(inner)
        elif isinstance(item, dict):
            for key, inner in item.items():
                yield key
                yield from texts(inner)

    # An escaped surrogate is decoded only as half of a pair; the compiler's
    # reader refuses a lone or reversed one.
    if any(re.search(r'[\ud800-\udfff]', text) for text in texts(value)):
        invalid('a string holds an unpaired surrogate escape')
    required = {'schema', 'bundle_id', 'knob_prefixes', 'drafters'}
    if not isinstance(value, dict) or not required <= value.keys() or not value.keys() <= required | {
            'paths', 'booleans', 'presence', 'numbers', 'multiples', 'taps', 'drafter_only', 'disabled_by'}:
        invalid('expected its schema, bundle_id, knob_prefixes and drafters and no unknown field')
    value = {'paths': [], 'booleans': [], 'presence': [], 'numbers': {}, 'multiples': {}, 'taps': [],
             'drafter_only': {}, 'disabled_by': [], **value}
    if value['schema'] != POLICY_SCHEMA:
        invalid(f'schema must be {POLICY_SCHEMA}')
    if not isinstance(value['bundle_id'], str) or not re.fullmatch(BUNDLE_ID, value['bundle_id']):
        invalid('invalid bundle ID')
    if not strings(value['knob_prefixes'], '[A-Z][A-Z0-9_]*_') or not value['knob_prefixes']:
        invalid('knob prefixes must be distinct names ending in _')
    if not strings(value['drafters'], DRAFTER_KIND):
        invalid('drafters must be distinct drafter kinds')
    prefixes = tuple(value['knob_prefixes'])
    if any(name.startswith(prefixes) for name in SHARED):
        invalid('a knob prefix covers a shared variable')
    numbers, multiples = value['numbers'], value['multiples']
    if not isinstance(numbers, dict) or not all(
            isinstance(bounds, list) and len(bounds) == 2 and all(map(integer, bounds))
            and bounds[0] <= bounds[1] for bounds in numbers.values()):
        invalid('numbers must map names to an inclusive integer range')
    declared = [*value['paths'], *value['booleans'], *value['presence'], *numbers, *value['taps']] \
        if all(strings(value[key], '[A-Z][A-Z0-9_]*') for key in ('paths', 'booleans', 'presence', 'taps')) \
        else None
    if declared is None or len(set(declared)) != len(declared):
        invalid('variables must be declared once each')
    if not all(re.fullmatch('[A-Z][A-Z0-9_]*', name) and name.startswith(prefixes) for name in declared):
        invalid('every variable must start with a knob prefix')
    if SHARED.intersection(declared):
        invalid('a variable is a shared one')
    if not isinstance(multiples, dict) or not all(
            name in numbers and integer(multiple, 1) for name, multiple in multiples.items()):
        invalid('multiples must name declared numbers')
    settable = set(declared) - set(value['taps'])
    drafter_only = value['drafter_only']
    if not isinstance(drafter_only, dict) or not all(
            name in settable and strings(kinds, DRAFTER_KIND) and kinds
            and set(kinds) <= set(value['drafters']) for name, kinds in drafter_only.items()):
        invalid('drafter_only must name declared variables and drafters')
    rules = value['disabled_by']
    if not isinstance(rules, list) or not all(
            isinstance(rule, dict) and rule.keys() == {'variable', 'value', 'prefix', 'reason'}
            and all(isinstance(rule[key], str) and rule[key] for key in rule)
            and rule['variable'] in settable and re.fullmatch('[A-Z][A-Z0-9_]*', rule['prefix'])
            and rule['prefix'].startswith(prefixes) for rule in rules):
        invalid('disabled_by must disable a prefix the model owns, with a value and reason')
    return value


def installed(binary):
    """The policies the model helpers installed beside `binary` report. A
    helper that reports none, or cannot, adds none."""
    policies = []
    # By absolute path: a relative BINARY such as ./chopinfer globs to bare
    # names, which subprocess would look up on PATH instead.
    for helper in sorted(Path(os.path.abspath(binary)).parent.glob('chopinfer-model-*')):
        if not helper.is_file() or not os.access(helper, os.X_OK):
            continue
        result = subprocess.run([str(helper), 'deployment-policy'], stdin=subprocess.DEVNULL,
                                capture_output=True)
        if result.returncode == 0 and result.stdout.strip():
            policies.append(policy(result.stdout, helper.name))
    return policies


def component(root, manifest, role, limit):
    """The bytes of the one component of `manifest` that serves `role`, read
    from the package at `root` without following a link and refused unless
    they have the declared length and SHA-256; None when none serves it."""
    declared = [entry for entry in manifest.get('components') or []
                if isinstance(entry, dict) and entry.get('role') == role]
    if not declared:
        return None
    if len(declared) > 1:
        raise ValueError(f'prepared package declares several {role} components')
    entry = declared[0]
    path, length, digest = entry.get('path'), entry.get('byte_length'), entry.get('sha256')
    parts = path.split('/') if isinstance(path, str) else []
    if not parts or any(part in ('', '.', '..') for part in parts) or type(length) is not int \
            or not 0 <= length <= limit or not isinstance(digest, str):
        raise ValueError(f'prepared package declares an invalid {role} component')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            inner = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = inner
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, 'rb') as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise ValueError(f'prepared package {role} component is not a regular file')
        data = file.read(limit + 1)
    if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'prepared package {role} component differs from its manifest')
    return data


def prepared(reply_path):
    """The bundle ID, drafter kind and deployment policy of the package a
    model helper prepared, from its reply at `reply_path` and the package's
    own files. Only data is read; package contents are never sourced. The
    drafter is the one the package declares."""
    reply = json.loads(Path(reply_path).read_text())
    if reply.get("result") != "prepared-direct":
        raise ValueError("model helper did not report a prepared package")
    bundle = reply.get("bundle_id")
    if not isinstance(bundle, str) or not re.fullmatch(BUNDLE_ID, bundle):
        raise ValueError("invalid deployment bundle ID")
    if not isinstance(reply.get("bundle"), str):
        raise ValueError("model helper reported no package directory")
    path = os.path.join(reply["bundle"], "manifest.json")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW)) as manifest:
        manifest = json.load(manifest)
    if manifest.get("id") != bundle:
        raise ValueError("prepared package manifest differs from the helper reply")
    speculation = manifest.get("speculation") or {}
    drafter = speculation.get("drafter_kind") if speculation.get("bundle_drafter") is True else None
    if drafter is not None and (not isinstance(drafter, str) or not re.fullmatch(DRAFTER_KIND, drafter)):
        raise ValueError("invalid package drafter kind")
    data = component(reply["bundle"], manifest, POLICY_ROLE, POLICY_MAX_BYTES)
    carried = None if data is None else policy(data, 'prepared package')
    if carried is not None and carried['bundle_id'] != bundle:
        raise ValueError(f'the prepared package deployment policy governs {carried["bundle_id"]}, '
                         f'not {bundle}')
    return bundle, drafter or '', carried


def refuse(name, reason):
    raise ValueError(f'CHOPINFER_{name}: {reason}')


def health_url(listener):
    try:
        host, port = listener.rsplit(':', 1)
        address = ipaddress.ip_address(host.strip('[]'))
        if address.version == 6 and not (host.startswith('[') and host.endswith(']')):
            raise ValueError('IPv6 addresses require brackets')
        if address.version == 4 and host != str(address):
            raise ValueError('IPv4 addresses must not use brackets')
        if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError('port must be from 1 through 65535')
    except ValueError as error:
        refuse('LISTEN', f'expected an IP socket address: {error}')
    if address.is_unspecified:
        address = ipaddress.ip_address('127.0.0.1' if address.version == 4 else '::1')
    host = str(address) if address.version == 4 else f'[{address}]'
    return f'http://{host}:{port}/health'


def validate(bundle=None, drafter=None, arguments=(), policies=(), package=None):
    """Validate the CHOPINFER_ environment and `arguments`: before
    preparation against the shared variables and `policies`, those of the
    installed models; with the prepared `bundle`, its `drafter` and the
    deployment policy its `package` carries (None when it carries none), also
    against that bundle."""
    if bundle and not re.fullmatch(BUNDLE_ID, bundle):
        raise ValueError('invalid bundle ID')
    if bundle and drafter and package is not None and drafter not in package['drafters']:
        raise ValueError(f'unsupported drafter {drafter} for bundle {bundle}')
    # A policy adds model variables; whatever it names, a shared variable
    # keeps its shared constraint.
    every = [*policies, *([package] if package is not None else [])]
    added = lambda kind: set().union(*(rules[kind] for rules in every)) - SHARED
    booleans, presence, taps = BOOLEANS | added('booleans'), PRESENCE | added('presence'), TAPS | added('taps')
    numbers, multiples = {}, {}
    for rules in every:
        numbers.update((name, tuple(bounds)) for name, bounds in rules['numbers'].items())
        multiples.update(rules['multiples'])
    numbers = {**numbers, **NUMBERS}
    multiples = {name: multiple for name, multiple in multiples.items() if name not in SHARED}
    known = KNOWN | added('paths') | booleans | presence | numbers.keys()
    prefixes = tuple(package['knob_prefixes']) if bundle and package is not None else ()
    model_prefixes = tuple(sorted({prefix for rules in every for prefix in rules['knob_prefixes']})) \
        if bundle else ()
    drafter_only = {name: kinds for name, kinds in package['drafter_only'].items() if name not in SHARED} \
        if bundle and package is not None else {}
    env = {k.removeprefix('CHOPINFER_'): v for k, v in os.environ.items()
           if k.startswith('CHOPINFER_')}
    for name, value in env.items():
        if name in GONE:
            raise ValueError(f'CHOPINFER_{name} is gone: {GONE[name]}')
        if name in taps:
            refuse(name, 'per-layer taps need a bundle built with its diagnostic-taps feature; '
                         'this image serves, so unset it and capture with a diagnostic build')
        if name not in known:
            refuse(name, 'unknown deployment variable')
        if not value and name != 'SERVE_ARGS':
            refuse(name, 'must not be empty')
        if name == 'SERVE_ARGS' and value and shlex.split(value) != list(arguments):
            refuse(name, 'must be expanded into launcher arguments; Docker users pass arguments after the image')
        if name in presence and value != '1':
            refuse(name, 'expected 1 to enable; unset the variable to disable')
        if name in booleans and value not in {'0', '1'}:
            refuse(name, 'expected 0 or 1')
        if name in numbers:
            low, high = numbers[name]
            if not re.fullmatch('[0-9]+', value) or not low <= int(value) <= high:
                refuse(name, f'expected an integer from {low} through {high}')
        if name in multiples and int(value) % multiples[name]:
            refuse(name, f'must be a multiple of {multiples[name]}')
        if name in CHOICES and value not in CHOICES[name]:
            refuse(name, 'expected ' + ', '.join(sorted(CHOICES[name])))
        if name == 'MODEL_OPTIONS':
            entries = value.split(',')
            names = [entry.partition('=')[0] for entry in entries]
            if len(set(names)) != len(names) or any(
                    not re.fullmatch(r'[A-Za-z0-9_.]{1,128}=.{0,1024}', entry)
                    for entry in entries):
                refuse(name, 'expected unique name=value entries separated by commas')
        if name == 'EXPECTED_BUNDLE_ID' and not re.fullmatch(r'[a-z0-9][a-z0-9.-]*', value):
            refuse(name, 'expected a bundle ID')
        if name == 'MODALITIES' and value not in ('auto', 'all'):
            names = value.split(',')
            if any(n not in MODALITIES for n in names) or len(set(names)) != len(names) \
                    or 'text' not in names:
                refuse(name, 'expected auto, all or a comma list of ' + ', '.join(MODALITIES)
                       + ' that includes text, each once')
        if name == 'LISTEN':
            health_url(value)
        if name == 'KV_MEMORY_FRACTION':
            try:
                valid = math.isfinite(float(value)) and 0 < float(value) <= 1
            except ValueError:
                valid = False
            if not valid:
                refuse(name, 'expected a fraction greater than 0 and at most 1')
        if bundle:
            if name not in SHARED and name.startswith(model_prefixes) and not name.startswith(prefixes):
                refuse(name, f'does not apply to bundle {bundle}')
            if name in SPECULATIVE or name in drafter_only:
                if not drafter or name in drafter_only and drafter not in drafter_only[name]:
                    refuse(name, f'does not apply to the selected drafter ({drafter or "none"})')
    # The encoder capture variables are declared by the model whose policy
    # owns the ENCODER_ prefix; capture of all layers needs a directory.
    for dependent, enabled in [('SAMPLING_CAPTURE_ROWS', bool(env.get('SAMPLING_CAPTURE'))),
                               ('STEP_PROFILE_CAPACITY', env.get('STEP_PROFILE_EVERY', '0') != '0'),
                               ('ENCODER_CAPTURE_ALL_LAYERS', bool(env.get('ENCODER_CAPTURE_DIR')))]:
        if dependent in env and not enabled:
            refuse(dependent, 'its capture or profiling facility is disabled')
    if 'VERIFY_MANIFEST' in env and env.get('VERIFY_MODEL') != '1':
        refuse('VERIFY_MANIFEST', 'applies only with CHOPINFER_VERIFY_MODEL=1')
    if bundle:
        expected = env.get('EXPECTED_BUNDLE_ID')
        if expected and expected != bundle:
            refuse('EXPECTED_BUNDLE_ID', f'expected {expected}, the checkpoint prepares {bundle}')
        tokens = env.get('SPECULATIVE_TOKENS', '')
        if '--speculative-tokens' in arguments:
            index = arguments.index('--speculative-tokens') + 1
            tokens = arguments[index] if index < len(arguments) else '0'
        if re.fullmatch('0+', tokens):
            for name in ('SPECULATION', 'DRAFT_COST'):
                if name in env:
                    refuse(name, 'does not apply when speculation is disabled')
        for rule in package['disabled_by'] if package is not None else ():
            if env.get(rule['variable']) == rule['value']:
                for name in env:
                    if name not in SHARED and name.startswith(rule['prefix']):
                        refuse(name, f'does not apply when {rule["reason"]}')


if __name__ == '__main__':
    # environment BINARY ARGS: before preparation, with the policies of the
    # helpers installed beside BINARY. prepared BINARY REPLY ARGS: the package
    # a helper prepared, as its reply at REPLY names it.
    try:
        if sys.argv[1:2] == ['environment'] and len(sys.argv) > 2:
            validate(arguments=sys.argv[3:], policies=installed(sys.argv[2]))
            sys.exit(0)
        if sys.argv[1:] == ['healthcheck']:
            url = health_url(os.environ.get('CHOPINFER_LISTEN', '0.0.0.0:8000'))
            os.execvp('curl', ['curl', '--noproxy', '*', '-fsS', '--max-time', '2', url])
        if sys.argv[1:2] == ['prepared'] and len(sys.argv) > 3:
            bundle, drafter, package = prepared(sys.argv[3])
            validate(bundle, drafter, sys.argv[4:], policies=installed(sys.argv[2]), package=package)
            sys.exit(0)
        sys.exit('usage: deployment.py environment BINARY [ARGS...] | prepared BINARY REPLY [ARGS...] '
                 '| healthcheck')
    except (OSError, ValueError) as error:
        sys.exit(str(error))
