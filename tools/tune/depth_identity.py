"""Portable capture bindings and resolved depth-sweep adoption targets."""
import hashlib
import json
import os
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sealed_capture(seal):
    """Retain content identities from the verified seal, without local paths."""
    import tune
    installs = {}
    for name, install in tune.sealed_installs(seal)[0].items():
        roots = {'install': Path(install['install']),
                 'package': Path(install.get('prepared', {}).get('bundle', '.'))}
        files = {}
        for role, root in roots.items():
            files[role] = {str(Path(path).relative_to(root)): sha for path, sha in seal['sha256'].items()
                           if root in Path(path).parents}
        package = package_identity(install) if seal.get('ready') else {}
        installs[name] = {'revision': install['revision'], 'artifacts': files, **package}
    return {'schema': 'chopinfer-depth-capture.v1', 'config_sha256': digest(seal['config']),
            'prompts_sha256': digest(seal['prompts']), 'installs': installs}


def package_identity(install):
    """Read the selected frozen package's immutable checkpoint binding."""
    bundle = Path(install['prepared']['bundle'])
    import faults
    with os.fdopen(faults.open_regular(bundle / 'manifest.json'), 'rb') as stream:
        body = stream.read()
    manifest = json.loads(body)
    identity = {'bundle_id': manifest.get('id'), 'package_identity': hashlib.sha256(body).hexdigest()}
    record = Path(install['cache']) / 'records' / f'{bundle.name}.json'
    if record.is_file():
        with os.fdopen(faults.open_regular(record), 'rb') as stream:
            entry = json.load(stream)
        value = entry['value']
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        if entry.get('version') != 1 or hashlib.sha256(encoded).hexdigest() != entry.get('checksum'):
            raise ValueError('selected package cache record checksum differs')
        if value.get('state') != 'complete':
            raise ValueError('selected package is incomplete')
        identity['checkpoint_identity'] = value['payload'].get('checkpoint')
    else:
        identity['checkpoint_identity'] = install['prepared'].get('checkpoint_identity')
    return identity


def normalized(check):
    if (check.get('schema') != 'chopinfer.serve-check.v1' or not check.get('profile')
            or not isinstance(check.get('settings'), dict) or not check['settings']
            or not isinstance(check.get('model_options'), list)):
        raise ValueError('unresolved serving settings')
    settings = {k: v for k, v in check['settings'].items() if k != 'speculation-depths'}
    if (any(not isinstance(k, str) or not isinstance(v, str) for k, v in settings.items())
            or not {'speculation', 'speculative-tokens', 'speculative-rows', 'max-sequences',
                    'prefill-policy', 'speculation-confidence', 'speculation-widths'} <= set(settings)):
        raise ValueError('incomplete resolved depth-sweep settings')
    options = {entry['name']: entry['value'] for entry in check['model_options']}
    if len(options) != len(check['model_options']):
        raise ValueError('duplicate resolved model option')
    return {'profile': check['profile'], 'settings': settings, 'model_options': options}


def launch_arguments(argv):
    """Remove only launcher details that cannot change model execution."""
    ignored = {'--listen', '--weight-keeper', '--weight-keeper-policy', '--weight-keeper-identity'}
    result, index = [], 0
    while index < len(argv):
        word = argv[index]
        if word.split('=', 1)[0] in ignored:
            index += 1 if '=' in word else 2
        elif word in ('--check', '--list-model-options'):
            index += 1
        else:
            result.append(word)
            index += 1
    return result


def check_health(live, install, resolved, model):
    for key in ('package_identity', 'checkpoint_identity'):
        if not re.fullmatch('[0-9a-f]{64}', install.get(key) or '') or live.get(key) != install[key]:
            raise ValueError(f'live {key} differs from the frozen capture')
    if live.get('bundle_id') != install['bundle_id'] or live.get('source_commit') != install['revision']:
        raise ValueError('live bundle or revision differs from the frozen capture')
    if not re.fullmatch('[0-9a-f]{64}', live.get('metadata_identity') or ''):
        raise ValueError('missing live checkpoint metadata identity')
    settings, profile = resolved['settings'], live.get('profile', {})
    expected = {'name': resolved['profile'], 'tensor_parallel_size': model['tensor_parallel_size'],
                'max_sequences': int(settings['max-sequences']), 'prefix_cache': settings['prefix-cache'] == 'on'}
    if settings['kv-cache-dtype'] != 'auto':
        expected['kv_cache_dtype'] = settings['kv-cache-dtype']
    if profile.get('context_parallel_size') is not None or int(settings.get('context-parallel-size', 1)) != 1:
        expected['context_parallel_size'] = int(settings.get('context-parallel-size', 1))
    if any(profile.get(k) != v for k, v in expected.items()):
        raise ValueError('live serving profile differs from preflight')
    spec = profile.get('speculation', {})
    if spec.get('mode') != settings['speculation'] or spec.get('tokens') != int(settings['speculative-tokens']):
        raise ValueError('live speculation differs from preflight')
    actual_options = {k: str(v).lower() if isinstance(v, bool) else str(v)
                      for k, v in live.get('model_options', {}).items()}
    if actual_options != resolved['model_options']:
        raise ValueError('live model controls differ from preflight')


def validate(run, config, rows, summary):
    import tune
    capture = summary.get('capture_identity', {})
    prompts = json.loads((Path(run) / 'prompts.json').read_text())
    if (capture.get('schema') != 'chopinfer-depth-capture.v1'
            or capture.get('config_sha256') != digest(config) or capture.get('prompts_sha256') != digest(prompts)):
        raise ValueError('config or prompt binding differs from the sealed capture')
    for prompt in prompts:
        content = {k: prompt[k] for k in ('prompt', 'messages') if k in prompt}
        if (len(content) != 1 or prompt.get('sha256') != hashlib.sha256(
                json.dumps(content, sort_keys=True).encode()).hexdigest()):
            raise ValueError('prompt content digest differs')
    installs = capture.get('installs', {})
    for name, install in installs.items():
        artifacts = install.get('artifacts', {})
        if (not artifacts.get('install', {}).get('bin/chopinfer')
                or artifacts.get('package', {}).get('manifest.json') != install.get('package_identity')
                or not install.get('bundle_id') or not re.fullmatch('[0-9a-f]{40}', install.get('revision') or '')
                or any(not re.fullmatch('[0-9a-f]{64}', sha) for files in artifacts.values() for sha in files.values())):
            raise ValueError(f'{name}: incomplete frozen artifact identities')
    preflight = summary.get('depth_preflight', {})
    if preflight.get('capture_sha256') != digest(capture):
        raise ValueError('preflight is not bound to the capture')
    target = config['model'].get('adoption_target')
    if not target or preflight.get('target_declaration') != target:
        raise ValueError('missing sealed adoption target')
    target_settings = normalized(preflight['target']['resolved'])
    if target_settings['profile'] != target['profile']:
        raise ValueError('resolved target profile differs from the declaration')
    cases = {case['arm']: case for case in preflight.get('cases', [])}
    arms = tune.arms(config)
    if len(cases) != len(preflight.get('cases', [])) or set(cases) != {arm['label'] for arm in arms}:
        raise ValueError('missing or duplicate preflight arms')
    live_common, artifact_common = None, None
    for visit in tune.visits(config):
        case, process = cases[visit['arm']], summary['processes'][visit['label']]
        resolved = normalized(case['resolved'])
        if resolved != target_settings or case.get('env', {}) != target.get('env', {}):
            raise ValueError(f"{visit['arm']}: serving settings differ from the adoption target")
        if case['resolved']['settings'].get('speculation-depths') != f"1:{visit['depth']}":
            raise ValueError('preflight depth differs from the declared sweep')
        if (process.get('capture_sha256') != digest(capture)
                or launch_arguments(process.get('serve_argv', [])) != launch_arguments(case['argv'])
                or process.get('env', {}) != case.get('env', {})):
            raise ValueError('actual launch differs from preflight')
        install = installs[process['install']]
        if process['revision'] != install['revision']:
            raise ValueError('process revision differs from sealed content')
        check_health(process.get('serving', {}), install, resolved, config['model'])
        live = {k: process['serving'][k] for k in ('package_identity', 'checkpoint_identity', 'metadata_identity', 'profile')}
        if live_common is not None and live != live_common or artifact_common is not None and install != artifact_common:
            raise ValueError('sweep processes use different package or checkpoint content')
        live_common, artifact_common = live, install
    for row in rows:
        count = row['concurrency'] * row['groups']
        recorded = summary['processes'][row['label']]['windows'][row['_file']]
        if (row.get('capture_sha256') != digest(capture)
                or row.get('prompts') != [p['sha256'] for p in prompts[:count]] or count > len(prompts)
                or any(recorded.get(key) != row.get(key) for key in ('install', 'revision'))):
            raise ValueError('window prompt or capture identity differs')
    # Planning limits are retained below, but never stand in for resolved capacity.
    model = config['model']
    common = {'model': model['model'], 'topology': model['tensor_parallel_size'],
              'workload': config['workload'], 'prompts_sha256': capture['prompts_sha256'],
              'sampling': config['sampling'], 'kinds': config['kinds'], 'tokens': config['tokens'],
              'selection': config['selection'], 'install': artifact_common, 'live': live_common,
              'target': target_settings, 'environment': target.get('env', {})}
    return {**describe(config, summary), 'common': common}


def describe(config, summary):
    import tune
    model = config['model']
    preflight = summary.get('depth_preflight', {})
    cases = {case['arm']: case for case in preflight.get('cases', [])}
    resolved = {}
    for visit in tune.visits(config):
        for _, windows in visit['passes']:
            for _, c in windows:
                resolved.setdefault(str(c), {})[visit['arm']] = cases.get(visit['arm'], {}).get('resolved')
    return {'concurrencies': model['concurrencies'], 'verify_rows': model['verify_rows'],
            'extra_arguments': model.get('extra_arguments', []), 'serve_arguments': model.get('serve_arguments', []),
            'environment': {arm['label']: arm.get('env', {}) for arm in tune.arms(config)},
            'adoption_target': model.get('adoption_target'), 'resolved_by_concurrency': resolved}


def partition_text(partition):
    return ('\n## Capture and serving target\n\n'
            + 'Declared planning verify rows: ' + str(partition['verify_rows']) + '.\n\n'
            + 'Launch arguments: `' + json.dumps(partition['serve_arguments'] + partition['extra_arguments']) + '`.\n\n'
            + 'Environment: `' + json.dumps(partition['environment'], sort_keys=True) + '`.\n\n'
            + 'Resolved settings by concurrency:\n\n```json\n'
            + json.dumps(partition['resolved_by_concurrency'], indent=2, sort_keys=True) + '\n```\n')
