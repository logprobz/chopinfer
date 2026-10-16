"""A model's serving facts, read from models/<id>/model.toml and profiles.json."""
import json
from pathlib import Path
import tomllib

from common import need, read, regular, digest

MODELS = Path(__file__).resolve().parents[3] / 'models'


def facts(model, models=MODELS):
    """The install files, bundle ID and drafters a model declares.

    An install holds bin/chopinfer, the model's compiler bin/chopinfer-model-ID
    and its one bundle cdylib. Every serve profile names the model's single
    bundle ID; the drafters are the profiles' drafters other than "none"."""
    need(type(model) is str and model and '/' not in model and not model.startswith('.'), 'invalid model id')
    root = Path(models) / model
    toml_path = regular(root / 'model.toml')
    with toml_path.open('rb') as stream:
        inventory = tomllib.load(stream)
    compiler = f'chopinfer-model-{model}'
    need(compiler in inventory.get('compiler', {}).get('bin', []), f'{model}: model.toml declares no {compiler} bin')
    cdylibs = inventory.get('bundle', {}).get('cdylib', [])
    need(type(cdylibs) is list and len(cdylibs) == 1 and type(cdylibs[0]) is str, f'{model}: expected one bundle cdylib')
    profiles_path = regular(root / 'profiles.json')
    profiles = read(profiles_path)
    need(profiles.get('model') == model and type(profiles.get('profiles')) is dict and profiles['profiles'],
         f'{model}: profiles.json names another model or no profiles')
    bundle_ids = {p.get('bundle_id') for p in profiles['profiles'].values()}
    need(len(bundle_ids) == 1 and type(next(iter(bundle_ids))) is str, f'{model}: profiles must share one bundle ID')
    drafters = sorted({p.get('speculation', {}).get('drafter') for p in profiles['profiles'].values()} - {'none', None})
    return dict(model=model, bundle_id=next(iter(bundle_ids)), drafters=drafters,
                server='bin/chopinfer', compiler='bin/' + compiler, bundle=f'lib/lib{cdylibs[0]}.so',
                profiles=str(profiles_path), profile_names=sorted(profiles['profiles']),
                model_toml_sha256=digest(toml_path), profiles_sha256=digest(profiles_path))


# A serve profile's options as the host expands one (Profile::arguments in
# crates/runtime-host/src/profiles.rs), less selection (bundle, topology,
# devices, served name) and speculation, which a variant sets. `serve --model`
# refuses --profile (it applies the family's own profile under explicit
# options), so the runner passes a profile's options explicitly.
OPTIONS = [('tensor_parallel', '--tensor-parallel-size'), ('max_sequences', '--max-sequences'),
           ('max_batch_tokens', '--max-batch-tokens'), ('prefill_chunk_tokens', '--prefill-chunk-tokens'),
           ('kv_memory_fraction', '--kv-memory-fraction'), ('kv_cache_dtype', '--kv-cache-dtype'),
           ('prefix_cache', '--prefix-cache'), ('queue_depth', '--queue-depth'),
           ('context_parallel_size', '--context-parallel-size'), ('prefill_budget', '--prefill-budget'),
           ('decode_tbt_target_ms', '--decode-tbt-target-ms')]
DESCRIPTIVE = {'bundle_id', 'topology', 'devices_required', 'devices', 'speculation', 'model_options',
               'qualified_max_sequences', 'qualification', 'status', 'evidence'}


def option_text(value):
    """A model option value as the host writes it: a string as itself, any
    other value as its JSON text."""
    return value if type(value) is str else json.dumps(value, separators=(',', ':'))


def serving(name, profile):
    """(serve arguments, /health `serving` subset) of a serve profile. The
    expectation holds what /health reports: the tensor-parallel size, the
    sequence limit, the prefix cache, an explicit KV dtype, a declared
    context-parallel size (any value, 1 included) and the model options."""
    need(type(profile) is dict and type(profile.get('tensor_parallel')) is int, f'profile {name}: no tensor_parallel')
    unknown = profile.keys() - DESCRIPTIVE - {key for key, _ in OPTIONS}
    need(not unknown, f'profile {name}: options the runner does not apply: {sorted(unknown)}')
    args = []
    for key, flag in OPTIONS:
        value = profile.get(key)
        if value is None:
            continue
        if key == 'prefix_cache':
            need(type(value) is bool, f'profile {name}: prefix_cache must be true or false')
            value = 'on' if value else 'off'
        need(type(value) in (str, int, float), f'profile {name}: {key} must be a scalar')
        args += [flag, str(value)]
    speculation = profile.get('speculation') or {}
    if speculation.get('draft_cost') is not None:
        args += ['--draft-cost', speculation['draft_cost']]
    options = profile.get('model_options') or {}
    need(type(options) is dict, f'profile {name}: model_options must be an object')
    for option, value in sorted(options.items()):
        args += ['--model-option', f'{option}={option_text(value)}']
    expected = {'tensor_parallel_size': profile['tensor_parallel']}
    for key in ('max_sequences', 'prefix_cache', 'context_parallel_size'):
        if profile.get(key) is not None:
            expected[key] = profile[key]
    # "auto" resolves to the package's dtype.
    if profile.get('kv_cache_dtype') not in (None, 'auto'):
        expected['kv_cache_dtype'] = profile['kv_cache_dtype']
    expect = {'profile': expected}
    if options:
        expect['model_options'] = dict(options)
    return args, expect


def serves(manifest, profile):
    """Whether a prepared package (its manifest) serves a profile: the
    profile's topology at its rank count, its context-parallel size and its
    model option values are ones the package declares."""
    topologies = [t for t in manifest.get('target', {}).get('topologies', [])
                  if t.get('id') == profile.get('topology') and t.get('rank_count') == profile['tensor_parallel']]
    if not topologies:
        return False
    cp = profile.get('context_parallel_size')
    if cp is not None and cp > 1 and cp not in topologies[0].get('context_parallel_sizes', []):
        return False
    declared = {o.get('name'): o for o in manifest.get('model_options', {}).get('options', [])}
    for option, value in (profile.get('model_options') or {}).items():
        allowed = declared.get(option, {}).get('definition', {}).get('allowed_values')
        if option not in declared or (allowed is not None and value not in allowed):
            return False
    return True
