"""Independent package admission and release selection evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from .source import identity, sha, tree
from .workflow import require, run, http_check


def write_manifest(root, value):
    (root / 'manifest.json').write_text(json.dumps(value, sort_keys=True, separators=(',', ':')))


def derive(source, destination):
    shutil.copytree(source, destination)
    return json.loads((destination / 'manifest.json').read_text())


def unavailable_operation(manifest):
    # A valid exact-copy shape with deliberately unsupported numerics. Package
    # validation accepts it; host operation admission must refuse it.
    key = {'operation':'chop.generic.copy-f32','variant':'v1'}
    def tensor(role, access, dependencies):
        return {'access':access,'alias':{'kind':'disjoint'},'alignment_bytes':256,
                'dependencies':dependencies,'dimensions':[64],'dtype':'chop.dtype.f32.v1',
                'layout':'chop.layout.dense-row-major.v1','role':role,
                'scale_block_dimensions':[],'scale_format':'chop.scale.none.v1'}
    manifest['operation_requirements'] = [{'key':key,
        'provisioning':{'kind':'host-generic-primitive','primitive_id':'chop.generic.copy-f32.v1'},
        'semantic_contract':{'numerical_contract':'example.unsupported-numerics.v1',
            'operation_abi':'chop.generic.copy-f32','operation_abi_version':1,
            'required_capabilities':[],
            'scalar_arguments':[{'argument':'element-count','dtype':'chop.dtype.u32.v1','width_bits':32}],
            'tensors':[tensor('source','read-only',[]), tensor('destination','write-only',
                        [{'tensor_index':0,'dependency':'source'}])]}}]
    manifest['graph_buckets'][0]['operations'] = [key]


def refusals(host, package, omitted_package, output):
    host, package, output = map(Path, (host, package, output))
    output.mkdir(parents=True)
    before = sha(host)
    rows = []
    cases = [('schema','unsupported bundle schema'), ('abi','unsupported bundle ABI version'),
             ('capability','requires unsupported host capability'), ('architecture','architecture'),
             ('topology','topology'), ('operation',"this host cannot run the selected bundle's operation"),
             ('corruption','SHA-256'), ('extra','file set differs')]
    for case, error in cases:
        destination = output / case
        manifest = derive(package, destination)
        if case == 'schema':
            manifest['schema'] = 'chop-model-bundle-v2'
        elif case == 'abi':
            manifest['abi_version'] = 2
        elif case == 'capability':
            caps = manifest['resource_requirements']['required_capabilities']
            caps.append('example.unavailable.v1')
            caps.sort()
        elif case == 'architecture':
            manifest['target']['architectures'] = ['sm_90']
        elif case == 'topology':
            manifest['target']['topologies'][0]['id'] = 'other'
        elif case == 'operation':
            unavailable_operation(manifest)
        elif case == 'corruption':
            (destination / 'checkpoint.bin').write_bytes(b'corrupt!')
        elif case == 'extra':
            (destination / 'extra.txt').write_text('not declared')
        # An authenticated but unloadable module distinguishes pre-entry
        # admission from a refusal reached after native code has run.
        if case in {'schema','abi','capability','architecture'}:
            native = destination / manifest['native_module']['path']
            native.write_bytes(b'not a native module\n')
            manifest['native_module'].update(byte_length=native.stat().st_size, sha256=sha(native))
        write_manifest(destination, manifest)
        result = subprocess.run([str(host), '--bundle', str(destination), '--fixture-request',
                                 'external-bundle-host', '--topology','tp1','--fake-host-services'],
                                text=True, capture_output=True)
        (output / f'{case}.log').write_text(result.stdout + result.stderr)
        require(result.returncode != 0 and error.lower() in result.stderr.lower(),
                f'{case} did not fail at its admission boundary: {result.stderr}')
        boundary = 'before instance creation' if case == 'topology' else 'before native entry'
        marker = 'startup bundle create:' if case == 'topology' else 'startup module load:'
        require(marker not in result.stderr, f'{case} passed its admission boundary')
        rows.append({'case':case, 'boundary':boundary, 'error':result.stderr.strip()})
    missing = output / 'missing-callback'
    manifest = derive(omitted_package, missing)
    marker = output / 'created'
    env = dict(os.environ, EXTERNAL_MODEL_CREATE_MARKER=str(marker))
    command = [str(host),'--bundle',str(missing),'--fixture-request','external-bundle-host',
               '--topology','tp1','--fake-host-services']
    serving = [str(host),'serve','--bundle',str(missing),'--topology','tp1',
               '--fake-host-services','--max-sequences','4','--kv-pages','512',
               '--max-context-tokens','1024','--prefill-chunk-tokens','512','--listen','127.0.0.1:0']
    result = subprocess.run(serving, text=True, capture_output=True, env=env, timeout=15)
    (output/'missing-callback.log').write_text(result.stdout + result.stderr)
    require(result.returncode != 0 and 'retire' in result.stderr.lower(),
            f'missing callback was not refused: {result.stderr}')
    require(not marker.exists(), 'callback incompatibility reached instance creation')
    rows.append({'case':'missing-callback','boundary':'before instance creation','error':result.stderr.strip()})
    manifest['resource_requirements']['required_capabilities'].remove('chop.request-retirement.v1')
    write_manifest(missing, manifest)
    accepted = json.loads(run(command, env=env, log=output/'optional-omission.log'))
    require(marker.exists() and accepted['live_resources'] == 0 and accepted['status'] == 'committed',
            'omitting an unused optional callback should remain supported')
    require(sha(host) == before, 'prebuilt host changed during refusal checks')
    (output/'refusals.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def releases(host, first, second, output, *, first_seed=7, second_seed=8):
    host, first, second, output = map(Path, (host, first, second, output))
    output.mkdir(parents=True)
    before = {str(p):identity(tree(p)) for p in [first, second]}
    host_hash = sha(host)
    bundle = json.loads((first/'manifest.json').read_text())['id']
    require(json.loads((second/'manifest.json').read_text())['id'] == bundle, 'release IDs differ')
    # A release is its package directory: selecting one serves it, and
    # rolling back serves the earlier directory again.
    selections = []
    for label, package, seed in [('first',first,first_seed),('updated',second,second_seed),('rollback',first,first_seed)]:
        selections.append(http_check(host, package, output/f'{label}.log', seed))
    require(selections[0] == selections[2], 'rollback did not restore the earlier release')
    require(selections[0] != selections[1], 'new checkpoint did not change the model response')
    require({p:identity(tree(Path(p))) for p in before} == before, 'a retained package changed')
    require(sha(host) == host_hash, 'release selection changed the host')
    receipt = {'bundle':bundle,'host_sha256':host_hash,'packages':before,'selections':selections}
    (output/'releases.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
