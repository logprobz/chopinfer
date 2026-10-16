"""Build external roles and exercise canonical packages with prebuilt hosts."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request

from . import inventory
from .graph import PolicyError, metadata, selected_graph
from .source import check, sha, tree, identity, verify_source

TOOLCHAIN_IMAGE = 'chopinfer-sdk-toolchain@sha256:d1d3f837e6c1f0015ceba694717268235f9fc6884238957d12762f6e0bac1b58'
# The relock candidate directory is set only by `relock` itself.
RELOCK_ENV = 'CHOPINFER_MODEL_RELOCK_DIR'
BYPASSES = ('KERNELS_LOCK_UPDATE', 'CHOPINFER_KERNEL_LINEINFO', RELOCK_ENV,
            'EXTERNAL_MODEL_CREATE_KERNEL_LOCK', 'EXTERNAL_MODEL_CREATE_PRECISION_LOCK')
KERNEL_BUILD_REPORT = 'kernel-build-report.json'
TARGET_OWNER = '.chopinfer-sdk-workspace'
# The lock file of a held workspace or target directory.
SDK_LOCK = '.chopinfer-sdk-lock'


def require(condition, message):
    if not condition:
        raise PolicyError(message)


def environment():
    for key in BYPASSES:
        require(os.environ.get(key, '') in ('', '0'), f'{key} is forbidden in the supported workflow')
    version = subprocess.check_output(['rustc', '--version'], text=True).strip()
    require(version.startswith('rustc 1.89.0 '), 'the supported SDK requires Rust 1.89.0')
    result = dict(os.environ, CARGO_NET_OFFLINE='true')
    return result


def cargo_config(relock_dir=''):
    """Cargo CLI configuration forcing every bypass empty for build scripts.

    The model's own Cargo configuration can set `[env]` for build scripts, and
    CLI configuration overrides it key by key, so each forced variable also
    sets `relative = false`: otherwise a configured `relative = true` would
    resolve the empty value to the workspace directory. Only `relock` passes
    its candidate directory."""
    config = []
    for key in BYPASSES:
        value = json.dumps(str(relock_dir) if key == RELOCK_ENV else '')
        # Cargo accepts dotted keys, not inline tables, on the command line.
        for field in (f'value={value}', 'force=true', 'relative=false'):
            config += ['--config', f'env.{key}.{field}']
    return config


def cargo_messages(output):
    """The JSON messages in a cargo command's output, skipping other lines."""
    for line in output.splitlines():
        if line.startswith('{'):
            try:
                yield json.loads(line)
            except ValueError:
                continue


def build_script_outputs(output):
    """The out directory of every build script a cargo command used, as its
    JSON messages report them, including scripts reused from an earlier build."""
    return [Path(m['out_dir']) for m in cargo_messages(output) if m.get('reason') == 'build-script-executed']


def role_packages(root):
    """Every declared role's package, for `--package` selection: workspace
    default members can never drop a role from a build."""
    return sorted({item.name for item in inventory.load_model(root, root).packages.values()})


def package_selection(packages):
    return [arg for package in packages for arg in ('--package', package)]


def produced_artifacts(output, root, model):
    """Each role's bin and cdylib from this cargo invocation's compiler-artifact
    messages, fresh or rebuilt. A file an earlier build left is not evidence."""
    produced = {}
    for message in cargo_messages(output):
        manifest = Path(message.get('manifest_path', '/')).resolve()
        if message.get('reason') != 'compiler-artifact' or not manifest.is_relative_to(root) \
                or manifest.is_relative_to(root / 'sdk'):
            continue
        name, kinds = message['target']['name'], message['target']['kind']
        if 'bin' in kinds and message.get('executable'):
            produced[(name, 'bin')] = message['executable']
        if 'cdylib' in kinds:
            for filename in message['filenames']:
                if Path(filename).name == f'lib{name}.so':
                    produced[(name, 'cdylib')] = filename
    artifacts = {}
    for role, item in model.packages.items():
        for name, kind in item.targets:
            path = produced.get((name, kind))
            require(path and Path(path).is_file(), f'this build produced no {role} {kind} {name}')
            artifacts[name] = {'path': str(path), 'sha256': sha(path)}
    return artifacts


def require_verified(directories, workspace):
    """Every kernel build report in these build script outputs must show a
    verified lock of this workspace, recorded with the lock's current digest.

    This backs the bypass refusal however a build script was configured, and
    covers reports a cached build script left from an earlier build: releases
    require each lock `Verified`, by this workspace's build, as it is now."""
    workspace = Path(workspace).resolve()
    for directory in directories:
        for path in sorted(Path(directory).rglob(KERNEL_BUILD_REPORT)):
            report = json.loads(path.read_text())
            status = report.get('lock_status')
            require(status == 'Verified', f'{path}: kernel lock {status}, not Verified; '
                    'an SDK build only verifies locks (create or update model locks with `relock`)')
            locks = [(f['lock_path'], f['lock_sha256']) for f in report.get('lock_families', [])]
            for lock, digest in locks or [(report['lock_path'], report['lock_sha256'])]:
                lock = Path(lock)
                require(lock.is_relative_to(workspace),
                        f'{path}: verified {lock}, outside this workspace; use one target directory per workspace')
                require(lock.is_file() and digest == sha(lock), f'{path}: {lock} changed since it was verified')


def verified_cargo(command, workspace, *, env, log):
    """Run a cargo command in `workspace` whose build scripts, fresh or reused,
    all verified this workspace's current locks."""
    output = run([*command, '--message-format=json-render-diagnostics'], env=env, cwd=workspace, log=log)
    require_verified(build_script_outputs(output), workspace)
    return output


def claim_target(target, workspace):
    """Dedicate a target directory to one workspace. Cargo keys a path
    package's build script output independently of its location, so exports
    sharing a target would reuse each other's outputs, lock reports included."""
    target.mkdir(parents=True, exist_ok=True)
    owner = target / TARGET_OWNER
    # Link a complete claim into place: exactly one of racing workspaces
    # creates it, and each then validates the owner it reads.
    temporary = target / f'.{TARGET_OWNER}.{secrets.token_hex(8)}'
    temporary.write_text(str(workspace))
    try:
        os.link(temporary, owner)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()
    claimed = owner.read_text()
    require(claimed == str(workspace),
            f'{target} is the target directory of {claimed}; use one target directory per workspace')


@contextmanager
def exclusive(directory, alternative=''):
    """Hold `directory` exclusively through a lock file in it; a second holder
    is refused at once, naming the first one's process ID."""
    descriptor = os.open(Path(directory) / SDK_LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(descriptor, 64, 0).decode(errors='replace').strip() or 'unknown'
            raise PolicyError(f'{directory} is in use by another SDK build or relock (pid {holder}); '
                              f'wait for it to finish{alternative}') from None
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, str(os.getpid()).encode(), 0)
        yield
    finally:
        os.close(descriptor)


def workspace_lock(workspace):
    """Hold the model workspace for a whole build or relock: runs with
    different target directories would otherwise write its locks together."""
    return exclusive(workspace)


def target_lock(target):
    """Hold the target directory for a whole build or relock: concurrent
    invocations on one target would otherwise rebuild each other's artifacts
    between a build and its receipt."""
    return exclusive(target, ' or use another target directory')


def external_output(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    # An output beside the SDK inside the model checkout would become release
    # source on its next export. Keep all generated material outside both trees.
    sdk_tools = Path(__file__).resolve().parents[2]
    for source in (workspace, sdk_tools):
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                'generated output must use an unrelated external directory')
    return output


def run(command, *, env=None, cwd=None, log=None):
    result = subprocess.run([str(a) for a in command], env=env, cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if log:
        Path(log).write_text(result.stdout + result.stderr)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        if 'kernels.lock' in message or ('lock' in message.lower() and 'mismatch' in message.lower()):
            message = message.replace('; rerun with KERNELS_LOCK_UPDATE=1 to accept', '')
            message += (f'\nPinned kernel inputs or the toolchain differ. Use {TOOLCHAIN_IMAGE} '
                        'or investigate the source difference; do not rewrite SDK locks. '
                        'Review a model-owned lock change with `relock`.')
        raise PolicyError(f'{Path(command[0]).name} failed ({result.returncode}): {message}')
    return result.stdout


def native_toolchain(graph, roots, env):
    owners = graph.cuda_build_owners(roots)
    result = {'cuda_build_owners': owners, 'nvcc': None}
    if owners:
        nvcc = Path(env.get('CHOPINFER_CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc'
        require(nvcc.is_file(), f'CUDA 13.2.86 is required by {", ".join(owners)}; '
                'set CHOPINFER_CUDA_HOME to the installed toolchain root')
        version = run([nvcc, '--version']).strip()
        require('V13.2.86' in version, 'the selected native build requires nvcc 13.2.86')
        result['nvcc'] = {'version': version, 'sha256': sha(nvcc)}
    return result


def build(workspace, target, logs, *, features=(), release_pin=False, contributor_roots=()):
    root = Path(workspace).resolve()
    target, logs = (external_output(root, p) for p in (target, logs))
    env = environment()
    # Held for the whole run, before the target: one run at a time per workspace.
    with workspace_lock(root):
        # Provenance follows the actual exported workspace, including test edges.
        # A historical SDK may compile kernels even without a CUDA model feature.
        model, packages = inventory.load_model(root, root), role_packages(root)
        graph, roots = selected_graph(metadata(root / 'Cargo.toml'), root / 'Cargo.toml',
                                      packages=packages, features=features)
        pin = check(root, release=release_pin, features=features,
                    contributor_roots=contributor_roots, selected=graph)
        require(all(p.stat().st_mode & 0o222 == 0 for p in (root / 'sdk').rglob('*')),
                'SDK inputs must remain read-only; use the exported checkout')
        native = native_toolchain(graph, roots, env)
        logs.mkdir(parents=True, exist_ok=True)
        claim_target(target, root)
        # Held for the whole run, so concurrent invocations never share outputs.
        with target_lock(target):
            env['CARGO_TARGET_DIR'] = str(target)
            command = [env.get('CARGO', 'cargo'), '--locked', '--offline', '--manifest-path', root / 'Cargo.toml',
                       *package_selection(packages), *cargo_config()]
            if features:
                command += ['--features', ','.join(features)]
            try:
                verified_cargo([command[0], 'test', *command[1:]], root, env=env, log=logs / 'test.log')
                output = verified_cargo([command[0], 'build', *command[1:]], root, env=env, log=logs / 'build.log')
            finally:
                verify_source(root, release=release_pin)
            artifacts = produced_artifacts(output, root, model)
            receipt = {'schema': 'chopinfer-sdk-build-v1', 'sdk': {k:v for k,v in pin.items() if k != 'files'},
                       'cargo_lock_sha256': sha(root / 'Cargo.lock'), 'features': list(features),
                       'rustc': run(['rustc', '-vV']).strip(), 'platform': platform.system() + '-' + platform.machine(),
                       'artifacts': artifacts}
            receipt.update(native)
            (logs / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
            return receipt


def package(workspace, compiler, module, checkpoint, output):
    root = Path(workspace).resolve()
    output = external_output(root, output)
    check(root)
    require(not output.exists(), 'package destination already exists')
    run([compiler, '--checkpoint', checkpoint, '--module', module, '--output', output], cwd=root)
    verify_source(root)
    return json.loads((output / 'manifest.json').read_text())['id']


def request(port, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


@contextmanager
def server(host, package, log):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with Path(log).open('w') as stream:
        process = subprocess.Popen([str(host), 'serve', '--bundle', str(package),
                                    '--topology', 'tp1', '--listen', f'127.0.0.1:{port}',
                                    '--fake-host-services', '--max-sequences', '4', '--kv-pages', '512',
                                    '--max-context-tokens', '1024', '--prefill-chunk-tokens', '512'],
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                require(process.poll() is None, f'host exited: {Path(log).read_text()}')
                try:
                    request(port, 'GET', '/health')
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.05)
            else:
                raise PolicyError('host health timed out')
            yield port
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def http_check(host, package, log, seed):
    with server(host, package, log) as port:
        models = request(port, 'GET', '/v1/models')['data']
        require([m['id'] for m in models] == ['example/stateful'], 'unexpected model listing')
        encoded = request(port, 'POST', '/tokenize', {'prompt': 'hi', 'add_special_tokens': False})
        require(encoded['tokens'] == [104, 105] and encoded['count'] == 2, 'byte tokenization differs')
        # More distinct requests than the four available state slots requires
        # the declared retirement callback to run after each finished request.
        texts = []
        for _ in range(6):
            body = request(port, 'POST', '/v1/completions', {
                'model': 'example/stateful', 'prompt': 'hi', 'add_special_tokens': False,
                'max_tokens': 3, 'temperature': 0})
            texts.append(body['choices'][0]['text'])
            require(body['usage']['completion_tokens'] == 3, 'completion length differs')
        state = seed
        for token in [104, 105]:
            state = (state * 17 + token) % (1 << 64)
        expected = ''
        for _ in range(3):
            token = 97 + state % 3
            expected += chr(token)
            state = (state * 17 + token) % (1 << 64)
        require(texts == [expected] * 6, f'deterministic completion or retirement differs: {texts}')
        return {'model': models[0]['id'], 'tokens': encoded['tokens'], 'completions': texts}


def check_frontend(host, package_root, vectors, log):
    """Check a package's served frontend against the model's vector files.

    `vectors` maps each kind (render, tokenizer, streaming) to its files.
    Returns the command's per-case lines; any failing case is an error."""
    command = [host, 'bundle', 'check-frontend', '--bundle', package_root]
    for kind, paths in vectors.items():
        for path in paths:
            command += [f'--{kind}', path]
    lines = run(command, log=log).splitlines()
    cases = [line for line in lines if line.startswith(('pass ', 'FAIL '))]
    require(cases and all(line.startswith('pass ') for line in cases),
            f'frontend vectors did not all pass: {lines}')
    return cases


def exercise(host, conformance_host, package_root, output, *, seed=7):
    host, conformance_host, package_root, output = map(lambda p: Path(p).resolve(),
                                                    (host, conformance_host, package_root, output))
    require(not output.exists(), 'exercise output already exists')
    external_output(package_root, output)
    hashes = {str(p):sha(p) for p in (host, conformance_host)}
    package_hash = identity(tree(package_root))
    output.mkdir(parents=True)
    manifest = json.loads((package_root / 'manifest.json').read_text())
    bundle = manifest['id']
    run([host, 'bundle', 'verify', '--bundle', package_root], log=output / 'verify.log')
    receipts = []
    try:
        for scenario, status, count in [('commit', 'committed', 1), ('cancel', 'cancelled', 2), ('drop', 'dropped', 1)]:
            value = json.loads(run([conformance_host, '--bundle', package_root,
                                   '--fixture-request', 'external-bundle-host', '--topology', 'tp1',
                                   '--fake-host-services', '--scenario', scenario], log=output / f'{scenario}.log'))
            expected = {'status':status, 'scenario':scenario, 'bundle':bundle, 'live_resources':0,
                        'executions':count, 'rows':count, 'logits_per_row':272, 'device':'host-memory'}
            require(all(value[k] == v for k,v in expected.items()), f'host scenario differs: {value}')
            receipts.append(value)
        http = http_check(host, package_root, output / 'http.log', seed)
    finally:
        require({p:sha(p) for p in hashes} == hashes, 'prebuilt host changed')
        require(identity(tree(package_root)) == package_hash, 'the served package changed')
    receipt = {'schema':'chopinfer-sdk-host-evidence-v1', 'host_sha256':hashes,
               'package_sha256':package_hash, 'scenarios':receipts, 'http':http,
               'evidence':'synthetic CPU boundary mechanics; no CUDA or model quality qualification'}
    (output / 'evidence.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
