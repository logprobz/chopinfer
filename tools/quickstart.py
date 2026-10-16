"""Build chopinfer and serve one model-owned JSON profile straight from a checkpoint.

`chopinfer serve --model CHECKPOINT` loads the Hugging Face safetensors
checkpoint directly: the installed model helper writes a small package into
the model cache and the host serves it with the profile the helper selects
for the checkpoint. Quickstart asks the helper for that profile first and
serves only the chosen one, or an autoregressive choice as the selected
profile with speculation off (tensor parallel size and
`--speculative-tokens 0` become explicit serve options)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory  # noqa: E402
# Flags from the rank-image flow, refused with the replacement to use.
REMOVED = {
    '--skip-materialize': 'serve --model loads the checkpoint directly; there are no rank images to skip',
    '--skip-package': 'serve --model writes its own package into the model cache; drop the flag',
    '--skip-install': 'nothing is installed any more; serve --model reads the checkpoint directly, drop the flag',
    '--space-margin-gib': 'no rank images are written, so there is no space preflight; drop the flag',
    '--drafter-checkpoint': 'the DFlash drafter is read from CHECKPOINT/dflash/; place it there and drop the flag',
}
# Serve options after `--` that quickstart owns or that no longer exist.
OWNED_SERVE_OPTIONS = {
    '--model': 'use --checkpoint',
    '--tensor-parallel-size': 'choose a profile with that tensor parallel size',
    '--model-cache-dir': 'use the quickstart option --model-cache-dir',
    '--listen': 'use the quickstart option --listen',
    '--install-root': 'removed; serve --model reads the checkpoint directly',
    '--bundle': 'removed from quickstart; serve --model reads the checkpoint directly',
    '--profile': 'removed from quickstart; choose the profile as the second argument',
    '--load': 'removed; serve --model always loads the checkpoint directly',
}


def run(command, **kwargs):
    command = [str(word) for word in command]
    print('+ ' + shlex.join(command), flush=True)
    kwargs.setdefault("cwd", ROOT)
    return subprocess.run(command, check=True, **kwargs)


def binaries(model):
    """(directory, name) of every file the private install holds."""
    return [('bin', 'chopinfer'), ('bin', f'chopinfer-model-{model}'), ('lib', f'libbundle_{model}.so')]


def install(target, release, model):
    """Copy the host, the model helper and its bundle module into a private
    directory. The host refuses a helper directory that is group or world
    writable, which Cargo target directories often are."""
    root = release / 'install'
    for directory, name in binaries(model):
        (root / directory).mkdir(parents=True, exist_ok=True)
        destination = root / directory / name
        # Replace by rename so a binary another process runs is never rewritten.
        with tempfile.NamedTemporaryFile(dir=root / directory, prefix=f'.{name}.', delete=False) as stream:
            partial = Path(stream.name)
        shutil.copyfile(target / 'release' / name, partial)
        partial.chmod(0o755)
        os.replace(partial, destination)
    for path in (root, root / 'bin', root / 'lib'):
        path.chmod(0o755)
    for path in [root, *root.rglob('*')]:
        if path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise SystemExit(f'{path} is group or world writable')
    return root


def verify_command(host, model, checkpoint):
    """Offline SHA-256 check of the checkpoint files. MiMo records digests for
    its default checkpoint; other files there are reported, not failed."""
    command = [host, 'verify', '--model', checkpoint]
    if model == 'mimo26':
        default = json.loads((ROOT / 'models/mimo26/checkpoints.json').read_text())['default']
        command += ['--manifest', f'{ROOT / "models/mimo26/checkpoint-files.json"}#{default}', '--allow-unrecorded']
    return command


def check_layout(parser, model, profile_name, drafter, checkpoint):
    """The helper picks MiMo's drafter from the checkpoint layout; refuse a
    profile it would not serve from this checkpoint."""
    if model != 'mimo26' or drafter == 'none':
        return
    has_dflash = (checkpoint / 'dflash').is_dir()
    if drafter == 'dflash' and not has_dflash:
        parser.error(f'{profile_name} needs the DFlash drafter at {checkpoint}/dflash/')
    if drafter == 'mtp' and has_dflash:
        parser.error(f'{checkpoint} has dflash/, so serve --model picks tp4-dflash; choose tp4-dflash, '
                     'or serve MTP with tools/dev/mimo-redeploy.sh --drafter mtp')
    if drafter == 'mtp' and not (checkpoint / 'model_mtp.safetensors').is_file():
        parser.error(f'{profile_name} needs the MTP weights at {checkpoint}/model_mtp.safetensors')


def selected_profile(helper, checkpoint, ranks):
    """The profile `serve --model` serves `checkpoint` with at `ranks`: the
    helper's own choice, which depends on the checkpoint (a precision without
    MTP, a missing drafter)."""
    command = [str(helper), 'profile', '--model', str(checkpoint), '--tensor-parallel-size', str(ranks)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f'{helper.name} names no serving profile for {checkpoint}: '
                         f'{(result.stderr or result.stdout).strip()}')
    return json.loads(result.stdout)['name']


def served_profile(parser, profiles, requested, selected):
    """The profile quickstart serves for `requested`, given the helper's
    `selected` one: the requested profile itself; for an autoregressive
    request, the selected profile of the same bundle and rank count run with
    speculation off; otherwise the request is refused."""
    if selected == requested:
        return requested
    wanted, chosen = profiles[requested], profiles.get(selected)
    if chosen and wanted['speculation']['drafter'] == 'none' \
            and (chosen['bundle_id'], chosen['tensor_parallel']) == (wanted['bundle_id'], wanted['tensor_parallel']):
        return selected
    parser.error(f'this checkpoint serves {selected} at {wanted["tensor_parallel"]} ranks, not {requested}; '
                 f'choose {selected}')


def serve_command(host, checkpoint, profile, cache, listen, extra):
    command = [host, 'serve', '--model', checkpoint, '--tensor-parallel-size', str(profile['tensor_parallel']),
               '--model-cache-dir', cache]
    chosen = {word.split('=', 1)[0] for word in extra}
    if profile['speculation']['drafter'] == 'none' and '--speculative-tokens' not in chosen:
        # The helper may pick a speculative profile at this size; run autoregressive.
        command += ['--speculative-tokens', '0']
    if listen:
        command += ['--listen', listen]
    return [*command, *extra]


def checkpoint_variables(root=ROOT):
    """The checkpoint variable of each released model, by name."""
    return {model.name: f'CHOPINFER_{model.checkpoint_env}_CHECKPOINT'
            for model in inventory.load(root) if model.release.enabled and model.checkpoint_env}


def main():
    variables = checkpoint_variables()
    parser = argparse.ArgumentParser(prog="tools/quickstart.sh", description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=list(variables))
    parser.add_argument('profile')
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--release', type=Path,
                        help="directory for this run's private binaries; default: fresh directory under DATA/releases")
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-cache-dir', type=Path, help='default: DATA/model-cache')
    parser.add_argument('--listen', help='serve address HOST:PORT (default: the host default)')
    parser.add_argument('--verify', action='store_true', help='check checkpoint file digests before serving')
    for step in ['build', 'serve']:
        parser.add_argument('--skip-' + step, action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print build, install and serving commands without running them')
    parser.add_argument('--allow-all-gpus', action='store_true', help='explicitly allow serving without CUDA_VISIBLE_DEVICES')
    argv = sys.argv[1:]
    for word in argv[:argv.index('--')] if '--' in argv else argv:
        flag = word.split('=', 1)[0]
        if flag in REMOVED:
            parser.error(f'{flag} was removed: {REMOVED[flag]}')
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == '--':
        extra = extra[1:]
    elif extra:
        parser.error('serve overrides must follow --')
    for word in extra:
        flag = word.split('=', 1)[0]
        if flag in OWNED_SERVE_OPTIONS:
            parser.error(f'serve override {flag}: {OWNED_SERVE_OPTIONS[flag]}')
    if not args.skip_serve and not args.dry_run and not args.allow_all_gpus and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('set CUDA_VISIBLE_DEVICES to assigned devices, use --skip-serve, or explicitly pass --allow-all-gpus')
    profile_file = ROOT / 'models' / args.model / 'profiles.json'
    document = json.loads(profile_file.read_text())
    if args.profile not in document['profiles']:
        parser.error(f'unknown profile {args.profile!r}')
    profile = document['profiles'][args.profile]
    checkpoint_env = variables[args.model]
    checkpoint = args.checkpoint or os.environ.get(checkpoint_env)
    if not checkpoint:
        parser.error(f'set --checkpoint or {checkpoint_env}')
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.is_dir():
        check_layout(parser, args.model, args.profile, profile['speculation']['drafter'], checkpoint)
    elif not args.dry_run:
        parser.error(f'checkpoint directory not found: {checkpoint}')
    data = args.data_dir.resolve()
    cache = (args.model_cache_dir or data / 'model-cache').resolve()
    if args.release and not args.release.resolve().is_relative_to(data):
        parser.error('--release must be inside the data directory')
    if args.release and args.release.resolve().is_dir():
        leftovers = sorted(path.name for path in args.release.resolve().iterdir() if path.name != 'install')
        if leftovers:
            parser.error(f'--release holds {", ".join(leftovers)} (an older rank-image release?); '
                         'use a fresh release directory')
    if args.dry_run:
        release = args.release.resolve() if args.release else data / 'releases' / 'NEW-RELEASE'
    else:
        (data / 'releases').mkdir(parents=True, exist_ok=True)
        release = args.release.resolve() if args.release else Path(tempfile.mkdtemp(prefix=args.model + '-', dir=data / 'releases'))
    target = Path(os.environ.get('CARGO_TARGET_DIR', str(data / 'target'))).resolve()
    os.environ.update(CARGO_TARGET_DIR=str(target), CARGO_NET_OFFLINE='true', CARGO_BUILD_JOBS='8',
                      CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='0', CARGO_PROFILE_TEST_DEBUG='0')
    if not args.dry_run:
        scratch = data / 'tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(scratch)
    host = release / 'install' / 'bin' / 'chopinfer'
    helper = release / 'install' / 'bin' / f'chopinfer-model-{args.model}'
    print(f'Release: {release}', flush=True)
    print(f'Model cache: {cache}', flush=True)

    def execute(command):
        if args.dry_run:
            print('+ ' + shlex.join(map(str, command)), flush=True)
        else:
            run(command)

    if not args.skip_build:
        # Cargo checks freshness, including source changes, before reusing binaries.
        execute(['nice', '-n', '19', 'cargo', 'build', '--release', '--locked', '--offline',
                 '-p', 'runtime-host', '-p', f'model-{args.model}', '-p', f'bundle-{args.model}',
                 '--lib', '--bin', 'chopinfer', '--bin', f'chopinfer-model-{args.model}',
                 '--features', 'runtime-host/linux-cuda-host'])
    if args.dry_run:
        for directory, name in binaries(args.model):
            print(f'+ install -m 0755 {shlex.quote(str(target / "release" / name))} '
                  f'{shlex.quote(str(release / "install" / directory / name))}', flush=True)
    else:
        missing = [str(target / 'release' / name) for _, name in binaries(args.model)
                   if not (target / 'release' / name).is_file()]
        if missing:
            parser.error('missing build artifacts: ' + ', '.join(missing) + '; rerun without --skip-build')
        install(target, release, args.model)
    # What the checkpoint serves, before any qualification is shown for it.
    if args.dry_run:
        print('+ ' + shlex.join([str(helper), 'profile', '--model', str(checkpoint), '--tensor-parallel-size',
                                 str(profile['tensor_parallel'])]), flush=True)
        served = args.profile
    else:
        served = served_profile(parser, document['profiles'], args.profile,
                                selected_profile(helper, checkpoint, profile['tensor_parallel']))
    shown = document['profiles'][served]
    if served == args.profile:
        print(f'Profile: {served}: {shown["status"]}: {shown["qualification"]}', flush=True)
    else:
        print(f'Profile: {args.profile} runs this checkpoint\'s {served} with speculation off; '
              f'{served}: {shown["status"]}: {shown["qualification"]}', flush=True)
    execute([host, 'profiles', 'show', profile_file, served])
    if args.verify:
        execute(verify_command(host, args.model, checkpoint))
    if not args.skip_serve:
        command = [str(word) for word in serve_command(host, checkpoint, profile, cache, args.listen, extra)]
        print('+ ' + shlex.join(command), flush=True)
        if not args.dry_run:
            os.execvp(command[0], command)


if __name__ == '__main__':
    main()
