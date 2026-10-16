"""Exercise the maintained external example without building a host."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .conformance import refusals, releases
from . import inventory
from .source import export, provision_lock, sha, git
from .workflow import build, package, exercise, check_frontend, require, run, server, external_output


def qualify(source, workspace, host, package, output):
    output.mkdir(parents=True)
    with server(host, package, output/'qualification-server.log') as port:
        env = dict(os.environ, CHOPINFER_DATA=str(output), PYTHONDONTWRITEBYTECODE='1',
                   CHOPINFER_EXAMPLE_PACKAGE=str(package))
        run(['python3','-B',source/'tools/qual/release_gate.py', '--config',workspace/'qualification.json',
             '--model-root',workspace,'--url',f'http://127.0.0.1:{port}', '--profile','quick'],
            env=env, log=output/'qualification.log')
    summaries = list(output.rglob('summary.json'))
    require(len(summaries) == 1, 'qualification must produce one summary')
    summary = json.loads(summaries[0].read_text())
    require(summary['passed'] and summary['mechanics_only'] and not summary['qualified'],
            'the synthetic declaration must remain unqualified even without the CLI mechanics flag')
    return summary


def smoke(source, output, host, conformance_host, *, revision='HEAD', development=True, common=False):
    source, output, host, conformance_host = map(lambda p: Path(p).resolve(), (source,output,host,conformance_host))
    external_output(source, output)
    require(not output.exists(), 'smoke output already exists')
    hosts = {str(p):sha(p) for p in [host, conformance_host]}
    includes = git(source, 'ls-files','--others','--exclude-standard','-z','--',
                   'examples/external-model','tools/model_sdk','tests/test_model_sdk.py').decode().split('\0')
    workspace = export(source, output/'model', revision=revision, development=development, includes=includes)
    provision_lock(workspace)
    features = [] if common else ['sm120-helpers', 'cuda-probe']
    # The compiler contributor path stays CUDA-free even when the bundle's
    # explicit CUDA probe is selected. Historical cells retain their requirements.
    contributor_roots = () if common else (inventory.load_model(workspace, workspace).packages['compiler'].name,)
    try:
        build_receipt = build(workspace, output/'target', output/'build', features=features,
                              release_pin=not development, contributor_roots=contributor_roots)
        compiler = output/'target/debug/example-compile'
        module = output/'target/debug/libexample_bundle.so'
        package(workspace, compiler, module, workspace/'fixtures/checkpoint.json', output/'package-1')
        evidence = exercise(host, conformance_host, output/'package-1', output/'host')
        if not common:
            # The example proves its frontend against its own vectors through
            # the prebuilt host, as an external model does.
            evidence['frontend_vectors'] = check_frontend(
                host, output/'package-1',
                {kind: [workspace/f'fixtures/{kind}-vectors.json']
                 for kind in ('render', 'tokenizer', 'streaming')},
                output/'host'/'check-frontend.log')
        checkpoint = output/'checkpoint-2.json'
        checkpoint.write_text('{"seed":8}\n')
        package(workspace, compiler, module, checkpoint, output/'package-2')
        update = releases(host, output/'package-1', output/'package-2', output/'release-checks')
        build(workspace, output/'target', output/'omitted-build', features=[*features,'omit-retirement'],
              release_pin=not development, contributor_roots=contributor_roots)
        package(workspace, compiler, module, workspace/'fixtures/checkpoint.json', output/'omitted-package')
        refusal = refusals(conformance_host, output/'package-1', output/'omitted-package', output/'refusals')
        # Use the current runner for its external-declaration support even in an
        # S0 cell. The package and prebuilt host remain independent of it.
        qualification = qualify(source, workspace, host, output/'package-1', output/'qualification')
        receipt = {'build':build_receipt, 'host':evidence,'release_checks':update,'refusals':refusal,
                   'qualification':{'passed':qualification['passed'],'qualified':qualification['qualified']}}
        if not common:
            receipt['precision_build'] = build(
                workspace, output/'target', output/'precision-build',
                features=['precision-exercise'], release_pin=not development)
            run(['python3', '-B', workspace/'tests/precision_preparation.py',
                 '--compiler', compiler, '--module', module,
                 '--work', output/'precision-preparation'],
                log=output/'precision-preparation.log')
            # The example's kernel tests run on the SDK's kernel harness from its
            # own test crate, which no build above selects: it is not a default
            # member, since it links the CUDA driver. Compile it; its tests need a
            # GPU and a verified build's reports.
            run([os.environ.get('CARGO', 'cargo'), 'test', '--locked', '--offline', '--manifest-path',
                 workspace/'Cargo.toml', '-p', 'example-lab', '--no-run'],
                env=dict(os.environ, CARGO_NET_OFFLINE='true', CARGO_TARGET_DIR=str(output/'target')),
                cwd=workspace, log=output/'lab-build.log')
        (output/'smoke.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt
    finally:
        require({p:sha(p) for p in hosts} == hosts, 'building external roles changed a prebuilt host')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source','output','host','conformance-host']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--release-pin', action='store_true')
    parser.add_argument('--common', action='store_true', help='omit SDK inputs introduced after S0')
    args = parser.parse_args()
    smoke(args.source,args.output,args.host,args.conformance_host, revision=args.revision,
          development=not args.release_pin, common=args.common)
    print('external workspace, packages, unchanged hosts, refusals and rollback passed')


if __name__ == '__main__':
    main()
