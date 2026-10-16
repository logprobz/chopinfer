"""Pinned external model authoring commands. Run with python3 -B -m tools.model_sdk."""
import argparse
import json
from pathlib import Path
import subprocess

from .graph import PolicyError
from .inventory import InventoryError
from .source import check, export, provision_lock, update_lock
from .workflow import build, package, exercise
from .relock import relock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('export')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision', default='HEAD')
    p.add_argument('--template', type=Path)
    p.add_argument('--model', help='export the in-tree model directory models/<m> as the workspace. Only '
                   'static path literals are rewritten; references the export cannot resolve are listed in '
                   "model-export.json and printed as warnings. The exported workspace's own check and tests "
                   'are the final verification.')
    p.add_argument('--development', action='store_true')
    p.add_argument('--include', action='append', default=[])
    for name in ['lock', 'check']:
        p = commands.add_parser(name)
        p.add_argument('--workspace', type=Path, required=True)
        if name == 'check':
            p.add_argument('--release', action='store_true')
        else:
            p.add_argument('--update', action='store_true',
                           help="resolve the existing Cargo.lock again after the model's own dependencies "
                           "changed, keeping the SDK's pins, and report the packages added and removed")
    p = commands.add_parser('build')
    for key in ['workspace', 'target', 'logs']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--feature', action='append', default=[])
    p.add_argument('--release-pin', action='store_true')
    p = commands.add_parser('relock', help='rewrite model-owned kernel locks and report the changes')
    for key in ['workspace', 'target', 'logs']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--feature', action='append', default=[])
    p = commands.add_parser('package')
    for key in ['workspace', 'compiler', 'module', 'checkpoint', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p = commands.add_parser('exercise')
    for key in ['host', 'conformance-host', 'package', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    try:
        if args.command == 'export':
            print(export(args.source, args.output, revision=args.revision, template=args.template,
                         development=args.development, includes=args.include, model=args.model))
        elif args.command == 'build':
            print(json.dumps(build(args.workspace, args.target, args.logs, features=args.feature, release_pin=args.release_pin)))
        elif args.command == 'relock':
            print(json.dumps(relock(args.workspace, args.target, args.logs, features=args.feature)))
        elif args.command == 'package':
            print(package(args.workspace, args.compiler, args.module, args.checkpoint, args.output))
        elif args.command == 'exercise':
            print(json.dumps(exercise(args.host, args.conformance_host, args.package, args.output, seed=args.seed)))
        else:
            if args.command == 'check':
                result = check(args.workspace, release=args.release)
            else:
                result = (update_lock if args.update else provision_lock)(args.workspace)
            print(json.dumps({k:v for k,v in result.items() if k != 'files'}, sort_keys=True))
    except (PolicyError, InventoryError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'external model SDK: {error}\n')


if __name__ == '__main__':
    main()
