#!/usr/bin/env python3
"""Print validated release selections for staging and deployment consumers."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from model_sdk import inventory

ROOT = Path(__file__).resolve().parents[2]


def read(root, metadata=None):
    models = inventory.load(root)
    if metadata is None:
        metadata = json.loads(subprocess.check_output([
            'cargo', 'metadata', '--format-version', '1', '--no-deps', '--locked', '--offline',
        ], cwd=root))
    inventory.check_metadata(models, metadata, root)
    return inventory.release_plan(models)


def binaries(plan):
    # Host discovery defines the helper prefix. Keep tools first, then helpers
    # in reverse model order, as the existing release builds and copies them.
    tools = tuple(target for model in plan for target in model.targets
                  if target.kind == 'bin' and not target.name.startswith('chopinfer-model-'))
    helpers = tuple(target for model in reversed(plan) for target in model.targets
                    if target.kind == 'bin' and target.name.startswith('chopinfer-model-'))
    return tools + helpers


def build_args(plan, kind):
    targets = binaries(plan) if kind == 'bin' else tuple(
        target for model in plan for target in model.targets if target.kind == 'cdylib')
    owners = dict.fromkeys(target.package for target in targets)
    args = [arg for owner in owners for arg in ('-p', owner)]
    if kind == 'bin':
        args += [arg for target in targets for arg in ('--bin', target.name)]
    return tuple(args)


def variants(root, directory):
    """The checkpoint variants a model directory declares beside its
    qual.json, each a qual.VARIANT.json (tools/qual/declaration.py), by name."""
    return sorted(match.group(1) for path in (Path(root) / directory).glob('qual.*.json')
                  if (match := re.fullmatch(r'qual\.(.+)\.json', path.name)))


def stage(plan, root):
    """The staging rows build.sh installs. An `eval` row installs the files
    `chopinfer eval files` lists for the model's declaration, at their
    checkout paths; its third column names a checkpoint variant, which
    build.sh passes as `--variant`, or is `-` for the model's qual.json."""
    rows = [('binary', name, 'bin/' + name, '755') for name in ('chopinfer', 'chopinfer-host', 'chopinfer-top')]
    rows += [('binary', target.name, inventory.destination(target), '755') for target in binaries(plan)]
    for model in plan:
        rows += [('binary', Path(inventory.destination(target)).name, inventory.destination(target), '755')
                 for target in model.targets if target.kind == 'cdylib']
        rows += [('asset', asset.source, asset.destination, 'copy') for asset in model.assets]
        rows.append(('eval', model.directory, '-', '-'))
        rows += [('eval', model.directory, variant, '-') for variant in variants(root, model.directory)]
    return tuple(rows)


def serving(plan, name, *, extras=False):
    model = next((model for model in plan if model.name == name), None)
    if model is None:
        raise ValueError(f'model {name!r} is not selected for release')
    helpers = [target for target in model.targets if target.kind == 'bin' and
               target.name.startswith('chopinfer-model-')]
    if len(helpers) != 1:
        raise ValueError(f'{name}: serving needs exactly one selected chopinfer-model- helper')
    if extras:
        targets = [target for target in model.targets if target.kind == 'bin' and target not in helpers]
        return tuple((target.name, inventory.destination(target)) for target in targets)
    libraries = [target for target in model.targets if target.kind == 'cdylib']
    if not libraries:
        raise ValueError(f'{name}: serving needs a selected cdylib')
    return (('chopinfer', 'bin/chopinfer'), *((Path(inventory.destination(target)).name,
            inventory.destination(target)) for target in [*helpers, *libraries]))


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--view', choices=('stage', 'build-bins', 'build-libs', 'libraries', 'serving', 'extras'), default='stage')
    parser.add_argument('--model')
    args = parser.parse_args(arguments)
    if args.view in ('serving', 'extras') and not args.model:
        parser.error('--model is required for serving or extras')
    try:
        root = args.root.resolve()
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        plan = read(root, metadata)
        if args.view.startswith('build-'):
            rows = [(arg,) for arg in build_args(plan, 'bin' if args.view == 'build-bins' else 'cdylib')]
        elif args.view == 'libraries':
            rows = [(inventory.destination(target),) for model in plan for target in model.targets if target.kind == 'cdylib']
        elif args.view in ('serving', 'extras'):
            rows = serving(plan, args.model, extras=args.view == 'extras')
        else:
            rows = stage(plan, root)
        for row in rows:
            print('\t'.join(row))
    except (inventory.InventoryError, ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'release plan: {error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
