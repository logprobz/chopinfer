"""Cargo package-ID traversal shared by core and external boundary checks."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from . import inventory


class PolicyError(ValueError):
    pass


class Graph:
    def __init__(self, metadata):
        self.packages = {p['id']: p for p in metadata['packages']}
        self.nodes = {n['id']: n for n in metadata['resolve']['nodes']}

    def linked(self, package, *, tests=False):
        return [d['pkg'] for d in self.nodes[package]['deps']
                if any(k['kind'] in ((None, 'build', 'dev') if tests else (None, 'build'))
                       for k in d['dep_kinds'])]

    def closure(self, package, *, tests=False):
        reached = set()
        # Cargo builds the selected test root's dev dependencies, not their tests.
        pending = self.linked(package, tests=tests)
        while pending:
            key = pending.pop()
            if key not in reached:
                reached.add(key)
                pending.extend(self.linked(key))
        return reached

    def package_ids(self, names):
        result = []
        for name in names:
            found = [key for key, package in self.packages.items() if package['name'] == name]
            if len(found) != 1:
                raise PolicyError(f'expected one selected package named {name}, found {len(found)}')
            result.extend(found)
        return result

    def cuda_build_owners(self, roots):
        reached = set(roots)
        for root in roots:
            reached.update(self.closure(root, tests=True))
        return sorted({self.packages[key]['name'] for key in reached
                       if any(self.packages[edge['pkg']]['name'] == 'kernel-build'
                              and any(kind['kind'] == 'build' for kind in edge['dep_kinds'])
                              for edge in self.nodes[key]['deps'])})

    def require_cuda_free(self, names):
        owners = self.cuda_build_owners(self.package_ids(names))
        if owners:
            raise PolicyError(f'contributor checks for {", ".join(names)} require CUDA through '
                              f'build dependencies of {", ".join(owners)}')


def selected_graph(data, manifest, *, packages=(), features=(), target=None):
    """Use Cargo's selected feature/target edges, keeping metadata package IDs.

    Workspace-wide metadata unifies features of unrelated members. Cargo tree
    selects the same roots as build/test. Retain its dependency-kind headings:
    one package can be a dev dependency while its optional build role is off.
    The all-feature architecture audit still uses metadata.
    """
    command = [os.environ.get('CARGO', 'cargo'), 'tree', '--locked', '--offline',
               '--manifest-path', str(manifest), '--edges', 'normal,build,dev',
               '--prefix', 'indent', '--charset', 'ascii', '--format', '{p}',
               '--no-dedupe', '--color', 'never']
    for package in packages:
        command += ['--package', package]
    if features:
        command += ['--features', ','.join(features)]
    if target:
        command += ['--target', target]
    output = subprocess.check_output(command, cwd=Path(manifest).resolve().parent, text=True)
    return graph_from_tree(data, output)


def graph_from_tree(data, output):
    graph = Graph(data)
    edges, roots, stack = set(), [], []
    for line in output.splitlines():
        if not line.strip():
            continue
        layout = re.fullmatch(r'((?:\|   |    )*)(?:(\|-- |`-- ))?(.+)', line)
        if not layout:
            raise PolicyError(f'unrecognized Cargo tree layout: {line}')
        prefix, branch, text = layout.groups()
        depth = len(prefix) // 4 + bool(branch)
        if text in ('[build-dependencies]', '[dev-dependencies]'):
            if branch or depth >= len(stack):
                raise PolicyError(f'dependency heading has no Cargo parent: {line}')
            stack = stack[:depth + 1]
            stack[-1]['kind'] = 'build' if text == '[build-dependencies]' else 'dev'
            continue
        match = re.fullmatch(r'(\S+) v(\S+)(.*)', text)
        if not match:
            raise PolicyError(f'unrecognized Cargo tree entry: {line}')
        name, version, suffix = match.groups()
        candidates = [key for key, package in graph.packages.items()
                      if package['name'] == name and package['version'] == version]
        if len(candidates) > 1:
            candidates = [key for key in candidates
                          if f'({Path(graph.packages[key]["manifest_path"]).parent})' in suffix]
        if len(candidates) != 1 or depth > len(stack):
            raise PolicyError(f'ambiguous or incomplete selected Cargo graph: {line}')
        key = candidates[0]
        stack = stack[:depth]
        if depth:
            parent, kind = stack[-1]['package'], stack[-1]['kind']
            if not any(edge['pkg'] == key and any(k['kind'] == kind for k in edge['dep_kinds'])
                       for edge in graph.nodes[parent]['deps']):
                raise PolicyError(f'selected Cargo edge is absent from metadata: {line}')
            edges.add((parent, key, kind))
        else:
            roots.append(key)
        stack.append({'package': key, 'kind': None})
    if not roots:
        raise PolicyError('selected Cargo graph contains no roots')
    active = set(roots) | {key for parent, child, _ in edges for key in (parent, child)}
    graph.packages = {key: package for key, package in graph.packages.items() if key in active}
    graph.nodes = {key: {**node, 'deps': [
        {**edge, 'dep_kinds': [kind for kind in edge['dep_kinds']
                             if (key, edge['pkg'], kind['kind']) in edges]}
        for edge in node['deps']
        if any((key, edge['pkg'], kind['kind']) in edges for kind in edge['dep_kinds'])]}
                   for key, node in graph.nodes.items() if key in active}
    return graph, roots


def metadata(manifest):
    return json.loads(subprocess.check_output([
        os.environ.get('CARGO', 'cargo'), 'metadata', '--locked', '--offline',
        '--all-features', '--format-version', '1', '--manifest-path', str(manifest),
    ], cwd=Path(manifest).resolve().parent))


HOST_CRATES = frozenset({'runtime-cuda', 'runtime-engine', 'runtime-eval', 'runtime-host', 'runtime-server'})
# Every role may link these: the ABI, its SDK, operation contracts, and the
# encoder and projection declarations every model role reads.
CONTRACT = frozenset({'runtime-bundle-abi', 'runtime-bundle-sdk', 'kernel-core', 'encoder-spec', 'projection-spec'})
SURFACE = {
    # A bundle exports its chat frontend, the declared format among them.
    'bundle': CONTRACT | {'bundle-runtime', 'kernel-build', 'kernels-sm120', 'frontend-format',
                          'runtime-grammar'},
    # A compiler builds the model's tokenizer contract and plans with the
    # kernel catalog.
    'compiler': CONTRACT | {'model-compiler-sdk', 'runtime-bundle', 'runtime-weights',
                            'runtime-frontend-contract', 'runtime-tokenizer', 'kernels-sm120'},
    # A model's contract is shared by its bundle and compiler: it plans with
    # the model-side SDK, declares the kernels the bundle builds and holds
    # the chat format the bundle exports and the compiler's tests render.
    'contract': CONTRACT | {'model-compiler-sdk', 'runtime-bundle', 'runtime-weights', 'bundle-runtime',
                            'kernel-build', 'kernels-sm120', 'frontend-format'},
}
INTERNAL = frozenset().union(*SURFACE.values()) | {
    'runtime-kernel-contract', 'runtime-kernels', 'runtime-kv', 'runtime-model-api', 'runtime-tokenizer', 'runtime-sampling',
}
UNSUPPORTED = {'kernel-lab', 'runtime-conformance', 'bundle-recurrent-fixture', 'bundle-tp4-fixture'}
# A model's own test crates, the workspace members no role declares, run its
# kernels on a GPU through the model-free kernel harness, which drives the CUDA
# driver, and may plan weights with the host's direct loader. Behind those
# entry points they may reach any SDK crate but the unsupported ones: a model
# never registers its tests in kernel-lab.
TEST_SURFACE = frozenset().union(*SURFACE.values()) | {'kernel-harness', 'runtime-cuda', 'runtime-host'}
SDK_NAMES = INTERNAL | HOST_CRATES | UNSUPPORTED | TEST_SURFACE


def check_external(data, manifest, inventory_path, sdk_root, *, selected=None, contributor_roots=()):
    root = Path(manifest).resolve().parent
    sdk = Path(sdk_root).resolve()
    if Path(data['workspace_root']).resolve() != root:
        raise PolicyError('metadata does not describe the external workspace')
    inventory_path = Path(inventory_path).resolve()
    if inventory_path.name != 'model.toml' or not inventory_path.is_relative_to(root):
        raise PolicyError('inventory must be a model.toml inside the external workspace')
    model = inventory.load_model(root, inventory_path.parent)
    inventory.check_metadata([model], data, root, external=True)
    graph = Graph(data)
    manifests = {Path(p['manifest_path']).resolve(): key for key, p in graph.packages.items()}
    starts = {role: manifests[root / declared.manifest] for role, declared in model.packages.items()}
    # A crossing into the SDK is checked against the surface of the role
    # that owns the crossing crate; another model-owned crate takes the
    # surface of the role whose closure reaches it.
    owners = {start: role for role, start in starts.items()}
    roots = [(role, model.packages[role].name, [(start, None)]) for role, start in starts.items()]
    # A test crate's own dev dependencies are among its entry points.
    roots += [('tests', graph.packages[key]['name'], [(child, key) for child in graph.linked(key, tests=True)])
              for key in data['workspace_members'] if key not in starts.values()]
    surfaces = dict(SURFACE, tests=TEST_SURFACE)
    for role, label, pending in roots:
        seen = set()
        while pending:
            key, parent = pending.pop()
            # Check every crossing, even if another path reached this package.
            package = graph.packages[key]
            name = package['name']
            path = Path(package['manifest_path']).resolve()
            in_sdk = path.is_relative_to(sdk)
            if role == 'bundle' and name in HOST_CRATES:
                raise PolicyError(f'{label} links forbidden host crate {name}')
            if in_sdk:
                relative = path.relative_to(sdk)
                supported = name not in UNSUPPORTED if role == 'tests' else name in INTERNAL
                if not supported or relative != Path('crates') / name / 'Cargo.toml':
                    raise PolicyError(f'{label} links unsupported SDK package {name}')
                parent_in_sdk = parent is not None and Path(
                    graph.packages[parent]['manifest_path']).resolve().is_relative_to(sdk)
                owner = owners.get(parent, role)
                if not parent_in_sdk and name not in surfaces[owner]:
                    raise PolicyError(f'{owner} entry point {name} is outside the supported SDK surface')
            elif name in SDK_NAMES:
                raise PolicyError(f'{name} resolves outside the pinned SDK checkout')
            if key not in seen:
                seen.add(key)
                pending.extend((child, key) for child in graph.linked(key))
    if contributor_roots:
        if selected is None:
            raise PolicyError('contributor checks require the selected external Cargo graph')
        selected.require_cuda_free(contributor_roots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    args = parser.parse_args()
    try:
        check_external(metadata(args.manifest), args.manifest, args.inventory, args.sdk)
    except (PolicyError, inventory.InventoryError) as error:
        parser.exit(1, f'external model boundary: {error}\n')
    print('external model boundary passed')


if __name__ == '__main__':
    main()
