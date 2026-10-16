"""Exercise the supplied precision compiler and module without build tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import unittest

WEIGHTS_ROLE = 'example.precision.weights.v1'
WEIGHT_PLAN = 'chop.weight-plan.v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Preparation(unittest.TestCase):
    def source(self, name, schema='nested', seed=7, expert=False):
        path = self.work / name
        self.write_checkpoint(path, schema, seed, expert)
        return path

    def prepare(self, checkpoint, policy=None, success=True, extra=()):
        # Direct preparation takes no module argument: the helper packages the
        # module installed beside it or in its ../lib.
        command = [str(self.compiler), 'prepare', '--direct', '--model', str(checkpoint),
                   '--tensor-parallel-size', '1', '--cache-dir', str(self.work / 'cache')]
        if policy is not None:
            command += ['--activation-policy', policy]
        result = subprocess.run([*command, *extra], text=True, capture_output=True,
                                env=dict(os.environ, PATH='/unavailable-build-tools',
                                         CHOPINFER_ACTIVATION_POLICY='invalid-environment-default'))
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data['result'], 'prepared-direct')
            self.assertEqual(data['qualification'], 'compatible-not-evaluated')
            self.assertEqual(Path(data['bundle']).name, 'p-' + data['package_identity'])
            self.assertEqual(digest(Path(data['bundle']) / 'manifest.json'), data['package_identity'])
            self.assertEqual(data['profile']['name'], 'tp1-ar')
            return data
        self.assertNotEqual(result.returncode, 0)
        return result.stdout + result.stderr

    def edit_scales(self, root, remove=(), add=()):
        """Rewrite the scale shard and the index without the `remove` tensors
        and with the `add` (name, dtype, shape, bytes) tensors."""
        path = root / 'scales.safetensors'
        raw = path.read_bytes()
        size = struct.unpack_from('<Q', raw)[0]
        tensors = [(name, record['dtype'], record['shape'],
                    raw[8 + size + record['data_offsets'][0]:8 + size + record['data_offsets'][1]])
                   for name, record in json.loads(raw[8:8 + size]).items() if name not in remove]
        header, payload = {}, bytearray()
        for name, dtype, shape, data in [*tensors, *add]:
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(payload), len(payload) + len(data)]}
            payload.extend(data)
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        index_path = root / 'model.safetensors.index.json'
        index = json.loads(index_path.read_text())
        for name in remove:
            del index['weight_map'][name]
        for name, *_ in add:
            index['weight_map'][name] = 'scales.safetensors'
        index_path.write_text(json.dumps(index))

    def manifest(self, prepared):
        return json.loads((Path(prepared['bundle']) / 'manifest.json').read_text())

    def planned_weights(self, prepared):
        """The package plans its weights from the checkpoint: no weight file."""
        root = Path(prepared['bundle'])
        manifest = self.manifest(prepared)
        text = json.dumps(manifest)
        self.assertIn(WEIGHTS_ROLE, text)
        self.assertIn(WEIGHT_PLAN, text)
        self.assertFalse((root / 'precision-weights.bin').exists())
        self.assertFalse(any(p.suffix in ('.img', '.bin') and p.stat().st_size > 65536
                             for p in root.rglob('*') if p.is_file() and p.parent.name != 'native'))

    def document(self, prepared):
        return json.loads((Path(prepared['bundle']) / 'precision.json').read_text())

    @staticmethod
    def layout(document):
        """The weight image layout: spans, packing and sources, not the decode
        order, which follows the selected activation format."""
        weights = json.loads(json.dumps(document['weights']))
        for weight in weights:
            del weight['packing']['decode_order']
        return weights, document['image_bytes']

    def test_all_policies_and_changed_values_use_one_compiler_and_package_cache(self):
        before = digest(self.module)
        a = self.source('uniform-a')
        automatic = self.prepare(a, 'auto')
        checkpoint = self.prepare(a, 'checkpoint')
        bf16 = self.prepare(a, 'bf16')
        self.assertEqual(len({p['package_identity'] for p in [automatic, checkpoint, bf16]}), 3)
        self.assertEqual(len({p['checkpoint_identity'] for p in [automatic, checkpoint, bf16]}), 1)
        for prepared, expected in [(automatic, 'Nvfp4'), (checkpoint, 'Nvfp4'), (bf16, 'Bf16')]:
            self.planned_weights(prepared)
            document = self.document(prepared)
            self.assertEqual(document['version'], 2)
            self.assertEqual(document['checkpoint_identity'], prepared['checkpoint_identity'])
            for operation in document['precision']['operations']:
                effective = operation['effective']
                self.assertEqual(len(operation['bindings']), 3)
                self.assertEqual(effective['arithmetic']['accumulator'], 'F32')
                for operand in effective['activations']:
                    self.assertEqual(operand['format'], expected)
                    self.assertEqual(operand['storage'], 'Bf16')
                for binding in operation['bindings']:
                    has_quantizer = any('_quant_' in s for s in binding['implementation']['stages'])
                    self.assertEqual(has_quantizer, expected == 'Nvfp4' and effective['id'] != 'lookup')
                if expected == 'Bf16' and effective['id'] != 'lookup':
                    self.assertEqual(operation['reuse'], 'PlainNvfp4Bf16V1')
                    self.assertEqual(operation['source']['activations'][0]['format'], 'Nvfp4')
            for weights in document['weights']:
                # The plan reads codes, local and global scales by name.
                packing = weights['packing']
                self.assertEqual([name.rsplit('.', 1)[1] for name in packing['sources']],
                                 ['weight', 'weight_scale', 'weight_scale_2'])
                plan = packing['plan']
                self.assertEqual(len(packing['matrices']), plan['experts'] * plan['projections'])
                self.assertEqual(weights['packing']['input_scale_tensor'] is None,
                                 weights['operation'] == 'lookup')
            self.assertLess(document['image_bytes'], 400000)
        # Policy-specific packages plan the same weight image.
        self.assertEqual(self.layout(self.document(automatic)), self.layout(self.document(bf16)))
        b = self.source('uniform-b', schema='flat', seed=19)
        changed = self.prepare(b, 'auto')
        self.assertNotEqual(changed['package_identity'], automatic['package_identity'])
        self.assertNotEqual(changed['bundle'], automatic['bundle'])
        # Preparing the same inputs again reuses the published package.
        again = self.prepare(a, 'auto')
        self.assertEqual(again['package_identity'], automatic['package_identity'])
        self.assertEqual(again['bundle'], automatic['bundle'])
        self.assertIn('reused completed direct package', again['measurements']['decisions'])
        self.assertIn('emitted direct package', automatic['measurements']['decisions'])
        # The identity covers safetensors headers, not payloads: new values
        # under the same headers select the same package, and the host packs
        # whatever values the checkpoint holds when it loads.
        payload = self.source('payload-only')
        path = payload / 'weights.safetensors'
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0x11
        path.write_bytes(raw)
        self.assertEqual(self.prepare(payload, 'auto')['package_identity'], automatic['package_identity'])
        self.assertEqual(digest(self.module), before)
        # Keep the real packages and their checkpoints for the separate
        # idle-GPU exercise, which packs the weights by each package's plan.
        (self.work / 'packages.json').write_text(json.dumps({
            name: {'package': prepared['bundle'], 'checkpoint': str(root)}
            for name, prepared, root in [('auto', automatic, a), ('checkpoint', checkpoint, a),
                                         ('bf16', bf16, a), ('changed-auto', changed, b)]}) + '\n')

    def test_per_expert_recipe_refuses_checkpoint_and_falls_back_whole_operation(self):
        source = self.source('per-expert', expert=True)
        refusal = self.prepare(source, 'checkpoint', success=False)
        self.assertIn('operation switch', refusal)
        self.assertTrue('per-expert' in refusal or 'bank-uniform' in refusal, refusal)
        auto = self.prepare(source, 'auto')
        document = self.document(auto)
        switch = document['precision']['operations'][1]
        self.assertEqual(switch['effective']['weights'][0]['layout']['id'], 'moe-packed-ExpertProjection')
        self.assertEqual(switch['effective']['activations'][0]['format'], 'Bf16')
        self.assertTrue(switch['fallback_reasons'])
        for binding in switch['bindings']:
            self.assertEqual(binding['implementation']['activations'][0]['format'], 'Bf16')
        self.assertEqual(switch['source']['activations'][0]['quantizer']['calibration']['Static']['scope'], 'Expert')
        overridden = self.prepare(source, 'bf16')
        self.assertNotEqual(auto['package_identity'], overridden['package_identity'])
        self.assertEqual(self.layout(document), self.layout(self.document(overridden)))
        path = self.work / 'packages.json'
        packages = json.loads(path.read_text())
        packages['per-expert-auto'] = {'package': auto['bundle'], 'checkpoint': str(source)}
        path.write_text(json.dumps(packages) + '\n')

    def test_metadata_scales_and_policy_fail_before_publication(self):
        source = self.source('bad-policy')
        self.assertIn('invalid-environment-default', self.prepare(source, success=False))
        self.assertIn('unsupported or repeated', self.prepare(source, 'auto', success=False,
                      extra=['--activation-policy', 'bf16']))
        transformed = self.source('transform')
        config = json.loads((transformed / 'hf_quant_config.json').read_text())
        config['quantization']['input_transform'] = 'hadamard'
        (transformed / 'hf_quant_config.json').write_text(json.dumps(config))
        self.assertIn('input_transform', self.prepare(transformed, 'bf16', success=False))
        invalid = self.source('invalid-scale')
        path = invalid / 'scales.safetensors'
        payload = bytearray(path.read_bytes())
        header_size = struct.unpack_from('<Q', payload)[0]
        payload[8 + header_size] = 0x80  # Negative zero is not a valid unsigned scale.
        path.write_bytes(payload)
        # Direct preparation reads headers only. The host refuses this scale
        # when it packs the weights by the plan; the compiler's CPU test
        # the_weight_plan_refuses_other_checkpoints_and_invalid_scales covers it.
        self.planned_weights(self.prepare(invalid, 'bf16'))
        missing = self.source('missing-calibration')
        self.edit_scales(missing, remove=['adapter.projection.input_scale'])
        self.assertIn('adapter.projection.input_scale', self.prepare(missing, 'bf16', success=False))

    def test_undeclared_tensors_are_refused_by_name(self):
        # Binding claims every checkpoint tensor for a declared projection.
        source = self.source('undeclared')
        self.edit_scales(source, add=[('router.weight', 'BF16', [4, 256], bytes(2048))])
        self.assertIn('router.weight', self.prepare(source, 'bf16', success=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    # Install the helper and module as a release lays them out: the helper in
    # bin/, the module in lib/, where the helper looks for it.
    install = args.work.resolve() / 'install'
    (install / 'bin').mkdir(parents=True)
    (install / 'lib').mkdir()
    Preparation.compiler = Path(shutil.copy2(args.compiler, install / 'bin' / 'example-compile'))
    Preparation.module = Path(shutil.copy2(args.module, install / 'lib' / 'libexample_bundle.so'))
    Preparation.work = args.work.resolve()
    Preparation.write_checkpoint = staticmethod(runpy.run_path(
        str(Path(__file__).resolve().with_name('make_precision.py')))['write_checkpoint'])
    unittest.main(argv=['precision_preparation'], verbosity=2)
