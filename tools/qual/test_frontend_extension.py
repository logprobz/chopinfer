"""Owning frontend producers preserve existing pinned full workloads."""
from qual_test_config import setUpModule  # noqa: F401

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration
import frontend_goldens
import refs

QUALIFICATION_RUNS_ONCE = 'Every owning frontend adapter is checked against its own declared inputs'


class OwnedFrontendWorkload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = [path for path in declaration.shipped()
                     if path.name == declaration.NAME and (path.parent / 'frontend_capture.py').is_file()]
        if not cls.paths:
            raise AssertionError('no owning frontend extension exercised')

    def test_complete_workload_preserves_pinned_requests_and_resolves_images(self):
        for path in self.paths:
            config = declaration.load(path)
            with self.subTest(model=config['model']), patch.dict(os.environ, {'CHOPINFER_QUAL_CONFIG': str(path)}):
                # Like the reference-binding tests, require the real external
                # workload for this integration test. Present inputs still
                # undergo their normal pinned-byte verification.
                missing = []
                for name in ('media_fixtures', 'needle'):
                    self.assertIn(name, config['inputs'])
                    location = common.resolve_input(name, optional=True, config=config)
                    if location is None or (not location.exists() and not location.is_symlink()):
                        missing.append(name)
                    else:
                        common.input_path(name, config=config)
                if missing:
                    self.skipTest(f"needs mounted {config['model']} frontend inputs: {', '.join(missing)}")
                root = common.resolve_input('frontend_capture', config=config)
                renders = json.loads(frontend_goldens.reference_bytes(config, 'frontend_capture', root / 'chat-renders.json'))
                before = copy.deepcopy(renders)
                model = config['qualification']['checkpoint']['served_model_name']
                plan = frontend_goldens.extension(config, renders, model)
                self.assertEqual(renders, before)
                self.assertEqual(len(plan['renders']), 75)
                self.assertNotIn('sha256:', json.dumps(plan['renders']))
                api = {row['name']: row['body'] for row in plan['api']}
                shapes = json.loads(frontend_goldens.reference_bytes(config, 'frontend_capture', root / 'api-shapes.json'))
                self.assertEqual(set(api), set(shapes))
                for row in json.loads((root / 'api-requests.json').read_text()):
                    self.assertEqual(api[row['name']], {'model': model, **row['request']})
                self.assertEqual(api['chat-effort-high']['reasoning_effort'], 'high')
                self.assertEqual(len(api['error-too-long']['prompt']), 262145)
                self.assertEqual([row['body']['max_tokens'] for row in plan['needles']], [128, 128, 128])
                self.assertEqual([row['total_context'] for row in plan['needles']],
                                 config['qualification']['needle']['lengths'])
                self.assertEqual(sum(len(row['body']['prompt']) for row in plan['needles']), 401024)
                self.assertEqual(sum(len(row['body']['prompt']) + row['body']['max_tokens']
                                     for row in plan['needles']), 401408)

    def test_pinned_workload_structure_needs_no_external_payloads(self):
        for path in self.paths:
            config = declaration.load(path)
            with self.subTest(model=config['model']):
                root = common.resolve_input('frontend_capture', config=config)
                renders = json.loads(frontend_goldens.reference_bytes(config, 'frontend_capture', root / 'chat-renders.json'))
                shapes = json.loads(frontend_goldens.reference_bytes(config, 'frontend_capture', root / 'api-shapes.json'))
                tokens = json.loads(frontend_goldens.reference_bytes(config, 'tokenizer_goldens',
                    common.resolve_input('tokenizer_goldens', config=config)))
                frozen = common.ROOT / config['qualification']['needle']['frozen']
                needles = json.loads(refs.read_member(config, 'needle_frozen', frozen))['needles']
                self.assertEqual([len(renders), len(tokens), len(shapes), len(needles)], [75, 81, 15, 3])
                self.assertEqual(sum(map(len, (renders, tokens, shapes, needles))), 174)
                self.assertIn('sha256:', json.dumps(renders))

    def test_missing_or_changed_media_refuses_before_image_generation(self):
        for path in self.paths:
            config = declaration.load(path)
            variable = config['inputs']['media_fixtures']['env']
            for present in (False, True):
                with self.subTest(model=config['model'], present=present), tempfile.TemporaryDirectory() as temporary:
                    if present:
                        (Path(temporary) / config['inputs']['media_fixtures']['path']).mkdir()
                    env = {'CHOPINFER_QUAL_CONFIG': str(path), variable: temporary}
                    error = 'sha256 differs' if present else 'missing input'
                    with patch.dict(os.environ, env), patch('multimodal.probe_images') as images:
                        with self.assertRaisesRegex(ValueError, 'lock entry media_fixtures: ' + error):
                            frontend_goldens.extension(config, [], config['qualification']['checkpoint']['served_model_name'])
                        images.assert_not_called()

    def test_changed_unadopted_api_request_bytes_refuse_before_capture(self):
        path = self.paths[0]
        config = declaration.load(path)
        root = common.resolve_input('frontend_capture', config=config)
        actual = Path.read_bytes
        def altered(member):
            raw = actual(member)
            return raw + b' ' if member == root / 'api-requests.json' else raw
        with patch.dict(os.environ, {'CHOPINFER_QUAL_CONFIG': str(path)}), patch.object(Path, 'read_bytes', altered), \
                self.assertRaisesRegex(ValueError, 'pinned manifest'):
            frontend_goldens.extension(config, [], config['qualification']['checkpoint']['served_model_name'])


if __name__ == '__main__':
    unittest.main()
