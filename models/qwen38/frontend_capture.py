"""The existing full frontend workload, including its six surface cases.

Nine answer requests come from the pinned HF workload. The six transport
requests preserve the original capture recipes, with the fixed tool-result
history retained from the HF migration. No reference is produced here.
"""

import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile


def prepare(config, renders, model):
    import common
    import multimodal
    import needle
    import refs

    root = common.resolve_input('frontend_capture', config=config)
    manifest_bytes = refs.read_member(config, 'fixture_root', root / 'manifest.json')
    manifest = json.loads(manifest_bytes)
    request_bytes = (root / 'api-requests.json').read_bytes()
    if hashlib.sha256(request_bytes).hexdigest() != manifest['files']['api-requests.json']['sha256']:
        raise ValueError('frontend API request bytes differ from the pinned manifest')
    identity_bytes = refs.read_member(config, 'fixture_root', root / 'identity.json')
    hashes = json.loads(identity_bytes)['images']
    with tempfile.TemporaryDirectory(prefix='frontend-images-') as temporary:
        images = multimodal.probe_images(Path(temporary), common.input_path('media_fixtures', config=config), hashes)
    urls = {name: 'data:image/png;base64,' + base64.b64encode(data).decode('ascii') for name, data in images.items()}
    by_hash = {'sha256:' + hashes[name]: urls[name] for name in images}

    def payload(value):
        if isinstance(value, str) and value.startswith('sha256:'):
            if value not in by_hash:
                raise ValueError('frontend image placeholder has no authenticated payload')
            return by_hash[value]
        if isinstance(value, dict):
            return {key: payload(child) for key, child in value.items()}
        if isinstance(value, list):
            return [payload(child) for child in value]
        return value

    requests = json.loads(request_bytes)
    bodies = {row['name']: copy.deepcopy(row['request']) for row in requests}
    if len(bodies) != len(requests):
        raise ValueError('duplicate frontend API request')
    bodies['chat-effort-high'] = {**copy.deepcopy(bodies['chat-thinking']), 'reasoning_effort': 'high'}
    bodies['tool-call-stream'] = {**copy.deepcopy(bodies['tool-call']), 'stream': True,
                                  'stream_options': {'include_usage': True}}
    bodies['chat-stream'] = {'messages': [{'role': 'user', 'content': 'Count from 1 to 5.'}],
                             'max_tokens': 256, 'temperature': 0, 'stream': True,
                             'stream_options': {'include_usage': True}}
    bodies['error-video'] = {'max_tokens': 8, 'messages': [{'role': 'user', 'content': [
        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,AAAA'}},
        {'type': 'text', 'text': 'Describe.'}]}]}
    bodies['error-five-images'] = {'max_tokens': 8, 'messages': [{'role': 'user', 'content': [
        *[{'type': 'image_url', 'image_url': {'url': urls['blue-square']}} for _ in range(5)],
        {'type': 'text', 'text': 'Count.'}]}]}
    bodies['error-too-long'] = {'prompt': [1] * 262145, 'max_tokens': 1}
    shapes = json.loads(refs.read_member(config, 'frontend_capture', root / 'api-shapes.json'))
    if set(bodies) != set(shapes):
        raise ValueError('frontend API request coverage differs from the declared shapes')
    api = [{'name': name, 'endpoint': '/v1/chat/completions' if 'messages' in body else '/v1/completions',
            'body': {'model': model, **body}} for name, body in bodies.items()]
    declared = needle.spec()
    prompts = needle.loader(declared)
    frozen_bytes = refs.read_member(config, 'needle_frozen', common.ROOT / declared['frozen'])
    frozen = {row['total_context']: row for row in json.loads(frozen_bytes)['needles']}
    needles = []
    for length in declared['lengths']:
        ids, budget, expected = prompts(length)
        needles.append({'name': frozen[length]['name'], 'total_context': length, 'expected': expected,
                        'endpoint': '/v1/completions', 'body': {'model': model, 'prompt': ids,
                        'max_tokens': budget, 'temperature': 0, 'return_token_ids': True}})
    return {'renders': payload(renders), 'api': api, 'needles': needles,
            'identity': {'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                         'api_requests_sha256': hashlib.sha256(request_bytes).hexdigest(),
                         'image_identity_sha256': hashlib.sha256(identity_bytes).hexdigest(),
                         'needle_sha256': hashlib.sha256(common.input_bytes('needle', config=config)).hexdigest(),
                         'needle_frozen_sha256': hashlib.sha256(frozen_bytes).hexdigest()}}
