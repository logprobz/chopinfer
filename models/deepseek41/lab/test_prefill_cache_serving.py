import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import prefill_cache_serving as job


def response():
    return dict(usage=dict(prompt_tokens=32768, completion_tokens=1), choices=[dict(
        token_ids=[42], logprobs=dict(tokens=['x'], token_logprobs=[-0.0], top_logprobs=[{'x': -0.0}]))])


def profile(path, cached):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE StringIds(id INTEGER,value TEXT)')
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,deviceId INTEGER)')
        names = {}
        def add(name, start, rank):
            if name not in names:
                names[name] = len(names)+1
                db.execute('INSERT INTO StringIds VALUES (?,?)', (names[name], 'deepseek41_'+name))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)', (start, start+8, names[name], rank))
            return start+10
        for rank in range(4):
            at = 0
            for offset in job.OFFSETS:
                at = add(job.EMBED, at, rank)
                for layer in range(40):
                    window = layer < 2
                    page = 32 if 2 <= layer < 20 else 64
                    decoded = cached and (window or offset > 0)
                    if decoded:
                        at = add('chop_latent_prefill_clear_pages_v1', at, rank)
                        if not window:
                            at = add(f'chop_latent_prefill_mark_pages_p{page}_v1', at, rank)
                        at = add('chop_latent_prefill_decode_window_d512_v1', at, rank)
                        if not window:
                            at = add(f'chop_latent_prefill_decode_pages_d512_p{page}_v1', at, rank)
                    name = 'chop_latent_sources_'+('window_' if window else 'mixed_')+('decoded_' if decoded else '')
                    name += 'prefill_'+('p32_' if page == 32 else '')+'reuse2_v1'
                    at = add(name, at, rank)
                    at = add('chop_latent_sources_prefill_merge_v1', at, rank)


class Gates(unittest.TestCase):
    def test_canonical_preserves_every_reported_float_bit(self):
        original = response()
        self.assertEqual(job.canonical(original)['selected'], ['8000000000000000'])
        other = copy.deepcopy(original)
        other['choices'][0]['logprobs']['top_logprobs'][0]['x'] = 0.0
        self.assertNotEqual(job.canonical(original), job.canonical(other))
        for mutate in (
            lambda r: r['usage'].update(prompt_tokens=8192),
            lambda r: r['usage'].update(prompt_tokens_details=dict(cached_tokens=1)),
            lambda r: r['choices'][0]['logprobs'].update(token_logprobs=[float('nan')]),
            lambda r: r['choices'][0]['logprobs'].update(top_logprobs=[{}]),
        ):
            bad = response()
            mutate(bad)
            with self.assertRaises(AssertionError):
                job.canonical(bad)

    def test_all_layers_ranks_offsets_and_producers_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            for cached in (False, True):
                path = Path(directory)/f'{cached}.sqlite'
                profile(path, cached)
                chunks = job.analyze(path, cached)
                self.assertEqual(len(chunks), 16)
                self.assertTrue(all(len(c['layers']) == 40 for c in chunks))
                with self.assertRaises(AssertionError):
                    job.analyze(path, not cached)
            path = Path(directory)/'True.sqlite'
            with sqlite3.connect(path) as db:
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=(SELECT id FROM StringIds WHERE value LIKE '%decode_window%')")
            with self.assertRaises(AssertionError):
                job.analyze(path, True)

    def test_missing_rank_and_merge_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for missing in ('rank', 'merge'):
                path = Path(directory)/f'{missing}.sqlite'
                profile(path, True)
                with sqlite3.connect(path) as db:
                    if missing == 'rank':
                        db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=3')
                    else:
                        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=(SELECT id FROM StringIds WHERE value LIKE '%_merge_%')")
                with self.assertRaises(AssertionError):
                    job.analyze(path, True)

    def test_exclusive_union_does_not_double_count(self):
        self.assertAlmostEqual(job.union_ms([(1, 11), (5, 21), (30, 40)]), 30e-6)

    def test_trace_marker_can_follow_a_prefix(self):
        row = dict(schema='chopinfer.step.v1', prefill_tokens=8192)
        self.assertEqual(job.trace_rows('prefix '+json.dumps(row)), [row])

    def test_seal_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root/'binary'
            item.write_bytes(b'original')
            item.chmod(0o444)
            (root/'prompt.json').write_text(json.dumps([1]*32768))
            (root/'artifacts.json').write_text(json.dumps(dict(
                base_revision='0' * 40,
                sha256={str(item): job.digest(item)})))
            job.verify(root)
            item.chmod(0o644)
            item.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                job.verify(root)
