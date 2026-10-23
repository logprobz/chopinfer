"""CPU self-test of the depth tuner's planning and selection."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402
import report  # noqa: E402
import tune  # noqa: E402

# A synthetic model: depths 0-5 over 32 verify rows, three processes each.
MODEL = {'max_depth': 5, 'verify_rows': 32, 'concurrencies': [1, 2, 4, 8, 16, 32], 'processes_per_depth': 3}
CONFIG = {'name': 'synthetic', 'model': MODEL, 'kinds': {'greedy': 0.5, 'sampled': 0.5},
          'selection': {'margin': 0.015, 'spread': 0.03}}


def window(run, label, kind, c, depth, rate, histogram):
    record = {'name': f'{kind}-c{c}', 'kind': kind, 'concurrency': c, 'depth': depth, 'tokens': 256,
              'request_seconds': c * 256 / rate, 'request_tokens_per_second': rate,
              'full': {'tokens_per_second': rate * 1.2, 'histogram': histogram},
              'streams': {'p10': rate / c}, 'energy': {'joules_per_token': 1000 / rate},
              'acceptance': {'draft_tokens': 10 * depth, 'accepted_tokens': 5 * depth, 'verify_items': 10,
                             'per_position': [{'position': p, 'drafted': 10, 'accepted': 10 - 2 * p}
                                              for p in range(1, depth + 1)]}}
    path = run / label / 'windows' / f'{kind}-c{c}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


class Tuner(unittest.TestCase):
    def test_every_prompt_prefix_keeps_the_mix(self):
        mix = [{'kind': 'completion', 'weight': 5}, {'kind': 'edit', 'weight': 3}, {'kind': 'chat', 'weight': 2}]
        kinds = prompts.order(mix, 32)
        self.assertEqual(kinds[:4], ['completion', 'edit', 'chat', 'completion'])
        for n in range(1, 33):
            for entry in mix:
                share = kinds[:n].count(entry['kind'])
                self.assertLessEqual(abs(share - n * entry['weight'] / 10), 1, (n, entry))

    def test_a_depth_measures_only_the_concurrencies_its_rows_can_change(self):
        config = {'model': MODEL}
        cs = lambda d, every=False: sorted({c for _, c in tune.windows_for(config, d, every)})
        self.assertEqual(cs(0), [1, 2, 4, 8, 16, 32])
        self.assertEqual(cs(1), [1, 2, 4, 8, 16])
        self.assertEqual(cs(3), [1, 2, 4, 8])
        self.assertEqual(cs(5), [1, 2, 4])
        self.assertEqual(cs(5, True), [1, 2, 4, 8, 16, 32])

    def test_processes_alternate_and_cover_every_depth(self):
        labels = tune.order({'model': MODEL})
        self.assertEqual(labels[:7], ['d0p1', 'd1p1', 'd2p1', 'd3p1', 'd4p1', 'd5p1', 'd5p2'])
        self.assertEqual(sorted(labels), sorted(f'd{d}p{p}' for d in range(6) for p in (1, 2, 3)))

    def test_the_shallowest_depth_within_the_margin_wins(self):
        rates = {  # C: {depth: request tok/s}, the same in both kinds
            1: {0: 100, 1: 150, 2: 180, 3: 200, 4: 210, 5: 212},
            8: {0: 400, 1: 430, 2: 470, 3: 520},
            16: {0: 600, 1: 560},
            32: {0: 850},
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps(CONFIG))
            for c, row in rates.items():
                for depth, rate in row.items():
                    for p in (1, 2, 3):
                        for kind in ('greedy', 'sampled'):
                            window(run, f'd{depth}p{p}', kind, c, depth, rate * (1 + 0.002 * p),
                                   {f'i{c}-r{c * (depth + 1)}': 10})
            # C4: depth 5 lands on depth 4's plans, which beat depth 3 clearly.
            for depth, rate, rows in ((0, 250, 4), (3, 330, 16), (4, 360, 20), (5, 361, 20)):
                for p in (1, 2, 3):
                    for kind in ('greedy', 'sampled'):
                        window(run, f'd{depth}p{p}', kind, 4, depth, rate, {f'i4-r{rows}': 10})
            out = report.evaluate(run)
        chosen = {c: v['depth'] for c, v in out['selection'].items()}
        # C1: depth 5 beats depth 4 by 1%, within the margin; C4: depth 5
        # serves depth 4's shapes; C16: drafting loses.
        self.assertEqual(chosen, {'1': 4, '4': 4, '8': 3, '16': 0, '32': 0})
        self.assertEqual(out['selection']['4']['depths'][5]['equivalent_to'], 4)
        self.assertEqual(out['depths'], '1:4,8:3,16:0')

    def test_unstable_depths_are_not_chosen(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps(CONFIG))
            for depth, rate in ((0, 100), (1, 150), (2, 190)):
                for p in (1, 2):
                    for kind in ('greedy', 'sampled'):
                        shapes = {f'i1-r{depth + 1}': 10 + (p if depth == 2 else 0)}
                        window(run, f'd{depth}p{p}', kind, 1, depth, rate, shapes)
            out = report.evaluate(run)
        self.assertEqual(out['selection']['1']['depth'], 1)
        self.assertFalse(out['selection']['1']['depths'][2]['stable'])


if __name__ == '__main__':
    unittest.main()
