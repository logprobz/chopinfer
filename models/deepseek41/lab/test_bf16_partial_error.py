import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import bf16_partial_error as report


class PartialErrors(unittest.TestCase):
    def test_rounding_ties_and_canonical_nan(self):
        values = np.array([1.00390625, 1.01171875, -0.0, np.nan], dtype=np.float32)
        self.assertEqual(report.bf16_bits(values).tolist(), [0x3f80, 0x3f82, 0x8000, 0x7fff])

    def test_rank_order_is_not_an_array_tree_reduction(self):
        a = [np.array([x], dtype=np.float32) for x in [1e8, -1e8, 1, 0]]
        b = [a[0], a[2], a[1], a[3]]
        self.assertEqual(report.widen(report.folded(a, False)).tolist(), [1.0])
        self.assertEqual(report.widen(report.folded(b, False)).tolist(), [0.0])

    def test_original_bound_violations_are_reported_not_hidden(self):
        ranks = [np.array([x], dtype=np.float32) for x in [1.00390625, -1, 0, 0]]
        arrays, counts = report.errors(ranks)
        self.assertEqual(counts['changed_bits'], 1)
        self.assertEqual(counts['current_outside_original_bound'], 0)
        self.assertEqual(counts['candidate_outside_original_bound'], 1)
        self.assertEqual(arrays['current_abs_f64'].tolist(), [0.0])
        self.assertEqual(arrays['candidate_abs_f64'].tolist(), [0.00390625])

    def test_reject_nonfinite_or_missing_ranks(self):
        with self.assertRaises(ValueError):
            report.errors([np.array([1], dtype=np.float32)]*3)
        with self.assertRaises(ValueError):
            report.errors([np.array([np.inf], dtype=np.float32)]*4)
        with self.assertRaises(ValueError):
            report.distribution([])

    def fixture(self, root):
        directory = root/'ar/operands/decode-1-backbone'
        directory.mkdir(parents=True)
        records=[]
        for rank in range(4):
            data = np.full(5120, rank+1, dtype='<f4').tobytes()
            name=f'rank-{rank}.bin';(directory/name).write_bytes(data)
            records.append(dict(rank=rank,site='layer-0-sum',file=name,bytes=len(data),sha256=report.digest(data),
                tensors=[dict(name='output',dtype='f32',shape=[1,5120],offset=0,bytes=len(data))]))
        document=dict(schema='chopinfer.deepseek41.decode-operands.v1',ranks=4,
            dimensions=dict(hidden=5120,first_layer=0,layers=1),live_rows=1,launch_rows=1,
            transaction='transaction-1.json',records=records,phase='Decode(1)',bank='backbone')
        path=directory/'capture.json';path.write_text(json.dumps(document))
        transaction=dict(committed=True,members=[dict(directory=directory.name,manifest_sha256=report.digest(path.read_bytes()))])
        (directory.parent/'transaction-1.json').write_text(json.dumps(transaction))
        (root/'source.json').write_text(json.dumps(dict(diagnostic_only=True)))
        return path

    def test_authenticated_fixture_exports_nonzero_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=self.fixture(root)
            result=report.analyze(root,root/'report.json',root/'inputs',expected_captures=1)
            self.assertEqual(result['cases'],1)
            self.assertEqual(result['counts']['elements'],5120)
            self.assertEqual(result['counts']['changed_bits'],0)
            manifest=json.loads((root/'inputs/manifest.json').read_text())
            self.assertEqual(len(manifest['cases']),1)
            self.assertEqual(manifest['cases'][0]['bytes'],4*5120*4)
            (path.parent/'rank-0.bin').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash or size'):
                report.capture(path)

    def test_uncommitted_and_empty_captures_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,'capture count'):
                report.analyze(root,root/'report.json',expected_captures=1)
            path=self.fixture(root)
            transaction=path.parent.parent/'transaction-1.json'
            data=json.loads(transaction.read_text());data['committed']=False
            transaction.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'not committed'):
                report.capture(path)


if __name__=='__main__':unittest.main(verbosity=2)
