import json
import unittest
import indexer_guard_edges_job as job


def logs():
    result=[]
    for shard in range(2):
        records=[c for c in job.cases() if c['id'] % 2 == shard]
        text='\n'.join('test prefix ... GUARD_EDGE_CASE '+json.dumps(c) for c in records)
        text+=f'\nGUARD_EDGES_COMPLETE shard={shard} cases={len(records)} total=801 native_oracle_checks={sum(c["native_oracle_checks"] for c in records)}'
        text+='\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured'
        result.append(text)
    return result


class Coverage(unittest.TestCase):
    def test_complete_edge_and_precision_coverage(self):
        self.assertEqual(job.analyze(logs())['native_oracle_checks'], 253440)

    def test_missing_or_repaired_native_evidence_is_rejected(self):
        source=logs()
        for before,after in [('cases=401','cases=400'),('0 failed','1 failed'),
                             ('"native_oracle_checks": 384','"native_oracle_checks": 0'),
                             ('"mismatches": 0','"mismatches": 1'),
                             ('"accepted": true','"accepted": false'),
                             ('"spread": 7','"spread": 6')]:
            bad=[source[0].replace(before,after,1),source[1]]
            self.assertNotEqual(bad,source)
            with self.assertRaises(AssertionError):job.analyze(bad)


if __name__ == '__main__':
    unittest.main()
