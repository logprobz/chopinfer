import copy
import json
import unittest
import indexer_prefill_tile_job as job


def logs(memcheck=False):
    texts=[]
    for shard in (0,1):
        lines=[];count=0
        for i,c in enumerate(job.cases()):
            if i%2!=shard or i not in job.selected(memcheck):continue
            count+=1
            batches=(c['rows']+(512 if c['layer']==20 else 4096)-1)//(512 if c['layer']==20 else 4096)
            row=dict(c,id=i,live=c['rows']-int(c['kind']==12),kept=31 if c['kind']==11 else 512,
                     batches=batches,raw=4,pipelines=4,partitions=1,controls=2,mismatches=0,
                     timed=job.timed(c) and not memcheck,memcheck=memcheck)
            lines.append('test prefix ... PREFILL_TILE_CASE '+json.dumps(row))
            if row['timed']:
                fine=dict(admitted=100,repairs=3,coarse_safe=0,fine_checked=0,pack_fallback_rows=0,timed=False)
                coarse=dict(fine,coarse_safe=90,fine_checked=10)
                lines.append('PREFILL_TILE_GUARDS '+json.dumps(dict(id=i,public=fine,coarse=coarse,compact_tile=coarse if c['layer']>=24 else None)))
                for stage in ('pack_score','all'):
                    for arm in ('coarse','narrow','wide','tile'):
                        lines.append('PREFILL_TILE_TIMING '+json.dumps(dict(id=i,stage=stage,candidate=arm,reference='public' if arm=='coarse' else 'coarse',batches=batches,median_ms=[1.,.8],paired_ratio=.8,exclusive=True)))
        lines.append(f'PREFILL_TILE_COMPLETE shard={shard} cases={count} memcheck={str(memcheck).lower()}')
        lines.append('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 27 filtered out; finished in 2.0s')
        texts.append('\n'.join(lines))
    return texts


class Parser(unittest.TestCase):
    def test_complete_counts_and_prefixed_markers(self):
        value=job.results(logs())
        self.assertEqual((value['cases'],value['raw'],value['timing_pairs']),(94,376,176))
        self.assertEqual(job.results(logs(True),True)['cases'],23)

    def test_missing_or_duplicate_case_fails(self):
        for duplicate in (False,True):
            text=logs();line=text[0].splitlines()[0]
            text[0]=text[0]+'\n'+line if duplicate else text[0].replace(line+'\n','',1)
            with self.assertRaises(AssertionError):job.results(text)

    def test_changed_bits_or_missing_controls_fail(self):
        for old,new in [('"mismatches": 0','"mismatches": 1'),('"controls": 2','"controls": 0'),('"raw": 4','"raw": 3'),('"context": 8192','"context": 4096')]:
            text=logs();text[0]=text[0].replace(old,new,1)
            with self.assertRaises(AssertionError):job.results(text)

    def test_missing_timing_fails(self):
        text=logs();line=next(s for s in text[0].splitlines() if 'PREFILL_TILE_TIMING ' in s)
        text[0]=text[0].replace(line+'\n','',1)
        with self.assertRaises(AssertionError):job.results(text)

    def test_invalid_duration_fails(self):
        text=logs();text[0]=text[0].replace('"median_ms": [1.0, 0.8]','"median_ms": [1.0, NaN]',1)
        with self.assertRaises(AssertionError):job.results(text)

    def test_guard_disagreement_fails(self):
        text=logs();text[0]=text[0].replace('"repairs": 3','"repairs": 4',1)
        with self.assertRaises(AssertionError):job.results(text)

    def test_memcheck_cannot_skip_tests(self):
        text=logs(True);text[0]=text[0].replace('1 passed; 0 failed; 0 ignored','0 passed; 0 failed; 1 ignored')
        with self.assertRaises(AssertionError):job.results(text,True)

    def test_complete_marker_must_match(self):
        text=logs();text[0]=text[0].replace('cases=47','cases=46')
        with self.assertRaises(AssertionError):job.results(text)


if __name__=='__main__':unittest.main()
