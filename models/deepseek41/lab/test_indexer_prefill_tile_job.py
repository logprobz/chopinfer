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
            # Fixed expectations from the executed matrix, independent of the
            # parser's planner calculation. Dense 65-row cases split at 64.
            counts={32:1,65:1,2496:1,8191:2,8192:2}
            if c['layer']<24:counts[65]=2
            if c['layer']==20:counts.update({2496:5,8191:16,8192:16})
            batches=counts[c['rows']]
            row=dict(c,id=i,live=c['rows']-int(c['kind']==12),kept=31 if c['kind']==11 else 512,
                     batches=batches,raw=4,pipelines=4,reuse_pipelines=1,partitions=1,controls=2,mismatches=0,
                     timed=job.timed(c) and not memcheck,memcheck=memcheck)
            lines.append('test prefix ... PREFILL_TILE_CASE '+json.dumps(row))
            if row['timed']:
                fine=dict(admitted=100,repairs=3,coarse_safe=0,fine_checked=0,pack_fallback_rows=0,timed=False)
                coarse=dict(fine,coarse_safe=90,fine_checked=10)
                lines.append('PREFILL_TILE_GUARDS '+json.dumps(dict(id=i,public=fine,coarse=coarse,compact_tile=coarse if c['layer']>=24 else None)))
                for stage in ('pack_score','all'):
                    for arm in ('coarse','narrow','wide','tile'):
                        lines.append('PREFILL_TILE_TIMING '+json.dumps(dict(id=i,stage=stage,candidate=arm,reference='public' if arm=='coarse' else 'coarse',batches=batches,median_ms=[1.,.8],paired_ratio=.8,exclusive=True,score_buffer_reused=True)))
        lines.append(f'PREFILL_TILE_COMPLETE shard={shard} cases={count} memcheck={str(memcheck).lower()}')
        lines.append('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 27 filtered out; finished in 2.0s')
        texts.append('\n'.join(lines))
    return texts


class Parser(unittest.TestCase):
    def test_batch_planner_boundaries(self):
        # rows, dense ratio 2, dense layer 20, compact retained candidates
        for rows,r2,r1,compact in [
            (32,1,1,1),(63,1,1,1),(64,1,1,1),(65,2,2,1),
            (127,2,2,1),(128,1,1,1),(511,2,2,1),(512,1,1,1),
            (513,2,2,1),(2496,1,5,1),(4033,2,8,1),(4096,1,8,1),
            (4097,2,9,2),(8191,2,16,2),(8192,2,16,2),
        ]:
            for layer,expected in ((2,r2),(8,r2),(14,r2),(20,r1),(24,compact)):
                for context in (8192,16384,24576,32768):
                    for kind in (0,6,12):
                        c=dict(rows=rows,layer=layer,context=context,kind=kind)
                        with self.subTest(**c):
                            self.assertEqual(job.batch_count(c),expected)

    def test_partial_dense_rows_use_two_batches(self):
        value=job.results(logs())
        # The same 65-row shape executes as 64 + 1 for dense scoring, and
        # as one compact batch. Cover all offsets and poisoned variants.
        partial=[c for c in value['records'] if c['rows']==65]
        self.assertEqual(len(partial),48)
        self.assertEqual({c['batches'] for c in partial if c['layer']<24},{2})
        self.assertEqual({c['batches'] for c in partial if c['layer']==24},{1})

    def test_wrong_batch_count_fails_with_case_details(self):
        for layer,wrong in ((2,1),(20,1),(24,2)):
            for memcheck in (False,True):
                text=logs(memcheck)
                for shard in (0,1):
                    lines=text[shard].splitlines()
                    for n,line in enumerate(lines):
                        if 'PREFILL_TILE_CASE ' not in line:continue
                        c=json.loads(line.split('PREFILL_TILE_CASE ',1)[1])
                        if c['layer']==layer and c['rows']==65:
                            c['batches']=wrong
                            lines[n]='PREFILL_TILE_CASE '+json.dumps(c)
                            text[shard]='\n'.join(lines)
                            break
                with self.assertRaisesRegex(AssertionError,'batch count'):
                    job.results(text,memcheck)

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
