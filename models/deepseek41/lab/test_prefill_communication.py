"""Synthetic peer traces: independent outgoing traffic may overlap a local sum."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import prefill_dma_order as peer_order
import prefill_communication as p


def fixture(path):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE StringIds(id INTEGER,value TEXT)')
        db.executemany('INSERT INTO StringIds VALUES (?,?)', [(1,p.EMBED),(2,p.SUM),(3,p.REUSE),(4,p.SCATTER),(5,'tail')])
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,deviceId INTEGER,streamId INTEGER)')
        db.execute('CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT)')
        db.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (10,'CUDA_MEMCPY_KIND_PTOP')")
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,copyKind INTEGER,streamId INTEGER,bytes INTEGER,srcDeviceId INTEGER,dstDeviceId INTEGER)')
        for rank in range(4):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,5,1,?,0)',(rank,))
        for index, rows in enumerate(p.ROWS):
            first = 1000 + index * 100
            for rank in range(4):
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,2,?,0)',
                           (first+(25 if rank==3 else 18),first+(29 if rank==3 else 22),rank))
                for phase in range(3):
                    destination = (rank+phase+1)%4
                    start=first+phase*6
                    end=start+5
                    if phase==2 and destination==3:end=first+24
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,10,0,?,?,?)',
                               (start,end,rank,rows//4*p.HIDDEN*4,rank,destination))
                    start=first+30+phase*6
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,10,0,?,?,?)',
                               (start,start+5,rank,rows//4*p.HIDDEN*2,rank,destination))
            if index<20:db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,3,0,0)',(first+50,first+80))
        for index in range(8):db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,4,0,0)',(8320+index*2,8321+index*2))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (8400,8410,5,0,0)')


class Ordering(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'trace.sqlite'
        fixture(self.path)

    def change(self,sql):
        with sqlite3.connect(self.path) as db:db.execute(sql)

    def test_independent_outgoing_scatter_is_legal(self):
        result=peer_order.audit(self.path,1)
        self.assertEqual(result['valid_outgoing_scatter_overlap_count'],73)
        self.assertEqual(result['min_incoming_scatter_to_sum_gap_ns'],1)
        self.assertEqual(result['min_all_sums_to_any_gather_gap_ns'],1)
        c=p.analyze(self.path,True,1)[0]
        self.assertEqual(c['reduction_groups']['full']['calls'],41)
        self.assertEqual(c['reduction_groups']['upper']['rows'],49152)
        self.assertEqual(c['reduction_groups']['full']['peer_bytes'],41*8192*p.HIDDEN*9//2)
        self.assertEqual(c['compute_communication_overlap_ms'],0)

    def test_outgoing_activity_is_not_the_incoming_dependency(self):
        with sqlite3.connect(self.path) as db:
            outgoing = db.execute('SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                                  'WHERE srcDeviceId=0 AND start<1030').fetchone()[0]
            incoming = db.execute('SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                                  'WHERE dstDeviceId=0 AND start<1030').fetchone()[0]
            begin = db.execute('SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL '
                               'WHERE shortName=2 AND deviceId=0').fetchone()[0]
        self.assertGreater(outgoing, begin)
        self.assertLess(incoming, begin)
        self.assertFalse(peer_order.audit(self.path, 1)['ordering_failures'])

    def test_incoming_scatter_overlap_is_rejected(self):
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET end=1020 WHERE start=1012 AND dstDeviceId=0')
        with self.assertRaisesRegex(RuntimeError,'incoming scatter overlaps owner sum'):
            p.analyze(self.path,True,1)

    def test_outgoing_gather_overlap_is_rejected(self):
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET start=1020 WHERE start=1030 AND srcDeviceId=0')
        with self.assertRaisesRegex(RuntimeError,'outgoing gather overlaps owner sum'):
            p.analyze(self.path,True,1)

    def test_global_sum_join_is_checked(self):
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET start=1027 WHERE start=1030 AND srcDeviceId=0')
        with self.assertRaisesRegex(RuntimeError,'gather precedes global sum join'):
            p.analyze(self.path,True,1)

    def test_pushed_copies_may_arrive_out_of_phase_order(self):
        # Rank 2 sends its first call's scatters 7 ns earlier, in its own
        # phase order, so rank 0 receives rank 2's before rank 3's: three
        # sources issue a destination's incoming copies when each pushes.
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET start=start-7,end=end-7 '
                    'WHERE srcDeviceId=2 AND start BETWEEN 1000 AND 1012')
        with self.assertRaisesRegex(RuntimeError,'peer phases differ'):
            peer_order.audit(self.path,1)
        self.assertEqual(peer_order.audit(self.path,1,issuer='source')['issuer'],'source')

    def test_the_issuer_keeps_the_phase_order(self):
        # Rank 0 sends its second phase first: wrong for either issuer.
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET start=999 WHERE srcDeviceId=0 AND dstDeviceId=2 AND start=1006')
        for issuer in ('destination','source'):
            with self.assertRaisesRegex(RuntimeError,'peer phases differ'):
                peer_order.audit(self.path,1,issuer=issuer)
        with self.assertRaisesRegex(RuntimeError,'unknown copy issuer'):
            peer_order.audit(self.path,1,issuer='peer')

    def test_wrong_bytes_are_rejected(self):
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=bytes+4 WHERE start=1000 AND srcDeviceId=0')
        with self.assertRaisesRegex(RuntimeError,'peer bytes differ'):
            p.analyze(self.path,True,1)

    def test_wrong_source_attribution_is_rejected(self):
        self.change('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET deviceId=dstDeviceId WHERE start=1000 AND srcDeviceId=0')
        with self.assertRaisesRegex(RuntimeError,'activity device is not its source'):
            p.analyze(self.path,True,1)

    def test_baseline_and_silent_fallback(self):
        self.change('DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY')
        self.change(f"UPDATE StringIds SET value='{p.ROUND}' WHERE id=2")
        self.assertEqual(p.analyze(self.path,False,1)[0]['kernels'][p.ROUND]['launches'],73)
        with self.assertRaisesRegex(RuntimeError,'wrong DMA trace counts'):
            p.analyze(self.path,True,1)

    def test_multiple_chunks_keep_collective_identity(self):
        with sqlite3.connect(self.path) as db:
            kernels=db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
            copies=db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
            for chunk in range(1,4):
                shift=chunk*10000
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)',
                               [(a+shift,b+shift,n,d,s) for a,b,n,d,s in kernels])
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?)',
                               [(a+shift,b+shift,*rest) for a,b,*rest in copies])
            for chunk in range(3):
                db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=4 AND start IN (?,?)',
                           (8332+chunk*10000,8334+chunk*10000))
        chunks=p.analyze(self.path,True)
        self.assertEqual([c['offset'] for c in chunks],[0,8192,16384,24576])
        self.assertEqual(peer_order.audit(self.path)['sums_checked'],1168)

    def test_cli_writes_ordering_and_wall_time(self):
        out=Path(self.directory.name)/'result.json'
        steps=Path(self.directory.name)/'steps.json'
        steps.write_text(json.dumps([dict(prefill_tokens=8192, execution_us=20)]))
        p.main([str(self.path),'--dma','--chunks','1','--steps',str(steps),'--json',str(out)])
        result=json.loads(out.read_text())
        self.assertEqual(result['ordering']['peer_copies_checked'],73*24)
        self.assertEqual(result['chunks'][0]['wall_ms'],0.02)

    def test_signed_zero(self):
        with self.assertRaisesRegex(RuntimeError,'served output bits differ'):
            p.exact({'x':-0.0},{'x':0.0})

    def test_attention_variant_does_not_change_dma_dependency_contract(self):
        self.change("UPDATE StringIds SET value='chop_latent_sources_decoded_head_async_finish_v1' WHERE id=3")
        chunks = p.analyze(self.path, True, 1)
        self.assertEqual(chunks[0]['paired_attention_launches'], 0)
        self.assertEqual(chunks[0]['kernels'][p.SUM]['launches'], 73)


def bf16(path):
    """The fixture with BF16 partials: rounded sends, half the scatter bytes."""
    with sqlite3.connect(path) as db:
        db.execute('UPDATE StringIds SET value=? WHERE id=2',(peer_order.BF16_SUM,))
        db.execute('INSERT INTO StringIds VALUES (6,?)',(peer_order.BF16_ROUND,))
        for index,rows in enumerate(p.ROWS):
            first=1000+index*100
            db.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=? WHERE start BETWEEN ? AND ?',
                       (rows//4*p.HIDDEN*2,first,first+12))
            for rank in range(4):
                for piece in range(peer_order.pieces(rank)):
                    start=first-20+piece*5
                    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,6,?,0)',(start,start+4,rank))


class Bf16Ordering(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'trace.sqlite'
        fixture(self.path);bf16(self.path)

    def change(self,sql):
        with sqlite3.connect(self.path) as db:db.execute(sql)

    def test_bf16_payload_rounds_before_its_scatters(self):
        result=peer_order.audit(self.path,1,'bf16')
        self.assertEqual((result['payload'],result['sums_checked'],result['rounds_checked']),('bf16',73*4,73*6))
        self.assertEqual(result['peer_copies_checked'],73*24)
        self.assertEqual(result['min_round_to_scatter_gap_ns'],11)

    def test_payloads_do_not_pass_for_each_other(self):
        with self.assertRaisesRegex(RuntimeError,'another DMA payload executed'):
            peer_order.audit(self.path,1,'f32')
        other=Path(self.directory.name)/'f32.sqlite';fixture(other)
        with self.assertRaisesRegex(RuntimeError,'another DMA payload executed'):
            peer_order.audit(other,1,'bf16')

    def test_f32_scatter_bytes_are_rejected(self):
        self.change(f'UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=8192/4*{p.HIDDEN}*4 WHERE start=1000 AND srcDeviceId=0')
        with self.assertRaisesRegex(RuntimeError,'peer bytes differ'):
            peer_order.audit(self.path,1,'bf16')

    def test_rounding_order_is_checked(self):
        self.change('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=1001 WHERE shortName=6 AND start=980 AND deviceId=0')
        with self.assertRaisesRegex(RuntimeError,'scatter reads unrounded rows'):
            peer_order.audit(self.path,1,'bf16')
        self.change('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=984 WHERE shortName=6 AND start=980 AND deviceId=0')
        self.change('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=1010,end=1014 WHERE shortName=6 AND start=1080 AND deviceId=1')
        with self.assertRaisesRegex(RuntimeError,'rounding overwrites rows still scattering'):
            peer_order.audit(self.path,1,'bf16')

    def split(self, ranks=range(4)):
        """Round each sent interval of call 0 on its own before its scatter."""
        for rank in ranks:
            self.change(f'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=6 AND deviceId={rank} AND start BETWEEN 980 AND 990')
            for phase in range(3):
                start=1000+phase*6-4
                self.change(f'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({start},{start+3},6,{rank},0)')

    def test_split_rounds_each_precede_their_scatter(self):
        self.split()
        result=peer_order.audit(self.path,1,'bf16',split=True)
        self.assertEqual(result['split_calls'],4)
        self.assertEqual(result['rounds_checked'],73*6-6+12)
        self.assertEqual(result['min_round_to_scatter_gap_ns'],1)
        with self.assertRaisesRegex(RuntimeError,'wrong BF16 rounding counts'):
            peer_order.audit(self.path,1,'bf16')
        # Unsplit calls still pass, and the fixed form is the default.
        other=Path(self.directory.name)/'plain.sqlite';fixture(other);bf16(other)
        self.assertEqual(peer_order.audit(other,1,'bf16',split=True)['split_calls'],0)

    def test_split_round_after_its_scatter_starts_is_rejected(self):
        self.split()
        self.change('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=1007 WHERE shortName=6 AND start=1002 AND deviceId=0')
        with self.assertRaisesRegex(RuntimeError,'scatter reads unrounded rows'):
            peer_order.audit(self.path,1,'bf16',split=True)

    def test_split_missing_interval_round_is_rejected(self):
        self.split()
        self.change('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=6 AND deviceId=0 AND start=1008')
        with self.assertRaisesRegex(RuntimeError,'scatter reads unrounded rows|wrong BF16 rounding counts'):
            peer_order.audit(self.path,1,'bf16',split=True)

    def test_missing_rounds_are_rejected(self):
        self.change('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=6 AND deviceId=2 AND start=1085')
        with self.assertRaisesRegex(RuntimeError,'wrong BF16 rounding counts'):
            peer_order.audit(self.path,1,'bf16')


if __name__=='__main__':unittest.main(verbosity=2)
