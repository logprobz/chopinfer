"""Mapped gather records keep typed handles and literal dimensions."""
import unittest

from engine import Unproven, launch as parse_launch
from history import Mapper, classify, stream_of, whole_replays
from testkit import GEN, launch


class MappedHistoryTests(unittest.TestCase):
    def mapper(self, closed=True):
        return Mapper({GEN + 4: GEN + 6}, {}, {2: 3}, {2: 3, 20: 24}, closed=closed)

    def test_namespaced_kernel_launch_maps_stream_and_buffer(self):
        line = launch('family_chop_operation_v1', GEN + 4).replace('launch 1 ', 'launch 2 ')
        self.assertEqual(parse_launch(line)['symbol'], 'family_chop_operation_v1')
        self.assertEqual(stream_of(line), 2)
        for closed in (False, True):
            self.assertEqual(self.mapper(closed)(line), line.replace('launch 2 ', 'launch 3 ').replace(
                f'raw: {GEN + 4}', f'raw: {GEN + 6}'))

    def test_gather_handles_and_wait_streams_map_in_every_owner(self):
        for closed in (False, True):
            mapper = self.mapper(closed)
            for before, after in (('async mapped create 20', 'async mapped create 24'),
                                  ('async mapped release 20', 'async mapped release 24'),
                                  ('async mapped wait 2', 'async mapped wait 3')):
                self.assertEqual(mapper(before), after)
            for line in ('async mapped create 21', 'async mapped release 21', 'async mapped wait 20'):
                with self.assertRaises(Unproven):
                    # A wait on an issued non-stream is refused in history.
                    self.mapper(True)(line)

    def test_gather_lengths_rows_rank_and_view_stay_literal(self):
        for line in ('async mapped ids 20 2', 'async mapped complete',
                     'mapped gather 2 layers.20.engram.embed.weight',
                     'mapped gather 2 layers.20.engram.embed.scale', 'serving kv', 'serving kv 20'):
            self.assertIsNotNone(classify(line))
            self.assertEqual(self.mapper()(line), line)

    def test_wait_breaks_replay_ordering_on_its_stream(self):
        self.assertEqual(stream_of('async mapped wait 2'), 2)
        self.assertEqual(whole_replays(['replay 2 40', 'async mapped wait 2', 'replay 2 41'], ['40', '41']), 1)

    def test_malformed_and_unknown_forms_remain_refused(self):
        for line in ('async mapped create 20 extra', 'async mapped ids 20', 'async mapped ids -1 2',
                     'async mapped wait unknown', 'async mapped complete 20', 'async mapped mystery 20',
                     'mapped gather 2 layers.20 extra', 'serving kv unknown'):
            with self.subTest(line=line), self.assertRaises(Unproven):
                self.mapper()(line)
        for symbol in ('family-chop_operation', 'family_chop_operation!', '42_chop_operation'):
            with self.subTest(symbol=symbol), self.assertRaises(Unproven):
                parse_launch(launch(symbol))
