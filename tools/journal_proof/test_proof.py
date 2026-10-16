#!/usr/bin/env python3
"""The recorded reviewer false acceptances, the pin manifest and the
identity checklist, on sealed synthetic captures of one device."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine import RAW
from proof import main
from provenance import init
from testkit import CREATE, GEN, MODEL, STEP, GitFixture, checkpoint, create_events, fixtures, launch, seal_capture, write_device


def rule(spec):
    return dict(added_resolves=[dict(symbol='chop_new', ids=[5], count=1)],
                added_modules=[dict(name='cubin_new', fnv1a='cubin_new_hash', bytes=100, ids=[4], count=1)],
                create_policy=spec['journals'][CREATE])


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-proof-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        authority = patch('provenance.AUTHORITY', self.root / 'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.git = GitFixture(self.root / 'objects.git')
        self.base, self.head, self.spec = fixtures()
        self.spec['auxiliary_rules'] = {'host': rule(self.spec)}
        self.captures = 0

    def capture(self, bodies, missing=(), source_extra='', model_toml=None):
        """A sealed capture of one device that records the create, then every
        other journal in order. `missing` journals are pinned in the source
        but absent from the capture."""
        self.captures += 1
        root = self.root / f'capture-{self.captures}'
        revision = self.git.commit(bodies, source_extra=source_extra, model_toml=model_toml)
        events = create_events(bodies[CREATE])
        ends = {CREATE: len(events)}
        for name, lines in bodies.items():
            if name != CREATE:
                events.extend(['note', line] for line in lines)
                ends[name] = len(events)
        write_device(root, 1, events)
        captured = {name: lines for name, lines in bodies.items() if name not in missing}
        for name in captured:
            checkpoint(root, name, [(1, events, ends[name])])
        return revision, seal_capture(root, revision, captured)

    def prove(self, base, head, spec, *extra):
        (self.root / 'spec.json').write_text(json.dumps(spec))
        report = self.root / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--repo', str(self.git.repo), '--model', MODEL, '--base', base[0], '--head', head[0],
                         '--spec', str(self.root / 'spec.json'), '--base-capture-index', str(base[1]),
                         '--head-capture-index', str(head[1]), '--report', str(report), *extra])
        return code, json.loads(report.read_text())

    def run_case(self, expected, head=None, spec=None, missing=(), extra=()):
        base = self.capture(self.base)
        other = self.capture(self.head if head is None else head, missing)
        code, report = self.prove(base, other, self.spec if spec is None else spec, *extra)
        self.assertEqual(code, expected, json.dumps(report.get('problems') or report.get('journals'), indent=1)[:3000])
        if not extra:
            self.assertEqual(report['passed'], expected == 0)
            if expected == 1:
                self.assertEqual(report['status'], 'NOT PROVEN')
        return report


class ReviewerCases(Harness):
    def test_valid_create_mapping_control(self):
        report = self.run_case(0)
        self.assertEqual(report['fixtures']['host']['buffer_map'], {str(GEN + 4): GEN + 6})
        self.assertEqual(report['counts']['added_resolves'], 1)

    def test_symbol_swap_needs_a_locked_module_replacement(self):
        # A resolved symbol changes its device identity; a substitution in the
        # journals alone does not authorize that (test_extensions covers the
        # lock-authenticated replacement).
        head = copy.deepcopy(self.base)
        head[CREATE][2] = 'resolve 3 chop_new'
        head[STEP][0] = launch('chop_new')
        spec = copy.deepcopy(self.spec)
        spec['journals'] = {name: dict(symbol_substitutions=[dict(**{'from': 'chop_old', 'to': 'chop_new'}, count=1)])
                            for name in (CREATE, STEP)}
        spec['auxiliary_rules']['host'] = dict(added_resolves=[], added_modules=[], create_policy=spec['journals'][CREATE])
        self.run_case(1, head, spec)

    def test_b11_step_copied_wholesale_from_base(self):
        # The original B11 decode mutation changes handles only. No symbol
        # or launch schedule change may be responsible for this rejection.
        self.spec['journals'][STEP] = {}
        self.head[STEP] = self.base[STEP][:]
        report = self.run_case(1)
        self.assertIn('unexplained line differences', report['journals'][STEP]['reasons'][0])

    def test_b11_plus_two_on_every_nonfixed_raw(self):
        self.head[STEP] = [RAW.sub(lambda m: f'raw: {int(m[1]) + 2}' if int(m[1]) & 0xffffffff < 1_000_000 else m[0], line)
                           for line in self.head[STEP]]
        self.run_case(1)

    def test_b13_kept_beta_group_moved_past_consumer(self):
        lines = self.head[STEP]
        self.head[STEP] = lines[:1] + lines[3:] + lines[1:3]
        self.run_case(1)

    def test_b13_kept_copy_live_count_offset_changed(self):
        self.head[STEP][2] = self.head[STEP][2].replace(f'raw: {GEN + 5000} }}, offset: 128', f'raw: {GEN + 5000} }}, offset: 0')
        self.run_case(1)

    def test_b13_rank_zero_one_weight_handles_swapped(self):
        a, b = GEN + 1_000_000, GEN + 1_000_001
        self.head[STEP] = [RAW.sub(lambda m: f'raw: {b if int(m[1]) == a else a if int(m[1]) == b else int(m[1])}', line)
                           for line in self.head[STEP]]
        self.run_case(1)

    def test_b15_generation_shift_every_handle(self):
        self.head[STEP] = [RAW.sub(lambda m: f'raw: {int(m[1]) + GEN}', line) for line in self.head[STEP]]
        self.run_case(1)

    def test_b15_stale_base_handles_with_authorized_new_schedule(self):
        self.head[STEP] = [line.replace(f'raw: {GEN + 6}', f'raw: {GEN + 4}') for line in self.head[STEP]]
        self.run_case(1)

    def test_b15_partial_manifest_and_missing_body(self):
        spec = copy.deepcopy(self.spec)
        spec['fixtures'][0]['journals'] = [CREATE]
        report = self.run_case(1, spec=spec)
        self.assertTrue(any('fixture manifest is incomplete' in p for p in report['problems']))
        report = self.run_case(2, missing=(STEP,))
        self.assertIn('captured journals do not match the source pins', report['problems'][0])

    def test_b17_fixed_block_table_shift(self):
        self.head[STEP] = [line.replace(f'raw: {GEN + 6003}', f'raw: {GEN + 6015}') for line in self.head[STEP]]
        self.run_case(1)

    def test_b17_added_resolve_removed(self):
        self.head[CREATE].remove('resolve 5 chop_new')
        self.run_case(1)

    def test_create_generation_and_unexplained_counter_shift(self):
        for raw in (GEN * 2 + 6, GEN + 8):
            head = copy.deepcopy(self.head)
            head[CREATE][5] = f'allocate 64 256 -> raw: {raw}'
            head[STEP] = [line.replace(f'raw: {GEN + 6}', f'raw: {raw}') for line in head[STEP]]
            self.run_case(1, head)

    def test_exact_added_resolve_id_required(self):
        self.head[CREATE][4] = 'resolve 9 chop_new'
        self.run_case(1)

    def test_declared_move_and_state_size_control(self):
        head = copy.deepcopy(self.base)
        head[CREATE][-1] = 'state 64'
        spec = copy.deepcopy(self.spec)
        spec['journals'] = {CREATE: dict(state_sizes=[dict(line=5, **{'from': 'state 32', 'to': 'state 64'})]),
                            STEP: dict(moves=[dict(lines=[2, 3], symbols=['chop_keep_beta', 'chop_keep_copy'], before=1)])}
        spec['auxiliary_rules']['host'] = dict(added_resolves=[], added_modules=[], create_policy=spec['journals'][CREATE])
        # Move lines 2/3 before the first launch, with all arguments unchanged.
        head[STEP] = self.base[STEP][1:3] + self.base[STEP][:1] + self.base[STEP][3:]
        self.run_case(0, head, spec)

    def test_merge_requires_exact_outputs_and_order(self):
        from engine import launch as parse, render
        head = copy.deepcopy(self.base)
        part = parse(self.base[STEP][1])
        second = parse(self.base[STEP][2])
        merged = dict(part, symbol='chop_merged', args=[part['args'][0], second['args'][0]])
        head[STEP] = [head[STEP][0], render(merged), head[STEP][3]]
        spec = copy.deepcopy(self.spec)
        spec['journals'] = {STEP: dict(merges=[dict(lines=[2, 3], symbols=['chop_keep_beta', 'chop_keep_copy'], symbol='chop_merged',
                                                    config={}, args=[dict(launch=0, arg=0), dict(launch=1, arg=0)],
                                                    outputs=[dict(from_launch=0, from_arg=0, to_arg=0),
                                                             dict(from_launch=1, from_arg=0, to_arg=1)])])}
        spec['auxiliary_rules']['host'] = dict(added_resolves=[], added_modules=[], create_policy={})
        self.run_case(0, head, spec)
        head[STEP][1] = head[STEP][1].replace('offset: 0', 'offset: 2', 1)
        self.run_case(1, head, spec)

    def test_undeclared_journal_addition(self):
        self.head['toy-extra.txt'] = ['state 32']
        report = self.run_case(1)
        self.assertIn('added journals differ from exact declaration', report['problems'])

    def test_fixture_cannot_be_split_into_per_step_mappings(self):
        self.spec['fixtures'] = [dict(name='create', create=CREATE, journals=[CREATE]),
                                 dict(name='step', create=STEP, journals=[STEP])]
        self.spec['auxiliary_rules'] = {'create': rule(self.spec), 'step': rule(self.spec)}
        report = self.run_case(1)
        self.assertTrue(any('splits a source fixture' in p for p in report['problems']))

    def test_removal_control(self):
        head = copy.deepcopy(self.base)
        head[STEP].pop(1)
        spec = copy.deepcopy(self.spec)
        spec['journals'] = {STEP: dict(removals=[dict(line=2, symbol='chop_keep_beta')])}
        spec['auxiliary_rules']['host'] = dict(added_resolves=[], added_modules=[], create_policy={})
        report = self.run_case(0, head, spec)
        self.assertEqual(report['counts']['removals'], 1)

    def test_cubin_substitution_needs_a_locked_module_replacement(self):
        head = copy.deepcopy(self.base)
        head[CREATE][1] = 'module 2 cubin_replacement'
        spec = copy.deepcopy(self.spec)
        spec['journals'] = {CREATE: dict(cubin_substitutions=[dict(**{'from': 'cubin_old', 'to': 'cubin_replacement'}, count=1)])}
        spec['auxiliary_rules']['host'] = dict(added_resolves=[], added_modules=[], create_policy=spec['journals'][CREATE])
        self.run_case(1, head, spec)

    def test_live_copy_is_exact_even_when_a_sibling_group_merges(self):
        # A declaration on one launch must never normalize untouched copies.
        self.head[STEP][2] = self.head[STEP][2].replace('length: 4', 'length: 8')
        self.run_case(1)

    def test_merge_output_subspan_is_bounded_and_exact(self):
        from engine import launch as parse, render, argument
        head = copy.deepcopy(self.base)
        parts = [parse(self.base[STEP][i]) for i in (1, 2)]
        changed = dict(parts[0], symbol='chop_subspan', args=[argument(dict(launch=i, arg=0, subspan=8 * i), parts) for i in (0, 1)])
        head[STEP] = [head[STEP][0], render(changed), head[STEP][3]]
        spec = copy.deepcopy(self.spec)
        spec['journals'] = {STEP: dict(merges=[dict(lines=[2, 3], symbols=['chop_keep_beta', 'chop_keep_copy'], symbol='chop_subspan',
                                                    config={}, args=[dict(launch=i, arg=0, subspan=8 * i) for i in (0, 1)],
                                                    outputs=[dict(from_launch=i, from_arg=0, to_arg=i, byte_offset=8 * i)
                                                             for i in (0, 1)])])}
        spec['auxiliary_rules']['host'] = dict(added_resolves=[], added_modules=[], create_policy={})
        self.run_case(0, head, spec)
        spec['journals'][STEP]['merges'][0]['args'][1]['subspan'] = 65
        self.run_case(1, head, spec)

    def test_count_requirement_rejects_silent_noop(self):
        self.spec['journals'][STEP]['launch_rules'][0]['from_symbol'] = 'chop_missing'
        self.run_case(1)


class PinTests(Harness):
    def refused(self, pattern, base=None, head=None):
        code, report = self.prove(base or self.capture(self.base), head or self.capture(self.head), self.spec)
        self.assertEqual(code, 2, report)
        self.assertRegex(report['problems'][0], pattern)

    def test_unreached_source_pin_is_invalid(self):
        ghost = 'fn ghost() { check_journal(&lines, (3, 0x0123456789abcdef), "toy-ghost.txt").unwrap(); }\n'
        self.refused(r'unreached \[\(3, .0123456789abcdef.\)\]', head=self.capture(self.head, source_extra=ghost))

    def test_unreached_computed_pin_is_invalid(self):
        # Review C1 r1: a computed expected pin adds no literal, so the
        # multiset alone would not notice its journal is missing.
        ghost = ('#[test]\n#[ignore]\nfn computed_pin_is_never_reached() {\n'
                 '    let lines = vec!["ghost".to_string()];\n'
                 '    check_journal(&lines, (1, 0xd933_2f4a_8ef9_d5d0 + 1), "toy-ghost.txt").unwrap();\n}\n')
        self.refused('pin tuple is computed, not a literal', head=self.capture(self.head, source_extra=ghost))

    def test_computed_pin_tuple_bound_elsewhere_is_invalid(self):
        ghost = ('fn ghost() { let pin = (1, 0xd933_2f4a_8ef9_d5d0 + 1);\n'
                 '    check_journal(&lines, pin, "toy-ghost.txt").unwrap(); }\n')
        self.refused('a pin tuple is computed, not a literal', head=self.capture(self.head, source_extra=ghost))

    def test_pin_returned_by_a_call_is_invalid(self):
        ghost = 'fn ghost() { check_journal(&lines, pin_of("toy-ghost.txt"), "toy-ghost.txt").unwrap(); }\n'
        self.refused('check_journal computes its expected pin', head=self.capture(self.head, source_extra=ghost))

    def test_aliased_or_indirect_checker_is_invalid(self):
        for extra, pattern in (('use crate::fake_host::check_journal as check;\n', 'imported under an alias'),
                               ('fn ghost() { let check = check_journal; }\n', 'used other than as a direct call')):
            with self.subTest(pattern=pattern):
                self.refused(pattern, head=self.capture(self.head, source_extra=extra))

    def test_imports_methods_and_pins_read_from_tables_are_accepted(self):
        extra = ('use crate::fake_host::{check_journal, journal_digest};\n'
                 'fn other() { PROGRAM.check_journal(catalogs, rows); }\n'
                 'fn table() { for (name, expected) in PINS { check_journal(&lines, expected, name).unwrap(); }\n'
                 '    check_journal(&lines, self.create, &self.file("create")).unwrap();\n'
                 '    check_journal(&lines, expected[index], &file).unwrap();\n'
                 '    check_journal(&lines, (lines, digest), &file).unwrap(); }\n')
        code, report = self.prove(self.capture(self.base), self.capture(self.head, source_extra=extra), self.spec)
        self.assertEqual(code, 0, report)

    def test_captured_journal_without_a_source_pin_is_invalid(self):
        revision, index = self.capture(self.head)
        payload = json.loads(index.read_text())
        # Re-pin the source without the step: the sealed step is then unpinned.
        repinned = self.git.commit({CREATE: self.head[CREATE]})
        from provenance import seal
        payload.pop('attestation')
        payload['revision'] = payload['archive_commit'] = repinned
        index.write_text(json.dumps(seal(payload)))
        self.refused(r'unpinned \[\(4, ', head=(repinned, index))

    def test_commented_out_pin_is_not_a_source_pin(self):
        ghost = '// check_journal(&lines, (3, 0x0123456789abcdef), "toy-ghost.txt");\n'
        code, report = self.prove(self.capture(self.base), self.capture(self.head, source_extra=ghost), self.spec)
        self.assertEqual(code, 0, report)

    def test_disagreeing_pin_sidecar_is_invalid(self):
        revision, index = self.capture(self.head)
        (index.parent / f'{STEP}.pin').write_text('(4, 0x0000000000000001) expected (4, 0x0000000000000002)\n')
        self.refused('the run checked', head=(revision, index))

    def test_agreeing_pin_sidecar_is_accepted(self):
        from common import fnv
        revision, index = self.capture(self.head)
        data = (index.parent / STEP).read_bytes()
        (index.parent / f'{STEP}.pin').write_text(f'(4, 0x{fnv(data)}) expected (4, 0x{fnv(data)})\n')
        code, report = self.prove(self.capture(self.base), (revision, index), self.spec)
        self.assertEqual(code, 0, report)

    def test_tampered_body_is_invalid(self):
        revision, index = self.capture(self.head)
        path = index.parent / STEP
        path.write_text(path.read_text().replace('offset: 128', 'offset: 129', 1))
        self.refused('body differs from helper output', head=(revision, index))

    def test_model_without_a_bundle_is_invalid(self):
        self.refused('declares no \\[bundle\\] package', head=self.capture(self.head, model_toml='[compiler]\npackage = "x"\n'))

    def test_capture_of_another_revision_is_invalid(self):
        base = self.capture(self.base)
        head = self.capture(self.head)
        self.refused('revision differs from requested commit', base=base, head=(base[0], head[1]))


class IdentityTests(Harness):
    def setUp(self):
        super().setUp()
        self.spec = dict(schema='chopinfer.journal-proof.spec.v1',
                         fixtures=[dict(name='host', create=CREATE, journals=[CREATE, STEP])],
                         auxiliary_rules={'host': dict(added_resolves=[], added_modules=[], create_policy={})})

    def test_identical_captures_pass_the_identity_checklist(self):
        report = self.run_case(0, head=self.base, extra=('--require', 'identity'))
        self.assertTrue(all(report['identity_checklist'].values()), report['identity_checklist'])
        self.assertTrue(report['coverage']['full_history_proof'])

    def test_a_declared_change_fails_the_identity_checklist(self):
        head = copy.deepcopy(self.base)
        head[STEP].pop(1)
        self.spec['journals'] = {STEP: dict(removals=[dict(line=2, symbol='chop_keep_beta')])}
        report = self.run_case(1, head=head, extra=('--require', 'identity'))
        self.assertTrue(report['passed'])
        self.assertFalse(report['identity_checklist']['every journal IDENTICAL'])
        self.assertFalse(report['identity_checklist']['no journal policies'])


if __name__ == '__main__':
    unittest.main()
