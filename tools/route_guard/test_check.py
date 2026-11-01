#!/usr/bin/env python3
"""CPU tests for selected-entry profile and journal checks."""
import contextlib
import importlib.util
import io
import json
import pathlib
import sqlite3
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('route_check', pathlib.Path(__file__).with_name('check.py'))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.classes = self.root / 'classes.json'
        self.tags = [{'entry': 'projection/fallback/shared', 'symbol': 'shared', 'class': 'Baseline',
                      'family': 'projection', 'admitted': [[512, 512]], 'wrappers': ['projection']}]
        self.write_classes()

    def write_classes(self):
        self.classes.write_text(json.dumps({'schema': 'chopinfer.route-tags.v1', 'profile': 'tp2-ar', 'tags': self.tags}))

    def run_check(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return CHECK.main(['--classes', str(self.classes), *map(str, args)])

    def test_journal_baseline_fails_and_justified_job_allowance_passes(self):
        path = self.root / 'journal'
        path.write_text('launch 4 shared config arguments\nlaunch 4 fast\n')
        self.assertEqual(self.run_check('--journal', path), 1)
        allow = self.root / 'allow.json'
        allow.write_text(json.dumps({'allow_fallbacks': {'projection': 'Awaiting range qualification'}}))
        self.assertEqual(self.run_check('--journal', path, '--allow', allow), 0)
        allow.write_text(json.dumps({'allow_fallbacks': {'projection': ''}}))
        self.assertEqual(self.run_check('--journal', path, '--allow', allow), 2)

    def test_phase_scoped_job_allowance_admits_only_its_phases(self):
        path = self.root / 'journal'
        path.write_text('launch 4 shared config arguments\n')
        allow = self.root / 'allow.json'
        allow.write_text(json.dumps({'allow_fallbacks': {
            'projection': {'reason': 'Prefill range pending', 'phases': ['prefill']}}}))
        self.assertEqual(self.run_check('--journal', path, '--allow', allow, '--phase', 'prefill'), 0)
        self.assertEqual(self.run_check('--journal', path, '--allow', allow, '--phase', 'verify'), 1)
        self.assertEqual(self.run_check('--journal', path, '--allow', allow), 1)
        for scope in ({'reason': 'Prefill range pending'}, {'reason': '', 'phases': ['prefill']},
                      {'reason': 'Pending', 'phases': ['draft']}, {'reason': 'Pending', 'phases': []}):
            allow.write_text(json.dumps({'allow_fallbacks': {'projection': scope}}))
            self.assertEqual(self.run_check('--journal', path, '--allow', allow, '--phase', 'prefill'), 2)

    def test_sqlite_uses_string_table_and_device_duration(self):
        path = self.root / 'capture.sqlite'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE StringIds(id INTEGER, value TEXT)')
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, shortName INTEGER)')
            db.execute('INSERT INTO StringIds VALUES (1, "shared")')
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1)', [(10, 30), (40, 90)])
        self.assertEqual(CHECK.nsys(path), {'shared': (2, 70)})
        self.assertEqual(self.run_check('--nsys', path), 1)

    def test_active_exception_prints_scope_reason_and_removal_task(self):
        journal = self.root / 'journal'
        journal.write_text('launch 4 shared\n')
        allow = self.root / 'allow.json'
        item = {'profiles': ['tp2-ar'], 'phase': 'prefill', 'rows': [[513, 513]],
                'entry': self.tags[0]['entry'], 'symbol': 'shared', 'family': 'projection',
                'reason': 'Existing gap awaiting range qualification', 'task': 'projection-range'}
        allow.write_text(json.dumps({'allow': [item]}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(CHECK.main(['--classes', str(self.classes), '--journal', str(journal),
                                        '--allow', str(allow), '--phase', 'prefill', '--rows', '513']), 0)
        records = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['exception'], item)
        self.assertEqual(records[0]['profile'], 'tp2-ar')
        self.assertEqual(self.run_check('--journal', journal, '--allow', allow,
                                        '--phase', 'prefill', '--rows', '514'), 1)

    def test_reused_symbol_needs_entry_evidence_and_fast_entry_passes(self):
        self.tags.append({**self.tags[0], 'entry': 'projection/standalone/shared', 'class': 'Fast'})
        self.write_classes()
        path = self.root / 'journal'
        path.write_text('launch 4 shared\n')
        self.assertEqual(self.run_check('--journal', path), 2)
        steps = self.root / 'steps'
        route = {'fallback_launches': 0, 'families': {}, 'entries': {
            'projection/standalone/shared': {'class': 'Fast', 'family': 'projection', 'symbol': 'shared', 'launches': 1}}}
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
        self.assertEqual(self.run_check('--journal', path, '--steps', steps), 0)
        route['fallback_launches'] = 1
        route['families'] = {'projection': 1}
        route['entries'] = {'projection/fallback/shared': {'class': 'Baseline', 'family': 'projection', 'symbol': 'shared', 'launches': 1}}
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
        self.assertEqual(self.run_check('--journal', path, '--steps', steps), 1)

    def test_irrelevant_steps_cannot_clear_fallbacks_or_ambiguity(self):
        journal = self.root / 'journal'
        journal.write_text('launch 4 shared\n')
        steps = self.root / 'steps'
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': {
            'fallback_launches': 0, 'families': {}, 'entries': {}}}) + '\n')
        self.assertEqual(self.run_check('--journal', journal, '--steps', steps), 2)
        self.tags.append({**self.tags[0], 'entry': 'projection/standalone/shared', 'class': 'Fast'})
        self.write_classes()
        self.assertEqual(self.run_check('--journal', journal, '--steps', steps), 2)

    def test_allowance_cannot_clear_incomplete_supplied_step_evidence(self):
        journal = self.root / 'journal'
        journal.write_text('launch 4 shared\nlaunch 4 shared\n')
        steps = self.root / 'steps'
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': {
            'fallback_launches': 1, 'families': {'projection': 1}, 'entries': {
                self.tags[0]['entry']: {'class': 'Baseline', 'family': 'projection',
                    'symbol': 'shared', 'launches': 1}}}}) + '\n')
        allow = self.root / 'allow.json'
        allow.write_text(json.dumps({'allow_fallbacks': {'projection': 'Range work pending'}}))
        self.assertEqual(self.run_check('--journal', journal, '--steps', steps, '--allow', allow), 2)

    def test_step_metadata_must_match_the_classes_revision(self):
        self.tags[0]['class'] = 'Fast'
        self.write_classes()
        steps = self.root / 'steps'
        for changed in ({'class': 'Baseline'}, {'family': 'other'}, {'symbol': 'other'}):
            value = {'class': 'Fast', 'family': 'projection', 'symbol': 'shared', 'launches': 1, **changed}
            fallback = value['class'] != 'Fast'
            steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': {
                'fallback_launches': int(fallback), 'families': {value['family']: 1} if fallback else {},
                'entries': {self.tags[0]['entry']: value}}}) + '\n')
            self.assertEqual(self.run_check('--steps', steps), 2)

    def test_ambiguous_evidence_takes_precedence_over_policy_failure(self):
        self.tags += [{**self.tags[0], 'entry': 'projection/standalone/shared', 'class': 'Fast'},
                      {**self.tags[0], 'entry': 'projection/fallback/other', 'symbol': 'other'}]
        self.write_classes()
        journal = self.root / 'journal'
        journal.write_text('launch 4 shared\nlaunch 4 other\n')
        self.assertEqual(self.run_check('--journal', journal), 2)

    def test_multiple_captures_need_their_combined_selected_evidence(self):
        self.tags.append({**self.tags[0], 'entry': 'projection/standalone/shared', 'class': 'Fast'})
        self.write_classes()
        first, second = self.root / 'first.journal', self.root / 'second.journal'
        first.write_text('launch 4 shared\n')
        second.write_text('launch 4 shared\n')
        steps = self.root / 'steps'
        route = {'fallback_launches': 0, 'families': {}, 'entries': {
            'projection/standalone/shared': {'class': 'Fast', 'family': 'projection',
                                             'symbol': 'shared', 'launches': 1}}}
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
        self.assertEqual(self.run_check('--journal', first, '--journal', second, '--steps', steps), 2)
        route['entries']['projection/standalone/shared']['launches'] = 2
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
        self.assertEqual(self.run_check('--journal', first, '--journal', second, '--steps', steps), 0)

    def host_steps(self, prepared=False):
        tag = {'entry': 'engram/prepared/prefill' if prepared else 'engram/sync/ar_prefill',
               'symbol': 'host.enqueue_mapped_gather' if prepared else 'host.gather_mapped_rows',
               'class': 'Fast' if prepared else 'Baseline', 'family': 'engram_gather',
               'admitted': [], 'wrappers': []}
        self.tags.append(tag)
        self.write_classes()
        route = {'fallback_launches': 0 if prepared else 2,
                 'families': {} if prepared else {'engram_gather': 2},
                 'entries': {tag['entry']: {**tag, 'launches': 2}}}
        path = self.root / 'host-steps'
        path.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
        return path, route

    def test_sync_host_symbol_rejects_fast_metadata(self):
        for baseline in (0, 2):
            with self.subTest(baseline=baseline):
                self.tags = [tag for tag in self.tags if tag['family'] != 'engram_gather']
                steps, route = self.host_steps()
                sync = self.tags.pop()
                fast = {**sync, 'entry': 'engram/mislabelled', 'class': 'Fast'}
                self.tags.append(fast)
                route['entries'] = {fast['entry']: {**fast, 'launches': 4 - baseline}}
                if baseline:
                    self.tags.append(sync)
                    route['entries'][sync['entry']] = {**sync, 'launches': baseline}
                route.update(fallback_launches=baseline,
                             families={'engram_gather': baseline} if baseline else {})
                self.write_classes()
                steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
                with self.assertRaisesRegex(CHECK.EvidenceError, 'synchronous mapped gather'):
                    CHECK.load_classes(self.classes)
                self.assertEqual(self.run_check('--steps', steps), 2)
                self.tags = [tag for tag in self.tags if tag['family'] != 'engram_gather']

    def test_host_calls_use_steps_without_nsys(self):
        steps, _ = self.host_steps()
        self.assertEqual(self.run_check('--steps', steps), 1)
        allow = self.root / 'allow.json'
        allow.write_text(json.dumps({'allow_fallbacks': {'engram_gather': 'AR prefill pending'}}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(CHECK.main(['--classes', str(self.classes), '--steps', str(steps),
                                        '--allow', str(allow)]), 0)
        self.assertIn('host symbol=host.gather_mapped_rows calls=2', output.getvalue())
        self.assertNotIn('device_ns', output.getvalue())
        steps, _ = self.host_steps(prepared=True)
        self.assertEqual(self.run_check('--steps', steps), 0)

    def test_host_calls_require_metadata_and_consistent_totals(self):
        steps, route = self.host_steps()
        self.tags.pop()
        self.write_classes()
        self.assertEqual(self.run_check('--steps', steps), 2)
        steps, route = self.host_steps()
        for changed in ({'fallback_launches': 1}, {'families': {}}):
            steps.write_text(json.dumps({'schema': 'chopinfer.step.v1',
                                         'route': {**route, **changed}}) + '\n')
            self.assertEqual(self.run_check('--steps', steps), 2)

    def test_host_steps_do_not_cover_missing_kernel_evidence(self):
        steps, route = self.host_steps(prepared=True)
        capture = self.root / 'capture.sqlite'
        with sqlite3.connect(capture) as db:
            db.execute('CREATE TABLE StringIds(id INTEGER, value TEXT)')
            db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, shortName INTEGER)')
            db.execute('INSERT INTO StringIds VALUES (1, "shared")')
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 30, 1)')
        allow = self.root / 'allow.json'
        allow.write_text(json.dumps({'allow_fallbacks': {'projection': 'Range work pending'}}))
        self.assertEqual(self.run_check('--nsys', capture, '--steps', steps, '--allow', allow), 2)
        tag = self.tags[0]
        route['entries'][tag['entry']] = {**tag, 'launches': 1}
        route['fallback_launches'] = 1
        route['families'] = {'projection': 1}
        steps.write_text(json.dumps({'schema': 'chopinfer.step.v1', 'route': route}) + '\n')
        self.assertEqual(self.run_check('--nsys', capture, '--steps', steps, '--allow', allow), 0)

    def test_empty_or_missing_step_evidence_does_not_pass(self):
        path = self.root / 'steps'
        for content in ['', '{"schema":"chopinfer.step.v1"}\n']:
            path.write_text(content)
            self.assertEqual(self.run_check('--steps', path), 2)


if __name__ == '__main__':
    unittest.main()
