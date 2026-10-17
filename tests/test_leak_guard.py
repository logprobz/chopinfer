"""Exercise tests/leak_guard.py's Git modes on temporary repositories."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('leak_guard', ROOT / 'tests/leak_guard.py')
leak_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leak_guard)

# Split so this file does not carry the shapes it tests.
HOST = 'zz-bench' + '-host-7'
PERSONAL = 'jane.doe' + '@mail.org'
NOREPLY = '1+test' + '@users.noreply.github.com'
TRAILER = 'Co-Authored-By: Claude <noreply' + '@anthropic.com>'
WORDING = 'The ' + 'operator' + ' decided this.'
DATA_PATH = 'Mount the models at /' + 'data/models.\n'


class Repo:
    def __init__(self, root):
        self.root = root

    def git(self, *args, env=None):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True,
                              text=True, env=env).stdout.strip()

    def commit(self, message, files=None, remove=(), env=None):
        for name, text in (files or {}).items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            self.git('add', name)
        for name in remove:
            self.git('rm', '-q', name)
        self.git('commit', '-q', '--allow-empty', '-m', message, env=env)
        return self.git('rev-parse', 'HEAD')


class Isolated(unittest.TestCase):
    def setUp(self):
        # Temporary repositories must not inherit the validation runner's Git context.
        env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1',
                   GIT_AUTHOR_NAME='Test', GIT_AUTHOR_EMAIL=NOREPLY,
                   GIT_COMMITTER_NAME='Test', GIT_COMMITTER_EMAIL=NOREPLY)
        context = patch.dict(os.environ, env, clear=True)
        context.start()
        self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.denylist = self.tmp / 'denylist'
        self.denylist.write_text('zz-bench-host-[0-9]+\n')

    def personal(self):
        return dict(os.environ, GIT_AUTHOR_EMAIL=PERSONAL, GIT_COMMITTER_EMAIL=PERSONAL)


class GitModes(Isolated):
    def setUp(self):
        super().setUp()
        self.repo = Repo(self.tmp / 'repo')
        subprocess.run(['git', 'init', '-q', '-b', 'main', str(self.repo.root)], check=True)
        self.base = self.repo.commit('Add the readme', {'README.md': '# Fixture\n'})

    def guard(self, *args):
        stderr = io.StringIO()
        with patch.object(leak_guard, 'ROOT', self.repo.root), \
                patch.object(leak_guard, 'DENYLIST_FILES', [self.denylist, self.repo.root / 'leak-denylist.local']), \
                contextlib.redirect_stderr(stderr):
            code = leak_guard.main(args)
        return code, stderr.getvalue()

    def outgoing(self, head, base=None):
        return self.guard('--rev', head, '--messages', head, '--base', base or self.base)

    def test_a_clean_push_passes(self):
        head = self.repo.commit(f'Add notes\n\n{TRAILER}', {'notes.md': 'Plain text.\n'})
        self.assertEqual(self.outgoing(head), (0, ''))

    def test_a_deny_list_hit_in_a_changed_file_fails(self):
        head = self.repo.commit('Add notes', {'notes.md': f'Built on {HOST}.\n'})
        code, err = self.outgoing(head)
        self.assertEqual(code, 1)
        self.assertIn(f'{head[:12]}:notes.md:1: local deny-list pattern', err)

    def test_a_version_a_later_commit_removes_still_fails(self):
        added = self.repo.commit('Add notes', {'notes.md': f'Built on {HOST}.\n'})
        head = self.repo.commit('Reword notes', {'notes.md': 'Built here.\n'})
        code, err = self.outgoing(head)
        self.assertEqual(code, 1)
        self.assertIn(f'{added[:12]}:notes.md:1', err)
        # The final tree alone is clean.
        self.assertEqual(self.guard('--rev', head)[0], 0)

    def test_a_copy_or_rename_loses_the_source_paths_allowance(self):
        allowed = self.repo.commit('Add the preview', {'docs/developer-preview.md': DATA_PATH})
        self.assertEqual(self.outgoing(allowed)[0], 0)
        shutil.copy(self.repo.root / 'docs/developer-preview.md', self.repo.root / 'docs/notes.md')
        self.repo.git('add', 'docs/notes.md')
        copied = self.repo.commit('Copy the preview')
        code, err = self.outgoing(copied, allowed)
        self.assertEqual(code, 1)
        self.assertIn('docs/notes.md:1: local /data path', err)
        self.repo.git('reset', '-q', '--hard', allowed)
        self.repo.git('mv', 'docs/developer-preview.md', 'docs/moved.md')
        moved = self.repo.commit('Move the preview')
        code, err = self.outgoing(moved, allowed)
        self.assertEqual(code, 1)
        self.assertIn('docs/moved.md:1: local /data path', err)

    def test_files_below_the_base_are_not_scanned(self):
        published = self.repo.commit('Add old notes', {'old.md': f'Built on {HOST}.\n'})
        head = self.repo.commit('Edit the readme', {'README.md': '# Fixture\n\nMore.\n'})
        self.assertEqual(self.outgoing(head, published)[0], 0)
        self.assertEqual(self.guard('--rev', head)[0], 1)
        self.assertEqual(self.guard('--rev', head, '--history')[0], 1)

    def test_messages_with_internal_wording_or_personal_addresses_fail(self):
        for message in (f'Fix it\n\n{WORDING}', f'Fix it\n\nAsk {PERSONAL}.', f'Fix it on {HOST}'):
            with self.subTest(message=message):
                self.repo.git('reset', '-q', '--hard', self.base)
                head = self.repo.commit(message)
                code, err = self.outgoing(head)
                self.assertEqual(code, 1)
                self.assertIn(f'commit {head[:12]} message:', err)
        self.repo.git('reset', '-q', '--hard', self.base)
        head = self.repo.commit('Fix it', env=self.personal())
        code, err = self.outgoing(head)
        self.assertEqual(code, 1)
        self.assertIn('message:1: email address: author Test <', err)

    def test_citations_outside_the_history_fail(self):
        self.repo.git('checkout', '-q', '--orphan', 'side')
        self.repo.git('rm', '-rqf', '.')
        side = self.repo.commit('Unrelated history', {'side.md': 'Side.\n'})
        self.repo.git('checkout', '-q', 'main')
        head = self.repo.commit(f'Port {side[:9]}')
        code, err = self.outgoing(head)
        self.assertEqual(code, 1)
        self.assertIn(f'cites commit {side[:9]} outside this history', err)
        self.repo.git('reset', '-q', '--hard', self.base)
        head = self.repo.commit(f'Follow up on {self.base[:9]}')
        self.assertEqual(self.outgoing(head)[0], 0)
        head = self.repo.commit('Add notes', {'notes.md': f'Measured at {side[:10]}.\n'})
        code, err = self.outgoing(head)
        self.assertEqual(code, 1)
        self.assertIn(f'{head[:12]}:notes.md added line:1: cites commit {side[:10]}', err)
        # Lines already there are left alone.
        published = head
        head = self.repo.commit('Extend notes', {'notes.md': f'Measured at {side[:10]}.\nAgain.\n'})
        self.assertEqual(self.outgoing(head, published)[0], 0)

    def test_annotated_tags(self):
        self.repo.git('tag', '-a', 'v1', '-m', 'Release v1')
        self.assertEqual(self.guard('--tag', 'v1'), (0, ''))
        self.repo.git('tag', 'light')
        self.assertEqual(self.guard('--tag', 'light')[0], 0)
        self.repo.git('tag', '-a', 'v2', '-m', f'Release v2, built on {HOST}')
        code, err = self.guard('--tag', 'v2')
        self.assertEqual(code, 1)
        self.assertIn('tag v2:3: local deny-list pattern', err)
        self.repo.git('tag', '-a', 'v3', '-m', 'Release v3', env=self.personal())
        code, err = self.guard('--tag', 'v3')
        self.assertEqual(code, 1)
        self.assertIn('email address: tagger Test', err)
        self.repo.git('tag', '-a', 'outer', '-m', 'Outer', 'v2')
        self.assertEqual(self.guard('--tag', 'outer')[0], 1)
        self.repo.git('tag', '-a', 'tree', '-m', 'A tree', 'HEAD^{tree}')
        code, err = self.guard('--tag', 'tree')
        self.assertEqual(code, 1)
        self.assertIn('tag tree: does not peel to a commit', err)

    def test_base_needs_a_revision(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            leak_guard.arguments(['--base', 'main'])


if __name__ == '__main__':
    unittest.main()
