"""Tests for the helper authority: sealing never creates a key, and the
explicit init command creates one private key exactly once."""
import errno
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import Invalid
from provenance import canonical, init, key, seal, verify

TOOL = Path(__file__).resolve().parent


def payload():
    files = {'capture.py': '0' * 64}
    return dict(schema='chopinfer.journal-captures.v3', producer='tools/journal_proof/capture.py', real_build=True,
                helper_files=files, helper_digest=hashlib.sha256(canonical(files)).hexdigest())


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-proof-authority-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.authority = self.root / 'data/journal-proof/authority'
        authority = patch('provenance.AUTHORITY', self.authority)
        authority.start()
        self.addCleanup(authority.stop)

    def refused_seal(self):
        with self.assertRaises(Invalid) as caught:
            seal(payload())
        message = str(caught.exception)
        self.assertIn(str(self.authority / 'key'), message)
        self.assertIn('python3 tools/journal_proof/provenance.py init', message)

    def test_seal_without_directory_refuses_and_creates_nothing(self):
        self.refused_seal()
        self.assertEqual(os.listdir(self.root), [])

    def test_seal_without_key_refuses_and_creates_nothing(self):
        self.authority.mkdir(mode=0o700, parents=True)
        self.refused_seal()
        self.assertEqual(os.listdir(self.authority), [])

    def test_verify_without_key_refuses_and_creates_nothing(self):
        with self.assertRaisesRegex(Invalid, 'trusted helper authority is unavailable'):
            verify(dict(payload(), attestation='0' * 64))
        self.assertEqual(os.listdir(self.root), [])

    def test_init_creates_private_directory_and_key(self):
        old = os.umask(0)
        try:
            path = init()
        finally:
            os.umask(old)
        self.assertEqual(path, self.authority / 'key')
        self.assertEqual(stat.S_IMODE(self.authority.lstat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o600)
        self.assertTrue(stat.S_ISREG(path.lstat().st_mode))
        self.assertEqual(len(path.read_bytes()), 32)
        self.assertEqual(key(), path.read_bytes())

    def test_init_refuses_existing_key(self):
        path = init()
        value = path.read_bytes()
        with self.assertRaisesRegex(Invalid, 'already exists'):
            init()
        self.assertEqual(path.read_bytes(), value)

    def test_init_refuses_symlinked_directory(self):
        target = self.root / 'elsewhere'
        target.mkdir()
        self.authority.parent.mkdir(parents=True)
        self.authority.symlink_to(target)
        with self.assertRaisesRegex(Invalid, 'symlink'):
            init()
        self.assertEqual(os.listdir(target), [])

    def test_init_refuses_symlinked_key(self):
        self.authority.mkdir(mode=0o700, parents=True)
        target = self.root / 'target'
        (self.authority / 'key').symlink_to(target)
        with self.assertRaisesRegex(Invalid, 'already exists'):
            init()
        self.assertFalse(target.exists())
        target.write_bytes(b'k' * 32)
        target.chmod(0o600)
        with self.assertRaisesRegex(Invalid, 'already exists'):
            init()
        self.assertEqual(target.read_bytes(), b'k' * 32)
        with self.assertRaisesRegex(Invalid, 'trusted helper authority is unavailable'):
            seal(payload())

    def test_init_refuses_a_directory_replaced_or_removed_before_the_key_is_created(self):
        real_open = os.open
        for replace in (True, False):
            with self.subTest(replace=replace):
                displaced = self.root / f'displaced-{replace}'

                def swap(name, flags, mode=0o777, *, dir_fd=None):
                    if dir_fd is not None and name == 'key':
                        self.authority.rename(displaced)
                        if replace:
                            self.authority.mkdir(mode=0o700)
                    return real_open(name, flags, mode, dir_fd=dir_fd)
                with patch('provenance.os.open', side_effect=swap):
                    with self.assertRaisesRegex(Invalid, 'replaced or removed during init'):
                        init()
                self.assertEqual(os.listdir(displaced), [])
                if replace:
                    self.assertEqual(os.listdir(self.authority), [])
                    self.authority.rmdir()
                self.assertFalse(self.authority.exists())

    def test_entropy_failure_creates_nothing(self):
        with patch('provenance.secrets.token_bytes', side_effect=OSError(errno.EIO, 'no entropy')):
            with self.assertRaises(OSError):
                init()
        self.assertEqual(os.listdir(self.root), [])

    def test_failed_write_removes_the_new_key_and_a_retry_succeeds(self):
        failures = {'write': ('provenance.os.write', dict(side_effect=OSError(errno.ENOSPC, 'no space'))),
                    'short write': ('provenance.os.write', dict(return_value=0)),
                    'fsync': ('provenance.os.fsync', dict(side_effect=OSError(errno.EIO, 'fsync failed')))}
        for name, (target, behavior) in failures.items():
            with self.subTest(name):
                with patch(target, **behavior):
                    with self.assertRaises((OSError, Invalid)):
                        init()
                self.assertEqual(os.listdir(self.authority), [])
        path = init()
        self.assertEqual(len(path.read_bytes()), 32)
        self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o600)
        self.assertEqual(verify(seal(payload())), payload())

    def swap_key(self, data, name='key'):
        """Atomically put another file at `name`, as a concurrent writer would."""
        other = self.authority / 'other'
        other.write_bytes(data)
        os.replace(other, self.authority / name)

    def test_cleanup_never_removes_a_file_swapped_in_after_creation(self):
        real_fsync = os.fsync

        def swap_then_fail(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                self.swap_key(b'existing' * 4)
                raise OSError(errno.EIO, 'fsync failed')
            return real_fsync(fd)
        with patch('provenance.os.fsync', side_effect=swap_then_fail):
            with self.assertRaisesRegex(Invalid, 'fsync failed.*not the file init created; it was left in place'):
                init()
        self.assertEqual(os.listdir(self.authority), ['key'])
        self.assertEqual((self.authority / 'key').read_bytes(), b'existing' * 4)

    def test_cleanup_puts_back_a_file_swapped_in_before_its_rename(self):
        real_rename = os.rename

        def swap_then_rename(*args, **kwargs):
            self.swap_key(b'existing' * 4)
            return real_rename(*args, **kwargs)
        with patch('provenance.os.write', return_value=0), patch('provenance.os.rename', side_effect=swap_then_rename):
            with self.assertRaisesRegex(Invalid, 'short write.*not the file init created; it was left in place'):
                init()
        self.assertEqual(os.listdir(self.authority), ['key'])
        self.assertEqual((self.authority / 'key').read_bytes(), b'existing' * 4)

    def test_cleanup_leaves_a_swapped_in_file_it_cannot_put_back(self):
        real_rename = os.rename

        def swap_rename_and_refill(*args, **kwargs):
            self.swap_key(b'existing' * 4)
            real_rename(*args, **kwargs)
            self.swap_key(b'newest' * 4)
        with patch('provenance.os.write', return_value=0), \
             patch('provenance.os.rename', side_effect=swap_rename_and_refill):
            with self.assertRaisesRegex(Invalid, 'short write.*during cleanup was left at'):
                init()
        held = [name for name in os.listdir(self.authority) if name != 'key']
        self.assertEqual(len(held), 1)
        self.assertEqual((self.authority / held[0]).read_bytes(), b'existing' * 4)
        self.assertEqual((self.authority / 'key').read_bytes(), b'newest' * 4)

    def test_seal_and_verify_after_init(self):
        init()
        sealed = seal(payload())
        self.assertEqual(verify(sealed), payload())
        with self.assertRaisesRegex(Invalid, 'not sealed by the trusted helper or was modified'):
            verify(dict(sealed, real_build=False))
        self.assertEqual(os.listdir(self.authority), ['key'])

    def test_no_authority_configured_refuses(self):
        with patch('provenance.AUTHORITY', None):
            for call in (init, lambda: seal(payload())):
                with self.assertRaisesRegex(Invalid, 'set CHOPINFER_JOURNAL_AUTHORITY or CHOPINFER_DATA'):
                    call()


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-proof-authority-command-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_init(self, **env):
        clean = {k: v for k, v in os.environ.items() if k not in ('CHOPINFER_JOURNAL_AUTHORITY', 'CHOPINFER_DATA')}
        return subprocess.run([sys.executable, '-B', str(TOOL / 'provenance.py'), 'init'], env={**clean, **env},
                              capture_output=True, timeout=60)

    def test_init_prints_only_the_path_and_refuses_a_second_run(self):
        authority = self.root / 'configured'
        env = dict(CHOPINFER_JOURNAL_AUTHORITY=str(authority), CHOPINFER_DATA=str(self.root / 'data'))
        result = self.run_init(**env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.decode(), f'{authority / "key"}\n')
        self.assertEqual(result.stderr, b'')
        value = (authority / 'key').read_bytes()
        self.assertEqual(len(value), 32)
        self.assertFalse((self.root / 'data').exists())
        again = self.run_init(**env)
        self.assertEqual(again.returncode, 2)
        self.assertEqual(again.stdout, b'')
        self.assertIn(b'already exists', again.stderr)
        for output in (result.stdout + result.stderr, again.stdout + again.stderr):
            self.assertNotIn(value, output)
            self.assertNotIn(value.hex().encode(), output)
        self.assertEqual((authority / 'key').read_bytes(), value)

    def test_init_defaults_under_chopinfer_data(self):
        result = self.run_init(CHOPINFER_DATA=str(self.root / 'data'))
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.root / 'data/journal-proof/authority/key'
        self.assertEqual(result.stdout.decode(), f'{path}\n')
        self.assertEqual(len(path.read_bytes()), 32)

    def test_init_without_authority_fails(self):
        result = self.run_init()
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'set CHOPINFER_JOURNAL_AUTHORITY or CHOPINFER_DATA', result.stderr)


if __name__ == '__main__':
    unittest.main()
