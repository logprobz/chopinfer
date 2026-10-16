"""Local helper authority. Capture files cannot appoint their own signer.

Sealing never creates the key. Create it once, explicitly:
    python3 tools/journal_proof/provenance.py init
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
from common import Invalid, need

INIT='python3 tools/journal_proof/provenance.py init'


def authority():
    """The directory holding the local helper key: CHOPINFER_JOURNAL_AUTHORITY,
    else journal-proof/authority under CHOPINFER_DATA. It lives outside the
    repository; a capture never names its own signer."""
    configured=os.environ.get('CHOPINFER_JOURNAL_AUTHORITY')
    if configured:
        return Path(configured)
    data=os.environ.get('CHOPINFER_DATA')
    return Path(data)/'journal-proof/authority' if data else None


AUTHORITY=authority()
HELPER_FILES=('allocation_args.py','auxiliary.py','auxproof.py','capture.py','common.py','engine.py','history.py','instrument.py','layout.py',
              'pins.py','proof.py','provenance.py','replacements.py','text.py','trace_runtime.rs')
# The producer of each sealed index schema: the external helper for v2, this
# tool for v3.
PRODUCERS={'chopinfer.journal-captures.v2':'capture_revision.py','chopinfer.journal-captures.v3':'tools/journal_proof/capture.py'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper_identity(root):
    files={name:digest(Path(root)/name) for name in HELPER_FILES}
    return hashlib.sha256(canonical(files)).hexdigest(),files


def init():
    """Create the authority directory (0700) and a fresh 32-byte key (0600).
    Refuses an existing key and a symlinked authority directory or key
    (parent directories are followed). Returns the key path; the key bytes
    are never printed. On any failure after the key file is created, that
    file is removed (see discard), so a retry can succeed; any other file
    at the key path is never removed."""
    need(AUTHORITY is not None,'no helper authority: set CHOPINFER_JOURNAL_AUTHORITY or CHOPINFER_DATA')
    path=AUTHORITY/'key'
    need(not AUTHORITY.is_symlink(),f'helper authority {AUTHORITY} is a symlink')
    need(not path.exists() and not path.is_symlink(),f'helper authority key {path} already exists; init never replaces it')
    value=secrets.token_bytes(32)
    AUTHORITY.mkdir(mode=0o700,parents=True,exist_ok=True)
    directory=os.open(AUTHORITY,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fchmod(directory,0o700)
        try:
            fd=os.open('key',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        except FileExistsError:
            raise Invalid(f'helper authority key {path} already exists; init never replaces it') from None
        created=os.fstat(fd)
        try:
            try:
                os.fchmod(fd,0o600)
                need(os.write(fd,value)==len(value),f'short write to helper authority key {path}')
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(directory)
            opened=os.fstat(directory)
            try:
                current=os.lstat(AUTHORITY)
            except FileNotFoundError:
                current=None
            need(current is not None and same(current,opened),
                 f'helper authority {AUTHORITY} was replaced or removed during init; no key was created')
        except BaseException as error:
            left=discard(directory,created)
            if left and isinstance(error,Exception):
                raise Invalid(f'{error}; {left}') from error
            raise
    finally:
        os.close(directory)
    return path


def same(a,b):
    return (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)


def discard(directory,created):
    """Remove the key file init created, and only that file. Returns None when
    it is gone, else what was left in place. The candidate is first renamed to
    a unique name and re-checked there, so a file swapped in under `key`
    between the check and the rename is put back, never unlinked. A file
    swapped in under the unique name itself between its check and the unlink
    would still be removed; nothing else uses that name."""
    path=AUTHORITY/'key'
    try:
        found=os.stat('key',dir_fd=directory,follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not same(found,created):
        return f'{path} is not the file init created; it was left in place'
    hold=f'.key-discard-{secrets.token_hex(8)}'
    try:
        os.rename('key',hold,src_dir_fd=directory,dst_dir_fd=directory)
        if same(os.stat(hold,dir_fd=directory,follow_symlinks=False),created):
            os.unlink(hold,dir_fd=directory)
            return None
    except FileNotFoundError:
        return None
    except OSError as error:
        return f'could not remove the partial key {path}: {error}'
    try:
        os.link(hold,'key',src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
        os.unlink(hold,dir_fd=directory)
    except OSError:
        return f'a file that replaced {path} during cleanup was left at {AUTHORITY/hold}'
    return f'{path} is not the file init created; it was left in place'


def key():
    need(AUTHORITY is not None,'no helper authority: set CHOPINFER_JOURNAL_AUTHORITY or CHOPINFER_DATA')
    path=AUTHORITY/'key'
    need(path.is_file() and not path.is_symlink() and path.stat().st_mode&0o077==0,'trusted helper authority is unavailable')
    value=path.read_bytes()
    need(len(value)==32,'invalid helper authority key')
    return value


def sealing_key():
    """The key seal() signs with. Refuses, naming the configured key path and
    the init command, when the key is missing; never creates one."""
    need(AUTHORITY is not None,'no helper authority: set CHOPINFER_JOURNAL_AUTHORITY or CHOPINFER_DATA')
    path=AUTHORITY/'key'
    need(path.exists() or path.is_symlink(),
         f'no helper authority key at {path}; sealing never creates one. Create it once with: {INIT}')
    return key()


def seal(payload):
    value=dict(payload)
    value['attestation']=hmac.new(sealing_key(),canonical(payload),hashlib.sha256).hexdigest()
    return value


def verify(value):
    need(isinstance(value,dict) and isinstance(value.get('attestation'),str),'auxiliary capture has no helper attestation')
    payload={k:v for k,v in value.items() if k!='attestation'}
    signature=hmac.new(key(),canonical(payload),hashlib.sha256).hexdigest()
    need(hmac.compare_digest(signature,value['attestation']),'auxiliary capture was not sealed by the trusted helper or was modified')
    need(payload.get('producer')==PRODUCERS.get(payload.get('schema')) and payload.get('real_build') is True,
         'auxiliary capture was not produced by a real helper build')
    need(payload.get('helper_digest')==hashlib.sha256(canonical(payload['helper_files'])).hexdigest(),'helper digest mismatch')
    return payload


def main(argv=None):
    p=argparse.ArgumentParser(description='Create the local helper authority key under $CHOPINFER_JOURNAL_AUTHORITY '
                                          '(default $CHOPINFER_DATA/journal-proof/authority). Sealing never creates one.')
    p.add_argument('command',choices=['init'])
    p.parse_args(argv)
    try:
        path=init()
    except (OSError,Invalid) as error:
        print(f'INIT FAILED: {error}',file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__=='__main__':
    sys.exit(main())
