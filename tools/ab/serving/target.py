#!/usr/bin/env python3
"""Keep a profiler target session anchored until the supervisor releases it."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import read, write, need
from processes import Proc, TOKEN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--token', required=True)
    p.add_argument('--timeout', type=float, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    need(command, 'target command missing')
    os.environ[TOKEN] = a.token
    if os.getsid(0) != os.getpid():
        os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    process = Proc().get(os.getpid())
    write(a.record, dict(pid=process.pid, start=process.start, uid=process.uid,
                        pgid=process.group, sid=process.session, token=a.token), 0o600)
    end = time.monotonic() + a.timeout
    while time.monotonic() < end:
        if Path(a.release).exists():
            need(read(a.release) == {'token': a.token}, 'target release token mismatch')
            # Reset signal dispositions in the exec'd child without using a
            # preexec_fn in the multithreaded serving process.
            argv = [sys.executable, '-c',
                    'import os,signal,sys; signal.signal(signal.SIGTERM,signal.SIG_DFL); '
                    'signal.signal(signal.SIGINT,signal.SIG_DFL); os.execvpe(sys.argv[1],sys.argv[1:],os.environ)',
                    *command]
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, env=os.environ)
            identity = Proc().get(child.pid)
            need(identity is not None, 'server identity missing immediately after spawn')
            write(Path(a.record).with_name('server.json'), dict(pid=identity.pid, start=identity.start,
                  uid=identity.uid, sid=identity.session, pgid=identity.group, token=a.token), 0o600)
            # Never reap the child or exit on its behalf. The supervisor checks
            # its birth identity and kills this anchor only after other members.
            while True:
                time.sleep(1)
        time.sleep(0.02)
    raise RuntimeError('target identity was not acknowledged before its deadline')


if __name__ == '__main__':
    main()
