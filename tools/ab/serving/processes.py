"""Linux process/session ownership and bounded commands. Never infer ownership from a GPU PID."""
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
import uuid

from common import Refusal, need, write

TOKEN = 'CHOPINFER_AB_OWNER'


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    group: int
    session: int
    start: int
    uid: int
    state: str

    @property
    def identity(self):
        return self.pid, self.start, self.uid

    @property
    def live(self):
        return self.state not in ('Z', 'X')


def parse_stat(pid, uid, raw):
    fields = raw.rsplit(')', 1)[1].split()
    return Process(pid, int(fields[1]), int(fields[2]), int(fields[3]),
                   int(fields[19]), uid, fields[0])


class Proc:
    def get(self, pid):
        directory = Path('/proc') / str(pid)
        try:
            return parse_stat(pid, directory.stat().st_uid, (directory / 'stat').read_text())
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, ValueError, IndexError) as error:
            raise Refusal(f'cannot establish process identity for {pid}: {error}') from error

    def scan(self):
        out = []
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():
                continue
            try:
                if directory.stat().st_uid != os.getuid():
                    continue
            except FileNotFoundError:
                continue
            process = self.get(int(directory.name))
            if process is not None:
                out.append(process)
        return out

    def tagged(self, process, token, nsys_session):
        try:
            root = Path('/proc') / str(process.pid)
            env = (root / 'environ').read_bytes().split(b'\0')
            if (TOKEN + '=' + token).encode() in env:
                return True
            if nsys_session:
                args = (root / 'cmdline').read_bytes().split(b'\0')
                tag = nsys_session.encode()
                return b'--start-agent' in args and any(
                    item == b'--session-name=' + tag or
                    (item == b'--session-name' and i + 1 < len(args) and args[i + 1] == tag)
                    for i, item in enumerate(args))
        except (FileNotFoundError, ProcessLookupError):
            return False
        except PermissionError:
            # A foreign process can restrict its environment. It is never
            # considered ours by a failed marker read.
            return False
        return False

    def send(self, process, sig):
        current = self.get(process.pid)
        if current is None or current.identity != process.identity:
            return False
        try:
            fd = os.pidfd_open(process.pid)
        except ProcessLookupError:
            return False
        try:
            current = self.get(process.pid)
            if current is None or current.identity != process.identity:
                return False
            signal.pidfd_send_signal(fd, sig)
            return True
        except ProcessLookupError:
            return False
        finally:
            os.close(fd)


def capabilities():
    need(os.name == 'posix' and Path('/proc/self/stat').exists(), 'the serving runner needs Linux /proc')
    need(all(hasattr(os, name) for name in ('pidfd_open', 'waitid', 'WNOWAIT', 'WEXITED'))
         and hasattr(signal, 'pidfd_send_signal'), 'the serving runner needs pidfds and non-reaping waitid')


class Owned:
    def __init__(self, record=None, *, proc=None, token=None, nsys_session=None):
        self.proc = proc or Proc()
        self.token = token or uuid.uuid4().hex
        self.nsys_session = nsys_session
        self.record = Path(record) if record else None
        self.process = None
        self.leader = None
        self.pidfd = None
        self.known = {}
        self.anchors = {}
        self.retired_anchors = set()
        self.state = 'not-started'
        self.reaped = False

    def save(self):
        if self.record:
            write(self.record, dict(state=self.state, token=self.token, nsys_session=self.nsys_session,
                pid=self.process.pid if self.process else None,
                pgid=self.process.pid if self.process else None,
                sid=self.process.pid if self.process else None,
                leader=asdict(self.leader) if self.leader else None,
                members=[asdict(p) for p in self.known.values()], anchors=[asdict(p) for p in self.anchors.values()],
                retired_anchors=[list(x) for x in self.retired_anchors], reaped=self.reaped), 0o600)

    def spawn(self, argv, *, stdout, stderr=None, env=None, cwd=None):
        need(self.process is None, 'cannot replace an owned launch')
        environment = dict(os.environ if env is None else env)
        environment[TOKEN] = self.token
        self.state = 'starting'
        # Assignment precedes identity discovery. Cancellation handlers only
        # set a flag, so they cannot abandon the child between these steps.
        self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr if stderr is not None else stdout, env=environment, cwd=cwd,
            start_new_session=True)
        self.pidfd = os.pidfd_open(self.process.pid)
        self.save()
        self.leader = self.proc.get(self.process.pid)
        need(self.leader is not None and self.leader.session == self.process.pid
             and self.leader.group == self.process.pid, 'spawn did not establish its own session')
        self.known[self.leader.identity] = self.leader
        self.save()

    def exited(self):
        if self.process is None:
            return True
        if self.reaped:
            return True
        # Popen.poll() would reap the leader and permit its SID to be reused.
        return os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def wait(self, seconds):
        """Sleeps up to `seconds`, waking as soon as the leader exits."""
        if self.pidfd is None or self.reaped:
            time.sleep(seconds)
            return
        poller = select.poll()
        poller.register(self.pidfd, select.POLLIN)
        poller.poll(seconds * 1000)

    def members(self):
        if self.process is None:
            return []
        processes = self.proc.scan()
        identities = set(self.known)
        pinned = not self.reaped
        observed = {p.identity: p for p in processes}
        for identity in self.anchors.keys() - self.retired_anchors:
            need(identity in observed, 'target session anchor vanished before owned cleanup')
        sessions = {p.session for identity, p in self.anchors.items()
                    if identity not in self.retired_anchors and identity in observed}
        owned = {p.pid: p for p in processes if
                 (pinned and p.session == self.process.pid) or p.session in sessions or p.identity in identities
                 or self.proc.tagged(p, self.token, self.nsys_session)}
        # Track descendants before a profiler reparents them. The target
        # wrapper also supplies a marker and an identity handshake before exec.
        while True:
            children = {p.pid: p for p in processes if p.parent in owned and p.pid not in owned}
            if not children:
                break
            owned.update(children)
        for process in owned.values():
            self.known[process.identity] = process
        self.save()
        return list(owned.values())

    def terminate(self, term_seconds, kill_seconds, *, deadline=None):
        if self.process is None:
            return
        self.state = 'stopping'
        self.save()
        error = None
        # The retained pidfd remains a safe way to stop a child even when
        # /proc identity discovery failed before readiness.
        if self.pidfd is not None and not self.reaped:
            try:
                signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for sig, seconds in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
            end = time.monotonic() + seconds
            if deadline is not None:
                end = min(end, deadline)
            sent = set()
            while True:
                try:
                    live = [p for p in self.members() if p.live]
                    error = None
                except (OSError, Refusal) as failure:
                    error = failure
                    live = []
                    # A failed scan cannot prove release, but known birth
                    # identities remain safe targets for best-effort cleanup.
                    for known in self.known.values():
                        try:
                            current = self.proc.get(known.pid)
                        except (OSError, Refusal):
                            continue
                        if current is not None and current.identity == known.identity and current.live:
                            live.append(current)
                if live == [] and error is None:
                    self.state = 'release-pending'
                    self.save()
                    return
                others = [p for p in live or [] if p.identity not in self.anchors]
                for process in live or []:
                    if process.identity in self.anchors:
                        if sig == signal.SIGTERM or others or error is not None:
                            continue
                        self.retired_anchors.add(process.identity)
                    if process.identity not in sent:
                        self.proc.send(process, sig)
                        sent.add(process.identity)
                if (sig == signal.SIGKILL and self.pidfd is not None and not self.reaped
                    and (self.leader is None or self.leader.identity not in self.anchors or (not others and error is None))):
                    try:
                        signal.pidfd_send_signal(self.pidfd, sig)
                    except ProcessLookupError:
                        pass
                # Anchors take no SIGTERM: once only they remain, the KILL
                # round retires them without waiting out the TERM window.
                if sig == signal.SIGTERM and not others and error is None:
                    break
                if time.monotonic() >= end:
                    break
                time.sleep(min(0.05, max(0, end - time.monotonic())))
        self.state = 'cleanup-failed'
        self.save()
        raise Refusal(f'owned session {self.process.pid} release unproved: {error or "process survived KILL"}')

    def reap(self, timeout=1.0):
        need(not [p for p in self.members() if p.live], 'cannot reap while owned processes survive')
        if self.process is not None and not self.reaped:
            self.process.wait(timeout=max(0.001, timeout))
            self.reaped = True
            if self.pidfd is not None:
                os.close(self.pidfd)
                self.pidfd = None
        self.state = 'released'
        self.save()


class Commands:
    """Regular output files avoid descendant-held pipe EOF hangs."""
    def __init__(self, term_seconds=0.2, kill_seconds=1.0, records=None):
        self.term_seconds = term_seconds
        self.kill_seconds = kill_seconds
        self.pending = []
        self.records = Path(records) if records else None
        if self.records:
            self.records.mkdir(exist_ok=True)

    def run(self, argv, timeout, *, env=None, stdout_path=None, check=None, deadline=None):
        need(timeout > 0, 'command deadline exhausted before spawn')
        end = time.monotonic() + timeout
        if deadline is not None:
            end = min(end, deadline)
        need(time.monotonic() < end, 'command absolute deadline exhausted before spawn')
        owned = Owned(self.records / (uuid.uuid4().hex + '.json') if self.records else None)
        self.pending.append(owned)
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            error = None
            try:
                owned.spawn(argv, stdout=out, stderr=err, env=env)
                while not owned.exited():
                    need(os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size <= 16 * 1024 * 1024, 'command output exceeds limit')
                    if check:
                        check()
                    if time.monotonic() >= end:
                        raise Refusal(f'command timed out: {argv[0]}')
                    owned.wait(min(0.05, max(0, end - time.monotonic())))
                need(time.monotonic() <= end, 'command completed after its deadline')
            except BaseException as failure:
                error = failure
            finally:
                # This also removes children left by a normally exited leader.
                owned.terminate(self.term_seconds, self.kill_seconds, deadline=deadline)
                owned.reap(timeout=min(1.0, max(.001, deadline - time.monotonic())) if deadline is not None else 1.0)
                self.pending.remove(owned)
                out.seek(0); err.seek(0)
                stdout, stderr = out.read(16 * 1024 * 1024 + 1), err.read(16 * 1024 * 1024 + 1)
                need(len(stdout) + len(stderr) <= 16 * 1024 * 1024, 'command output exceeds limit')
                if stdout_path:
                    Path(stdout_path).write_bytes(stdout + stderr)
            if error:
                raise error
            return subprocess.CompletedProcess(argv, owned.process.returncode, stdout, stderr)

    def cleanup(self, deadline=None):
        failures = []
        for owned in list(self.pending):
            try:
                owned.terminate(self.term_seconds, self.kill_seconds, deadline=deadline)
                owned.reap(timeout=min(1.0, max(.001, deadline - time.monotonic())) if deadline is not None else 1.0)
                self.pending.remove(owned)
            except Exception as error:
                failures.append(str(error))
        need(not failures, '; '.join(failures))
