"""Private adapter for Qwen's recorded CPU lifecycle, not an SDK schema.

The model supplies labels and decoded row context. No serving logs, CUDA
observations or numerical results are inferred here. Public descriptors belong
to tools/model_sdk when that independent surface lands.
"""
from contextlib import contextmanager
from collections import deque
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import shlex
import stat
import struct

EXTENSION = 'qwen38.lifecycle.v1'
METADATA_LIMIT = 16 << 20
DATA_LIMIT = 64 << 20
ACTION_LIMIT = 64
SHA256 = re.compile(r'[0-9a-f]{64}')
KINDS = ['create', 'execute', 'commit', 'execute', 'commit', 'execute', 'cancel', 'retire', 'destroy']


class Incompatible(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


@contextmanager
def opened(path, limit):
    """No-follow descriptor walk, regular files only, bounded before and during IO."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'lifecycle paths must be absolute and external')
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            try:
                os.stat('.git', dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError('lifecycle artifacts must be outside repositories')
        file = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(file, 'rb') as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'file exceeds lifecycle budget or is not regular')
            yield stream
    finally:
        os.close(descriptor)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read_json(path):
    with opened(path, METADATA_LIMIT) as stream:
        raw = stream.read(METADATA_LIMIT + 1)
    require(len(raw) <= METADATA_LIMIT, 'metadata budget exceeded')
    return decode(raw)


def digest(path, limit=METADATA_LIMIT):
    size = 0
    sha = hashlib.sha256()
    with opened(path, limit) as stream:
        while chunk := stream.read(1 << 20):
            size += len(chunk)
            require(size <= limit, 'artifact grew beyond lifecycle budget')
            sha.update(chunk)
    return dict(bytes=size, sha256=sha.hexdigest())


def lines(path):
    total = 0
    with opened(path, DATA_LIMIT) as stream:
        while line := stream.readline(METADATA_LIMIT + 1):
            total += len(line)
            require(len(line) <= METADATA_LIMIT and total <= DATA_LIMIT, 'record exceeds lifecycle budget')
            require(line.endswith(b'\n'), 'unterminated lifecycle record')
            yield line


class Wire:
    """Read the bounded V4/V3 subset using the SDK's existing byte layout."""
    def __init__(self, data):
        require(isinstance(data, list) and len(data) <= METADATA_LIMIT
                and all(type(n) is int and 0 <= n <= 255 for n in data), 'invalid batch bytes')
        self.data = bytes(data)
        self.position = 0

    def take(self, size):
        require(0 <= size <= len(self.data) - self.position, 'truncated batch')
        value = self.data[self.position:self.position + size]
        self.position += size
        return value

    def number(self, form):
        return struct.unpack('<' + form, self.take(struct.calcsize('<' + form)))[0]


def wire_rows(data):
    wire = Wire(data)
    require(wire.take(12) == b'chopbatch-v4', 'unsupported batch wire')
    enclosed = wire.number('I')
    end = wire.position + enclosed
    require(wire.take(12) == b'chopbatch-v3', 'unsupported enclosed batch')
    count = wire.number('I')
    require(0 < count <= 32, 'unsupported live rows')
    rows = []
    for _ in range(count):
        request, phase, tokens, flags, inputs = (wire.number(f) for f in ('Q', 'B', 'I', 'B', 'I'))
        require(phase in (0, 1) and flags == 1 and tokens == inputs and inputs > 0, 'unsupported batch item')
        wire.take(inputs * 4)
        rows.append(dict(request=request, phase=['prefill', 'decode'][phase], tokens=tokens))
    require(wire.number('I') == count, 'incomplete appends')
    for row in rows:
        context, additional, new, pages = (wire.number(f) for f in ('Q', 'Q', 'Q', 'I'))
        require(pages <= 1024 and additional == row['tokens'] and new == context + additional, 'invalid append')
        row.update(context=context, additional=additional, end=new, pages=[wire.number('Q') for _ in range(pages)])
        require(wire.number('B') == 0, 'copy-on-write replay is unsupported')
    require(wire.take(16) == bytes(16) and wire.position == end, 'media or trailing batch data')
    require(wire.number('I') == count, 'incomplete slot mapping')
    for row in rows:
        row['slot'] = wire.number('Q')
    require(wire.position == len(wire.data), 'trailing wire data')
    return rows


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        value = self.value = read_json(self.root / 'diagnostic.json')
        require(value['format'] == 'chopinfer.diagnostic.v1' and value['mode'] == 'journal'
                and value['extension'] == EXTENSION and value['backend'] == 'qwen38-fake-host', 'unsupported lifecycle format/backend')
        artifacts = value['artifacts']
        require(set(artifacts) == {'input.json', 'journal.txt', 'observations.jsonl'}, 'incomplete or unknown lifecycle artifacts')
        require(all(type(a['bytes']) is int and a['bytes'] > 0 and isinstance(a['sha256'], str)
                    and SHA256.fullmatch(a['sha256']) for a in artifacts.values()), 'invalid artifact descriptor')
        require(sum(artifacts[n]['bytes'] for n in ('journal.txt', 'observations.jsonl')) <= DATA_LIMIT,
                'combined lifecycle data budget exceeded')
        for name, expected in artifacts.items():
            actual = digest(self.root / name, METADATA_LIMIT if name == 'input.json' else DATA_LIMIT)
            require(equal(actual, expected), 'stale, truncated or trailing artifact ' + name)
        inp = self.input = read_json(self.root / 'input.json')
        require(inp['extension'] == EXTENSION and inp['backend'] == value['backend'], 'unsupported input format')
        require(equal(value['identity'], dict(input_sha256=artifacts['input.json']['sha256'], settings=inp['settings'])),
                'input identity disagrees with capture')
        require(equal(inp['producer'], value['producer']) and all(isinstance(h, str) and SHA256.fullmatch(h)
                for h in value['producer'].values()) and set(value['producer']) == {'binary_sha256', 'kernel_lock_sha256'},
                'unbound producer identity')
        require(value['declared_differences'] == [], 'unsupported declared producer/configuration difference')
        actions = self.actions = inp['actions']
        require(isinstance(actions, list) and len(actions) <= ACTION_LIMIT, 'action budget exceeded')
        require([a['kind'] for a in actions] == KINDS and [a['id'] for a in actions] == list(range(9)),
                'unsupported or reordered lifecycle actions')
        require(all(type(a['id']) is int for a in actions), 'invalid action ID')
        require(isinstance(inp['configuration'], list) and inp['configuration'] and
                all(type(n) is int and 0 <= n <= 255 for n in inp['configuration']), 'invalid encoded configuration')
        require(actions[7] == dict(id=7, kind='retire', request=22), 'unsupported retirement')
        workload = []
        for action in actions:
            kind = action['kind']
            require(set(action) == ({'id', 'kind', 'name', 'wire'} if kind == 'execute' else
                                   {'id', 'kind', 'request'} if kind == 'retire' else {'id', 'kind'}), 'unknown action payload')
            if kind == 'execute':
                workload.append(dict(action=action['id'], name=action['name'], rows=wire_rows(action['wire'])))
        require(equal(value['workload'], workload), 'decoded wire and workload observations disagree')
        require(value['expected_actions'] == len(actions), 'missing action coverage')
        require(value['execution_result'] == ('passed' if value['completion'] == 'full' else 'failed'), 'unbound execution result')
        require(value['arm'] in ('baseline', 'control', 'candidate') and value['fault'] in ('none', 'restore-copy-rank-1'), 'unknown arm/fault')
        require(value['arm'] == 'candidate' or value['fault'] == 'none', 'baseline/control cannot inject a fault')
        require(equal(value['device'], dict(execution='not_run', reason='CPU fake host; idle-device lifecycle evidence is missing')),
                'CPU journal cannot establish device evidence')
        require(value['reproduction'] and all(isinstance(x, str) and x for x in value['reproduction']), 'missing reproduction command')
        # Validate the entire recording, including the tail after the first difference.
        for _ in self.events():
            pass

    def events(self):
        value = self.value
        action = -1
        active = False
        count = journal_count = 0
        copy_pending = None
        failed_copy = False
        last = None
        journal_hash, observations_hash = hashlib.sha256(), hashlib.sha256()
        journal_bytes = observation_bytes = 0
        with opened(self.root / 'journal.txt', DATA_LIMIT) as journal:
            for raw in lines(self.root / 'observations.jsonl'):
                observations_hash.update(raw)
                observation_bytes += len(raw)
                event = decode(raw)
                require(event['ordinal'] == count and type(event['ordinal']) is int, 'missing or reordered observation')
                count += 1
                kind = event['kind']
                if kind == 'begin':
                    require(not active and copy_pending is None, 'missing lifecycle result')
                    action += 1
                    active = True
                require(active and event['action'] == action and action < len(self.actions), 'missing or reordered action boundary')
                if copy_pending is not None:
                    require(kind == 'copy_result', 'missing copy service result')
                if kind == 'journal':
                    require(event['line'] == journal_count, 'missing or reordered journal reference')
                    raw_line = journal.readline(METADATA_LIMIT + 1)
                    require(raw_line.endswith(b'\n') and len(raw_line) <= METADATA_LIMIT, 'truncated journal record')
                    journal_bytes += len(raw_line)
                    require(journal_bytes + observation_bytes <= DATA_LIMIT, 'combined lifecycle data budget exceeded')
                    journal_hash.update(raw_line)
                    text = raw_line[:-1].decode('utf-8')
                    require(event['service'] == text.split(' ', 1)[0], 'service label differs from journal')
                    event = dict(event, text=text)
                    if event['service'] == 'copy':
                        copy_pending = (journal_count, text)
                    journal_count += 1
                elif kind == 'copy_result':
                    require(copy_pending is not None and copy_pending[0] == event['line'], 'unjoined copy result')
                    require(event['status'] in ('Success', 'HostError'), 'unknown copy status')
                    for span in ('source', 'destination'):
                        require(set(event[span]) == {'buffer', 'generation', 'offset', 'length'} and
                                all(type(n) is int and n >= 0 for n in event[span].values()), 'missing copy spans')
                    spans = re.findall(r'raw: (\d+) \}, offset: (\d+), length: (\d+)', copy_pending[1])
                    require(len(spans) == 2, 'missing copy arguments in journal')
                    for key, (handle, offset, length) in zip(('source', 'destination'), spans):
                        span = event[key]
                        require((int(handle), int(offset), int(length)) ==
                                ((span['generation'] << 32) | span['buffer'], span['offset'], span['length']),
                                'copy result disagrees with executed arguments')
                    require(event['rank'] is None or type(event['rank']) is int and 0 <= event['rank'] < 4, 'invalid model rank probe')
                    copy_pending = None
                    if event['status'] == 'HostError':
                        require(not failed_copy and action == 6 and event['rank'] == 1 and value['fault'] == 'restore-copy-rank-1',
                                'unsupported failure prefix')
                        failed_copy = True
                elif kind == 'result':
                    require(event['probe'] == 'qwen38.slot-transaction' and
                            all(key in event for key in ('pending', 'owners', 'poisoned', 'captures', 'replays', 'acquired', 'released')),
                            'missing required state probe')
                    require(type(event['poisoned']) is bool and (event['pending'] is None or type(event['pending']) is bool), 'invalid transaction probe')
                    require(isinstance(event['owners'], list) and len(event['owners']) == 3, 'missing slot owners')
                    require(all(type(event[k]) is int and event[k] >= 0 for k in ('captures', 'replays', 'acquired', 'released')), 'invalid resource probe')
                    require(event['status'] == ('HostError' if failed_copy else 'Success'), 'unclosed service failure')
                    if failed_copy:
                        require(event['pending'] is True and event['poisoned'] is True, 'missing failed transaction state')
                    last = event
                    active = False
                else:
                    require(kind == 'begin', 'unknown observation')
                yield event
            require(not journal.read(1), 'unreferenced trailing journal data')
        for name, size, sha in (('journal.txt', journal_bytes, journal_hash),
                                ('observations.jsonl', observation_bytes, observations_hash)):
            require(equal(dict(bytes=size, sha256=sha.hexdigest()), value['artifacts'][name]),
                    'artifact changed during lifecycle verification')
        require(not active and copy_pending is None and last is not None, 'interrupted lifecycle capture')
        require(count == value['observations'] and journal_count == value['events'], 'incomplete observation coverage')
        require(action + 1 == value['completed_actions'], 'incomplete action results')
        teardown = value['teardown']
        require((teardown['acquired'], teardown['released']) == (last['acquired'], last['released']), 'unbound teardown counters')
        if failed_copy:
            require(value['completion'] == 'closed-failure' and value['completed_actions'] == 7 and value['failed_action'] == 6
                    and teardown['status'] == 'Poisoned' and teardown['acquired'] > teardown['released'], 'incomplete failure publication')
        else:
            require(value['completion'] == 'full' and value['completed_actions'] == 9 and value['failed_action'] is None
                    and teardown['status'] == 'Success' and teardown['acquired'] == teardown['released']
                    and last['pending'] is None and last['owners'] == [None]*3 and not last['poisoned'], 'incomplete lifecycle cleanup')


def compatibility(a, b):
    for key in ('identity', 'producer'):
        if not equal(a.value[key], b.value[key]):
            raise Incompatible('incompatible lifecycle ' + key)
    if not equal(a.input, b.input):
        raise Incompatible('incompatible lifecycle inputs/configuration')


def bound(binding, root):
    require(binding is not None and Path(binding['path']) == root
            and binding['sha256'] == digest(root / 'diagnostic.json')['sha256'], 'missing or stale baseline/control binding')


def first_difference(a, b):
    first = None
    graph = {'expected': dict(captures=0, replays=0), 'observed': dict(captures=0, replays=0)}
    context = deque(maxlen=2)
    for expected, observed in zip_longest(a.events(), b.events()):
        for side, event in (('expected', expected), ('observed', observed)):
            if event and event['kind'] == 'result':
                graph[side] = {key: event[key] for key in ('captures', 'replays')}
            elif event and event['kind'] == 'journal' and event['service'] in ('capture', 'captured', 'replay'):
                graph[side]['last_service'] = event['text']
        if not equal(expected, observed) and first is None:
            event = observed or expected
            action = event['action']
            preceding = [s for s in b.value['workload'] if s['action'] <= action]
            step = preceding[-1] if preceding else None
            first = dict(ordinal=event['ordinal'], action=action, operation=b.actions[action]['kind'],
                         stage=event['kind'], step=step, expected=expected, observed=observed,
                         service='copy_device_to_device' if event['kind'] == 'copy_result' else event.get('service'),
                         live_rows=len(step['rows']) if step else 0,
                         graph=json.loads(json.dumps(graph)), context=list(context),
                         configured_capacity=b.input['settings']['served_capacity'],
                         effective_hc=b.input['settings']['effective_hc'])
        if observed:
            context.append(observed)
    return first


def compare(reference, candidate):
    report = dict(format='chopinfer.diagnostic.v1', mode='journal', extension=EXTENSION,
                  status='insufficient', evidence='unavailable')
    try:
        a, b = Capture(reference), Capture(candidate)
        compatibility(a, b)
        require(a.value['arm'] == 'baseline' and a.value['completion'] == 'full' and a.value['fault'] == 'none'
                and a.value['reference'] is None and a.value['control'] is None, 'reference is not a complete baseline')
        require(b.value['arm'] in ('control', 'candidate'), 'candidate must bind its baseline')
        bound(b.value['reference'], a.root)
        if b.value['arm'] == 'candidate':
            require(b.value['control'] is not None, 'missing unchanged baseline repeat')
            control_root = Path(b.value['control']['path'])
            bound(b.value['control'], control_root)
            control = Capture(control_root)
            compatibility(a, control)
            require(control.value['arm'] == 'control' and control.value['control'] is None and control.value['fault'] == 'none'
                    and control.value['completion'] == 'full', 'invalid unchanged baseline repeat')
            bound(control.value['reference'], a.root)
            require(first_difference(a, control) is None, 'baseline repeat changes the baseline')
        else:
            require(b.value['control'] is None, 'control cannot bind another control')
        first = first_difference(a, b)
        require(first is not None or b.value['completion'] == 'full', 'incomplete matching lifecycle')
        report.update(status='diverged' if first else 'match', evidence='cpu-structural-state', first=first,
                      device=b.value['device'], completed_actions=b.value['completed_actions'],
                      expected_actions=b.value['expected_actions'], completion=b.value['completion'],
                      reproduction=b.value['reproduction'], command=shlex.join(b.value['reproduction']),
                      teardown=b.value['teardown'])
    except Incompatible as error:
        report.update(status='incompatible', reason=str(error))
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, struct.error) as error:
        report.update(status='insufficient', reason=str(error))
    return report
