"""Model-neutral offline probe metadata, bounded reads and immutable seals.

Models supply probe inventories, operand dependencies and numerical contracts.
No numerical pass/fail policy or model layout conversion belongs here.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

SCHEMA = 'chopinfer.oracle-attachment.v1'
FORMAT = 'chopinfer.diagnostic.v1'
MAX_ENTRIES = 65536
MAX_BYTES = 8 << 30
METADATA_BYTES = 16 << 20
WIDTH = {'BF16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'U32': 4, 'U8': 1, 'BOOL': 1}


class Incompatible(ValueError):
    """Two observed computations cannot support the requested comparison."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def reject_constant(value):
    raise ValueError('nonfinite JSON constant ' + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key ' + key)
        result[key] = value
    return result


def decode_json(data):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def external(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('diagnostic paths must be absolute and external')
    for part in (path, *path.parents):
        if part.is_symlink() or (part / '.git').exists():
            raise ValueError('diagnostic path enters a repository or symlink')
    return path


def stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def opened(path):
    """Walk with retained directory descriptors and refuse all symbolic links."""
    path = external(path)
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('diagnostic artifact is not a regular file')
            yield stream, before.st_size
            if (stamp(before) != stamp(os.fstat(stream.fileno())) or
                    stamp(before) != stamp(os.stat(path.name, dir_fd=parent, follow_symlinks=False))):
                raise ValueError('diagnostic artifact changed during read')
    finally:
        os.close(parent)


def read_bytes(path, limit=METADATA_BYTES):
    with opened(path) as (stream, size):
        if size > limit:
            raise ValueError('artifact exceeds declared byte budget')
        data = stream.read(limit + 1)
        if len(data) != size:
            raise ValueError('truncated or growing artifact')
        return data


def read_json(path):
    return decode_json(read_bytes(path))


def identity(path, limit=MAX_BYTES):
    with opened(path) as (stream, size):
        if size > limit:
            raise ValueError('artifact exceeds declared byte budget')
        return {'bytes': size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def write_new(path, value):
    path = external(path)
    data = canonical(value) + b'\n'
    if len(data) > METADATA_BYTES:
        raise ValueError('metadata exceeds budget')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def component(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('artifact must be a single relative component')
    return name


def elements(shape):
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 8 or
            any(type(d) is not int or d <= 0 for d in shape)):
        raise ValueError('invalid nonempty tensor shape')
    count = math.prod(shape)
    if count > MAX_BYTES:
        raise ValueError('tensor shape exceeds budget')
    return count


def named(entries, label):
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        raise ValueError('invalid ' + label + ' inventory')
    out = {}
    for entry in entries:
        name = entry.get('name')
        if not isinstance(name, str) or not name or name in out:
            raise ValueError('missing or duplicate ' + label + ' name')
        out[name] = entry
    return out


def validate_scope(scope):
    if (not scope.get('model') or not scope.get('contract') or
            scope.get('evidence') not in ('chained', 'op-isolated')):
        raise ValueError('missing model/evidence/contract scope')
    probes = named(scope['probes'], 'expected probe')
    if not probes:
        raise ValueError('zero requested probes')
    for probe in probes.values():
        elements(probe['shape'])
    inputs = scope['inputs']
    rows = inputs['rows']
    if (not isinstance(rows, list) or not 0 < len(rows) <= MAX_ENTRIES or
            not isinstance(inputs.get('state'), dict) or not inputs['state'].get('kind')):
        raise ValueError('missing exact input rows/state')
    if len({canonical(row) for row in rows}) != len(rows):
        raise ValueError('duplicate logical input row')
    return probes


def tensor_descriptor(name, shape, dtype, file, offset=0, **metadata):
    value = dict(name=name, shape=shape, dtype=dtype, file=file, offset=offset,
                 bytes=elements(shape) * WIDTH[dtype], **metadata)
    validate_tensor(value)
    return value


def validate_tensor(tensor):
    component(tensor['file'])
    size = elements(tensor['shape']) * WIDTH[tensor['dtype']]
    if (type(tensor['offset']) is not int or tensor['offset'] < 0 or
            tensor['bytes'] != size or tensor['offset'] + size > MAX_BYTES):
        raise ValueError('tensor shape/dtype/span mismatch')


def inventory(root, description, observed, producer_identity):
    """Complete a producer that wrote scope.json before collecting tensors."""
    root = external(root)
    if read_json(root / 'scope.json') != description:
        raise ValueError('producer changed its declared scope')
    expected = validate_scope(description['scope'])
    tensors = named(observed, 'observed probe')
    missing = [name for name, probe in expected.items()
               if name not in tensors or probe['shape'] != tensors[name]['shape']]
    for ordinal, tensor in enumerate(observed):
        validate_tensor(tensor)
        tensor['ordinal'] = ordinal
    value = dict(schema=SCHEMA, description=description, identity=producer_identity,
                 tensors=observed, completion='incomplete' if missing else 'full', missing=missing)
    write_new(root / 'producer.json', value)
    return seal(root)


def seal(root):
    root = external(root)
    value = read_json(root / 'producer.json')
    description = read_json(root / 'scope.json')
    if value['schema'] != SCHEMA or value['description'] != description:
        raise ValueError('producer scope/completion binding differs')
    validate_scope(description['scope'])
    limits = description['limits']
    if (type(limits['tensor_bytes']) is not int or not 0 < limits['tensor_bytes'] <= MAX_BYTES or
            type(limits['metadata_bytes']) is not int or not 0 < limits['metadata_bytes'] <= METADATA_BYTES):
        raise ValueError('invalid capture budgets')
    artifacts = {name: identity(root / name, limits['metadata_bytes']) for name in ('scope.json', 'producer.json')}
    for tensor in value['tensors']:
        validate_tensor(tensor)
        name = tensor['file']
        if name in ('scope.json', 'producer.json', 'diagnostic.json'):
            raise ValueError('tensor aliases metadata')
        if name not in artifacts:
            artifacts[name] = identity(root / name, limits['tensor_bytes'])
        if tensor['offset'] + tensor['bytes'] > artifacts[name]['bytes']:
            raise ValueError('tensor span outside artifact')
    if sum(a['bytes'] for n, a in artifacts.items() if n not in ('scope.json', 'producer.json')) > limits['tensor_bytes']:
        raise ValueError('total tensor budget exhausted')
    result = dict(format=FORMAT, mode='oracle', attachment=SCHEMA, description=description,
                  identity=value['identity'], tensors=value['tensors'], artifacts=artifacts,
                  completion=value['completion'], missing=value['missing'],
                  scope_sha256=sha256(canonical(description['scope'])))
    write_new(root / 'diagnostic.json', result)
    return result


def load(root):
    root = external(root)
    value = read_json(root / 'diagnostic.json')
    if (value['format'], value['mode'], value['attachment']) != (FORMAT, 'oracle', SCHEMA):
        raise ValueError('unsupported oracle attachment')
    scope = value['description']['scope']
    probes = validate_scope(scope)
    if value['scope_sha256'] != sha256(canonical(scope)):
        raise ValueError('stale scope digest')
    expected_files = {'scope.json', 'producer.json'}
    tensors = named(value['tensors'], 'observed probe')
    for ordinal, tensor in enumerate(value['tensors']):
        validate_tensor(tensor)
        if tensor['ordinal'] != ordinal:
            raise ValueError('probe order is not contiguous')
        expected_files.add(tensor['file'])
    if set(value['artifacts']) != expected_files:
        raise ValueError('incomplete or unknown artifact inventory')
    limits = value['description']['limits']
    if not 0 < limits['tensor_bytes'] <= MAX_BYTES or not 0 < limits['metadata_bytes'] <= METADATA_BYTES:
        raise ValueError('unsupported capture budget')
    total = 0
    for name, recorded in value['artifacts'].items():
        component(name)
        bound = limits['metadata_bytes'] if name in ('scope.json', 'producer.json') else limits['tensor_bytes']
        if identity(root / name, bound) != recorded:
            raise ValueError('stale or truncated artifact ' + name)
        if name not in ('scope.json', 'producer.json'):
            total += recorded['bytes']
    if total > limits['tensor_bytes']:
        raise ValueError('total tensor budget exhausted')
    producer = read_json(root / 'producer.json')
    if (producer['description'] != value['description'] or read_json(root / 'scope.json') != value['description'] or
            any(producer[key] != value[key] for key in ('identity', 'tensors', 'completion', 'missing'))):
        raise ValueError('seal differs from producer completion')
    missing = [name for name, probe in probes.items()
               if name not in tensors or probe['shape'] != tensors[name]['shape']]
    if missing or value['completion'] != 'full' or value['missing']:
        raise ValueError('incomplete required probe coverage: ' + ', '.join(missing or value['missing']))
    for tensor in tensors.values():
        if tensor['offset'] + tensor['bytes'] > value['artifacts'][tensor['file']]['bytes']:
            raise ValueError('tensor span outside artifact')
    if not value['description'].get('reproduction'):
        raise ValueError('missing producer reproduction command')
    return value


def raw_tensor(root, tensor, expected=None):
    validate_tensor(tensor)
    with opened(Path(root) / tensor['file']) as (stream, size):
        if expected is not None:
            if size != expected['bytes'] or hashlib.file_digest(stream,'sha256').hexdigest()!=expected['sha256']:
                raise ValueError('tensor changed after attachment validation')
        if tensor['offset'] + tensor['bytes'] > size:
            raise ValueError('tensor span outside artifact')
        stream.seek(tensor['offset'])
        raw = stream.read(tensor['bytes'])
        if len(raw) != tensor['bytes']:
            raise ValueError('truncated tensor')
        return raw


def values(raw, dtype):
    width = WIDTH[dtype]
    if len(raw) % width:
        raise ValueError('trailing partial element')
    if dtype == 'BF16':
        return [struct.unpack('<f', b'\0\0' + raw[i:i+2])[0] for i in range(0, len(raw), 2)]
    code = {'F32':'f', 'F64':'d', 'I64':'q', 'U32':'I', 'U8':'B', 'BOOL':'B'}[dtype]
    return [v[0] for v in struct.iter_unpack('<' + code, raw)]


def safetensors(root, filename):
    """Read only the bounded header; tensor payloads remain referenced by span."""
    with opened(Path(root) / component(filename)) as (stream, size):
        first = stream.read(8)
        if len(first) != 8:
            raise ValueError('truncated safetensors header')
        length, = struct.unpack('<Q', first)
        if length > METADATA_BYTES or length + 8 > size:
            raise ValueError('invalid safetensors header length')
        header = decode_json(stream.read(length))
    descriptors = []
    spans = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        start, end = info['data_offsets']
        if type(start) is not int or type(end) is not int or start < 0 or end < start or end + 8 + length > size:
            raise ValueError('invalid safetensors data span')
        tensor = tensor_descriptor(name, info['shape'], info['dtype'], filename, start + 8 + length)
        if tensor['bytes'] != end - start:
            raise ValueError('safetensors shape/dtype span differs')
        spans.append((start, end))
        descriptors.append(tensor)
    previous = 0
    for start, end in sorted(spans):
        if start != previous:
            raise ValueError('overlapping or incomplete safetensors payload')
        previous = end
    if previous + 8 + length != size:
        raise ValueError('trailing safetensors payload')
    return descriptors
