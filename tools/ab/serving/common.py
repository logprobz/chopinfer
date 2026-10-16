"""Small, strict file and value helpers for the serving A/B runner."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat


class Refusal(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise Refusal(message)


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, f'duplicate JSON key: {key}')
        out[key] = value
    return out


def loads(text):
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(Refusal(f'nonfinite JSON: {x}')))


def read(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'r') as stream:
        need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'JSON input is not a regular file')
        return loads(stream.read())


def write(path, value, mode=0o644):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write('\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'digest input is not a regular file')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def regular(path, protected=False):
    path = Path(path)
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f'not a regular non-symlink file: {path}')
    if protected:
        need(not mode & 0o022, f'group/other-writable file: {path}')
    return path


def finite(value, where, minimum=None):
    need(type(value) in (int, float) and math.isfinite(value), f'{where}: expected finite number')
    need(minimum is None or value >= minimum, f'{where}: below {minimum}')
    return value


def keys(value, required, optional, where):
    need(type(value) is dict, f'{where}: expected object')
    need(required <= value.keys(), f'{where}: missing {sorted(required - value.keys())}')
    need(value.keys() <= required | optional, f'{where}: unknown {sorted(value.keys() - required - optional)}')


def subset(expected, actual, where='health'):
    if isinstance(expected, dict):
        need(isinstance(actual, dict), f'{where}: expected object')
        for key, value in expected.items():
            need(key in actual, f'{where}.{key}: missing')
            subset(value, actual[key], f'{where}.{key}')
    else:
        need(type(actual) is type(expected) and actual == expected,
             f'{where}: got {actual!r}, expected {expected!r}')
