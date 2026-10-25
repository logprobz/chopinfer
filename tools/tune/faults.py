"""Read process fault counters without counting child totals twice."""
from pathlib import Path
import json,time

def validate(evidence):
    """Refuse an incomplete or over-budget measured visit."""
    count = evidence.get('major_faults')
    if evidence.get('complete') is not True or type(count) is not int or not 0 <= count <= 1000:
        raise ValueError(f'invalid visit fault evidence: {evidence}')

def combine(parts):
    return {'complete': bool(parts) and all(p.get('complete') is True for p in parts),
            'major_faults': sum(p['major_faults'] for p in parts),
            'minor_faults': sum(p.get('minor_faults', 0) for p in parts),
            'scope': 'Measured windows only; startup, warmup and drain excluded.'}

def parse_stat(text):
    pid=int(text.split(' ',1)[0]); fields=text.rsplit(')',1)[1].split()
    return {'pid':pid,'state':fields[0],'pgrp':int(fields[2]),'start_ticks':int(fields[19]),
            'minor_faults':int(fields[7]),'major_faults':int(fields[9])}

def snapshot(server):
    group=json.loads(server.pidfile.read_text())['pgid'];rows={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            row=parse_stat(path.read_text())
            if row['pgrp']==group and row['state']!='Z':rows[str(row['pid'])]=row
        except (OSError,ValueError):pass
    if not rows:raise RuntimeError('no live server processes for fault counters')
    return {'monotonic_seconds':time.monotonic(),'process_group':group,'processes':rows}

def difference(before,after):
    if before['process_group']!=after['process_group']:raise ValueError('process group changed')
    rows=[]
    for pid,end in after['processes'].items():
        start=before['processes'].get(pid)
        if start is not None and start['start_ticks']!=end['start_ticks']:raise ValueError('PID reused within a timing window')
        row={'pid':int(pid),'start_ticks':end['start_ticks'],'born_during_window':start is None}
        for key in ('minor_faults','major_faults'):
            row[key]=end[key]-(start[key] if start else 0)
            if row[key]<0:raise ValueError('fault counter decreased')
        rows.append(row)
    missing=sorted(set(before['processes'])-set(after['processes']))
    return {'process_group':after['process_group'],'processes':rows,
            'minor_faults':sum(r['minor_faults'] for r in rows),
            'major_faults':sum(r['major_faults'] for r in rows),
            'exited_processes_without_final_counter':missing,'complete':not missing,
            'scope':'Each live process own minflt/majflt, excluding child-total fields; deltas bracket this window.'}


def open_regular(path):
    """Open an absolute regular file without following any path symlink."""
    import os
    import stat
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('cold image path must be absolute without parent traversal')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('cold image must be a regular file')
    return fd


def file_identity(path):
    """Seal immutable mapped input metadata without reading all file pages."""
    import os
    fd = open_regular(path)
    try:
        value = os.fstat(fd)
        return {'path': str(path), 'size': value.st_size, 'inode': value.st_ino,
                'device': value.st_dev, 'mtime_ns': value.st_mtime_ns}
    finally:
        os.close(fd)


def residency(fd, offset, length):
    """Count file-cache pages with mincore on an untouched PROT_NONE map."""
    import ctypes
    import mmap
    import os
    if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or offset + length > os.fstat(fd).st_size:
        raise ValueError('invalid Engram view range')
    page = mmap.PAGESIZE
    start = offset // page * page
    size = offset + length - start
    pages = (size + page - 1) // page
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    address = libc.mmap(None, size, 0, mmap.MAP_SHARED, fd, start)
    if address == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), 'map Engram view for mincore')
    try:
        vector = (ctypes.c_ubyte * pages)()
        if libc.mincore(address, size, vector):
            raise OSError(ctypes.get_errno(), 'mincore Engram view')
        resident = sum(v & 1 for v in vector)
        return {'pages': pages, 'resident_pages': resident, 'resident_fraction': resident / pages}
    finally:
        libc.munmap(address, size)


class ColdEngram:
    """Apply the existing host measurement protocol, then advise file pages."""

    def __init__(self, server, spec):
        self.server, self.spec, self.epoch = server, spec, 0
        self.path = server.directory / 'mapped-row-control.json'
        server.env['CHOPINFER_MAPPED_ROW_CONTROL'] = str(self.path)
        self.images = []
        try:
            for image in spec.get('images', []):
                fd = open_regular(image['path'])
                self.images.append((fd, image))
                import os
                if os.fstat(fd).st_size != image['bytes'] or not image.get('views'):
                    raise ValueError('cold image size or Engram views missing')
                for view in image['views']:
                    residency(fd, view['offset'], view['bytes'])
            if not self.images and not server.dry:
                raise ValueError('cold Engram requires explicit rank image sources and view ranges')
            threshold = spec.get('warm_fraction', .1)
            if not 0 <= threshold <= 1:
                raise ValueError('invalid partly warm residency threshold')
        except BaseException:
            self.close()
            raise

    def close(self):
        import os
        for fd, _ in self.images:
            os.close(fd)
        self.images = []

    def sample(self):
        if self.server.dry:
            import os
            fraction = float(os.environ.get('TUNE_STUB_RESIDENCY', '0'))
            return [{'name': 'synthetic', 'pages': 100, 'resident_pages': round(fraction * 100),
                     'resident_fraction': fraction}]
        return [{'image': image['path'], 'name': view['name'], **residency(fd, view['offset'], view['bytes'])}
                for fd, image in self.images for view in image['views']]

    def prepare(self):
        import os
        import runner
        for _, image in self.images:
            if image.get('identity') and file_identity(image['path']) != image['identity']:
                raise RuntimeError('sealed cold image metadata changed')
        self.epoch += 1
        command = {'epoch': self.epoch, 'mode': 'both', 'reset_cache': True, 'release_page_tables': True}
        body = json.dumps(command).encode()
        if len(body) > 1024 or self.path.is_symlink():
            raise ValueError('unsafe mapped-row control file')
        temporary = self.path.with_suffix('.pending')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(body)
        log = self.server.directory / 'server.log'
        offset = log.stat().st_size
        temporary.replace(self.path)
        deadline = time.monotonic() + runner.remaining(10)
        ack = None
        while time.monotonic() < deadline:
            with log.open() as stream:
                stream.seek(offset)
                for line in stream:
                    if 'chopinfer.mapped-row-policy.v1' not in line:
                        continue
                    value = json.loads(line[line.index('{'):])
                    if value.get('status') == 'refused':
                        raise RuntimeError(f'mapped-row control refused: {value}')
                    if value.get('epoch') == self.epoch and value.get('status') == 'applied':
                        ack = value
            if ack:
                break
            time.sleep(.02)
        if not ack or ack.get('reset_cache') is not True or ack.get('released_mapping_bytes', 0) <= 0:
            raise RuntimeError('mapped-row control lacks acknowledged cache reset and page-table release')
        before = self.sample()
        for fd, _ in self.images:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        after = self.sample()
        warm = any(v['resident_fraction'] >= self.spec.get('warm_fraction', .1) for v in after)
        return {'epoch': self.epoch, 'acknowledgement': ack, 'before_advice': before, 'after_advice': after,
                'state': 'partly warm' if warm else 'cold', 'synthetic': self.server.dry}
