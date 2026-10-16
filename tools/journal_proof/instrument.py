"""Patch only an archive copy: the journal dump and the fake-host trace.

The patches touch the bundle runtime's fake_host.rs (the dump in
check_journal, the trace hooks, and the trace module) and the model's
instance create boundary. They never change what a journal records.
"""
import hashlib
import json
from pathlib import Path
import re

from common import need, uncomment

ROOT = Path(__file__).resolve().parent
ANCHOR = '    let observed = journal_digest(lines);\n    if observed == expected {'
DUMP = '''    let observed = journal_digest(lines);
    if let Ok(dir) = std::env::var("CHOPINFER_JOURNAL_DUMP") {
        let site = std::panic::Location::caller();
        // A dump that fails ends the test process. A panic could be caught
        // (catch_unwind, should_panic) and leave an earlier dump of the name
        // to stand for a check that was refused.
        let dumped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use std::io::Write;
            assert_eq!(std::path::Path::new(file).components().count(), 1, "a journal name is a file name");
            // Each test checks a journal once; checks by several tests are
            // merged by the capture, which requires the same body and pin.
            let dir = std::path::Path::new(&dir).join(jproof_trace::check_directory());
            std::fs::create_dir_all(&dir).unwrap();
            for (name, data) in [
                (file.to_string(), lines.join("\\n")),
                (format!("{file}.pin"), format!("({}, {:#018x}) expected ({}, {:#018x})\\n",
                                                observed.0, observed.1, expected.0, expected.1)),
                (format!("{file}.site"), format!("{}:{}\\n", site.file(), site.line())),
            ] {
                std::fs::OpenOptions::new().write(true).create_new(true).open(dir.join(&name))
                    .unwrap_or_else(|error| panic!("{name}: {error}; a test checks a journal twice"))
                    .write_all(data.as_bytes()).unwrap();
            }
            jproof_trace::checkpoint(file);
        }));
        if let Err(payload) = dumped {
            use std::io::Write;
            let why = payload.downcast_ref::<String>().map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("panic");
            // Written to the process's stderr: the test harness's output
            // capture is lost on abort.
            let _ = writeln!(std::io::stderr(), "journal dump {file}: {why}; aborting the test process");
            std::process::abort();
        }
    }
    if observed == expected {'''
CHECKER = 'pub fn check_journal(lines: &[String], expected: (usize, u64), file: &str) -> Result<(), String> {'
CREATE = (r'(    fn create\(services: HostServicesV1, configuration: &\[u8\]\) -> Result<Self, Failed<Self>> \{)'
          r'\s*create\(services, configuration\)\s*\}')


def runtime_crate(source):
    """The bundle runtime crate's directory and Rust name at this source tree:
    crates/bundle-runtime (bundle_runtime), or crates/bundle-runtime-sm120
    before its rename."""
    source = Path(source)
    for directory, ident in (('crates/bundle-runtime', 'bundle_runtime'),
                             ('crates/bundle-runtime-sm120', 'bundle_runtime_sm120')):
        if (source / directory / 'src/fake_host.rs').is_file():
            return directory, ident
    need(False, 'no bundle runtime fake_host.rs in the source tree')


def test_names(source, bundle):
    """Every #[test] function name in the bundle crate's sources. A test
    thread is traced when its name is one of these or ends in ::name."""
    names = set()
    for path in sorted((Path(source) / bundle / 'src').rglob('*.rs')):
        text = uncomment(path.read_text())
        names.update(re.findall(r'#\[test\]\s*(?:#\[[^\]]*\]\s*)*fn\s+(\w+)\s*\(', text))
    need(names, f'no tests in {bundle}/src')
    return sorted(names)


def record(source, path, before):
    return dict(path=str(path.relative_to(source)),
                before_sha256=hashlib.sha256(before.encode()).hexdigest() if before is not None else None,
                after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def patch(source, bundle):
    """Patch the archive at `source` for the bundle crate in directory
    `bundle` (models/ID/bundle). Returns the patched files' records."""
    source = Path(source)
    crate, ident = runtime_crate(source)
    shared = source / crate / 'src/fake_host.rs'
    original = shared.read_text()
    text = original
    def trace(event):
        return f'if jproof_trace::enabled() {{ jproof_trace::event(jproof_trace::identity(&self.jproof_cookie), {event}); }}'
    edits = [
        (ANCHOR, DUMP),
        # The dump records each check's call site.
        (CHECKER, '#[track_caller]\n' + CHECKER),
        ('pub struct FakeDevice<O = NoObserver> {', 'pub struct FakeDevice<O = NoObserver> {\n    pub jproof_cookie: Cell<usize>,'),
        ('        self.next.set(next);\n        next',
         '        self.next.set(next);\n        ' + trace('serde_json::json!(["id", next])') + '\n        next'),
        ('    pub fn note(&self, entry: String) {',
         '    pub fn note(&self, entry: String) {\n        ' + trace('serde_json::json!(["note", &entry])')),
        ('        self.buffer_order.borrow_mut().push(id);\n        BundleDeviceBufferHandleV1::from_parts(1, id)', '''        self.buffer_order.borrow_mut().push(id);
        let value = BundleDeviceBufferHandleV1::from_parts(1, id);
        if jproof_trace::enabled() {
            jproof_trace::event(jproof_trace::identity(&self.jproof_cookie), serde_json::json!([
                "allocation", value.raw, self.stream_ranks.borrow().get(&stream.slot()).copied(), byte_length, alignment, stream.raw]));
        }
        value'''),
        ('        self.stream_ranks.borrow_mut().insert(id, rank);',
         '        self.stream_ranks.borrow_mut().insert(id, rank);\n        ' + trace('serde_json::json!(["stream", id, rank])')),
        ('        BundleCudaModuleHandleV1::from_parts(1, id)\n    }', '''        if jproof_trace::enabled() {
            let name = self.module_cubins.borrow().get(&id).copied().unwrap_or("?");
            jproof_trace::event(jproof_trace::identity(&self.jproof_cookie), serde_json::json!([
                "module", id, name, format!("{:016x}", fnv1a(module_bytes)), module_bytes.len(), jproof_trace::module_payload(module_bytes)]));
        }
        BundleCudaModuleHandleV1::from_parts(1, id)
    }'''),
        ('        self.symbol_names.borrow_mut().insert(id, name);',
         '        ' + trace('serde_json::json!(["resolve", id, module.slot(), &name])') + '\n        self.symbol_names.borrow_mut().insert(id, name);'),
    ]
    for before, after in edits:
        need(text.count(before) == 1, 'unsupported trace anchor: ' + before[:80])
        text = text.replace(before, after, 1)
    text += '\n#[doc(hidden)]\npub mod jproof_trace;\n'
    shared.write_text(text)
    runtime = (ROOT / 'trace_runtime.rs').read_text().replace('__TEST_NAMES__', json.dumps(test_names(source, bundle)))
    runtime_path = source / crate / 'src/fake_host/jproof_trace.rs'
    need(not runtime_path.exists(), f'{runtime_path} already exists')
    runtime_path.parent.mkdir(exist_ok=True)
    runtime_path.write_text(runtime)
    instance = source / bundle / 'src/instance.rs'
    old = instance.read_text()
    replacement = r'''\1
        #[cfg(test)]
        let trace = IDENT::fake_host::jproof_trace::begin_create();
        let result = create(services, configuration);
        #[cfg(test)]
        IDENT::fake_host::jproof_trace::end_create(trace, result.is_ok());
        result
    }'''.replace('IDENT', ident)
    changed, count = re.subn(CREATE, replacement, old)
    need(count == 1, f'unsupported instance create boundary in {bundle}/src/instance.rs')
    instance.write_text(changed)
    return [record(source, shared, original), record(source, runtime_path, None), record(source, instance, old)]
