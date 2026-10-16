//! Archive-only trace instrumentation. It never changes the pinned journal.
//!
//! Each test thread keeps its own trace. Its devices are numbered from 1 in
//! the order the test creates them, their records go to
//! `aux/tests/<test>/device-<n>.jsonl`, and module payloads are named by
//! their bytes. Nothing depends on which tests ran before or beside a test,
//! so serial, threaded and one-process-per-test runs write the same files.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use serde_json::{Value, json};

const TESTS: &[&str] = &__TEST_NAMES__;
#[derive(Default)]
struct Device {
    /// Records not yet written to the device file.
    pending: Vec<Value>,
    records: usize,
    notes: usize,
}
struct Trace {
    directory: PathBuf,
    test: String,
    next: usize,
    devices: BTreeMap<usize, Device>,
}
thread_local! {
    static TRACE: RefCell<Option<Trace>> = RefCell::new({
        let test = std::thread::current().name().unwrap_or("").to_owned();
        std::env::var("CHOPINFER_JOURNAL_DUMP").ok().and_then(|directory| {
            TESTS.iter().any(|name| test == *name || test.ends_with(&format!("::{name}")))
                .then(|| Trace { directory: PathBuf::from(directory), test, next: 0, devices: BTreeMap::new() })
        })
    });
}
pub fn enabled() -> bool { TRACE.with(|t| t.borrow().is_some()) }
/// A test's directory under aux/tests: its path with `::` as `.`.
fn test_directory(test: &str) -> String { test.replace("::", ".") }
pub fn identity(cookie: &std::cell::Cell<usize>) -> usize {
    if !enabled() { return 0; }
    TRACE.with(|t| {
        let mut trace = t.borrow_mut();
        let trace = trace.as_mut().unwrap();
        if cookie.get() == 0 {
            trace.next += 1;
            cookie.set(trace.next);
        }
        trace.devices.entry(cookie.get()).or_default();
    });
    cookie.get()
}
pub fn event(id: usize, value: Value) {
    if id == 0 { return; }
    TRACE.with(|t| {
        let mut trace = t.borrow_mut();
        let device = trace.as_mut().unwrap().devices.get_mut(&id).unwrap();
        if value[0] == "note" { device.notes += 1; }
        device.records += 1;
        device.pending.push(value);
    });
}
pub fn begin_create() -> BTreeMap<usize, (usize, usize)> {
    TRACE.with(|t| t.borrow().as_ref().map(|trace| trace.devices.iter()
        .map(|(&id, device)| (id, (device.records, device.notes))).collect()).unwrap_or_default())
}
pub fn end_create(before: BTreeMap<usize, (usize, usize)>, succeeded: bool) {
    if !succeeded { return; }
    TRACE.with(|t| {
        if let Some(trace) = t.borrow_mut().as_mut() {
            for (&id, device) in &mut trace.devices {
                let (events, notes) = before.get(&id).copied().unwrap_or_default();
                if device.records != events {
                    let marker = json!(["created", events, notes, device.records, device.notes]);
                    device.records += 1;
                    device.pending.push(marker);
                }
            }
        }
    });
}
/// Where this test thread's checks dump their journals, under the dump
/// directory: `checks/<test>`. Several tests may check one journal; the
/// capture merges their checks after the run.
pub fn check_directory() -> String {
    TRACE.with(|t| {
        let trace = t.borrow();
        let trace = trace.as_ref().expect("pinned fixture was not selected for tracing");
        format!("checks/{}", test_directory(&trace.test))
    })
}
pub fn checkpoint(file: &str) {
    TRACE.with(|t| {
        let mut trace = t.borrow_mut();
        let trace = trace.as_mut().expect("pinned fixture was not selected for tracing");
        let aux = trace.directory.join("aux");
        let relative = format!("tests/{}", test_directory(&trace.test));
        std::fs::create_dir_all(aux.join(&relative)).unwrap();
        let mut devices = Vec::new();
        for (&id, device) in &mut trace.devices {
            let name = format!("{relative}/device-{id}.jsonl");
            let file = std::fs::OpenOptions::new().create(true).append(true)
                .open(aux.join(&name)).unwrap();
            let mut output = std::io::BufWriter::new(file);
            for record in device.pending.drain(..) {
                serde_json::to_writer(&mut output, &record).unwrap();
                writeln!(output).unwrap();
            }
            output.flush().unwrap();
            devices.push(json!({"id":id,"path":name,"records":device.records,"notes":device.notes}));
        }
        let value = json!({"schema":"chopinfer.journal-trace.checkpoint.v1", "journal":file,
                           "test":trace.test,"devices":devices});
        std::fs::write(aux.join(&relative).join(format!("{file}.checkpoint.json")),
                       serde_json::to_vec(&value).unwrap()).unwrap();
    });
}

/// The payload file of a module's bytes, `modules/<fnv>-<length>.bin` under
/// aux, written once. Concurrent writers of the same bytes rename identical
/// files into place; different bytes under one name fail the test.
pub fn module_payload(bytes: &[u8]) -> String {
    static TEMPORARY: AtomicUsize = AtomicUsize::new(0);
    let name = format!("modules/{:016x}-{}.bin", super::fnv1a(bytes), bytes.len());
    let aux = PathBuf::from(std::env::var("CHOPINFER_JOURNAL_DUMP").unwrap()).join("aux");
    std::fs::create_dir_all(aux.join("modules")).unwrap();
    let path = aux.join(&name);
    if let Ok(existing) = std::fs::read(&path) {
        assert!(existing == bytes, "two module payloads share {name}");
        return name;
    }
    let temporary = aux.join(format!("modules/.{}-{}.tmp", std::process::id(),
                                     TEMPORARY.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&temporary, bytes).unwrap();
    std::fs::rename(&temporary, &path).unwrap();
    assert!(std::fs::read(&path).unwrap() == bytes, "two module payloads share {name}");
    name
}
