//! A frontend exported with `export_frontend_v1!`, driven through the same
//! trampolines in process by the host's safe handles: answers, refusals, a
//! caught panic in every entry, and the parser and close lifecycle.

use std::sync::{Arc, Mutex};

use runtime_bundle_abi::frontend::{
    FRONTEND_OPENED_BYTES_MAX_V1, FRONTEND_REFUSAL_BYTES_MAX_V1, FrontendCallErrorV1,
    FrontendExportV1, FrontendHandleV1, FrontendParserV1, FrontendV1,
};

/// What each frontend and parser did, by the name it was opened with, so
/// tests running in parallel each read their own record.
static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn log(entry: String) {
    LOG.lock().unwrap().push(entry);
}

fn logged(entry: &str) -> bool {
    LOG.lock().unwrap().iter().any(|logged| logged == entry)
}

/// Opened over a component naming it; "refuse" and "panic" refuse or panic
/// in open, and a name ending in "!" panics when closed.
struct Fake {
    name: String,
}

struct FakeParser {
    name: String,
    output: Vec<u8>,
}

/// Requests and texts "refuse" and "panic" refuse or panic in any entry.
fn answer(input: &[u8], prefix: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
    match input {
        b"panic" => panic!("a test panic"),
        b"refuse" => Err(b"refused: test".to_vec()),
        b"refuse-large" => Err(vec![b'r'; FRONTEND_REFUSAL_BYTES_MAX_V1 + 1]),
        _ => Ok([prefix, input].concat()),
    }
}

impl FrontendV1 for Fake {
    type Parser = FakeParser;

    fn open(component: &[u8]) -> Result<(Self, Vec<u8>), Vec<u8>> {
        let opened = answer(component, b"opened:")?;
        Ok((
            Self {
                name: String::from_utf8(component.to_vec()).unwrap(),
            },
            opened,
        ))
    }

    fn render(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        answer(request, b"render:")
    }

    fn constraint(&self, request: &[u8]) -> Result<Vec<u8>, Vec<u8>> {
        answer(request, b"constraint:")
    }

    fn open_parser(&self, request: &[u8]) -> Result<FakeParser, Vec<u8>> {
        answer(request, b"")?;
        Ok(FakeParser {
            name: format!("{}/{}", self.name, String::from_utf8_lossy(request)),
            output: Vec::new(),
        })
    }
}

impl FrontendParserV1 for FakeParser {
    fn push(&mut self, text: &[u8], last: bool) -> Result<Vec<u8>, Vec<u8>> {
        let event = answer(text, b"")?;
        self.output.extend_from_slice(&event);
        Ok(if last { self.output.clone() } else { event })
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        log(format!("closed {}", self.name));
        assert!(!self.name.ends_with('!'), "a test panic in close");
    }
}

impl Drop for FakeParser {
    fn drop(&mut self) {
        log(format!("closed {}", self.name));
        assert!(!self.name.ends_with('!'), "a test panic in parser close");
    }
}

runtime_bundle_abi::export_frontend_v1!(Fake);

fn export() -> FrontendExportV1 {
    FrontendExportV1::in_process::<Fake>()
}

fn open(name: &str) -> FrontendHandleV1 {
    let (frontend, answer) = export().open(name.as_bytes()).unwrap();
    assert_eq!(answer, [b"opened:", name.as_bytes()].concat());
    frontend
}

fn refused(document: &str) -> FrontendCallErrorV1 {
    FrontendCallErrorV1::Refused(document.as_bytes().to_vec())
}

#[test]
fn answers_and_refusals_cross_the_export() {
    let frontend = open("answers");
    assert_eq!(frontend.render(b"chat", 1 << 20).unwrap(), b"render:chat");
    assert_eq!(
        frontend.constraint(b"tools", 1 << 20).unwrap(),
        b"constraint:tools"
    );
    assert_eq!(frontend.render(b"", 1 << 20).unwrap(), b"render:");
    assert_eq!(
        frontend.render(b"refuse", 1 << 20).unwrap_err(),
        refused("refused: test")
    );
    assert_eq!(
        frontend.constraint(b"refuse", 1 << 20).unwrap_err(),
        refused("refused: test")
    );
    assert_eq!(
        frontend.open_parser(b"refuse").err(),
        Some(refused("refused: test"))
    );
    assert_eq!(
        export().open(b"refuse").err(),
        Some(refused("refused: test"))
    );
    // A refusal leaves the frontend usable.
    assert_eq!(frontend.render(b"again", 1 << 20).unwrap(), b"render:again");
    frontend.close().unwrap();
    assert!(logged("closed answers"));
}

/// The handles print without pointers, so a bundle test can `unwrap_err` an
/// open.
#[test]
fn handles_debug_print_without_pointers() {
    assert_eq!(
        export().open(b"refuse").unwrap_err(),
        refused("refused: test")
    );
    let frontend = open("debug");
    assert_eq!(
        frontend.open_parser(b"refuse").unwrap_err(),
        refused("refused: test")
    );
    assert_eq!(
        format!("{frontend:?}"),
        "FrontendHandleV1 { export: FrontendExportV1 { loaded_module: false, .. }, .. }"
    );
    let mut parser = frontend.open_parser(b"request").unwrap();
    assert_eq!(
        format!("{parser:?}"),
        "FrontendParserHandleV1 { finished: false, closed: false, .. }"
    );
    parser.push(b"", true, 1 << 20).unwrap();
    assert_eq!(
        format!("{parser:?}"),
        "FrontendParserHandleV1 { finished: true, closed: false, .. }"
    );
    parser.close().unwrap();
    frontend.close().unwrap();
}

#[test]
fn a_panic_in_any_entry_is_caught_and_reported() {
    assert_eq!(
        export().open(b"panic").err(),
        Some(FrontendCallErrorV1::Panicked)
    );
    let frontend = open("panics");
    assert_eq!(
        frontend.render(b"panic", 1 << 20).unwrap_err(),
        FrontendCallErrorV1::Panicked
    );
    assert_eq!(
        frontend.constraint(b"panic", 1 << 20).unwrap_err(),
        FrontendCallErrorV1::Panicked
    );
    assert_eq!(
        frontend.open_parser(b"panic").err(),
        Some(FrontendCallErrorV1::Panicked)
    );
    // The frontend still answers after a caught panic.
    assert_eq!(frontend.render(b"chat", 1 << 20).unwrap(), b"render:chat");

    let mut parser = frontend.open_parser(b"push").unwrap();
    assert_eq!(
        parser.push(b"panic", false, 1 << 20).unwrap_err(),
        FrontendCallErrorV1::Panicked
    );
    // After a panic the parser takes no further push.
    assert_eq!(
        parser.push(b"more", false, 1 << 20).unwrap_err(),
        FrontendCallErrorV1::ParserFinished
    );
    parser.close().unwrap();

    let parser = frontend.open_parser(b"close!").unwrap();
    assert_eq!(parser.close().unwrap_err(), FrontendCallErrorV1::Panicked);
    assert!(logged("closed panics/close!"));
    frontend.close().unwrap();

    let frontend = open("close!");
    assert_eq!(frontend.close().unwrap_err(), FrontendCallErrorV1::Panicked);
    assert!(logged("closed close!"));
}

#[test]
fn a_parser_runs_to_its_final_push_and_then_only_closes() {
    let frontend = open("parser");
    let mut parser = frontend.open_parser(b"request").unwrap();
    assert_eq!(parser.push(b"Hel", false, 1 << 20).unwrap(), b"Hel");
    assert_eq!(parser.push(b"", false, 1 << 20).unwrap(), b"");
    assert_eq!(parser.push(b"lo", false, 1 << 20).unwrap(), b"lo");
    assert_eq!(parser.push(b"!", true, 1 << 20).unwrap(), b"Hello!");
    assert_eq!(
        parser.push(b"late", false, 1 << 20).unwrap_err(),
        FrontendCallErrorV1::ParserFinished
    );
    assert_eq!(
        parser.push(b"", true, 1 << 20).unwrap_err(),
        FrontendCallErrorV1::ParserFinished
    );
    parser.close().unwrap();
    assert!(logged("closed parser/request"));

    // A refusal ends the parser too.
    let mut parser = frontend.open_parser(b"refusal").unwrap();
    assert_eq!(
        parser.push(b"refuse", false, 1 << 20).unwrap_err(),
        refused("refused: test")
    );
    assert_eq!(
        parser.push(b"more", false, 1 << 20).unwrap_err(),
        FrontendCallErrorV1::ParserFinished
    );
    // Dropping a parser closes it.
    drop(parser);
    assert!(logged("closed parser/refusal"));
    frontend.close().unwrap();
}

/// A frontend closes exactly once, after its handle and every parser opened
/// from it are gone, whichever goes last.
#[test]
fn a_frontend_closes_after_its_last_parser() {
    let frontend = open("ordered");
    let mut parser = frontend.open_parser(b"open").unwrap();
    assert_eq!(
        frontend.close().unwrap_err(),
        FrontendCallErrorV1::Contract("a frontend closed while a parser is open")
    );
    assert!(!logged("closed ordered"));
    assert_eq!(parser.push(b"still", true, 1 << 20).unwrap(), b"still");
    drop(parser);
    assert!(logged("closed ordered/open"));
    assert!(logged("closed ordered"));
    let count = LOG
        .lock()
        .unwrap()
        .iter()
        .filter(|entry| *entry == "closed ordered")
        .count();
    assert_eq!(count, 1);

    drop(open("dropped"));
    assert!(logged("closed dropped"));
}

/// Render and constraint run concurrently on one frontend, and a parser
/// moves to another thread.
#[test]
fn handles_cross_threads_as_the_contract_allows() {
    let frontend = Arc::new(open("threads"));
    let workers = (0..8)
        .map(|worker| {
            let frontend = Arc::clone(&frontend);
            std::thread::spawn(move || {
                for round in 0..64 {
                    let request = format!("{worker}.{round}");
                    assert_eq!(
                        frontend.render(request.as_bytes(), 1 << 20).unwrap(),
                        format!("render:{request}").as_bytes()
                    );
                    assert_eq!(
                        frontend.constraint(request.as_bytes(), 1 << 20).unwrap(),
                        format!("constraint:{request}").as_bytes()
                    );
                }
            })
        })
        .collect::<Vec<_>>();
    let mut parser = frontend.open_parser(b"moved").unwrap();
    parser.push(b"a", false, 1 << 20).unwrap();
    let parser = std::thread::spawn(move || {
        assert_eq!(parser.push(b"b", true, 1 << 20).unwrap(), b"ab");
        parser
    })
    .join()
    .unwrap();
    for worker in workers {
        worker.join().unwrap();
    }
    drop(parser);
    Arc::into_inner(frontend).unwrap().close().unwrap();
    assert!(logged("closed threads"));
}

/// The header names the capability and export these constants spell, and
/// the threading contract the handles rely on.
#[test]
fn the_header_declares_the_capability_and_export() {
    use runtime_bundle_abi::frontend::{FRONTEND_API_SYMBOL_V1, FRONTEND_CAPABILITY_V1};
    let header = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../include/chop_bundle_v1.h"),
    )
    .unwrap();
    assert!(header.contains(&format!(
        "#define CHOP_FRONTEND_CAPABILITY_V1 \"{FRONTEND_CAPABILITY_V1}\""
    )));
    let symbol = std::str::from_utf8(FRONTEND_API_SYMBOL_V1)
        .unwrap()
        .trim_end_matches('\0');
    assert!(header.contains(&format!(
        "const chop_bundle_frontend_table_v1 *{symbol}(void);"
    )));
    // The threading contract the safe handles rely on: FrontendHandleV1 is
    // Sync because these three entries may share a frontend.
    let prose = header
        .split_whitespace()
        .filter(|word| *word != "*")
        .collect::<Vec<_>>()
        .join(" ");
    assert!(prose.contains(
        "render, constraint and parser_open may run concurrently on one frontend from any thread"
    ));
}

/// parser_open, render and constraint run concurrently on one frontend,
/// as do pushes and closes of different parsers, and the frontend closes
/// once they are all closed.
#[test]
fn parsers_open_concurrently_on_one_frontend() {
    let frontend = open("concurrent");
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let frontend = &frontend;
            scope.spawn(move || {
                for round in 0..64 {
                    let request = format!("{worker}.{round}");
                    let mut parser = frontend.open_parser(request.as_bytes()).unwrap();
                    assert_eq!(
                        frontend.render(request.as_bytes(), 1 << 20).unwrap(),
                        format!("render:{request}").as_bytes()
                    );
                    assert_eq!(parser.push(b"a", false, 1 << 20).unwrap(), b"a");
                    assert_eq!(
                        frontend.constraint(request.as_bytes(), 1 << 20).unwrap(),
                        format!("constraint:{request}").as_bytes()
                    );
                    assert_eq!(parser.push(b"b", true, 1 << 20).unwrap(), b"ab");
                    parser.close().unwrap();
                }
            });
        }
    });
    frontend.close().unwrap();
    assert!(logged("closed concurrent"));
    assert!(logged("closed concurrent/7.63"));
}

/// A document past its call's limit, or a refusal past the fixed refusal
/// limit, fails the call before the host copies it.
#[test]
fn documents_past_their_limit_are_refused_uncopied() {
    let frontend = open("limits");
    let past = |length: u64, maximum: usize| FrontendCallErrorV1::Oversized { length, maximum };
    assert_eq!(frontend.render(b"ab", 9).unwrap(), b"render:ab");
    assert_eq!(frontend.render(b"abc", 9), Err(past(10, 9)));
    assert_eq!(frontend.constraint(b"a", 12).unwrap(), b"constraint:a");
    assert_eq!(frontend.constraint(b"ab", 12), Err(past(13, 12)));
    let refusal = (FRONTEND_REFUSAL_BYTES_MAX_V1 + 1) as u64;
    assert_eq!(
        frontend.render(b"refuse-large", usize::MAX),
        Err(past(refusal, FRONTEND_REFUSAL_BYTES_MAX_V1))
    );
    assert_eq!(
        export().open(b"refuse-large").map(|_| ()),
        Err(past(refusal, FRONTEND_REFUSAL_BYTES_MAX_V1))
    );
    // An open answer past its limit fails the open, and the bundle drops
    // the frontend it would have returned.
    let long = "o".repeat(FRONTEND_OPENED_BYTES_MAX_V1 - b"opened:".len() + 1);
    assert_eq!(
        export().open(long.as_bytes()).map(|_| ()),
        Err(past(
            FRONTEND_OPENED_BYTES_MAX_V1 as u64 + 1,
            FRONTEND_OPENED_BYTES_MAX_V1
        ))
    );
    assert!(logged(&format!("closed {long}")));
    let (at_the_limit, _) = export()
        .open(&long.as_bytes()[1..])
        .expect("an answer at the limit opens");
    at_the_limit.close().unwrap();
    let mut parser = frontend.open_parser(b"limits").unwrap();
    assert_eq!(parser.push(b"ab", false, 2).unwrap(), b"ab");
    assert_eq!(parser.push(b"abc", false, 2), Err(past(3, 2)));
    // A push refused for its size ends the parser.
    assert_eq!(
        parser.push(b"a", false, 2),
        Err(FrontendCallErrorV1::ParserFinished)
    );
}
