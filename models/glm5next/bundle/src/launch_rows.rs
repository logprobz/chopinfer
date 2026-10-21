//! `crate::submit` in test builds: the shared submission functions, with
//! every launch the bundle submits or captures recorded as its lane (rank or
//! stream), symbol and rows, and every launch of the shared runtime's
//! recorded programs (the encoder's) with its live and prepared rows. The
//! fake-host journals do not record rows, so these records keep the rows of
//! every launch comparable.
//!
//! With `CHOPINFER_GLM5NEXT_ROWS_DIR` set, each test thread writes its
//! records to `<dir>/<test name>.rows.tsv` as it makes them.

use std::cell::RefCell;
use std::collections::BTreeMap;

use bundle_runtime::dispatch::{self, BufferBindings, DispatchError};
use bundle_runtime::geometry::BundleGeometry;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::pipeline::{self as shared, Collectives, PreparedLaunch, Program};
use kernel_core::contract::{ArgumentList, LaunchPlan};
use runtime_bundle_abi::{BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesV1};

/// Where each test thread writes its records.
const ROWS_DIR: &str = "CHOPINFER_GLM5NEXT_ROWS_DIR";

/// One recorded launch: the slot of the stream it goes to, its symbol, and
/// its lane, symbol and rows as one line.
pub struct Recorded {
    pub stream: u32,
    pub symbol: String,
    pub line: String,
}

/// A thread's records since `start_recording` that `take_recorded` has not
/// taken, and where the thread writes every record: unread until the first
/// one, then the file or nowhere.
#[derive(Default)]
struct Records {
    pending: Option<Vec<Recorded>>,
    export: Option<Option<std::fs::File>>,
}

thread_local! {
    static RECORDED: RefCell<Records> = RefCell::default();
}

/// The file this thread's records go to, if `ROWS_DIR` is set.
fn export_file() -> Option<std::fs::File> {
    let root = std::path::PathBuf::from(std::env::var_os(ROWS_DIR)?);
    let name = std::thread::current().name()?.to_owned();
    std::fs::create_dir_all(&root).expect("create rows directory");
    Some(std::fs::File::create(root.join(format!("{name}.rows.tsv"))).expect("export launch rows"))
}

fn push(stream: u32, symbol: &str, line: String) {
    use std::io::Write;
    RECORDED.with(|recorded| {
        let mut recorded = recorded.borrow_mut();
        if let Some(file) = recorded.export.get_or_insert_with(export_file) {
            writeln!(file, "{line}").expect("export launch rows");
        }
        if let Some(pending) = recorded.pending.as_mut() {
            pending.push(Recorded {
                stream,
                symbol: symbol.to_owned(),
                line,
            });
        }
    });
}

fn record(stream: BundleStreamHandleV1, lane: &str, launches: &[PreparedLaunch]) {
    for launch in launches {
        push(
            stream.slot(),
            &launch.symbol,
            format!("{lane}\t{}\t{}", launch.symbol, launch.rows),
        );
    }
}

pub(crate) fn record_dma_sum(stream: BundleStreamHandleV1, sum: &PreparedLaunch) {
    record(
        stream,
        &format!("stream {}", stream.slot()),
        std::slice::from_ref(sum),
    );
}

/// Records the launches this thread's recorded programs (the encoder's)
/// submit from now on.
pub fn observe_programs() {
    bundle_runtime::recorded::observe::launches(Some(Box::new(|stream, symbol, rows| {
        push(
            stream.slot(),
            symbol,
            format!(
                "encoder stream {}\t{symbol}\t{} of {}",
                stream.slot(),
                rows.live,
                rows.capacity
            ),
        )
    })));
}

/// Keeps this thread's records from now on, for `take_recorded`.
pub fn start_recording() {
    RECORDED.with(|recorded| recorded.borrow_mut().pending = Some(Vec::new()));
}

/// This thread's records since `start_recording` or the last call.
pub fn take_recorded() -> Vec<Recorded> {
    RECORDED.with(|recorded| {
        std::mem::take(
            recorded
                .borrow_mut()
                .pending
                .as_mut()
                .expect("recording started"),
        )
    })
}

pub fn launch_all(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    record(stream, &format!("stream {}", stream.slot()), launches);
    shared::launch_all(services, stream, symbols, launches)
}

/// The shared `run_ranks`, recording each rank's lowered launches when they
/// are submitted or captured; a cached graph's replay launches nothing.
#[allow(clippy::too_many_arguments)]
pub fn run_ranks<G: BundleGeometry>(
    services: &HostServicesV1,
    streams: &[BundleStreamHandleV1],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    graph: Option<(u32, u32)>,
    graphs: &mut GraphCache<G>,
    collectives: &Collectives,
    mut prepare: impl FnMut(usize) -> Result<Program, DispatchError>,
    submitted: impl FnMut(usize),
) -> Result<(), DispatchError> {
    shared::run_ranks(
        services,
        streams,
        symbols,
        graph,
        graphs,
        collectives,
        |rank| {
            let program = prepare(rank)?;
            record(streams[rank], &format!("rank {rank}"), &program.launches);
            Ok(program)
        },
        submitted,
    )
}

/// The shared single launch, which carries no rows: recorded as `-`.
pub fn launch(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    symbol: &str,
    plan: LaunchPlan,
    planned: &ArgumentList,
    bindings: &BufferBindings,
) -> Result<(), DispatchError> {
    push(
        stream.slot(),
        symbol,
        format!("stream {}\t{symbol}\t-", stream.slot()),
    );
    dispatch::launch(services, stream, symbols, symbol, plan, planned, bindings)
}
