//! `crate::pipeline` in test builds: the shared pipeline, with every launch
//! list the bundle submits or captures recorded as its stream, symbol and
//! rows, and every launch of the shared runtime's recorded programs (the
//! encoder's) with its live and prepared rows. The fake-host journals do not
//! record rows, so tests compare these records to keep the rows of every
//! launch unchanged.

use std::cell::RefCell;
use std::collections::BTreeMap;

pub use bundle_runtime::pipeline::*;

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::geometry::BundleGeometry;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::pipeline as shared;
use runtime_bundle_abi::{BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesV1};

/// One recorded launch: the slot of the stream it goes to, its symbol, and
/// its lane (rank or stream), symbol and rows as one line.
pub struct Recorded {
    pub stream: u32,
    pub symbol: String,
    pub line: String,
    /// The launch plan, for the bundle's own launch lists (`launch_all`,
    /// `run_ranks`); none for the shared runtime's recorded programs.
    pub plan: Option<String>,
}

thread_local! {
    static RECORDED: RefCell<Vec<Recorded>> = const { RefCell::new(Vec::new()) };
}

fn record(stream: BundleStreamHandleV1, lane: &str, launches: &[PreparedLaunch]) {
    RECORDED.with(|recorded| {
        recorded
            .borrow_mut()
            .extend(launches.iter().map(|launch| Recorded {
                stream: stream.slot(),
                symbol: launch.symbol.clone(),
                line: format!("{lane}\t{}\t{}", launch.symbol, launch.rows),
                plan: Some(format!("{:?}", launch.plan)),
            }))
    });
}

/// Records the launches this thread's recorded programs submit from now on.
fn observe_programs() {
    bundle_runtime::recorded::observe::launches(Some(Box::new(|stream, symbol, rows| {
        RECORDED.with(|recorded| {
            recorded.borrow_mut().push(Recorded {
                stream: stream.slot(),
                symbol: symbol.to_owned(),
                line: format!(
                    "encoder stream {}\t{symbol}\t{} of {}",
                    stream.slot(),
                    rows.live,
                    rows.capacity
                ),
                plan: None,
            })
        })
    })));
}

/// This thread's records since the last call; later calls also hold the
/// launches of the thread's recorded programs.
pub fn take_recorded() -> Vec<Recorded> {
    observe_programs();
    RECORDED.with(|recorded| recorded.take())
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
