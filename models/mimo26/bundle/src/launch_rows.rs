//! `crate::pipeline` in test builds: the shared pipeline, with every launch
//! list the bundle submits or captures recorded as its rank (or stream),
//! symbol and rows. The fake-host journals do not record
//! `PreparedLaunch::rows`, so tests compare these records to keep the rows
//! of every launch unchanged.

use std::cell::RefCell;
use std::collections::BTreeMap;

pub use bundle_runtime::pipeline::*;

use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::geometry::BundleGeometry;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::pipeline as shared;
use runtime_bundle_abi::{BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesV1};

/// One recorded launch: the slot of the stream it goes to, and its rank
/// (or stream), symbol and rows as one line.
pub struct Recorded {
    pub stream: u32,
    pub symbol: String,
    pub line: String,
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
            }))
    });
}

/// This thread's records since the last call.
pub fn take_recorded() -> Vec<Recorded> {
    RECORDED.with(|recorded| recorded.take())
}

pub fn launch_all(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    record(stream, &format!("stream {}", stream.raw), launches);
    shared::launch_all(services, stream, symbols, launches)
}

#[allow(clippy::too_many_arguments)]
pub fn launch_all_traced(
    services: &HostServicesV1,
    context: runtime_bundle_abi::BundleContextHandleV1,
    stream: BundleStreamHandleV1,
    rank: u32,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
    first: usize,
    trace: &mut Vec<TracedLaunch>,
) -> Result<(), DispatchError> {
    record(stream, &format!("rank {rank}"), launches);
    shared::launch_all_traced(
        services, context, stream, rank, symbols, launches, first, trace,
    )
}

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

pub fn run_rank<G: BundleGeometry>(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    rank: u32,
    graph: Option<(u32, u32)>,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    graphs: &mut GraphCache<G>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    // As the shared `run_rank`, recording only what a capture or an eager
    // run launches: a cached graph replays without launching.
    let launch = |range: std::ops::Range<usize>| {
        record(stream, &format!("rank {rank}"), &launches[range.clone()]);
        shared::launch_all(services, stream, symbols, &launches[range])
    };
    match graph {
        Some((bucket, items)) => graphs.run(
            services,
            stream,
            bundle_runtime::graphs::GraphKey {
                rank,
                bucket,
                items,
            },
            launches.len(),
            launch,
        ),
        None => launch(0..launches.len()),
    }
}
