//! Test-only launch observation at submission and graph preparation.
//! Model adapters own their records and journal formatting.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::dispatch::DispatchError;
use crate::geometry::BundleGeometry;
use crate::graphs::GraphCache;
use crate::pipeline::{self as shared, Collectives, PreparedLaunch, Program};
use runtime_bundle_abi::{BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesV1};

/// A thread-local collection whose record representation belongs to its caller.
pub struct Records<R>(RefCell<Vec<R>>);

impl<R> Records<R> {
    pub const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    pub fn extend(&self, records: impl IntoIterator<Item = R>) {
        self.0.borrow_mut().extend(records);
    }

    pub fn take(&self) -> Vec<R> {
        self.0.take()
    }
}

impl<R> Default for Records<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Records the list before submitting it, including a submission that fails.
pub fn launch_all(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
    record: impl FnOnce(BundleStreamHandleV1, &[PreparedLaunch]),
) -> Result<(), DispatchError> {
    record(stream, launches);
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
    mut record: impl FnMut(BundleStreamHandleV1, &str, &[PreparedLaunch]),
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
