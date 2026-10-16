//! The functions every launch the bundle submits or captures goes through:
//! the shared pipeline's and dispatch's. Test builds wrap them to record
//! each launch's rank or stream, symbol and rows (`launch_rows.rs`).

use std::collections::BTreeMap;

use bundle_runtime::dispatch::{self, BufferBindings, DispatchError};
use bundle_runtime::geometry::BundleGeometry;
use bundle_runtime::graphs::GraphCache;
use bundle_runtime::pipeline::{self as shared, Collectives, PreparedLaunch, Program};
use kernel_core::contract::{ArgumentList, LaunchPlan};
use runtime_bundle_abi::{BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesV1};

/// The shared `launch_all`.
pub fn launch_all(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    shared::launch_all(services, stream, symbols, launches)
}

/// The shared `run_ranks`.
#[allow(clippy::too_many_arguments)]
pub fn run_ranks<G: BundleGeometry>(
    services: &HostServicesV1,
    streams: &[BundleStreamHandleV1],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    graph: Option<(u32, u32)>,
    graphs: &mut GraphCache<G>,
    collectives: &Collectives,
    prepare: impl FnMut(usize) -> Result<Program, DispatchError>,
    submitted: impl FnMut(usize),
) -> Result<(), DispatchError> {
    shared::run_ranks(
        services,
        streams,
        symbols,
        graph,
        graphs,
        collectives,
        prepare,
        submitted,
    )
}

/// The shared single launch.
pub fn launch(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    symbol: &str,
    plan: LaunchPlan,
    planned: &ArgumentList,
    bindings: &BufferBindings,
) -> Result<(), DispatchError> {
    dispatch::launch(services, stream, symbols, symbol, plan, planned, bindings)
}
