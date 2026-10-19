//! Submit qualified cache producers with their measured stream ordering.
use bundle_runtime::dispatch::{self, DispatchError};
use bundle_runtime::pipeline::{self, PreparedLaunch};
use runtime_bundle_abi::{BundleCudaSymbolHandleV1, BundleStreamHandleV1, HostServicesV1};
use std::collections::BTreeMap;

fn producer(symbol: &str) -> bool {
    matches!(
        symbol,
        "chop_latent_prefill_clear_pages_v1"
            | "chop_latent_prefill_decode_window_d512_v1"
            | "chop_latent_prefill_mark_pages_p32_v1"
            | "chop_latent_prefill_mark_pages_p64_v1"
            | "chop_latent_prefill_decode_pages_d512_p32_v1"
            | "chop_latent_prefill_decode_pages_d512_p64_v1"
    )
}

pub(super) fn launch(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    if launches.iter().all(|l| l.plan.pdl) {
        return pipeline::launch_all(services, stream, symbols, launches);
    }
    if launches.iter().any(|l| !l.plan.pdl && !producer(&l.symbol)) {
        return Err(DispatchError::InvalidPlan);
    }
    // Resolve the entire segment before any submission, preserving the
    // collective unit's refusal/poisoning boundary. Never promote a producer
    // to PDL: its ordinary launch waits for the preceding native KV writes.
    let resolved = launches
        .iter()
        .map(|l| dispatch::resolve(symbols, &l.symbol, l.plan, &l.arguments, &l.bindings))
        .collect::<Result<Vec<_>, _>>()?;
    for call in resolved {
        call.submit(services, stream)?;
    }
    Ok(())
}
