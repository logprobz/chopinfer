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

#[cfg(test)]
thread_local! {
    static REFUSED: std::cell::RefCell<Option<(String, u32)>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `run` with the `nth` eager segment (counting from zero) that
/// launches a symbol containing `symbol` refused before any of it is
/// submitted, as a symbol missing from the rank's table would refuse it: a
/// local refusal, which poisons only a unit that already submitted.
#[cfg(test)]
pub(crate) fn with_refused<T>(symbol: &str, nth: u32, run: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            REFUSED.with_borrow_mut(|refused| *refused = None);
        }
    }
    let _reset = Reset;
    REFUSED.with_borrow_mut(|refused| *refused = Some((symbol.to_owned(), nth)));
    run()
}

/// The symbol `with_refused` refuses in `launches`, counting this segment.
#[cfg(test)]
fn refused(launches: &[PreparedLaunch]) -> Option<String> {
    REFUSED.with_borrow_mut(|refused| {
        let (symbol, left) = refused.as_mut()?;
        let launch = launches
            .iter()
            .find(|l| l.symbol.contains(symbol.as_str()))?;
        if *left > 0 {
            *left -= 1;
            return None;
        }
        let name = launch.symbol.clone();
        *refused = None;
        Some(name)
    })
}

pub(super) fn launch(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    symbols: &BTreeMap<String, BundleCudaSymbolHandleV1>,
    launches: &[PreparedLaunch],
) -> Result<(), DispatchError> {
    #[cfg(test)]
    if let Some(symbol) = refused(launches) {
        return Err(DispatchError::MissingSymbol(symbol));
    }
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
