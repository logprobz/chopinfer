//! One prefill planned on the fake host outside a step, for the template
//! tests and the release benchmark in `fake_host_tests::prefill_plan_templates`.
//! Nothing runs on a device; each plan packs the step again.

use super::skip;
use crate::instance::Instance;
use bundle_runtime::entry::Bundle;
use runtime_bundle_abi::{BundleHandleV1, BundleStatusV1};
use std::time::{Duration, Instant};

fn with<T>(handle: BundleHandleV1, run: impl FnOnce(&mut Instance) -> T) -> T {
    let mut result = None;
    Instance::instances().with(|table| {
        bundle_runtime::table::with_instance(table, handle, |instance| {
            result = Some(run(instance));
            BundleStatusV1::Success
        })
    });
    result.expect("a live instance")
}

/// A prefill planned outside a step (`plan`).
pub(crate) struct Planned {
    pub elapsed: Duration,
    /// Each rank's upper-skip launches.
    pub launches: Vec<usize>,
    /// Whether the step's Engram plan was a kept template's, none without
    /// an Engram plan.
    pub engram: Option<bool>,
}

/// Plans the prefill of the original batch `request` on the instance
/// `handle` as an upper-skip step, as `plan_run` does in execute.
pub(crate) fn plan(handle: BundleHandleV1, request: &[u8]) -> Result<Planned, String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let reuses = skip::engram_reuses();
        let started = Instant::now();
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        let elapsed = started.elapsed();
        let launches = run
            .execution
            .skip()
            .ok_or("not an upper-skip plan")?
            .launches();
        let engram = run
            .execution
            .prefill_engram()
            .map(|_| skip::engram_reuses() > reuses);
        Ok(Planned {
            elapsed,
            launches,
            engram,
        })
    })
}

/// The median host time of `repeats` plans of `request` (`plan`), and of
/// the same upper-skip plans lowered in full each time.
pub(crate) fn time(handle: BundleHandleV1, request: &[u8], repeats: usize) -> (Duration, Duration) {
    let median = |mut times: Vec<Duration>| {
        times.sort_unstable();
        times[times.len() / 2]
    };
    let mut planned = Vec::new();
    let mut lowered = Vec::new();
    for _ in 0..repeats {
        planned.push(plan(handle, request).unwrap().elapsed);
        lowered.push(with(handle, |instance| {
            let step = instance.pack(request).unwrap();
            let hashed = match (&instance.dspark, &instance.engram) {
                (Some(_), Some(engram)) => Some(engram.hash(&step.prefill).unwrap()),
                _ => None,
            };
            let run = instance.plan_run(step.prefill, None, hashed).unwrap();
            let started = Instant::now();
            skip::Plan::prepare(instance, &run).unwrap();
            started.elapsed()
        }));
    }
    (median(planned), median(lowered))
}

/// A prefix restore's bookkeeping without its snapshot copy: `request`
/// holds `slot` as if it had committed `tokens`, so a fake-host step may
/// continue it at a long context. No check reads the fake device's state
/// bytes.
pub(crate) fn restore_without_snapshot(
    handle: BundleHandleV1,
    request: u64,
    slot: u32,
    tokens: &[u32],
) -> Result<(), String> {
    with(handle, |instance| {
        let count = tokens.len() as u64;
        instance
            .ledger
            .check_restore(request, count, Some(slot))
            .map_err(|e| format!("{e:?}"))?;
        if instance.dspark.is_some() {
            instance
                .dspark_restored(request, slot, tokens)
                .map_err(|e| format!("{e:?}"))?;
        } else if let Some(engram) = &mut instance.engram {
            engram
                .restore(request, tokens)
                .map_err(|e| format!("{e:?}"))?;
        }
        instance
            .ledger
            .restore(request, count, Some(slot))
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
    })
}
