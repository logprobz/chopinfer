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
    /// Rank 0's body launches as (symbol, stage), and its head's symbols.
    pub symbols: (Vec<(String, crate::program::Stage)>, Vec<String>),
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
        let skip = run.execution.skip().ok_or("not an upper-skip plan")?;
        let launches = skip.launches();
        let symbols = skip.symbols(0);
        let engram = run
            .execution
            .prefill_engram()
            .map(|_| skip::engram_reuses() > reuses);
        Ok(Planned {
            elapsed,
            launches,
            engram,
            symbols,
        })
    })
}

/// A fresh upper-skip lowering of the original batch `request` on the
/// instance `handle` as lines (`skip::Plan::describe`), then its Engram
/// plan for a run that takes prepared gathers.
pub(crate) fn describe(handle: BundleHandleV1, request: &[u8]) -> Result<Vec<String>, String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        let plan = skip::Plan::prepare(instance, &run)?;
        let mut lines = plan.describe();
        if let Some(gathers) = instance
            .prefill_engram
            .as_ref()
            .filter(|_| run.route.prepared_gathers)
        {
            lines.push(format!("engram {:?}", plan.engram(gathers, false)?));
        }
        Ok(lines)
    })
}

/// Whether two fresh lowerings of the original batch `request` are the
/// same plan (`skip::Plan::same`): as lowered, after the second's first
/// site moves, both keeping the same Engram plan, and after the second's
/// kept Engram plan moves its first DMA join.
pub(crate) fn same(handle: BundleHandleV1, request: &[u8]) -> Result<[bool; 4], String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        let mut a = skip::Plan::prepare(instance, &run)?;
        let mut b = skip::Plan::prepare(instance, &run)?;
        let lowered = a.same(&b)?;
        let mut moved = skip::Plan::prepare(instance, &run)?;
        if !moved.move_first_site() {
            return Err("the plan has no sites".into());
        }
        let site = a.same(&moved)?;
        let gathers = instance
            .prefill_engram
            .as_ref()
            .ok_or("the instance has no prepared gathers")?;
        let engram = a.engram(gathers, false)?;
        a.keep_engram_plan(engram.clone());
        b.keep_engram_plan(engram.clone());
        let kept = a.same(&b)?;
        let mut stale = engram;
        if !stale.move_first_join() {
            return Err("the Engram plan has no DMA join".into());
        }
        b.keep_engram_plan(stale);
        Ok([lowered, site, kept, a.same(&b)?])
    })
}

/// One group of a lowering in groups (`lower_in_groups`).
pub(crate) struct Group {
    /// The group's layer, none for the embedding and the feature packs.
    pub layer: Option<u32>,
    /// Each rank's body launches once the group is lowered.
    pub launches: Vec<usize>,
    /// The median host time of the group's lowering on every rank.
    pub elapsed: Duration,
}

/// A lowering in groups (`lower_in_groups`).
pub(crate) struct InGroups {
    pub groups: Vec<Group>,
    /// The median host time of the layout.
    pub layout: Duration,
    /// The median host time of the head, copies, collective checks and
    /// DMA plan after the last group.
    pub finish: Duration,
    /// Rank 0's body stages of the finished plan.
    pub stages: Vec<crate::program::Stage>,
    /// Each rank's body launches of the finished plan.
    pub launches: Vec<usize>,
}

/// Lowers the upper-skip plan of the original batch `request` one group at
/// a time, `repeats` times (`skip::Plan::lower_in_groups`).
pub(crate) fn lower_in_groups(
    handle: BundleHandleV1,
    request: &[u8],
    repeats: usize,
) -> Result<InGroups, String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        let skip::InGroups {
            groups,
            layout,
            finish,
            plan,
        } = skip::Plan::lower_in_groups(instance, &run, repeats)?;
        let groups = groups
            .into_iter()
            .map(|(layer, launches, elapsed)| Group {
                layer,
                launches,
                elapsed,
            })
            .collect();
        let stages = plan
            .symbols(0)
            .0
            .into_iter()
            .map(|(_, stage)| stage)
            .collect();
        let launches = (0..plan.launches().len())
            .map(|rank| plan.symbols(rank).0.len())
            .collect();
        Ok(InGroups {
            groups,
            layout,
            finish,
            stages,
            launches,
        })
    })
}

/// The lowerings checked of each kind, and every stage and row count of
/// `rows` and `heads` at which the upper-skip lowering finds no route,
/// lowered on rank 0 with the inputs of the original batch `request`,
/// which holds at least `rows`' largest count (`skip::sweep`).
pub(crate) fn sweep(
    handle: BundleHandleV1,
    request: &[u8],
    rows: &[u32],
    heads: &[u32],
) -> Result<([usize; 4], Vec<String>), String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        let layout = skip::Layout::of(instance, &run)?;
        skip::sweep(instance, &run, &layout, rows, heads)
    })
}

/// The DMA blocks prepared after each group of the upper-skip lowering of
/// the original batch `request`, then in all, and whether they equal the
/// whole body's (`skip::Plan::dma_in_groups`).
pub(crate) fn dma_in_groups(
    handle: BundleHandleV1,
    request: &[u8],
) -> Result<(Vec<usize>, bool), String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        skip::Plan::dma_in_groups(instance, &run)
    })
}

/// Whether the Engram plan of the original batch `request`, made as its
/// upper-skip body is lowered in groups, equals the one made from the
/// whole plan (`skip::Plan::engram_in_groups`).
pub(crate) fn engram_in_groups(handle: BundleHandleV1, request: &[u8]) -> Result<bool, String> {
    with(handle, |instance| {
        instance.skip_prefill = true;
        let step = instance.pack(request).map_err(|e| format!("pack: {e:?}"))?;
        let hashed = match (&instance.dspark, &instance.engram) {
            (Some(_), Some(engram)) => Some(engram.hash(&step.prefill)?),
            _ => None,
        };
        let run = instance
            .plan_run(step.prefill, None, hashed)
            .map_err(|e| format!("plan: {e:?}"))?;
        let gathers = instance
            .prefill_engram
            .as_ref()
            .ok_or("the instance has no prepared gathers")?;
        skip::Plan::engram_in_groups(instance, &run, gathers)
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
