//! Upper-skip plans of short single-item text prompts, kept by their exact
//! lowering inputs and reused instead of lowering every rank again.
//!
//! `skip::Plan::prepare` reads the phase's rows, item extents, prefix
//! lengths, positions, logits rows, input layout and cache range, and
//! otherwise only state the instance fixes at create (buffers, arenas, ops,
//! KV extent, prefill counts, and a DSpark target's tap, feature and Engram
//! staging spans). An eligible run starts its one item at row zero with no
//! prefix and positions `0..rows`, has no prefill cache range, media or
//! position axes, and scores only its last row, so its rows and layout fix
//! every input, a DSpark target's retained feature rows included; they are
//! the key. Token IDs, block tables, slots, window and tail inputs, index
//! tags and Engram rows reach the kernels through the per-step uploads and
//! are never part of a plan.
//!
//! A plan this cache does not keep may still be kept as a template by its
//! structure, with its index selections lowered again each step
//! (`plan_templates`).
//!
//! `prefill_plan_cache_verify` lowers every hit, and every reused template,
//! again and refuses the step if the fresh plan differs from the kept one.
//! It defaults to false.
//!
//! At most `prefill_plan_cache_plans` plans are kept (1 to 64, default 16).
//! Keeping one more drops the least recently used. A dropped plan is
//! lowered again on its next use, with the same launches, so the limit
//! moves only host time and memory.

use super::{PREFILL, Run, plan_templates, skip};
use crate::instance::Instance;
use crate::program::Phase;
use bundle_runtime::upload::InputLayout;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// The longest prompt a plan is kept for. At most one plan of each length
/// is kept; one TP4 Flash plan holds about ten thousand launches in about
/// 20 MiB of host memory.
pub(crate) const MAX_ROWS: u32 = model_deepseek41_contract::execution_options::MAX_PLANS as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    On,
    Verify,
}

#[derive(Debug, Eq, PartialEq)]
struct Key {
    rows: u32,
    layout: InputLayout,
}

struct Kept {
    key: Key,
    plan: Rc<skip::Plan>,
    /// The cache's use count when this plan was last kept or reused.
    used: u64,
}

pub(crate) struct Cache {
    mode: Mode,
    /// At most a plan of each length, `rows - 1` its index.
    plans: RefCell<Vec<Option<Kept>>>,
    /// The most plans kept at once.
    limit: usize,
    uses: Cell<u64>,
    hits: Cell<u64>,
    misses: Cell<u64>,
    evictions: Cell<u64>,
    /// Templates of the plans this cache does not keep.
    templates: plan_templates::Templates,
}

impl Cache {
    pub(crate) fn new(options: &model_deepseek41_contract::execution_options::Options) -> Self {
        let mode = if options.prefill_plan_cache_verify {
            Mode::Verify
        } else {
            Mode::On
        };
        let limit = options.prefill_plan_cache_plans;
        Self {
            mode,
            plans: RefCell::new((0..MAX_ROWS).map(|_| None).collect()),
            limit,
            uses: Cell::new(0),
            hits: Cell::new(0),
            misses: Cell::new(0),
            evictions: Cell::new(0),
            templates: plan_templates::Templates::new(),
        }
    }

    /// Whether every reused plan is lowered again and compared.
    pub(super) fn verifies(&self) -> bool {
        self.mode == Mode::Verify
    }

    /// What the templates did.
    #[cfg(test)]
    pub(super) fn template_counts(&self) -> bundle_runtime::template::Counts {
        self.templates.counts()
    }

    /// Plans served from the cache and plans lowered for it.
    #[cfg(test)]
    fn counts(&self) -> (u64, u64) {
        (self.hits.get(), self.misses.get())
    }

    /// The rows of every kept plan, shortest first, and the plans dropped
    /// so far.
    #[cfg(test)]
    fn kept(&self) -> (Vec<u32>, u64) {
        let rows = self
            .plans
            .borrow()
            .iter()
            .flatten()
            .map(|kept| kept.key.rows)
            .collect();
        (rows, self.evictions.get())
    }

    fn next_use(&self) -> u64 {
        self.uses.set(self.uses.get() + 1);
        self.uses.get()
    }

    /// `run`'s upper-skip plan: the kept plan of its key, or a fresh one,
    /// kept when the run is eligible.
    pub(super) fn plan(&self, instance: &Instance, run: &Run) -> Result<Rc<skip::Plan>, String> {
        let key = key(run);
        let Some(key) = key else {
            return self.templates.plan(instance, run, self.verifies());
        };
        let at = key.rows as usize - 1;
        let kept = match &mut self.plans.borrow_mut()[at] {
            Some(kept) if kept.key == key => {
                kept.used = self.next_use();
                Some(Rc::clone(&kept.plan))
            }
            _ => None,
        };
        if let Some(plan) = kept {
            self.hits.set(self.hits.get() + 1);
            if self.mode == Mode::Verify && !plan.same(&skip::Plan::prepare(instance, run)?)? {
                return Err("a kept prefill plan differs from a fresh lowering".into());
            }
            return Ok(plan);
        }
        let plan = Rc::new(skip::Plan::prepare(instance, run)?);
        // A DMA plan carries transport state of its own; keep only plain ones.
        if plan.uses_dma() {
            return Ok(plan);
        }
        self.misses.set(self.misses.get() + 1);
        let mut plans = self.plans.borrow_mut();
        // A plan of the same length in another layout gives up its place;
        // otherwise a full cache drops its least recently used plan.
        let mut dropped = None;
        if plans[at].is_none() && plans.iter().flatten().count() >= self.limit {
            let oldest = (0..plans.len())
                .filter_map(|index| plans[index].as_ref().map(|kept| (kept.used, index)))
                .min()
                .map(|(_, index)| index);
            if let Some(oldest) = oldest {
                plans[oldest] = None;
                self.evictions.set(self.evictions.get() + 1);
                dropped = Some(oldest + 1);
            }
        }
        plans[at] = Some(Kept {
            key,
            plan: Rc::clone(&plan),
            used: self.next_use(),
        });
        let dropped = dropped.map_or_else(String::new, |rows| format!(", dropped {rows} rows"));
        instance.log(&format!(
            "prefill plan cache: kept {} rows{dropped}; {} of {} plans, {} hits, {} misses, {} \
             dropped",
            at + 1,
            plans.iter().flatten().count(),
            self.limit,
            self.hits.get(),
            self.misses.get(),
            self.evictions.get()
        ));
        Ok(plan)
    }
}

/// `run`'s key, or `None` if its plan may depend on more than the key.
fn key(run: &Run) -> Option<Key> {
    let inputs = &run.inputs;
    let rows = inputs.live_tokens();
    let eligible = run.arena == PREFILL
        && run.phase == Phase::Prefill { rows }
        && (1..=MAX_ROWS).contains(&rows)
        && run.media.is_none()
        && run.cache.is_none()
        && inputs.request_ids.len() == 1
        && inputs.q_start == [0]
        && inputs.q_len == [rows as i32]
        && inputs.prefix_len == [0]
        && inputs.position_axes.is_empty()
        && inputs.positions.iter().copied().eq(0..rows as i32)
        && inputs.logit_rows() == [rows - 1]
        && run.head_rows == 1;
    eligible.then_some(Key {
        rows,
        layout: run.layout,
    })
}

/// Whether two programs issue the same launches with the same arguments as
/// the host resolves them, the same selected routes, the same collectives
/// and the same stages.
pub(super) fn same_programs(a: &super::Programs, b: &super::Programs) -> Result<bool, String> {
    Ok(a.2 == b.2
        && bundle_runtime::template::same_programs(&a.0, &a.1, &b.0, &b.1)
            .map_err(|e| format!("{e:?}"))?)
}

/// The (hits, misses) of the instance `handle`.
#[cfg(test)]
pub(crate) fn counts(handle: runtime_bundle_abi::BundleHandleV1) -> (u64, u64) {
    use bundle_runtime::entry::Bundle;
    let mut counts = None;
    Instance::instances().with(|table| {
        bundle_runtime::table::with_instance(table, handle, |instance| {
            counts = Some(instance.plans.counts());
            runtime_bundle_abi::BundleStatusV1::Success
        })
    });
    counts.expect("a live instance")
}

/// The rows of every plan the instance `handle` keeps, and the plans it
/// dropped so far.
#[cfg(test)]
pub(crate) fn kept(handle: runtime_bundle_abi::BundleHandleV1) -> (Vec<u32>, u64) {
    use bundle_runtime::entry::Bundle;
    let mut kept = None;
    Instance::instances().with(|table| {
        bundle_runtime::table::with_instance(table, handle, |instance| {
            kept = Some(instance.plans.kept());
            runtime_bundle_abi::BundleStatusV1::Success
        })
    });
    kept.expect("a live instance")
}

#[cfg(test)]
mod tests {
    use super::same_programs;
    use crate::program::Stage;
    use bundle_runtime::dispatch::BufferBindings;
    use bundle_runtime::pipeline::PreparedLaunch;
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    /// One launch of `symbol` reading `offset` of a buffer and `scalar`,
    /// bound after `unused` other views, so planner IDs differ.
    fn launch(symbol: &str, offset: u64, scalar: u32, unused: u64) -> PreparedLaunch {
        let mut bindings = BufferBindings::default();
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        for at in 0..unused {
            bindings.bind(DeviceBufferSpanV1::new(buffer, at * 512, 256));
        }
        let view = bindings.bind(DeviceBufferSpanV1::new(buffer, offset, 256));
        PreparedLaunch {
            symbol: symbol.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: false,
            },
            arguments: ArgumentList(vec![Argument::DeviceBuffer(view), Argument::U32(scalar)]),
            bindings,
        }
    }

    fn program(launch: PreparedLaunch, waits: bool) -> super::super::Programs {
        (
            vec![vec![launch]],
            vec![vec![waits]],
            vec![vec![Stage::HeadCollapse]],
        )
    }

    #[test]
    fn programs_compare_by_what_the_host_resolves() {
        let same = |a, b| same_programs(&a, &b).unwrap();
        let base = || program(launch("k", 256, 7, 0), false);
        assert!(same(base(), base()));
        // Other planner IDs of the same views resolve to the same launch.
        assert!(same(base(), program(launch("k", 256, 7, 3), false)));
        assert!(!same(base(), program(launch("j", 256, 7, 0), false)));
        assert!(!same(base(), program(launch("k", 768, 7, 0), false)));
        assert!(!same(base(), program(launch("k", 256, 8, 0), false)));
        assert!(!same(base(), program(launch("k", 256, 7, 0), true)));
    }
}
