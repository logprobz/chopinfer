//! Upper-skip plans kept as templates by their structure, at most `LIMIT`
//! at once.
//!
//! `skip::Plan::lower` reads of its run the phase, the input layout, the
//! scored rows, the cached-attention range and the layout `skip::Layout`
//! derives from the items, and of the instance only what create fixes (the
//! list in `plan_cache`). Of the positions, only the index selections read
//! anything: their batches, widths and kernels follow the context. The
//! upper-skip DMA reads none of those launches (`skip::Plan::templatable`).
//! So a plan the short-prompt cache does not keep is kept under every other
//! input (`Key`), and a later run of an equal key lowers only its index
//! selections again from its own positions (`skip::Plan::patch`). Token
//! IDs, block tables, slots, positions, window, tail and index inputs and
//! Engram rows reach the kernels through the per-step uploads, as always.
//! A selection that lowers to another launch count or other collective
//! flags refuses the patch, so the step tries the key's other templates,
//! then lowers in full and keeps one more: a key keeps a template for each
//! set of index batches its contexts take. Anchors key only their packed
//! rows and spare indices; slot seeds and ledger ownership are rebuilt
//! each step outside the template. Plans that read media rows or take
//! position axes are never kept.
//!
//! A template keeps the Engram plan of its first step with prepared
//! gathers, and later steps audit only its index selections against the
//! gathers again (`skip::Plan::engram`).
//!
//! `prefill_plan_cache_verify` lowers every reused template again and
//! makes every reused Engram plan again, and refuses the step if either
//! fresh plan differs. The kept plan stores no request data: anchor slot
//! seeds and ledger request/boundary pairs are rebuilt for each step.

use super::{PREFILL, Run, skip};
use crate::instance::Instance;
use crate::ops::latent::CacheRange;
use crate::program::Phase;
#[cfg(test)]
use bundle_runtime::template::Counts;
use bundle_runtime::template::Templates as Kept;
use bundle_runtime::upload::InputLayout;
use std::rc::Rc;

/// Everything but the positions a kept plan's launches were lowered from.
#[derive(Debug, PartialEq)]
pub(super) struct Key {
    phase: Phase,
    layout: InputLayout,
    head_rows: u32,
    cache: Option<CacheRange>,
    dma: bool,
    structure: skip::Structure,
}

impl Key {
    /// `run`'s key, or `None` if its plan may read more than the key and
    /// the positions.
    fn of(instance: &Instance, run: &Run, layout: &skip::Layout) -> Option<Self> {
        let eligible = run.arena == PREFILL
            && run.phase.prefill()
            && run.media.is_none()
            && run.inputs.position_axes.is_empty();
        if !eligible {
            return None;
        }
        Some(Self {
            phase: run.phase,
            layout: run.layout,
            head_rows: run.head_rows,
            cache: run.cache,
            dma: skip::Plan::dma_applies(instance, run),
            structure: layout.structure(),
        })
    }
}

/// The most templates kept at once. A template of a 512-row chunk after a
/// 100K or 200K context holds about 23 MiB of host memory and one of an
/// 8192-row chunk about 32 MiB, so a full cache holds 1.3 to 1.9 GiB
/// (`fake_host_tests::prefill_plan_templates::template_memory`).
pub(crate) const LIMIT: usize = 64;

pub(super) struct Templates {
    kept: Kept<Key, skip::Plan>,
}

impl Templates {
    pub(super) fn new() -> Self {
        Self {
            kept: Kept::new(LIMIT),
        }
    }

    #[cfg(test)]
    pub(super) fn counts(&self) -> Counts {
        self.kept.counts()
    }

    /// `run`'s upper-skip plan: its key's template with the index
    /// selections lowered again, or a fresh plan, kept when it can be.
    pub(super) fn plan(
        &self,
        instance: &Instance,
        run: &Run,
        verify: bool,
    ) -> Result<Rc<skip::Plan>, String> {
        let layout = skip::Layout::of(instance, run)?;
        let Some(key) = Key::of(instance, run, &layout) else {
            return skip::Plan::lower(instance, run, &layout).map(Rc::new);
        };
        let refused = self.kept.counts().refused;
        if let Some(plan) = self.kept.reuse(&key, |plan| plan.patch(instance, run))? {
            if verify && !plan.same(&skip::Plan::lower(instance, run, &layout)?)? {
                return Err("a reused prefill plan template differs from a fresh lowering".into());
            }
            return Ok(plan);
        }
        let mut plan = skip::Plan::lower(instance, run, &layout)?;
        if !plan.templatable() {
            return Ok(Rc::new(plan));
        }
        plan.keep_engram();
        let plan = Rc::new(plan);
        let dropped = self.kept.keep(key, Rc::clone(&plan));
        let counts = self.kept.counts();
        let refused = if counts.refused > refused {
            ", its key's templates refused"
        } else {
            ""
        };
        let dropped = if dropped { ", dropped the oldest" } else { "" };
        instance.log(&format!(
            "prefill plan templates: kept {} rows{refused}{dropped}; {} of {} templates, {} \
             hits, {} misses, {} busy, {} refused, {} dropped",
            run.phase.rows(),
            self.kept.len(),
            LIMIT,
            counts.hits,
            counts.misses,
            counts.busy,
            counts.refused,
            counts.evicted
        ));
        Ok(plan)
    }
}

/// What the templates of the instance `handle` did; read from a poisoned
/// instance too.
#[cfg(test)]
pub(crate) fn counts(handle: runtime_bundle_abi::BundleHandleV1) -> Counts {
    use bundle_runtime::entry::Bundle;
    Instance::instances().with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .plans
            .template_counts()
    })
}
