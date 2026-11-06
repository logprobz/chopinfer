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
//!
//! Freeing a dropped template's launches takes about 4 ms at the served
//! long-context shapes. On a DSpark instance a dropped template waits
//! until the step's prefill launched (`release`), so its freeing
//! runs while the device works instead of before the launch; a step that
//! fails before its launch leaves them to the next step's entry.

use super::{PREFILL, Run, skip};
use crate::instance::Instance;
use crate::ops::latent::CacheRange;
use crate::program::Phase;
#[cfg(test)]
use bundle_runtime::template::Counts;
use bundle_runtime::template::Templates as Kept;
use bundle_runtime::upload::InputLayout;
use std::cell::RefCell;
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

#[cfg(test)]
thread_local! { static TEST_LIMIT: std::cell::Cell<usize> = const { std::cell::Cell::new(LIMIT) }; }

/// Runs `run` with instances it creates keeping at most `limit` templates,
/// so a test can drop templates in a few steps.
#[cfg(test)]
pub(crate) fn with_limit<T>(limit: usize, run: impl FnOnce() -> T) -> T {
    struct Reset(usize);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_LIMIT.set(self.0);
        }
    }
    let _reset = Reset(TEST_LIMIT.replace(limit));
    run()
}

/// A run's upper-skip plan: its key's template or a fresh lowering, or a
/// miss to lower while it issues (`prefill_streamed_lowering`).
pub(super) enum Planned {
    Ready(Rc<skip::Plan>),
    Miss(Box<Miss>),
}

/// A run no template serves: its layout, its key when it has one, and
/// whether that key's templates refused the run.
pub(super) struct Miss {
    pub(super) layout: skip::Layout,
    pub(super) key: Option<Key>,
    pub(super) refused: bool,
}

pub(super) struct Templates {
    kept: Kept<Key, skip::Plan>,
    /// Templates dropped to stay within the limit and held until the
    /// step's prefill launched (`release`); empty unless `late`.
    retired: RefCell<Vec<Rc<skip::Plan>>>,
    late: bool,
}

impl Templates {
    /// Templates that free the ones they drop only once the step's prefill
    /// launched when `late`.
    pub(super) fn new(late: bool) -> Self {
        #[cfg(test)]
        let limit = TEST_LIMIT.get();
        #[cfg(not(test))]
        let limit = LIMIT;
        Self {
            kept: Kept::new(limit),
            retired: RefCell::new(Vec::new()),
            late,
        }
    }

    /// Frees the templates this step dropped: after its prefill launched,
    /// or at the next step's entry. The vector keeps its room, so holding
    /// a step's drops allocates nothing once the cache is full.
    pub(super) fn release(&self) {
        let mut retired = self.retired.borrow_mut();
        if !retired.is_empty() {
            retired.clear();
        }
    }

    /// The dropped templates held for the step's launch.
    #[cfg(test)]
    pub(super) fn retired(&self) -> usize {
        self.retired.borrow().len()
    }

    #[cfg(test)]
    pub(super) fn counts(&self) -> Counts {
        self.kept.counts()
    }

    /// The templates kept now, the most kept at once and the counts so far.
    pub(super) fn state(&self) -> crate::dspark::staging::PrefillTemplates {
        crate::dspark::staging::PrefillTemplates {
            templates: self.kept.len(),
            limit: self.kept.limit(),
            counts: self.kept.counts(),
        }
    }

    /// `run`'s upper-skip plan: its key's template with the index
    /// selections lowered again, or a fresh plan, kept when it can be; a
    /// miss when `streamed`.
    pub(super) fn plan(
        &self,
        instance: &Instance,
        run: &Run,
        verify: bool,
        streamed: bool,
    ) -> Result<Planned, String> {
        let layout = skip::Layout::of(instance, run)?;
        let Some(key) = Key::of(instance, run, &layout) else {
            if streamed {
                let (key, refused) = (None, false);
                return Ok(Planned::Miss(Box::new(Miss {
                    layout,
                    key,
                    refused,
                })));
            }
            return skip::Plan::lower(instance, run, &layout)
                .map(|plan| Planned::Ready(Rc::new(plan)));
        };
        let refused = self.kept.counts().refused;
        if let Some(plan) = self.kept.reuse(&key, |plan| plan.patch(instance, run))? {
            if verify && !plan.same(&skip::Plan::lower(instance, run, &layout)?)? {
                return Err("a reused prefill plan template differs from a fresh lowering".into());
            }
            return Ok(Planned::Ready(plan));
        }
        let refused = self.kept.counts().refused > refused;
        if streamed {
            let key = Some(key);
            return Ok(Planned::Miss(Box::new(Miss {
                layout,
                key,
                refused,
            })));
        }
        let plan = skip::Plan::lower(instance, run, &layout)?;
        Ok(Planned::Ready(
            self.keep(instance, run, key, refused, plan, None, self.late),
        ))
    }

    /// Keeps `plan`, lowered while it issued and launched, as a template
    /// of `key`, with the Engram plan `engram` it made; the plan to run. A
    /// template the keep drops is freed once the launch returns
    /// (`release`), as it already has.
    pub(super) fn keep_streamed(
        &self,
        instance: &Instance,
        run: &Run,
        key: Key,
        refused: bool,
        plan: skip::Plan,
        engram: Option<crate::instance::prefill_engram::Plan>,
    ) -> Rc<skip::Plan> {
        self.keep(instance, run, key, refused, plan, engram, true)
    }

    /// Keeps the fresh `plan` as a template of `key` when it can be, with
    /// its Engram plan when it made one, holding a template it drops until
    /// `release` when `hold`.
    #[allow(clippy::too_many_arguments)]
    fn keep(
        &self,
        instance: &Instance,
        run: &Run,
        key: Key,
        refused: bool,
        mut plan: skip::Plan,
        engram: Option<crate::instance::prefill_engram::Plan>,
        hold: bool,
    ) -> Rc<skip::Plan> {
        if !plan.templatable() {
            return Rc::new(plan);
        }
        match engram {
            Some(engram) => plan.keep_engram_plan(engram),
            None => plan.keep_engram(),
        }
        #[cfg(test)]
        skip::record_kept(|| plan.describe_template());
        let plan = Rc::new(plan);
        let dropped = match self.kept.keep(key, Rc::clone(&plan)) {
            Some(old) if hold => {
                self.retired.borrow_mut().push(old);
                true
            }
            dropped => dropped.is_some(),
        };
        let counts = self.kept.counts();
        let refused = if refused {
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
            self.kept.limit(),
            counts.hits,
            counts.misses,
            counts.busy,
            counts.refused,
            counts.evicted
        ));
        plan
    }
}

/// The dropped templates the instance `handle` holds for its next
/// prefill's launch.
#[cfg(test)]
pub(crate) fn retired(handle: runtime_bundle_abi::BundleHandleV1) -> usize {
    use bundle_runtime::entry::Bundle;
    Instance::instances().with(|table| {
        table
            .borrow_mut()
            .get_mut(handle)
            .expect("a live instance")
            .plans
            .retired_templates()
    })
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
