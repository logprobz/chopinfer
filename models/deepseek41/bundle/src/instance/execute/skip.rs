//! Exact upper-layer intervals, using original packed coordinates for carried
//! state and local row zero for each part's transient compute scratch.

use super::{PREFILL, Programs, Run};
use crate::instance::prefill_engram::{Gathers, Plan as Engram};
use crate::instance::{Instance, dspark::PrefillFeatureRange};
use crate::lower::Lowering;
use crate::ops::Lowered;
use crate::program::{Phase, Stage, Sublayer};
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch, Unit};
use bundle_runtime::template::{self, Patch, Site};
use bundle_runtime::upload::DeviceInputs;
use model_deepseek41_contract::prefill_cone::{self, Item};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span};
use std::cell::OnceCell;
use std::ops::Range;

type Copy = (Span, Span);

mod dma;
mod lower;
mod stream;
#[cfg(test)]
pub(crate) use dma::with_enabled as with_dma;
use lower::Lowerer;
#[cfg(test)]
pub(super) use lower::sweep;
pub(super) use stream::launch as launch_streamed;
#[cfg(test)]
pub(crate) use stream::{
    Failure as StreamedFailure, launches as streamed_launches,
    with_failure as with_streamed_failure, with_lead,
};

#[cfg(test)]
thread_local! { static PATCHES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) }; }

/// Runs `run` with every template reused as kept, its index selections not
/// lowered again, so a test can show the verify mode refuses it.
#[cfg(test)]
pub(crate) fn without_patches<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            PATCHES.set(self.0);
        }
    }
    let _reset = Reset(PATCHES.replace(false));
    run()
}

#[cfg(test)]
thread_local! { static LOWERINGS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

/// The whole plans lowered before their launch (`Plan::lower`) on this
/// thread.
#[cfg(test)]
pub(crate) fn lowerings() -> u64 {
    LOWERINGS.get()
}

#[cfg(test)]
thread_local! { static KEPT: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) }; }

/// Every template kept on this thread since the last take, as lines in
/// order: its description when it was kept (`Plan::describe_template`),
/// then its Engram plan once a step made it after the keep.
#[cfg(test)]
pub(crate) fn take_kept() -> Vec<String> {
    KEPT.take()
}

#[cfg(test)]
pub(super) fn record_kept(lines: Vec<String>) {
    KEPT.with_borrow_mut(|kept| kept.extend(lines));
}

#[cfg(test)]
thread_local! { static ENGRAM_REUSES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
thread_local! { static STALE_ENGRAM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

/// `plan`, its first DMA join moved under `with_stale_engram`.
#[cfg(test)]
fn stale(mut plan: Engram) -> Engram {
    if STALE_ENGRAM.get() {
        assert!(plan.move_first_join(), "a DMA Engram plan with a join");
    }
    plan
}

/// Runs `run` with every reused Engram plan's first DMA join moved to
/// another layer, as a stale kept plan would hold it, so a test can show
/// that the reuse audit accepts it and the verify mode refuses it.
#[cfg(test)]
pub(crate) fn with_stale_engram<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            STALE_ENGRAM.set(self.0);
        }
    }
    let _reset = Reset(STALE_ENGRAM.replace(true));
    run()
}

/// The Engram plans kept templates reused on this thread (`Plan::engram`).
#[cfg(test)]
pub(crate) fn engram_reuses() -> u64 {
    ENGRAM_REUSES.get()
}

pub(super) struct Plan {
    body: Programs,
    /// The body's index selections, the only launches that read the
    /// positions: a kept template lowers them again (`patch`).
    sites: Vec<Site<Selection>>,
    dma: Option<dma::Plan>,
    head: Programs,
    copies: Vec<Vec<Copy>>,
    /// A kept template's Engram plan, made at its first step with prepared
    /// gathers (`engram`); none for a plan not kept.
    engram: Option<OnceCell<Engram>>,
    /// Tells a test when the plan is freed (`super::observe_events`).
    #[cfg(test)]
    _freed: Freed,
}

#[cfg(test)]
struct Freed;

/// A group of a lowering in groups (`Plan::lower_in_groups`): its layer,
/// every rank's body launches after it, and its median host time.
#[cfg(test)]
pub(super) type LoweredGroup = (Option<u32>, Vec<usize>, std::time::Duration);

/// A lowering in groups (`Plan::lower_in_groups`): its groups, and the
/// median host times of its layout and of its finish (the head, copies,
/// collective checks and DMA plan).
#[cfg(test)]
pub(super) struct InGroups {
    pub groups: Vec<LoweredGroup>,
    pub layout: std::time::Duration,
    pub finish: std::time::Duration,
    pub plan: Plan,
}

#[cfg(test)]
impl Drop for Freed {
    fn drop(&mut self) {
        super::event("plan freed");
    }
}

/// An index selection of the body: its layer, and the packed rows of the
/// part whose lowering issued it, none for the full-width layers below the
/// cone. Its batches and their widths follow the positions of those rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Selection {
    layer: u32,
    part: Option<Range<u32>>,
}

/// What an upper-skip lowering reads of its run besides its positions: the
/// items, the anchors, the cone with its scored rows and, with DSpark, each
/// item's retained feature rows. The positions reach only the index
/// selections (`Selection`).
pub(super) struct Layout {
    items: Vec<Item>,
    anchors: Vec<super::anchors::Anchor>,
    cone: prefill_cone::Plan,
    features: Vec<PrefillFeatureRange>,
}

impl Layout {
    pub(super) fn of(instance: &Instance, run: &Run) -> Result<Self, String> {
        let g = instance.config.shape.geometry;
        let inputs = &run.inputs;
        let items: Vec<_> = inputs
            .q_start
            .iter()
            .zip(&inputs.q_len)
            .zip(&inputs.prefix_len)
            .map(|((&start, &len), &prefix)| Item {
                rows: start as u32..(start + len) as u32,
                prefix: prefix as u32,
            })
            .collect();
        let logits = inputs.logit_rows();
        let anchors = super::anchors::anchors(inputs, instance.anchor_spare_count())?;
        let mut ends = vec![None; if anchors.is_empty() { 0 } else { items.len() }];
        for anchor in &anchors {
            ends[anchor.item] = Some(anchor.tokens);
        }
        let cone = prefill_cone::Plan::with_anchors(
            g,
            &items,
            &ends,
            &logits,
            instance.dspark.is_some(),
            64,
        )?;
        let features: Vec<_> = if instance.dspark.is_some() {
            items
                .iter()
                .zip(&cone.endpoints)
                .enumerate()
                .map(|(index, (item, &end))| {
                    let rows = item.rows.end - item.rows.start;
                    let window = g.shape.attention.window;
                    let mut keep = rows - prefill_cone::feature_offset(item.prefix, rows, window);
                    // An anchor item also packs the raw rows a chunk ending at
                    // its anchor retains, for the anchor's history spare.
                    if let Some(&Some(anchor)) = ends.get(index) {
                        let first = prefill_cone::anchor_feature_start(item.prefix, anchor, window);
                        keep = keep.max(end - first);
                    }
                    PrefillFeatureRange {
                        packed_start: item.rows.end - keep,
                        rows: keep,
                        position: u64::from(end - keep),
                        end: u64::from(end),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            items,
            anchors,
            cone,
            features,
        })
    }

    /// The part of the layout a plan's launches other than its index
    /// selections depend on: the items' packed rows, the cone and the
    /// retained feature rows, without the absolute endpoints and feature
    /// positions, which lowering only checks against each other. Anchors
    /// contribute only their packed rows and spare indices; their slot
    /// seeds and ledger ownership are rebuilt outside the kept plan.
    pub(super) fn structure(&self) -> Structure {
        Structure {
            items: self.items.iter().map(|item| item.rows.clone()).collect(),
            anchors: self
                .anchors
                .iter()
                .map(|anchor| (anchor.row, anchor.spare))
                .collect(),
            cone: prefill_cone::Plan {
                endpoints: Vec::new(),
                ..self.cone.clone()
            },
            features: self
                .features
                .iter()
                .map(|feature| (feature.packed_start, feature.rows))
                .collect(),
        }
    }
}

/// `Layout::structure`.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct Structure {
    items: Vec<Range<u32>>,
    anchors: Vec<(u32, u32)>,
    cone: prefill_cone::Plan,
    features: Vec<(u32, u32)>,
}

fn byte_range(span: Span, first: u64, bytes: u64) -> Result<Span, String> {
    crate::dspark::device::slice(span, first, bytes)
}

pub(crate) fn count(instance: &Instance, rank: usize, value: u32) -> Result<Span, String> {
    if value > PREFILL_TOKENS {
        return Err("prefill count exceeds prepared capacity".into());
    }
    byte_range(
        *instance
            .prefill_counts
            .get(rank)
            .ok_or("prefill counts absent")?,
        u64::from(value) * 4,
        4,
    )
}

fn part(
    instance: &Instance,
    rank: usize,
    full: DeviceInputs,
    rows: &Range<u32>,
) -> Result<DeviceInputs, String> {
    let bytes = u64::from(rows.end - rows.start) * 4;
    let first = u64::from(rows.start) * 4;
    Ok(DeviceInputs {
        token_ids: byte_range(full.token_ids, first, bytes)?,
        positions: byte_range(full.positions, first, bytes)?,
        token_batch: byte_range(full.token_batch, first, bytes)?,
        live_tokens: count(instance, rank, rows.end - rows.start)?,
        ..full
    })
}

fn push(
    program: &mut (Vec<PreparedLaunch>, Vec<bool>, Vec<Stage>),
    stage: Stage,
    lowered: Vec<Lowered>,
) {
    for op in lowered {
        program.0.push(op.launch);
        program.1.push(op.collective);
        program.2.push(stage);
    }
}

fn required(lowered: Option<Vec<Lowered>>, stage: Stage) -> Result<Vec<Lowered>, String> {
    lowered.ok_or_else(|| format!("upper-layer skip stage {stage:?} is not lowered"))
}

fn prefix(layer: u32) -> [Stage; 3] {
    [
        Stage::HcMix(layer, Sublayer::Attention),
        Stage::InputNorm(layer, Sublayer::Attention),
        Stage::QuantizeInput(layer, Sublayer::Attention),
    ]
}

fn compressed(stage: Stage) -> bool {
    matches!(
        stage,
        Stage::Compress(_)
            | Stage::IndexKeyWrite(_)
            | Stage::CompressedKvWrite(_)
            | Stage::Project(_, "compressor.kv" | "compressor.gate" | "indexer.k")
    )
}

fn lowering<'a>(
    instance: &'a Instance,
    run: &Run,
    rank: usize,
    input: &'a DeviceInputs,
    origin: Option<u32>,
) -> Lowering<'a> {
    Lowering {
        prefill_row_origin: origin,
        shape: &instance.config.shape,
        ops: &instance.ops,
        buffers: &instance.buffers[rank],
        peers: &instance.buffers,
        inputs: input,
        rank: rank as u32,
        kv: instance.kv_extent(),
        head_rows: run.head_rows,
        prefill_hc_fusion: false,
    }
}

impl Plan {
    pub(super) fn prepare(instance: &Instance, run: &Run) -> Result<Self, String> {
        Self::lower(instance, run, &Layout::of(instance, run)?)
    }

    /// The plan of `run`, whose layout is `layout`: the lowerer drained,
    /// then the head and row copies (`Lowerer::finish`).
    pub(super) fn lower(instance: &Instance, run: &Run, layout: &Layout) -> Result<Self, String> {
        #[cfg(test)]
        LOWERINGS.set(LOWERINGS.get() + 1);
        Lowerer::new(instance, run, layout)?.finish(instance, run, layout)
    }

    /// Whether `other` issues the same launches, transport, copies and head
    /// at the same index-selection sites, with the same Engram plan where
    /// both are templates that made one.
    pub(super) fn same(&self, other: &Self) -> Result<bool, String> {
        let dma = match (&self.dma, &other.dma) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b)?,
            _ => false,
        };
        let engram = match (
            self.engram.as_ref().and_then(OnceCell::get),
            other.engram.as_ref().and_then(OnceCell::get),
        ) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        Ok(dma
            && engram
            && self.sites == other.sites
            && self.copies == other.copies
            && super::plan_cache::same_programs(&self.body, &other.body)?
            && super::plan_cache::same_programs(&self.head, &other.head)?)
    }

    /// Whether this plan may be kept as a template: its sites hold, and
    /// none lies among the launches its DMA reads or replaces, so a patch
    /// leaves the DMA as a fresh lowering would plan it.
    pub(super) fn templatable(&self) -> bool {
        template::check_sites(&self.body.0, &self.body.1, &self.sites).is_ok()
            && self.dma.as_ref().is_none_or(|dma| {
                self.sites.iter().all(|site| {
                    dma.reads(site.rank)
                        .iter()
                        .all(|read| site.at + site.len <= read.start || read.end <= site.at)
                })
            })
    }

    /// Lowers the body's index selections again from `run`'s positions, as
    /// `lower` does; `run` has this plan's layout structure.
    pub(super) fn patch(&mut self, instance: &Instance, run: &Run) -> Result<Patch, String> {
        #[cfg(test)]
        if !PATCHES.get() {
            return Ok(Patch::Applied);
        }
        let positions = &run.inputs.positions;
        let patched = template::patch(&mut self.body.0, &self.body.1, &self.sites, |site| {
            let stage = Stage::IndexSelect(site.input.layer);
            let full = DeviceInputs::within(instance.arenas[PREFILL][site.rank], &run.layout);
            let lowered = match &site.input.part {
                None => lowering(instance, run, site.rank, &full, None)
                    .stage_with_prefill_context(run.phase, stage, positions, run.cache)?,
                Some(range) => {
                    let input = part(instance, site.rank, full, range)?;
                    let phase = Phase::Prefill {
                        rows: range.end - range.start,
                    };
                    lowering(instance, run, site.rank, &input, Some(range.start))
                        .stage_with_prefill_context(
                            phase,
                            stage,
                            &positions[range.start as usize..range.end as usize],
                            run.cache,
                        )?
                }
            };
            Ok::<_, String>(
                required(lowered, stage)?
                    .into_iter()
                    .map(|op| (op.launch, op.collective))
                    .collect(),
            )
        })?;
        // A site holding a collective keeps the lowering's cross-rank check.
        let collective = self
            .sites
            .iter()
            .any(|site| self.body.1[site.rank][site.at..site.at + site.len].contains(&true));
        if patched == Patch::Applied && collective {
            bundle_runtime::pipeline::check_collectives(&self.body.0, &self.body.1)
                .map_err(|e| format!("prefill body collectives: {e:?}"))?;
        }
        Ok(patched)
    }

    /// Each rank's body and head launches.
    #[cfg(test)]
    pub(super) fn launches(&self) -> Vec<usize> {
        self.body
            .0
            .iter()
            .zip(&self.head.0)
            .map(|(body, head)| body.len() + head.len())
            .collect()
    }

    /// Lowers the plan of `run` one group at a time, `repeats` times
    /// (`InGroups`): each group's layer (`Lowerer::lowered`), every rank's
    /// body launches after it and its median host time.
    #[cfg(test)]
    pub(super) fn lower_in_groups(
        instance: &Instance,
        run: &Run,
        repeats: usize,
    ) -> Result<InGroups, String> {
        use std::time::{Duration, Instant};
        let median = |times: &mut Vec<Duration>| {
            times.sort_unstable();
            times[times.len() / 2]
        };
        let mut groups = Vec::new();
        let mut times: Vec<Vec<Duration>> = Vec::new();
        let (mut layouts, mut finishes) = (Vec::new(), Vec::new());
        let mut plan = None;
        for _ in 0..repeats.max(1) {
            let started = Instant::now();
            let layout = Layout::of(instance, run)?;
            layouts.push(started.elapsed());
            let mut lowering = Lowerer::new(instance, run, &layout)?;
            groups.clear();
            for at in 0.. {
                let started = Instant::now();
                if !lowering.lower_next(instance, run, &layout)? {
                    break;
                }
                let elapsed = started.elapsed();
                if times.len() == at {
                    times.push(Vec::new());
                }
                times[at].push(elapsed);
                groups.push(lowering.lowered(&layout));
            }
            let started = Instant::now();
            plan = Some(lowering.finish(instance, run, &layout)?);
            finishes.push(started.elapsed());
        }
        let groups = groups
            .into_iter()
            .zip(&mut times)
            .map(|((launches, layer), times)| (layer, launches, median(times)))
            .collect();
        Ok(InGroups {
            groups,
            layout: median(&mut layouts),
            finish: median(&mut finishes),
            plan: plan.expect("one lowering"),
        })
    }

    /// Prepares `run`'s DMA plan as its body is lowered one group at a time
    /// (`dma::Builder`): the blocks prepared after each group, and whether
    /// the plan equals the one prepared from the whole body.
    #[cfg(test)]
    pub(super) fn dma_in_groups(
        instance: &Instance,
        run: &Run,
    ) -> Result<(Vec<usize>, bool), String> {
        let layout = Layout::of(instance, run)?;
        let mut lowerer = Lowerer::new(instance, run, &layout)?;
        let mut builder = dma::Builder::new(instance, run).ok_or("the run takes no DMA")?;
        let mut ready = Vec::new();
        while lowerer.lower_next(instance, run, &layout)? {
            builder.advance(instance, lowerer.body(), false)?;
            ready.push(builder.blocks());
        }
        builder.advance(instance, lowerer.body(), true)?;
        ready.push(builder.blocks());
        let streamed = builder.finish(lowerer.body()).ok_or("no DMA block")?;
        let whole = dma::Plan::prepare(instance, run, lowerer.body())?.ok_or("no DMA block")?;
        Ok((ready, streamed.same(&whole)?))
    }

    /// Makes `run`'s Engram plan as its body is lowered one group at a time
    /// (`prefill_engram::Builder`), splitting each DMA block as it is
    /// prepared, and whether it equals the plan made from the whole plan
    /// (`engram`).
    #[cfg(test)]
    pub(super) fn engram_in_groups(
        instance: &Instance,
        run: &Run,
        gathers: &Gathers,
    ) -> Result<bool, String> {
        let layout = Layout::of(instance, run)?;
        let mut lowerer = Lowerer::new(instance, run, &layout)?;
        let mut builder = dma::Builder::new(instance, run);
        let mut engram =
            crate::instance::prefill_engram::Builder::new(gathers, instance.buffers.len());
        let mut split = 0;
        loop {
            let more = lowerer.lower_next(instance, run, &layout)?;
            let body = lowerer.body();
            engram.advance(gathers, &body.0, &body.1, &body.2)?;
            if let Some(dma) = &mut builder {
                dma.advance(instance, body, !more)?;
                while split < dma.blocks() {
                    dma.split_engram(split, &mut engram, gathers)?;
                    split += 1;
                }
            }
            if !more {
                break;
            }
        }
        let plan = lowerer.finish(instance, run, &layout)?;
        let dma = builder.and_then(|builder| builder.finish(&plan.body));
        if let Some(dma) = &dma {
            engram.split_dma(dma.tail())?;
        }
        let streamed = engram.finish(&plan.body.0, &plan.body.1)?;
        if dma.is_some() {
            streamed.check_dma_collectives(&plan.body.0, &plan.body.1)?;
        }
        Ok(streamed == plan.engram(gathers, false)?)
    }

    /// Rank `rank`'s body launches as (symbol, stage), then its head's
    /// symbols.
    #[cfg(test)]
    pub(super) fn symbols(&self, rank: usize) -> (Vec<(String, Stage)>, Vec<String>) {
        let body = self.body.0[rank]
            .iter()
            .zip(&self.body.2[rank])
            .map(|(launch, &stage)| (launch.symbol.clone(), stage))
            .collect();
        let head = self.head.0[rank].iter().map(|l| l.symbol.clone()).collect();
        (body, head)
    }

    /// The plan as lines, for the metadata pins: every body and head launch
    /// with its stage and collective flag, the sites, the DMA plan and the
    /// row copies.
    #[cfg(test)]
    pub(super) fn describe(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (name, programs) in [("body", &self.body), ("head", &self.head)] {
            for (rank, launches) in programs.0.iter().enumerate() {
                for (at, launch) in launches.iter().enumerate() {
                    lines.push(format!(
                        "{name} {rank} {at} {:?} {} {}",
                        programs.2[rank][at],
                        programs.1[rank][at],
                        super::dma::describe_launch(launch)
                    ));
                }
            }
        }
        for site in &self.sites {
            lines.push(format!(
                "site {} {} {} {} {:?}",
                site.rank, site.at, site.len, site.input.layer, site.input.part
            ));
        }
        if let Some(dma) = &self.dma {
            dma.describe(&mut lines);
        }
        for (rank, copies) in self.copies.iter().enumerate() {
            for (destination, source) in copies {
                lines.push(format!("copy {rank} {destination:?} {source:?}"));
            }
        }
        lines
    }

    /// The template as lines (`describe`), then its kept Engram plan once a
    /// step made it.
    #[cfg(test)]
    pub(super) fn describe_template(&self) -> Vec<String> {
        let mut lines = self.describe();
        if let Some(engram) = self.engram.as_ref().and_then(OnceCell::get) {
            lines.push(format!("engram {engram:?}"));
        }
        lines
    }

    pub(super) fn uses_dma(&self) -> bool {
        self.dma.is_some()
    }

    /// Whether `run`'s body may take upper-skip DMA (`dma::applies`).
    pub(super) fn dma_applies(instance: &Instance, run: &Run) -> bool {
        dma::applies(instance, run)
    }

    /// Whether a text prefill of `requests` requests may take upper-skip
    /// DMA (`dma::admits`).
    pub(super) fn dma_admits(instance: &Instance, requests: usize) -> bool {
        dma::admits(instance, requests)
    }

    /// Moves the first site to the next layer, as a template of another
    /// structure would hold it; whether there was one.
    #[cfg(test)]
    pub(super) fn move_first_site(&mut self) -> bool {
        self.sites
            .first_mut()
            .map(|site| site.input.layer += 1)
            .is_some()
    }

    /// Keeps `plan` as this template's Engram plan.
    pub(super) fn keep_engram_plan(&mut self, plan: Engram) {
        self.engram = Some(OnceCell::from(plan));
    }

    /// Keeps this template's first Engram plan for its later steps
    /// (`engram`).
    pub(super) fn keep_engram(&mut self) {
        self.engram = Some(OnceCell::new());
    }

    /// The body's Engram plan with `gathers`. A kept template made it at its
    /// first step with them and has since changed only its index
    /// selections, so it audits only those launches again
    /// (`prefill_engram::Plan::reuse`) and checks the collectives of its DMA
    /// pieces again; its DMA's own launches and the gathers' spans are those
    /// it planned with. `verify` makes the plan in full too and refuses a
    /// differing one.
    pub(super) fn engram(&self, gathers: &Gathers, verify: bool) -> Result<Engram, String> {
        let fresh = || {
            let mut plan = Engram::new(gathers, &self.body.0, &self.body.1, &self.body.2)?;
            self.prepare_engram(&mut plan, gathers)?;
            Ok::<_, String>(plan)
        };
        let Some(kept) = &self.engram else {
            return fresh();
        };
        let Some(plan) = kept.get() else {
            let plan = fresh()?;
            #[cfg(test)]
            record_kept(vec![format!("engram {plan:?}")]);
            return Ok(kept.get_or_init(|| plan).clone());
        };
        let sites = self
            .sites
            .iter()
            .map(|site| (site.rank, site.at..site.at + site.len));
        let plan = plan.reuse(gathers, &self.body.0, &self.body.1, sites)?;
        #[cfg(test)]
        let plan = stale(plan);
        if self.dma.is_some() {
            plan.check_dma_collectives(&self.body.0, &self.body.1)?;
        }
        if verify && plan != fresh()? {
            return Err("a reused prefill Engram plan differs from a fresh one".into());
        }
        #[cfg(test)]
        ENGRAM_REUSES.set(ENGRAM_REUSES.get() + 1);
        Ok(plan)
    }

    fn prepare_engram(&self, plan: &mut Engram, gathers: &Gathers) -> Result<(), String> {
        if let Some(dma) = &self.dma {
            dma.prepare_engram(plan, gathers)?;
            plan.check_dma_collectives(&self.body.0, &self.body.1)?;
        }
        self.audit_head(&gathers.pending_spans(plan.layer_count()))
    }

    /// The head and its row copies run beside a mixed step's verify-row
    /// gathers `verify`, which begin after the body's last Engram join:
    /// none touches their spans.
    fn audit_head(&self, verify: &[Span]) -> Result<(), String> {
        if verify.is_empty() {
            return Ok(());
        }
        for (head, copies) in self.head.0.iter().zip(&self.copies) {
            crate::shared_overlap::disjoint_copy_spans(verify, head)?;
            let aliased = copies
                .iter()
                .flat_map(|&(to, from)| [to, from])
                .any(|span| {
                    verify.iter().any(|pending| {
                        pending.buffer == span.buffer
                            && pending.length != 0
                            && span.length != 0
                            && pending.offset < span.offset.saturating_add(span.length)
                            && span.offset < pending.offset.saturating_add(pending.length)
                    })
                });
            if aliased {
                return Err("a prefill row copy aliases pending verify Engram rows".into());
            }
        }
        Ok(())
    }

    /// Issues the prefill as one unit: the body, the row copies and the
    /// head. Once any of it is submitted, the step's ring, KV and
    /// compressor writes have begun, so a later refusal, failed copy or
    /// panic poisons the instance instead of leaving the step to cancel.
    pub(super) fn execute(
        &self,
        instance: &Instance,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
    ) -> Result<(), BundleAbiError> {
        let _unwinding = PoisonOnUnwind(&instance.collectives);
        let mut unit = Unit::default();
        self.execute_body(instance, gathers, &mut unit)?;
        self.compact_rows(instance, &mut unit)?;
        self.execute_head(instance, &mut unit)
    }

    /// The body: through upper-skip DMA, between prepared gather joins, or
    /// in one eager unit, all part of `unit`.
    fn execute_body(
        &self,
        instance: &Instance,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
        unit: &mut Unit,
    ) -> Result<(), BundleAbiError> {
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        if let Some(dma) = &self.dma {
            // The DMA body poisons on any failure; once it ran, so does
            // every later failure of the unit.
            dma.execute(instance, &self.body, gathers)?;
            unit.record(&instance.collectives, Ok(()))
                .map_err(|e: DispatchError| instance.dispatch_failed(e))
        } else {
            if let Some(gathers) = gathers {
                gathers.execute(instance, unit, &self.body.0, &self.body.1)
            } else {
                instance.launch_eager_in(unit, &streams, &self.body.0, &self.body.1)
            }
            .map_err(|e| instance.dispatch_failed(e))
        }
    }

    /// Moves each scored row to its logits position for the head.
    fn compact_rows(&self, instance: &Instance, unit: &mut Unit) -> Result<(), BundleAbiError> {
        for (rank, copies) in self.copies.iter().enumerate() {
            for &(destination, source) in copies {
                let copied = instance.services.copy_device_to_device(
                    instance.ranks[rank].2,
                    destination,
                    source,
                );
                unit.record(&instance.collectives, copied.map_err(DispatchError::from))
                    .map_err(|e| instance.dispatch_failed(e))?;
            }
        }
        Ok(())
    }

    fn execute_head(&self, instance: &Instance, unit: &mut Unit) -> Result<(), BundleAbiError> {
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        instance
            .launch_eager_in(unit, &streams, &self.head.0, &self.head.1)
            .map_err(|e| instance.dispatch_failed(e))
    }
}
