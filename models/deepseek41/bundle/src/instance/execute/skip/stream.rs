//! An upper-skip prefill lowered while it issues (`prefill_streamed_lowering`).
//!
//! The body is lowered one group at a time (`Lowerer`). Each issue unit of
//! the whole-plan launch (`Plan::execute`) issues once every rank lowered
//! it: a DMA block once its call and the launches its consumers read are
//! prepared (`dma::Builder`), in the pieces the Engram joins split it into;
//! otherwise each collective round of the body, or of a prepared-gather
//! piece after its join. These are the whole-plan launch's units, rounds,
//! joins and calls, in its order and through one unit, so the host makes
//! the same submissions in the same order and only the lowering moves
//! between them. A unit issues once `LEAD` groups were lowered since it
//! could, so a peer-copy drain returns to units already lowered.
//!
//! Before the first submission a lowering or audit error refuses the step
//! as the whole-plan lowering does; after it, it poisons the instance
//! (`Unit::record`). A DMA body poisons on any failure of its own
//! submissions, its first included, as the whole-plan DMA launch does. The
//! head and its row copies issue last, and the complete plan and its
//! Engram plan return for the plan cache to keep.

use super::{Layout, Plan, dma, lower::Lowerer};
use crate::instance::Instance;
use crate::instance::execute::{Executor, PREFILL, Run};
use crate::instance::prefill_engram::{
    Builder as EngramBuilder, DmaPiece, Gathers, Plan as Engram,
};
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::pipeline::{PoisonOnUnwind, Unit};
use model_deepseek41_contract::serving::{PREFILL_TOKENS, SCORED_ROWS};
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1, BundleStreamHandleV1};
use std::collections::VecDeque;

/// A unit issues once this many groups, its own included, were lowered
/// since it could; the first unit and those after the last group issue at
/// once.
const LEAD: usize = 2;

#[cfg(test)]
thread_local! { static TEST_LEAD: std::cell::Cell<usize> = const { std::cell::Cell::new(LEAD) }; }

/// Runs `run` with streamed launches issuing each unit once `lead` groups
/// were lowered since it could, so a test or benchmark can compare leads.
#[cfg(test)]
pub(crate) fn with_lead<T>(lead: usize, run: impl FnOnce() -> T) -> T {
    struct Reset(usize);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_LEAD.set(self.0);
        }
    }
    let _reset = Reset(TEST_LEAD.replace(lead.max(1)));
    run()
}

#[cfg(test)]
thread_local! { static LAUNCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

/// The streamed launches that completed on this thread.
#[cfg(test)]
pub(crate) fn launches() -> u64 {
    LAUNCHES.get()
}

/// Where a test fails a streamed launch as an error of its lowering or
/// audits would (`with_failure`).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Failure {
    /// The lowering of group `n`, counting from zero.
    Group(usize),
    /// The head's audit, once the body issued.
    Head,
    /// The prepared gathers failing before group `n` is lowered, as a failed
    /// host callback leaves them.
    Gathers(usize),
    /// A panic in the lowering of group `n`.
    Panic(usize),
    /// Group `n`'s last launch on rank 0 also reading the last Engram
    /// layer's ids, which its gather still holds, for the pending-span audit
    /// to refuse.
    Alias(usize),
}

#[cfg(test)]
thread_local! { static FAILURE: std::cell::Cell<Option<Failure>> = const { std::cell::Cell::new(None) }; }

/// Runs `run` with each streamed launch failing at `failure`.
#[cfg(test)]
pub(crate) fn with_failure<T>(failure: Failure, run: impl FnOnce() -> T) -> T {
    struct Reset(Option<Failure>);
    impl Drop for Reset {
        fn drop(&mut self) {
            FAILURE.set(self.0);
        }
    }
    let _reset = Reset(FAILURE.replace(Some(failure)));
    run()
}

/// The failure `with_failure` injects at `at`.
#[cfg(test)]
fn injected(at: Failure) -> Result<(), String> {
    if FAILURE.get() == Some(at) {
        return Err(format!("an injected streamed failure at {at:?}"));
    }
    Ok(())
}

fn lead() -> usize {
    #[cfg(test)]
    return TEST_LEAD.get();
    #[cfg(not(test))]
    LEAD
}

/// Lowers `run`'s upper-skip plan, of layout `layout`, while it issues:
/// the complete plan and, for a run with prepared gathers, its Engram plan.
pub(in crate::instance::execute) fn launch(
    instance: &Instance,
    run: &Run,
    layout: &Layout,
) -> Result<(Plan, Option<Engram>), BundleAbiError> {
    if instance.collectives.is_poisoned() {
        return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
    }
    let streams: Vec<_> = instance.ranks.iter().map(|rank| rank.2).collect();
    let _unwinding = PoisonOnUnwind(&instance.collectives);
    let mut unit = Unit::default();
    validate(instance, run, layout).map_err(|error| fail(instance, &mut unit, error))?;
    let lowerer =
        Lowerer::new(instance, run, layout).map_err(|error| fail(instance, &mut unit, error))?;
    let gathers = instance
        .prefill_engram
        .as_ref()
        .filter(|_| run.route.prepared_gathers)
        .map(|gathers| (gathers, gathers.sequence()));
    let mut stream = Stream {
        instance,
        run,
        layout,
        streams: &streams,
        unit,
        lowerer,
        gathers,
        engram: gathers.map(|(gathers, _)| EngramBuilder::new(gathers, streams.len())),
        dma: dma::Builder::new(instance, run),
        dma_plan: None,
        groups: 0,
        issued: vec![0; streams.len()],
        joins: 0,
        blocks: 0,
        ready: Vec::new(),
        frontiers: VecDeque::new(),
        started: false,
    };
    loop {
        let more = stream.lower()?;
        stream.issue(!more)?;
        if !more {
            break;
        }
    }
    let launched = stream.finish()?;
    #[cfg(test)]
    LAUNCHES.set(LAUNCHES.get() + 1);
    Ok(launched)
}

/// `error` as the step's failure: a refusal before the prefill's first
/// submission, poisoning the instance after it.
fn fail(instance: &Instance, unit: &mut Unit, error: String) -> BundleAbiError {
    instance.log(&error);
    let recorded = unit.record::<()>(&instance.collectives, Err(DispatchError::InvalidPlan));
    instance.dispatch_failed(recorded.expect_err("a recorded failure"))
}

/// The checks of a streamed miss before its first submission: whatever its
/// lowering would refuse mid-launch for its rows, layout, head, anchors,
/// features and counts.
fn validate(instance: &Instance, run: &Run, layout: &Layout) -> Result<(), String> {
    let rows = run.phase.rows();
    if rows == 0 || rows > PREFILL_TOKENS || run.inputs.positions.len() != rows as usize {
        return Err("a streamed prefill's rows lie outside its prepared capacity".into());
    }
    let cone = &layout.cone;
    if run.head_rows > SCORED_ROWS
        || cone.logits.len() > SCORED_ROWS as usize
        || cone.logits.iter().any(|&row| row >= rows)
    {
        return Err("a streamed prefill's scored rows lie outside its head".into());
    }
    for layer in &cone.layers {
        for part in layer
            .keys
            .parts(4096)
            .iter()
            .chain(&layer.outputs.parts(4096))
        {
            if part.is_empty() || part.end > rows {
                return Err(format!(
                    "a streamed prefill's layer {} part {part:?} lies outside its rows",
                    layer.id
                ));
            }
        }
    }
    for feature in &layout.features {
        if feature.rows > PREFILL_TOKENS
            || feature
                .packed_start
                .checked_add(feature.rows)
                .is_none_or(|end| end > rows)
        {
            return Err("a streamed prefill's feature rows lie outside its rows".into());
        }
    }
    for anchor in &layout.anchors {
        if anchor.spare >= instance.anchor_spare_count() || anchor.row == 0 || anchor.row > rows {
            return Err("a streamed prefill's anchor lies outside its spares or rows".into());
        }
    }
    if instance.arenas[PREFILL]
        .iter()
        .any(|arena| run.layout.end > arena.arena.length)
    {
        return Err("a streamed prefill's inputs exceed their arena".into());
    }
    Ok(())
}

struct Stream<'a> {
    instance: &'a Instance,
    run: &'a Run,
    layout: &'a Layout,
    streams: &'a [BundleStreamHandleV1],
    unit: Unit,
    lowerer: Lowerer,
    /// The prepared gathers and the begin this step holds.
    gathers: Option<(&'a Gathers, u64)>,
    engram: Option<EngramBuilder>,
    dma: Option<dma::Builder>,
    dma_plan: Option<dma::Plan>,
    /// Groups lowered so far.
    groups: usize,
    /// Each rank's body launches issued, without DMA, and the joins issued.
    issued: Vec<usize>,
    joins: usize,
    /// DMA blocks issued, and the groups lowered when each could issue.
    blocks: usize,
    ready: Vec<usize>,
    /// Each rank's lowered body after each of the last `lead()` groups.
    frontiers: VecDeque<Vec<usize>>,
    /// Whether any unit was submitted.
    started: bool,
}

impl Stream<'_> {
    fn fail(&mut self, error: String) -> BundleAbiError {
        fail(self.instance, &mut self.unit, error)
    }

    /// Lowers the next group, then finds its Engram consumers and DMA
    /// blocks; false once every group is lowered.
    fn lower(&mut self) -> Result<bool, BundleAbiError> {
        let lowered = (|| {
            #[cfg(test)]
            {
                injected(Failure::Group(self.groups))?;
                if FAILURE.get() == Some(Failure::Gathers(self.groups))
                    && let Some((gathers, _)) = self.gathers
                {
                    gathers.fail();
                }
                assert_ne!(
                    FAILURE.get(),
                    Some(Failure::Panic(self.groups)),
                    "an injected streamed panic"
                );
            }
            let more = self
                .lowerer
                .lower_next(self.instance, self.run, self.layout)?;
            #[cfg(test)]
            if FAILURE.get() == Some(Failure::Alias(self.groups))
                && let (Some((gathers, _)), Some(engram)) = (self.gathers, &self.engram)
                && let Some(launch) = self.lowerer.last_mut(0)
            {
                let ids = gathers.pending_spans(engram.layers() - 1)[0];
                let id = launch.bindings.bind(ids);
                launch
                    .arguments
                    .0
                    .push(kernel_core::contract::Argument::DeviceBuffer(id));
            }
            let body = self.lowerer.body();
            if let (Some((gathers, sequence)), Some(engram)) = (self.gathers, &mut self.engram) {
                if !gathers.owned_by(sequence, self.joins) {
                    return Err("prefill Engram gathers changed while their plan was made".into());
                }
                engram.advance(gathers, &body.0, &body.1, &body.2)?;
            }
            if let Some(builder) = &mut self.dma {
                let before = builder.blocks();
                builder.advance(self.instance, body, !more)?;
                let groups = self.groups + usize::from(more);
                self.ready
                    .extend((before..builder.blocks()).map(|_| groups));
            }
            Ok::<_, String>(more)
        })();
        let more = lowered.map_err(|error| self.fail(error))?;
        if more {
            self.groups += 1;
        }
        let lowered = self.lowerer.body().0.iter().map(Vec::len).collect();
        self.frontiers.push_back(lowered);
        while self.frontiers.len() > lead() {
            self.frontiers.pop_front();
        }
        Ok(more)
    }

    /// Issues every unit the lowered body holds and the lead admits, every
    /// one once the body is `complete`.
    fn issue(&mut self, complete: bool) -> Result<(), BundleAbiError> {
        // A body that takes DMA issues nothing before its first block; one
        // whose reductions admit no call issues as a plan without DMA.
        match &self.dma {
            Some(builder) if builder.blocks() > 0 || !complete => self.issue_blocks(complete),
            _ => self.issue_rounds(complete),
        }
    }

    fn issue_blocks(&mut self, complete: bool) -> Result<(), BundleAbiError> {
        let lead = lead();
        loop {
            let Some(builder) = &self.dma else {
                return Ok(());
            };
            let index = self.blocks;
            if index == builder.blocks() {
                break;
            }
            if self.started && !complete && self.ready[index] + lead > self.groups + 1 {
                return Ok(());
            }
            let pieces = match (&mut self.engram, self.gathers) {
                (Some(engram), Some((gathers, _))) => builder.split_engram(index, engram, gathers),
                _ => Ok(vec![builder.piece(index)]),
            };
            let pieces = pieces.map_err(|error| self.fail(error))?;
            for piece in &pieces {
                self.issue_dma(piece)?;
            }
            let builder = self.dma.as_ref().expect("a DMA builder");
            let called = builder.execute_call(index, self.instance, self.streams);
            self.unit
                .record(&self.instance.collectives, called)
                .map_err(|error| self.instance.dispatch_failed(error))?;
            self.started = true;
            self.blocks += 1;
        }
        if complete {
            let builder = self.dma.take().expect("a DMA builder");
            let plan = builder
                .finish(self.lowerer.body())
                .expect("a DMA plan with a block");
            let tail = plan.tail().to_vec();
            let pieces = match &mut self.engram {
                Some(engram) => engram.split_dma(&tail).map(<[DmaPiece]>::to_vec),
                None => Ok(vec![DmaPiece {
                    spans: tail,
                    join: None,
                }]),
            };
            let pieces = pieces.map_err(|error| self.fail(error))?;
            for piece in &pieces {
                self.issue_dma(piece)?;
            }
            self.dma_plan = Some(plan);
        }
        Ok(())
    }

    /// Issues a DMA compute piece after its join; a failure poisons, as
    /// any of the DMA body's.
    fn issue_dma(&mut self, piece: &DmaPiece) -> Result<(), BundleAbiError> {
        let instance = self.instance;
        let issued = (|| {
            if let Some(layer) = piece.join {
                self.join(layer)?;
            }
            self.issue_ranges(&piece.spans)
        })();
        issued.inspect_err(|_| instance.collectives.poison())
    }

    /// Issues each rank's body launches `ranges` through the unit.
    fn issue_ranges(&mut self, ranges: &[std::ops::Range<usize>]) -> Result<(), BundleAbiError> {
        let body = self.lowerer.body();
        let programs: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(rank, range)| &body.0[rank][range.clone()])
            .collect();
        let flags: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(rank, range)| &body.1[rank][range.clone()])
            .collect();
        Executor::new(self.instance, self.streams)
            .issue_in(&mut self.unit, &programs, &flags)
            .map_err(|error| self.instance.dispatch_failed(error))?;
        self.started |= ranges.iter().any(|range| !range.is_empty());
        Ok(())
    }

    /// Joins Engram layer `layer` on every stream, as the whole-plan launch
    /// does before the piece that reads it.
    fn join(&mut self, layer: usize) -> Result<(), BundleAbiError> {
        let Some((gathers, sequence)) = self.gathers else {
            return Err(self.fail("a streamed prefill joins no prepared gathers".into()));
        };
        if layer != self.joins || !gathers.owned_by(sequence, layer) {
            return Err(self.fail("prefill Engram gathers changed before their join".into()));
        }
        let joined = self
            .instance
            .join_prefill_engram(self.streams, layer)
            .map_err(DispatchError::from);
        self.unit
            .record(&self.instance.collectives, joined)
            .map_err(|error| self.instance.dispatch_failed(error))?;
        self.joins += 1;
        Ok(())
    }

    /// Without DMA: issues the body's collective rounds the lowered body
    /// holds, as the plan's eager launch or its prepared-gather pieces
    /// issue them, each piece after its join.
    fn issue_rounds(&mut self, complete: bool) -> Result<(), BundleAbiError> {
        let lens: Vec<usize> = self.lowerer.body().0.iter().map(Vec::len).collect();
        let usable = match self.frontiers.front() {
            Some(front) if self.started && !complete => front.clone(),
            _ => lens.clone(),
        };
        loop {
            let layers = self.engram.as_ref().map_or(0, EngramBuilder::layers);
            // The current piece ends at its layer's consumer, the last at the
            // body's end.
            let end = match &self.engram {
                Some(engram) if self.joins < layers => engram.consumer(self.joins),
                _ if complete => Some(lens.clone()),
                _ => None,
            };
            if let Some(end) = &end
                && *end == self.issued
            {
                if self.joins < layers {
                    self.join(self.joins)?;
                    continue;
                }
                return Ok(());
            }
            let target = match &end {
                Some(end) if end.iter().zip(&usable).all(|(e, u)| e <= u) => end.clone(),
                _ => {
                    let limit: Vec<_> = match &end {
                        Some(end) => end.iter().zip(&usable).map(|(&e, &u)| e.min(u)).collect(),
                        None => usable.clone(),
                    };
                    self.rounds(&limit)
                }
            };
            if target == self.issued {
                return Ok(());
            }
            let ranges: Vec<_> = self
                .issued
                .iter()
                .zip(&target)
                .map(|(&from, &to)| from..to)
                .collect();
            self.issue_ranges(&ranges)?;
            self.issued = target;
        }
    }

    /// Each rank's end of the whole collective rounds after its issued
    /// launches and before `limit`: as many rounds on every rank as the
    /// rank with the fewest holds.
    fn rounds(&self, limit: &[usize]) -> Vec<usize> {
        let flags = &self.lowerer.body().1;
        let collectives: Vec<Vec<usize>> = self
            .issued
            .iter()
            .zip(limit)
            .enumerate()
            .map(|(rank, (&from, &to))| (from..to).filter(|&at| flags[rank][at]).collect())
            .collect();
        let rounds = collectives.iter().map(Vec::len).min().unwrap_or(0);
        if rounds == 0 {
            return self.issued.clone();
        }
        collectives.iter().map(|at| at[rounds - 1] + 1).collect()
    }

    /// The head and its row copies after the issued body, as the plan's
    /// launch issues them, then the complete plan and its Engram plan.
    fn finish(self) -> Result<(Plan, Option<Engram>), BundleAbiError> {
        let Stream {
            instance,
            run,
            layout,
            mut unit,
            lowerer,
            gathers,
            engram,
            dma_plan,
            ..
        } = self;
        let plan = lowerer
            .assemble(instance, run, layout, dma_plan)
            .map_err(|error| fail(instance, &mut unit, error))?;
        #[cfg(test)]
        injected(Failure::Head).map_err(|error| fail(instance, &mut unit, error))?;
        let engram = match (engram, gathers) {
            (Some(engram), Some((gathers, _))) => {
                let made = (|| {
                    let engram = engram.finish(&plan.body.0, &plan.body.1)?;
                    if plan.dma.is_some() {
                        engram.check_dma_collectives(&plan.body.0, &plan.body.1)?;
                    }
                    plan.audit_head(&gathers.pending_spans(engram.layer_count()))?;
                    Ok::<_, String>(engram)
                })();
                Some(made.map_err(|error| fail(instance, &mut unit, error))?)
            }
            _ => None,
        };
        plan.compact_rows(instance, &mut unit)?;
        plan.execute_head(instance, &mut unit)?;
        Ok((plan, engram))
    }
}
