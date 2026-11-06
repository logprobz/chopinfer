//! Replace only the existing cone's rounded reductions with ordered DMA.
//! Every compute launch and persistent mutation retains its original rows.
use super::{Instance, Programs, Run, Stage};
use crate::instance::execute::Executor;
use crate::instance::execute::dma::{Call, Consumers, Producers};
use crate::instance::prefill_engram::DmaPiece;
use bundle_runtime::dispatch::DispatchError;
use kernel_core::contract::Argument;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::BundleAbiError;
use std::ops::Range;

#[derive(Debug, Eq, PartialEq)]
struct Reduction {
    stage: Stage,
    first: u32,
    rows: u32,
    collective: usize,
    end: usize,
}

/// Rank `rank`'s reductions in the stage runs of its body from `from`, the
/// first launch of a run, and where the scan stopped: a run that reaches
/// the end of the body is scanned only once the body is `complete`, as the
/// next group of a lowering in groups may extend it.
fn reductions(
    instance: &Instance,
    body: &Programs,
    rank: usize,
    from: usize,
    complete: bool,
) -> Result<(Vec<Reduction>, usize), String> {
    let mut found = Vec::new();
    let (programs, flags, stages) = body;
    if programs[rank].len() != flags[rank].len() || programs[rank].len() != stages[rank].len() {
        return Err("sparse DMA program metadata differs".into());
    }
    let partial = instance.buffers[rank].scratch.partial;
    let row_bytes = u64::from(instance.config.shape.dims().hidden) * 4;
    let mut at = from;
    while at < stages[rank].len() {
        let stage = stages[rank][at];
        let end = at
            + stages[rank][at..]
                .iter()
                .take_while(|&&s| s == stage)
                .count();
        if end == stages[rank].len() && !complete {
            break;
        }
        if matches!(stage, Stage::EmbeddingReduce | Stage::Reduce(..)) {
            let collective = (at..end)
                .find(|&i| flags[rank][i])
                .ok_or("sparse reduction has no collective")?;
            let launch = &programs[rank][collective];
            // Only the admitted phased contracts; composed parts keep theirs.
            if crate::Kernels::prefill_dma_payload(&launch.symbol).is_none() {
                at = end;
                continue;
            }
            // The qualified phased contracts take their F32 partial and BF16
            // result as their first two arguments. Check the bound row views;
            // never infer it from a layer number or a compact output index.
            let Some(Argument::DeviceBuffer(input)) = launch.arguments.0.first() else {
                return Err("sparse reduction input is not a direct buffer".into());
            };
            let view = launch
                .bindings
                .get(*input)
                .map_err(|e| format!("sparse DMA input: {e:?}"))?;
            let offset = view
                .offset
                .checked_sub(partial.offset)
                .ok_or("sparse DMA input precedes partials")?;
            let bytes = u64::from(launch.rows) * row_bytes;
            if view.buffer != partial.buffer
                || !offset.is_multiple_of(row_bytes)
                || view.length != bytes
                || offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > partial.length)
            {
                return Err("sparse DMA input is not the original partial row interval".into());
            }
            let Some(Argument::DeviceBuffer(output)) = launch.arguments.0.get(1) else {
                return Err("sparse reduction output is not a direct buffer".into());
            };
            let output = launch
                .bindings
                .get(*output)
                .map_err(|e| format!("sparse DMA output: {e:?}"))?;
            let plane = if stage == Stage::EmbeddingReduce {
                instance.buffers[rank].scratch.embedded
            } else {
                instance.buffers[rank].scratch.reduced
            };
            if output.buffer != plane.buffer
                || plane.offset.checked_add(offset / 2) != Some(output.offset)
                || output.length != bytes / 2
                || (offset / 2)
                    .checked_add(output.length)
                    .is_none_or(|end| end > plane.length)
            {
                return Err("sparse DMA output is not the original rounded row interval".into());
            }
            found.push(Reduction {
                stage,
                first: u32::try_from(offset / row_bytes).map_err(|_| "sparse DMA row overflow")?,
                rows: launch.rows,
                collective,
                end,
            });
        }
        at = end;
    }
    Ok((found, at))
}

struct Block {
    compute: Vec<Range<usize>>,
    reduce: Call,
    /// The row-local launches after the call, run per owner interval; their
    /// program launches are then skipped.
    consume: Option<Consumers>,
    /// The launch before the call, run per owner interval inside it; its
    /// program launch is then skipped.
    produce: Option<Producers>,
}

pub(super) struct Plan {
    blocks: Vec<Block>,
    tail: Vec<Range<usize>>,
    /// Each rank's body launches the plan read to prepare its calls: each
    /// call's producer through its consumers. The launches it replaces lie
    /// among them.
    reads: Vec<Vec<Range<usize>>>,
}

impl Block {
    fn same(&self, other: &Self) -> Result<bool, String> {
        let consume = match (&self.consume, &other.consume) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b)?,
            _ => false,
        };
        let produce = match (&self.produce, &other.produce) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b)?,
            _ => false,
        };
        Ok(self.compute == other.compute
            && consume
            && produce
            && self.reduce.same(&other.reduce)?)
    }
}

impl Plan {
    /// Whether `other` issues the same compute spans, calls and owner-local
    /// launches.
    pub(super) fn same(&self, other: &Self) -> Result<bool, String> {
        if self.tail != other.tail
            || self.reads != other.reads
            || self.blocks.len() != other.blocks.len()
        {
            return Ok(false);
        }
        for (a, b) in self.blocks.iter().zip(&other.blocks) {
            if !a.same(b)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The blocks as lines (`skip::Plan::describe`): each block's compute
    /// spans, call, consumers and producers, then the tail and the reads.
    #[cfg(test)]
    pub(super) fn describe(&self, lines: &mut Vec<String>) {
        for (index, block) in self.blocks.iter().enumerate() {
            lines.push(format!("block {index} {:?}", block.compute));
            block.reduce.describe(lines);
            if let Some(consume) = &block.consume {
                consume.describe(lines);
            }
            if let Some(produce) = &block.produce {
                produce.describe(lines);
            }
        }
        lines.push(format!("tail {:?}", self.tail));
        lines.push(format!("reads {:?}", self.reads));
    }

    /// Each rank's launches after the last block.
    pub(super) fn tail(&self) -> &[Range<usize>] {
        &self.tail
    }

    /// Rank `rank`'s body launches this plan read (`reads`).
    pub(super) fn reads(&self, rank: usize) -> &[Range<usize>] {
        self.reads.get(rank).map_or(&[], Vec::as_slice)
    }

    pub(super) fn prepare_engram(
        &self,
        plan: &mut crate::instance::prefill_engram::Plan,
        gathers: &crate::instance::prefill_engram::Gathers,
    ) -> Result<(), String> {
        let mut joined = 0;
        let mut pieces = Vec::new();
        for block in &self.blocks {
            pieces.push(plan.split_dma(&block.compute, &mut joined)?);
            block.reduce.audit_pending(
                &gathers.pending_spans(joined),
                block.consume.as_ref(),
                block.produce.as_ref(),
            )?;
        }
        pieces.push(plan.split_dma(&self.tail, &mut joined)?);
        if joined != plan.layer_count() {
            return Err("DMA omitted an Engram join boundary".into());
        }
        plan.dma = pieces;
        Ok(())
    }

    pub(super) fn prepare(
        instance: &Instance,
        run: &Run,
        body: &Programs,
    ) -> Result<Option<Self>, String> {
        let Some(mut builder) = Builder::new(instance, run) else {
            return Ok(None);
        };
        builder.advance(instance, body, true)?;
        Ok(builder.finish(body))
    }

    pub(super) fn execute(
        &self,
        instance: &Instance,
        body: &Programs,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
    ) -> Result<(), BundleAbiError> {
        let streams = instance.ranks.iter().map(|rank| rank.2).collect::<Vec<_>>();
        let executor = Executor::new(instance, &streams);
        let issue = |spans: &[Range<usize>]| {
            executor.issue(
                &spans
                    .iter()
                    .enumerate()
                    .map(|(rank, span)| &body.0[rank][span.clone()])
                    .collect::<Vec<_>>(),
                &spans
                    .iter()
                    .enumerate()
                    .map(|(rank, span)| &body.1[rank][span.clone()])
                    .collect::<Vec<_>>(),
            )
        };
        let compute = |index: usize, spans: &[Range<usize>]| {
            if let Some(plan) = gathers {
                let pieces = plan
                    .dma
                    .get(index)
                    .ok_or(bundle_runtime::dispatch::DispatchError::InvalidPlan)?;
                for crate::instance::prefill_engram::DmaPiece { spans, join } in pieces {
                    if let Some(layer) = join {
                        instance.join_prefill_engram(&streams, *layer)?;
                    }
                    issue(spans)?;
                }
                Ok(())
            } else {
                issue(spans)
            }
        };
        let result = executor.run(|| {
            for (index, block) in self.blocks.iter().enumerate() {
                compute(index, &block.compute)?;
                block.reduce.execute(
                    instance,
                    &streams,
                    block.consume.as_ref(),
                    block.produce.as_ref(),
                )?;
            }
            compute(self.blocks.len(), &self.tail)
        });
        if result.is_err() {
            instance.collectives.poison();
        }
        result.map_err(|error| instance.dispatch_failed(error))
    }
}

/// The DMA plan of a body as its launches are lowered: each block is
/// prepared once every rank lowered its call and the launches after the
/// call its consumers read, in body order, as `Plan::prepare` prepares
/// them from the whole body.
pub(super) struct Builder {
    /// Each rank's reductions found so far, and where its scan stopped.
    found: Vec<Vec<Reduction>>,
    scanned: Vec<usize>,
    /// The next reduction to prepare a block for.
    next: usize,
    cursors: [usize; 4],
    blocks: Vec<Block>,
    reads: Vec<Vec<Range<usize>>>,
}

impl Builder {
    /// The builder of `run`'s DMA plan; none when its body takes no DMA.
    pub(super) fn new(instance: &Instance, run: &Run) -> Option<Self> {
        applies(instance, run).then(|| Self {
            found: (0..4).map(|_| Vec::new()).collect(),
            scanned: vec![0; 4],
            next: 0,
            cursors: [0; 4],
            blocks: Vec::new(),
            reads: vec![Vec::new(); 4],
        })
    }

    /// The blocks prepared so far.
    pub(super) fn blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Splits block `index`'s compute spans at the Engram joins they cross
    /// and audits its call against the spans pending there, as
    /// `Plan::prepare_engram` does for each block: its pieces.
    pub(super) fn split_engram(
        &self,
        index: usize,
        engram: &mut crate::instance::prefill_engram::Builder,
        gathers: &crate::instance::prefill_engram::Gathers,
    ) -> Result<Vec<DmaPiece>, String> {
        let block = &self.blocks[index];
        let pieces = engram.split_dma(&block.compute)?.to_vec();
        block.reduce.audit_pending(
            &gathers.pending_spans(engram.joined()),
            block.consume.as_ref(),
            block.produce.as_ref(),
        )?;
        Ok(pieces)
    }

    /// Block `index`'s compute spans as its one piece, for a body without
    /// prepared gathers.
    pub(super) fn piece(&self, index: usize) -> DmaPiece {
        DmaPiece {
            spans: self.blocks[index].compute.clone(),
            join: None,
        }
    }

    /// Issues block `index`'s call after its compute pieces, as
    /// `Plan::execute` does. A failure poisons, as any of the DMA body's.
    pub(super) fn execute_call(
        &self,
        index: usize,
        instance: &Instance,
        streams: &[runtime_bundle_abi::BundleStreamHandleV1],
    ) -> Result<(), DispatchError> {
        let block = &self.blocks[index];
        block
            .reduce
            .execute(
                instance,
                streams,
                block.consume.as_ref(),
                block.produce.as_ref(),
            )
            .inspect_err(|_| instance.collectives.poison())
    }

    /// Prepares the blocks `body`, lowered so far, holds: each whose call
    /// and consumers' launches every rank lowered, or every block once
    /// `complete`.
    pub(super) fn advance(
        &mut self,
        instance: &Instance,
        body: &Programs,
        complete: bool,
    ) -> Result<(), String> {
        if body.0.len() != 4 || body.1.len() != 4 || body.2.len() != 4 {
            return Err("sparse DMA requires all four rank programs".into());
        }
        for rank in 0..4 {
            let (found, at) = reductions(instance, body, rank, self.scanned[rank], complete)?;
            self.found[rank].extend(found);
            self.scanned[rank] = at;
        }
        let ranks = &self.found;
        if complete && ranks.iter().any(|rank| rank.len() != ranks[0].len()) {
            return Err("sparse DMA rank reduction counts differ".into());
        }
        let available = ranks.iter().map(Vec::len).min().unwrap_or(0);
        while self.next < available {
            let index = self.next;
            // The consumers read the launches after each rank's call.
            if !complete
                && (0..4)
                    .any(|rank| ranks[rank][index].end + Consumers::LAUNCHES > body.0[rank].len())
            {
                break;
            }
            self.next += 1;
            let reference = &ranks[0][index];
            if ranks.iter().any(|rank| {
                let r = &rank[index];
                (r.stage, r.first, r.rows) != (reference.stage, reference.first, reference.rows)
            }) {
                return Err("sparse DMA ranks disagree on original rows".into());
            }
            let collectives = (0..4)
                .map(|rank| body.0[rank][ranks[rank][index].collective].symbol.as_str())
                .collect::<Vec<_>>();
            let Some(reduce) = Call::prepare(
                instance,
                reference.first,
                reference.rows,
                reference.stage == Stage::EmbeddingReduce,
                &collectives,
            )?
            else {
                continue;
            };
            for (rank, read) in self.reads.iter_mut().enumerate() {
                let r = &ranks[rank][index];
                read.push(
                    r.collective.saturating_sub(1)
                        ..(r.end + Consumers::LAUNCHES).min(body.0[rank].len()),
                );
            }
            let following = (0..4)
                .map(|rank| &body.0[rank][ranks[rank][index].end..])
                .collect::<Vec<_>>();
            let flags = (0..4)
                .map(|rank| &body.1[rank][ranks[rank][index].end..])
                .collect::<Vec<_>>();
            let consume = Consumers::prepare(instance, &reduce, &following, &flags)?;
            let taken = if consume.is_some() {
                Consumers::LAUNCHES
            } else {
                0
            };
            // The producer is the last launch of each rank's compute span.
            let produce = if (0..4).all(|rank| ranks[rank][index].collective > self.cursors[rank]) {
                let launches = (0..4)
                    .map(|rank| &body.0[rank][ranks[rank][index].collective - 1])
                    .collect::<Vec<_>>();
                let flags = (0..4)
                    .map(|rank| body.1[rank][ranks[rank][index].collective - 1])
                    .collect::<Vec<_>>();
                Producers::prepare(instance, &reduce, &launches, &flags)?
            } else {
                None
            };
            let given = usize::from(produce.is_some());
            let compute = self
                .cursors
                .iter_mut()
                .enumerate()
                .map(|(rank, cursor)| {
                    let r = &ranks[rank][index];
                    let span = *cursor..r.collective - given;
                    *cursor = r.end + taken;
                    span
                })
                .collect();
            self.blocks.push(Block {
                compute,
                reduce,
                consume,
                produce,
            });
        }
        Ok(())
    }

    /// The plan of the complete `body`, once `advance` prepared every
    /// block: none without a block.
    pub(super) fn finish(self, body: &Programs) -> Option<Plan> {
        let tail = self
            .cursors
            .into_iter()
            .enumerate()
            .map(|(rank, at)| at..body.0[rank].len())
            .collect();
        (!self.blocks.is_empty()).then_some(Plan {
            blocks: self.blocks,
            tail,
            reads: self.reads,
        })
    }
}

/// Whether `run`'s body may take upper-skip DMA; it does when its
/// reductions admit any call.
pub(super) fn applies(instance: &Instance, run: &Run) -> bool {
    run.phase.prefill() && run.media.is_none() && admits(instance, run.inputs.request_ids.len())
}

/// Whether the instance may send an upper-skip text prefill of `requests`
/// requests through DMA, before its reductions' own admission.
pub(super) fn admits(instance: &Instance, requests: usize) -> bool {
    #[cfg(test)]
    if !ENABLED.get() {
        return false;
    }
    crate::instance::execute::dma::admits(instance, requests)
        && instance.config.kv.storage == KvStorage::Native
}

#[cfg(test)]
thread_local! { static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) }; }

#[cfg(test)]
pub(crate) fn with_enabled<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.set(self.0);
        }
    }
    let _reset = Reset(ENABLED.replace(enabled));
    run()
}
