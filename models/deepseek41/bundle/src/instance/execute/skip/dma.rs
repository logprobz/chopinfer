//! Replace only the existing cone's rounded reductions with ordered DMA.
//! Every compute launch and persistent mutation retains its original rows.
use super::{Instance, Programs, Run, Stage};
use crate::instance::execute::dma::Call;
use bundle_runtime::pipeline::PoisonOnUnwind;
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

fn reductions(instance: &Instance, body: &Programs, rank: usize) -> Result<Vec<Reduction>, String> {
    let mut found = Vec::new();
    let (programs, flags, stages) = body;
    if programs[rank].len() != flags[rank].len() || programs[rank].len() != stages[rank].len() {
        return Err("sparse DMA program metadata differs".into());
    }
    let partial = instance.buffers[rank].scratch.partial;
    let row_bytes = u64::from(instance.config.shape.dims().hidden) * 4;
    let mut at = 0;
    while at < stages[rank].len() {
        let stage = stages[rank][at];
        let end = at
            + stages[rank][at..]
                .iter()
                .take_while(|&&s| s == stage)
                .count();
        if matches!(stage, Stage::EmbeddingReduce | Stage::Reduce(..)) {
            let collective = (at..end)
                .find(|&i| flags[rank][i])
                .ok_or("sparse reduction has no collective")?;
            let launch = &programs[rank][collective];
            // Only the admitted phased contracts; composed parts keep theirs.
            if crate::Kernels::prefill_dma_payload(&launch.symbol, false).is_none() {
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
    Ok(found)
}

struct Block {
    compute: Vec<Range<usize>>,
    reduce: Call,
}

pub(super) struct Plan {
    blocks: Vec<Block>,
    tail: Vec<Range<usize>>,
}

impl Plan {
    pub(super) fn prepare(
        instance: &Instance,
        run: &Run,
        body: &Programs,
    ) -> Result<Option<Self>, String> {
        if !crate::instance::execute::dma::eligible(instance, run)
            || !instance.overlap.is_empty()
            || instance.config.kv.storage != KvStorage::Native
        {
            return Ok(None);
        }
        #[cfg(test)]
        if !ENABLED.get() {
            return Ok(None);
        }
        if body.0.len() != 4 || body.1.len() != 4 || body.2.len() != 4 {
            return Err("sparse DMA requires all four rank programs".into());
        }
        let ranks = (0..4)
            .map(|rank| reductions(instance, body, rank))
            .collect::<Result<Vec<_>, _>>()?;
        if ranks.iter().any(|rank| rank.len() != ranks[0].len()) {
            return Err("sparse DMA rank reduction counts differ".into());
        }
        let mut cursors = [0; 4];
        let mut blocks = Vec::new();
        for (index, reference) in ranks[0].iter().enumerate() {
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
                false,
            )?
            else {
                continue;
            };
            let compute = cursors
                .iter_mut()
                .enumerate()
                .map(|(rank, cursor)| {
                    let r = &ranks[rank][index];
                    let span = *cursor..r.collective;
                    *cursor = r.end;
                    span
                })
                .collect();
            blocks.push(Block { compute, reduce });
        }
        let tail = cursors
            .into_iter()
            .enumerate()
            .map(|(rank, at)| at..body.0[rank].len())
            .collect();
        if blocks.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self { blocks, tail }))
    }

    pub(super) fn execute(
        &self,
        instance: &Instance,
        body: &Programs,
    ) -> Result<(), BundleAbiError> {
        let streams = instance.ranks.iter().map(|rank| rank.2).collect::<Vec<_>>();
        let issue = |spans: &[Range<usize>]| {
            instance.launch_eager(
                &streams,
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
        let _unwind = PoisonOnUnwind(&instance.collectives);
        let result = (|| {
            for block in &self.blocks {
                issue(&block.compute)?;
                block.reduce.execute(instance, &streams)?;
            }
            issue(&self.tail)
        })();
        if result.is_err() {
            instance.collectives.poison();
        }
        result.map_err(|error| instance.dispatch_failed(error))
    }
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
