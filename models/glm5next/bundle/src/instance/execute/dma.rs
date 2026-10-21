//! Replace only admitted F32 MoE collective launches with ordered DMA.
use super::{PREFILL, Run, Stage};
use crate::buffers::Buffer;
use crate::instance::Instance;
use crate::program::{Sublayer, moe_sublayer};
use bundle_runtime::dispatch::{self, DispatchError};
use bundle_runtime::dma::{Call, subspan};
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch};
use runtime_bundle_abi::BundleStreamHandleV1;

pub(super) fn eligible_inputs(instance: &Instance, inputs: &PhaseInputs) -> bool {
    instance.dma.len() == 4
        && !instance.partial
        && !instance.sync_stages
        && !instance.taps.text()
        && inputs.request_ids.len() == 1
        && inputs.live_tokens() >= crate::prefill_dma::MIN_ROWS
        && instance.buffers.iter().all(|b| b.comm.phased)
}

pub(super) fn eligible(instance: &Instance, run: &Run) -> bool {
    run.arena == PREFILL
        && run.draft.is_none()
        && !run.inject
        && eligible_inputs(instance, &run.inputs)
}

struct Segment {
    programs: Vec<Vec<PreparedLaunch>>,
    stages: Vec<Vec<Stage>>,
    collectives: Vec<Vec<bool>>,
}

fn segment(
    programs: &[Vec<PreparedLaunch>],
    stages: &[Vec<Stage>],
    collectives: &[Vec<bool>],
    ranges: &[std::ops::Range<usize>],
) -> Segment {
    let mut copied = Segment {
        programs: Vec::new(),
        stages: Vec::new(),
        collectives: Vec::new(),
    };
    for (rank, range) in ranges.iter().enumerate() {
        copied.programs.push(
            programs[rank][range.clone()]
                .iter()
                .map(|p| PreparedLaunch {
                    symbol: p.symbol.clone(),
                    rows: p.rows,
                    plan: p.plan,
                    arguments: p.arguments.clone(),
                    bindings: p.bindings.clone(),
                })
                .collect(),
        );
        copied.stages.push(stages[rank][range.clone()].to_vec());
        copied
            .collectives
            .push(collectives[rank][range.clone()].to_vec());
    }
    copied
}

pub(super) struct Plan {
    before: Vec<Segment>,
    tail: Segment,
    call: Call,
    sums: Vec<PreparedLaunch>,
}

impl Plan {
    pub(super) fn prepare(
        instance: &Instance,
        rows: u32,
        programs: &[Vec<PreparedLaunch>],
        stages: &[Vec<Stage>],
        collectives: &[Vec<bool>],
    ) -> Result<Self, String> {
        if programs.len() != 4
            || stages.len() != 4
            || collectives.len() != 4
            || (0..4).any(|r| {
                programs[r].len() != stages[r].len() || programs[r].len() != collectives[r].len()
            })
        {
            return Err("DMA rank programs have inconsistent lengths".into());
        }
        let positions: Vec<Vec<usize>> = programs
            .iter()
            .map(|p| {
                p.iter()
                    .enumerate()
                    .filter_map(|(at, launch)| {
                        instance
                            .target
                            .prefill_dma_replaces(&launch.symbol)
                            .then_some(at)
                    })
                    .collect()
            })
            .collect();
        let expected = (0..instance.config.shape.geometry.layers())
            .filter(|&layer| moe_sublayer(instance.config.shape, layer, Sublayer::Mlp))
            .map(|layer| Stage::Reduce(layer, Sublayer::Mlp))
            .collect::<Vec<_>>();
        for rank in 0..4 {
            if positions[rank].len() != expected.len()
                || positions[rank]
                    .iter()
                    .zip(&expected)
                    .any(|(&at, stage)| stages[rank][at] != *stage || !collectives[rank][at])
            {
                return Err("DMA requires every target MoE F32 collective in stage order".into());
            }
        }
        let elements = u64::from(rows) * u64::from(instance.config.shape.profile.hidden);
        let inputs = instance
            .buffers
            .iter()
            .map(|b| subspan(b.get(Buffer::Partial), 0, elements * 4))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = instance
            .buffers
            .iter()
            .map(|b| subspan(b.get(Buffer::Reduced), 0, elements * 2))
            .collect::<Result<Vec<_>, _>>()?;
        let call = Call::prepare(
            &instance.dma,
            &inputs,
            &outputs,
            rows,
            instance.config.shape.profile.hidden,
        )?;
        let sums = call
            .owners()
            .iter()
            .map(|&owner| instance.target.prefill_dma_sum(owner))
            .collect::<Result<Vec<_>, _>>()?;
        let mut cursors = [0; 4];
        let mut before = Vec::new();
        for block in 0..expected.len() {
            let ranges = (0..4)
                .map(|rank| cursors[rank]..positions[rank][block])
                .collect::<Vec<_>>();
            before.push(segment(programs, stages, collectives, &ranges));
            for rank in 0..4 {
                // Replace precisely the collective. Trailing stage work,
                // including any widening or epilogue, remains in the next segment.
                cursors[rank] = positions[rank][block] + 1;
            }
        }
        let tail = segment(
            programs,
            stages,
            collectives,
            &(0..4)
                .map(|r| cursors[r]..programs[r].len())
                .collect::<Vec<_>>(),
        );
        Ok(Self {
            before,
            tail,
            call,
            sums,
        })
    }

    pub(super) fn execute(
        &self,
        instance: &Instance,
        streams: &[BundleStreamHandleV1],
    ) -> Result<(), DispatchError> {
        let sums = self
            .sums
            .iter()
            .enumerate()
            .map(|(rank, p)| {
                dispatch::resolve(
                    &instance.symbols[rank],
                    &p.symbol,
                    p.plan,
                    &p.arguments,
                    &p.bindings,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let issue = |segment: &Segment| {
            instance.launch_eager(
                streams,
                &segment.programs,
                &segment.stages,
                &segment.collectives,
            )
        };
        let _unwind = PoisonOnUnwind(&instance.collectives);
        let result = (|| {
            for segment in &self.before {
                issue(segment)?;
                #[cfg(test)]
                for (sum, &stream) in self.sums.iter().zip(streams) {
                    crate::submit::record_dma_sum(stream, sum);
                }
                self.call.execute(
                    &instance.services,
                    &instance.dma,
                    streams,
                    &sums,
                    &instance.collectives,
                )?;
            }
            issue(&self.tail)
        })();
        if result.is_err() {
            instance.collectives.poison();
        }
        result
    }
}
