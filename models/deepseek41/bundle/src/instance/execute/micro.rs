//! Two prefill parts on one compute stream and one ordered reduction stream.
//! Retained row planes use global origins; transient scratch stays local.
use super::{Run, Stage};
use crate::instance::Instance;
use crate::lower::Lowering;
use crate::ops::Lowered;
use crate::program::{Phase, Sublayer};
use bundle_runtime::dispatch::DispatchError;
use bundle_runtime::lanes::{self, Block, Lane, Op};
use bundle_runtime::pipeline::{PoisonOnUnwind, PreparedLaunch, check_collectives};
use bundle_runtime::resources::upload_bytes;
use bundle_runtime::upload::DeviceInputs;
use model_deepseek41_contract::serving::{PREFILL_GRANULE, PREFILL_TOKENS};
use runtime_bundle_abi::DeviceBufferSpanV1;

#[derive(Default)]
struct Calls {
    launches: Vec<PreparedLaunch>,
    flags: Vec<bool>,
}

impl Calls {
    fn append(&mut self, calls: Vec<Lowered>) {
        for call in calls {
            self.flags.push(call.collective);
            self.launches.push(call.launch);
        }
    }

    fn copied(run: &Run, rank: usize, pick: impl Fn(usize, Stage) -> bool) -> Self {
        let mut calls = Self::default();
        for (at, &stage) in run.stages[rank].iter().enumerate() {
            if pick(at, stage) {
                let launch = &run.programs[rank][at];
                calls.launches.push(PreparedLaunch {
                    symbol: launch.symbol.clone(),
                    rows: launch.rows,
                    plan: launch.plan,
                    arguments: launch.arguments.clone(),
                    bindings: launch.bindings.clone(),
                });
                calls.flags.push(run.collectives[rank][at]);
            }
        }
        calls
    }
}

struct Part {
    compute: Calls,
    reduce: Calls,
}

#[derive(Clone, Copy)]
enum Work {
    Compute(usize),
    Reduce(usize),
}

pub(super) struct Plan {
    rows: [u32; 2],
    whole: Vec<Calls>,
    blocks: Vec<Vec<[Part; 2]>>,
    tail: Vec<Calls>,
    dma: Vec<[Option<super::dma::Call>; 2]>,
}

fn split(rows: u32) -> Option<[u32; 2]> {
    if !(1024..=PREFILL_TOKENS).contains(&rows) {
        return None;
    }
    let a = rows.div_ceil(2).div_ceil(PREFILL_GRANULE) * PREFILL_GRANULE;
    Some([a, rows - a])
}

fn inputs(
    full: DeviceInputs,
    first: u32,
    rows: u32,
    live: DeviceBufferSpanV1,
) -> Result<DeviceInputs, String> {
    let slice = |span: DeviceBufferSpanV1| -> Result<DeviceBufferSpanV1, String> {
        let offset = u64::from(first) * 4;
        let length = u64::from(rows) * 4;
        if offset + length > span.length {
            return Err("prefill part exceeds uploaded input rows".into());
        }
        Ok(DeviceBufferSpanV1::new(
            span.buffer,
            span.offset + offset,
            length,
        ))
    };
    Ok(DeviceInputs {
        token_ids: slice(full.token_ids)?,
        positions: slice(full.positions)?,
        token_batch: slice(full.token_batch)?,
        live_tokens: live,
        ..full
    })
}

impl Plan {
    pub(super) fn prepare(instance: &Instance, run: &Run) -> Result<Option<Self>, String> {
        if !instance.layer_overlap
            || !run.phase.prefill()
            || instance.config.shape.ranks() != 4
            || instance.dspark.is_some()
            || run.media.is_some()
            || run.inputs.request_ids.len() != 1
            || instance.overlap.len() != instance.ranks.len()
            || instance.overlap.is_empty()
        {
            return Ok(None);
        }
        #[cfg(feature = "diagnostic-taps")]
        if instance.taps.text() {
            return Ok(None);
        }
        let Some(rows) = split(run.phase.rows()) else {
            return Ok(None);
        };
        let Some(&start) = run.inputs.positions.first() else {
            return Ok(None);
        };
        if start < 0
            || !(start as u32).is_multiple_of(PREFILL_GRANULE)
            || run
                .inputs
                .positions
                .iter()
                .enumerate()
                .any(|(i, &p)| p != start + i as i32)
            || run.inputs.token_batch.iter().any(|&item| item != 0)
        {
            return Ok(None);
        }
        check_collectives(&run.programs, &run.collectives)?;
        let order = crate::program::stages(instance.config.shape.geometry, run.phase);
        if order.get(..5)
            != Some(&[
                Stage::Embedding,
                Stage::EmbeddingReduce,
                Stage::ExpandStreams,
                Stage::RopeRows,
                Stage::WindowKeys,
            ])
        {
            return Err("prefill prologue no longer matches the part schedule".into());
        }
        let mut specs = vec![(vec![Stage::EmbeddingReduce], None)];
        let mut pending = vec![Stage::ExpandStreams];
        for stage in order.into_iter().skip(5) {
            if stage == Stage::HeadCollapse {
                break;
            }
            if matches!(stage, Stage::Reduce(..)) {
                specs.push((std::mem::take(&mut pending), Some(stage)));
            } else {
                pending.push(stage);
            }
        }
        let last = Stage::HcExpand(instance.config.shape.geometry.layers() - 1, Sublayer::Mlp);
        if pending != [last]
            || specs.len() != 1 + 2 * instance.config.shape.geometry.layers() as usize
        {
            return Err("prefill part boundaries do not cover every sublayer".into());
        }
        let mut whole = Vec::new();
        let mut tail = Vec::new();
        for rank in 0..instance.ranks.len() {
            whole.push(Calls::copied(run, rank, |_, s| {
                matches!(s, Stage::Embedding | Stage::RopeRows | Stage::WindowKeys)
            }));
            let first = run.stages[rank]
                .iter()
                .position(|&s| s == last)
                .ok_or("missing prefill epilogue")?;
            tail.push(Calls::copied(run, rank, |at, _| at >= first));
        }
        let mut blocks: Vec<Vec<[Part; 2]>> = Vec::new();
        for (stages, reduce) in specs {
            let mut block: Vec<[Part; 2]> = Vec::new();
            for (rank, buffers) in instance.buffers.iter().enumerate() {
                let full = DeviceInputs::within(instance.arenas[run.arena][rank], &run.layout);
                let mut parts = Vec::new();
                for (part, first) in [0, rows[0]].into_iter().enumerate() {
                    let counts = instance.overlap[rank].counts;
                    let live =
                        DeviceBufferSpanV1::new(counts.buffer, counts.offset + part as u64 * 4, 4);
                    let inputs = inputs(full, first, rows[part], live)?;
                    let lowering = Lowering {
                        prefill_row_origin: Some(first),
                        shape: &instance.config.shape,
                        ops: &instance.ops,
                        buffers,
                        peers: &instance.buffers,
                        inputs: &inputs,
                        rank: rank as u32,
                        kv: instance.kv_extent(),
                        head_rows: run.head_rows,
                    };
                    let phase = Phase::Prefill { rows: rows[part] };
                    let lower = |stage| {
                        lowering
                            .stage_with_prefill_context(
                                phase,
                                stage,
                                &run.inputs.positions
                                    [first as usize..(first + rows[part]) as usize],
                                run.cache,
                            )?
                            .ok_or_else(|| format!("prefill part has no lowering for {stage:?}"))
                    };
                    let mut compute = Calls::default();
                    for &stage in &stages {
                        compute.append(lower(stage)?);
                    }
                    let mut reduction = Calls::default();
                    if let Some(stage) = reduce {
                        reduction.append(lower(stage)?);
                    } else {
                        let at = compute
                            .flags
                            .iter()
                            .position(|&v| v)
                            .ok_or("embedding part has no collective")?;
                        reduction.launches = compute.launches.split_off(at);
                        reduction.flags = compute.flags.split_off(at);
                    }
                    if compute.flags.iter().any(|&v| v) || !reduction.flags.iter().any(|&v| v) {
                        return Err("prefill part crosses an unexpected collective".into());
                    }
                    parts.push(Part {
                        compute,
                        reduce: reduction,
                    });
                }
                crate::shared_overlap::disjoint_branches(
                    &parts[0].reduce.launches,
                    &parts[1].compute.launches,
                )?;
                if let Some(previous) = blocks.last() {
                    crate::shared_overlap::disjoint_branches(
                        &previous[rank][1].reduce.launches,
                        &parts[0].compute.launches,
                    )?;
                }
                block.push(parts.try_into().ok().expect("two prefill parts"));
            }
            for part in 0..2 {
                check_collectives(
                    &block
                        .iter()
                        .map(|p| p[part].reduce.launches.as_slice())
                        .collect::<Vec<_>>(),
                    &block
                        .iter()
                        .map(|p| p[part].reduce.flags.as_slice())
                        .collect::<Vec<_>>(),
                )?;
            }
            blocks.push(block);
        }
        let mut dma = Vec::new();
        for block in 0..blocks.len() {
            let mut calls = Vec::new();
            for part in 0..2 {
                let first = if part == 0 { 0 } else { rows[0] };
                calls.push(if super::dma::eligible(instance, run) {
                    super::dma::Call::prepare(instance, first, rows[part], block == 0)?
                } else {
                    None
                });
            }
            let calls: [Option<super::dma::Call>; 2] =
                calls.try_into().ok().expect("two DMA parts");
            if let Some(call) = &calls[0] {
                for rank in &blocks[block] {
                    call.audit(&rank[1].compute.launches)?;
                }
            }
            if let Some(call) = &calls[1]
                && let Some(next) = blocks.get(block + 1)
            {
                for rank in next {
                    call.audit(&rank[0].compute.launches)?;
                }
            }
            dma.push(calls);
        }
        Ok(Some(Self {
            rows,
            whole,
            blocks,
            tail,
            dma,
        }))
    }

    fn schedule(&self) -> Vec<Op<Work>> {
        lanes::Program {
            whole: vec![],
            prologue: vec![],
            epilogue: vec![],
            blocks: (0..self.blocks.len())
                .flat_map(|i| {
                    [
                        Block {
                            lane: Lane::Compute,
                            mark: 0,
                            stages: vec![Work::Compute(i)],
                        },
                        Block {
                            lane: Lane::Comm,
                            mark: 1,
                            stages: vec![Work::Reduce(i)],
                        },
                    ]
                })
                .collect(),
        }
        .schedule(2)
    }

    pub(super) fn execute(&self, instance: &Instance) -> Result<(), DispatchError> {
        let main = instance.ranks.iter().map(|r| r.2).collect::<Vec<_>>();
        let comm = instance
            .overlap
            .iter()
            .map(|r| r.lane.stream)
            .collect::<Vec<_>>();
        let issue = |calls: Vec<&Calls>, lane| {
            instance.launch_eager(
                if lane == Lane::Compute { &main } else { &comm },
                &calls
                    .iter()
                    .map(|c| c.launches.as_slice())
                    .collect::<Vec<_>>(),
                &calls.iter().map(|c| c.flags.as_slice()).collect::<Vec<_>>(),
            )
        };
        let _unwind = PoisonOnUnwind(&instance.collectives);
        let result = (|| {
            let bytes = self
                .rows
                .iter()
                .flat_map(|r| r.to_le_bytes())
                .collect::<Vec<_>>();
            for (rank, state) in instance.overlap.iter().enumerate() {
                upload_bytes(&instance.services, main[rank], state.counts, &bytes)?;
            }
            issue(self.whole.iter().collect(), Lane::Compute)?;
            for op in self.schedule() {
                match op {
                    Op::Launch { stage, part, lane } => {
                        let part = part.expect("prefill blocks have parts");
                        let (Work::Compute(block) | Work::Reduce(block)) = stage;
                        if matches!(stage, Work::Reduce(_))
                            && let Some(call) = &self.dma[block][part]
                        {
                            call.execute(instance, &comm)?;
                            continue;
                        }
                        issue(
                            self.blocks[block]
                                .iter()
                                .map(|p| match stage {
                                    Work::Compute(_) => &p[part].compute,
                                    Work::Reduce(_) => &p[part].reduce,
                                })
                                .collect(),
                            lane,
                        )?;
                    }
                    Op::Record(lane, mark) | Op::Wait(lane, mark) => {
                        for (rank, state) in instance.overlap.iter().enumerate() {
                            let stream = if lane == Lane::Compute {
                                main[rank]
                            } else {
                                comm[rank]
                            };
                            let event = state.lane.events[mark.index(2)];
                            match op {
                                Op::Record(..) => instance.services.record_event(event, stream)?,
                                Op::Wait(..) => instance.services.wait_event(stream, event)?,
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
            issue(self.tail.iter().collect(), Lane::Compute)
        })();
        if result.is_err() {
            instance.collectives.poison();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn micro_parts_preserve_granules_and_fit_reindex_counts() {
        assert_eq!(split(1023), None);
        assert_eq!(split(8193), None);
        for rows in 1024..=8192 {
            let [a, b] = split(rows).unwrap();
            assert_eq!(a % PREFILL_GRANULE, 0);
            assert!(a > 0 && b > 0 && a <= 4096 && b <= 4096);
            assert_eq!(a + b, rows);
        }
    }
}
