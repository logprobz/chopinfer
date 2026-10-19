//! Exact upper-layer intervals, using original packed coordinates for carried
//! state and local row zero for each part's transient compute scratch.

use super::{PREFILL, Programs, Run};
use crate::buffers::row_range;
use crate::instance::{Instance, dspark::PrefillFeatureRange};
use crate::lower::Lowering;
use crate::ops::Lowered;
use crate::program::{Phase, Stage, Sublayer};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::upload::DeviceInputs;
use model_deepseek41_contract::prefill_cone::{self, Item};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span};
use std::ops::Range;

type Copy = (Span, Span);

pub(super) struct Plan {
    body: Programs,
    head: Programs,
    copies: Vec<Vec<Copy>>,
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
    }
}

impl Plan {
    pub(super) fn prepare(instance: &Instance, run: &Run) -> Result<Self, String> {
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
        let logits: Vec<u32> = inputs.logit_rows().iter().map(|&row| row as u32).collect();
        let cone = prefill_cone::Plan::new(g, &items, &logits, instance.dspark.is_some(), 64)?;
        let features: Vec<_> = if instance.dspark.is_some() {
            items
                .iter()
                .map(|item| {
                    let rows = item.rows.end - item.rows.start;
                    let keep = rows
                        - prefill_cone::feature_offset(item.prefix, rows, g.shape.attention.window);
                    PrefillFeatureRange {
                        packed_start: item.rows.end - keep,
                        rows: keep,
                        position: u64::from(item.prefix + rows - keep),
                        end: u64::from(item.prefix + rows),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let full_stages = crate::program::stages(g, run.phase);
        let mut body = (Vec::new(), Vec::new(), Vec::new());
        let mut head = (Vec::new(), Vec::new(), Vec::new());
        let mut copies = Vec::new();
        for (rank, buffers) in instance.buffers.iter().enumerate() {
            let full = DeviceInputs::within(instance.arenas[PREFILL][rank], &run.layout);
            let base = lowering(instance, run, rank, &full, None);
            let mut program = (Vec::new(), Vec::new(), Vec::new());
            for &stage in full_stages
                .iter()
                .take_while(|stage| stage.layer().is_none_or(|id| id < cone.last_source))
            {
                let ops = match stage {
                    Stage::EngramRows(id) if instance.dspark.is_some() => {
                        instance.dspark_engram(&base, run.phase, id)?
                    }
                    Stage::DsparkFeature(id) if instance.dspark.is_some() => {
                        Some(instance.dspark_feature(&base, run.phase, id)?)
                    }
                    _ => base.stage(run.phase, stage)?,
                };
                push(&mut program, stage, required(ops, stage)?);
            }
            // The final compressed and indexed source still consumes all rows.
            for stage in prefix(cone.last_source).into_iter().chain(
                crate::program::attention(g, cone.last_source, run.phase)
                    .into_iter()
                    .filter(|&stage| compressed(stage)),
            ) {
                push(
                    &mut program,
                    stage,
                    required(base.stage(run.phase, stage)?, stage)?,
                );
            }
            for layer in &cone.layers {
                for &feature in features
                    .iter()
                    .filter(|_| g.shape.dspark.targets.contains(&layer.id))
                {
                    let range = feature.packed_start..feature.packed_start + feature.rows;
                    let input = part(instance, rank, full, &range)?;
                    let lowering = lowering(instance, run, rank, &input, Some(range.start));
                    push(
                        &mut program,
                        Stage::DsparkFeature(layer.id),
                        instance.dspark_feature_range(&lowering, layer.id, feature)?,
                    );
                }
                if layer
                    .keys
                    .ranges()
                    .first()
                    .is_some_and(|range| range.start > 0)
                {
                    push(
                        &mut program,
                        Stage::WindowKv(layer.id),
                        required(
                            base.prefill_window_prefix(layer.id)?,
                            Stage::WindowKv(layer.id),
                        )?,
                    );
                }
                // Produce every needed key before any output overwrites its HC streams.
                for range in layer.keys.parts(4096) {
                    let input = part(instance, rank, full, &range)?;
                    let lowering = lowering(instance, run, rank, &input, Some(range.start));
                    let phase = Phase::Prefill {
                        rows: range.end - range.start,
                    };
                    for stage in prefix(layer.id).into_iter().chain([
                        Stage::Project(layer.id, "attention.kv"),
                        Stage::WindowKv(layer.id),
                        Stage::WindowRingWrite(layer.id),
                    ]) {
                        push(
                            &mut program,
                            stage,
                            required(lowering.stage(phase, stage)?, stage)?,
                        );
                    }
                }
                for range in layer.outputs.parts(4096) {
                    let input = part(instance, rank, full, &range)?;
                    let lowering = lowering(instance, run, rank, &input, Some(range.start));
                    let phase = Phase::Prefill {
                        rows: range.end - range.start,
                    };
                    for &stage in full_stages
                        .iter()
                        .filter(|stage| stage.layer() == Some(layer.id))
                    {
                        if compressed(stage)
                            || matches!(
                                stage,
                                Stage::DsparkFeature(_)
                                    | Stage::WindowKv(_)
                                    | Stage::WindowRingWrite(_)
                                    | Stage::Project(_, "attention.kv")
                            )
                        {
                            continue;
                        }
                        push(
                            &mut program,
                            stage,
                            required(lowering.stage(phase, stage)?, stage)?,
                        );
                    }
                }
            }
            for &feature in &features {
                let counts = instance
                    .dspark
                    .as_ref()
                    .ok_or("DSpark is absent")?
                    .prepared
                    .feature_pack_counts(feature.rows, feature.rows)?
                    .into_iter()
                    .map(|n| count(instance, rank, n))
                    .collect::<Result<Vec<_>, _>>()?;
                push(
                    &mut program,
                    Stage::HeadCollapse,
                    instance.dspark_pack_feature_range(rank, feature, &counts)?,
                );
            }
            body.0.push(program.0);
            body.1.push(program.1);
            body.2.push(program.2);
            let mut program = (Vec::new(), Vec::new(), Vec::new());
            let mut rank_copies = Vec::new();
            if run.head_rows > 0 {
                // Ascending original rows imply destination <= source. No copy
                // can overwrite the source of a later logits row.
                for (destination, &source) in cone.logits.iter().enumerate() {
                    if destination as u32 == source {
                        continue;
                    }
                    for span in [buffers.scratch.streams, buffers.scratch.pre_odd] {
                        rank_copies.push((
                            row_range(span, destination as u32, 1)?,
                            row_range(span, source, 1)?,
                        ));
                    }
                }
                let input = DeviceInputs {
                    live_tokens: count(instance, rank, run.head_rows)?,
                    ..full
                };
                let lowering = lowering(instance, run, rank, &input, None);
                let phase = Phase::Prefill {
                    rows: run.head_rows,
                };
                for stage in [
                    Stage::HeadCollapse,
                    Stage::FinalNorm,
                    Stage::GatherHeadRows,
                    Stage::LmHead,
                    Stage::GatherLogits,
                ] {
                    push(
                        &mut program,
                        stage,
                        required(lowering.stage(phase, stage)?, stage)?,
                    );
                }
            }
            head.0.push(program.0);
            head.1.push(program.1);
            head.2.push(program.2);
            copies.push(rank_copies);
        }
        bundle_runtime::pipeline::check_collectives(&body.0, &body.1)
            .map_err(|e| format!("prefill body collectives: {e:?}"))?;
        bundle_runtime::pipeline::check_collectives(&head.0, &head.1)
            .map_err(|e| format!("prefill head collectives: {e:?}"))?;
        Ok(Self { body, head, copies })
    }

    pub(super) fn execute(&self, instance: &Instance) -> Result<(), BundleAbiError> {
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        instance
            .launch_eager(&streams, &self.body.0, &self.body.1)
            .map_err(|e| instance.dispatch_failed(e))?;
        for (rank, copies) in self.copies.iter().enumerate() {
            for &(destination, source) in copies {
                instance
                    .services
                    .copy_device_to_device(streams[rank], destination, source)?;
            }
        }
        instance
            .launch_eager(&streams, &self.head.0, &self.head.1)
            .map_err(|e| instance.dispatch_failed(e))
    }
}
