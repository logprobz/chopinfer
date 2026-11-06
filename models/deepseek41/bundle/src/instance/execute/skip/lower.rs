//! An upper-skip body lowered one group of stages at a time on every rank:
//! the full-width stages of each layer below the cone, the last source's
//! full-width prefix and compressed stages, each cone layer, then the
//! DSpark feature packs. Each rank lowers its stages in the order a
//! whole-plan lowering does, so the drained lowerer makes the same plan
//! (`fake_host_tests::prefill_plan_templates::metadata`).

use super::{
    Layout, Plan, Selection, byte_range, compressed, count, lowering, part, prefix, push, required,
};
use crate::buffers::row_range;
use crate::instance::Instance;
use crate::instance::execute::{PREFILL, Run};
use crate::lower::Lowering;
use crate::program::{Phase, Stage};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::template::Site;
use bundle_runtime::upload::DeviceInputs;
use std::ops::Range;

type Program = (Vec<PreparedLaunch>, Vec<bool>, Vec<Stage>);

/// Stages lowered on every rank in turn.
#[derive(Clone, Debug)]
enum Group {
    /// The full-width stages of one layer below the cone, or of none: a
    /// range of `Lowerer::stages`.
    Full(Range<usize>),
    /// The last source's prefix and compressed stages, at full width.
    LastSource,
    /// The cone layer at this index of the cone's layers.
    Cone(usize),
    /// The DSpark feature packs.
    Packs,
}

/// One rank's body so far.
#[derive(Default)]
struct Body {
    program: Program,
    sites: Vec<Site<Selection>>,
}

pub(super) struct Lowerer {
    /// The phase's stages; those below the cone's last source are lowered
    /// at full width.
    stages: Vec<Stage>,
    groups: Vec<Group>,
    next: usize,
    /// The request state's window-ring buffer, which anchor spares copy.
    ring_buffer: usize,
    ranks: Vec<Body>,
}

impl Lowerer {
    pub(super) fn new(instance: &Instance, run: &Run, layout: &Layout) -> Result<Self, String> {
        let g = instance.config.shape.geometry;
        let cone = &layout.cone;
        let ring_buffer = instance
            .state
            .buffers
            .iter()
            .position(|buffer| buffer.name == "window.ring")
            .ok_or("the request state has no window rings")?;
        let stages = crate::program::stages(g, run.phase);
        let below = stages
            .iter()
            .take_while(|stage| stage.layer().is_none_or(|id| id < cone.last_source))
            .count();
        let mut groups = Vec::new();
        let mut start = 0;
        for at in 1..=below {
            if at == below || stages[at].layer() != stages[start].layer() {
                groups.push(Group::Full(start..at));
                start = at;
            }
        }
        groups.push(Group::LastSource);
        groups.extend((0..cone.layers.len()).map(Group::Cone));
        groups.push(Group::Packs);
        Ok(Self {
            stages,
            groups,
            next: 0,
            ring_buffer,
            ranks: instance.buffers.iter().map(|_| Body::default()).collect(),
        })
    }

    /// Lowers the next group on every rank, in rank order; false once every
    /// group is lowered. A lowering error leaves the lowerer spent.
    pub(super) fn lower_next(
        &mut self,
        instance: &Instance,
        run: &Run,
        layout: &Layout,
    ) -> Result<bool, String> {
        let Some(group) = self.groups.get(self.next).cloned() else {
            return Ok(false);
        };
        for rank in 0..self.ranks.len() {
            self.lower_group(instance, run, layout, rank, &group)?;
        }
        self.next += 1;
        Ok(true)
    }

    fn lower_group(
        &mut self,
        instance: &Instance,
        run: &Run,
        layout: &Layout,
        rank: usize,
        group: &Group,
    ) -> Result<(), String> {
        let g = instance.config.shape.geometry;
        let (cone, features) = (&layout.cone, &layout.features);
        let full = DeviceInputs::within(instance.arenas[PREFILL][rank], &run.layout);
        let base = lowering(instance, run, rank, &full, None);
        let ring_buffer = self.ring_buffer;
        let Body { program, sites } = &mut self.ranks[rank];
        match group {
            Group::Full(range) => {
                for &stage in &self.stages[range.clone()] {
                    let ops = match stage {
                        Stage::EngramRows(id) if instance.dspark.is_some() => {
                            instance.dspark_engram(&base, run.phase, id)?
                        }
                        Stage::DsparkFeature(id) if instance.dspark.is_some() => {
                            Some(instance.dspark_feature(&base, run.phase, id)?)
                        }
                        // Only this full-width prefix has qualified cached
                        // attention. Retained parts below keep their native
                        // row origins.
                        _ => base.stage_with_prefill_context(
                            run.phase,
                            stage,
                            &run.inputs.positions,
                            run.cache,
                        )?,
                    };
                    let at = program.0.len();
                    push(program, stage, required(ops, stage)?);
                    if let Stage::IndexSelect(layer) = stage {
                        sites.push(Site {
                            rank,
                            at,
                            len: program.0.len() - at,
                            input: Selection { layer, part: None },
                        });
                    }
                    if let Stage::WindowRingWrite(layer) = stage {
                        let rows = 0..cone.rows;
                        anchor_writes(
                            instance,
                            layout,
                            rank,
                            ring_buffer,
                            full,
                            program,
                            &base,
                            layer,
                            &rows,
                        )?;
                    }
                }
            }
            Group::LastSource => {
                // The final compressed and indexed source still consumes all
                // rows.
                for stage in prefix(cone.last_source).into_iter().chain(
                    crate::program::attention(g, cone.last_source, run.phase)
                        .into_iter()
                        .filter(|&stage| compressed(stage)),
                ) {
                    push(
                        program,
                        stage,
                        required(base.stage(run.phase, stage)?, stage)?,
                    );
                }
            }
            Group::Cone(index) => {
                let layer = &cone.layers[*index];
                for &feature in features
                    .iter()
                    .filter(|_| g.shape.dspark.targets.contains(&layer.id))
                {
                    let range = feature.packed_start..feature.packed_start + feature.rows;
                    let input = part(instance, rank, full, &range)?;
                    let lowering = lowering(instance, run, rank, &input, Some(range.start));
                    push(
                        program,
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
                        program,
                        Stage::WindowKv(layer.id),
                        required(
                            base.prefill_window_prefix(layer.id)?,
                            Stage::WindowKv(layer.id),
                        )?,
                    );
                }
                // Produce every needed key before any output overwrites its
                // HC streams.
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
                            program,
                            stage,
                            required(
                                lowering.stage_with_positions(
                                    phase,
                                    stage,
                                    &run.inputs.positions[range.start as usize..range.end as usize],
                                )?,
                                stage,
                            )?,
                        );
                    }
                    anchor_writes(
                        instance,
                        layout,
                        rank,
                        ring_buffer,
                        full,
                        program,
                        &lowering,
                        layer.id,
                        &range,
                    )?;
                }
                for range in layer.outputs.parts(4096) {
                    let input = part(instance, rank, full, &range)?;
                    let lowering = lowering(instance, run, rank, &input, Some(range.start));
                    let phase = Phase::Prefill {
                        rows: range.end - range.start,
                    };
                    for &stage in self
                        .stages
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
                        let at = program.0.len();
                        push(
                            program,
                            stage,
                            required(
                                lowering.stage_with_prefill_context(
                                    phase,
                                    stage,
                                    &run.inputs.positions[range.start as usize..range.end as usize],
                                    run.cache,
                                )?,
                                stage,
                            )?,
                        );
                        if let Stage::IndexSelect(layer) = stage {
                            sites.push(Site {
                                rank,
                                at,
                                len: program.0.len() - at,
                                input: Selection {
                                    layer,
                                    part: Some(range.clone()),
                                },
                            });
                        }
                    }
                }
            }
            Group::Packs => {
                for &feature in features {
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
                        program,
                        Stage::HeadCollapse,
                        instance.dspark_pack_feature_range(rank, feature, &counts)?,
                    );
                }
            }
        }
        Ok(())
    }

    /// Lowers every remaining group, then each rank's head and row copies,
    /// and checks the collectives and prepares the DMA as `Plan::lower`
    /// does.
    pub(super) fn finish(
        mut self,
        instance: &Instance,
        run: &Run,
        layout: &Layout,
    ) -> Result<Plan, String> {
        while self.lower_next(instance, run, layout)? {}
        let cone = &layout.cone;
        let mut body = (Vec::new(), Vec::new(), Vec::new());
        let mut sites = Vec::new();
        let mut head = (Vec::new(), Vec::new(), Vec::new());
        let mut copies = Vec::new();
        for (rank, (lowered, buffers)) in self.ranks.into_iter().zip(&instance.buffers).enumerate()
        {
            body.0.push(lowered.program.0);
            body.1.push(lowered.program.1);
            body.2.push(lowered.program.2);
            sites.extend(lowered.sites);
            let full = DeviceInputs::within(instance.arenas[PREFILL][rank], &run.layout);
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
        let dma = super::dma::Plan::prepare(instance, run, &body)?;
        Ok(Plan {
            body,
            sites,
            dma,
            head,
            copies,
            engram: None,
            #[cfg(test)]
            _freed: super::Freed,
        })
    }
}

/// After a layer's window write into the slots, writes the same key rows
/// within a ring of each anchor into its spare ring. `part` holds the
/// lowering's packed rows, from its transient row zero.
#[allow(clippy::too_many_arguments)]
fn anchor_writes(
    instance: &Instance,
    layout: &Layout,
    rank: usize,
    ring_buffer: usize,
    full: DeviceInputs,
    program: &mut Program,
    lowering: &Lowering<'_>,
    layer: u32,
    part: &Range<u32>,
) -> Result<(), String> {
    let g = instance.config.shape.geometry;
    let cells = crate::instance::execute::anchors::ring_cells(g);
    for anchor in &layout.anchors {
        let ring = anchor.ring_rows(cells);
        let rows = ring.start.max(part.start)..ring.end.min(part.end);
        if rows.is_empty() {
            continue;
        }
        let n = rows.end - rows.start;
        let spares = &instance.anchor_spares[rank];
        let spare = crate::lower::spare_ring(
            g,
            instance.config.kv.storage,
            instance.anchor_spare_count(),
            spares.buffers[ring_buffer].ok_or("the window rings have no spare")?,
            layer,
            anchor.spare,
        )?;
        let stage = Stage::WindowRingWrite(layer);
        push(
            program,
            stage,
            required(
                lowering.spare_ring_write(
                    layer,
                    rows.start - part.start,
                    rows.start,
                    n,
                    spare,
                    byte_range(full.positions, u64::from(rows.start) * 4, u64::from(n) * 4)?,
                    byte_range(full.positions, u64::from(anchor.row - 1) * 4, 4)?,
                    byte_range(spares.zeros, 0, u64::from(n) * 4)?,
                    count(instance, rank, n)?,
                )?,
                stage,
            )?,
        );
    }
    Ok(())
}
