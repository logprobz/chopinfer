//! Exact upper-layer intervals, using original packed coordinates for carried
//! state and local row zero for each part's transient compute scratch.

use super::{PREFILL, Programs, Run};
use crate::buffers::row_range;
use crate::instance::{Instance, dspark::PrefillFeatureRange};
use crate::lower::Lowering;
use crate::ops::Lowered;
use crate::program::{Phase, Stage, Sublayer};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::template::{self, Patch, Site};
use bundle_runtime::upload::DeviceInputs;
use model_deepseek41_contract::prefill_cone::{self, Item};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span};
use std::ops::Range;

type Copy = (Span, Span);

mod dma;
#[cfg(test)]
pub(crate) use dma::with_enabled as with_dma;

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

pub(super) struct Plan {
    /// Each rank's copies before the step writes its slots: the anchor
    /// spares' seeds (`seeds`).
    seeds: Vec<Vec<Copy>>,
    body: Programs,
    /// The body's index selections, the only launches that read the
    /// positions: a kept template lowers them again (`patch`).
    sites: Vec<Site<Selection>>,
    dma: Option<dma::Plan>,
    head: Programs,
    copies: Vec<Vec<Copy>>,
    /// The (request, tokens) each anchor item keeps in its spare.
    kept: Vec<(u64, u64)>,
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
    /// positions, which lowering only checks against each other. `None`
    /// when the plan keeps anchors, whose spare seeds follow the items'
    /// slots.
    pub(super) fn structure(&self) -> Option<Structure> {
        if !self.anchors.is_empty() {
            return None;
        }
        Some(Structure {
            items: self.items.iter().map(|item| item.rows.clone()).collect(),
            cone: prefill_cone::Plan {
                endpoints: Vec::new(),
                ..self.cone.clone()
            },
            features: self
                .features
                .iter()
                .map(|feature| (feature.packed_start, feature.rows))
                .collect(),
        })
    }
}

/// `Layout::structure`.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct Structure {
    items: Vec<Range<u32>>,
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

    /// The plan of `run`, whose layout is `layout`.
    pub(super) fn lower(instance: &Instance, run: &Run, layout: &Layout) -> Result<Self, String> {
        let g = instance.config.shape.geometry;
        let inputs = &run.inputs;
        let (anchors, cone, features) = (&layout.anchors, &layout.cone, &layout.features);
        let cells = super::anchors::ring_cells(g);
        let ring_buffer = instance
            .state
            .buffers
            .iter()
            .position(|buffer| buffer.name == "window.ring")
            .ok_or("the request state has no window rings")?;
        let full_stages = crate::program::stages(g, run.phase);
        let mut seeds = Vec::new();
        let mut body = (Vec::new(), Vec::new(), Vec::new());
        let mut sites = Vec::new();
        let mut head = (Vec::new(), Vec::new(), Vec::new());
        let mut copies = Vec::new();
        for (rank, buffers) in instance.buffers.iter().enumerate() {
            let full = DeviceInputs::within(instance.arenas[PREFILL][rank], &run.layout);
            let base = lowering(instance, run, rank, &full, None);
            let mut program = (Vec::new(), Vec::new(), Vec::new());
            let mut rank_seeds = Vec::new();
            if !anchors.is_empty() {
                let held: Vec<_> = buffers
                    .state
                    .buffers
                    .iter()
                    .map(|&(_, span)| span)
                    .collect();
                for anchor in anchors {
                    let slot = u32::try_from(inputs.slot_ids[anchor.item])
                        .map_err(|_| "an anchor item has no state slot")?;
                    rank_seeds.extend(super::anchors::seed_copies(
                        &instance.state,
                        instance.extent,
                        &held,
                        &instance.anchor_spares[rank],
                        slot,
                        anchor,
                        cells,
                    )?);
                }
            }
            seeds.push(rank_seeds);
            // After a layer's window write into the slots, write the same
            // key rows within a ring of each anchor into its spare ring.
            // `part` holds the lowering's packed rows, from its transient
            // row zero.
            let anchor_writes = |program: &mut (Vec<PreparedLaunch>, Vec<bool>, Vec<Stage>),
                                 lowering: &Lowering<'_>,
                                 layer: u32,
                                 part: &Range<u32>|
             -> Result<(), String> {
                for anchor in anchors {
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
                                byte_range(
                                    full.positions,
                                    u64::from(rows.start) * 4,
                                    u64::from(n) * 4,
                                )?,
                                byte_range(full.positions, u64::from(anchor.row - 1) * 4, 4)?,
                                byte_range(spares.zeros, 0, u64::from(n) * 4)?,
                                count(instance, rank, n)?,
                            )?,
                            stage,
                        )?,
                    );
                }
                Ok(())
            };
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
                    // Only this full-width prefix has qualified cached attention.
                    // Retained parts below keep their native row origins.
                    _ => base.stage_with_prefill_context(
                        run.phase,
                        stage,
                        &run.inputs.positions,
                        run.cache,
                    )?,
                };
                let at = program.0.len();
                push(&mut program, stage, required(ops, stage)?);
                if let Stage::IndexSelect(layer) = stage {
                    sites.push(Site {
                        rank,
                        at,
                        len: program.0.len() - at,
                        input: Selection { layer, part: None },
                    });
                }
                if let Stage::WindowRingWrite(layer) = stage {
                    anchor_writes(&mut program, &base, layer, &(0..cone.rows))?;
                }
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
                    anchor_writes(&mut program, &lowering, layer.id, &range)?;
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
                        let at = program.0.len();
                        push(
                            &mut program,
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
        let dma = dma::Plan::prepare(instance, run, &body)?;
        Ok(Self {
            seeds,
            body,
            sites,
            dma,
            head,
            copies,
            kept: anchors
                .iter()
                .map(|anchor| (anchor.request, u64::from(anchor.tokens)))
                .collect(),
        })
    }

    /// Whether `other` issues the same launches, transport, copies and head.
    pub(super) fn same(&self, other: &Self) -> Result<bool, String> {
        let dma = match (&self.dma, &other.dma) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b)?,
            _ => false,
        };
        Ok(dma
            && self.seeds == other.seeds
            && self.kept == other.kept
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

    pub(super) fn body(&self) -> &Programs {
        &self.body
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

    pub(super) fn uses_dma(&self) -> bool {
        self.dma.is_some()
    }

    /// Whether `run`'s body may take upper-skip DMA (`dma::applies`).
    pub(super) fn dma_applies(instance: &Instance, run: &Run) -> bool {
        dma::applies(instance, run)
    }

    pub(super) fn prepare_engram(
        &self,
        plan: &mut crate::instance::prefill_engram::Plan,
        gathers: &crate::instance::prefill_engram::Gathers,
    ) -> Result<(), String> {
        if let Some(dma) = &self.dma {
            dma.prepare_engram(plan, gathers)?;
            plan.check_dma_collectives(&self.body.0, &self.body.1)?;
        }
        Ok(())
    }

    /// The (request, tokens) each anchor item keeps in its spare, in spare
    /// order, for the step's ledger.
    pub(super) fn kept(&self) -> &[(u64, u64)] {
        &self.kept
    }

    /// Rank `rank`'s copies, as (destination, source), that seed the anchor
    /// spares from the slots. They precede every write of the step to the
    /// slots, the compressor tails' own seed copies included.
    pub(super) fn seeds(&self, rank: usize) -> &[Copy] {
        self.seeds.get(rank).map_or(&[], Vec::as_slice)
    }

    pub(super) fn execute(
        &self,
        instance: &Instance,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
    ) -> Result<(), BundleAbiError> {
        self.execute_body(instance, gathers)?;
        self.compact_rows(instance)?;
        self.execute_head(instance)
    }

    /// The body: through upper-skip DMA, between prepared gather joins, or
    /// in one eager unit.
    fn execute_body(
        &self,
        instance: &Instance,
        gathers: Option<&crate::instance::prefill_engram::Plan>,
    ) -> Result<(), BundleAbiError> {
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        if let Some(dma) = &self.dma {
            dma.execute(instance, &self.body, gathers)
        } else {
            if let Some(gathers) = gathers {
                gathers.execute(instance, &self.body.0, &self.body.1)
            } else {
                instance.launch_eager(&streams, &self.body.0, &self.body.1)
            }
            .map_err(|e| instance.dispatch_failed(e))
        }
    }

    /// Moves each scored row to its logits position for the head.
    fn compact_rows(&self, instance: &Instance) -> Result<(), BundleAbiError> {
        for (rank, copies) in self.copies.iter().enumerate() {
            for &(destination, source) in copies {
                instance.services.copy_device_to_device(
                    instance.ranks[rank].2,
                    destination,
                    source,
                )?;
            }
        }
        Ok(())
    }

    fn execute_head(&self, instance: &Instance) -> Result<(), BundleAbiError> {
        let streams: Vec<_> = instance.ranks.iter().map(|r| r.2).collect();
        instance
            .launch_eager(&streams, &self.head.0, &self.head.1)
            .map_err(|e| instance.dispatch_failed(e))
    }
}
