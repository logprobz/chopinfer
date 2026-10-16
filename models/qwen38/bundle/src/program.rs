//! The ordered rank program of one prefill or decode phase.
//!
//! The residual is four hyper-connection streams wide. Every sublayer
//! (attention, then MoE) reads `mix(norm(X))` and ends in one all-reduce
//! that injects its output into X and normalizes X for the next sublayer.
//! Layer 1 first adds the per-layer n-gram embedding (PLE) to X.
//!
//! The MTP drafter's program (`drafter_stages`) runs its one full-attention
//! layer, numbered `MTP_LAYER`, on X built from the target's final X and the
//! next token, then its own mixer, the shared head and the draft argmax.

#[cfg(test)]
use crate::contract::{LAYERS, MTP_LAYER, is_full};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Sublayer {
    Attention,
    Mlp,
}

/// Which prepared shapes a phase runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Eager, planned at the live token count.
    Prefill,
    /// A graph bucket of this many items, one token each.
    Decode(u32),
    /// A graph bucket of `rows` rows over at most `items` items of one to
    /// `MAX_VERIFY_ROWS` rows each for target verify. Drafter catch-up runs
    /// may put up to `DECODE_BATCH` rows in one item.
    Verify { rows: u32, items: u32 },
}

impl Phase {
    /// The prepared rows of a graph phase.
    pub const fn bucket(self) -> Option<u32> {
        match self {
            Self::Prefill => None,
            Self::Decode(rows) | Self::Verify { rows, .. } => Some(rows),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    /// Each item's PLE history, from its slot or fresh, into the phase's
    /// staging rows.
    PleStage,
    /// Prefill continues each item's linear state in a fresh pool entry.
    LinearStateCopy,
    Embedding,
    /// Prefill rows of media placeholders take their encoder rows from the
    /// media arena; nothing launches for a step without injects.
    MediaInject,
    /// X = the embedding repeated into every stream.
    ExpandStreams,
    FirstNorm,
    /// PLE, before layer 1's attention.
    NgramGather,
    NgramAllGather,
    PleProject,
    PleKeyNorm,
    PleGate,
    PleConvNorm,
    PleConv,
    /// Verify rows: the PLE state after each row, for the commit to keep.
    PleCheckpoints,
    /// Prefill: the PLE state at each item's anchor, for a prefix capture.
    PleAnchor,
    PleResidual,
    PleNorm,
    /// Hyper-connection input: down and inject projections, the rank-320
    /// bottleneck, the up projection and the stream mix.
    HcIn(u32, Sublayer),
    HcBottleneck(u32, Sublayer),
    HcUp(u32, Sublayer),
    HcMix(u32, Sublayer),
    LinearInProj(u32),
    LinearConv(u32),
    LinearDelta(u32),
    LinearNorm(u32),
    LinearOut(u32),
    AttnInProj(u32),
    AttnQkNormRopeKv(u32),
    IndexQuery(u32),
    IndexKeyWrite(u32),
    /// Block scores and top-k selection; prefill scores and selects in
    /// pieces of `SCORE_ROWS` rows, one after the other.
    BlockSelect(u32),
    SparseAttention(u32),
    AttnGate(u32),
    AttnOut(u32),
    Router(u32),
    Route(u32),
    ExpertsGateUp(u32),
    ExpertsDown(u32),
    SharedGateUp(u32),
    SharedDown(u32),
    Combine(u32),
    /// All-reduce of the sublayer's rank partials, inject into X and the
    /// next sublayer's grouped norm.
    Reduce(u32, Sublayer),
    /// The final hyper-connection mixer (no inject).
    MixerIn,
    MixerBottleneck,
    MixerUp,
    MixerMix,
    /// Prefill gathers the rows that need logits before the head.
    GatherHeadRows,
    LmHead,
    GatherLogits,
    /// The drafter's input: the target X of each row and its next token.
    MtpGather,
    /// Gemma RMSNorm of the target X over all four streams, and of the next
    /// token's embedding.
    MtpNorms,
    /// `fc_hidden` of each stream and `fc_embedding` of the token.
    MtpFc,
    /// X = fc_hidden(...)_s + fc_embedding(...) for each stream s.
    MtpCombine,
    /// Each rank's best logit of every draft row, its exchange, and the
    /// drafts written where the next drafter run and the verify rows read
    /// them.
    DraftBest,
    DraftExchange,
    DraftFinal,
}

impl Stage {
    #[cfg(test)]
    pub const fn layer(self) -> Option<u32> {
        self.layer_for(crate::contract::Shape::TP4)
    }
    pub const fn layer_for(self, shape: crate::contract::Shape) -> Option<u32> {
        match self {
            Self::HcIn(layer, _)
            | Self::HcBottleneck(layer, _)
            | Self::HcUp(layer, _)
            | Self::HcMix(layer, _)
            | Self::Reduce(layer, _)
            | Self::LinearInProj(layer)
            | Self::LinearConv(layer)
            | Self::LinearDelta(layer)
            | Self::LinearNorm(layer)
            | Self::LinearOut(layer)
            | Self::AttnInProj(layer)
            | Self::AttnQkNormRopeKv(layer)
            | Self::IndexQuery(layer)
            | Self::IndexKeyWrite(layer)
            | Self::BlockSelect(layer)
            | Self::SparseAttention(layer)
            | Self::AttnGate(layer)
            | Self::AttnOut(layer)
            | Self::Router(layer)
            | Self::Route(layer)
            | Self::ExpertsGateUp(layer)
            | Self::ExpertsDown(layer)
            | Self::SharedGateUp(layer)
            | Self::SharedDown(layer)
            | Self::Combine(layer) => Some(layer),
            Self::NgramGather
            | Self::NgramAllGather
            | Self::PleProject
            | Self::PleKeyNorm
            | Self::PleGate
            | Self::PleConvNorm
            | Self::PleConv
            | Self::PleCheckpoints
            | Self::PleAnchor
            | Self::PleResidual
            | Self::PleNorm => Some(shape.ple_layer()),
            _ => None,
        }
    }
}

fn attention(layer: u32) -> [Stage; 8] {
    [
        Stage::AttnInProj(layer),
        Stage::AttnQkNormRopeKv(layer),
        Stage::IndexQuery(layer),
        Stage::IndexKeyWrite(layer),
        Stage::BlockSelect(layer),
        Stage::SparseAttention(layer),
        Stage::AttnGate(layer),
        Stage::AttnOut(layer),
    ]
}

fn linear(layer: u32) -> [Stage; 5] {
    [
        Stage::LinearInProj(layer),
        Stage::LinearConv(layer),
        Stage::LinearDelta(layer),
        Stage::LinearNorm(layer),
        Stage::LinearOut(layer),
    ]
}

fn moe(layer: u32) -> [Stage; 7] {
    [
        Stage::Router(layer),
        Stage::Route(layer),
        Stage::ExpertsGateUp(layer),
        Stage::ExpertsDown(layer),
        Stage::SharedGateUp(layer),
        Stage::SharedDown(layer),
        Stage::Combine(layer),
    ]
}

fn hc_in(layer: u32, sublayer: Sublayer) -> [Stage; 4] {
    [
        Stage::HcIn(layer, sublayer),
        Stage::HcBottleneck(layer, sublayer),
        Stage::HcUp(layer, sublayer),
        Stage::HcMix(layer, sublayer),
    ]
}

/// Every stage of one phase, in issue order. Prefill alone copies linear
/// state and gathers the rows that need logits; verify alone keeps the PLE
/// state after each row.
pub fn stages_for(shape: crate::contract::Shape, phase: Phase) -> Vec<Stage> {
    let prefill = phase == Phase::Prefill;
    let mut result = Vec::with_capacity(1100);
    result.push(Stage::PleStage);
    if prefill {
        result.push(Stage::LinearStateCopy);
    }
    result.push(Stage::Embedding);
    if prefill {
        result.push(Stage::MediaInject);
    }
    result.extend([Stage::ExpandStreams, Stage::FirstNorm]);
    for layer in 0..shape.layers() {
        if layer == shape.ple_layer() {
            result.extend([
                Stage::NgramGather,
                Stage::NgramAllGather,
                Stage::PleProject,
                Stage::PleKeyNorm,
                Stage::PleGate,
                Stage::PleConvNorm,
                Stage::PleConv,
            ]);
            if matches!(phase, Phase::Verify { .. }) {
                result.push(Stage::PleCheckpoints);
            }
            if phase == Phase::Prefill {
                result.push(Stage::PleAnchor);
            }
            result.extend([Stage::PleResidual, Stage::PleNorm]);
        }
        result.extend(hc_in(layer, Sublayer::Attention));
        if shape.is_full(layer) {
            result.extend(attention(layer));
        } else {
            result.extend(linear(layer));
        }
        result.push(Stage::Reduce(layer, Sublayer::Attention));
        result.extend(hc_in(layer, Sublayer::Mlp));
        result.extend(moe(layer));
        result.push(Stage::Reduce(layer, Sublayer::Mlp));
    }
    result.extend([
        Stage::MixerIn,
        Stage::MixerBottleneck,
        Stage::MixerUp,
        Stage::MixerMix,
    ]);
    if prefill {
        result.push(Stage::GatherHeadRows);
    }
    result.extend([Stage::LmHead, Stage::GatherLogits]);
    result
}

/// The drafter's stages: its input, its layer and, when the run drafts, its
/// mixer, the head over the draft rows and the argmax.
pub fn drafter_stages_for(shape: crate::contract::Shape, drafts: bool) -> Vec<Stage> {
    let mut result = vec![
        Stage::MtpGather,
        Stage::Embedding,
        Stage::MtpNorms,
        Stage::MtpFc,
        Stage::MtpCombine,
        Stage::FirstNorm,
    ];
    result.extend(hc_in(shape.mtp_layer(), Sublayer::Attention));
    result.extend(attention(shape.mtp_layer()));
    result.push(Stage::Reduce(shape.mtp_layer(), Sublayer::Attention));
    result.extend(hc_in(shape.mtp_layer(), Sublayer::Mlp));
    result.extend(moe(shape.mtp_layer()));
    result.push(Stage::Reduce(shape.mtp_layer(), Sublayer::Mlp));
    if drafts {
        result.extend([
            Stage::MixerIn,
            Stage::MixerBottleneck,
            Stage::MixerUp,
            Stage::MixerMix,
            Stage::GatherHeadRows,
            Stage::LmHead,
            Stage::DraftBest,
            Stage::DraftExchange,
            Stage::DraftFinal,
        ]);
    }
    result
}

#[cfg(test)]
pub fn stages(phase: Phase) -> Vec<Stage> {
    stages_for(crate::contract::Shape::TP4, phase)
}

#[cfg(test)]
pub fn drafter_stages(drafts: bool) -> Vec<Stage> {
    drafter_stages_for(crate::contract::Shape::TP4, drafts)
}

/// The opt-in prefill's two parts and four reusable event classes.
pub const OVERLAP_PARTS: usize = 2;
pub const OVERLAP_EVENTS: usize = 4 * OVERLAP_PARTS;

/// Keep the original stage order inside each part. Setup and the head run
/// once; every complete Reduce pair runs on the communication lane.
pub fn overlap_schedule(shape: crate::contract::Shape) -> Vec<bundle_runtime::lanes::Op<Stage>> {
    use bundle_runtime::lanes::{Block, Lane, Program};
    let mut program = Program::<Stage> {
        whole: Vec::new(),
        prologue: Vec::new(),
        blocks: Vec::new(),
        epilogue: Vec::new(),
    };
    let mut compute = Vec::new();
    for stage in stages_for(shape, Phase::Prefill) {
        match stage {
            Stage::PleStage | Stage::LinearStateCopy => program.whole.push(stage),
            Stage::Embedding | Stage::MediaInject | Stage::ExpandStreams | Stage::FirstNorm => {
                program.prologue.push(stage)
            }
            Stage::GatherHeadRows | Stage::LmHead | Stage::GatherLogits => {
                program.epilogue.push(stage)
            }
            Stage::Reduce(_, sublayer) => {
                let mark = if sublayer == Sublayer::Attention {
                    0
                } else {
                    2
                };
                program.blocks.push(Block {
                    lane: Lane::Compute,
                    mark,
                    stages: std::mem::take(&mut compute),
                });
                program.blocks.push(Block {
                    lane: Lane::Comm,
                    mark: mark + 1,
                    stages: vec![stage],
                });
            }
            _ => compute.push(stage),
        }
    }
    program.blocks.push(Block {
        lane: Lane::Compute,
        mark: 0,
        stages: compute,
    });
    debug_assert_eq!(program.events(OVERLAP_PARTS), OVERLAP_EVENTS);
    program.schedule(OVERLAP_PARTS)
}

/// Select the nearest legal contiguous boundary to the chunk's midpoint.
/// Crossings preserve both the item's delta chunk grid and absolute granules.
pub fn overlap_split(
    inputs: &crate::inputs::PhaseInputs,
    anchor_ends: &[Option<i32>],
) -> Option<u32> {
    let total = inputs.live_tokens() as i32;
    let middle = total / 2;
    let granule = crate::contract::PREFILL_GRANULE as i32;
    inputs
        .q_start
        .iter()
        .zip(&inputs.q_len)
        .zip(&inputs.prefix_len)
        .flat_map(|((&start, &length), &prefix)| {
            let position = prefix + (middle - start).clamp(0, length);
            let below = position / granule * granule;
            [
                start,
                start + length,
                start + below - prefix - granule,
                start + below - prefix,
                start + below + granule - prefix,
            ]
            .into_iter()
            .filter(move |&at| {
                at >= start
                    && at <= start + length
                    && (at == start
                        || at == start + length
                        || (prefix % granule == 0 && (at - start) % granule == 0))
            })
        })
        .filter(|&at| at > 0 && at < total)
        .filter(|&at| inputs.can_split_at_with_rows(at as u32, anchor_ends))
        // Linear kernels capture only a partial final granule. A split at
        // the anchor frontier would leave neither half writing that state.
        .filter(|&at| {
            !anchor_ends.iter().enumerate().any(|(item, end)| {
                end.is_some_and(|end| at == inputs.q_start[item] + end - inputs.prefix_len[item])
            })
        })
        .min_by_key(|&at| ((at - middle).unsigned_abs(), at))
        .map(|at| at as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_keeps_each_parts_stage_order_and_records_before_waits() {
        use bundle_runtime::lanes::{Lane, Op};
        for geometry in model_qwen38_contract::geometry::GEOMETRIES {
            for ranks in [crate::contract::Ranks::TP2, crate::contract::Ranks::TP4] {
                let shape = crate::contract::Shape::of_geometry(geometry, ranks).unwrap();
                let stages = stages_for(shape, Phase::Prefill);
                let ops = overlap_schedule(shape);
                let mut recorded = [false; OVERLAP_EVENTS];
                let mut latest = [None; OVERLAP_PARTS];
                let mut marks = [None; OVERLAP_EVENTS];
                let mut visible = [[0_usize; OVERLAP_PARTS]; 2];
                for (sequence, op) in ops.iter().enumerate() {
                    match *op {
                        Op::Launch {
                            part: Some(part),
                            lane,
                            ..
                        } => {
                            if let Some((before, position)) = latest[part]
                                && before != lane
                            {
                                assert!(
                                    visible[lane as usize][part] >= position,
                                    "part {part}: {lane:?} did not wait for {before:?}"
                                );
                            }
                            latest[part] = Some((lane, sequence + 1));
                        }
                        Op::Record(lane, mark) => {
                            let (before, position) = latest[mark.part].unwrap();
                            assert_eq!(before, lane);
                            marks[mark.index(OVERLAP_PARTS)] = Some(position);
                        }
                        Op::Wait(lane, mark) => {
                            visible[lane as usize][mark.part] =
                                marks[mark.index(OVERLAP_PARTS)].unwrap();
                        }
                        _ => {}
                    }
                    match *op {
                        Op::Record(_, mark) => recorded[mark.index(OVERLAP_PARTS)] = true,
                        Op::Wait(_, mark) => assert!(recorded[mark.index(OVERLAP_PARTS)]),
                        Op::Launch { stage, lane, .. } => {
                            assert_eq!(lane == Lane::Comm, matches!(stage, Stage::Reduce(..)))
                        }
                    }
                }
                for part in 0..OVERLAP_PARTS {
                    let actual = ops
                        .iter()
                        .filter_map(|op| match *op {
                            Op::Launch {
                                stage,
                                part: launched,
                                ..
                            } if launched.is_none() || launched == Some(part) => Some(stage),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(actual, stages);
                }
            }
        }
    }

    #[test]
    fn overlap_split_rejects_the_same_inputs_as_row_splitting() {
        let mut inputs = crate::inputs::PhaseInputs {
            token_ids: vec![0; 128],
            q_start: vec![0],
            q_len: vec![128],
            prefix_len: vec![0],
            anchors: vec![false],
            item_order: vec![0],
            ..crate::inputs::PhaseInputs::default()
        };
        for verify in [false, true] {
            inputs.verify = verify;
            inputs.decode = !verify;
            assert!(inputs.split_at_with_rows(64, &[None]).is_none());
            assert!(overlap_split(&inputs, &[None]).is_none());
        }
        inputs.decode = false;
        inputs.verify = false;
        assert!(inputs.split_at_with_rows(64, &[]).is_none());
        assert!(overlap_split(&inputs, &[]).is_none());
    }

    #[test]
    fn overlap_split_preserves_linear_anchor_at_the_midpoint() {
        for (first, second) in [(1025, 1023), (1030, 1000)] {
            let inputs = crate::inputs::PhaseInputs {
                token_ids: vec![0; first + second],
                q_start: vec![0, first as i32],
                q_len: vec![first as i32, second as i32],
                prefix_len: vec![0, 0],
                anchors: vec![true, false],
                item_order: vec![0, 1],
                ..crate::inputs::PhaseInputs::default()
            };
            let at = overlap_split(&inputs, &[Some(1024), None]).unwrap();
            assert_ne!(
                at, 1024,
                "the linear kernels do not write a 64-aligned tail anchor"
            );
            assert_eq!(at, first as u32);
        }
    }

    #[test]
    fn generated_batches_preserve_each_crossing_chunk_grid_and_anchor_owner() {
        let mut seed = 17_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        let mut admitted = 0;
        let mut anchored = 0;
        for _ in 0..256 {
            let mut inputs = crate::inputs::PhaseInputs::default();
            let mut ends = Vec::new();
            let mut start = 0;
            for item in 0..1 + next() % 8 {
                let length = 1 + (next() % 1024) as i32;
                let prefix = [0, 64, 128, 193, 10, 2048][next() as usize % 6];
                let end = prefix + length;
                let anchor = end / 64 * 64;
                let keep =
                    next() % 3 == 0 && prefix % 64 == 0 && end % 64 != 0 && anchor - prefix >= 64;
                inputs.item_order.push(item as usize);
                inputs.request_ids.push(u64::from(item) + 10);
                inputs.slot_ids.push(item as i32);
                inputs.q_start.push(start);
                inputs.q_len.push(length);
                inputs.prefix_len.push(prefix);
                inputs.seq_lens.push(end);
                inputs.last_positions.push(i64::from(end - 1));
                inputs.emits_row.push(true);
                inputs.prompt_rows.push(false);
                inputs.anchors.push(keep);
                inputs
                    .block_tables
                    .push(vec![0; (end as usize).div_ceil(64)]);
                inputs.token_ids.extend(vec![7; length as usize]);
                inputs.positions.extend(prefix..end);
                inputs
                    .token_batch
                    .extend(vec![item as i32; length as usize]);
                ends.push(keep.then_some(anchor));
                start += length;
            }
            let Some(at) = overlap_split(&inputs, &ends) else {
                continue;
            };
            admitted += 1;
            for (item, &start) in inputs.q_start.iter().enumerate() {
                if start < at as i32 && (at as i32) < start + inputs.q_len[item] {
                    assert_eq!((at as i32 - start) % 64, 0);
                    assert_eq!(inputs.prefix_len[item] % 64, 0);
                }
                if let Some(end) = ends[item] {
                    assert_ne!(at as i32, start + end - inputs.prefix_len[item]);
                }
            }
            let (a, b) = inputs.split_at_with_rows(at, &ends).unwrap();
            for half in [&a, &b] {
                for (item, &keep) in half.anchors.iter().enumerate() {
                    if !keep {
                        continue;
                    }
                    anchored += 1;
                    let original = inputs
                        .request_ids
                        .iter()
                        .position(|&id| id == half.request_ids[item])
                        .unwrap();
                    assert_eq!(half.prefix_len[item] % 64, 0);
                    assert_ne!(half.seq_lens[item] % 64, 0);
                    assert!(ends[original].unwrap() - half.prefix_len[item] >= 64);
                }
            }
        }
        assert!(admitted > 100 && anchored > 20);
    }

    #[test]
    fn unaligned_continuing_item_does_not_change_delta_chunks() {
        let inputs = crate::inputs::PhaseInputs {
            token_ids: vec![0; 2048],
            q_start: vec![0],
            q_len: vec![2048],
            prefix_len: vec![193],
            anchors: vec![false],
            item_order: vec![0],
            ..crate::inputs::PhaseInputs::default()
        };
        assert_eq!(overlap_split(&inputs, &[None]), None);
    }

    #[test]
    fn overlap_split_aligns_absolute_positions_of_crossing_items() {
        use crate::inputs::PhaseInputs;
        for prefix in [0, 64, 128, 193] {
            for length in [2, 63, 64, 65, 1000, 2048, 2050, 8190] {
                let inputs = PhaseInputs {
                    token_ids: vec![0; length + 3],
                    q_start: vec![0, 3],
                    q_len: vec![3, length as i32],
                    prefix_len: vec![0, prefix],
                    anchors: vec![false, false],
                    item_order: vec![0, 1],
                    ..PhaseInputs::default()
                };
                let at = overlap_split(&inputs, &[None, None]).unwrap() as i32;
                assert!(at > 0 && at < length as i32 + 3);
                if at > 3 {
                    assert_eq!(prefix % crate::contract::PREFILL_GRANULE as i32, 0);
                    assert_eq!((at - 3) % crate::contract::PREFILL_GRANULE as i32, 0);
                }
            }
        }
    }

    #[test]
    fn layers_interleave_linear_and_full_attention_with_wrapped_sublayers() {
        for phase in [Phase::Prefill, Phase::Decode(4)] {
            let stages = stages(phase);
            let at = |wanted| stages.iter().position(|stage| *stage == wanted).unwrap();
            assert_eq!(
                stages
                    .iter()
                    .filter(|stage| matches!(stage, Stage::SparseAttention(_)))
                    .count(),
                12
            );
            assert_eq!(
                stages
                    .iter()
                    .filter(|stage| matches!(stage, Stage::LinearDelta(_)))
                    .count(),
                36
            );
            assert_eq!(
                stages
                    .iter()
                    .filter(|stage| matches!(stage, Stage::Reduce(..)))
                    .count(),
                96
            );
            // PLE follows layer 0 and precedes layer 1's attention wrapper.
            assert!(at(Stage::Reduce(0, Sublayer::Mlp)) < at(Stage::NgramGather));
            assert!(at(Stage::PleNorm) < at(Stage::HcIn(1, Sublayer::Attention)));
            for layer in 0..LAYERS {
                let mix = at(Stage::HcMix(layer, Sublayer::Attention));
                let reduce = at(Stage::Reduce(layer, Sublayer::Attention));
                let core = if is_full(layer) {
                    at(Stage::SparseAttention(layer))
                } else {
                    at(Stage::LinearDelta(layer))
                };
                assert!(mix < core && core < reduce);
                assert!(reduce < at(Stage::Combine(layer)));
                assert!(at(Stage::Combine(layer)) < at(Stage::Reduce(layer, Sublayer::Mlp)));
            }
            assert_eq!(stages.last(), Some(&Stage::GatherLogits));
            assert!(!stages.contains(&Stage::PleCheckpoints));
            assert_eq!(
                stages.contains(&Stage::GatherHeadRows),
                phase == Phase::Prefill
            );
            assert_eq!(
                stages.contains(&Stage::MediaInject),
                phase == Phase::Prefill
            );
        }
    }

    #[test]
    fn verify_keeps_ple_checkpoints_and_the_drafter_runs_one_layer() {
        let verify = stages(Phase::Verify { rows: 8, items: 2 });
        let at = |wanted| verify.iter().position(|stage| *stage == wanted).unwrap();
        assert!(at(Stage::PleConv) < at(Stage::PleCheckpoints));
        assert!(at(Stage::PleCheckpoints) < at(Stage::PleResidual));
        assert_eq!(verify.len(), stages(Phase::Decode(8)).len() + 1);
        let drafter = drafter_stages(true);
        assert!(
            drafter
                .iter()
                .all(|stage| stage.layer().is_none_or(|layer| layer == MTP_LAYER))
        );
        assert!(drafter.contains(&Stage::SparseAttention(MTP_LAYER)));
        assert_eq!(drafter.last(), Some(&Stage::DraftFinal));
        let catch_up = drafter_stages(false);
        assert_eq!(
            catch_up.last(),
            Some(&Stage::Reduce(MTP_LAYER, Sublayer::Mlp))
        );
    }
}
