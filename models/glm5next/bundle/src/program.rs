//! The ordered rank program of one phase, in config order.
//!
//! The residual is `hc_mult` (4) BF16 streams. Every sublayer reads the
//! Sinkhorn mixer's collapse of the streams, normalized, and its rank
//! partials end in one all-reduce whose sum the mixer expands back into the
//! streams. The token mixer is KDA (34 layers) or latent attention over the
//! pooled indexer's selection (11 layers, 3, 7, ..., 43); the MLP is dense
//! (layers 0-2) or MoE. The head collapses the streams by their mean, then
//! the final norm and the vocabulary-split head.

use model_glm5next_contract::contract::{DRAFT_ITEMS, MAX_VERIFY_ROWS, PREFILL_TOKENS, Shape};
use model_glm5next_contract::geometry::{LayerKind, MlpKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Sublayer {
    Attention,
    Mlp,
}

/// Which prepared shapes a phase runs, and how many rows it carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Eager, planned at the live row count, at most `PREFILL_TOKENS`.
    Prefill { rows: u32 },
    /// A graph bucket of this many items, one row each.
    Decode(u32),
    /// A graph bucket of `rows` rows over at most `items` items of one to
    /// `MAX_VERIFY_ROWS` rows each.
    Verify { rows: u32, items: u32 },
}

impl Phase {
    /// The rows the phase's entries are compiled for: the prefill capacity
    /// or the graph bucket.
    pub const fn capacity(self) -> u32 {
        match self {
            Self::Prefill { .. } => PREFILL_TOKENS,
            Self::Decode(rows) | Self::Verify { rows, .. } => rows,
        }
    }

    /// The rows a launch carries.
    pub const fn rows(self) -> u32 {
        match self {
            Self::Prefill { rows } | Self::Decode(rows) | Self::Verify { rows, .. } => rows,
        }
    }

    pub const fn attention(self) -> kernel_core::contract::AttentionPhase {
        use kernel_core::contract::AttentionPhase;
        match self {
            Self::Prefill { .. } => AttentionPhase::Prefill,
            Self::Decode(_) => AttentionPhase::Decode,
            Self::Verify { .. } => AttentionPhase::Verify,
        }
    }

    pub const fn recurrent(self) -> kernel_core::contract::RecurrentPhase {
        use kernel_core::contract::RecurrentPhase;
        match self {
            Self::Prefill { .. } => RecurrentPhase::Prefill,
            Self::Decode(_) => RecurrentPhase::Decode,
            Self::Verify { .. } => RecurrentPhase::Verify {
                max_rows: MAX_VERIFY_ROWS,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    /// Each rank gathers its vocabulary rows (zero elsewhere); the
    /// all-reduce assembles the embedding.
    Embedding,
    EmbeddingReduce,
    /// X = the embedding copied into every stream.
    ExpandStreams,
    /// The sublayer's Sinkhorn mixer: pre, post and comb from the streams,
    /// and the collapsed row the sublayer reads.
    HcMix(u32, Sublayer),
    /// The sublayer's input RMSNorm of the collapsed row.
    InputNorm(u32, Sublayer),
    /// One projection by projection ID.
    Project(u32, &'static str),
    KdaConv(u32),
    KdaDeltaRule(u32),
    KdaGatedNorm(u32),
    /// RMSNorm of the query-down output (`q_a_layernorm`).
    QueryNorm(u32),
    /// RMSNorm of the latent (`kv_a_layernorm`).
    LatentNorm(u32),
    LatentKvWrite(u32),
    /// `q_abs = W_UK^T q_nope` per head.
    QueryAbsorb(u32),
    IndexKeyWrite(u32),
    IndexSelect(u32),
    LatentAttention(u32),
    /// `out = W_UV o` per head.
    ValueUp(u32),
    Route(u32),
    LocalRoutes(u32),
    DenseGateUp(u32),
    ExpertsGateUp(u32),
    SharedGateUp(u32),
    ExpertsDown(u32),
    /// The rank's weighted routes summed per token in F32: in a prefill
    /// into the MLP output, which the shared expert's down projection then
    /// adds; in the decode and verify graphs plus the shared expert's
    /// output, into the partial ([`moe`]).
    RoutedSum(u32),
    /// The all-reduce of the sublayer's rank partials.
    Reduce(u32, Sublayer),
    /// `X'_j = post_j * y + sum_i comb[i][j] * X_i`.
    HcExpand(u32, Sublayer),
    /// The mean of the streams before the final norm.
    HeadCollapse,
    FinalNorm,
    /// Prefill gathers the rows that need logits before the head.
    GatherHeadRows,
    LmHead,
    /// Each rank's F32 logits shard to the rows' owners: decode and verify
    /// rows exchanged to the rank that owns each block of rows, prefill
    /// rows gathered to rank 0.
    GatherLogits,
    /// The MTP drafter's `enorm` of the next tokens' embedding.
    DraftEmbedNorm,
    /// Its `hnorm` of the target's hidden row (the final norm of the
    /// streams' mean).
    DraftHiddenNorm,
    /// The two normalized rows copied side by side into the input
    /// projection's `[rows, 2 * hidden]` input.
    DraftEhInput,
    /// The single residual stream plus the sublayer's reduced output.
    DraftResidual(Sublayer),
    /// `shared_head.norm` of the stream, which the main head reads and the
    /// next chained draft step takes as its hidden row.
    DraftHeadNorm,
    /// A draft step reuses the extend pass's key list, compacted to each
    /// item's last row.
    DraftShareSelection,
    /// Each drafter row's hidden input `h_p`: a slot's retained row, or a
    /// row the previous program left (the target's, or the previous draft
    /// step's), and its token, by the run's source codes
    /// (`crate::drafter::Run::codes`).
    DraftGather,
    /// The draft rows of the head's norm, which alone the head scores.
    DraftHeadRows,
    /// Each row's argmax over the vocabulary: the next draft token.
    DraftArgmax,
}

/// The MTP drafter's passes (`index_share_for_mtp_iteration`): the extend
/// pass over the accepted tokens runs the drafter's indexer; each later
/// draft step reuses its selection, compacted to the last token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftPass {
    Extend,
    Step,
}

/// The draft rows a drafter run of `phase` scores: its head, the gather of
/// its draft rows and its argmax carry one row per item of its graph
/// bucket (a draft step's decode bucket, an extend pass's verify item
/// bucket), since each drafting item drafts once per run; an extend pass in
/// the prefill layout carries every drafter item. Each row's arithmetic is
/// the same at any bucket (the head's uses keep the one-split tiled GEMM's
/// per-row bits; the row kernels read the live count on the device).
pub const fn draft_rows(phase: Phase) -> u32 {
    match phase {
        Phase::Decode(items) | Phase::Verify { items, .. } => items,
        Phase::Prefill { .. } => DRAFT_ITEMS,
    }
}

/// The MTP drafter's stages of one pass: a single-stream layer after the
/// main ones (no mHC), whose latent KV is a twelfth sparse layer of the
/// MTP KV formats. Its input is `eh_proj(cat[enorm(embed(t + 1)),
/// hnorm(h_t)])`; it ends with `shared_head.norm(x)` into the main head and
/// each row's argmax, the next draft token.
pub fn drafter_stages(shape: Shape, pass: DraftPass, phase: Phase) -> Vec<Stage> {
    use Stage::*;
    let layer = shape.geometry.layers();
    let mut stages = vec![
        DraftGather,
        Embedding,
        EmbeddingReduce,
        DraftEmbedNorm,
        DraftHiddenNorm,
        DraftEhInput,
        Project(layer, "mtp.eh"),
        InputNorm(layer, Sublayer::Attention),
    ];
    for stage in sparse(layer) {
        let stage = match stage {
            Project(layer, id) => Project(layer, mtp_id(id)),
            other => other,
        };
        let indexer = matches!(
            stage,
            Project(_, id) if id.starts_with("mtp.latent.indexer.")
        ) || matches!(stage, IndexKeyWrite(_) | IndexSelect(_));
        match (pass, indexer) {
            (DraftPass::Step, true) => {
                if matches!(stage, IndexSelect(_)) {
                    stages.push(DraftShareSelection);
                }
            }
            _ => stages.push(stage),
        }
    }
    stages.extend([
        Reduce(layer, Sublayer::Attention),
        DraftResidual(Sublayer::Attention),
        InputNorm(layer, Sublayer::Mlp),
    ]);
    for stage in moe(layer, phase) {
        stages.push(match stage {
            Project(layer, id) => Project(layer, mtp_id(id)),
            other => other,
        });
    }
    stages.extend([
        Reduce(layer, Sublayer::Mlp),
        DraftResidual(Sublayer::Mlp),
        DraftHeadNorm,
        DraftHeadRows,
        LmHead,
        // Each row's argmax over the rank's logits shards and across ranks:
        // the drafter's logits never reach the sink or the target's logits
        // rows.
        DraftArgmax,
    ]);
    stages
}

/// The MTP layer's declaration of a main-layer projection ID.
fn mtp_id(id: &'static str) -> &'static str {
    match id {
        "latent.q-a" => "mtp.latent.q-a",
        "latent.q-b" => "mtp.latent.q-b",
        "latent.kv-a" => "mtp.latent.kv-a",
        "latent.output" => "mtp.latent.output",
        "latent.indexer.q" => "mtp.latent.indexer.q",
        "latent.indexer.k" => "mtp.latent.indexer.k",
        "latent.indexer.weights" => "mtp.latent.indexer.weights",
        "latent.indexer.pool-gate" => "mtp.latent.indexer.pool-gate",
        "moe.router" => "mtp.moe.router",
        "moe.shared.down" => "mtp.moe.shared.down",
        other => other,
    }
}

impl Stage {
    pub const fn layer(self) -> Option<u32> {
        match self {
            Self::HcMix(layer, _)
            | Self::InputNorm(layer, _)
            | Self::Project(layer, _)
            | Self::KdaConv(layer)
            | Self::KdaDeltaRule(layer)
            | Self::KdaGatedNorm(layer)
            | Self::QueryNorm(layer)
            | Self::LatentNorm(layer)
            | Self::LatentKvWrite(layer)
            | Self::QueryAbsorb(layer)
            | Self::IndexKeyWrite(layer)
            | Self::IndexSelect(layer)
            | Self::LatentAttention(layer)
            | Self::ValueUp(layer)
            | Self::Route(layer)
            | Self::LocalRoutes(layer)
            | Self::DenseGateUp(layer)
            | Self::ExpertsGateUp(layer)
            | Self::SharedGateUp(layer)
            | Self::ExpertsDown(layer)
            | Self::RoutedSum(layer)
            | Self::Reduce(layer, _)
            | Self::HcExpand(layer, _) => Some(layer),
            _ => None,
        }
    }
}

fn kda(layer: u32) -> Vec<Stage> {
    use Stage::*;
    vec![
        Project(layer, "linear.q"),
        Project(layer, "linear.k"),
        Project(layer, "linear.v"),
        KdaConv(layer),
        Project(layer, "linear.forget-a"),
        Project(layer, "linear.forget-b"),
        Project(layer, "linear.beta"),
        KdaDeltaRule(layer),
        Project(layer, "linear.gate-a"),
        Project(layer, "linear.gate-b"),
        KdaGatedNorm(layer),
        Project(layer, "linear.output"),
    ]
}

/// Latent attention: the indexer scores pooled keys from the normalized
/// query-down output and the hidden row, and selects the key list the
/// latent attention reads. The latent is written before the attention of
/// the same step reads it.
fn sparse(layer: u32) -> Vec<Stage> {
    use Stage::*;
    vec![
        Project(layer, "latent.q-a"),
        QueryNorm(layer),
        Project(layer, "latent.q-b"),
        Project(layer, "latent.kv-a"),
        LatentNorm(layer),
        LatentKvWrite(layer),
        QueryAbsorb(layer),
        Project(layer, "latent.indexer.q"),
        Project(layer, "latent.indexer.k"),
        Project(layer, "latent.indexer.weights"),
        Project(layer, "latent.indexer.pool-gate"),
        IndexKeyWrite(layer),
        IndexSelect(layer),
        LatentAttention(layer),
        ValueUp(layer),
        Project(layer, "latent.output"),
    ]
}

fn dense(layer: u32) -> Vec<Stage> {
    vec![
        Stage::DenseGateUp(layer),
        Stage::Project(layer, "dense.down"),
    ]
}

/// A MoE sublayer of `phase`. The shared expert's down projection
/// (`moe.shared.down`'s F32 use profile and residual add) reads the
/// activated rows (`DenseIntermediate`), and the sublayer's F32 partial
/// (`Partial`, which the reduce sums) is the shared expert's output plus
/// the F32 routed sum, one add of the same two values in every phase:
/// - a prefill sums the routes into `MlpOut`, then the shared down
///   projection adds it as its residual into `Partial`;
/// - the decode and verify graphs run the shared gate/up and down
///   projection beside the routed experts (`crate::shared_overlap`): the
///   down projection adds `ZeroRows` into `MlpOut`, and the routed sum
///   adds `MlpOut` into `Partial` last.
fn moe(layer: u32, phase: Phase) -> Vec<Stage> {
    use Stage::*;
    let mut stages = vec![
        Project(layer, "moe.router"),
        Route(layer),
        LocalRoutes(layer),
        ExpertsGateUp(layer),
        ExpertsDown(layer),
    ];
    if matches!(phase, Phase::Prefill { .. }) {
        stages.extend([
            RoutedSum(layer),
            SharedGateUp(layer),
            Project(layer, "moe.shared.down"),
        ]);
    } else {
        stages.extend([
            SharedGateUp(layer),
            Project(layer, "moe.shared.down"),
            RoutedSum(layer),
        ]);
    }
    stages
}

/// Whether `(layer, sublayer)` of `shape`'s target or drafter is a MoE
/// sublayer, whose F32 partials reduce through the rounded all-reduce: the
/// MLP of a MoE layer or of the drafter's layer.
pub fn moe_sublayer(shape: Shape, layer: u32, sublayer: Sublayer) -> bool {
    let g = shape.geometry;
    sublayer == Sublayer::Mlp && (layer >= g.layers() || g.mlp_kind(layer) == MlpKind::Moe)
}

/// A sublayer between its mixer and its expansion.
fn sublayer(layer: u32, sublayer: Sublayer, body: Vec<Stage>) -> Vec<Stage> {
    let mut stages = vec![
        Stage::HcMix(layer, sublayer),
        Stage::InputNorm(layer, sublayer),
    ];
    stages.extend(body);
    stages.push(Stage::Reduce(layer, sublayer));
    stages.push(Stage::HcExpand(layer, sublayer));
    stages
}

/// The target model's stages of `phase`, in order.
pub fn stages(shape: Shape, phase: Phase) -> Vec<Stage> {
    let g = shape.geometry;
    let mut stages = vec![
        Stage::Embedding,
        Stage::EmbeddingReduce,
        Stage::ExpandStreams,
    ];
    for layer in 0..g.layers() {
        let mixer = match g.layer_kind(layer) {
            LayerKind::Linear => kda(layer),
            LayerKind::Sparse => sparse(layer),
        };
        stages.extend(sublayer(layer, Sublayer::Attention, mixer));
        let mlp = match g.mlp_kind(layer) {
            MlpKind::Dense => dense(layer),
            MlpKind::Moe => moe(layer, phase),
        };
        stages.extend(sublayer(layer, Sublayer::Mlp, mlp));
    }
    stages.push(Stage::HeadCollapse);
    stages.push(Stage::FinalNorm);
    if matches!(phase, Phase::Prefill { .. }) {
        stages.push(Stage::GatherHeadRows);
    }
    stages.push(Stage::LmHead);
    stages.push(Stage::GatherLogits);
    stages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_follow_the_config_order() {
        let shape = Shape::TP4;
        let stages = stages(shape, Phase::Decode(8));
        let count = |f: &dyn Fn(&Stage) -> bool| stages.iter().filter(|s| f(s)).count();
        assert_eq!(count(&|s| matches!(s, Stage::KdaDeltaRule(_))), 34);
        assert_eq!(count(&|s| matches!(s, Stage::LatentAttention(_))), 11);
        assert_eq!(count(&|s| matches!(s, Stage::DenseGateUp(_))), 3);
        assert_eq!(count(&|s| matches!(s, Stage::ExpertsGateUp(_))), 42);
        // A mixer and an expansion around each of the 90 sublayers.
        assert_eq!(count(&|s| matches!(s, Stage::HcMix(..))), 90);
        assert_eq!(count(&|s| matches!(s, Stage::HcExpand(..))), 90);
        let sparse: Vec<u32> = stages
            .iter()
            .filter_map(|s| match s {
                Stage::LatentAttention(layer) => Some(*layer),
                _ => None,
            })
            .collect();
        assert_eq!(sparse, (0..11).map(|i| 4 * i + 3).collect::<Vec<_>>());
        // Layers appear in order.
        let layers: Vec<u32> = stages.iter().filter_map(|s| s.layer()).collect();
        assert!(layers.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(stages.last(), Some(&Stage::GatherLogits));
        assert!(!stages.contains(&Stage::GatherHeadRows));
        assert!(super::stages(shape, Phase::Prefill { rows: 64 }).contains(&Stage::GatherHeadRows));
    }

    /// Every projection ID the program runs is declared in `projections.json`
    /// with a stem holding the stage's layer, and every main-layer
    /// declaration runs, by ID or through the stage that owns it.
    #[test]
    fn projection_stages_name_declared_projections() {
        use model_glm5next_contract::projections;
        let shape = Shape::TP4;
        let d = projections::declaration(shape.ranks()).unwrap();
        let g = projections::geometry(shape.ranks()).unwrap();
        let layers = |id: &str| -> Option<Vec<u64>> {
            let p = d.projections.iter().find(|p| p.id == id)?;
            Some(match p.stem() {
                Some(stem) => {
                    let stem = d.layer_stems.iter().find(|s| s.id == stem).unwrap();
                    stem.indices.resolve(&g).unwrap()
                }
                None => Vec::new(),
            })
        };
        // The pooling gate is a side tensor until the shared importer reads
        // a projection source without a `.weight` suffix.
        let pending = |id: &str| d.side_tensors.iter().any(|s| s.id == id);
        let mut run = std::collections::BTreeSet::new();
        let mtp = |layer: u32, id: &'static str, drafted: &'static str| {
            if layer >= shape.geometry.layers() {
                drafted
            } else {
                id
            }
        };
        let programs = [
            Phase::Prefill { rows: 64 },
            Phase::Decode(8),
            Phase::Verify { rows: 32, items: 8 },
        ]
        .map(|phase| stages(shape, phase))
        .into_iter()
        .chain(
            [DraftPass::Extend, DraftPass::Step]
                .map(|pass| drafter_stages(shape, pass, Phase::Decode(8))),
        );
        for program in programs {
            for stage in program {
                let id = match stage {
                    Stage::Project(layer, id) => {
                        match layers(id) {
                            Some(layers) => assert!(layers.contains(&layer.into()), "{id} {layer}"),
                            None => assert!(pending(id), "{id} is not declared"),
                        }
                        id
                    }
                    Stage::Embedding => "embedding",
                    Stage::LmHead => "head",
                    Stage::DenseGateUp(_) => "dense.gate-up",
                    Stage::SharedGateUp(layer) => {
                        mtp(layer, "moe.shared.gate-up", "mtp.moe.shared.gate-up")
                    }
                    Stage::ExpertsGateUp(layer) => {
                        mtp(layer, "moe.experts.gate-up", "mtp.moe.experts.gate-up")
                    }
                    Stage::ExpertsDown(layer) => {
                        mtp(layer, "moe.experts.down", "mtp.moe.experts.down")
                    }
                    _ => continue,
                };
                assert!(layers(id).is_some() || pending(id), "{id}");
                run.insert(id.to_owned());
            }
        }
        // Every declaration runs, the drafter's in its passes.
        let declared: std::collections::BTreeSet<_> = d
            .projections
            .iter()
            .map(|p| p.id.clone())
            .chain(["latent.indexer.pool-gate", "mtp.latent.indexer.pool-gate"].map(str::to_owned))
            .collect();
        assert_eq!(run, declared);
    }

    /// The drafter's passes run no mixer stage (the MTP layer adds its
    /// sublayers' outputs to one stream), in every phase kind.
    #[test]
    fn drafter_passes_run_no_mixer_stage() {
        let mixer = |stage: &Stage| {
            matches!(
                stage,
                Stage::HcMix(..) | Stage::HcExpand(..) | Stage::HeadCollapse
            )
        };
        for pass in [DraftPass::Step, DraftPass::Extend] {
            for phase in [
                Phase::Prefill { rows: 1 },
                Phase::Decode(1),
                Phase::Verify { items: 1, rows: 1 },
            ] {
                let stages = drafter_stages(Shape::TP4, pass, phase);
                assert!(!stages.is_empty(), "{pass:?} {phase:?}");
                assert!(!stages.iter().any(mixer), "{pass:?} {phase:?}");
            }
        }
    }
}
