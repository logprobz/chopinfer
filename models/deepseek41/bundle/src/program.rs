//! The ordered rank program of one phase, in layer order.
//!
//! The residual is `hc_mult` (4) BF16 streams. Every sublayer's Sinkhorn
//! mixer computes its `pre`, `post` and `comb` from the streams, but the
//! sublayer reads the streams collapsed by the previous mixer's `pre`
//! (lagged, `model.py:970-997`); the first attention collapses to stream 0.
//! Its rank partials end in one F32 all-reduce whose sum the mixer expands
//! back into the streams in F32.
//!
//! Attention is latent MQA over two key sources (`model.py:766-791`): the
//! layer's window ring and, at compress ratio r > 0, the compressed
//! positions its selection keeps. The CSA2 schedule (`Geometry::layer_role`)
//! says which layer compresses and which selects: a `Full` layer runs its
//! compressor, writes the index keys and runs its indexer; a `Reindex`
//! layer runs its own indexer over the latest source's index keys, inside
//! the candidate source's blocks; a `Reuse` layer reads the latest
//! source's compressed KV and selection. Every MLP is MoE with a shared
//! expert. Engram rewrites the streams before its layers; each
//! DSpark target layer's attention input is retained for the drafter. The
//! head collapses the streams by the last FFN mixer's `pre`, then the final
//! norm and the vocabulary-split head. Every prompt row runs the whole
//! decoder (D13).

use model_deepseek41_contract::geometry::{Geometry, LayerRole};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Sublayer {
    Attention,
    Mlp,
}

/// Which prepared shapes a phase runs, and how many rows it carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Eager, planned at the live row count.
    Prefill { rows: u32 },
    /// A graph bucket of this many items, one row each.
    Decode(u32),
    /// A graph bucket of `rows` rows over at most `items` items of one to
    /// `MAX_VERIFY_ROWS` rows each.
    Verify { rows: u32, items: u32 },
}

impl Phase {
    /// The rows a launch carries.
    pub const fn rows(self) -> u32 {
        match self {
            Self::Prefill { rows } | Self::Decode(rows) | Self::Verify { rows, .. } => rows,
        }
    }

    pub const fn prefill(self) -> bool {
        matches!(self, Self::Prefill { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    /// Each rank gathers its vocabulary rows (zero elsewhere); the
    /// all-reduce assembles the embedding.
    Embedding,
    EmbeddingReduce,
    /// The embedding copied into every stream.
    ExpandStreams,
    /// Each RoPE table's cos and sin rows at the step's positions, which
    /// the tail rotations read.
    RopeRows,
    /// Each row's window key list: its visible ring cells, oldest first.
    /// Every layer's ring is addressed alike.
    WindowKeys,
    /// The Engram rows the host hashed and gathered for the step's tokens,
    /// staged to the device and decoded to BF16 (`model.py:312-326`).
    EngramRows(u32),
    /// The v2 Engram gate reads the projection's F32 keys and value, adds
    /// into the BF16 streams in F32 and rounds once to BF16. Image rows
    /// use the closed-gate expression (`model.py:350-366`).
    EngramGate(u32),
    /// The mean of the streams at the layer's attention input, retained for
    /// the DSpark drafter (`model.py:1265-1266`).
    DsparkFeature(u32),
    /// The sublayer's Sinkhorn mixer: its `pre`, `post` and `comb` from the
    /// streams, and the streams collapsed by the previous mixer's `pre`.
    HcMix(u32, Sublayer),
    /// The sublayer's input RMSNorm of the collapsed row.
    InputNorm(u32, Sublayer),
    /// The normed row quantized to E4M3 per 32 with a ceil UE8M0 scale,
    /// the activation every FP8 and FP4 projection of the sublayer reads
    /// (D10).
    QuantizeInput(u32, Sublayer),
    /// One projection by projection ID.
    Project(u32, &'static str),
    /// `q_norm` of the query-down output into F32: the attention and the
    /// indexer read it.
    QueryNorm(u32),
    /// The queries' RoPE tail in F32, rounded once into the BF16 attention
    /// query.
    QueryRope(u32),
    /// The step rows' window keys: `kv_norm`, the RoPE tail at each row's
    /// position, then the E4M3 per-32 quantization. A decode or verify
    /// step writes them into the layer's ring, which has room for its
    /// rows, before its queries read it; a prefill stages them after the
    /// window before its chunk.
    WindowKv(u32),
    /// The KV source's compressor: each completed group's pooled latent
    /// (F32 softmax over the group at ratio 2) and its RMSNorm; an open
    /// group's rows wait in the compressor's tail.
    Compress(u32),
    /// The index keys of the completed groups: `k_norm`, the RoPE tail at
    /// the group's position, MXFP4 quant-dequant, into the pages.
    IndexKeyWrite(u32),
    /// The compressed latents of the completed groups: the RoPE tail at the
    /// group's position, E2M1 per-16 quant-dequant, into the pages.
    CompressedKvWrite(u32),
    /// The index query: the RoPE tail at each row's position, MXFP4
    /// quant-dequant.
    IndexQuery(u32),
    /// The index source's top-k compressed positions, in its candidate
    /// blocks when it reindexes. The candidate source also publishes each
    /// query's best candidate blocks of compressed positions, its newest
    /// block pinned (`model.py:584-611`), from the same scores.
    IndexSelect(u32),
    /// Latent attention over the window and the selected compressed
    /// positions, with the sink; the output's RoPE tail rotated back.
    LatentAttention(u32),
    /// A prefill's window keys written into the layer's ring after its
    /// attention: each sequence's last ring cells' positions.
    WindowRingWrite(u32),
    /// The router's scores, the per-row bias choice, the top-k weights and
    /// the rank's local routes, padded to the experts' route tile.
    Route(u32),
    ExpertsGateUp(u32),
    ExpertsDown(u32),
    /// The shared expert's gate and up projections (column split).
    SharedGateUp(u32),
    /// Its clamp, `silu(g) * u` in F32, rounded to BF16 and quantized to
    /// E4M3 per 32 for the down projection.
    SharedActivation(u32),
    /// Its down projection (row split) into F32 partials.
    SharedDown(u32),
    /// The rank's weighted routes summed per row in F32, plus the shared
    /// expert's partial.
    RoutedSum(u32),
    /// The all-reduce of the sublayer's F32 rank partials.
    Reduce(u32, Sublayer),
    /// `X'_j = post_j * y + sum_i comb[i][j] * X_i` in F32, stored BF16.
    HcExpand(u32, Sublayer),
    /// The streams weighted by the last FFN mixer's `pre`.
    HeadCollapse,
    FinalNorm,
    /// Prefill gathers the rows that need logits before the head.
    GatherHeadRows,
    LmHead,
    /// Each rank's F32 logits shard to the rows' owners.
    GatherLogits,
}

impl Stage {
    pub const fn layer(self) -> Option<u32> {
        match self {
            Self::EngramRows(layer)
            | Self::EngramGate(layer)
            | Self::DsparkFeature(layer)
            | Self::HcMix(layer, _)
            | Self::InputNorm(layer, _)
            | Self::QuantizeInput(layer, _)
            | Self::Project(layer, _)
            | Self::QueryNorm(layer)
            | Self::QueryRope(layer)
            | Self::WindowKv(layer)
            | Self::Compress(layer)
            | Self::IndexKeyWrite(layer)
            | Self::CompressedKvWrite(layer)
            | Self::IndexQuery(layer)
            | Self::IndexSelect(layer)
            | Self::LatentAttention(layer)
            | Self::WindowRingWrite(layer)
            | Self::Route(layer)
            | Self::ExpertsGateUp(layer)
            | Self::ExpertsDown(layer)
            | Self::SharedGateUp(layer)
            | Self::SharedActivation(layer)
            | Self::SharedDown(layer)
            | Self::RoutedSum(layer)
            | Self::Reduce(layer, _)
            | Self::HcExpand(layer, _) => Some(layer),
            _ => None,
        }
    }
}

/// A sublayer between its mixer and its expansion.
fn sublayer(stages: &mut Vec<Stage>, layer: u32, sublayer: Sublayer, body: Vec<Stage>) {
    stages.extend([
        Stage::HcMix(layer, sublayer),
        Stage::InputNorm(layer, sublayer),
        Stage::QuantizeInput(layer, sublayer),
    ]);
    stages.extend(body);
    stages.extend([
        Stage::Reduce(layer, sublayer),
        Stage::HcExpand(layer, sublayer),
    ]);
}

/// The KV source whose compressed KV and index keys `layer` reads: the
/// latest at or before it (`SharedAttentionRuntime`, `model.py:1169-1183`).
pub fn kv_source(g: &Geometry, layer: u32) -> Option<u32> {
    model_deepseek41_contract::indexer::sources(g, layer).map(|sources| sources.keys)
}

/// The index source whose selection `layer` attends over: the latest at or
/// before it.
pub fn index_source(g: &Geometry, layer: u32) -> Option<u32> {
    model_deepseek41_contract::indexer::sources(g, layer).map(|sources| sources.selection)
}

/// The attention of `layer` in `phase`. The indexer reads the latent
/// before RoPE, so index keys are written before the compressed latents
/// (`model.py:740-764`); both precede the selection that scores them and
/// the attention that reads them.
pub fn attention(g: &Geometry, layer: u32, phase: Phase) -> Vec<Stage> {
    use Stage::*;
    let l = layer;
    let mut stages = vec![
        Project(l, "attention.q-a"),
        QueryNorm(l),
        Project(l, "attention.q-b"),
        QueryRope(l),
        Project(l, "attention.kv"),
        WindowKv(l),
    ];
    let indexer = [
        Project(l, "indexer.q"),
        IndexQuery(l),
        Project(l, "indexer.weights"),
    ];
    match g.layer_role(l) {
        LayerRole::Full => {
            stages.push(Project(l, "compressor.kv"));
            if g.compress_ratio(l) > 1 {
                stages.push(Project(l, "compressor.gate"));
            }
            stages.extend([
                Compress(l),
                Project(l, "indexer.k"),
                IndexKeyWrite(l),
                CompressedKvWrite(l),
            ]);
            stages.extend(indexer);
            stages.push(IndexSelect(l));
        }
        LayerRole::Reindex => {
            stages.extend(indexer);
            stages.push(IndexSelect(l));
        }
        LayerRole::Reuse | LayerRole::Window => {}
    }
    stages.push(LatentAttention(l));
    if phase.prefill() {
        stages.push(WindowRingWrite(l));
    }
    stages.extend([Project(l, "attention.o-a"), Project(l, "attention.o-b")]);
    stages
}

/// The MoE of `layer`: routed experts and the shared expert, whose F32
/// partial the routed sum adds before the all-reduce.
pub fn moe(layer: u32) -> Vec<Stage> {
    use Stage::*;
    vec![
        Project(layer, "moe.router"),
        Route(layer),
        ExpertsGateUp(layer),
        ExpertsDown(layer),
        SharedGateUp(layer),
        SharedActivation(layer),
        SharedDown(layer),
        RoutedSum(layer),
    ]
}

/// The target model's stages of `phase`, in order.
pub fn stages(g: &Geometry, phase: Phase) -> Vec<Stage> {
    let mut stages = vec![
        Stage::Embedding,
        Stage::EmbeddingReduce,
        Stage::ExpandStreams,
        Stage::RopeRows,
        Stage::WindowKeys,
    ];
    for l in 0..g.layers() {
        if g.is_engram_layer(l) {
            stages.extend([
                Stage::EngramRows(l),
                Stage::Project(l, "engram.kv"),
                Stage::EngramGate(l),
            ]);
        }
        if g.shape.dspark.targets.contains(&l) {
            stages.push(Stage::DsparkFeature(l));
        }
        sublayer(&mut stages, l, Sublayer::Attention, attention(g, l, phase));
        sublayer(&mut stages, l, Sublayer::Mlp, moe(l));
    }
    stages.extend([Stage::HeadCollapse, Stage::FinalNorm]);
    if phase.prefill() {
        stages.push(Stage::GatherHeadRows);
    }
    stages.extend([Stage::LmHead, Stage::GatherLogits]);
    stages
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    fn layers_of(stages: &[Stage], pick: impl Fn(&Stage) -> bool) -> Vec<u32> {
        stages
            .iter()
            .filter(|s| pick(s))
            .filter_map(|s| s.layer())
            .collect()
    }

    #[test]
    fn engram_projection_feeds_the_gate_without_an_intermediate_round() {
        for phase in [
            Phase::Prefill { rows: 8 },
            Phase::Decode(8),
            Phase::Verify { rows: 8, items: 2 },
        ] {
            let stages = stages(&FLASH, phase);
            for layer in FLASH.shape.engram.layers {
                let at = stages
                    .iter()
                    .position(|s| *s == Stage::EngramGate(*layer))
                    .unwrap();
                assert_eq!(
                    &stages[at - 2..at],
                    &[
                        Stage::EngramRows(*layer),
                        Stage::Project(*layer, "engram.kv")
                    ]
                );
            }
        }
    }

    #[test]
    fn the_flash_program_follows_the_csa2_schedule() {
        let stages = stages(&FLASH, Phase::Decode(8));
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::LatentAttention(_))),
            (0..40).collect::<Vec<_>>()
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::Compress(_))),
            [2, 8, 14, 20]
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::IndexSelect(_))),
            [2, 8, 14, 20, 24, 28, 32, 36]
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(
                s,
                Stage::Project(_, "compressor.gate")
            )),
            [2, 8, 14]
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::EngramGate(_))),
            [1, 14]
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::DsparkFeature(_))),
            [37, 38, 39]
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::HcMix(..))).len(),
            80
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::Reduce(..))).len(),
            80
        );
        assert_eq!(
            layers_of(&stages, |s| matches!(s, Stage::ExpertsGateUp(_))).len(),
            40
        );
        // Layers appear in order, Engram and the retained feature before
        // the layer's first mixer.
        let layers: Vec<u32> = stages.iter().filter_map(|s| s.layer()).collect();
        assert!(layers.windows(2).all(|pair| pair[0] <= pair[1]));
        let at = |stage: Stage| stages.iter().position(|s| *s == stage).unwrap();
        assert!(at(Stage::EngramGate(14)) < at(Stage::HcMix(14, Sublayer::Attention)));
        assert!(at(Stage::DsparkFeature(37)) < at(Stage::HcMix(37, Sublayer::Attention)));
        // The index keys are written before the compressed latents, both
        // before the selection and the attention that read them.
        assert!(at(Stage::IndexKeyWrite(20)) < at(Stage::CompressedKvWrite(20)));
        assert!(at(Stage::CompressedKvWrite(20)) < at(Stage::IndexSelect(20)));
        assert!(at(Stage::IndexSelect(20)) < at(Stage::LatentAttention(20)));
        // A decode step writes its window keys into the ring before its
        // queries read them; a prefill writes the ring after its attention.
        assert!(at(Stage::WindowKv(20)) < at(Stage::LatentAttention(20)));
        assert!(!stages.contains(&Stage::WindowRingWrite(20)));
        assert_eq!(stages.last(), Some(&Stage::GatherLogits));
        assert!(!stages.contains(&Stage::GatherHeadRows));
        let prefill = super::stages(&FLASH, Phase::Prefill { rows: 64 });
        assert!(prefill.contains(&Stage::GatherHeadRows));
        let at = |stage: Stage| prefill.iter().position(|s| *s == stage).unwrap();
        assert!(at(Stage::LatentAttention(20)) < at(Stage::WindowRingWrite(20)));
    }

    #[test]
    fn reuse_layers_read_the_latest_sources() {
        let indexer = |stages: &[Stage]| {
            stages.iter().any(|s| {
                matches!(
                    s,
                    Stage::IndexQuery(_) | Stage::IndexSelect(_) | Stage::Project(_, "indexer.q")
                )
            })
        };
        let compressor = |stages: &[Stage]| {
            stages
                .iter()
                .any(|s| matches!(s, Stage::Compress(_) | Stage::Project(_, "compressor.kv")))
        };
        for layer in [0, 1, 5, 23, 39] {
            let stages = attention(&FLASH, layer, Phase::Decode(1));
            assert!(!indexer(&stages), "{layer}");
            assert!(!compressor(&stages), "{layer}");
        }
        assert!(indexer(&attention(&FLASH, 24, Phase::Decode(1))));
        assert!(!compressor(&attention(&FLASH, 24, Phase::Decode(1))));
        assert_eq!(
            [1, 2, 7, 19, 20, 39].map(|l| kv_source(&FLASH, l)),
            [None, Some(2), Some(2), Some(14), Some(20), Some(20)]
        );
        assert_eq!(
            [5, 23, 27, 39].map(|l| index_source(&FLASH, l)),
            [Some(2), Some(20), Some(24), Some(36)]
        );
    }
}
