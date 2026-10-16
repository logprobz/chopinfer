//! Launch records: each stage of a phase as the contract it launches and
//! what prepares its launches: the projection API by projection ID
//! ([`Planner::Projection`]), the ops an instance prepares on its kernel
//! backend ([`Planner::Backend`], `crate::ops`) or GLM's own ops
//! ([`Planner::Own`], `model_glm5next_contract::own_ops`). A record whose
//! stage does not lower (a projection without the precision component)
//! journals against its contract.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use kernel_core::collective::ShardedRows;
use kernel_core::contract::{
    ActivationFormat, CommQuery, DType, Epilogues, GateActivation, GemmQuery, GroupedGemm,
    MoeQuery, NumericalContract, RecurrentQuery, Refusal, RmsNormKind, RoutedCapacity, RouterKind,
    RowQuery, ScaleMode, SwigluColumns, Validate, WeightFormat,
};
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::latent_attention::{
    HeadBatchedGemmQuery, HeadWeightLayout, LatentAttentionOutput, LatentAttentionQuery,
    LatentKvFormat,
};
use kernel_core::per_channel_decay_delta_rule::PerChannelDecayDeltaRuleQuery;
use kernel_core::pooled_indexer::{
    IndexerGranularity, IndexerKeyFormat, IndexerSelectionMode, PooledIndexerQuery,
};
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, SinkhornHyperConnectionQuery, SinkhornMix, SinkhornOrder,
    StreamHeadMode,
};
use model_glm5next_contract::catalog::DType as StoredDType;
use model_glm5next_contract::contract::{PAGE_TOKENS, Shape};
use model_glm5next_contract::quantization::{FP8_BLOCK128, Layout};
use model_glm5next_contract::views::{Binding, checkpoint_rank_views};

use crate::lower::formats::Class;
use crate::program::{Phase, Stage, stages};

/// The indexer key LayerNorm's eps.
const INDEXER_KEY_NORM_EPS: f32 = 1e-6;

/// The contract a launch runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Gemm(GemmQuery),
    GatedMlp(GatedMlpProjectionQuery),
    HeadBatched(HeadBatchedGemmQuery),
    Latent(LatentAttentionQuery),
    Indexer(PooledIndexerQuery),
    Sinkhorn(SinkhornHyperConnectionQuery),
    Kda(PerChannelDecayDeltaRuleQuery),
    Recurrent(RecurrentQuery),
    Row(RowQuery),
    Moe(MoeQuery),
    Comm(CommQuery),
    /// A move of rows whose columns are split across the ranks: bits
    /// only.
    ShardedRows(ShardedRows),
    /// One of GLM's own ops (`model_glm5next_contract::own_ops`), by name:
    /// its contract is `glm5next.{name}.v1`.
    Own(&'static str),
}

impl Op {
    /// The op family the journal names.
    pub const fn family(&self) -> &'static str {
        match self {
            Self::Gemm(_) => "gemm",
            Self::GatedMlp(_) => "gated-mlp",
            Self::HeadBatched(_) => "head-batched-gemm",
            Self::Latent(_) => "latent-attention",
            Self::Indexer(_) => "pooled-indexer",
            Self::Sinkhorn(_) => "sinkhorn-hc",
            Self::Kda(_) => "per-channel-decay-delta-rule",
            Self::Recurrent(_) => "recurrent",
            Self::Row(_) => "row",
            Self::Moe(_) => "moe",
            Self::Comm(_) => "comm",
            Self::ShardedRows(_) => "sharded-rows",
            Self::Own(_) => "glm5next-row",
        }
    }

    /// The numerical contract ID, validating the query.
    pub fn contract(&self) -> Result<String, Refusal> {
        let id = match self {
            Self::Gemm(q) => q.numerical_contract_id(),
            Self::GatedMlp(q) => q.numerical_contract_id(),
            Self::HeadBatched(q) => q.numerical_contract_id(),
            Self::Latent(q) => q.numerical_contract_id(),
            Self::Indexer(q) => q.numerical_contract_id(),
            Self::Sinkhorn(q) => q.numerical_contract_id(),
            Self::Kda(q) => q.numerical_contract_id(),
            Self::Recurrent(q) => q.numerical_contract_id(),
            Self::Row(q) => q.numerical_contract_id(),
            Self::Moe(q) => q.numerical_contract_id(),
            Self::Comm(q) => q.numerical_contract_id(),
            Self::ShardedRows(rows) => {
                rows.validate()?;
                return Ok("copy".into());
            }
            Self::Own(name) => return Ok(format!("glm5next.{name}.v1")),
        }?;
        // Each family's ID validates its query; the row and comm IDs name
        // only the algorithm, so their shapes are validated here.
        match self {
            Self::Row(q) => q.validate()?,
            Self::Comm(q) => q.validate()?,
            _ => {}
        }
        Ok(id.as_str().to_owned())
    }
}

/// What prepares a record's launches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Planner {
    /// The projection API, by projection ID (`projections.json`).
    Projection(&'static str),
    /// Ops the instance prepares at create on its kernel backend
    /// (`crate::ops`).
    Backend,
    /// GLM's own ops, prepared at create on the target's own kernels
    /// (`crate::ops::own`).
    Own,
}

/// One launch of one rank's program.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub stage: Stage,
    pub op: Op,
    pub planner: Planner,
}

/// One projection's per-rank GEMM shape: its output columns (gate and up
/// together), K and stored weight dtype.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Projected {
    n: u32,
    k: u32,
    dtype: StoredDType,
}

/// Per-rank projection shapes by (layer, projection ID); the head has none.
type Shapes = BTreeMap<(Option<u32>, &'static str), Projected>;

type Computed = Result<Arc<Shapes>, String>;

/// Every projection shape map [`shared_projections`] computed, with its
/// key, in first-request order.
static PROJECTIONS: Mutex<Vec<(Shape, Layout, Computed)>> = Mutex::new(Vec::new());

/// [`projections`], computed once per process for each (shape, checkpoint
/// layout) and shared read-only after: the shapes are a pure function of
/// the two, and computing them walks every rank view of the checkpoint,
/// 20 to 30 ms on every program that records lower (each eager prefill
/// chunk, each graph capture).
fn shared_projections(shape: Shape, layout: &Layout) -> Computed {
    // A panic while computing leaves no entry, so the list stays usable.
    let mut computed = PROJECTIONS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let found = computed
        .iter()
        .find(|(of, stored_as, _)| *of == shape && stored_as == layout);
    if let Some((.., shapes)) = found {
        return shapes.clone();
    }
    let shapes = projections(shape, layout).map(Arc::new);
    computed.push((shape, *layout, shapes.clone()));
    shapes
}

/// Per-rank projection shapes by (layer, projection ID), read from the rank
/// views of an export stored in `layout` so the program and the weight plan
/// cannot disagree, at their logical K (U8 codes hold two E2M1 values per
/// byte). The head has no layer.
fn projections(shape: Shape, layout: &Layout) -> Result<Shapes, String> {
    let mut shapes = BTreeMap::new();
    for view in checkpoint_rank_views(shape, layout, 0)? {
        let Binding::Projection(id) = view.binding else {
            continue;
        };
        // Weights only (the pooling gate has no `.weight` suffix), never a
        // scale companion; routed experts plan through the grouped engine,
        // not by ID.
        if SCALE_COMPANIONS
            .iter()
            .any(|suffix| view.name.ends_with(suffix))
            || view.shape.len() != 2
            || id.contains("moe.experts.")
        {
            continue;
        }
        let layer = view
            .name
            .split_once("layers.")
            .and_then(|(_, rest)| rest.split('.').next())
            .and_then(|layer| layer.parse().ok());
        let packed = if view.dtype == StoredDType::U8 { 2 } else { 1 };
        let (n, k) = (view.shape[0] as u32, packed * view.shape[1] as u32);
        let entry = shapes.entry((layer, id)).or_insert(Projected {
            n: 0,
            k,
            dtype: view.dtype,
        });
        // Gate and up share a projection ID.
        entry.n += n;
    }
    Ok(shapes)
}

/// The scale companions of a projection's weight, which are not weights.
const SCALE_COMPANIONS: [&str; 4] = [
    ".weight_scale_inv",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
];

/// The format of a projection's stored weight: block FP8, NVFP4 (U8 E2M1
/// codes) or BF16.
fn weight(dtype: StoredDType) -> WeightFormat {
    match dtype {
        StoredDType::F8E4M3 => WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        StoredDType::U8 => WeightFormat::Nvfp4,
        _ => WeightFormat::Bf16,
    }
}

/// The records of every rank's program (they differ only in the rank
/// scalar of local routes and the head's vocabulary rows) for `phase` over
/// latent KV in `kv`, at context-parallel size 1.
pub fn records(shape: Shape, kv: LatentKvFormat, phase: Phase) -> Result<Vec<Record>, String> {
    records_for(shape, &FP8_BLOCK128, kv, phase)
}

/// [`records`] of an export stored in `layout`.
pub fn records_for(
    shape: Shape,
    layout: &'static Layout,
    kv: LatentKvFormat,
    phase: Phase,
) -> Result<Vec<Record>, String> {
    records_of(shape, layout, kv, phase, stages(shape, phase))
}

/// The records of the MTP drafter's `pass` at `phase`'s rows.
pub fn drafter_records(
    shape: Shape,
    kv: LatentKvFormat,
    phase: Phase,
    pass: crate::program::DraftPass,
) -> Result<Vec<Record>, String> {
    drafter_records_for(shape, &FP8_BLOCK128, kv, phase, pass)
}

/// [`drafter_records`] of an export stored in `layout`.
pub fn drafter_records_for(
    shape: Shape,
    layout: &'static Layout,
    kv: LatentKvFormat,
    phase: Phase,
    pass: crate::program::DraftPass,
) -> Result<Vec<Record>, String> {
    records_of(
        shape,
        layout,
        kv,
        phase,
        crate::program::drafter_stages(shape, pass, phase),
    )
}

fn records_of(
    shape: Shape,
    layout: &'static Layout,
    kv: LatentKvFormat,
    phase: Phase,
    stages: Vec<Stage>,
) -> Result<Vec<Record>, String> {
    let g = shape.geometry;
    let p = shape.profile;
    let capacity = phase.capacity();
    let hidden = p.hidden;
    let eps = g.eps();
    let projected = shared_projections(shape, layout)?;
    let norm = |width| {
        Op::Row(RowQuery::RmsNorm {
            width,
            eps,
            kind: RmsNormKind::HfBf16Intermediate,
            fused_residual: false,
            output_quantization: None,
        })
    };
    let mix = SinkhornMix {
        streams: p.hc_streams,
        width: hidden,
        iterations: g.shape.hc.sinkhorn_iterations,
        eps_bits: (g.shape.hc.eps as f32).to_bits(),
        norm_eps_bits: eps.to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::PerSublayer,
        fn_dtype: DType::Bf16,
    };
    let clamp = SwigluClamp::symmetric(g.shape.swiglu_limit as f32);
    // Each chained projection's weights in the checkpoint's layout.
    let chained = |layer, class| crate::lower::formats::weight_format(layout, shape, layer, class);
    let gated = |n, k, projection, grouped: bool, weight| GatedMlpProjectionQuery {
        gemm: GemmQuery {
            m_capacity: capacity,
            n,
            k,
            weight,
            activation: ActivationFormat::Bf16,
            // The experts' weighted down rows combine into the F32 routed
            // sum.
            output: if grouped && projection == GatedMlpProjection::Down {
                DType::F32
            } else {
                DType::Bf16
            },
            epilogues: Epilogues {
                swiglu: (projection == GatedMlpProjection::GateUp)
                    .then_some(SwigluColumns::ConcatenatedHalves),
                weighted_moe_combine: grouped && projection == GatedMlpProjection::Down,
                ..Epilogues::default()
            },
            grouped: grouped.then_some(GroupedGemm {
                experts: p.local_experts,
                top_k: p.top_k,
                capacity: RoutedCapacity::TotalRows(capacity * p.top_k),
            }),
        },
        projection,
        swiglu_clamp: (projection == GatedMlpProjection::GateUp).then_some(clamp),
        route_weight: grouped.then_some(RouteWeightPoint::AfterDown),
    };
    // Each rank's F32 partial rounded once to BF16, summed in rank order
    // and rounded once (`ops::reduce`).
    let reduce = Op::Comm(CommQuery::AllReduce {
        ranks: shape.ranks(),
        elements: u64::from(capacity) * u64::from(hidden),
        dtype: DType::Bf16,
        fused_residual: false,
        fused_rmsnorm: None,
    });
    // A MoE sublayer's F32 partials summed in F32 in rank order and rounded
    // once (`ops::reduce`).
    let rounded = Op::Comm(CommQuery::AllReduceRounded {
        ranks: shape.ranks(),
        elements: u64::from(capacity) * u64::from(hidden),
    });
    let select = PooledIndexerQuery::Select {
        heads: p.indexer_heads,
        head_dim: p.indexer_dim,
        group: p.index_pool,
        max_top_k: p.index_topk_max,
        always_tail: g.semantics.index_pool_tail,
        granularity: IndexerGranularity::Pool,
        key_format: IndexerKeyFormat::Bf16,
        mode: IndexerSelectionMode::Global,
        page_size: PAGE_TOKENS as u32,
        capacity,
    };
    let max_keys = select
        .list_width()
        .ok_or("the indexer key list width is not representable")?;
    let record = |stage: Stage| -> Result<Record, String> {
        use Planner::{Backend, Own};
        let (op, planner) = match stage {
            Stage::Embedding => (
                Op::Row(RowQuery::EmbeddingGather {
                    vocab: p.vocab_rows,
                    width: hidden,
                }),
                Backend,
            ),
            Stage::Reduce(layer, sublayer)
                if crate::program::moe_sublayer(shape, layer, sublayer) =>
            {
                (rounded, Backend)
            }
            Stage::EmbeddingReduce | Stage::Reduce(..) => (reduce, Backend),
            Stage::ExpandStreams => (Op::Own("expand-streams"), Own),
            Stage::GatherHeadRows => (Op::Own("gather-rows"), Own),
            Stage::HcMix(..) => (
                Op::Sinkhorn(SinkhornHyperConnectionQuery::MixCollapse { mix, capacity }),
                Backend,
            ),
            Stage::HcExpand(..) => (
                Op::Sinkhorn(SinkhornHyperConnectionQuery::Expand {
                    streams: p.hc_streams,
                    width: hidden,
                    capacity,
                }),
                Backend,
            ),
            Stage::HeadCollapse => (
                Op::Sinkhorn(SinkhornHyperConnectionQuery::Head {
                    streams: p.hc_streams,
                    width: hidden,
                    mode: StreamHeadMode::MeanOfStreams,
                    capacity,
                }),
                Backend,
            ),
            Stage::InputNorm(..)
            | Stage::FinalNorm
            | Stage::DraftEmbedNorm
            | Stage::DraftHiddenNorm
            | Stage::DraftHeadNorm => (norm(hidden), Backend),
            Stage::DraftResidual(_) => (Op::Own("residual-add"), Own),
            Stage::DraftGather => (Op::Own("draft-gather"), Own),
            Stage::DraftShareSelection => (
                Op::Indexer(PooledIndexerQuery::CompactSelection {
                    width: max_keys,
                    capacity: model_glm5next_contract::contract::DRAFT_ITEMS,
                }),
                Backend,
            ),
            Stage::DraftHeadRows => (Op::Own("gather-rows"), Own),
            Stage::DraftEhInput => (Op::Own("eh-input"), Own),
            // The partial, the pairs' gathers and the final
            // (`model_glm5next_contract::draft_argmax`), prepared on the
            // backend and recorded as the partial's contract.
            Stage::DraftArgmax => (
                Op::Row(model_glm5next_contract::draft_argmax::partial_query(shape)),
                Backend,
            ),
            Stage::QueryNorm(_) => (norm(p.q_rank), Backend),
            Stage::LatentNorm(_) => (norm(p.kv_rank), Backend),
            Stage::Project(layer, id) => {
                let shape = projected
                    .get(&(Some(layer), id))
                    .ok_or_else(|| format!("layer {layer} has no projection {id}"))?;
                let gemm = GemmQuery {
                    m_capacity: capacity,
                    n: shape.n,
                    k: shape.k,
                    weight: weight(shape.dtype),
                    activation: ActivationFormat::Bf16,
                    // The router scores in F32. The row-split projections
                    // write the F32 partials the all-reduce sums, the shared
                    // expert's down projection after adding its F32 residual,
                    // the routed sum.
                    output: if matches!(
                        id,
                        "moe.router"
                            | "mtp.moe.router"
                            | "linear.output"
                            | "latent.output"
                            | "mtp.latent.output"
                            | "dense.down"
                            | "moe.shared.down"
                            | "mtp.moe.shared.down"
                    ) {
                        DType::F32
                    } else {
                        DType::Bf16
                    },
                    epilogues: Epilogues {
                        // The shared expert's output adds into the routed sum.
                        residual_add: matches!(id, "moe.shared.down" | "mtp.moe.shared.down"),
                        ..Epilogues::default()
                    },
                    grouped: None,
                };
                (Op::Gemm(gemm), Planner::Projection(id))
            }
            Stage::LmHead => {
                let head = projected
                    .get(&(None, "head"))
                    .ok_or("the head has no projection")?;
                (
                    Op::Gemm(GemmQuery {
                        m_capacity: capacity,
                        n: head.n,
                        k: head.k,
                        weight: weight(head.dtype),
                        activation: ActivationFormat::Bf16,
                        // F32 logits, which the row exchange moves.
                        output: DType::F32,
                        epilogues: Epilogues::default(),
                        grouped: None,
                    }),
                    Planner::Projection("head"),
                )
            }
            // Decode and verify rows exchanged to their block owners, a
            // prefill's logits rows gathered to rank 0 (`ops::logits`).
            Stage::GatherLogits => {
                let head = projected
                    .get(&(None, "head"))
                    .ok_or("the head has no projection")?;
                (
                    Op::ShardedRows(ShardedRows {
                        ranks: shape.ranks(),
                        shard: u64::from(head.n),
                        dtype: DType::F32,
                    }),
                    Backend,
                )
            }
            Stage::KdaConv(_) => (
                Op::Kda(shape.kda_conv(phase.recurrent(), capacity)),
                Backend,
            ),
            Stage::KdaDeltaRule(_) => (
                Op::Kda(shape.kda_delta_rule(phase.recurrent(), capacity)),
                Backend,
            ),
            Stage::KdaGatedNorm(_) => (
                Op::Recurrent(RecurrentQuery::GatedRmsNorm {
                    width: p.linear_heads * p.linear_head_dim,
                    head_dim: p.linear_head_dim,
                    eps_bits: eps.to_bits(),
                    gate: GateActivation::Sigmoid,
                    capacity,
                }),
                Backend,
            ),
            Stage::LatentKvWrite(_) => (
                Op::Latent(LatentAttentionQuery::KvWrite {
                    latent: p.kv_rank,
                    page_size: PAGE_TOKENS as u32,
                    kv,
                    capacity,
                }),
                Backend,
            ),
            Stage::QueryAbsorb(_) => (
                Op::HeadBatched(HeadBatchedGemmQuery {
                    heads: p.latent_heads,
                    n: p.kv_rank,
                    k: p.qk_nope,
                    layout: HeadWeightLayout::KByN,
                    output: DType::Bf16,
                    m_capacity: capacity,
                }),
                Backend,
            ),
            Stage::LatentAttention(_) => (
                Op::Latent(LatentAttentionQuery::Attention {
                    heads: p.latent_heads,
                    latent: p.kv_rank,
                    max_keys,
                    softmax_scale_bits: (1.0 / (g.qk_head() as f32).sqrt()).to_bits(),
                    page_size: PAGE_TOKENS as u32,
                    kv,
                    phase: phase.attention(),
                    output: LatentAttentionOutput::Normalized,
                    sink: false,
                    tail_rope_dims: 0,
                    second_fp4_keys: false,
                    capacity,
                }),
                Backend,
            ),
            Stage::ValueUp(_) => (
                Op::HeadBatched(HeadBatchedGemmQuery {
                    heads: p.latent_heads,
                    n: p.v_head,
                    k: p.kv_rank,
                    layout: HeadWeightLayout::NByK,
                    output: DType::Bf16,
                    m_capacity: capacity,
                }),
                Backend,
            ),
            Stage::IndexKeyWrite(_) => (
                Op::Indexer(PooledIndexerQuery::KeyWrite {
                    pooling: shape.pooling(),
                    key_norm_eps_bits: INDEXER_KEY_NORM_EPS.to_bits(),
                    page_size: PAGE_TOKENS as u32,
                    key_format: IndexerKeyFormat::Bf16,
                    phase: phase.recurrent(),
                    capacity,
                }),
                Backend,
            ),
            Stage::IndexSelect(_) => (Op::Indexer(select), Backend),
            Stage::Route(_) => (
                Op::Row(RowQuery::Router {
                    experts: g.shape.moe.experts,
                    top_k: p.top_k,
                    kind: RouterKind::SigmoidBiasSelectNormalized {
                        scaling_factor: g.shape.moe.routed_scale as f32,
                    },
                }),
                Backend,
            ),
            Stage::LocalRoutes(_) => (
                Op::Moe(MoeQuery::LocalRoutes {
                    experts: g.shape.moe.experts,
                    local_experts: p.local_experts,
                    top_k: p.top_k,
                    capacity,
                }),
                Backend,
            ),
            Stage::DenseGateUp(layer) => (
                Op::GatedMlp(gated(
                    2 * p.dense_intermediate,
                    hidden,
                    GatedMlpProjection::GateUp,
                    false,
                    chained(layer, Class::Dense),
                )),
                Backend,
            ),
            Stage::SharedGateUp(layer) => (
                Op::GatedMlp(gated(
                    2 * p.shared_intermediate,
                    hidden,
                    GatedMlpProjection::GateUp,
                    false,
                    chained(layer, Class::Shared),
                )),
                Backend,
            ),
            Stage::ExpertsGateUp(layer) => (
                Op::GatedMlp(gated(
                    2 * p.expert_intermediate,
                    hidden,
                    GatedMlpProjection::GateUp,
                    true,
                    chained(layer, Class::Experts),
                )),
                Backend,
            ),
            Stage::ExpertsDown(layer) => (
                Op::GatedMlp(gated(
                    hidden,
                    p.expert_intermediate,
                    GatedMlpProjection::Down,
                    true,
                    chained(layer, Class::Experts),
                )),
                Backend,
            ),
            // The decode and verify graphs' sum adds the shared expert's
            // F32 output after it (`chained::RoutedSum::add_entry`).
            Stage::RoutedSum(_) => (
                Op::Moe(MoeQuery::LocalRoutedSum {
                    experts: g.shape.moe.experts,
                    local_experts: p.local_experts,
                    top_k: p.top_k,
                    width: hidden,
                    output: DType::F32,
                    capacity,
                }),
                Backend,
            ),
        };
        Ok(Record { stage, op, planner })
    };
    stages.into_iter().map(record).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::DraftPass;

    /// For every checkpoint layout, the shared projection shapes are the
    /// ones computed fresh, and later requests return that same map.
    #[test]
    fn shared_projection_shapes_equal_fresh_ones() {
        for layout in model_glm5next_contract::quantization::LAYOUTS {
            let fresh = projections(Shape::TP4, layout);
            let shared = shared_projections(Shape::TP4, layout);
            assert_eq!(
                shared.as_deref().map_err(String::as_str),
                fresh.as_ref().map_err(String::as_str),
                "{}",
                layout.id
            );
            let again = shared_projections(Shape::TP4, layout).unwrap();
            assert!(Arc::ptr_eq(&shared.unwrap(), &again));
        }
    }

    /// Every projection record's output dtype and residual add are its
    /// declaration's (`projections.json`): its use profile's output at
    /// every use and its epilogue, in the target's program and both
    /// drafter passes, so lowering from either agrees: the row-split
    /// projections write the F32 partials the all-reduce sums (the shared
    /// expert's down projection after adding the routed sum), the router
    /// and the head F32 scores.
    #[test]
    fn projection_records_agree_with_their_declarations() {
        let declaration: serde_json::Value =
            serde_json::from_str(model_glm5next_contract::projections::DECLARATION).unwrap();
        let shape = Shape::TP4;
        let kv = LatentKvFormat::Fp8E4M3Block128;
        let mut records = Vec::new();
        for phase in [
            Phase::Prefill { rows: 64 },
            Phase::Decode(8),
            Phase::Verify { rows: 32, items: 8 },
        ] {
            records.extend(super::records(shape, kv, phase).unwrap());
        }
        for pass in [DraftPass::Extend, DraftPass::Step] {
            records.extend(drafter_records(shape, kv, Phase::Decode(8), pass).unwrap());
        }
        let mut checked = std::collections::BTreeSet::new();
        for record in records {
            let (id, gemm) = match (record.stage, record.op) {
                (Stage::Project(_, id), Op::Gemm(gemm)) => (id, gemm),
                (Stage::LmHead, Op::Gemm(gemm)) => ("head", gemm),
                _ => continue,
            };
            let declared = |kind: &str| {
                declaration[kind]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|declared| declared["id"] == id)
            };
            // The indexer's pooling gate is a side tensor its own kernel
            // reads.
            let Some(projection) = declared("projections") else {
                assert!(declared("side_tensors").is_some(), "{id} is not declared");
                continue;
            };
            assert_eq!(
                projection["epilogues"]["residual_add"].as_bool(),
                Some(gemm.epilogues.residual_add),
                "{id}"
            );
            // A row-split projection's output is a partial of the sum.
            if projection["split"]["class"] == "row" {
                assert_eq!(gemm.output, DType::F32, "{id}");
            }
            let profile = projection["use_profile"].as_str().unwrap();
            for used in declaration["use_profiles"][profile].as_array().unwrap() {
                assert_eq!(
                    used["contract"]["output_dtype"].as_str(),
                    Some(format!("{:?}", gemm.output).as_str()),
                    "{id}"
                );
            }
            checked.insert(id);
        }
        for id in [
            "head",
            "moe.router",
            "linear.output",
            "latent.output",
            "dense.down",
            "moe.shared.down",
            "mtp.latent.output",
            "mtp.moe.shared.down",
        ] {
            assert!(checked.contains(id), "{id}");
        }
    }
}
