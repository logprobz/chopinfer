//! Every new static query maps to one op family, and only a dense GEMM
//! entry declares a dense kernel.

use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    ActivationFormat, AttentionPhase, CompileKey, CompileKnobs, DType, DenseKernel, Epilogues,
    GemmQuery, InstantiationSpec, OpFamily, RecurrentPhase, Refusal, ScaleMode, StaticQuery,
    SwigluColumns, Target, Validate, WeightFormat,
};
use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, SwigluClamp};
use kernel_core::gated_pooling::GatedGroupPooling;
use kernel_core::latent_attention::{
    HeadBatchedGemmQuery, HeadWeightLayout, LatentAttentionOutput, LatentAttentionQuery,
    LatentKvFormat,
};
use kernel_core::per_channel_decay_delta_rule::{DecayGate, PerChannelDecayDeltaRuleQuery};
use kernel_core::pooled_indexer::{
    IndexerGranularity, IndexerKeyFormat, IndexerSelectionMode, PooledIndexerQuery,
};
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, SinkhornHyperConnectionQuery, SinkhornMix, SinkhornOrder,
};

fn spec(family: OpFamily, query: StaticQuery, dense: Option<DenseKernel>) -> InstantiationSpec {
    InstantiationSpec {
        family,
        key: CompileKey {
            query,
            knobs: CompileKnobs {
                tile_m: 16,
                tile_n: 64,
                tile_k: 64,
                stages: 2,
                swizzle: 1,
                warps: 4,
                split_k_capacity: 1,
            },
            dense,
            target: Target::Sm120f,
        },
        symbol: "chop_contract_probe_v1".to_owned(),
    }
}

fn mix() -> SinkhornMix {
    SinkhornMix {
        streams: 4,
        width: 4096,
        iterations: 20,
        eps_bits: 1e-6_f32.to_bits(),
        norm_eps_bits: 1e-5_f32.to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::PerSublayer,
        fn_dtype: DType::Bf16,
    }
}

fn dense_gate_up() -> GatedMlpProjectionQuery {
    GatedMlpProjectionQuery {
        gemm: GemmQuery {
            m_capacity: 32,
            n: 2 * 3072,
            k: 4096,
            weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            activation: ActivationFormat::Bf16,
            output: DType::Bf16,
            epilogues: Epilogues {
                swiglu: Some(SwigluColumns::ConcatenatedHalves),
                ..Epilogues::default()
            },
            grouped: None,
        },
        projection: GatedMlpProjection::GateUp,
        swiglu_clamp: Some(SwigluClamp::symmetric(10.0)),
        route_weight: None,
    }
}

#[test]
fn new_queries_map_to_their_op_families() {
    let cases = [
        (
            OpFamily::Gemm,
            StaticQuery::GatedMlpProjection(dense_gate_up()),
        ),
        (
            OpFamily::Gemm,
            StaticQuery::HeadBatchedGemm(HeadBatchedGemmQuery {
                heads: 16,
                n: 512,
                k: 256,
                layout: HeadWeightLayout::KByN,
                output: DType::Bf16,
                m_capacity: 256,
            }),
        ),
        (
            OpFamily::Attention,
            StaticQuery::LatentAttention(LatentAttentionQuery::Attention {
                heads: 16,
                latent: 512,
                max_keys: 2051,
                softmax_scale_bits: 0.0625_f32.to_bits(),
                page_size: 64,
                kv: LatentKvFormat::Fp8E4M3Block128,
                phase: AttentionPhase::Decode,
                output: LatentAttentionOutput::Normalized,
                sink: false,
                tail_rope_dims: 0,
                second_fp4_keys: false,
                capacity: 256,
            }),
        ),
        (
            OpFamily::Row,
            StaticQuery::LatentAttention(LatentAttentionQuery::KvWrite {
                latent: 512,
                page_size: 64,
                kv: LatentKvFormat::Bf16,
                capacity: 256,
            }),
        ),
        (
            OpFamily::Row,
            StaticQuery::PooledIndexer(PooledIndexerQuery::KeyWrite {
                pooling: GatedGroupPooling {
                    group: 4,
                    width: 128,
                    position_dtype: DType::F32,
                },
                key_norm_eps_bits: 1e-6_f32.to_bits(),
                page_size: 64,
                key_format: IndexerKeyFormat::Bf16,
                phase: RecurrentPhase::Decode,
                capacity: 256,
            }),
        ),
        (
            OpFamily::Row,
            StaticQuery::PooledIndexer(PooledIndexerQuery::Select {
                heads: 32,
                head_dim: 128,
                group: 4,
                max_top_k: 2048,
                always_tail: true,
                granularity: IndexerGranularity::Pool,
                key_format: IndexerKeyFormat::Bf16,
                mode: IndexerSelectionMode::Global,
                page_size: 64,
                capacity: 256,
            }),
        ),
        (
            OpFamily::Row,
            StaticQuery::SinkhornHyperConnection(SinkhornHyperConnectionQuery::MixCollapse {
                mix: mix(),
                capacity: 256,
            }),
        ),
        (
            OpFamily::Comm,
            StaticQuery::SinkhornHyperConnection(
                SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
                    ranks: 4,
                    dtype: DType::F32,
                    mix: mix(),
                    rmsnorm_eps_bits: 1e-5_f32.to_bits(),
                    norm: kernel_core::rmsnorm::RmsNormReduction::WarpTree { threads: 256 },
                    capacity: 256,
                },
            ),
        ),
        (
            OpFamily::Recurrent,
            StaticQuery::PerChannelDecayDeltaRule(PerChannelDecayDeltaRuleQuery::DeltaRule {
                heads: 16,
                head_dim: 128,
                gate: DecayGate::BoundedSigmoid {
                    lower_bound_bits: (-5.0_f32).to_bits(),
                },
                phase: RecurrentPhase::Verify { max_rows: 4 },
                capacity: 256,
            }),
        ),
        (
            OpFamily::Row,
            StaticQuery::ContextParallel(ContextParallelQuery::PartialMerge {
                ranks: 4,
                heads: 16,
                width: 512,
                capacity: 256,
            }),
        ),
    ];
    for (family, query) in cases {
        spec(family, query.clone(), None).validate().unwrap();
        let other = if family == OpFamily::Row {
            OpFamily::Gemm
        } else {
            OpFamily::Row
        };
        assert_eq!(
            spec(other, query, None).validate(),
            Err(Refusal("op family does not match static query"))
        );
    }
}

#[test]
fn a_dense_clamped_projection_declares_a_dense_kernel() {
    let dense = StaticQuery::GatedMlpProjection(dense_gate_up());
    let kernel = Some(DenseKernel::FragmentDecode { splits: 1 });
    spec(OpFamily::Gemm, dense, kernel).validate().unwrap();
    let head_batched = StaticQuery::HeadBatchedGemm(HeadBatchedGemmQuery {
        heads: 16,
        n: 256,
        k: 512,
        layout: HeadWeightLayout::NByK,
        output: DType::Bf16,
        m_capacity: 256,
    });
    assert_eq!(
        spec(OpFamily::Gemm, head_batched, kernel).validate(),
        Err(Refusal("only a dense GEMM entry declares a dense kernel"))
    );
}
