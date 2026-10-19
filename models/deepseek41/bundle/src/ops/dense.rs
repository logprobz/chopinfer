//! The attention's dense FP8 projections: E4M3 weights with a UE8M0 scale
//! per row and 32 K (the checkpoint's 32x32 blocks repacked to 1x32), over
//! activations quantized to E4M3 per 32 with a ceil UE8M0 scale, summed in
//! F32 folds of 128 K (`Mxfp8FoldGemmQuery`). `wq_a` and `wkv` read the
//! layer input the sublayer quantized; `wq_b`, the indexer's `wq_b`,
//! `wo_a` and `wo_b` quantize their own input first. Every projection
//! writes F32, and only a quantizer's input rounds to BF16: `q_norm` and
//! the window write read `wq_a` and `wkv`, both `wq_b` read the F32
//! `q_norm` output and keep F32 to the tail RoPE. `wo_a` is the grouped
//! form: each of the rank's groups projects its heads' slice of the
//! attention output, the vendor's BF16 einsum over the same decoded
//! values. `wo_b` writes the F32 rank partial the all-reduce sums.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::precision::quantizer::{BlockQuantizeBuffers, BlockQuantizeQuery};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::moe;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::moe::QuantizeViews;
use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use crate::shape::Shape;

/// The instruction path the dense projections declare: native block-scaled
/// MMA, folded every 128 K.
const EXECUTION: BlockScaledExecution = BlockScaledExecution::NativeBlockScaledMma;

/// What a dense projection reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Input {
    /// The sublayer input the sublayer quantized.
    Layer,
    /// The normed query down projection.
    Query,
    /// The attention output, its tail inverse-rotated.
    Attention,
    /// The `wo_a` output.
    OutputA,
    /// The shared expert's activation, which quantizes itself.
    SharedActivation,
    /// An Engram layer's table rows decoded to BF16, `[rows, engram_k]`.
    Engram,
}

/// A dense projection of the program: its stage ID, its checkpoint tensor
/// under the layer, what it reads, and its GEMM at `shape`.
pub(crate) struct Dense {
    pub id: &'static str,
    pub tensor: &'static str,
    pub input: Input,
    pub query: Mxfp8FoldGemmQuery,
}

/// The dense FP8 projections of an attention layer at `shape`.
pub(crate) fn projections(shape: &Shape) -> Vec<Dense> {
    let g = shape.geometry;
    let a = &g.shape.attention;
    let heads = shape.local_heads() * a.head_dim;
    let groups = shape.local_groups();
    let group_k = a.heads / a.o_groups * a.head_dim;
    let inter = shape.local_inter();
    let gemm = |n: u32, k: u32, output: DType| Mxfp8FoldGemmQuery {
        m_capacity: PREFILL_TOKENS,
        n,
        k,
        groups: 1,
        activation_row_stride: k,
        activation_group_stride: k,
        output,
        execution: EXECUTION,
    };
    let mut dense = vec![
        Dense {
            id: "attention.q-a",
            tensor: "attn.wq_a",
            input: Input::Layer,
            query: gemm(a.q_rank, g.shape.hidden, DType::F32),
        },
        Dense {
            id: "attention.q-b",
            tensor: "attn.wq_b",
            input: Input::Query,
            query: gemm(heads, a.q_rank, DType::F32),
        },
        Dense {
            id: "indexer.q",
            tensor: "attn.indexer.wq_b",
            input: Input::Query,
            query: gemm(
                g.shape.indexer.heads * g.shape.indexer.dim,
                a.q_rank,
                DType::F32,
            ),
        },
        Dense {
            id: "attention.kv",
            tensor: "attn.wkv",
            input: Input::Layer,
            query: gemm(a.head_dim, g.shape.hidden, DType::F32),
        },
        Dense {
            id: "attention.o-a",
            tensor: "attn.wo_a",
            input: Input::Attention,
            query: Mxfp8FoldGemmQuery {
                groups,
                activation_row_stride: heads,
                ..gemm(a.o_rank, group_k, DType::F32)
            },
        },
        Dense {
            id: "attention.o-b",
            tensor: "attn.wo_b",
            input: Input::OutputA,
            query: gemm(g.shape.hidden, groups * a.o_rank, DType::F32),
        },
        // The shared expert: gate and up split by columns, down by K.
        Dense {
            id: "moe.shared-gate",
            tensor: "ffn.shared_experts.w1",
            input: Input::Layer,
            query: gemm(inter, g.shape.hidden, DType::F32),
        },
        Dense {
            id: "moe.shared-up",
            tensor: "ffn.shared_experts.w3",
            input: Input::Layer,
            query: gemm(inter, g.shape.hidden, DType::F32),
        },
        Dense {
            id: "moe.shared-down",
            tensor: "ffn.shared_experts.w2",
            input: Input::SharedActivation,
            query: gemm(g.shape.hidden, inter, DType::F32),
        },
    ];
    // An Engram layer's `wkv`: from the decoded table rows to a key per
    // stream and the value, replicated on every rank.
    if g.engram_layers() > 0 {
        dense.push(Dense {
            id: "engram.kv",
            tensor: "engram.wkv",
            input: Input::Engram,
            query: gemm(engram_kv_width(g), engram_k(g), DType::F32),
        });
    }
    dense
}

/// The width of an Engram layer's decoded table rows: every hash column's
/// row of `engram_dim`.
pub(crate) const fn engram_k(g: &Geometry) -> u32 {
    let e = &g.shape.engram;
    (e.max_ngram - 1) * e.heads * e.head_dim
}

/// The width of an Engram layer's F32 `wkv` output: a key per residual
/// stream, then the value.
pub(crate) const fn engram_kv_width(g: &Geometry) -> u32 {
    (g.shape.hc.streams + 1) * g.shape.hidden
}

impl Input {
    /// The rows' type: the F32 `q_norm`, `wo_a` and shared gate and up
    /// outputs, the BF16 layer input, attention output and Engram rows.
    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::Query | Self::OutputA | Self::SharedActivation => DType::F32,
            Self::Layer | Self::Attention | Self::Engram => DType::Bf16,
        }
    }

    /// Whether the projection quantizes its input itself; the sublayer and
    /// the shared expert's activation quantize the others'.
    pub(crate) const fn quantizes(self) -> bool {
        matches!(
            self,
            Self::Query | Self::Attention | Self::OutputA | Self::Engram
        )
    }
}

/// The inputs the dense projections quantize themselves: each width and
/// type. The quantizer rounds an F32 input to BF16 before its amax.
pub(crate) fn quantized_inputs(shape: &Shape) -> Vec<(u32, DType)> {
    let mut inputs: Vec<(u32, DType)> = Vec::new();
    for dense in projections(shape) {
        let input = (dense.query.activation_row_stride, dense.input.dtype());
        if dense.input.quantizes() && !inputs.contains(&input) {
            inputs.push(input);
        }
    }
    inputs
}

/// A quantized input's width and type.
type QuantizedInput = (u32, DType);

pub(crate) struct DenseOps {
    /// Each quantized input's quantizer, by width and type.
    quantizers: Vec<(QuantizedInput, Prepared<BlockQuantizeBuffers<View>>)>,
    /// Each projection's GEMM, by stage ID.
    gemms: Vec<(&'static str, Prepared<Mxfp8FoldGemmBuffers<View>>)>,
    decode: Vec<(&'static str, u32, Prepared<Mxfp8FoldGemmBuffers<View>>)>,
    paired: Vec<(u32, Prepared<[Mxfp8FoldGemmBuffers<View>; 2]>)>,
}

/// A step's spans of a dense projection: the activation codes and scales,
/// the weight codes and scales, the output rows and their status.
pub(crate) struct GemmViews {
    pub activations: DeviceBufferSpanV1,
    pub activation_scales: DeviceBufferSpanV1,
    pub weights: DeviceBufferSpanV1,
    pub weight_scales: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    /// The phase's row status, which every fold shares from its first row.
    pub status: DeviceBufferSpanV1,
}

impl DenseOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<DenseOps, Refused> {
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let gemms = projections(shape)
            .into_iter()
            .map(|dense| {
                let gemm = asks.ask(format!("dense.{}", dense.id), every, |need| {
                    backend.mxfp8_fold_gemm(&dense.query, need)
                })?;
                Ok((dense.id, gemm))
            })
            .collect::<Result<_, Refused>>()?;
        let quantizers = quantized_inputs(shape)
            .into_iter()
            .map(|(width, input)| {
                let query = BlockQuantizeQuery {
                    input,
                    ..moe::quantizer(width)
                };
                let what = format!("dense.quantize {input:?} w{width}");
                let op = asks.ask(what, every, |need| backend.block_quantize(&query, need))?;
                Ok(((width, input), op))
            })
            .collect::<Result<_, Refused>>()?;
        let mut decode = Vec::new();
        let mut paired = Vec::new();
        for dense in projections(shape) {
            let query = Mxfp8FoldGemmQuery {
                m_capacity: 32,
                ..dense.query
            };
            for capacity in crate::sm120::mxfp8_fold::DECODE_BUCKETS {
                let need =
                    exact::need(capacity, exact::of_phase(Phase::Decode(capacity), capacity));
                let op = asks.ask(
                    format!("dense.{} decode {capacity}", dense.id),
                    need,
                    |need| backend.mxfp8_fold_gemm(&query, need),
                )?;
                decode.push((dense.id, capacity, op));
                if dense.id == "moe.shared-gate" {
                    let op = asks.ask(
                        format!("dense.shared-pair decode {capacity}"),
                        need,
                        |need| backend.mxfp8_fold_gemm_pair(&query, need),
                    )?;
                    paired.push((capacity, op));
                }
            }
        }
        Ok(DenseOps {
            quantizers,
            gemms,
            decode,
            paired,
        })
    }

    /// The quantization of `rows` rows of `width` columns of `input`.
    pub(crate) fn quantize(
        &self,
        width: u32,
        input: DType,
        rows: u32,
        views: &QuantizeViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .quantizers
            .iter()
            .find(|(of, _)| *of == (width, input))
            .ok_or(Refusal(
                "DeepSeek V4.1 quantizes no dense input of this width",
            ))?;
        let mut bind = Bind::default();
        let buffers = BlockQuantizeBuffers {
            input: bind.view(views.input),
            output: bind.view(views.codes),
            scales: Some(bind.view(views.scales)),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: 0,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// Projection `id` of `rows` rows.
    pub(crate) fn project(
        &self,
        id: &str,
        phase: Phase,
        views: &GemmViews,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
        let op = if matches!(phase, Phase::Decode(_)) {
            &self
                .decode
                .iter()
                .find(|(of, capacity, _)| *of == id && *capacity == rows)
                .ok_or(Refusal("DeepSeek V4.1 has no dense decode bucket"))?
                .2
        } else {
            &self
                .gemms
                .iter()
                .find(|(of, _)| *of == id)
                .ok_or(Refusal("DeepSeek V4.1 has no dense projection of this ID"))?
                .1
        };
        let mut bind = Bind::default();
        let buffers = Mxfp8FoldGemmBuffers {
            activations: bind.view(views.activations),
            activation_scales: bind.view(views.activation_scales),
            weights: bind.view(views.weights),
            weight_scales: bind.view(views.weight_scales),
            output: bind.view(views.output),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: 0,
        };
        bind.launch_within(op, &buffers, workspace, Counts::rows(rows))
    }

    pub(crate) fn project_pair(
        &self,
        rows: u32,
        views: &[GemmViews; 2],
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = &self
            .paired
            .iter()
            .find(|(capacity, _)| *capacity == rows)
            .ok_or(Refusal("DeepSeek V4.1 has no paired dense decode bucket"))?
            .1;
        let [first, second] = views;
        if first.activations != second.activations
            || first.activation_scales != second.activation_scales
            || first.live_rows != second.live_rows
            || first.status != second.status
        {
            return Err(Refusal(
                "paired dense projections require shared input spans",
            ));
        }
        let mut bind = Bind::default();
        let first = Mxfp8FoldGemmBuffers {
            activations: bind.view(first.activations),
            activation_scales: bind.view(first.activation_scales),
            weights: bind.view(first.weights),
            weight_scales: bind.view(first.weight_scales),
            output: bind.view(first.output),
            live_rows: bind.view(first.live_rows),
            status: bind.view(first.status),
            status_row_offset: 0,
        };
        let second = Mxfp8FoldGemmBuffers {
            weights: bind.view(second.weights),
            weight_scales: bind.view(second.weight_scales),
            output: bind.view(second.output),
            ..first
        };
        bind.launch_within(op, &[first, second], workspace, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::contract::Validate;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn decode_buckets_use_split_folds_and_prefill_keeps_the_original_entry() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = DenseOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
        let span = |index, length| {
            DeviceBufferSpanV1::new(
                runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(index, 1),
                0,
                length,
            )
        };
        let workspace = span(99, 1 << 30);
        for dense in projections(&shape) {
            let q = dense.query;
            for phase in [
                Phase::Decode(1),
                Phase::Decode(2),
                Phase::Decode(4),
                Phase::Decode(8),
                Phase::Decode(16),
                Phase::Decode(32),
                Phase::Prefill { rows: 1 },
                Phase::Prefill { rows: 64 },
                Phase::Verify { rows: 6, items: 1 },
            ] {
                let rows = u64::from(phase.rows());
                let views = GemmViews {
                    activations: span(1, rows * u64::from(q.activation_row_stride)),
                    activation_scales: span(2, rows * u64::from(q.activation_row_stride) / 32),
                    weights: span(3, u64::from(q.groups) * u64::from(q.n) * u64::from(q.k)),
                    weight_scales: span(
                        4,
                        u64::from(q.groups) * u64::from(q.n) * u64::from(q.k) / 32,
                    ),
                    output: span(5, rows * u64::from(q.groups) * u64::from(q.n) * 4),
                    live_rows: span(6, 4),
                    status: span(7, rows * 4),
                };
                let calls = ops.project(dense.id, phase, &views, workspace).unwrap();
                if matches!(phase, Phase::Decode(_)) {
                    assert_eq!(calls.len(), 3);
                    assert!(calls[1].launch.symbol.contains("_partials_v1"));
                } else {
                    assert_eq!(calls.len(), 1);
                    assert!(calls[0].launch.symbol.contains("mxfp8_fold_m8192"));
                    assert_eq!(calls[0].launch.plan.grid[0], q.n.div_ceil(32));
                }
                if dense.id == "moe.shared-gate" && matches!(phase, Phase::Decode(_)) {
                    let second = GemmViews {
                        weights: span(8, views.weights.length),
                        weight_scales: span(9, views.weight_scales.length),
                        output: span(10, views.output.length),
                        ..views
                    };
                    let calls = ops
                        .project_pair(phase.rows(), &[views, second], workspace)
                        .unwrap();
                    assert_eq!(calls.len(), 3);
                    assert!(calls.iter().all(|c| c.launch.symbol.contains("_pair_v1")));
                }
            }
        }
    }

    /// At four ranks each rank projects its 16 heads' queries, its two
    /// `wo_a` groups of 4096 K to 1024, and `wo_b` from their 2048 outputs
    /// to the hidden width, as the vendor's TP splits do.
    #[test]
    fn flash_projections_split_heads_and_groups_by_rank() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let dense = projections(&shape);
        let query = |id: &str| dense.iter().find(|d| d.id == id).unwrap().query;
        for d in &dense {
            d.query.validate().unwrap();
        }
        // Every projection writes F32.
        assert!(dense.iter().all(|d| d.query.output == DType::F32));
        assert_eq!(
            (query("attention.q-a").n, query("attention.q-a").k),
            (1280, 5120)
        );
        let q_b = query("attention.q-b");
        assert_eq!((q_b.n, q_b.k, q_b.output), (8192, 1280, DType::F32));
        let index_q = query("indexer.q");
        assert_eq!(
            (index_q.n, index_q.k, index_q.output),
            (4096, 1280, DType::F32)
        );
        assert_eq!(
            (query("attention.kv").n, query("attention.kv").k),
            (512, 5120)
        );
        let o_a = query("attention.o-a");
        assert_eq!((o_a.groups, o_a.n, o_a.k), (2, 1024, 4096));
        assert_eq!(
            (o_a.activation_row_stride, o_a.activation_group_stride),
            (8192, 4096)
        );
        let o_b = query("attention.o-b");
        assert_eq!((o_b.n, o_b.k, o_b.output), (5120, 2048, DType::F32));
        // Engram's `wkv`: three n-gram sizes of eight heads' 256-wide rows
        // to four keys and the value.
        let kv = query("engram.kv");
        assert_eq!((kv.n, kv.k, kv.output), (25600, 6144, DType::F32));
        assert_eq!(
            quantized_inputs(&shape),
            [
                (1280, DType::F32),
                (8192, DType::Bf16),
                (2048, DType::F32),
                (6144, DType::Bf16)
            ]
        );
    }
}
