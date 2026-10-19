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
use super::row_buckets::RowBuckets;
use super::{Asks, Bind, Lowered, exact, fold_stream};
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
    stream: Vec<(&'static str, u32, fold_stream::Op)>,
    stream_pair: Vec<(u32, fold_stream::Op)>,
    tensors: Vec<(&'static str, &'static str)>,
    prefill_gemms: Vec<(&'static str, Prepared<Mxfp8FoldGemmBuffers<View>>)>,
    /// Each quantized input's quantizer, by width and type.
    quantizers: Vec<(QuantizedInput, RowBuckets<BlockQuantizeBuffers<View>>)>,
    /// Each projection's GEMM, by stage ID.
    gemms: Vec<(&'static str, Prepared<Mxfp8FoldGemmBuffers<View>>)>,
    decode: Vec<(&'static str, u32, Prepared<Mxfp8FoldGemmBuffers<View>>)>,
    paired: Vec<(u32, Prepared<[Mxfp8FoldGemmBuffers<View>; 2]>)>,
}

/// A step's spans of a dense projection: the activation codes and scales,
/// the weight codes and scales, the output rows and their status.
pub(crate) struct GemmViews {
    pub stream: Option<fold_stream::Storage>,
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
        let prefill_gemms = projections(shape)
            .into_iter()
            .map(|dense| {
                let gemm = asks.ask(format!("dense.prefill.{}", dense.id), every, |need| {
                    backend.mxfp8_prefill_gemm(&dense.query, need)
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
                let op = RowBuckets::prepare(asks, what, every, |need| {
                    backend.block_quantize(&query, need)
                })?;
                Ok(((width, input), op))
            })
            .collect::<Result<_, Refused>>()?;
        let mut stream = Vec::new();
        let mut stream_pair = Vec::new();
        let tensors = projections(shape)
            .iter()
            .map(|d| (d.id, d.tensor))
            .collect();
        let mut decode = Vec::new();
        let mut paired = Vec::new();
        for dense in projections(shape) {
            for capacity in 1..=crate::Kernels::dense_decode_capacity() {
                let query = crate::Kernels::dense_decode_query(dense.query, capacity);
                let need = exact::need(capacity, exact::verify(capacity));
                let op = asks.ask(
                    format!("dense.{} decode {capacity}", dense.id),
                    need,
                    |need| backend.mxfp8_fold_gemm(&query, need),
                )?;
                decode.push((dense.id, capacity, op));
                if let Some(op) =
                    fold_stream::prepare(backend, shape.geometry, query, capacity, false)?
                {
                    asks.record(
                        format!("dense.{} stream {capacity}", dense.id),
                        need,
                        op.info(),
                    );
                    stream.push((dense.id, capacity, op));
                }
                if dense.id == "moe.shared-gate" {
                    let op = asks.ask(
                        format!("dense.shared-pair decode {capacity}"),
                        need,
                        |need| backend.mxfp8_fold_gemm_pair(&query, need),
                    )?;
                    paired.push((capacity, op));
                    if let Some(op) =
                        fold_stream::prepare(backend, shape.geometry, query, capacity, true)?
                    {
                        asks.record(
                            format!("dense.shared-pair stream {capacity}"),
                            need,
                            op.info(),
                        );
                        stream_pair.push((capacity, op));
                    }
                }
            }
        }
        Ok(DenseOps {
            stream,
            stream_pair,
            tensors,
            quantizers,
            gemms,
            decode,
            paired,
            prefill_gemms,
        })
    }

    /// Largest selected bucket initializes one persistent operation per weight.
    pub(crate) fn stream_weight_ops(&self, weight: &str) -> Vec<(bool, &fold_stream::Op)> {
        let Some((id, _)) = self
            .tensors
            .iter()
            .find(|(_, tensor)| weight.ends_with(&format!(".{tensor}.weight")))
        else {
            return Vec::new();
        };
        let mut result = Vec::new();
        if let Some((_, _, op)) = self
            .stream
            .iter()
            .filter(|(of, _, _)| of == id)
            .max_by_key(|(_, rows, _)| *rows)
        {
            result.push((false, op));
        }
        if *id == "moe.shared-gate"
            && let Some((_, op)) = self.stream_pair.iter().max_by_key(|(rows, _)| *rows)
        {
            result.push((true, op));
        }
        result
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
        bind.launch(op.at(rows), &buffers, Counts::rows(rows))
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
        let op = if phase.prefill() && rows >= 128 {
            &self
                .prefill_gemms
                .iter()
                .find(|(of, _)| *of == id)
                .ok_or(Refusal(
                    "DeepSeek V4.1 has no dense prefill projection of this ID",
                ))?
                .1
        } else if !phase.prefill() && rows <= crate::Kernels::dense_decode_capacity() {
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
        if !phase.prefill()
            && let Some((_, _, op)) = self
                .stream
                .iter()
                .find(|(of, capacity, _)| *of == id && *capacity == rows)
        {
            return fold_stream::launch(op, bind, buffers, None, views.stream, rows);
        }
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
        let stream = first.stream;
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
        if let Some((_, op)) = self
            .stream_pair
            .iter()
            .find(|(capacity, _)| *capacity == rows)
        {
            return fold_stream::launch(op, bind, first, Some(second), stream, rows);
        }
        bind.launch_within(op, &[first, second], workspace, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::contract::Validate;
    use model_deepseek41_contract::geometry::FLASH;

    fn expect_split(ranks: u32, projection: &str, rows: u32) -> bool {
        match (ranks, projection) {
            (2, "attention.q-b") => matches!(rows, 1..=4 | 7..=11),
            (4, "attention.q-b") => matches!(rows, 1..=2 | 5..=12 | 16),
            (2, "moe.shared-down") => matches!(rows, 1..=2 | 9..=16),
            (4, "moe.shared-down") => matches!(rows, 1..=28),
            _ => false,
        }
    }

    fn expected_family(query: Mxfp8FoldGemmQuery, rows: u32, paired: bool) -> &'static str {
        use crate::sm120::mxfp8_fold::{DecodePath, decode_path};
        match decode_path(&FLASH, query, rows, paired) {
            DecodePath::Split => "mxfp8_decode_",
            DecodePath::Ticket => "mxfp8_ticket_",
            DecodePath::Prefetch => "mxfp8_prefetch_",
        }
    }

    #[test]
    fn tied_dense_buckets_use_split_folds_at_each_topology() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                DenseOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            for (id, rows, op) in &ops.decode {
                let split = expect_split(shape.ranks(), id, *rows);
                assert_eq!(
                    op.info().symbols.len(),
                    if split { 3 } else { 2 },
                    "{} {id} rows={rows}",
                    shape.topology()
                );
                let q = projections(&shape)
                    .into_iter()
                    .find(|d| d.id == *id)
                    .unwrap()
                    .query;
                assert!(
                    op.info()
                        .symbols
                        .iter()
                        .all(|(s, _)| s.contains(expected_family(q, *rows, false)))
                );
            }
            assert_eq!(ops.decode.len(), projections(&shape).len() * 64);
        }
    }

    #[test]
    fn small_prefills_use_fold_and_full_tiles_use_staging() {
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
                Phase::Verify { rows: 48, items: 8 },
                Phase::Verify {
                    rows: 64,
                    items: 16,
                },
                Phase::Verify {
                    rows: 96,
                    items: 16,
                },
                Phase::Prefill { rows: 1 },
                Phase::Prefill { rows: 17 },
                Phase::Prefill { rows: 64 },
                Phase::Prefill { rows: 127 },
                Phase::Prefill { rows: 128 },
                Phase::Prefill { rows: 129 },
                Phase::Prefill { rows: 1000 },
                Phase::Prefill { rows: 4095 },
                Phase::Prefill { rows: 8150 },
                Phase::Prefill { rows: 8191 },
                Phase::Verify { rows: 6, items: 1 },
            ] {
                let rows = u64::from(phase.rows());
                let views = GemmViews {
                    stream: Some(fold_stream::Storage {
                        packed: [span(100, 1 << 30), span(101, 1 << 30)],
                        state: span(102, 1 << 20),
                        workspace: span(103, 1 << 30),
                    }),
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
                if !phase.prefill() && phase.rows() <= 64 {
                    let split = expect_split(shape.ranks(), dense.id, phase.rows());
                    let packed =
                        crate::Kernels::stream_fold_qualified(&FLASH, q, phase.rows(), false);
                    assert_eq!(
                        calls.len(),
                        if packed {
                            1
                        } else if split {
                            3
                        } else {
                            2
                        }
                    );
                    assert!(
                        calls
                            .iter()
                            .all(|call| call.launch.symbol.contains(if packed {
                                "mxfp8_stream_"
                            } else {
                                expected_family(q, phase.rows(), false)
                            }))
                    );
                } else if phase.prefill() && phase.rows() >= 128 {
                    let split =
                        crate::sm120::mxfp8_fold::prefill_tail_policy(q).splits(phase.rows());
                    assert_eq!(calls.len(), if split { 3 } else { 2 });
                    assert!(calls[0].launch.symbol.ends_with("_admit_v1"));
                    assert!(calls[1].launch.symbol.ends_with(if split {
                        "_split_full_v1"
                    } else {
                        "_prefill_tma_v1"
                    }));
                    assert_eq!(calls[1].launch.plan.grid[0], q.n.div_ceil(128));
                    if split {
                        assert!(calls[2].launch.symbol.ends_with("_split_tail_v1"));
                        assert_eq!(calls[2].launch.plan.grid[0], q.n.div_ceil(32));
                    }
                } else {
                    assert_eq!(calls.len(), 1);
                    assert!(calls[0].launch.symbol.contains("mxfp8_fold_m8192"));
                    assert_eq!(calls[0].launch.plan.grid[0], q.n.div_ceil(32));
                }
                if dense.id == "moe.shared-gate" && !phase.prefill() && phase.rows() <= 64 {
                    let second = GemmViews {
                        weights: span(8, views.weights.length),
                        weight_scales: span(9, views.weight_scales.length),
                        output: span(10, views.output.length),
                        ..views
                    };
                    let calls = ops
                        .project_pair(phase.rows(), &[views, second], workspace)
                        .unwrap();
                    assert_eq!(calls.len(), 2);
                    assert!(calls.iter().all(|c| {
                        c.launch.symbol.contains("_pair_f32_")
                            && c.launch
                                .symbol
                                .contains(expected_family(q, phase.rows(), true))
                    }));
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
    #[test]
    fn dense_ceil_prefill_keeps_decode_and_verify_entries() {
        use model_deepseek41_contract::serving::DECODE_BATCH;
        assert_eq!(
            super::super::verify_rows(),
            DECODE_BATCH,
            "extend the preserved row buckets before increasing verify capacity"
        );
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
        for entry in crate::sm120::ceil_prefill::baseline_quantizers(&FLASH, 4).unwrap() {
            let width = entry.query.width;
            for rows in [1, DECODE_BATCH, DECODE_BATCH + 1, 1024, 2048, 8192] {
                let elements = u64::from(rows) * u64::from(width);
                let views = QuantizeViews {
                    input: span(
                        1,
                        elements
                            * if entry.query.input == DType::F32 {
                                4
                            } else {
                                2
                            },
                    ),
                    codes: span(2, elements),
                    scales: span(3, elements / 32),
                    live_rows: span(4, 4),
                    status: span(5, u64::from(rows) * 4),
                };
                let calls = ops
                    .quantize(width, entry.query.input, rows, &views)
                    .unwrap();
                assert_eq!(calls.len(), 1);
                let call = &calls[0].launch;
                if rows <= DECODE_BATCH {
                    assert!(call.symbol.ends_with("_parallel_v1"), "{}", call.symbol);
                } else {
                    let warps = if width >= 8192 { 16 } else { 8 };
                    assert!(
                        call.symbol.ends_with(&format!("_prefill_w{warps}_v1")),
                        "{}",
                        call.symbol
                    );
                    assert_eq!(call.plan.block, [warps * 32, 1, 1]);
                }
            }
        }
    }
}
