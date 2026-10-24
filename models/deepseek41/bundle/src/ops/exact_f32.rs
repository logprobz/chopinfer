//! The exact F32 dense projections (F32-TQ): BF16 checkpoint weights over
//! F32 activations into F32 rows (`ActivationFormat::F32`,
//! `DecodeOrder::ExactF32Operands`), each bound through
//! `Backend::projection` at the served prefill capacity, which every
//! phase's rows fit. The prepared op chooses the ordered split entry at
//! rows <= 256 for decode and verification. Larger rows select only the
//! measured tiles declared in `sm120::f32_prefill` and `sm120::f32_blocked`; other counts retain
//! the original entry. Both optimized paths read the weight-domain
//! table cached at instance creation.
//!
//! Beside the head (`ops::head`) they are the vendor's F32 or plain
//! `F.linear` projections, replicated on every rank:
//! - the router, `linear(x.float(), weight.float())` (`Gate`);
//! - a KV source's compressor `wkv` and, at ratio 2, `wgate` over
//!   `x.float()` (`Compressor`; ratio 1's `wkv` writes F32 under F32-TQ);
//! - an index source's `weights_proj` (`Indexer`);
//! - a KV source's indexer `wk` over the normalized F32 latent, never a
//!   BF16-rounded one.
//!
//! All but `wk` read the sublayer's normed rows widened exactly to F32
//! (`own_ops::widen`), which the sublayer's input norm writes once.

#[allow(dead_code, reason = "wide consumers follow paired qualification")]
pub(crate) mod cached;

use kernel_core::backend::{Backend, Counts, OpInfo, Prepared, Refused, View};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, NumericalContract, Refusal, Target, WeightFormat,
};
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights};
use kernel_core::precision::format::{KernelFamily, registry};
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, CodeEncoding, DecodeOrder, ExecutionPhase, OperationKind,
    OperationSpec, OperationUse, Rounding, TensorSpan, WeightLayout, WeightSpec,
};
use model_deepseek41_contract::geometry::{Geometry, LayerRole};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::program::Sublayer;
use crate::shape::Shape;

/// The use every exact F32 projection is asked at.
pub(crate) const USE: OperationUse = OperationUse {
    phase: ExecutionPhase::Prefill,
    capacity: PREFILL_TOKENS,
    graph: false,
};

/// The exact F32 projection of `n` outputs from `k` inputs.
pub(crate) fn query(n: u32, k: u32) -> GemmQuery {
    GemmQuery {
        m_capacity: USE.capacity,
        n,
        k,
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::F32,
        output: DType::F32,
        epilogues: Epilogues::default(),
        grouped: None,
    }
}

/// The projection `q` of the BF16 weight `tensor` as an operation the
/// backend binds.
pub(crate) fn operation(id: &str, tensor: &str, q: GemmQuery) -> Result<OperationSpec, Refused> {
    let dims = vec![u64::from(q.n), u64::from(q.k)];
    Ok(OperationSpec {
        id: id.into(),
        kind: OperationKind::Dense,
        target: Target::Sm120f,
        weights: vec![WeightSpec {
            codes: TensorSpan {
                tensor: tensor.into(),
                dtype: DType::Bf16,
                shape: dims.clone(),
                offset: 0,
                byte_length: u64::from(q.n) * u64::from(q.k) * 2,
            },
            logical_shape: dims,
            format: WeightFormat::Bf16,
            encoding: CodeEncoding::Bf16LittleEndian,
            layout: WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        }],
        geometry: None,
        activations: vec![ActivationSpec::f32()],
        arithmetic: ArithmeticSpec {
            weight_operand: DType::F32,
            accumulator: DType::F32,
            output: DType::F32,
            decode_order: DecodeOrder::ExactF32Operands,
            rounding: Rounding::NearestEven,
            numerical_contract: q.numerical_contract_id()?.to_string(),
        },
        epilogues: q.epilogues,
        uses: vec![USE],
    })
}

/// The launches of `op` over `rows` rows: F32 `activation` rows through the
/// BF16 `weight` into F32 `output` rows.
pub(crate) fn project(
    op: &Prepared<ProjectionBuffers<View>>,
    rows: u32,
    activation: DeviceBufferSpanV1,
    weight: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    live_rows: DeviceBufferSpanV1,
    state: ProjectionState,
) -> Result<Vec<Lowered>, Refusal> {
    let mut bind = Bind::default();
    let buffers = bind_buffers(
        &mut bind,
        activation,
        weight,
        output,
        live_rows,
        state.weight_domain,
    );
    bind.launch_within(op, &buffers, state.workspace, Counts::rows(rows))
}

/// Immutable weight metadata and the shared per-launch workspace.
#[derive(Clone, Copy)]
pub struct ProjectionState {
    pub weight_domain: Option<DeviceBufferSpanV1>,
    pub workspace: DeviceBufferSpanV1,
}

pub(crate) fn domain_name(weight: &str) -> String {
    format!("{weight}.f32_domain")
}

pub(crate) fn state(buffers: &crate::buffers::RankBuffers, weight: &str) -> ProjectionState {
    ProjectionState {
        weight_domain: buffers.views.get(&domain_name(weight)).copied(),
        workspace: buffers.workspace,
    }
}

/// Classify immutable weights once, after upload and before graph capture.
pub(crate) fn initialize(
    op: &Prepared<ProjectionBuffers<View>>,
    weight: DeviceBufferSpanV1,
    domain: DeviceBufferSpanV1,
) -> Result<Vec<Lowered>, Refusal> {
    let mut bind = Bind::default();
    let buffers = bind_buffers(&mut bind, weight, weight, weight, weight, Some(domain));
    bind.initialize(op, &buffers)
}

fn bind_buffers(
    bind: &mut Bind,
    activation: DeviceBufferSpanV1,
    weight: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    live_rows: DeviceBufferSpanV1,
    weight_domain: Option<DeviceBufferSpanV1>,
) -> ProjectionBuffers<View> {
    let activation = bind.view(activation);
    ProjectionBuffers {
        activation,
        live_rows: bind.view(live_rows),
        output: bind.view(output),
        residual: None,
        weights: QuantizedWeights {
            codes: bind.view(weight),
            local_scales: None,
            global_scales: None,
        },
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            weight_domain: weight_domain.map(|span| bind.view(span)),
            partial: activation,
            semaphores: activation,
            unused: activation,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
        },
        row_offset: 0,
        shard_rows: 0,
    }
}

/// What a dense F32 projection reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Input {
    /// The sublayer's normed rows widened to F32.
    Normed,
    /// The KV source's normalized F32 latent.
    Latent,
}

/// A dense F32 projection of the program: its stage ID, its checkpoint
/// tensor under the layer, what it reads and its outputs and inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct F32Dense {
    pub id: &'static str,
    pub tensor: &'static str,
    pub input: Input,
    pub n: u32,
    pub k: u32,
}

/// The dense F32 projections of `g`.
pub(crate) fn projections(g: &Geometry) -> Vec<F32Dense> {
    let s = &g.shape;
    let dense = |id, tensor, input, n, k| F32Dense {
        id,
        tensor,
        input,
        n,
        k,
    };
    let mut projections = vec![dense(
        "moe.router",
        "ffn.gate",
        Input::Normed,
        s.moe.experts,
        s.hidden,
    )];
    if !s.csa.kv_sources.is_empty() {
        let head_dim = s.attention.head_dim;
        projections.extend([
            dense(
                "compressor.kv",
                "attn.compressor.wkv",
                Input::Normed,
                head_dim,
                s.hidden,
            ),
            dense(
                "compressor.gate",
                "attn.compressor.wgate",
                Input::Normed,
                head_dim,
                s.hidden,
            ),
            dense(
                "indexer.k",
                "attn.indexer.wk",
                Input::Latent,
                s.indexer.dim,
                head_dim,
            ),
        ]);
    }
    if !s.csa.index_sources.is_empty() {
        projections.push(dense(
            "indexer.weights",
            "attn.indexer.weights_proj",
            Input::Normed,
            s.indexer.heads,
            s.hidden,
        ));
    }
    projections
}

/// Whether sublayer `(layer, sublayer)` of `g` runs a dense F32 projection
/// of its normed rows: every MoE's router, and an attention's compressor or
/// index weights.
pub(crate) fn widens(g: &Geometry, layer: u32, sublayer: Sublayer) -> bool {
    match sublayer {
        Sublayer::Mlp => true,
        Sublayer::Attention => matches!(g.layer_role(layer), LayerRole::Full | LayerRole::Reindex),
    }
}

/// The dense F32 projections of an instance and the widening of the
/// normed rows they read.
/// The views of the router's certified projection, which reads the
/// MoE-input norm's rows and certificate (router-in-norm, launched through
/// `crate::Kernels::router_in_norm_router`).
pub(crate) struct CertifiedRouterViews {
    /// The norm's BF16 rows, which the SIMT path widens itself.
    pub(crate) normed: View,
    pub(crate) certified: View,
    pub(crate) weight: View,
    pub(crate) output: View,
    pub(crate) live_rows: View,
    pub(crate) weight_domain: View,
}

pub(crate) struct ExactF32Ops {
    widen: Prepared<WidenRowsBuffers<View>>,
    projections: Vec<(F32Dense, Prepared<ProjectionBuffers<View>>)>,
    cached_router: Option<cached::Wide>,
}

/// A step's spans of the normed rows' widening.
pub(crate) struct WidenViews {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl ExactF32Ops {
    pub(crate) fn weight_op(&self, name: &str) -> Option<&Prepared<ProjectionBuffers<View>>> {
        self.projections.iter().find_map(|(dense, op)| {
            name.ends_with(&format!(".{}.weight", dense.tensor))
                .then_some(op)
        })
    }
    pub(crate) fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: &Shape,
        asks: &Asks,
        experts: u32,
    ) -> Result<ExactF32Ops, Refused> {
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let widen = asks.ask("f32.widen", every, |need| {
            own.widen_rows(
                &WidenRows {
                    width: shape.dims().hidden,
                },
                need,
            )
        })?;
        let draft_backend = if experts == shape.geometry.shape.dspark.experts
            && experts != shape.geometry.shape.moe.experts
        {
            crate::Kernels::drafter_f32_backend(shape)?
        } else {
            None
        };
        let projection_backend = draft_backend
            .as_ref()
            .map_or(backend, |b| b as &dyn Backend);
        let projections = projections(shape.geometry)
            .into_iter()
            .map(|mut dense| {
                if dense.id == "moe.router" {
                    dense.n = experts;
                }
                let op = operation(dense.id, dense.tensor, query(dense.n, dense.k))?;
                let prepared = projection_backend.projection(&op, USE, every)?.handle;
                let prepared = if dense.id == "moe.router" {
                    crate::Kernels::prefill_router(shape, experts, prepared)?
                } else {
                    prepared
                };
                asks.record(format!("f32.{}", dense.id), every, prepared.info());
                Ok((dense, prepared))
            })
            .collect::<Result<_, Refused>>()?;
        let cached_router = cached::prepare(
            shape,
            asks,
            "f32.moe.router wide verify",
            experts,
            shape.dims().hidden,
            experts == shape.geometry.shape.moe.experts,
        )?;
        Ok(ExactF32Ops {
            widen,
            projections,
            cached_router,
        })
    }

    pub(crate) fn cached_router(&self, phase: crate::program::Phase) -> Option<&cached::Wide> {
        self.cached_router.as_ref().filter(|op| op.for_phase(phase))
    }

    /// The widening of `rows` normed rows to F32.
    pub(crate) fn widen(&self, rows: u32, views: &WidenViews) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = WidenRowsBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(&self.widen, &buffers, Counts::rows(rows))
    }

    /// Each weight view of `g` a projection reads, with the codes layout
    /// its op reads it in.
    pub(crate) fn weight_layouts(&self, g: &Geometry) -> Result<Vec<(String, String)>, String> {
        let mut views = Vec::new();
        for (dense, op) in &self.projections {
            let layout = codes_layout(op.info())?;
            for layer in 0..g.layers() {
                views.push((
                    format!("layers.{layer}.{}.weight", dense.tensor),
                    layout.clone(),
                ));
            }
        }
        Ok(views)
    }

    /// The dense F32 projection `id`, if the instance has it.
    pub(crate) fn projection(
        &self,
        id: &str,
    ) -> Option<&(F32Dense, Prepared<ProjectionBuffers<View>>)> {
        self.projections.iter().find(|(dense, _)| dense.id == id)
    }
}

/// The codes layout a rank index names for the BF16 weight `info` reads
/// (`OpInfo::weight_layouts`, role `weights`): the format registry's
/// layout whose weight layout the op declares.
pub(crate) fn codes_layout(info: &OpInfo) -> Result<String, String> {
    let (_, layout) = info
        .weight_layouts
        .iter()
        .find(|(role, _)| role == "weights")
        .ok_or_else(|| format!("{} declares no weight layout", info.implementation))?;
    let format = registry::by_format(WeightFormat::Bf16).map_err(|error| error.to_string())?;
    for family in [KernelFamily::Row, KernelFamily::Dense] {
        let choice = format
            .layout(family, None)
            .map_err(|error| error.to_string())?;
        if choice.weight == *layout {
            return Ok(choice.codes.to_owned());
        }
    }
    Err(format!("BF16 has no codes layout for {layout:?}"))
}
