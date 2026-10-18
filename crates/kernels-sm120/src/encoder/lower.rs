//! What each shared variant runs, as kernels on the rows it reads. A model's
//! registered variants answer through `Registered`.

use encoder_spec::components::attention::{Mask, Sink};
use encoder_spec::components::rotary::RotaryComponent as _;
use encoder_spec::components::{
    Boundary, Cast, Conv1d, Data, Element, Gelu, Layout, Linear, Merge, Mlp, Norm, Position,
    RotarySlot, StageOp, Transformer, Variant,
};
use encoder_spec::policy::Precision;
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GeluMode, GemmQuery, MultimodalQuery, RotaryStyle,
    SwigluColumns, WeightFormat,
};

use super::{Kernel, OVERRIDE_ROWS, Registered, Slot};

/// The ids of the linear variants a composite stage's GEMMs run as.
const LINEAR_BF16: &str = "linear.bf16.v1";
const LINEAR_BF16_ROUND_LINEAR: &str = "linear.bf16-round-linear.v1";
const LINEAR_PAIRS: &str = "linear.pairs.v1";

/// One GEMM's shape and epilogue.
#[derive(Clone, Copy, Default)]
struct Gemm {
    n: u32,
    k: u32,
    bias: bool,
    gelu: Option<Gelu>,
    swiglu: bool,
    round_linear: bool,
    residual: bool,
    output: Option<Element>,
}

fn gelu_mode(gelu: Option<Gelu>) -> Option<GeluMode> {
    gelu.map(|gelu| match gelu {
        Gelu::Erf => GeluMode::Erf,
        Gelu::Tanh => GeluMode::Tanh,
    })
}

fn dtype(element: Element) -> DType {
    match element {
        Element::F32 => DType::F32,
        Element::Bf16 => DType::Bf16,
    }
}

fn narrow(value: u64, what: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{what} {value} does not fit 32 bits"))
}

/// Walks one declaration's stages, collecting the kernels they run and the
/// slot each fills.
pub(super) struct Lowering<'a> {
    pub precision: Precision,
    pub capacity: u32,
    pub registered: &'a dyn Registered,
    pub kernels: Vec<(Slot, Kernel)>,
    /// The slot the next kernels fill.
    pub slot: Slot,
}

impl Lowering<'_> {
    /// Names the part (and block) of the current stage the next kernels run.
    fn at(&mut self, part: &'static str, block: Option<u32>) {
        self.slot.part = part;
        self.slot.block = block;
    }

    fn push(&mut self, variant: &str, role: Option<&'static str>, query: MultimodalQuery) {
        self.kernels.push((
            self.slot.clone(),
            Kernel {
                variant: variant.to_owned(),
                role,
                query,
            },
        ));
    }

    fn extend(&mut self, kernels: Vec<Kernel>) {
        let slot = self.slot.clone();
        self.kernels
            .extend(kernels.into_iter().map(|kernel| (slot.clone(), kernel)));
    }

    /// A GEMM as the linear variant the policy and the rounding pick.
    fn gemm(&mut self, gemm: Gemm) -> Result<(), String> {
        let pairs = self.precision == Precision::F32Pairs;
        let mut query = GemmQuery {
            m_capacity: self.capacity,
            n: gemm.n,
            k: gemm.k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::Bf16,
            output: DType::Bf16,
            epilogues: Epilogues {
                bias: gemm.bias,
                gelu: gelu_mode(gemm.gelu),
                swiglu: gemm.swiglu.then_some(SwigluColumns::ConcatenatedHalves),
                ..Epilogues::default()
            },
            grouped: None,
        };
        let (variant, query) = match (pairs, gemm.round_linear) {
            (false, false) => (LINEAR_BF16, MultimodalQuery::Gemm(query)),
            (false, true) => (LINEAR_BF16_ROUND_LINEAR, MultimodalQuery::GemmBf16(query)),
            (true, false) if !gemm.swiglu => {
                query.output = dtype(gemm.output.unwrap_or(Element::F32));
                query.epilogues.residual_add = gemm.residual;
                (LINEAR_PAIRS, MultimodalQuery::GemmPairs(query))
            }
            (true, _) => {
                return Err("linear.pairs.v1 has no SwiGLU or rounded-linear epilogue".into());
            }
        };
        self.push(variant, None, query);
        Ok(())
    }

    /// The kernels of stage `op` reading `input`.
    pub fn stage(&mut self, op: &StageOp, input: &Boundary) -> Result<(), String> {
        let width = || input.width();
        match op {
            StageOp::Linear(linear) => {
                self.at("linear", None);
                let params = linear.params();
                let gemm = Gemm {
                    n: params.out,
                    k: width()?,
                    bias: params.bias,
                    gelu: params.gelu,
                    round_linear: matches!(linear, Linear::Bf16RoundLinear(_)),
                    output: params.output,
                    ..Gemm::default()
                };
                self.gemm(gemm)
            }
            StageOp::Conv1d(conv) => self.conv1d(conv, input),
            StageOp::Position(Position::LearnedGridBilinearF32(params)) => {
                self.at("position", None);
                let table_rows = narrow(
                    u64::from(params.table[0]) * u64::from(params.table[1]),
                    "table rows",
                )?;
                self.push(
                    &op.id(),
                    None,
                    MultimodalQuery::InterpolatedPositionAddF32 {
                        width: width()?,
                        table_rows,
                        capacity: self.capacity,
                    },
                );
                Ok(())
            }
            StageOp::Norm(norm) => {
                self.at("norm", None);
                self.norm(norm, width()?)
            }
            StageOp::Transformer(transformer) => self.transformer(transformer, width()?),
            // A reinterpretation of consecutive rows, and copies: no kernel.
            StageOp::Merge(Merge::Consecutive(_)) | StageOp::Group(_) => Ok(()),
            StageOp::Registered(registered) => {
                // Each kernel's slot part is its role, else `registered`.
                let kernels =
                    self.registered
                        .stage(registered.component.as_ref(), input, self.capacity)?;
                for kernel in kernels {
                    self.at(kernel.role.unwrap_or("registered"), None);
                    self.kernels.push((self.slot.clone(), kernel));
                }
                Ok(())
            }
            StageOp::Projector(projector) => {
                let params = projector.params();
                let width = width()?;
                self.at("norm", None);
                self.norm(&params.norm, width)?;
                match params.gelu {
                    Gelu::Erf => {
                        self.at("gelu", None);
                        self.push(
                            &op.id(),
                            Some("gelu"),
                            MultimodalQuery::GeluRowsF32 {
                                width,
                                capacity: self.capacity,
                            },
                        );
                    }
                    Gelu::Tanh => {
                        return Err(format!("`{}`: the GELU row pass is erf only", op.id()));
                    }
                }
                self.mlp(&params.mlp, width, false, params.output)
            }
            StageOp::Pad(pad) => {
                self.at("pad", None);
                self.push(
                    &op.id(),
                    None,
                    MultimodalQuery::PadColumnsF32 {
                        input: width()?,
                        output: pad.columns(),
                        capacity: self.capacity,
                    },
                );
                Ok(())
            }
            StageOp::Merge(Merge::SpaceToDepth(params)) => {
                if self.precision != Precision::F32Pairs
                    || !matches!(
                        input.data,
                        Data::Rows {
                            element: Element::F32,
                            ..
                        }
                    )
                {
                    return Err(format!(
                        "`{}` gathers F32 rows under f32-pairs.v1 only",
                        op.id()
                    ));
                }
                self.at("gather", None);
                self.push(
                    &op.id(),
                    None,
                    MultimodalQuery::GatherRowsF32 {
                        width: width()?,
                        group: narrow(
                            u64::from(params.factor) * u64::from(params.factor),
                            "space-to-depth cell",
                        )?,
                        capacity: self.capacity,
                    },
                );
                Ok(())
            }
            StageOp::Cast(Cast::Bf16(_)) => {
                if self.precision != Precision::F32Pairs
                    || !matches!(
                        input.data,
                        Data::Rows {
                            element: Element::F32,
                            ..
                        }
                    )
                {
                    return Err(format!(
                        "`{}` rounds F32 rows under f32-pairs.v1 only",
                        op.id()
                    ));
                }
                self.at("cast", None);
                self.push(
                    &op.id(),
                    None,
                    MultimodalQuery::CastBf16F32 {
                        width: width()?,
                        capacity: self.capacity,
                    },
                );
                Ok(())
            }
            StageOp::Unresolved(_) => Err(format!("`{}` has no encoder kernel yet", op.id())),
        }
    }

    /// The output layout's kernels: the language model's prefill writes the
    /// encoder rows over its token rows.
    pub fn layout(&mut self, layout: &Layout) -> Result<(), String> {
        // The encoder runtime writes a learned-delimiter span whole into the
        // arena (its cells at their placed rows, the learned rows between
        // them), so both layouts inject arena rows by the same override.
        let (Layout::Tokens(params) | Layout::LearnedDelimiters(params)) = layout;
        self.at("layout", None);
        self.push(
            &layout.id(),
            None,
            MultimodalQuery::EmbeddingOverride {
                width: params.hidden,
                capacity: OVERRIDE_ROWS,
            },
        );
        Ok(())
    }

    fn conv1d(&mut self, conv: &Conv1d, input: &Boundary) -> Result<(), String> {
        let params = conv.params();
        let Data::Rows { width, element } = input.data else {
            return Err("conv1d reads rows".into());
        };
        let pairs = self.precision == Precision::F32Pairs;
        if pairs && params.gelu_rows.is_some() {
            return Err(format!(
                "`{}`: under f32-pairs.v1 the GELU is the GEMM's (`gelu`), not a row pass",
                conv.id()
            ));
        }
        self.at("unfold", None);
        let unfold = if pairs {
            if element != Element::F32 {
                return Err(format!("`{}` reads F32 rows under f32-pairs.v1", conv.id()));
            }
            MultimodalQuery::Im2col1dF32 {
                channels: width,
                kernel: params.kernel,
                stride: params.stride,
                pad_left: params.pad[0],
                pad_right: params.pad[1],
                capacity: self.capacity,
            }
        } else {
            MultimodalQuery::Im2col1d {
                channels: width,
                kernel: params.kernel,
                stride: params.stride,
                pad_left: params.pad[0],
                pad_right: params.pad[1],
                input: dtype(element),
                capacity: self.capacity,
            }
        };
        self.push(&conv.id(), Some("unfold"), unfold);
        self.at("linear", None);
        self.gemm(Gemm {
            n: params.out,
            k: narrow(u64::from(params.kernel) * u64::from(width), "conv1d K")?,
            bias: params.bias,
            gelu: params.gelu,
            round_linear: params.round_linear,
            ..Gemm::default()
        })?;
        match params.gelu_rows {
            None => Ok(()),
            Some(Gelu::Erf) => {
                self.at("gelu_rows", None);
                self.push(
                    &conv.id(),
                    Some("gelu_rows"),
                    MultimodalQuery::GeluRows {
                        width: params.out,
                        capacity: self.capacity,
                    },
                );
                Ok(())
            }
            Some(Gelu::Tanh) => Err(format!("`{}`: GELU rows are erf only", conv.id())),
        }
    }

    fn norm(&mut self, norm: &Norm, width: u32) -> Result<(), String> {
        let eps_bits = (norm.params().eps as f32).to_bits();
        let capacity = self.capacity;
        let query = match norm {
            Norm::Layer(_) => MultimodalQuery::LayerNorm {
                width,
                eps_bits,
                capacity,
            },
            Norm::LayerF32(_) => MultimodalQuery::LayerNormF32 {
                width,
                eps_bits,
                capacity,
            },
            // BF16 RMS rows run the row engine's plain-weight RMS entries
            // (`crate::row`) until the shared encoder runtime lowers them.
            Norm::Rms(_) => return Ok(()),
            Norm::RmsF32(_) => MultimodalQuery::RmsNormF32 {
                width,
                eps_bits,
                capacity,
            },
        };
        self.push(&norm.id(), None, query);
        Ok(())
    }

    /// An MLP slot over F32 rows of `width` under `f32-pairs.v1`: the up
    /// GEMM (gate and up halves for a SwiGLU, then its row pass), then the
    /// down GEMM, added in place to the residual stream (`residual`) or
    /// written as `output`. The transformer's `mlp.gelu.v1` and
    /// `mlp.swiglu.v1` keep their own lowering.
    fn mlp(
        &mut self,
        mlp: &Mlp,
        width: u32,
        residual: bool,
        output: Option<Element>,
    ) -> Result<(), String> {
        let capacity = self.capacity;
        let inner = narrow(mlp.inner(), "MLP width")?;
        self.at("up", None);
        match mlp {
            Mlp::Gelu(params) => self.gemm(Gemm {
                n: inner,
                k: width,
                bias: params.bias,
                gelu: Some(params.gelu),
                ..Gemm::default()
            })?,
            Mlp::Swiglu(_) | Mlp::SwigluClamped(_) => {
                self.gemm(Gemm {
                    n: narrow(2 * u64::from(inner), "SwiGLU width")?,
                    k: width,
                    bias: mlp.bias(),
                    ..Gemm::default()
                })?;
                let query = match mlp {
                    Mlp::SwigluClamped(params) => MultimodalQuery::SwigluClampedF32 {
                        hidden: inner,
                        limit_bits: (params.limit as f32).to_bits(),
                        capacity,
                    },
                    _ => MultimodalQuery::SwigluF32 {
                        hidden: inner,
                        capacity,
                    },
                };
                self.at("swiglu", None);
                self.push(&mlp.id(), Some("swiglu"), query);
            }
        }
        self.at("down", None);
        self.gemm(Gemm {
            n: width,
            k: inner,
            bias: mlp.bias(),
            residual,
            output,
            ..Gemm::default()
        })
    }

    fn transformer(&mut self, transformer: &Transformer, width: u32) -> Result<(), String> {
        let params = transformer.params();
        let attention = params.attention.params();
        let qk_norm_eps = params.attention.qk_norm_eps()?;
        let pairs = params.attention.pairs();
        let qkv = narrow(params.attention.qkv_width()?, "qkv width")?;
        let heads = narrow(params.attention.attention_width()?, "attention width")?;
        let capacity = self.capacity;
        self.at("norm", None);
        self.norm(&params.norm, width)?;
        self.at("qkv", None);
        self.gemm(Gemm {
            n: qkv,
            k: width,
            bias: attention.qkv_bias,
            ..Gemm::default()
        })?;
        if let Some(eps) = qk_norm_eps {
            self.at("qk_norm", None);
            self.push(
                &params.attention.id(),
                Some("qk_norm"),
                MultimodalQuery::QkRmsNormF32 {
                    q_heads: attention.q_heads,
                    kv_heads: attention.kv_heads,
                    head_dim: attention.head_dim,
                    eps_bits: eps.to_bits(),
                    capacity,
                },
            );
        }
        if !pairs {
            self.at("split", None);
            self.push(
                &params.attention.id(),
                Some("split"),
                MultimodalQuery::SplitQkv {
                    q_width: heads,
                    kv_width: attention.kv_heads * attention.head_dim,
                    capacity,
                },
            );
        }
        self.at("rotary", None);
        match &params.rotary {
            Some(RotarySlot::Shared(rotary)) => {
                if !pairs {
                    return Err(format!("`{}` rotates pair rows only", rotary.id()));
                }
                let rotary_dim = narrow(rotary.rotary_dims()?, "rotary dims")?;
                self.push(
                    &rotary.id(),
                    None,
                    MultimodalQuery::RopeQkvPairs {
                        q_heads: attention.q_heads,
                        kv_heads: attention.kv_heads,
                        head_dim: attention.head_dim,
                        rotary_dim,
                        style: RotaryStyle::Neox,
                        capacity,
                    },
                );
            }
            Some(RotarySlot::Registered(registered)) => {
                let kernels = self.registered.rotary(
                    registered.component.as_ref(),
                    &params.attention,
                    capacity,
                )?;
                self.extend(kernels);
            }
            Some(RotarySlot::Unresolved(unresolved)) => {
                return Err(format!("`{}` is not registered", unresolved.id));
            }
            None if pairs => {
                return Err(
                    "attention.varlen.pairs.v1 reads the pair rows the rotary kernel writes; \
                     it needs a rotary slot"
                        .into(),
                );
            }
            None => {}
        }
        // One attention entry per distinct mask, in block order.
        for block in 0..params.blocks {
            let (mask, sink) = transformer.attention_at(block);
            let (causal, window_left, window_right) = match mask {
                Mask::Full => (false, None, None),
                Mask::Causal => (true, None, None),
                Mask::Window([left, right]) => (false, Some(left), Some(right)),
                Mask::CausalWindow(left) => (true, Some(left), None),
            };
            let key0_sink = sink == Sink::Key0;
            let query = if pairs {
                MultimodalQuery::ContiguousAttentionPairs {
                    q_heads: attention.q_heads,
                    kv_heads: attention.kv_heads,
                    head_dim: attention.head_dim,
                    causal,
                    window_left,
                    window_right,
                    key0_sink,
                    capacity,
                }
            } else {
                MultimodalQuery::ContiguousAttention {
                    q_heads: attention.q_heads,
                    kv_heads: attention.kv_heads,
                    head_dim: attention.head_dim,
                    causal,
                    window_left,
                    window_right,
                    key0_sink,
                    capacity,
                }
            };
            self.at("attention", Some(block));
            self.push(&params.attention.id(), None, query);
        }
        // Under f32-pairs.v1 the residual adds are the projections' epilogue.
        self.at("proj", None);
        self.gemm(Gemm {
            n: width,
            k: heads,
            bias: attention.proj_bias,
            residual: pairs,
            ..Gemm::default()
        })?;
        let residual = |lowering: &mut Self| {
            lowering.push(
                &transformer.id(),
                Some("residual_add"),
                MultimodalQuery::ResidualAdd { width, capacity },
            )
        };
        if !pairs {
            self.at("residual_add", None);
            residual(self);
        }
        for reorder in &params.reorder {
            self.at("reorder", None);
            let query = if pairs {
                MultimodalQuery::PermuteF32 { width, capacity }
            } else {
                MultimodalQuery::Permute { width, capacity }
            };
            self.push(&reorder.reorder.id(), None, query);
        }
        match &params.mlp {
            Mlp::Gelu(mlp) => {
                let hidden = mlp.pad_to.unwrap_or(mlp.hidden);
                self.at("up", None);
                self.gemm(Gemm {
                    n: hidden,
                    k: width,
                    bias: mlp.bias,
                    gelu: Some(mlp.gelu),
                    round_linear: mlp.round_linear,
                    ..Gemm::default()
                })?;
                self.at("down", None);
                self.gemm(Gemm {
                    n: width,
                    k: hidden,
                    bias: mlp.bias,
                    residual: pairs,
                    ..Gemm::default()
                })?;
            }
            Mlp::Swiglu(mlp) => {
                let gate_up = narrow(2 * u64::from(mlp.hidden), "SwiGLU width")?;
                self.at("up", None);
                // Under f32-pairs.v1 the GEMM writes the F32 gate and up
                // columns and a row pass takes their SwiGLU.
                self.gemm(Gemm {
                    n: gate_up,
                    k: width,
                    bias: mlp.bias,
                    swiglu: !pairs,
                    round_linear: mlp.round_linear,
                    ..Gemm::default()
                })?;
                if pairs {
                    self.at("swiglu", None);
                    self.push(
                        &params.mlp.id(),
                        Some("swiglu"),
                        MultimodalQuery::SwigluF32 {
                            hidden: mlp.hidden,
                            capacity,
                        },
                    );
                }
                self.at("down", None);
                self.gemm(Gemm {
                    n: width,
                    k: mlp.hidden,
                    bias: mlp.bias,
                    residual: pairs,
                    ..Gemm::default()
                })?;
            }
            Mlp::SwigluClamped(_) => self.mlp(&params.mlp, width, true, None)?,
        }
        if !params.carry.is_empty() && pairs {
            self.at("carry", None);
            self.push(
                &transformer.id(),
                Some("carry"),
                MultimodalQuery::ResidualAddF32 { width, capacity },
            );
        }
        Ok(())
    }
}
