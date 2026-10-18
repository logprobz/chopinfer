//! Lowering: one declaration's stages as a canonical launch list over the
//! regions of its buffer plan and the weight views of its encoder image,
//! fixed at create. Per record only the live row counts, the side inputs
//! and the attention sequence count change; every kernel grids for its
//! capacity and reads its live rows on the device.
//!
//! The regions each stage reads and writes are the buffer plan's (its uses
//! name them), so the plan's aliasing is the launch list's. The rows the
//! last stage writes (`Operand::Output`) sit at the start of its region.
//!
//! `f32-pairs.v1` keeps F32 rows in `stream`, `a` and `b`, with residual
//! adds in the projection and down GEMMs. Staged declarations upload
//! per-block side inputs in command order and reorder through `a`.

use std::collections::BTreeMap;

use encoder_spec::buffers::{self, BufferPlan};
use encoder_spec::components::attention::Sink;
use encoder_spec::components::rotary::{Rotary, Table};
use encoder_spec::components::{
    Data, Item, Layout, Linear, Merge, Mlp, Norm, RotarySlot, StageOp, Transformer,
};
use encoder_spec::placement::{Delimiter, Piece};
use encoder_spec::weights::Source;
use encoder_spec::{Declaration, ImageLayout};
use kernel_core::allocation::Traversal;
use kernel_core::backend::{Backend, Counts, Need, Prepared, View};
use kernel_core::contract::MultimodalQuery;
use kernel_core::multimodal::{
    EmbeddingOverrideBuffers, EncoderAttentionBuffers, EncoderGemmBuffers, EncoderHeadBuffers,
    EncoderRowBuffers, EncoderStage,
};

use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use super::registered::{Bind, Registered};
use super::side::{Input, Order};

/// What a launch argument is bound to.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Operand {
    /// A whole region of the buffer plan, shared by its uses.
    Region(&'static str),
    /// A whole side-input region of the stage that owns it.
    Side(&'static str, usize),
    /// The rows the last stage writes, at the start of its region.
    Output,
    /// The live row count of slot `n`: a u32 at byte `4 n` of `counts`.
    Count(usize),
    /// A view of the encoder image.
    Weight(String),
    Null,
    /// The run's live attention sequences of a transformer stage.
    Sequences(usize),
    U32(u32),
    /// The run's rows of the payload region (staged declarations).
    Payload,
    /// The run's rows (`Run::output`, of these bytes each) of a region a
    /// later stage joins the runs from.
    RunOutput(&'static str, u64),
}

/// One command of a stage's part of a run.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Op {
    /// `Lowered::launches[n]`.
    Launch(usize),
    /// Stages the run's side input into a region, in command order.
    Upload(Input, &'static str),
    /// Copies the stage's live rows, `row_bytes` each, from one region to
    /// the start of another.
    Copy {
        from: &'static str,
        to: &'static str,
        row_bytes: u64,
    },
    /// Copies the joined rows a run gathers (`Run::gather`) to the start of
    /// a region, each run of consecutive source rows in one copy.
    Gather {
        from: &'static str,
        to: &'static str,
        row_bytes: u64,
    },
    /// A transformer block's output rows, in the rows' own order, read back
    /// as block tap `name` (`enc.<tower>.B<nn>.out`, blocks numbered across
    /// the declaration's transformers) when block taps are on; for a
    /// transformer's first block also its input (`B<nn>.in`).
    Tap {
        name: String,
        region: &'static str,
        row_bytes: u64,
    },
    /// The codes a quantizer stage writes, read back as `enc.<tower>.x.codes`
    /// when code taps are on: the step's written rows, or with `run_output`
    /// the run's output rows of a region a joining stage reads.
    CodesTap {
        name: String,
        region: &'static str,
        row_bytes: u64,
        run_output: bool,
    },
}

pub(super) struct Launch {
    /// The live count slot the kernel reads: its stage's input rows, or for
    /// the GEMM and row pass of a strided conv1d its output rows.
    pub count: usize,
    /// `Lowered::prepared[op]`.
    pub op: usize,
    /// Its buffers by role.
    pub roles: Box<Roles>,
}

/// A launch's buffers by the roles of its op's record.
#[derive(Clone)]
pub(super) enum Roles {
    Rows(EncoderRowBuffers<Operand>),
    Heads(EncoderHeadBuffers<Operand>),
    /// The attention's buffers; its live sequences are the run's of the
    /// transformer stage.
    Attention(EncoderAttentionBuffers<Operand>, usize),
    Gemm(EncoderGemmBuffers<Operand>),
    /// A model's own op, its buffers by the names its registered stage
    /// gives.
    Own(Named<Operand>),
}

/// An op prepared for a slot, by the record its buffers fill.
pub(super) enum Ready {
    Rows(Prepared<EncoderRowBuffers<View>>),
    Heads(Prepared<EncoderHeadBuffers<View>>),
    Attention(Prepared<EncoderAttentionBuffers<View>>),
    Gemm(Prepared<EncoderGemmBuffers<View>>),
    Own(Prepared<Named<View>>),
}

/// The language model's prefill embedding override.
pub(super) type Inject = std::sync::Arc<Prepared<EmbeddingOverrideBuffers<View>>>;

/// Buffers by name.
pub type Named<V> = Vec<(&'static str, V)>;

/// A declaration's slot: its stage, part and block.
pub(super) type SlotKey = (usize, &'static str, Option<u32>);

/// What a declaration's slot runs: a shared encoder stage's query, which
/// the runtime prepares through the backend's multimodal methods, or a
/// model-owned op of a registered stage, which binds its buffers by the
/// names the stage's `registered::Launch` gives.
pub enum SlotOp {
    Query(MultimodalQuery),
    Own(Prepared<Named<View>>),
}

/// Resolves a declaration's slot (stage, part, block) into what it runs.
/// The queries are the ones the backend's catalog was built from.
pub type Slots<'a> = dyn Fn(usize, &'static str, Option<u32>) -> Result<SlotOp, String> + 'a;

/// How a declaration's slots resolve.
#[derive(Clone, Copy)]
pub(super) struct Resolver<'a> {
    backend: &'a dyn Backend,
    slots: &'a Slots<'a>,
    /// Rows of one prefill step: the override's capacity.
    prefill_rows: u32,
}

/// One declaration lowered onto its buffer plan.
pub struct Lowered {
    pub declaration: Declaration,
    /// The ops prepared for its slots (stage, part, block).
    pub(super) prepared: Vec<(SlotKey, Ready)>,
    pub(super) launches: Vec<Launch>,
    /// Per stage, its commands in order.
    pub(super) ops: Vec<Vec<Op>>,
    /// Bytes staged once per record after the payload, by region.
    pub(super) record: Vec<(&'static str, Vec<u8>)>,
    /// Whether the run's uploads are staged in command order and the live
    /// row count is uploaded as it changes (`Declaration::staged`), rather
    /// than written before the first launch.
    pub(super) staged: bool,
    /// The full-run rows of each live count slot.
    pub(super) counts: Vec<u64>,
    /// Per stage, the count slots of the rows it reads and writes.
    pub(super) stage_counts: Vec<[usize; 2]>,
    /// The region the payload is uploaded to, and the bytes of one payload
    /// row.
    pub(super) payload: &'static str,
    pub(super) payload_row_bytes: u64,
    /// The region of the output rows and their bytes in a full run.
    pub(super) output: (&'static str, u64),
    /// The region a joining stage gathers from and the stage that writes
    /// it, for staged declarations with one.
    pub(super) joined: Option<(&'static str, usize)>,
    /// The layout's embedding override, run in the language model's prefill.
    pub(super) inject: Inject,
    /// Under `layout.learned-delimiters.v1`, the views of its learned start,
    /// newline and end rows ([`Delimiter`] order), each one BF16 row.
    pub(super) delimiters: Option<[String; 3]>,
}

/// What one placed row of an item's span is written from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacedSource {
    /// `rows` encoder output rows from row `first` of the run's output.
    Output { first: u64, rows: u64 },
    /// One learned delimiter row.
    Learned(Delimiter),
}

/// One copy into an item's span: `source` to placed rows from `placed`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacedCopy {
    pub source: PlacedSource,
    pub placed: u64,
}

/// The copies that write a run's part of an item's span under `layout`,
/// from the item's rows after the last stage (`encoded`): the run's output
/// rows (item rows `output`) at their placed rows in `Layout::place` order,
/// and, in the run whose output starts the item, every learned delimiter
/// row. Under `layout.tokens.v1` the span is the encoder rows in order.
pub fn placed_copies(
    layout: &Layout,
    encoded: Item,
    output: std::ops::Range<u64>,
) -> Result<Vec<PlacedCopy>, String> {
    let placement = layout.place(encoded, 0)?;
    let mut copies = Vec::new();
    let mut placed = 0;
    for piece in placement.pieces {
        match piece {
            Piece::Rows { media_row, rows } => {
                let first = media_row.max(output.start);
                let end = (media_row + rows).min(output.end);
                if first < end {
                    copies.push(PlacedCopy {
                        source: PlacedSource::Output {
                            first: first - output.start,
                            rows: end - first,
                        },
                        placed: placed + (first - media_row),
                    });
                }
            }
            Piece::Learned(delimiter) => {
                if output.start == 0 {
                    copies.push(PlacedCopy {
                        source: PlacedSource::Learned(delimiter),
                        placed,
                    });
                }
            }
        }
        placed += piece.rows();
    }
    Ok(copies)
}

/// The views of the output layout's learned start, newline and end rows,
/// named as the stages' views are (the checkpoint prefix and the tensor).
fn delimiter_views(declaration: &Declaration) -> Result<[String; 3], String> {
    let view = |role: &str| {
        let source = declaration
            .output
            .weights
            .get(role)
            .ok_or_else(|| format!("the layout binds no `{role}` row"))?;
        let tensor = match (declaration.image_layout, source) {
            (ImageLayout::ViewOrder, Source::Tensor(tensor)) => tensor,
            (ImageLayout::ViewOrder, _) => {
                return Err(format!("`{role}`: view-order.v1 binds single tensors only"));
            }
            (ImageLayout::ProgramOrderStacked, source) => {
                first_tensor(source).ok_or_else(|| {
                    format!("`{role}`: program-order-stacked.v1 names views by their first tensor")
                })?
            }
        };
        Ok(format!(
            "{}{}",
            declaration.checkpoint.prefix, tensor.tensor
        ))
    };
    Ok([view("start")?, view("newline")?, view("end")?])
}

/// The region whose uses include `what` of `encoder` at `stage`.
/// The one region whose uses include `what` of `encoder` at `stage`.
fn region(
    plan: &BufferPlan,
    encoder: &str,
    stage: Option<usize>,
    what: &str,
) -> Result<&'static str, String> {
    let mut found = plan.regions.iter().filter(|region| {
        region
            .uses
            .iter()
            .any(|at| at.encoder == encoder && at.stage == stage && at.what == what)
    });
    match (found.next(), found.next()) {
        (Some(region), None) => Ok(region.name),
        (None, _) => Err(format!(
            "the buffer plan holds no `{what}` of stage {stage:?}"
        )),
        (Some(first), Some(second)) => Err(format!(
            "the buffer plan holds `{what}` of stage {stage:?} in both `{}` and `{}`",
            first.name, second.name
        )),
    }
}

/// Builds the launch list, preparing each slot's op once.
struct Builder<'a> {
    declaration: &'a Declaration,
    resolver: Resolver<'a>,
    prepared: Vec<(SlotKey, Ready)>,
    launches: Vec<Launch>,
    ops: Vec<Vec<Op>>,
    stage_counts: Vec<[usize; 2]>,
    staged: bool,
    record: Vec<(&'static str, Vec<u8>)>,
    joined: Option<(&'static str, usize)>,
    /// Transformer blocks lowered so far, across stages.
    blocks: u32,
}

/// The tensor whose name a source's image view takes: the source's own,
/// or the first of a concatenation's.
fn first_tensor(source: &Source) -> Option<&encoder_spec::weights::Tensor> {
    match source {
        Source::Tensor(tensor) => Some(tensor),
        Source::Concat(concat) => concat.concat_rows.first().and_then(first_tensor),
        Source::ConcatRange(_) | Source::Zeros(_) => None,
    }
}

impl Builder<'_> {
    /// The view of `role` of `stage` (at `block`), or a null pointer when
    /// the stage (or that block) has no such role.
    fn weight(&self, stage: usize, role: &str, block: Option<u32>) -> Result<Operand, String> {
        let Some(source) = self.declaration.stages[stage].weights.get(role) else {
            return Ok(Operand::Null);
        };
        let tensor = match (self.declaration.image_layout, source) {
            (ImageLayout::ViewOrder, Source::Tensor(tensor)) => tensor,
            (ImageLayout::ViewOrder, _) => {
                return Err(format!("`{role}`: view-order.v1 binds single tensors only"));
            }
            (ImageLayout::ProgramOrderStacked, source) => {
                first_tensor(source).ok_or_else(|| {
                    format!("`{role}`: program-order-stacked.v1 names views by their first tensor")
                })?
            }
        };
        if let (Some(set), Some(block)) = (&tensor.blocks, block)
            && !set.contains(block)
        {
            return Ok(Operand::Null);
        }
        let name = match block {
            Some(block) => tensor.tensor.replace("{i}", &block.to_string()),
            None => tensor.tensor.clone(),
        };
        Ok(Operand::Weight(format!(
            "{}{name}",
            self.declaration.checkpoint.prefix
        )))
    }

    /// The bias of a GEMM whose weights are `<role>weight`. A
    /// `program-order-stacked.v1` image holds one for every GEMM.
    fn bias(&self, stage: usize, role: &str, block: Option<u32>) -> Result<Operand, String> {
        if self.declaration.image_layout == ImageLayout::ViewOrder {
            return self.weight(stage, &format!("{role}bias"), block);
        }
        match self.weight(stage, &format!("{role}weight"), block)? {
            Operand::Weight(name) => name
                .strip_suffix(".weight")
                .map(|stem| Operand::Weight(format!("{stem}.bias")))
                .ok_or_else(|| format!("`{role}weight`: {name} names no bias view")),
            other => Ok(other),
        }
    }

    /// Adds the launch of slot (`stage`, `part`, `block`) with its operands
    /// named by the roles of its op's record.
    fn launch(
        &mut self,
        stage: usize,
        part: &'static str,
        block: Option<u32>,
        named: Vec<(&'static str, Operand)>,
    ) -> Result<(), String> {
        let count = self.stage_counts[stage][0];
        self.launch_counting(stage, (part, block), named, count)
    }

    /// `launch`, the kernel reading the live rows of count slot `count`:
    /// the slot's op, prepared through the backend the first time a launch
    /// names the slot, and the launch's buffers by its record's roles.
    fn launch_counting(
        &mut self,
        stage: usize,
        (part, block): (&'static str, Option<u32>),
        named: Vec<(&'static str, Operand)>,
        count: usize,
    ) -> Result<(), String> {
        let Resolver { backend, slots, .. } = self.resolver;
        let slot = (stage, part, block);
        let at = format!("stage {stage} ({part})");
        let op = match self.prepared.iter().position(|(known, _)| *known == slot) {
            Some(op) => op,
            None => {
                let prepared = match slots(stage, part, block)? {
                    SlotOp::Query(query) => self
                        .prepare(backend, &query, &named)
                        .map_err(|e| format!("{at}: {e}"))?,
                    SlotOp::Own(op) => Ready::Own(op),
                };
                self.prepared.push((slot, prepared));
                self.prepared.len() - 1
            }
        };
        let roles = roles(&self.prepared[op].1, stage, named).map_err(|e| format!("{at}: {e}"))?;
        self.ops[stage].push(Op::Launch(self.launches.len()));
        self.launches.push(Launch {
            count,
            op,
            roles: Box::new(roles),
        });
        Ok(())
    }

    /// The op of `query` for the declaration's runs (`Need::rows` of its
    /// `capacity_rows`), after checking that the query's runtime values
    /// are the declaration's ones `named` carries.
    fn prepare(
        &self,
        backend: &dyn Backend,
        query: &MultimodalQuery,
        named: &[(&str, Operand)],
    ) -> Result<Ready, String> {
        let value = |name: &str| {
            named.iter().find_map(|(known, operand)| match operand {
                Operand::U32(value) if *known == name => Some(*value),
                _ => None,
            })
        };
        let carried = match *query {
            MultimodalQuery::LayerNormF32 { eps_bits, .. }
            | MultimodalQuery::RmsNormF32 { eps_bits, .. }
            | MultimodalQuery::QkRmsNormF32 { eps_bits, .. } => Some(("eps_bits", eps_bits)),
            MultimodalQuery::SwigluClampedF32 { limit_bits, .. } => {
                Some(("limit_bits", limit_bits))
            }
            _ => None,
        };
        if let Some((name, bits)) = carried
            && value(name) != Some(bits)
        {
            return Err(format!("the query's `{name}` is not the declaration's"));
        }
        let need = Need::rows(self.declaration.input.capacity_rows);
        let refused = |refused: kernel_core::backend::Refused| format!("{refused:?}");
        let op = match EncoderStage::of(query).map_err(|refusal| refusal.0.to_owned())? {
            EncoderStage::Rows => Ready::Rows(backend.encoder_rows(query, need).map_err(refused)?),
            EncoderStage::Heads => {
                Ready::Heads(backend.encoder_heads(query, need).map_err(refused)?)
            }
            EncoderStage::Attention => {
                let scale = value("scale_bits").ok_or("an attention without its scale")?;
                Ready::Attention(
                    backend
                        .encoder_attention(query, f32::from_bits(scale), need)
                        .map_err(refused)?,
                )
            }
            EncoderStage::Gemm => Ready::Gemm(backend.encoder_gemm(query, need).map_err(refused)?),
            EncoderStage::Override => return Err("the override runs in the prefill".into()),
        };
        if op.info().workspace > 0 {
            return Err("the encoder runtime binds no backend workspace".into());
        }
        Ok(op)
    }

    /// The live row count a launch of `stage` reads.
    fn count(&self, stage: usize) -> Operand {
        self.count_slot(self.stage_counts[stage][0])
    }

    /// The live row count of count slot `slot`: under a staged declaration
    /// the one `count` the launch's slot is uploaded to.
    fn count_slot(&self, slot: usize) -> Operand {
        if self.staged {
            Operand::Region("count")
        } else {
            Operand::Count(slot)
        }
    }

    /// A norm of `stage` with the weights `<role>weight` (and
    /// `<role>bias`), reading `input` into `output`. BF16 RMS rows run the
    /// row engine's plain-weight RMS entries, without a residual.
    fn norm(
        &mut self,
        stage: usize,
        norm: &Norm,
        role: &str,
        block: Option<u32>,
        (input, output): (Operand, &'static str),
    ) -> Result<(), String> {
        let eps = Operand::U32((norm.params().eps as f32).to_bits());
        let weight = self.weight(stage, &format!("{role}weight"), block)?;
        let named = match norm {
            Norm::LayerF32(_) | Norm::Layer(_) => vec![
                ("input", input),
                ("weight", weight),
                ("bias", self.weight(stage, &format!("{role}bias"), block)?),
                ("output", Operand::Region(output)),
                ("live_rows", self.count(stage)),
                ("eps_bits", eps),
            ],
            Norm::Rms(_) => vec![
                ("x", input),
                ("weight", weight),
                ("residual", Operand::Null),
                ("residual_out", Operand::Null),
                ("y", Operand::Region(output)),
                ("live_rows", self.count(stage)),
                ("eps_bits", eps),
            ],
            Norm::RmsF32(_) => vec![
                ("input", input),
                ("weight", weight),
                ("output", Operand::Region(output)),
                ("live_rows", self.count(stage)),
                ("eps_bits", eps),
            ],
        };
        self.launch(stage, "norm", None, named)
    }

    /// A GEMM of part `part` reading `input` into `output` with the
    /// weights `<role>weight` and their bias.
    fn gemm(
        &mut self,
        (stage, part, block): (usize, &'static str, Option<u32>),
        role: &str,
        input: Operand,
        output: Operand,
    ) -> Result<(), String> {
        let named = vec![
            ("activation", input),
            (
                "weight",
                self.weight(stage, &format!("{role}weight"), block)?,
            ),
            ("bias", self.bias(stage, role, block)?),
            ("output", output),
            ("live_rows", self.count(stage)),
        ];
        self.launch(stage, part, None, named)
    }

    /// The SwiGLU row pass of `mlp` (clamped or not) of `stage`, from the
    /// gate and up rows in `input` to the product in `output`.
    fn swiglu(
        &mut self,
        stage: usize,
        mlp: &Mlp,
        input: &'static str,
        output: &'static str,
    ) -> Result<(), String> {
        let mut named = vec![
            ("input", Operand::Region(input)),
            ("output", Operand::Region(output)),
            ("live_rows", self.count(stage)),
        ];
        if let Mlp::SwigluClamped(params) = mlp {
            named.push(("limit_bits", Operand::U32((params.limit as f32).to_bits())));
        }
        self.launch(stage, "swiglu", None, named)
    }

    /// An `f32-pairs.v1` transformer over the residual stream `stream`,
    /// with norms and attention in `norm` and q, k, v pairs and the MLP's
    /// hidden rows in `qkv` and `mlp` (`b`); a SwiGLU's product goes to
    /// `norm`.
    fn transformer(
        &mut self,
        stage: usize,
        transformer: &Transformer,
        regions: [&'static str; 4],
    ) -> Result<(), String> {
        let [stream, norm, qkv, mlp] = regions;
        let params = transformer.params();
        match &params.rotary {
            Some(RotarySlot::Shared(Rotary::Axial2d(axial)))
                if axial.table == Table::HfF32Angle => {}
            Some(RotarySlot::Shared(Rotary::OneD(rotary))) if rotary.table == Table::HfF32Angle => {
            }
            _ => {
                return Err(
                    "only `rotary.axial-2d.v1` and `rotary.1d.v1` have host tables here".into(),
                );
            }
        }
        if !self.staged && !params.reorder.is_empty() {
            return Err("a reorder stages its maps and rotary rows per block".into());
        }
        if params.carry.len() > 1 {
            return Err("one `carry` region holds one carry's rows at most".into());
        }
        let qk_norm_eps = params.attention.qk_norm_eps()?;
        let up = match params.mlp {
            Mlp::Gelu(_) => "up.",
            Mlp::Swiglu(_) | Mlp::SwigluClamped(_) => "gate_up.",
        };
        let row_bytes = u64::from(params.width) * 4;
        let scale = params.attention.scale()?.to_bits();
        let count = self.count(stage);
        let side = |name| match self.staged {
            true => Operand::Region(name),
            false => Operand::Side(name, stage),
        };
        let (cos, sin, sequences) = (side("cos"), side("sin"), side("sequences"));
        // The first block's input: the rows the transformer reads.
        self.tap(stage, "in", stream, row_bytes);
        for block in 0..params.blocks {
            let at = |part| (stage, part, Some(block));
            let stream_rows = Operand::Region(stream);
            let reordered = params
                .reorder
                .iter()
                .any(|reorder| reorder.blocks.contains(block));
            if reordered {
                self.reorder(stage, Order::Gathered, [stream, norm], row_bytes)?;
            }
            // A staged transformer that reorders stages its rotary rows
            // before every block, in the block's order; one that does not,
            // once.
            if self.staged && (block == 0 || !params.reorder.is_empty()) {
                let order = if reordered {
                    Order::Gathered
                } else {
                    Order::Rows
                };
                self.ops[stage].push(Op::Upload(Input::Cos(stage, order), "cos"));
                self.ops[stage].push(Op::Upload(Input::Sin(stage, order), "sin"));
            }
            self.norm(
                stage,
                &params.norm,
                "norm1.",
                Some(block),
                (stream_rows.clone(), norm),
            )?;
            self.gemm(
                at("qkv"),
                "qkv.",
                Operand::Region(norm),
                Operand::Region(qkv),
            )?;
            if let Some(eps) = qk_norm_eps {
                let named = vec![
                    ("rows", Operand::Region(qkv)),
                    (
                        "q_weight",
                        self.weight(stage, "q_norm.weight", Some(block))?,
                    ),
                    (
                        "k_weight",
                        self.weight(stage, "k_norm.weight", Some(block))?,
                    ),
                    ("live_rows", count.clone()),
                    ("eps_bits", Operand::U32(eps.to_bits())),
                ];
                self.launch(stage, "qk_norm", None, named)?;
            }
            // q and k rotate in place, then each qkv row becomes its pair rows.
            let rotary = vec![
                ("rows", Operand::Region(qkv)),
                ("cos", cos.clone()),
                ("sin", sin.clone()),
                ("live_rows", count.clone()),
            ];
            self.launch(stage, "rotary", None, rotary)?;
            let sinks = match transformer.attention_at(block).1 {
                Sink::Key0 => self.weight(stage, "sinks", Some(block))?,
                Sink::None => Operand::Null,
            };
            let attention = vec![
                ("qkv", Operand::Region(qkv)),
                ("cu_seqlens", sequences.clone()),
                ("sinks", sinks),
                ("output", Operand::Region(norm)),
                ("live_tokens", count.clone()),
                ("live_sequences", Operand::Sequences(stage)),
                ("scale_bits", Operand::U32(scale)),
            ];
            self.launch(stage, "attention", Some(block), attention)?;
            self.gemm(
                at("proj"),
                "proj.",
                Operand::Region(norm),
                stream_rows.clone(),
            )?;
            self.norm(
                stage,
                &params.norm,
                "norm2.",
                Some(block),
                (stream_rows.clone(), norm),
            )?;
            self.gemm(at("up"), up, Operand::Region(norm), Operand::Region(mlp))?;
            let hidden = match params.mlp {
                Mlp::Gelu(_) => mlp,
                Mlp::Swiglu(_) => {
                    let swiglu = vec![
                        ("input", Operand::Region(mlp)),
                        ("output", Operand::Region(norm)),
                        ("live_rows", count.clone()),
                    ];
                    self.launch(stage, "swiglu", None, swiglu)?;
                    norm
                }
                Mlp::SwigluClamped(_) => {
                    self.swiglu(stage, &params.mlp, mlp, norm)?;
                    norm
                }
            };
            self.gemm(at("down"), "down.", Operand::Region(hidden), stream_rows)?;
            if reordered {
                self.reorder(stage, Order::Rows, [stream, norm], row_bytes)?;
            }
            self.tap(stage, "out", stream, row_bytes);
            // A carry saves the residual stream after one block and adds it
            // back after a later one.
            for carry in &params.carry {
                if carry.save_after == block {
                    self.ops[stage].push(Op::Copy {
                        from: stream,
                        to: "carry",
                        row_bytes,
                    });
                }
                if carry.add_after == block {
                    let named = vec![
                        ("input", Operand::Region(stream)),
                        ("residual", Operand::Region("carry")),
                        ("output", Operand::Region(stream)),
                        ("live_rows", count.clone()),
                    ];
                    self.launch(stage, "carry", None, named)?;
                }
            }
        }
        Ok(())
    }

    /// The tower the declaration encodes, as tap names spell it.
    fn tower(&self) -> &'static str {
        match self
            .declaration
            .input
            .recipes
            .first()
            .map(|recipe| recipe.tag())
        {
            Some(tag) if tag.starts_with("audio.") => "audio",
            _ => "vision",
        }
    }

    /// Taps the rows in `region` (`row_bytes` each): the current block's
    /// output (`out`, which moves on to the next block) or its input (`in`).
    fn tap(&mut self, stage: usize, site: &str, region: &'static str, row_bytes: u64) {
        let name = format!("enc.{}.B{:02}.{site}", self.tower(), self.blocks);
        if site == "out" {
            self.blocks += 1;
        }
        self.ops[stage].push(Op::Tap {
            name,
            region,
            row_bytes,
        });
    }

    /// Reorders the residual stream `stream` into `order` through `via`.
    fn reorder(
        &mut self,
        stage: usize,
        order: Order,
        [stream, via]: [&'static str; 2],
        row_bytes: u64,
    ) -> Result<(), String> {
        self.ops[stage].push(Op::Upload(Input::Reorder(stage, order), "reorder"));
        let named = vec![
            ("input", Operand::Region(stream)),
            ("indices", Operand::Region("reorder")),
            ("output", Operand::Region(via)),
            ("live_rows", self.count(stage)),
        ];
        self.launch(stage, "reorder", None, named)?;
        self.ops[stage].push(Op::Copy {
            from: via,
            to: stream,
            row_bytes,
        });
        Ok(())
    }
}

/// The full-run rows each stage reads and writes, from `capacity` rows.
fn full_rows(declaration: &Declaration) -> Result<Vec<[u64; 2]>, String> {
    let mut rows = u64::from(declaration.input.capacity_rows);
    let mut out = Vec::new();
    for (index, stage) in declaration.stages.iter().enumerate() {
        let after = match &stage.op {
            StageOp::Merge(Merge::Consecutive(params)) => rows / u64::from(params.rows),
            StageOp::Merge(Merge::SpaceToDepth(params)) => {
                declaration.space_to_depth_rows(index, params.factor, rows)
            }
            StageOp::Linear(_)
            | StageOp::Position(_)
            | StageOp::Norm(_)
            | StageOp::Transformer(_)
            | StageOp::Projector(_)
            | StageOp::Pad(_) => rows,
            StageOp::Conv1d(_) | StageOp::Registered(_) | StageOp::Group(_) => {
                let component = stage.op.component()?;
                // Runs after a joining stage start again at capacity.
                if component.joins_segments() {
                    rows = u64::from(declaration.input.capacity_rows);
                }
                component.item(Item { rows, grid: None })?.rows
            }
            other => return Err(format!("`{}` has no launch yet", other.id())),
        };
        out.push([rows, after]);
        rows = after;
    }
    Ok(out)
}

/// The `f32-pairs.v1` launch list; the region of the output rows. Each
/// stage writes the region the buffer plan gives it.
fn lower_pairs(
    builder: &mut Builder,
    plan: &BufferPlan,
    payload: &'static str,
    registered: &dyn Registered,
) -> Result<Option<&'static str>, String> {
    let declaration = builder.declaration;
    let name = declaration.name.as_str();
    let last = declaration.stages.len() - 1;
    let inputs = declaration.inputs().map_err(|error| error.0)?;
    let mut place = match builder.staged {
        true => Operand::Payload,
        false => Operand::Region(payload),
    };
    let mut output = None;
    for (index, stage) in declaration.stages.iter().enumerate() {
        let id = stage.op.id();
        let fail = |error: String| format!("{name} stages[{index}] ({id}): {error}");
        let writes = |what: &str| region(plan, name, Some(index), what).map_err(fail);
        let target = |region: &'static str| {
            if index == last {
                Operand::Output
            } else {
                Operand::Region(region)
            }
        };
        let joins_next = declaration
            .stages
            .get(index + 1)
            .and_then(|next| next.op.component().ok())
            .is_some_and(|next| next.joins_segments());
        match &stage.op {
            StageOp::Linear(Linear::Pairs(_)) => {
                let out = writes(&id)?;
                builder
                    .gemm((index, "linear", None), "", place, target(out))
                    .map_err(fail)?;
                place = Operand::Region(out);
            }
            StageOp::Conv1d(_) => {
                // Unfold the rows' taps, then the GEMM over the strided
                // output rows.
                let out = writes(&id)?;
                let [read, write] = builder.stage_counts[index];
                let im2col = Operand::Region("im2col");
                let unfold = vec![
                    ("input", place),
                    ("output", im2col.clone()),
                    ("live_frames", builder.count_slot(read)),
                ];
                builder
                    .launch_counting(index, ("unfold", None), unfold, read)
                    .map_err(fail)?;
                let gemm = vec![
                    ("activation", im2col),
                    (
                        "weight",
                        builder.weight(index, "weight", None).map_err(fail)?,
                    ),
                    ("bias", builder.bias(index, "", None).map_err(fail)?),
                    ("output", target(out)),
                    ("live_rows", builder.count_slot(write)),
                ];
                builder
                    .launch_counting(index, ("linear", None), gemm, write)
                    .map_err(fail)?;
                place = Operand::Region(out);
            }
            StageOp::Position(_) => {
                if builder.staged {
                    return Err(fail(
                        "staged learned positions require per-run tap uploads; unsupported by the shared runtime".into(),
                    ));
                }
                if declaration.stages[..index]
                    .iter()
                    .any(|stage| matches!(stage.op, StageOp::Merge(_)))
                {
                    return Err(fail(
                        "learned positions require the unmerged input grid".into(),
                    ));
                }
                let named = vec![
                    ("input", place.clone()),
                    ("table", builder.weight(index, "table", None).map_err(fail)?),
                    ("indices", Operand::Side("tap-indices", index)),
                    ("weights", Operand::Side("tap-weights", index)),
                    ("output", place.clone()),
                    ("live_rows", builder.count(index)),
                ];
                builder
                    .launch(index, "position", None, named)
                    .map_err(fail)?;
            }
            StageOp::Norm(norm) => {
                let out = writes(&id)?;
                if index == last {
                    return Err(fail("the last stage writes the output rows".into()));
                }
                builder
                    .norm(index, norm, "", None, (place, out))
                    .map_err(fail)?;
                place = Operand::Region(out);
            }
            StageOp::Transformer(transformer) => {
                let regions = [
                    writes("residual")?,
                    writes("norm")?,
                    writes("qkv, then its pair rows")?,
                    writes("mlp")?,
                ];
                if place != Operand::Region(regions[0]) || writes("attention")? != regions[1] {
                    return Err(fail("the plan's transformer regions differ".into()));
                }
                builder
                    .transformer(index, transformer, regions)
                    .map_err(fail)?;
            }
            StageOp::Pad(_) => {
                let out = writes(&id)?;
                let named = vec![
                    ("input", place),
                    ("output", target(out)),
                    ("live_rows", builder.count(index)),
                ];
                builder.launch(index, "pad", None, named).map_err(fail)?;
                place = Operand::Region(out);
            }
            StageOp::Projector(projector) => {
                // As the buffer plan places it: the norm (then GELU) rows
                // and the SwiGLU product in the region the stage does not
                // read, the up projection's rows in the other one.
                let Operand::Region(from) = place else {
                    return Err(fail("reads rows outside a region".into()));
                };
                let other = |region: &str| if region == "a" { "b" } else { "a" };
                let (scratch, up_rows) = (other(from), other(other(from)));
                let use_of = buffers::scratch_use(&id);
                for region in [scratch, up_rows] {
                    let planned = plan.regions.iter().any(|planned| {
                        planned.name == region
                            && planned.uses.iter().any(|at| {
                                at.encoder == name && at.stage == Some(index) && at.what == use_of
                            })
                    });
                    if !planned {
                        return Err(fail(format!("the plan holds no scratch in `{region}`")));
                    }
                }
                let out = writes(&id)?;
                let params = projector.params();
                builder
                    .norm(index, &params.norm, "norm.", None, (place, scratch))
                    .map_err(fail)?;
                let gelu = vec![
                    ("input", Operand::Region(scratch)),
                    ("output", Operand::Region(scratch)),
                    ("live_rows", builder.count(index)),
                ];
                builder.launch(index, "gelu", None, gelu).map_err(fail)?;
                let (up, hidden) = match params.mlp {
                    Mlp::Gelu(_) => ("up.", up_rows),
                    Mlp::Swiglu(_) | Mlp::SwigluClamped(_) => ("gate_up.", scratch),
                };
                builder
                    .gemm(
                        (index, "up", None),
                        up,
                        Operand::Region(scratch),
                        Operand::Region(up_rows),
                    )
                    .map_err(fail)?;
                if hidden == scratch {
                    builder
                        .swiglu(index, &params.mlp, up_rows, scratch)
                        .map_err(fail)?;
                }
                if out == hidden {
                    return Err(fail(format!("the down projection would overwrite `{out}`")));
                }
                builder
                    .gemm(
                        (index, "down", None),
                        "down.",
                        Operand::Region(hidden),
                        target(out),
                    )
                    .map_err(fail)?;
                place = Operand::Region(out);
            }
            StageOp::Merge(Merge::Consecutive(_)) => {}
            StageOp::Merge(Merge::SpaceToDepth(_)) => {
                // One CTA per cell: the live count is the cells written.
                let out = writes(&id)?;
                let write = builder.stage_counts[index][1];
                let named = vec![
                    ("input", place),
                    ("indices", Operand::Side("gather-indices", index)),
                    ("output", target(out)),
                    ("live_rows", builder.count_slot(write)),
                ];
                builder
                    .launch_counting(index, ("gather", None), named, write)
                    .map_err(fail)?;
                place = Operand::Region(out);
            }
            StageOp::Group(_) => {
                let out = writes(&id)?;
                let Operand::Region(from) = place else {
                    return Err(fail("gathers rows outside a region".into()));
                };
                let row_bytes = inputs[index].data.row_bytes();
                builder.ops[index].push(Op::Gather {
                    from,
                    to: out,
                    row_bytes,
                });
                place = Operand::Region(out);
            }
            StageOp::Registered(stage) => {
                // The region the stage writes: the one use of its id (its
                // scratch regions carry `buffers::scratch_use`).
                let out = writes(&id)?;
                let output = &inputs[index + 1];
                registered_launches(
                    builder,
                    registered,
                    (index, stage.component.as_ref()),
                    &inputs[index],
                    (place, output, out, joins_next),
                    plan,
                )
                .map_err(fail)?;
                place = Operand::Region(out);
            }
            _ => return Err(fail("no f32-pairs.v1 launch".into())),
        }
        if index == last {
            output = match &place {
                Operand::Region(region) => Some(*region),
                _ => return Err(fail("the last stage writes no region".into())),
            };
        }
    }
    Ok(output)
}

/// The launches of registered stage `index` (`component`) reading `input`
/// from `place` and writing `output` rows to `out` (each run's rows at its
/// offset when the next stage joins them), and its record uploads.
fn registered_launches(
    builder: &mut Builder,
    registered: &dyn Registered,
    (index, component): (usize, &dyn encoder_spec::components::Component),
    input: &encoder_spec::components::Boundary,
    (place, output, out, joins_next): (
        Operand,
        &encoder_spec::components::Boundary,
        &'static str,
        bool,
    ),
    plan: &BufferPlan,
) -> Result<(), String> {
    let last = builder.declaration.stages.len() - 1;
    let lowered = registered.stage(component, input)?;
    for launch in lowered.launches {
        let named = launch
            .arguments
            .into_iter()
            .map(|(param, bind)| {
                let operand = match bind {
                    Bind::Input => place.clone(),
                    Bind::Output if joins_next => Operand::RunOutput(out, output.data.row_bytes()),
                    Bind::Output if index == last => Operand::Output,
                    Bind::Output => Operand::Region(out),
                    Bind::Region(region) => Operand::Region(region),
                    Bind::View(view) => Operand::Weight(view),
                    Bind::Live => builder.count(index),
                };
                (param, operand)
            })
            .collect();
        builder.launch(index, launch.part, None, named)?;
    }
    if let Data::Codes { .. } = output.data {
        let name = format!("enc.{}.x.codes", builder.tower());
        builder.ops[index].push(Op::CodesTap {
            name,
            region: out,
            row_bytes: output.data.row_bytes(),
            run_output: joins_next,
        });
    }
    if joins_next {
        builder.joined = Some((out, index));
    }
    for (region, bytes) in lowered.record {
        if !plan.regions.iter().any(|known| known.name == region) {
            return Err(format!("the plan holds no region `{region}`"));
        }
        builder.record.push((region, bytes));
    }
    Ok(())
}

impl Lowered {
    /// Lowers `declaration` onto `plan`, preparing each slot's op on
    /// `backend` as `slots` resolves it, and the embedding override for
    /// prefill steps of up to `prefill_rows` rows.
    pub fn with_backend(
        declaration: &Declaration,
        plan: &BufferPlan,
        backend: &dyn Backend,
        slots: &Slots,
        registered: &dyn Registered,
        prefill_rows: u32,
    ) -> Result<Self, String> {
        let resolver = Resolver {
            backend,
            slots,
            prefill_rows,
        };
        Self::lower(declaration, plan, resolver, registered)
    }

    fn lower(
        declaration: &Declaration,
        plan: &BufferPlan,
        resolver: Resolver,
        registered: &dyn Registered,
    ) -> Result<Self, String> {
        let name = declaration.name.as_str();
        let staged = declaration.staged();
        declaration.precision.validate()?;
        let rows = full_rows(declaration).map_err(|error| format!("{name}: {error}"))?;
        let mut counts = Vec::new();
        let mut slot = |value: u64| match counts.iter().position(|&known| known == value) {
            Some(index) => index,
            None => {
                counts.push(value);
                counts.len() - 1
            }
        };
        let stage_counts = rows
            .iter()
            .map(|&[before, after]| [slot(before), slot(after)])
            .collect();
        let mut builder = Builder {
            declaration,
            resolver,
            prepared: Vec::new(),
            launches: Vec::new(),
            ops: vec![Vec::new(); declaration.stages.len()],
            stage_counts,
            staged,
            record: Vec::new(),
            joined: None,
            blocks: 0,
        };
        let payload = region(plan, name, None, "payload")?;
        let last = declaration.stages.len() - 1;
        let lowered = lower_pairs(&mut builder, plan, payload, registered)?;
        let place = lowered.ok_or_else(|| format!("{name}: no stages"))?;
        let delimiters = match &declaration.output.op {
            Layout::Tokens(_) => None,
            Layout::LearnedDelimiters(_) => {
                Some(delimiter_views(declaration).map_err(|error| format!("{name}: {error}"))?)
            }
        };
        let row_bytes = u64::from(declaration.output.op.params().hidden) * 2;
        let inject = inject(declaration, resolver).map_err(|error| format!("{name}: {error}"))?;
        let input = &declaration.input;
        // Every recipe's item traces through the declaration, with a grid
        // of its rows (`multiple` is one merge block) for a layout or a
        // space-to-depth merge that follows one.
        let follows_grid = delimiters.is_some()
            || declaration
                .stages
                .iter()
                .any(|stage| matches!(stage.op, StageOp::Merge(Merge::SpaceToDepth(_))));
        for recipe in &input.recipes {
            let rows = recipe.rows().map_err(|error| format!("{name}: {error}"))?;
            let side = rows.multiple.isqrt();
            let grid = (follows_grid && side * side == rows.multiple)
                .then(|| [rows.item / rows.grid, side, rows.grid / side]);
            let item = encoder_spec::components::Item {
                rows: rows.item,
                grid,
            };
            declaration.trace(item).map_err(|error| error.0)?;
        }
        Ok(Self {
            declaration: declaration.clone(),
            prepared: builder.prepared,
            launches: builder.launches,
            ops: builder.ops,
            record: builder.record,
            staged,
            counts,
            stage_counts: builder.stage_counts,
            payload,
            payload_row_bytes: u64::from(input.columns) * input.payload.bytes(),
            output: (place, rows[last][1] * row_bytes),
            joined: builder.joined,
            inject,
            delimiters,
        })
    }

    /// The symbols of the launch list, in order: an op's first symbol.
    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.launches
            .iter()
            .map(|launch| self.prepared[launch.op].1.info().symbols[0].0.as_str())
    }

    /// Checks every launch of the ops once at its count slot's full rows
    /// against the image views `weights` holds (every other buffer taken
    /// as large enough), so a view too small for its stage fails create
    /// rather than a record.
    pub(super) fn check(&self, weights: &BTreeMap<String, Span>) -> Result<(), String> {
        for name in self.delimiters.iter().flatten() {
            let span = weights
                .get(name)
                .ok_or_else(|| format!("{}: no encoder view {name}", self.declaration.name))?;
            if span.length != self.row_bytes() {
                return Err(format!(
                    "{}: the learned row {name} holds {} bytes, not one BF16 row of {}",
                    self.declaration.name,
                    span.length,
                    self.row_bytes()
                ));
            }
        }
        for launch in &self.launches {
            let roles = &launch.roles;
            let (slot, op) = &self.prepared[launch.op];
            let mut view = |operand: &Operand| match operand {
                Operand::Weight(name) => weights
                    .get(name)
                    .map(|span| View {
                        handle: 0,
                        bytes: span.length,
                    })
                    .ok_or_else(|| format!("no encoder view {name}")),
                _ => Ok(View {
                    handle: 0,
                    bytes: u64::MAX,
                }),
            };
            let rows = u32::try_from(self.counts[launch.count])
                .map_err(|_| "encoder rows exceed 32 bits".to_owned())?;
            let launches = op
                .launches(roles, (rows, 1), &mut view)
                .map_err(|error| format!("{} {slot:?}: {error}", self.declaration.name))?;
            if launches
                .iter()
                .any(|launch| launch.collective || !launch.plan.pdl)
            {
                return Err(format!(
                    "{} {slot:?}: an encoder launch is collective or not PDL",
                    self.declaration.name
                ));
            }
        }
        Ok(())
    }

    /// Bytes of one output row: the language model's BF16 hidden.
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.declaration.output.op.params().hidden) * 2
    }
}

/// The override of `declaration`'s layout.
fn inject(declaration: &Declaration, resolver: Resolver) -> Result<Inject, String> {
    let at = (declaration.stages.len(), "layout", None);
    let Resolver {
        backend,
        slots,
        prefill_rows,
    } = resolver;
    let SlotOp::Query(query) = slots(at.0, at.1, at.2)? else {
        return Err("the layout's override is a shared stage".into());
    };
    let op = backend
        .embedding_override(&query, Need::rows(prefill_rows))
        .map_err(|refused| format!("the override: {refused:?}"))?;
    if op.info().workspace > 0 {
        return Err("the encoder runtime binds no backend workspace".into());
    }
    Ok(std::sync::Arc::new(op))
}

impl Ready {
    /// The launches of a launch's `roles` over `rows` rows (and an
    /// attention's `sequences`), each operand bound to the view `view`
    /// gives it.
    pub(super) fn launches(
        &self,
        roles: &Roles,
        (rows, sequences): (u32, u32),
        view: &mut impl FnMut(&Operand) -> Result<View, String>,
    ) -> Result<Vec<kernel_core::backend::Launch>, String> {
        let mut view = |operand: Operand| view(&operand);
        let counts = Counts::rows(rows);
        let launches = match (self, roles) {
            (Ready::Rows(op), Roles::Rows(record)) => {
                op.launches(&record.clone().try_map(&mut view)?, View::EMPTY, counts)
            }
            (Ready::Heads(op), Roles::Heads(record)) => {
                op.launches(&record.clone().try_map(&mut view)?, View::EMPTY, counts)
            }
            (Ready::Attention(op), Roles::Attention(record, _)) => {
                let mut views = record.clone().try_map(&mut view)?;
                views.live_sequences = sequences;
                op.launches(&views, View::EMPTY, counts)
            }
            (Ready::Gemm(op), Roles::Gemm(record)) => {
                op.launches(&record.clone().try_map(&mut view)?, View::EMPTY, counts)
            }
            (Ready::Own(op), Roles::Own(named)) => {
                let views = named
                    .iter()
                    .map(|(name, operand)| Ok((*name, view(operand.clone())?)))
                    .collect::<Result<Vec<_>, String>>()?;
                op.launches(&views, View::EMPTY, counts)
            }
            _ => return Err("a launch's roles are not its op's".into()),
        };
        launches.map_err(|refusal| refusal.0.to_owned())
    }

    pub(super) fn info(&self) -> &kernel_core::backend::OpInfo {
        match self {
            Ready::Rows(op) => op.info(),
            Ready::Heads(op) => op.info(),
            Ready::Attention(op) => op.info(),
            Ready::Gemm(op) => op.info(),
            Ready::Own(op) => op.info(),
        }
    }
}

impl Launch {
    /// Every operand the launch binds.
    pub(super) fn operands(&self) -> Vec<Operand> {
        let mut operands = Vec::new();
        let mut push = |operand: &Operand| operands.push(operand.clone());
        match self.roles.as_ref() {
            Roles::Rows(record) => record.visit(&mut push),
            Roles::Heads(record) => record.visit(&mut push),
            Roles::Attention(record, _) => record.visit(&mut push),
            Roles::Gemm(record) => record.visit(&mut push),
            Roles::Own(named) => named.iter().for_each(|(_, operand)| push(operand)),
        }
        operands
    }
}

/// Takes a launch's operands by parameter name into its op's roles.
struct Take(Vec<(&'static str, Operand)>);

impl Take {
    /// The operand `name` binds, none for a null pointer.
    fn option(&mut self, name: &str) -> Option<Operand> {
        let at = self.0.iter().position(|(known, _)| *known == name)?;
        match self.0.remove(at).1 {
            Operand::Null => None,
            operand => Some(operand),
        }
    }

    fn value(&mut self, name: &str) -> Result<Operand, String> {
        self.option(name)
            .ok_or_else(|| format!("nothing binds `{name}`"))
    }

    /// Refuses an operand no role took, except the values the query
    /// carries.
    fn done(self) -> Result<(), String> {
        match self
            .0
            .iter()
            .find(|(name, _)| !matches!(*name, "eps_bits" | "limit_bits" | "scale_bits"))
        {
            Some((name, _)) => Err(format!("`{name}` is no role of the stage's op")),
            None => Ok(()),
        }
    }
}

/// The roles of `op`'s record that `named` binds, for a launch of `stage`.
fn roles(op: &Ready, stage: usize, named: Vec<(&'static str, Operand)>) -> Result<Roles, String> {
    let mut take = Take(named);
    let roles = match op {
        Ready::Rows(_) => {
            let live_rows = match take.option("live_rows") {
                Some(operand) => operand,
                None => take.value("live_frames")?,
            };
            Roles::Rows(EncoderRowBuffers {
                input: take.value("input")?,
                output: take.value("output")?,
                live_rows,
                weight: take.option("weight"),
                bias: take.option("bias"),
                residual: take.option("residual"),
                indices: take.option("indices"),
                table: take.option("table"),
                blend: take.option("weights"),
            })
        }
        Ready::Heads(_) => Roles::Heads(EncoderHeadBuffers {
            rows: take.value("rows")?,
            live_rows: take.value("live_rows")?,
            cos: take.option("cos"),
            sin: take.option("sin"),
            q_weight: take.option("q_weight"),
            k_weight: take.option("k_weight"),
        }),
        Ready::Attention(_) => {
            if take.value("live_sequences")? != Operand::Sequences(stage) {
                return Err("an attention counts its own stage's sequences".into());
            }
            let record = EncoderAttentionBuffers {
                qkv: take.value("qkv")?,
                cu_seqlens: take.value("cu_seqlens")?,
                sinks: take.option("sinks"),
                output: take.value("output")?,
                live_tokens: take.value("live_tokens")?,
                live_sequences: 0,
            };
            Roles::Attention(record, stage)
        }
        Ready::Gemm(_) => Roles::Gemm(EncoderGemmBuffers {
            activation: take.value("activation")?,
            weight: take.value("weight")?,
            bias: take.option("bias"),
            output: take.value("output")?,
            live_rows: take.value("live_rows")?,
        }),
        Ready::Own(_) => return Ok(Roles::Own(take.0)),
    };
    take.done()?;
    Ok(roles)
}

/// The backend path on SM120 catalogs.
#[cfg(test)]
mod backend_tests;

#[cfg(test)]
mod tests {
    use encoder_spec::buffers::{self, Serving};
    use encoder_spec::policy::Precision;

    use super::super::Shared;
    use encoder_spec::{Environment, parse};
    use kernel_core::backend::{Launch as BackendLaunch, OpInfo, PreparedOp};
    use kernel_core::contract::Refusal;
    use kernel_core::properties::Invariance;

    use super::*;

    /// A two-block F32 tower: a patch GEMM, the learned position grid, the
    /// blocks, the merger's norm, a 2 x 2 merge and its two GEMMs.
    pub(super) const TOWER: &str = r#"{
        "schema": "chopinfer.encoder.v1", "model": "alpha", "name": "alpha-vision",
        "precision": "f32-pairs.v1",
        "checkpoint": {"prefix": "visual.",
            "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 2}}]},
        "input": {"recipes": [{"tag": "image.patch-grid.v1", "patch": 16, "temporal_patch": 2,
                               "merge": 2, "maximum_pixels": 1048576}],
                  "patchify": "merge-blocks.v1", "payload": "f32", "columns": 1536,
                  "capacity_rows": 4096},
        "sequences": {"kind": "item"},
        "image_layout": "view-order.v1",
        "stages": [
            {"op": "linear.pairs.v1", "out": 256, "bias": true,
             "weights": {"weight": {"tensor": "patch.weight", "transforms": ["flatten"]},
                         "bias": "patch.bias"}},
            {"op": "position.learned-grid.bilinear-align-corners.f32.v1", "table": [16, 16],
             "weights": {"table": "pos.weight"}},
            {"op": "transformer.pre-norm.v1", "blocks": 2, "width": 256,
             "norm": {"op": "norm.layer-f32.v1", "eps": 1e-6},
             "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 4, "kv_heads": 4,
                           "head_dim": 64, "qkv_bias": true, "proj_bias": true, "mask": "full"},
             "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 16, "theta": 10000,
                        "style": "neox", "table": "table.hf-f32-angle.v1"},
             "mlp": {"op": "mlp.gelu.v1", "hidden": 1024, "gelu": "tanh", "bias": true},
             "weights": {
                "norm1.weight": "b.{i}.n1.weight", "norm1.bias": "b.{i}.n1.bias",
                "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
                "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
                "norm2.weight": "b.{i}.n2.weight", "norm2.bias": "b.{i}.n2.bias",
                "up.weight": "b.{i}.up.weight", "up.bias": "b.{i}.up.bias",
                "down.weight": "b.{i}.down.weight", "down.bias": "b.{i}.down.bias"}},
            {"op": "norm.layer-f32.v1", "eps": 1e-6,
             "weights": {"weight": "m.n.weight", "bias": "m.n.bias"}},
            {"op": "merge.consecutive.v1", "rows": 4},
            {"op": "linear.pairs.v1", "out": 1024, "bias": true, "gelu": "erf",
             "weights": {"weight": "m.fc1.weight", "bias": "m.fc1.bias"}},
            {"op": "linear.pairs.v1", "out": 512, "bias": true, "output": "bf16",
             "weights": {"weight": "m.fc2.weight", "bias": "m.fc2.bias"}}
        ],
        "output": {"op": "layout.tokens.v1", "hidden": 512, "positions": "grid-3axis.v1",
                   "media_mask": false},
        "parallel": ["replicated"]
    }"#;

    /// The parameters of the kernel the catalog gives `part`, in order.
    fn kernel_params(_: usize, part: &str) -> Result<Vec<&'static str>, String> {
        let mut params: Vec<&'static str> = match part {
            "linear" | "qkv" | "proj" | "up" | "down" => {
                vec!["activation", "weight", "bias", "output", "live_rows"]
            }
            "position" => vec![
                "input",
                "table",
                "indices",
                "weights",
                "output",
                "live_rows",
            ],
            "norm" => vec!["input", "weight", "bias", "output", "live_rows"],
            "qk_norm" => vec!["rows", "q_weight", "k_weight", "live_rows"],
            "swiglu" | "gelu" | "pad" => vec!["input", "output", "live_rows"],
            "gather" => vec!["input", "indices", "output", "live_rows"],
            "rotary" => vec!["rows", "cos", "sin", "live_rows"],
            "attention" => vec!["qkv", "cu_seqlens", "sinks", "output", "live_tokens"],
            _ => return Err(format!("no part {part}")),
        };
        match part {
            "norm" | "qk_norm" => params.push("eps_bits"),
            // The clamped SwiGLU's ABI.
            "swiglu" => params.push("limit_bits"),
            "attention" => params.extend(["live_sequences", "scale_bits"]),
            _ => {}
        }
        Ok(params)
    }

    /// The parameters of a slot's kernel (stage, part).
    pub(super) type Params = fn(usize, &str) -> Result<Vec<&'static str>, String>;

    /// A stand-in op of a slot, named `<part><stage>`: it binds every
    /// operand its launch names, and launches nothing.
    struct Stub(OpInfo);

    impl<B> PreparedOp<B> for Stub {
        fn info(&self) -> &OpInfo {
            &self.0
        }

        fn launches(&self, _: &B, _: View, _: Counts) -> Result<Vec<BackendLaunch>, Refusal> {
            Ok(Vec::new())
        }
    }

    fn stub(name: String) -> Stub {
        Stub(OpInfo::new(
            format!("test.{name}.v1"),
            Invariance::NONE,
            vec![(name, 0)],
        ))
    }

    /// A backend whose one op is the embedding override, named `layout`.
    pub(super) struct Overrides;

    impl Backend for Overrides {
        fn name(&self) -> &str {
            "overrides"
        }

        fn embedding_override(
            &self,
            _: &MultimodalQuery,
            _: Need,
        ) -> Result<Prepared<EmbeddingOverrideBuffers<View>>, kernel_core::backend::Refused>
        {
            Ok(Box::new(stub("layout".into())))
        }
    }

    /// Each slot of a part `params` knows as its stand-in op, the layout as
    /// the override's query.
    pub(super) fn stubs(
        params: Params,
    ) -> impl Fn(usize, &'static str, Option<u32>) -> Result<SlotOp, String> {
        move |stage, part, _| match part {
            "layout" => Ok(SlotOp::Query(MultimodalQuery::EmbeddingOverride {
                width: 512,
                capacity: 64,
            })),
            _ => {
                params(stage, part)?;
                Ok(SlotOp::Own(Box::new(stub(format!("{part}{stage}")))))
            }
        }
    }

    /// `declaration` lowered onto `plan` with the stand-in ops of `params`.
    pub(super) fn lower(
        declaration: &Declaration,
        plan: &BufferPlan,
        params: Params,
        registered: &dyn Registered,
    ) -> Result<Lowered, String> {
        Lowered::with_backend(
            declaration,
            plan,
            &Overrides,
            &stubs(params),
            registered,
            64,
        )
    }

    /// The stand-in op launch `index` runs.
    fn symbol(lowered: &Lowered, index: usize) -> &str {
        &lowered.prepared[lowered.launches[index].op]
            .1
            .info()
            .symbols[0]
            .0
    }

    /// The operands launch `index` names, by name.
    pub(super) fn named(lowered: &Lowered, index: usize) -> &Named<Operand> {
        match lowered.launches[index].roles.as_ref() {
            Roles::Own(named) => named,
            _ => panic!("launch {index} is no stand-in op's"),
        }
    }

    /// The operands of launch `index` in its kernel's parameter order
    /// (`params`), each named once.
    fn arguments(lowered: &Lowered, index: usize, params: Params) -> Vec<Operand> {
        let named = named(lowered, index);
        let (stage, part, _) = lowered.prepared[lowered.launches[index].op].0;
        let params = params(stage, part).unwrap();
        assert_eq!(named.len(), params.len(), "{}", symbol(lowered, index));
        params
            .iter()
            .map(|param| {
                named
                    .iter()
                    .find(|(name, _)| name == param)
                    .unwrap_or_else(|| panic!("nothing binds `{param}`"))
                    .1
                    .clone()
            })
            .collect()
    }

    fn lowered() -> (Lowered, BufferPlan) {
        let tower = parse(TOWER, &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        (lower(&tower, &plan, kernel_params, &Shared).unwrap(), plan)
    }

    /// The span of a merged `[t, h, w]` grid as the vendor lays an image
    /// (image_processor.py `image_token_types`, model.py
    /// `merge_image_embeddings`) per temporal grid: a start row, each grid
    /// row's cells in raster order then a newline row, an end row; cells
    /// numbered in the encoder's output order.
    fn vendor_span([t, h, w]: [u64; 3]) -> Vec<PlacedSource> {
        let mut span = Vec::new();
        for g in 0..t {
            span.push(PlacedSource::Learned(Delimiter::Start));
            for i in 0..h {
                for j in 0..w {
                    span.push(PlacedSource::Output {
                        first: (g * h + i) * w + j,
                        rows: 1,
                    });
                }
                span.push(PlacedSource::Learned(Delimiter::Newline));
            }
            span.push(PlacedSource::Learned(Delimiter::End));
        }
        span
    }

    /// The placed rows of every run of an item laid end to end: each run's
    /// copies, its output rows numbered as item rows.
    fn placed_span(layout: &Layout, grid: [u64; 3], cuts: &[u64]) -> Vec<Option<PlacedSource>> {
        let encoded = Item {
            rows: grid.iter().product(),
            grid: Some(grid),
        };
        let rows = layout.place(encoded, 0).unwrap().positions.len();
        let mut span = vec![None; rows];
        let mut bounds = vec![0];
        bounds.extend(
            cuts.iter()
                .copied()
                .filter(|&cut| cut > 0 && cut < encoded.rows),
        );
        bounds.push(encoded.rows);
        for run in bounds.windows(2) {
            for copy in placed_copies(layout, encoded, run[0]..run[1]).unwrap() {
                let row = copy.placed as usize;
                match copy.source {
                    PlacedSource::Output { first, rows } => {
                        for k in 0..rows as usize {
                            assert!(span[row + k].is_none(), "row {} written twice", row + k);
                            span[row + k] = Some(PlacedSource::Output {
                                first: run[0] + first + k as u64,
                                rows: 1,
                            });
                        }
                    }
                    learned => {
                        assert!(span[row].is_none(), "row {row} written twice");
                        span[row] = Some(learned);
                    }
                }
            }
        }
        span
    }

    /// Every placed row of a learned-delimiter span is written once, from the
    /// encoder row or the learned row the vendor puts there, however the
    /// item's output rows are cut into runs; a token layout's span is the
    /// encoder rows in order.
    #[test]
    fn learned_delimiter_spans_place_rows_as_the_vendor_lays_images() {
        let params = encoder_spec::components::layout::LayoutParams {
            hidden: 8,
            positions: encoder_spec::components::layout::PositionRule::Sequential,
            media_mask: true,
        };
        let delimited = Layout::LearnedDelimiters(params.clone());
        for grid in [[1, 2, 3], [1, 5, 7], [1, 1, 1], [2, 3, 4], [3, 1, 2]] {
            let cells: u64 = grid.iter().product();
            let expected: Vec<_> = vendor_span(grid).into_iter().map(Some).collect();
            assert_eq!(
                expected.len() as u64,
                grid[0] * (grid[1] * (grid[2] + 1) + 2)
            );
            for cuts in [
                vec![],
                vec![1],
                vec![cells / 2],
                vec![2, 5, 7, 11],
                (1..cells).collect(),
            ] {
                assert_eq!(
                    placed_span(&delimited, grid, &cuts),
                    expected,
                    "{grid:?} cut at {cuts:?}"
                );
            }
        }
        let tokens = Layout::Tokens(params);
        let span = placed_span(&tokens, [1, 2, 3], &[4]);
        let ordered: Vec<_> = (0..6)
            .map(|k| Some(PlacedSource::Output { first: k, rows: 1 }))
            .collect();
        assert_eq!(span, ordered);
        // a learned row is written in the run that starts the item only
        let later = placed_copies(
            &delimited,
            Item {
                rows: 6,
                grid: Some([1, 2, 3]),
            },
            3..6,
        )
        .unwrap();
        assert!(
            later
                .iter()
                .all(|copy| matches!(copy.source, PlacedSource::Output { .. }))
        );
    }

    /// A learned-delimiter tower lowers with its three learned rows as views
    /// of the image, and create refuses one that is missing or not one BF16
    /// row of the language model's hidden size.
    #[test]
    fn learned_delimiter_towers_lower_with_their_learned_rows() {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        value["output"] = serde_json::json!({
            "op": "layout.learned-delimiters.v1", "hidden": 512, "positions": "sequential",
            "media_mask": true,
            "weights": {"start": "img.start", "newline": "img.newline", "end": "img.end"}});
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        let delimited = lower(&tower, &plan, kernel_params, &Shared).unwrap();
        assert_eq!(
            delimited.delimiters,
            Some(["visual.img.start", "visual.img.newline", "visual.img.end"].map(String::from))
        );
        let (tokens, _) = lowered();
        assert_eq!(tokens.delimiters, None);
        let view = |bytes| {
            Span::new(
                runtime_bundle_abi::BundleDeviceBufferHandleV1::from_raw(0),
                0,
                bytes,
            )
        };
        let mut weights: BTreeMap<String, Span> = BTreeMap::new();
        for launch in &delimited.launches {
            for operand in launch.operands() {
                if let Operand::Weight(name) = operand {
                    weights.insert(name, view(u64::MAX));
                }
            }
        }
        for name in ["visual.img.start", "visual.img.newline"] {
            weights.insert(name.into(), view(1024));
        }
        let missing = delimited.check(&weights).unwrap_err();
        assert!(
            missing.contains("no encoder view visual.img.end"),
            "{missing}"
        );
        weights.insert("visual.img.end".into(), view(1022));
        let short = delimited.check(&weights).unwrap_err();
        assert!(
            short.contains("holds 1022 bytes, not one BF16 row of 1024"),
            "{short}"
        );
        weights.insert("visual.img.end".into(), view(1024));
        delimited.check(&weights).unwrap();
    }

    /// Zero-column padding before the patch GEMM: it reads the payload in
    /// `b` and writes the padded rows to `a`, which the patch GEMM reads.
    #[test]
    fn padded_payloads_launch_before_the_patch_gemm() {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        let stages = value["stages"].as_array_mut().unwrap();
        stages.insert(
            0,
            serde_json::json!({"op": "pad.zero-columns.v1", "columns": 1600}),
        );
        stages[1]["weights"]["weight"]["transforms"] =
            serde_json::json!(["flatten", {"pad_cols": 1600}]);
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        let lowered = lower(&tower, &plan, kernel_params, &Shared).unwrap();
        let symbol = |index: usize| symbol(&lowered, index);
        let operands = |index: usize| arguments(&lowered, index, kernel_params);
        assert_eq!((symbol(0), symbol(1)), ("pad0", "linear1"));
        assert_eq!(
            operands(0),
            [
                Operand::Region("b"),
                Operand::Region("a"),
                Operand::Count(0)
            ]
        );
        assert_eq!(operands(1)[0], Operand::Region("a"));
        assert_eq!(operands(1)[3], Operand::Region("stream"));
    }

    /// The tower with per-head q/k norms, a clamped SwiGLU MLP and a
    /// projector after its merge: the q/k norm runs on the qkv rows before
    /// the rotary, the clamped row pass gets its limit, and the projector
    /// keeps to `a` and `b` as the plan places it.
    #[test]
    fn vision_variants_launch_over_the_plans_regions() {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        let tower = &mut value["stages"][2];
        tower["attention"]["op"] = "attention.varlen.qk-rms.pairs.v1".into();
        tower["attention"]["qk_norm_eps"] = 1e-5.into();
        tower["mlp"] = serde_json::json!(
            {"op": "mlp.swiglu-clamped.v1", "hidden": 1024, "bias": true, "limit": 10});
        let weights = tower["weights"].as_object_mut().unwrap();
        weights.remove("up.weight");
        weights.remove("up.bias");
        for (role, tensor) in [
            ("q_norm.weight", "b.{i}.qn.weight"),
            ("k_norm.weight", "b.{i}.kn.weight"),
            ("gate_up.weight", "b.{i}.gate_up.weight"),
            ("gate_up.bias", "b.{i}.gate_up.bias"),
        ] {
            weights.insert(role.into(), tensor.into());
        }
        let stages = value["stages"].as_array_mut().unwrap();
        stages.truncate(5);
        stages.push(serde_json::json!(
            {"op": "linear.pairs.v1", "out": 1024, "bias": false,
             "weights": {"weight": "m.proj.weight"}}));
        stages.push(serde_json::json!(
            {"op": "projector.norm-gelu-mlp.v1",
             "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
             "gelu": "erf",
             "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 2048, "bias": false, "limit": 10},
             "output": "bf16",
             "weights": {"norm.weight": "m.n.weight", "norm.bias": "m.n.bias",
                         "gate_up.weight": "m.gate_up.weight", "down.weight": "m.down.weight"}}));
        value["output"]["hidden"] = 1024.into();
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        let lowered = lower(&tower, &plan, kernel_params, &Shared).unwrap();
        let operands = |index: usize| arguments(&lowered, index, kernel_params);
        let region = |name: &'static str| Operand::Region(name);
        let view = |name: &str| Operand::Weight(format!("visual.{name}"));
        let symbol = |index: usize| symbol(&lowered, index);
        // Each block: the q/k norm in place on the qkv rows in `b`, right
        // after the qkv GEMM and before the rotary.
        let norms = launches_of(&lowered, "qk_norm");
        assert_eq!(norms.len(), 2);
        for (block, &at) in norms.iter().enumerate() {
            assert_eq!(symbol(at - 1), "qkv2");
            assert_eq!(symbol(at + 1), "rotary2");
            assert_eq!(
                operands(at),
                [
                    region("b"),
                    view(&format!("b.{block}.qn.weight")),
                    view(&format!("b.{block}.kn.weight")),
                    Operand::Count(0),
                    Operand::U32(1e-5_f32.to_bits()),
                ]
            );
        }
        let limit = Operand::U32(10.0_f32.to_bits());
        let swiglus = launches_of(&lowered, "swiglu");
        assert_eq!(swiglus.len(), 3);
        // The tower's: gate and up in `b`, the product in `a`.
        assert_eq!(
            operands(swiglus[0]),
            [region("b"), region("a"), Operand::Count(0), limit.clone()]
        );
        // The projector reads the merger's linear in `b`: norm and GELU
        // into `a`, gate and up into `b`, the product into `a`, and the BF16
        // output rows from `a` into `b`, all over the merged rows' count.
        let gelu = launches_of(&lowered, "gelu6")[0];
        assert_eq!(symbol(gelu - 1), "norm6");
        assert_eq!(operands(gelu - 1)[0], region("b"));
        assert_eq!(operands(gelu - 1)[3], region("a"));
        assert_eq!(
            operands(gelu),
            [region("a"), region("a"), Operand::Count(1)]
        );
        assert_eq!(symbol(gelu + 1), "up6");
        assert_eq!(
            operands(gelu + 1),
            [
                region("a"),
                view("m.gate_up.weight"),
                Operand::Null,
                region("b"),
                Operand::Count(1)
            ]
        );
        assert_eq!(
            operands(gelu + 2),
            [region("b"), region("a"), Operand::Count(1), limit]
        );
        let last = lowered.launches.len() - 1;
        assert_eq!(symbol(last), "down6");
        assert_eq!(
            (&operands(last)[0], &operands(last)[3]),
            (&region("a"), &Operand::Output)
        );
        assert_eq!(lowered.output.0, "b");
    }

    #[test]
    fn staged_learned_positions_are_refused_at_lowering() {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        value["sequences"] = serde_json::json!({"kind": "temporal-grid"});
        value["input"]["chunking"] = serde_json::json!("whole-sequences.v1");
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        assert!(tower.staged());
        let plan = buffers::plan(
            &[&tower],
            4096,
            Serving {
                arena_rows: 1024,
                prefill_rows: 64,
                staging_bytes: 1 << 20,
            },
        )
        .unwrap();
        let error = lower(&tower, &plan, kernel_params, &Shared)
            .err()
            .expect("staged learned positions must not reach execution");
        assert_eq!(
            error,
            "alpha-vision stages[1] (position.learned-grid.bilinear-align-corners.f32.v1): staged learned positions require per-run tap uploads; unsupported by the shared runtime"
        );
    }

    #[test]
    fn staged_learned_positions_fail_create_before_host_allocation() {
        use runtime_bundle_abi::{BundleAbiError, HostServicesTableV1};
        use runtime_host::fake::FakeHostDeviceV1;
        use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
        use std::rc::Rc;

        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        value["sequences"] = serde_json::json!({"kind": "temporal-grid"});
        value["input"]["chunking"] = serde_json::json!("whole-sequences.v1");
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let backend = Rc::new(HostServicesV1Backend::new(
            FakeHostDeviceV1::new(),
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap()),
        ));
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let config = super::super::Config {
            label: "alpha",
            capture_schema: "alpha.capture.v1",
            arena: crate::media::MediaArena {
                rows: 1024,
                page_rows: 64,
                payload_limits: &[1 << 24],
            },
            prefill_rows: 64,
            registered: &Shared,
            staging_bytes: 1 << 20,
        };
        let slots = stubs(kernel_params);
        let error = super::super::Encoder::with_backend(
            table.caller(),
            &[Some(&tower)],
            config,
            &Overrides,
            &|_, stage, part, block| slots(stage, part, block),
            vec![std::collections::BTreeMap::new(); 2],
            |_, _| panic!("unsupported positions must fail before allocation"),
        )
        .err()
        .expect("unsupported positions must fail create");
        assert_eq!(error, BundleAbiError::InvalidRange("encoder plan"));
        assert_eq!(backend.live_resource_count(), 0);
    }

    #[test]
    fn learned_positions_after_a_merge_are_refused_at_lowering() {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        let stages = value["stages"].as_array_mut().unwrap();
        let position = stages.remove(1);
        stages.insert(4, position);
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let geometry = tower
            .geometry(
                "image.patch-grid.v1",
                encoder_spec::geometry::Media::Grid([1, 8, 12]),
            )
            .unwrap();
        assert_eq!(geometry.rows[3].grid, Some([1, 4, 6]));
        let plan = buffers::plan(
            &[&tower],
            4096,
            Serving {
                arena_rows: 1024,
                prefill_rows: 64,
                staging_bytes: 0,
            },
        )
        .unwrap();
        let error = lower(&tower, &plan, kernel_params, &Shared)
            .err()
            .expect("learned positions must not use the original grid after a merge");
        assert!(
            error.contains("learned positions require the unmerged input grid"),
            "{error}"
        );
    }

    #[test]
    fn the_launch_list_follows_the_stages_and_blocks() {
        let (lowered, _) = lowered();
        let block = [
            "norm2",
            "qkv2",
            "rotary2",
            "attention2",
            "proj2",
            "norm2",
            "up2",
            "down2",
        ];
        let mut expected = vec!["linear0", "position1"];
        expected.extend(block.iter().chain(&block));
        expected.extend(["norm3", "linear5", "linear6"]);
        assert_eq!(lowered.symbols().collect::<Vec<_>>(), expected);
        // One symbol per stage and part; the override is the layout's.
        let symbols = lowered
            .prepared
            .iter()
            .map(|(_, op)| &op.info().symbols[0].0);
        assert_eq!(symbols.collect::<std::collections::BTreeSet<_>>().len(), 12);
        assert_eq!(lowered.inject.info().symbols[0].0, "layout");
        // Counts: the patch rows, then the merged rows.
        assert_eq!(lowered.counts, [4096, 1024]);
        assert_eq!(lowered.stage_counts[4], [0, 1]);
        assert_eq!(lowered.stage_counts[5], [1, 1]);
    }

    #[test]
    fn launches_read_and_write_the_plans_regions() {
        let (lowered, plan) = lowered();
        let operands = |index: usize| arguments(&lowered, index, kernel_params);
        let region = |name: &'static str| Operand::Region(name);
        let view = |name: &str| Operand::Weight(format!("visual.{name}"));
        assert_eq!(lowered.payload, "b");
        // The patch GEMM reads the payload and starts the residual stream.
        assert_eq!(
            operands(0),
            [
                region("b"),
                view("patch.weight"),
                view("patch.bias"),
                region("stream"),
                Operand::Count(0)
            ]
        );
        // The position add works in place.
        assert_eq!(operands(1)[0], region("stream"));
        assert_eq!(operands(1)[4], region("stream"));
        // Block 1's attention: pair rows in `b`, no sinks, output in `a`.
        assert_eq!(
            operands(2 + 8 + 3),
            [
                region("b"),
                Operand::Side("sequences", 2),
                Operand::Null,
                region("a"),
                Operand::Count(0),
                Operand::Sequences(2),
                Operand::U32((64_f32).powf(-0.5).to_bits()),
            ]
        );
        assert_eq!(operands(2 + 8 + 4)[1], view("b.1.proj.weight"));
        // The merger's last GEMM writes the output rows over `a`, sized by
        // the merged rows of a full run.
        let last = operands(lowered.launches.len() - 1);
        assert_eq!((&last[0], &last[3]), (&region("b"), &Operand::Output));
        assert_eq!(lowered.output, ("a", 1024 * 512 * 2));
        assert!(plan.region("a").unwrap().bytes >= lowered.output.1);
    }

    #[test]
    fn a_slot_without_its_op_is_refused() {
        let tower = parse(TOWER, &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        let slots = stubs(kernel_params);
        let missing = |stage, part, block| match part {
            "rotary" => Err("no rotary entry".into()),
            _ => slots(stage, part, block),
        };
        let error = Lowered::with_backend(&tower, &plan, &Overrides, &missing, &Shared, 64)
            .err()
            .unwrap();
        assert!(error.contains("no rotary entry"), "{error}");
    }

    /// The bytes launch `index` reads through parameter `param`: the last
    /// upload to the region it binds there, as the program uploads every
    /// side input before its first launch.
    fn read(
        lowered: &Lowered,
        side: &super::super::side::Side,
        index: usize,
        param: &str,
    ) -> Vec<u8> {
        let Some((_, Operand::Side(region, stage))) = named(lowered, index)
            .iter()
            .find(|(name, _)| *name == param)
        else {
            panic!("{} binds no side region at {param}", symbol(lowered, index));
        };
        side.uploads
            .iter()
            .rev()
            .find(|(key, _)| *key == (*region, Some(*stage)))
            .map(|(_, bytes)| bytes.clone())
            .unwrap()
    }

    /// The tower with `change` applied, lowered, and its side inputs for an
    /// 8 x 12 image.
    fn variant(
        change: impl FnOnce(&mut Vec<serde_json::Value>),
    ) -> (Lowered, super::super::side::Side) {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        change(value["stages"].as_array_mut().unwrap());
        let declaration = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&declaration], 4096, serving).unwrap();
        let lowered = lower(&declaration, &plan, kernel_params, &Shared).unwrap();
        let geometry = declaration
            .geometry(
                "image.patch-grid.v1",
                encoder_spec::geometry::Media::Grid([1, 8, 12]),
            )
            .unwrap();
        let side =
            super::super::side::inputs(&lowered, &Shared, &geometry, 2, &geometry.runs[0]).unwrap();
        (lowered, side)
    }

    fn launches_of(lowered: &Lowered, part: &str) -> Vec<usize> {
        (0..lowered.launches.len())
            .filter(|&index| symbol(lowered, index).starts_with(part))
            .collect()
    }

    fn bytes<T: Copy, const N: usize>(values: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
        values.iter().flat_map(|&value| to(value)).collect()
    }

    /// Two stages that produce the same kind of side input each feed their
    /// own consumers: two transformers of different rotary theta (and their
    /// own sequence bounds), two learned position grids of different tables.
    /// Every side input uploads before the first launch, so a shared region
    /// would hand the first stage the second stage's table.
    #[test]
    fn each_stage_reads_its_own_side_inputs() {
        use encoder_spec::geometry::grid_positions;
        let positions = grid_positions(encoder_spec::Patchify::MergeBlocks, 2, 8, 12);
        let check = |lowered: &Lowered, side, launch, param: &str, expected: Vec<u8>| {
            assert!(
                read(lowered, side, launch, param) == expected,
                "launch {launch} reads another stage's {param}"
            );
        };
        let (lowered, side) = variant(|stages| {
            let mut second = stages[2].clone();
            second["rotary"]["theta"] = serde_json::json!(1000);
            stages.insert(3, second);
        });
        let rotary = launches_of(&lowered, "rotary");
        assert_eq!(rotary.len(), 4);
        for (launches, theta) in [(&rotary[..2], 10_000.0), (&rotary[2..], 1000.0)] {
            let (cos, sin) = super::super::side::rotary(&positions, theta, 32);
            for &launch in launches {
                check(
                    &lowered,
                    &side,
                    launch,
                    "cos",
                    bytes(&cos, f32::to_le_bytes),
                );
                check(
                    &lowered,
                    &side,
                    launch,
                    "sin",
                    bytes(&sin, f32::to_le_bytes),
                );
            }
        }
        let bounds = bytes(&[0, 96], i32::to_le_bytes);
        let attention = launches_of(&lowered, "attention");
        for (&launch, stage) in attention.iter().zip([2, 2, 3, 3]) {
            assert_eq!(
                arguments(&lowered, launch, kernel_params)[1],
                Operand::Side("sequences", stage)
            );
            check(&lowered, &side, launch, "cu_seqlens", bounds.clone());
        }
        assert_eq!(side.sequences, [(2, 1), (3, 1)]);
        let (lowered, side) = variant(|stages| {
            let mut second = stages[1].clone();
            second["table"] = serde_json::json!([32, 32]);
            second["weights"]["table"] = serde_json::json!("pos2.weight");
            stages.insert(2, second);
        });
        let position = launches_of(&lowered, "position");
        assert_eq!(position.len(), 2);
        for (&launch, table) in position.iter().zip([[16, 16], [32, 32]]) {
            let (indices, weights) = super::super::side::position_taps(&positions, [8, 12], table);
            check(
                &lowered,
                &side,
                launch,
                "indices",
                bytes(&indices, i32::to_le_bytes),
            );
            check(
                &lowered,
                &side,
                launch,
                "weights",
                bytes(&weights, f32::to_le_bytes),
            );
        }
    }

    /// A tower of row-major patches merged 2 x 2 by space-to-depth: the
    /// gather reads the norm's rows and the run's own gather indices, one
    /// launch row per cell, and writes the region the first aligner GEMM
    /// reads. A 5 x 7 grid gathers 3 x 4 cells, the last row and column of
    /// cells half past the grid; the plan holds the cells the recipe's
    /// budget admits.
    #[test]
    fn space_to_depth_gathers_each_runs_cells() {
        let mut value: serde_json::Value = serde_json::from_str(TOWER).unwrap();
        value["input"] = serde_json::json!({
            "recipes": [{"tag": "image.centered-pad.v1", "patch": 16, "downsample": 2,
                         "maximum_tokens": 259}],
            "patchify": "row-major.v1", "payload": "f32", "columns": 768,
            "capacity_rows": 1024});
        value["output"]["positions"] = "sequential".into();
        let stages = value["stages"].as_array_mut().unwrap();
        stages.remove(1);
        stages[3] =
            serde_json::json!({"op": "merge.space-to-depth.v1", "factor": 2, "pad": "zero"});
        let declaration = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 0,
        };
        let plan = buffers::plan(&[&declaration], 1024, serving).unwrap();
        assert_eq!(
            plan.owned("gather-indices", Some(3)).unwrap().bytes,
            256 * 4 * 4
        );
        let lowered = lower(&declaration, &plan, kernel_params, &Shared).unwrap();
        let [gather] = launches_of(&lowered, "gather")[..] else {
            panic!("one gather launch");
        };
        let write = lowered.stage_counts[3][1];
        let operands = arguments(&lowered, gather, kernel_params);
        assert_eq!(
            operands,
            [
                Operand::Region("a"),
                Operand::Side("gather-indices", 3),
                Operand::Region("b"),
                Operand::Count(write),
            ]
        );
        let linear = launches_of(&lowered, "linear4");
        assert_eq!(
            arguments(&lowered, linear[0], kernel_params)[0],
            Operand::Region("b")
        );

        let geometry = declaration
            .geometry(
                "image.centered-pad.v1",
                encoder_spec::geometry::Media::Grid([1, 5, 7]),
            )
            .unwrap();
        let side =
            super::super::side::inputs(&lowered, &Shared, &geometry, 1, &geometry.runs[0]).unwrap();
        assert_eq!(side.counts[write], 12);
        let indices = encoder_spec::geometry::space_to_depth_indices([1, 5, 7], 2).unwrap();
        assert_eq!(indices.len(), 12 * 4);
        assert_eq!(
            read(&lowered, &side, gather, "indices"),
            bytes(&indices, i32::to_le_bytes)
        );
    }

    /// A staged tower under `f32-pairs.v1`: F32 payload and rows, RMS F32
    /// norms, window and key-0 sink attention with a full-attention block,
    /// a reordered block, a SwiGLU row pass, a carry, the shared rotary.
    pub(super) const STAGED_PAIRS: &str = r#"{
        "schema": "chopinfer.encoder.v1", "model": "alpha", "name": "alpha-pairs",
        "precision": "f32-pairs.v1",
        "checkpoint": {"prefix": "visual.",
            "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 2}}]},
        "input": {"recipes": [{"tag": "image.patch-grid.v1", "patch": 16, "temporal_patch": 2,
                               "merge": 2, "maximum_pixels": 1048576}],
                  "patchify": "merge-blocks.v1", "payload": "f32", "columns": 1536,
                  "capacity_rows": 4096, "chunking": "whole-sequences.v1"},
        "sequences": {"kind": "temporal-grid"},
        "image_layout": "program-order-stacked.v1",
        "stages": [
            {"op": "linear.pairs.v1", "out": 128, "bias": false,
             "weights": {"weight": {"tensor": "patch.weight", "transforms": ["flatten"]}}},
            {"op": "transformer.pre-norm.v1", "blocks": 2, "width": 128,
             "norm": {"op": "norm.rms-f32.v1", "eps": 1e-6},
             "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 2, "kv_heads": 1,
                           "head_dim": 64, "qkv_bias": true, "proj_bias": true,
                           "mask": {"window": [64, 64]}, "sink": "key0"},
             "overrides": [{"blocks": "even", "attention": {"mask": "full", "sink": "none"}}],
             "reorder": [{"blocks": "odd", "reorder": {"op": "reorder.merge-group-columns.v1"}}],
             "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 8, "theta": 10000,
                        "style": "neox", "table": "table.hf-f32-angle.v1"},
             "carry": [{"save_after": 0, "add_after": 1}],
             "mlp": {"op": "mlp.swiglu.v1", "hidden": 256, "bias": true},
             "weights": {
                "norm1.weight": "b.{i}.n1.weight",
                "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
                "sinks": {"tensor": "b.{i}.sinks", "transforms": ["widen_f32"], "blocks": "odd"},
                "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
                "norm2.weight": "b.{i}.n2.weight",
                "gate_up.weight": {"concat_rows": ["b.{i}.gate.weight", "b.{i}.up.weight"]},
                "gate_up.bias": {"concat_rows": ["b.{i}.gate.bias", "b.{i}.up.bias"]},
                "down.weight": "b.{i}.down.weight", "down.bias": "b.{i}.down.bias"}},
            {"op": "norm.rms-f32.v1", "eps": 1e-6, "weights": {"weight": "m.n.weight"}},
            {"op": "merge.consecutive.v1", "rows": 4},
            {"op": "linear.pairs.v1", "out": 512, "bias": false, "gelu": "erf",
             "weights": {"weight": "m.fc1.weight"}},
            {"op": "linear.pairs.v1", "out": 256, "bias": false, "output": "bf16",
             "weights": {"weight": "m.fc2.weight"}}
        ],
        "output": {"op": "layout.tokens.v1", "hidden": 256, "positions": "sequential",
                   "media_mask": false},
        "parallel": ["replicated"]
    }"#;

    /// The parameters of the kernel each part of a staged pairs tower runs.
    fn staged_pairs_params(stage: usize, part: &str) -> Result<Vec<&'static str>, String> {
        let mut params = match part {
            "norm" => vec!["input", "weight", "output", "live_rows"],
            "swiglu" => vec!["input", "output", "live_rows"],
            "reorder" => vec!["input", "indices", "output", "live_rows"],
            "carry" => vec!["input", "residual", "output", "live_rows"],
            "registered" if stage == 3 => vec!["input", "scratch", "table", "output", "live_rows"],
            "registered" => vec!["input", "output", "live_rows"],
            _ => return kernel_params(stage, part),
        };
        if part == "norm" {
            params.push("eps_bits");
        }
        Ok(params)
    }

    #[test]
    fn staged_pair_towers_stage_their_side_inputs_and_reorder_through_a() {
        let tower = parse(STAGED_PAIRS, &Environment::new("alpha")).unwrap();
        assert!(tower.staged());
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 1 << 20,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        // Shared side-input regions and one live row count, as staged
        // declarations upload them in command order.
        for name in [
            "sequences",
            "cos",
            "sin",
            "reorder",
            "carry",
            "count",
            "payload",
        ] {
            assert_eq!(plan.region(name).unwrap().owner, None, "{name}");
        }
        assert!(plan.region("counts").is_none());
        // The SwiGLU's gate and up columns fill `b`; its product `a`.
        assert_eq!(plan.region("b").unwrap().bytes, 4096 * 512 * 4);
        let lowered = lower(&tower, &plan, staged_pairs_params, &Shared).unwrap();
        assert!(lowered.staged);
        assert_eq!(lowered.payload_row_bytes, 1536 * 4);
        let launch = |op: &Op| match op {
            Op::Launch(index) => symbol(&lowered, *index).to_owned(),
            Op::Upload(input, region) => format!("upload {input:?} {region}"),
            Op::Copy { from, to, .. } => format!("copy {from} {to}"),
            Op::Gather { from, to, .. } => format!("gather {from} {to}"),
            Op::Tap { name, region, .. } | Op::CodesTap { name, region, .. } => {
                format!("tap {name} {region}")
            }
        };
        let block = [
            "norm1",
            "qkv1",
            "rotary1",
            "attention1",
            "proj1",
            "norm1",
            "up1",
            "swiglu1",
            "down1",
        ];
        let mut expected = vec![
            "tap enc.vision.B00.in stream".to_owned(),
            "upload Cos(1, Rows) cos".to_owned(),
            "upload Sin(1, Rows) sin".to_owned(),
        ];
        expected.extend(block.iter().map(|name| name.to_string()));
        expected.push("tap enc.vision.B00.out stream".into());
        expected.push("copy stream carry".into());
        expected.push("upload Reorder(1, Gathered) reorder".into());
        expected.push("reorder1".into());
        expected.push("copy a stream".into());
        expected.push("upload Cos(1, Gathered) cos".into());
        expected.push("upload Sin(1, Gathered) sin".into());
        expected.extend(block.iter().map(|name| name.to_string()));
        expected.push("upload Reorder(1, Rows) reorder".into());
        expected.push("reorder1".into());
        expected.push("copy a stream".into());
        expected.push("tap enc.vision.B01.out stream".into());
        expected.push("carry1".into());
        assert_eq!(
            lowered.ops[1].iter().map(launch).collect::<Vec<_>>(),
            expected
        );
        let operands = |index: usize| arguments(&lowered, index, staged_pairs_params);
        let region = |name: &'static str| Operand::Region(name);
        let view = |name: &str| Operand::Weight(format!("visual.{name}"));
        // The patch GEMM reads the run's payload rows into the residual
        // stream; norms read the stream into `a`.
        assert_eq!(
            operands(0),
            [
                Operand::Payload,
                view("patch.weight"),
                view("patch.bias"),
                region("stream"),
                region("count")
            ]
        );
        assert_eq!(
            operands(1),
            [
                region("stream"),
                view("b.0.n1.weight"),
                region("a"),
                region("count"),
                Operand::U32(1e-6_f32.to_bits())
            ]
        );
        // Block 0's attention is full, without sinks; block 1's has them.
        let attention = |block: usize| {
            let index = (0..lowered.launches.len())
                .filter(|&index| symbol(&lowered, index) == "attention1")
                .nth(block)
                .unwrap();
            operands(index)
        };
        assert_eq!(attention(0)[1], region("sequences"));
        assert_eq!(attention(0)[2], Operand::Null);
        assert_eq!(attention(1)[2], view("b.1.sinks"));
        // The SwiGLU reads `b` into `a`, which the down GEMM adds into the
        // stream.
        let swiglu = (0..lowered.launches.len())
            .position(|index| symbol(&lowered, index) == "swiglu1")
            .unwrap();
        assert_eq!(operands(swiglu)[..2], [region("b"), region("a")]);
        assert_eq!(operands(swiglu + 1)[0], region("a"));
        assert_eq!(operands(swiglu + 1)[3], region("stream"));
        let last = operands(lowered.launches.len() - 1);
        assert_eq!(last[3], Operand::Output);
    }

    /// Two carries would share the one `carry` region, the second saving
    /// over rows the first still adds later: the plan and the lowering
    /// refuse them.
    #[test]
    fn overlapping_carries_are_refused() {
        let mut value: serde_json::Value = serde_json::from_str(STAGED_PAIRS).unwrap();
        value["stages"][1]["carry"] = serde_json::json!([
            {"save_after": 0, "add_after": 1},
            {"save_after": 0, "add_after": 1}
        ]);
        let tower = parse(&value.to_string(), &Environment::new("alpha")).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 1 << 20,
        };
        let error = buffers::plan(&[&tower], 4096, serving).unwrap_err();
        assert!(error.0.contains("one `carry` region"), "{error}");
        let single: Declaration = parse(STAGED_PAIRS, &Environment::new("alpha")).unwrap();
        let plan = buffers::plan(&[&single], 4096, serving).unwrap();
        let error = lower(&tower, &plan, staged_pairs_params, &Shared)
            .err()
            .unwrap();
        assert!(error.contains("one `carry` region"), "{error}");
    }

    /// Registered stages: a quantizer writing four codes per row whose
    /// scratch depends on its rows (one table, and a batch temporary only
    /// for runs of several rows), and an embedding of the codes as rows.
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(tag = "op")]
    enum Batched {
        #[serde(rename = "alpha:quantize.batched.v1")]
        V1(BatchedParams),
        #[serde(rename = "alpha:embed.codes.v1")]
        Embed(BatchedParams),
    }

    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BatchedParams {}

    impl encoder_spec::components::Typed for Batched {
        fn policies(&self) -> &'static [Precision] {
            encoder_spec::policy::ANY
        }
    }

    impl encoder_spec::components::Component for Batched {
        fn parallel(&self) -> encoder_spec::policy::ParallelClass {
            encoder_spec::policy::ParallelClass::RowLocal
        }

        fn output(
            &self,
            _: &encoder_spec::components::Context,
            input: &encoder_spec::components::Boundary,
        ) -> Result<encoder_spec::components::Boundary, String> {
            Ok(match self {
                Batched::V1(_) => input.with(Data::Codes {
                    bounds: vec![16; 4],
                }),
                Batched::Embed(_) => input.with(Data::Rows {
                    width: 128,
                    element: encoder_spec::components::Element::F32,
                }),
            })
        }

        fn roles(
            &self,
            _: &encoder_spec::components::Context,
            _: &encoder_spec::components::Boundary,
        ) -> Result<Vec<encoder_spec::weights::Role>, String> {
            Ok(Vec::new())
        }

        fn scratch(
            &self,
            rows: u64,
            input: &encoder_spec::components::Boundary,
        ) -> Result<Vec<encoder_spec::components::Scratch>, String> {
            if matches!(self, Batched::Embed(_)) {
                return Ok(Vec::new());
            }
            let mut scratch = vec![encoder_spec::components::Scratch {
                name: "table",
                bytes: 64,
            }];
            if rows > 1 {
                scratch.insert(
                    0,
                    encoder_spec::components::Scratch {
                        name: "batch-temporary",
                        bytes: rows * input.data.row_bytes(),
                    },
                );
            }
            Ok(scratch)
        }
    }

    static BATCHED: encoder_spec::registry::Provider = encoder_spec::registry::Provider {
        namespace: "alpha",
        registrations: &[
            encoder_spec::registry::Registration {
                id: "alpha:quantize.batched.v1",
                decoder: encoder_spec::registry::Decoder::Stage(
                    encoder_spec::registry::stage::<Batched>,
                ),
            },
            encoder_spec::registry::Registration {
                id: "alpha:embed.codes.v1",
                decoder: encoder_spec::registry::Decoder::Stage(
                    encoder_spec::registry::stage::<Batched>,
                ),
            },
        ],
    };

    /// The host side of `Batched`: one launch reading its rows, its two
    /// scratch regions and writing its output.
    struct BatchedHost;

    impl Registered for BatchedHost {
        fn rotary(
            &self,
            rotary: &dyn encoder_spec::components::rotary::RotaryComponent,
            _: &encoder_spec::geometry::Positions,
        ) -> Result<(Vec<f32>, Vec<f32>), String> {
            Err(format!("`{}` is not registered here", rotary.id()))
        }

        fn stage(
            &self,
            component: &dyn encoder_spec::components::Component,
            _: &encoder_spec::components::Boundary,
        ) -> Result<super::super::registered::Stage, String> {
            let arguments = if component.id() == "alpha:embed.codes.v1" {
                vec![
                    ("input", Bind::Input),
                    ("output", Bind::Output),
                    ("live_rows", Bind::Live),
                ]
            } else {
                vec![
                    ("input", Bind::Input),
                    ("scratch", Bind::Region("batch-temporary")),
                    ("table", Bind::Region("table")),
                    ("output", Bind::Output),
                    ("live_rows", Bind::Live),
                ]
            };
            Ok(super::super::registered::Stage {
                launches: vec![super::super::registered::Launch {
                    part: "registered",
                    arguments,
                }],
                record: Vec::new(),
            })
        }
    }

    /// A registered stage's output is the one region its id uses, apart
    /// from scratch it declares only for the planned rows (review
    /// counterexample: a quantizer's batch temporary, declared for runs of
    /// several rows and planned before its codes, was taken for the codes,
    /// which a one-row scratch query missed).
    #[test]
    fn registered_outputs_never_alias_row_dependent_scratch() {
        let mut value: serde_json::Value = serde_json::from_str(STAGED_PAIRS).unwrap();
        value["stages"].as_array_mut().unwrap().splice(
            3..3,
            [
                serde_json::json!({"op": "alpha:quantize.batched.v1"}),
                serde_json::json!({"op": "alpha:embed.codes.v1"}),
            ],
        );
        let mut environment = Environment::new("alpha");
        environment.registry.register(&BATCHED).unwrap();
        let tower = parse(&value.to_string(), &environment).unwrap();
        let serving = Serving {
            arena_rows: 1024,
            prefill_rows: 64,
            staging_bytes: 1 << 20,
        };
        let plan = buffers::plan(&[&tower], 4096, serving).unwrap();
        let id = "alpha:quantize.batched.v1";
        // The planned rows declare the batch temporary, under its scratch use.
        for name in ["batch-temporary", "table"] {
            let uses = &plan.region(name).unwrap().uses;
            assert!(
                uses.iter().all(|at| at.what == buffers::scratch_use(id)),
                "{name}"
            );
        }
        let lowered = lower(&tower, &plan, staged_pairs_params, &BatchedHost).unwrap();
        let launch = |name: &str| {
            let index = (0..lowered.launches.len())
                .find(|&index| symbol(&lowered, index) == name)
                .unwrap();
            arguments(&lowered, index, staged_pairs_params)
        };
        assert_eq!(
            launch("registered3")[..4],
            [
                Operand::Region("a"),
                Operand::Region("batch-temporary"),
                Operand::Region("table"),
                Operand::Region("codes"),
            ]
        );
        let planned = plan.region("codes").unwrap();
        assert!(
            planned
                .uses
                .iter()
                .any(|at| at.stage == Some(3) && at.what == id)
        );
        // The embedding reads the codes.
        assert_eq!(launch("registered4")[0], Operand::Region("codes"));
    }
}
