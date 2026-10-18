//! Media encoder stages through the model's encoder catalog entries
//! (`crate::encoder`): the entry whose code runs a stage's query, launched
//! with `encoder::kernel`'s plan and arguments bound by parameter name.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, DType, LaunchPlan, MultimodalQuery, Refusal, Validate};
use kernel_core::multimodal::{
    EmbeddingOverrideBuffers, EncoderAttentionBuffers, EncoderGemmBuffers, EncoderHeadBuffers,
    EncoderRowBuffers, EncoderStage,
};
use kernel_core::properties::Invariance;

use super::check_plan;
use super::select::{Candidate, choose};
use crate::encoder::{Entry, kernel};

/// The stage `op` of `stage` through the entry of `entries` whose code
/// runs it, of the least capacity that holds the op's rows. `scale` is an
/// attention's logit scale.
pub(super) fn prepare<B>(
    entries: &[Entry],
    op: &MultimodalQuery,
    stage: EncoderStage,
    scale: Option<f32>,
    need: Need,
) -> Result<Prepared<B>, Refused>
where
    EncoderOp: PreparedOp<B>,
{
    let op = EncoderOp::new(entries, *op, stage, scale, need)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The code an entry compiles for a query: equal codes are one kernel, as
/// `encoder::union` merges them. Runtime values (capacity, eps, limits)
/// are not code.
fn code(query: MultimodalQuery) -> Result<(&'static str, String, Vec<EntryParam>, u32), Refusal> {
    Ok((
        kernel::header(query)?,
        kernel::body(query)?,
        kernel::params(query),
        kernel::threads(query),
    ))
}

/// What an entry of the query's code launches: its symbol, parameters and
/// plan, which grids for the entry's capacity. Its other query values are
/// the model's at launch.
fn launches(entry: &Entry) -> String {
    let Entry {
        symbol, // the launched symbol
        query,  // its code selects the op; the plan grids for its capacity
        params, // the ABI
    } = entry;
    let plan = kernel::plan(symbol, *query, entry.capacity());
    format!("{symbol} {params:?} {plan:?}")
}

/// One encoder stage on its catalog entry. The plan grids for the entry's
/// capacity and reads the live rows on the device, so it serves every row
/// count; it is planned once. Launches pass the model's runtime values
/// (eps, limits), not the entry's, which `encoder::union` may have taken
/// from another declaration.
pub(super) struct EncoderOp {
    query: MultimodalQuery,
    symbol: String,
    params: Vec<EntryParam>,
    plan: LaunchPlan,
    /// The most rows a launch processes.
    capacity: u32,
    /// The logit scale of an attention.
    scale_bits: Option<u32>,
    persistent: bool,
    info: OpInfo,
}

impl EncoderOp {
    fn new(
        entries: &[Entry],
        query: MultimodalQuery,
        stage: EncoderStage,
        scale: Option<f32>,
        need: Need,
    ) -> Result<Self, Refused> {
        let capacity = need.capacity;
        if EncoderStage::of(&query)? != stage {
            return Err(Refused::Rule(format!(
                "{query:?} is not an encoder {stage:?} stage"
            )));
        }
        query.validate()?;
        kernel::limits(query)?;
        let wanted = code(query)?;
        let entry = choose(
            "encoder",
            entries
                .iter()
                .filter(|entry| code(entry.query).is_ok_and(|code| code == wanted))
                // The encoder entries declare no exactness
                // (`crate::invariance`).
                .map(|entry| {
                    Candidate::new(entry, &entry.symbol, launches(entry), Invariance::NONE)
                        .capacity(entry.capacity())
                }),
            need,
        )?
        .entry;
        kernel::contract(&entry.symbol, query)?;
        if capacity == 0 || capacity > entry.capacity() {
            return Err(Refused::Rule(format!(
                "{}: {capacity} rows exceed the entry's capacity of {}",
                entry.symbol,
                entry.capacity()
            )));
        }
        let scale_bits = match scale {
            Some(scale) if !scale.is_finite() => {
                return Err(Refused::Rule("an attention scale must be finite".into()));
            }
            scale => scale.map(f32::to_bits),
        };
        let plan = kernel::plan(&entry.symbol, entry.query, capacity)?;
        check_plan(&plan)?;
        let persistent = stage == EncoderStage::Gemm && plan.grid == [crate::gemm::SM_COUNT, 1, 1];
        let info = OpInfo::new(
            format!("sm120.encoder.v1[{}]", entry.symbol),
            // The encoder entries declare no exactness (`crate::invariance`).
            Invariance::NONE,
            vec![(
                entry.symbol.clone(),
                kernel::dynamic_shared_bytes(entry.query).unwrap_or(0),
            )],
        );
        Ok(Self {
            query,
            symbol: entry.symbol.clone(),
            params: entry.params.clone(),
            plan,
            capacity,
            scale_bits,
            persistent,
            info,
        })
    }

    fn rows(&self, counts: Counts) -> Result<u32, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("encoder rows exceed prepared capacity"));
        }
        Ok(counts.rows)
    }

    /// The launch over `rows` rows of the arguments `named`, which must
    /// read every one of the `sides` side roles the caller passed.
    fn launch(&self, rows: u32, named: Named, sides: usize) -> Result<Vec<Launch>, Refusal> {
        if named.sides != sides {
            return Err(Refusal("a buffer the encoder stage does not read"));
        }
        if named.values.len() != self.params.len() {
            return Err(Refusal("encoder argument count mismatch"));
        }
        let values = self
            .params
            .iter()
            .map(|param| {
                named
                    .values
                    .iter()
                    .find(|(role, _)| *role == name(param))
                    .map(|(_, value)| value.clone())
                    .ok_or(Refusal("an encoder parameter has no buffer"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows,
            plan: self.plan,
            arguments: kernel::pack_arguments(&self.params, values)?,
            collective: false,
            persistent: self.persistent,
        }])
    }
}

fn name(param: &EntryParam) -> &str {
    match param {
        EntryParam::DevicePointer { name, .. }
        | EntryParam::U32 { name }
        | EntryParam::U64 { name }
        | EntryParam::I32 { name }
        | EntryParam::TensorMap { name }
        | EntryParam::PointerArray { name, .. } => name,
    }
}

/// The bytes of `rows` rows of `width` elements of `bytes` bytes.
fn span(rows: u32, width: u64, bytes: u64) -> Result<u64, Refusal> {
    u64::from(rows)
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(bytes))
        .ok_or(Refusal("encoder buffer sizes overflow u64"))
}

/// A launch's arguments by parameter name, each view checked against the
/// bytes it reads or writes, and the side roles they read.
#[derive(Default)]
struct Named {
    values: Vec<(&'static str, Argument)>,
    sides: usize,
}

impl Named {
    fn view(&mut self, name: &'static str, view: View, bytes: u64) -> Result<&mut Self, Refusal> {
        view.covers(bytes)?;
        self.values
            .push((name, Argument::DeviceBuffer(view.handle)));
        Ok(self)
    }

    /// A side role the query reads.
    fn side(
        &mut self,
        name: &'static str,
        view: Option<View>,
        bytes: u64,
    ) -> Result<&mut Self, Refusal> {
        let view = view.ok_or(Refusal("an encoder stage misses a buffer its query reads"))?;
        self.sides += 1;
        self.view(name, view, bytes)
    }

    /// A side role the query reads when `read`, else a null pointer.
    fn optional(
        &mut self,
        name: &'static str,
        view: Option<View>,
        read: bool,
        bytes: u64,
    ) -> Result<&mut Self, Refusal> {
        if read {
            return self.side(name, view, bytes);
        }
        self.values.push((name, Argument::NullDevicePointer));
        Ok(self)
    }

    /// A side role the query reads when `read`. Otherwise a view the host
    /// passes is bound unread, as an image that holds the role for every
    /// stage binds it, and none binds a null pointer.
    fn bound(
        &mut self,
        name: &'static str,
        view: Option<View>,
        read: bool,
        bytes: u64,
    ) -> Result<&mut Self, Refusal> {
        match view {
            Some(view) if !read => {
                self.sides += 1;
                self.values
                    .push((name, Argument::DeviceBuffer(view.handle)));
                Ok(self)
            }
            view => self.optional(name, view, read, bytes),
        }
    }

    fn u32(&mut self, name: &'static str, value: u32) -> &mut Self {
        self.values.push((name, Argument::U32(value)));
        self
    }

    fn u64(&mut self, name: &'static str, value: u64) -> &mut Self {
        self.values.push((name, Argument::U64(value)));
        self
    }
}

fn present(views: &[Option<View>]) -> usize {
    views.iter().flatten().count()
}

impl PreparedOp<EncoderRowBuffers<View>> for EncoderOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &EncoderRowBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.rows(counts)?;
        let b = buffers;
        let mut named = Named::default();
        let f32_rows = |width: u32| span(rows, width.into(), 4);
        // The bytes of the input and output rows, and the live count's name.
        let (input, output, live) = match self.query {
            MultimodalQuery::InterpolatedPositionAddF32 {
                width, table_rows, ..
            } => {
                named
                    .side("table", b.table, span(table_rows, width.into(), 2)?)?
                    .side("indices", b.indices, span(rows, 4, 4)?)?
                    .side("weights", b.blend, span(rows, 4, 4)?)?;
                (f32_rows(width)?, f32_rows(width)?, "live_rows")
            }
            MultimodalQuery::LayerNormF32 {
                width, eps_bits, ..
            } => {
                named
                    .side("weight", b.weight, u64::from(width) * 2)?
                    .side("bias", b.bias, u64::from(width) * 2)?
                    .u32("eps_bits", eps_bits);
                (f32_rows(width)?, f32_rows(width)?, "live_rows")
            }
            MultimodalQuery::RmsNormF32 {
                width, eps_bits, ..
            } => {
                named
                    .side("weight", b.weight, u64::from(width) * 2)?
                    .u32("eps_bits", eps_bits);
                (f32_rows(width)?, f32_rows(width)?, "live_rows")
            }
            MultimodalQuery::PermuteF32 { width, .. } => {
                named.side("indices", b.indices, span(rows, 1, 4)?)?;
                (f32_rows(width)?, f32_rows(width)?, "live_rows")
            }
            MultimodalQuery::GatherRowsF32 { width, group, .. } => {
                // Every index names one of at most `rows * group` input
                // rows (each output row's blocks hold distinct inputs or
                // holes).
                let columns = u64::from(width) * u64::from(group);
                named.side("indices", b.indices, span(rows, group.into(), 4)?)?;
                (
                    span(rows, columns, 4)?,
                    span(rows, columns, 4)?,
                    "live_rows",
                )
            }
            MultimodalQuery::ResidualAddF32 { width, .. } => {
                named.side("residual", b.residual, f32_rows(width)?)?;
                (f32_rows(width)?, f32_rows(width)?, "live_rows")
            }
            MultimodalQuery::SwigluF32 { hidden, .. } => (
                span(rows, 2 * u64::from(hidden), 4)?,
                f32_rows(hidden)?,
                "live_rows",
            ),
            MultimodalQuery::SwigluClampedF32 {
                hidden, limit_bits, ..
            } => {
                named.u32("limit_bits", limit_bits);
                (
                    span(rows, 2 * u64::from(hidden), 4)?,
                    f32_rows(hidden)?,
                    "live_rows",
                )
            }
            MultimodalQuery::GeluRowsF32 { width, .. } => {
                (f32_rows(width)?, f32_rows(width)?, "live_rows")
            }
            MultimodalQuery::PadColumnsF32 { input, output, .. } => {
                (f32_rows(input)?, f32_rows(output)?, "live_rows")
            }
            MultimodalQuery::Im2col1dF32 {
                channels,
                kernel,
                stride,
                pad_left,
                pad_right,
                ..
            } => {
                // Rows are input frames. The kernel's one CTA per output row
                // covers its grid.
                let padded = u64::from(rows) + u64::from(pad_left) + u64::from(pad_right);
                let unfolded = padded
                    .checked_sub(kernel.into())
                    .map(|taps| taps / u64::from(stride) + 1)
                    .filter(|&unfolded| unfolded <= u64::from(self.plan.grid[0]))
                    .ok_or(Refusal("im2col frames do not fit its output rows"))?;
                let taps = u64::from(kernel) * u64::from(channels);
                (
                    f32_rows(channels)?,
                    span(unfolded as u32, taps, 4)?,
                    "live_frames",
                )
            }
            _ => return Err(Refusal("not an encoder row stage")),
        };
        named
            .view("input", b.input, input)?
            .view("output", b.output, output)?
            .view(live, b.live_rows, 4)?;
        let sides = [b.weight, b.bias, b.residual, b.indices, b.table, b.blend];
        self.launch(rows, named, present(&sides))
    }
}

impl PreparedOp<EncoderHeadBuffers<View>> for EncoderOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &EncoderHeadBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.rows(counts)?;
        let b = buffers;
        let mut named = Named::default();
        let width = |q_heads: u32, kv_heads: u32, head_dim: u32| {
            (u64::from(q_heads) + 2 * u64::from(kv_heads))
                .checked_mul(u64::from(head_dim))
                .ok_or(Refusal("encoder buffer sizes overflow u64"))
        };
        match self.query {
            MultimodalQuery::RopeQkvPairs {
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                ..
            } => {
                let table = span(rows, u64::from(rotary_dim / 2), 4)?;
                named
                    .view(
                        "rows",
                        b.rows,
                        span(rows, width(q_heads, kv_heads, head_dim)?, 4)?,
                    )?
                    .side("cos", b.cos, table)?
                    .side("sin", b.sin, table)?;
            }
            MultimodalQuery::QkRmsNormF32 {
                q_heads,
                kv_heads,
                head_dim,
                eps_bits,
                ..
            } => {
                named
                    .view(
                        "rows",
                        b.rows,
                        span(rows, width(q_heads, kv_heads, head_dim)?, 4)?,
                    )?
                    .side("q_weight", b.q_weight, u64::from(head_dim) * 2)?
                    .side("k_weight", b.k_weight, u64::from(head_dim) * 2)?
                    .u32("eps_bits", eps_bits);
            }
            _ => return Err(Refusal("not an encoder head stage")),
        }
        named.view("live_rows", b.live_rows, 4)?;
        let sides = [b.cos, b.sin, b.q_weight, b.k_weight];
        self.launch(rows, named, present(&sides))
    }
}

impl PreparedOp<EncoderAttentionBuffers<View>> for EncoderOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &EncoderAttentionBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.rows(counts)?;
        let b = buffers;
        let (
            MultimodalQuery::ContiguousAttentionPairs {
                q_heads,
                kv_heads,
                head_dim,
                key0_sink,
                ..
            },
            Some(scale_bits),
        ) = (self.query, self.scale_bits)
        else {
            return Err(Refusal("not an encoder attention stage"));
        };
        let sequences = b.live_sequences;
        if sequences == 0 || sequences > rows {
            return Err(Refusal(
                "attention needs one to its token count of sequences",
            ));
        }
        let mut named = Named::default();
        named
            .view(
                "qkv",
                b.qkv,
                span(
                    rows,
                    (u64::from(q_heads) + 2 * u64::from(kv_heads)) * u64::from(head_dim),
                    4,
                )?,
            )?
            .view("cu_seqlens", b.cu_seqlens, (u64::from(sequences) + 1) * 4)?
            .optional("sinks", b.sinks, key0_sink, u64::from(q_heads) * 4)?
            .view(
                "output",
                b.output,
                span(rows, u64::from(q_heads) * u64::from(head_dim), 4)?,
            )?
            .view("live_tokens", b.live_tokens, 4)?
            .u32("live_sequences", sequences)
            .u32("scale_bits", scale_bits);
        self.launch(rows, named, present(&[b.sinks]))
    }
}

impl PreparedOp<EncoderGemmBuffers<View>> for EncoderOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &EncoderGemmBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.rows(counts)?;
        let b = buffers;
        let MultimodalQuery::GemmPairs(query) = self.query else {
            return Err(Refusal("not an encoder GEMM stage"));
        };
        let (n, k) = (u64::from(query.n), u64::from(query.k));
        let output = if query.output == DType::F32 { 4 } else { 2 };
        let mut named = Named::default();
        named
            .view("activation", b.activation, span(rows, k, 4)?)?
            .view(
                "weight",
                b.weight,
                n.checked_mul(k * 2)
                    .ok_or(Refusal("encoder buffer sizes overflow u64"))?,
            )?
            .bound("bias", b.bias, query.epilogues.bias, n * 2)?
            .view("output", b.output, span(rows, n, output)?)?
            .view("live_rows", b.live_rows, 4)?;
        self.launch(rows, named, present(&[b.bias]))
    }
}

impl PreparedOp<EmbeddingOverrideBuffers<View>> for EncoderOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &EmbeddingOverrideBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.rows(counts)?;
        let b = buffers;
        let MultimodalQuery::EmbeddingOverride { width, .. } = self.query else {
            return Err(Refusal("not an embedding override"));
        };
        let row = u64::from(width) * 2;
        let arena = b
            .arena_rows
            .checked_mul(row)
            .ok_or(Refusal("encoder buffer sizes overflow u64"))?;
        let mut named = Named::default();
        named
            .view("activation", b.activation, span(rows, row, 1)?)?
            .view("arena", b.arena, arena)?
            .view("source", b.source, span(rows, 1, 8)?)?
            .view("live_rows", b.live_rows, 4)?
            .u64("arena_rows", b.arena_rows);
        self.launch(rows, named, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::encoder::{Kernel, OVERRIDE_ROWS};
    use kernel_core::backend::Backend;
    use kernel_core::contract::{
        ActivationFormat, Epilogues, GeluMode, GemmQuery, RotaryStyle, WeightFormat,
    };

    const CAPACITY: u32 = 4096;

    fn entry(variant: &str, role: Option<&'static str>, query: MultimodalQuery) -> Entry {
        Kernel::new(variant, role, query).entry().unwrap()
    }

    fn eps(value: f32) -> u32 {
        value.to_bits()
    }

    fn pairs(n: u32, k: u32, bias: bool, gelu: Option<GeluMode>, output: DType) -> MultimodalQuery {
        MultimodalQuery::GemmPairs(GemmQuery {
            m_capacity: CAPACITY,
            n,
            k,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::Bf16,
            output,
            epilogues: Epilogues {
                bias,
                gelu,
                residual_add: output == DType::F32 && gelu.is_none(),
                ..Epilogues::default()
            },
            grouped: None,
        })
    }

    fn attention(
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window_left: Option<u32>,
        key0_sink: bool,
    ) -> MultimodalQuery {
        MultimodalQuery::ContiguousAttentionPairs {
            q_heads,
            kv_heads,
            head_dim,
            causal: window_left.is_some(),
            window_left,
            window_right: None,
            key0_sink,
            capacity: CAPACITY,
        }
    }

    fn buf(handle: u64) -> Argument {
        Argument::DeviceBuffer(handle)
    }

    /// The launch the legacy builders give for `entry` over `rows` rows with
    /// `values` in parameter order.
    fn legacy(entry: &Entry, rows: u32, values: Vec<Argument>, persistent: bool) -> Launch {
        Launch {
            symbol: entry.symbol.clone(),
            rows,
            plan: kernel::plan(&entry.symbol, entry.query, rows).unwrap(),
            arguments: kernel::pack_arguments(&entry.params, values).unwrap(),
            collective: false,
            persistent,
        }
    }

    /// Views that cover anything a test launch reads.
    fn big(handle: u64) -> View {
        view(handle, 1 << 40)
    }

    fn rows_buffers() -> EncoderRowBuffers<View> {
        EncoderRowBuffers {
            input: big(1),
            output: big(2),
            live_rows: view(3, 4),
            weight: None,
            bias: None,
            residual: None,
            indices: None,
            table: None,
            blend: None,
        }
    }

    #[test]
    fn row_stages_launch_what_the_builders_plan_and_pack() {
        let b = rows_buffers();
        let cases = [
            (
                entry(
                    "norm.layer-f32.v1",
                    None,
                    MultimodalQuery::LayerNormF32 {
                        width: 1152,
                        eps_bits: eps(1e-6),
                        capacity: CAPACITY,
                    },
                ),
                EncoderRowBuffers {
                    weight: Some(big(4)),
                    bias: Some(big(5)),
                    ..b
                },
                vec![
                    buf(1),
                    buf(4),
                    buf(5),
                    buf(2),
                    buf(3),
                    Argument::U32(eps(1e-6)),
                ],
            ),
            (
                entry(
                    "norm.rms-f32.v1",
                    None,
                    MultimodalQuery::RmsNormF32 {
                        width: 1280,
                        eps_bits: eps(1e-6),
                        capacity: CAPACITY,
                    },
                ),
                EncoderRowBuffers {
                    weight: Some(big(4)),
                    ..b
                },
                vec![buf(1), buf(4), buf(2), buf(3), Argument::U32(eps(1e-6))],
            ),
            (
                entry(
                    "position.learned-grid.bilinear-align-corners.f32.v1",
                    None,
                    MultimodalQuery::InterpolatedPositionAddF32 {
                        width: 1152,
                        table_rows: 2304,
                        capacity: CAPACITY,
                    },
                ),
                EncoderRowBuffers {
                    table: Some(big(4)),
                    indices: Some(big(5)),
                    blend: Some(big(6)),
                    ..b
                },
                vec![buf(1), buf(4), buf(5), buf(6), buf(2), buf(3)],
            ),
            (
                entry(
                    "transformer.pre-norm.v1",
                    Some("reorder"),
                    MultimodalQuery::PermuteF32 {
                        width: 1280,
                        capacity: CAPACITY,
                    },
                ),
                EncoderRowBuffers {
                    indices: Some(big(4)),
                    ..b
                },
                vec![buf(1), buf(4), buf(2), buf(3)],
            ),
            (
                entry(
                    "merge.space-to-depth.v1",
                    None,
                    MultimodalQuery::GatherRowsF32 {
                        width: 1024,
                        group: 9,
                        capacity: CAPACITY,
                    },
                ),
                EncoderRowBuffers {
                    indices: Some(big(4)),
                    ..b
                },
                vec![buf(1), buf(4), buf(2), buf(3)],
            ),
            (
                entry(
                    "transformer.pre-norm.v1",
                    Some("carry"),
                    MultimodalQuery::ResidualAddF32 {
                        width: 1024,
                        capacity: CAPACITY,
                    },
                ),
                EncoderRowBuffers {
                    residual: Some(big(4)),
                    ..b
                },
                vec![buf(1), buf(4), buf(2), buf(3)],
            ),
            (
                entry(
                    "mlp.swiglu.v1",
                    Some("swiglu"),
                    MultimodalQuery::SwigluF32 {
                        hidden: 3420,
                        capacity: CAPACITY,
                    },
                ),
                b,
                vec![buf(1), buf(2), buf(3)],
            ),
            (
                entry(
                    "mlp.swiglu-clamped.v1",
                    Some("swiglu"),
                    MultimodalQuery::SwigluClampedF32 {
                        hidden: 4096,
                        limit_bits: 7.0_f32.to_bits(),
                        capacity: CAPACITY,
                    },
                ),
                b,
                vec![buf(1), buf(2), buf(3), Argument::U32(7.0_f32.to_bits())],
            ),
            (
                entry(
                    "projector.norm-gelu-mlp.v1",
                    Some("gelu"),
                    MultimodalQuery::GeluRowsF32 {
                        width: 4608,
                        capacity: CAPACITY,
                    },
                ),
                b,
                vec![buf(1), buf(2), buf(3)],
            ),
            (
                entry(
                    "pad.zero-columns.v1",
                    None,
                    MultimodalQuery::PadColumnsF32 {
                        input: 1176,
                        output: 1216,
                        capacity: CAPACITY,
                    },
                ),
                b,
                vec![buf(1), buf(2), buf(3)],
            ),
            (
                entry(
                    "conv1d.im2col.v1",
                    Some("unfold"),
                    MultimodalQuery::Im2col1dF32 {
                        channels: 128,
                        kernel: 3,
                        stride: 2,
                        pad_left: 1,
                        pad_right: 1,
                        capacity: CAPACITY,
                    },
                ),
                b,
                vec![buf(1), buf(2), buf(3)],
            ),
        ];
        let backend = Sm120::new().with_encoders(cases.iter().map(|(entry, ..)| entry.clone()));
        for (entry, buffers, values) in cases {
            let prepared = backend
                .encoder_rows(&entry.query, Need::rows(CAPACITY))
                .unwrap();
            assert_eq!(
                prepared.info().implementation,
                format!("sm120.encoder.v1[{}]", entry.symbol)
            );
            assert_eq!(prepared.info().guarantees, Invariance::NONE);
            assert_eq!(prepared.info().symbols, [(entry.symbol.clone(), 0)]);
            for rows in [1, 777, CAPACITY] {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [legacy(&entry, rows, values.clone(), false)],
                    "{}",
                    entry.symbol
                );
            }
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(0))
                    .is_err()
            );
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(CAPACITY + 1))
                    .is_err()
            );
            // A side role the stage does not read, and a short output.
            let extra = EncoderRowBuffers {
                residual: buffers.residual.xor(Some(big(9))),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&extra, View::EMPTY, Counts::rows(8))
                    .is_err()
            );
            let short = EncoderRowBuffers {
                output: view(2, 4),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(8))
                    .is_err()
            );
        }
    }

    #[test]
    fn row_stages_check_every_view_and_side_role() {
        let width = 1152;
        let norm = entry(
            "norm.layer-f32.v1",
            None,
            MultimodalQuery::LayerNormF32 {
                width,
                eps_bits: eps(1e-6),
                capacity: CAPACITY,
            },
        );
        let backend = Sm120::new().with_encoders([norm.clone()]);
        let prepared = backend.encoder_rows(&norm.query, Need::rows(64)).unwrap();
        let rows = 64 * u64::from(width) * 4;
        let exact = EncoderRowBuffers {
            input: view(1, rows),
            output: view(2, rows),
            live_rows: view(3, 4),
            weight: Some(view(4, u64::from(width) * 2)),
            bias: Some(view(5, u64::from(width) * 2)),
            ..rows_buffers()
        };
        assert!(
            prepared
                .launches(&exact, View::EMPTY, Counts::rows(64))
                .is_ok()
        );
        for short in [
            EncoderRowBuffers {
                input: view(1, rows - 4),
                ..exact
            },
            EncoderRowBuffers {
                live_rows: view(3, 2),
                ..exact
            },
            EncoderRowBuffers {
                bias: Some(view(5, u64::from(width) * 2 - 2)),
                ..exact
            },
            EncoderRowBuffers {
                bias: None,
                ..exact
            },
        ] {
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(64))
                    .is_err()
            );
        }
        // The im2col's frames must fill one output row, within its grid.
        let unfold = entry(
            "conv1d.im2col.v1",
            Some("unfold"),
            MultimodalQuery::Im2col1dF32 {
                channels: 128,
                kernel: 5,
                stride: 1,
                pad_left: 0,
                pad_right: 0,
                capacity: CAPACITY,
            },
        );
        let backend = Sm120::new().with_encoders([unfold.clone()]);
        let prepared = backend.encoder_rows(&unfold.query, Need::rows(16)).unwrap();
        let frames = |rows: u32| EncoderRowBuffers {
            input: view(1, u64::from(rows) * 128 * 4),
            output: view(2, u64::from(rows - 4) * 5 * 128 * 4),
            ..rows_buffers()
        };
        assert!(
            prepared
                .launches(&frames(16), View::EMPTY, Counts::rows(16))
                .is_ok()
        );
        assert!(
            prepared
                .launches(&frames(16), View::EMPTY, Counts::rows(4))
                .is_err()
        );
    }

    #[test]
    fn head_stages_launch_what_the_builders_plan_and_pack() {
        let rope = entry(
            "rotary.axial-2d.v1",
            None,
            MultimodalQuery::RopeQkvPairs {
                q_heads: 16,
                kv_heads: 16,
                head_dim: 72,
                rotary_dim: 36,
                style: RotaryStyle::Neox,
                capacity: CAPACITY,
            },
        );
        let norm = entry(
            "attention.varlen.pairs.v1",
            Some("qk_norm"),
            MultimodalQuery::QkRmsNormF32 {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 64,
                eps_bits: eps(1e-6),
                capacity: CAPACITY,
            },
        );
        let backend = Sm120::new().with_encoders([rope.clone(), norm.clone()]);
        let rows = |cos, sin, q_weight, k_weight| EncoderHeadBuffers {
            rows: big(1),
            live_rows: view(2, 4),
            cos,
            sin,
            q_weight,
            k_weight,
        };
        let cases = [
            (
                &rope,
                rows(Some(big(3)), Some(big(4)), None, None),
                vec![buf(1), buf(3), buf(4), buf(2)],
            ),
            (
                &norm,
                rows(None, None, Some(big(3)), Some(big(4))),
                vec![buf(1), buf(3), buf(4), buf(2), Argument::U32(eps(1e-6))],
            ),
        ];
        for (entry, buffers, values) in cases {
            let prepared = backend
                .encoder_heads(&entry.query, Need::rows(CAPACITY))
                .unwrap();
            for rows in [1, 1000, CAPACITY] {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [legacy(entry, rows, values.clone(), false)]
                );
            }
            let swapped = EncoderHeadBuffers {
                cos: buffers.q_weight,
                q_weight: buffers.cos,
                ..buffers
            };
            assert!(
                prepared
                    .launches(&swapped, View::EMPTY, Counts::rows(8))
                    .is_err()
            );
        }
        // Each row's cos and sin hold half its rotary dims.
        let prepared = backend.encoder_heads(&rope.query, Need::rows(8)).unwrap();
        let table = Some(view(3, 8 * 18 * 4));
        let exact = EncoderHeadBuffers {
            rows: view(1, 8 * 48 * 72 * 4),
            ..rows(table, table, None, None)
        };
        assert!(
            prepared
                .launches(&exact, View::EMPTY, Counts::rows(8))
                .is_ok()
        );
        let short = EncoderHeadBuffers {
            sin: Some(view(4, 8 * 18 * 4 - 4)),
            ..exact
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
    }

    #[test]
    fn attention_launches_what_the_builders_plan_and_pack() {
        let scale = 72_f32.powf(-0.5);
        let full = entry(
            "attention.varlen.pairs.v1",
            None,
            attention(16, 16, 72, None, false),
        );
        let sink = entry(
            "attention.varlen.pairs.v1",
            None,
            attention(8, 2, 64, Some(128), true),
        );
        let backend = Sm120::new().with_encoders([full.clone(), sink.clone()]);
        for (entry, sinks) in [(&full, None), (&sink, Some(big(3)))] {
            let prepared = backend
                .encoder_attention(&entry.query, scale, Need::rows(CAPACITY))
                .unwrap();
            assert_eq!(
                prepared.info().symbols,
                [(
                    entry.symbol.clone(),
                    kernel::dynamic_shared_bytes(entry.query).unwrap()
                )]
            );
            let buffers = EncoderAttentionBuffers {
                qkv: big(1),
                cu_seqlens: big(2),
                sinks,
                output: big(4),
                live_tokens: view(5, 4),
                live_sequences: 3,
            };
            let sink_argument = sinks.map_or(Argument::NullDevicePointer, |view| buf(view.handle));
            for rows in [3, 2048, CAPACITY] {
                let values = vec![
                    buf(1),
                    buf(2),
                    sink_argument.clone(),
                    buf(4),
                    buf(5),
                    Argument::U32(3),
                    Argument::U32(scale.to_bits()),
                ];
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [legacy(entry, rows, values, false)]
                );
            }
            let toggled = EncoderAttentionBuffers {
                sinks: sinks.xor(Some(big(3))),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&toggled, View::EMPTY, Counts::rows(8))
                    .is_err()
            );
            for sequences in [0, 9] {
                let buffers = EncoderAttentionBuffers {
                    live_sequences: sequences,
                    ..buffers
                };
                assert!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(8))
                        .is_err()
                );
            }
            let short = EncoderAttentionBuffers {
                cu_seqlens: view(2, 12),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(8))
                    .is_err()
            );
        }
        assert!(matches!(
            backend.encoder_attention(&full.query, f32::NAN, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn gemms_launch_what_the_builders_plan_and_pack() {
        let qkv = entry(
            "linear.pairs.v1",
            None,
            pairs(3456, 1152, true, None, DType::F32),
        );
        let fc1 = entry(
            "linear.pairs.v1",
            None,
            pairs(4608, 4608, true, Some(GeluMode::Erf), DType::Bf16),
        );
        let narrow = entry("linear.pairs.v1", None, {
            let MultimodalQuery::GemmPairs(query) = pairs(256, 1152, false, None, DType::F32)
            else {
                unreachable!()
            };
            MultimodalQuery::GemmPairs(GemmQuery {
                m_capacity: 256,
                ..query
            })
        });
        let backend = Sm120::new().with_encoders([qkv.clone(), fc1.clone(), narrow.clone()]);
        for (entry, bias, persistent) in [
            (&qkv, true, true),
            (&fc1, true, true),
            (&narrow, false, false),
        ] {
            let capacity = entry.capacity();
            let prepared = backend
                .encoder_gemm(&entry.query, Need::rows(capacity))
                .unwrap();
            assert!(prepared.info().weight_layouts.is_empty());
            assert_eq!(
                prepared.info().symbols,
                [(entry.symbol.clone(), kernel::PAIRS_GEMM_SHARED_BYTES)]
            );
            let buffers = EncoderGemmBuffers {
                activation: big(1),
                weight: big(2),
                bias: bias.then(|| big(3)),
                output: big(4),
                live_rows: view(5, 4),
            };
            let values = vec![
                buf(1),
                buf(2),
                if bias {
                    buf(3)
                } else {
                    Argument::NullDevicePointer
                },
                buf(4),
                buf(5),
            ];
            for rows in [1, capacity / 2, capacity] {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [legacy(entry, rows, values.clone(), persistent)]
                );
            }
            // A bias epilogue needs its bias; a GEMM without one binds a
            // passed bias unread.
            let toggled = EncoderGemmBuffers {
                bias: buffers.bias.xor(Some(big(3))),
                ..buffers
            };
            let launches = prepared.launches(&toggled, View::EMPTY, Counts::rows(1));
            if bias {
                assert!(launches.is_err());
            } else {
                let values = vec![buf(1), buf(2), buf(3), buf(4), buf(5)];
                assert_eq!(launches.unwrap(), [legacy(entry, 1, values, persistent)]);
            }
        }
        // The weight is [n, k] BF16 and the output's rows follow its dtype.
        let prepared = backend.encoder_gemm(&fc1.query, Need::rows(8)).unwrap();
        let exact = EncoderGemmBuffers {
            activation: view(1, 8 * 4608 * 4),
            weight: view(2, 4608 * 4608 * 2),
            bias: Some(view(3, 4608 * 2)),
            output: view(4, 8 * 4608 * 2),
            live_rows: view(5, 4),
        };
        assert!(
            prepared
                .launches(&exact, View::EMPTY, Counts::rows(8))
                .is_ok()
        );
        let short = EncoderGemmBuffers {
            weight: view(2, 4608 * 4608 * 2 - 2),
            ..exact
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
    }

    #[test]
    fn gemms_of_a_stacked_image_bind_its_bias_of_every_gemm() {
        // A program-order-stacked image holds a bias view for every GEMM,
        // and today's runtime binds it whether or not the GEMM adds it.
        let text = serde_json::json!({
            "schema": "chopinfer.encoder.v1",
            "model": "example",
            "name": "example-vision",
            "precision": "f32-pairs.v1",
            "checkpoint": {
                "prefix": "visual.",
                "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 2}}]
            },
            "input": {
                "recipes": [{"tag": "image.patch-grid.v1", "patch": 16, "temporal_patch": 2,
                             "merge": 2, "maximum_pixels": 1048576}],
                "patchify": "merge-blocks.v1", "payload": "f32", "columns": 1536,
                "capacity_rows": CAPACITY
            },
            "sequences": {"kind": "item"},
            "image_layout": "program-order-stacked.v1",
            "stages": [
                {"op": "linear.pairs.v1", "out": 1280, "bias": false,
                 "weights": {"weight": "patch.weight"}},
                {"op": "merge.consecutive.v1", "rows": 4},
                {"op": "linear.pairs.v1", "out": 5120, "bias": false, "gelu": "erf",
                 "weights": {"weight": "m.fc1.weight"}},
                {"op": "linear.pairs.v1", "out": 4096, "bias": true, "output": "bf16",
                 "weights": {"weight": "m.fc2.weight", "bias": "m.fc2.bias"}}
            ],
            "output": {"op": "layout.tokens.v1", "hidden": 4096, "positions": "grid-3axis.v1",
                       "media_mask": false},
            "parallel": ["replicated"]
        })
        .to_string();
        let declaration =
            encoder_spec::parse(&text, &encoder_spec::Environment::new("example")).unwrap();
        let entries = crate::encoder::catalog(&declaration, &crate::encoder::Shared).unwrap();
        let gemms = entries
            .iter()
            .filter(|entry| kernel::is_gemm(entry.query))
            .collect::<Vec<_>>();
        assert_eq!(gemms.len(), 3);
        let backend = Sm120::new().with_encoders(entries.iter().cloned());
        for entry in gemms {
            let MultimodalQuery::GemmPairs(query) = entry.query else {
                unreachable!()
            };
            let prepared = backend
                .encoder_gemm(&entry.query, Need::rows(CAPACITY))
                .unwrap();
            let buffers = EncoderGemmBuffers {
                activation: big(1),
                weight: big(2),
                bias: Some(view(3, u64::from(query.n) * 2)),
                output: big(4),
                live_rows: view(5, 4),
            };
            let values = vec![buf(1), buf(2), buf(3), buf(4), buf(5)];
            let persistent = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(1))
                .unwrap()[0]
                .persistent;
            for rows in [1, 1000, CAPACITY] {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [legacy(entry, rows, values.clone(), persistent)],
                    "{}",
                    entry.symbol
                );
            }
        }
    }

    #[test]
    fn the_embedding_override_launches_what_the_builders_plan_and_pack() {
        let width = 4096;
        let layout = entry(
            "layout.tokens.v1",
            None,
            MultimodalQuery::EmbeddingOverride {
                width,
                capacity: OVERRIDE_ROWS,
            },
        );
        let backend = Sm120::new().with_encoders([layout.clone()]);
        let prepared = backend
            .embedding_override(&layout.query, Need::rows(OVERRIDE_ROWS))
            .unwrap();
        let buffers = EmbeddingOverrideBuffers {
            activation: big(1),
            arena: view(2, 1024 * u64::from(width) * 2),
            source: big(3),
            live_rows: view(4, 4),
            arena_rows: 1024,
        };
        for rows in [1, 300, OVERRIDE_ROWS] {
            let values = vec![buf(1), buf(2), buf(3), buf(4), Argument::U64(1024)];
            assert_eq!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [legacy(&layout, rows, values, false)]
            );
        }
        let past = EmbeddingOverrideBuffers {
            arena_rows: 1025,
            ..buffers
        };
        assert!(
            prepared
                .launches(&past, View::EMPTY, Counts::rows(1))
                .is_err()
        );
    }

    #[test]
    fn stages_take_the_models_runtime_values_and_the_entrys_grid() {
        // A union entry keeps the first declaration's eps and the largest
        // capacity; a model's stage of the same code launches with its own.
        let union = entry(
            "norm.rms-f32.v1",
            None,
            MultimodalQuery::RmsNormF32 {
                width: 1280,
                eps_bits: eps(1e-6),
                capacity: 8192,
            },
        );
        let stage = MultimodalQuery::RmsNormF32 {
            width: 1280,
            eps_bits: eps(1e-5),
            capacity: 1024,
        };
        let backend = Sm120::new().with_encoders([union.clone()]);
        let prepared = backend.encoder_rows(&stage, Need::rows(1024)).unwrap();
        let buffers = EncoderRowBuffers {
            weight: Some(big(4)),
            ..rows_buffers()
        };
        let launch = &prepared
            .launches(&buffers, View::EMPTY, Counts::rows(512))
            .unwrap()[0];
        assert_eq!(launch.plan.grid, [8192, 1, 1]);
        assert_eq!(launch.arguments.0.last(), Some(&Argument::U32(eps(1e-5))));
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(1025))
                .is_err()
        );
    }

    #[test]
    fn prepare_refuses_unserved_unsupported_over_limit_and_inexact_stages() {
        let gelu = entry(
            "projector.norm-gelu-mlp.v1",
            Some("gelu"),
            MultimodalQuery::GeluRowsF32 {
                width: 4608,
                capacity: CAPACITY,
            },
        );
        let backend = Sm120::new().with_encoders([gelu.clone()]);
        let rule = |result: Result<Prepared<EncoderRowBuffers<View>>, Refused>| match result {
            Err(Refused::Rule(rule)) => rule,
            Err(other) => panic!("{other}"),
            Ok(_) => panic!("prepared"),
        };
        // A query of another stage, a BF16 variant no record describes,
        // and a query no entry runs.
        let gemm = pairs(256, 256, false, None, DType::F32);
        assert!(rule(backend.encoder_rows(&gemm, Need::rows(8))).contains("not an encoder"));
        let retired = MultimodalQuery::GeluRows {
            width: 4608,
            capacity: CAPACITY,
        };
        assert!(
            rule(backend.encoder_rows(&retired, Need::rows(8))).contains("fills no buffer record")
        );
        let other = MultimodalQuery::GeluRowsF32 {
            width: 1024,
            capacity: CAPACITY,
        };
        assert!(rule(backend.encoder_rows(&other, Need::rows(8))).contains("no SM120"));
        // Rows past the entry's grid, and none.
        assert!(
            rule(backend.encoder_rows(&gelu.query, Need::rows(CAPACITY + 1))).contains("capacity")
        );
        assert!(rule(backend.encoder_rows(&gelu.query, Need::rows(0))).contains("capacity"));
        // Head dims the contract allows and SM120's kernels do not serve,
        // refused by the backend and by today's builder.
        let wide = attention(16, 16, 96, None, false);
        let odd = MultimodalQuery::QkRmsNormF32 {
            q_heads: 16,
            kv_heads: 16,
            head_dim: 72,
            eps_bits: eps(1e-6),
            capacity: CAPACITY,
        };
        let ragged = MultimodalQuery::GatherRowsF32 {
            width: 1022,
            group: 9,
            capacity: CAPACITY,
        };
        for query in [wide, odd, ragged] {
            assert!(query.validate().is_ok());
            assert!(kernel::contract("chop_enc_test_v1", query).is_err());
        }
        assert!(matches!(
            backend.encoder_attention(&wide, 0.1, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("SM120")
        ));
        assert!(matches!(
            backend.encoder_heads(&odd, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("SM120")
        ));
        // GQA groups an m16 tile does not hold whole.
        assert!(matches!(
            backend.encoder_attention(&attention(12, 4, 64, None, false), 0.1, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        // Exactness the entries do not declare.
        assert!(matches!(
            backend.encoder_rows(
                &gelu.query,
                Need::rows(16).with_exact(Invariance::one_row(16))
            ),
            Err(Refused::Inexact { .. })
        ));
        // Two entries of one code under different variants.
        let twin = entry("transformer.pre-norm.v1", Some("gelu"), gelu.query);
        assert_ne!(twin.symbol, gelu.symbol);
        let backend = Sm120::new().with_encoders([gelu.clone(), twin]);
        assert!(rule(backend.encoder_rows(&gelu.query, Need::rows(8))).contains("two SM120"));
    }

    #[test]
    fn stages_prepare_alike_in_every_order() {
        let query = MultimodalQuery::SwigluF32 {
            hidden: 128,
            capacity: 16,
        };
        let small = entry("mlp.swiglu.v1", Some("swiglu"), query);
        let large = entry(
            "mlp.swiglu.v1",
            Some("swiglu"),
            crate::encoder::kernel::with_capacity(query, 32),
        );
        assert_eq!(small.symbol, large.symbol);
        let twin = entry("transformer.pre-norm.v1", Some("swiglu"), query);
        let buffers = rows_buffers();
        let prepare = |order: &[Entry], rows| {
            Sm120::new()
                .with_encoders(order.iter().cloned())
                .encoder_rows(&query, Need::rows(rows))
                .map(|op| {
                    op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                })
        };
        // The least capacity that holds the rows: the small grid for 8
        // rows, the large one for 24, whatever the order.
        for (rows, grid) in [(8, 16), (24, 32)] {
            for order in orders(&[small.clone(), large.clone(), small.clone()]) {
                let launches = prepare(&order, rows).unwrap();
                assert_eq!(launches[0].plan.grid, [grid, 1, 1]);
            }
        }
        // Entries of one code and capacity under two symbols tie.
        for order in orders(&[small.clone(), twin.clone()]) {
            let mut symbols = [small.symbol.as_str(), twin.symbol.as_str()];
            symbols.sort_unstable();
            assert!(matches!(
                prepare(&order, 8),
                Err(Refused::Rule(rule)) if rule.contains(&symbols.join(" and "))
            ));
        }
    }

    /// The encoder key holds what an entry launches, every field changed
    /// alone: its symbol, ABI and the grid its query's capacity plans; the
    /// query's code selects the op.
    #[test]
    fn launch_keys_hold_what_encoder_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let query = MultimodalQuery::SwigluF32 {
            hidden: 128,
            capacity: 16,
        };
        let base = entry("mlp.swiglu.v1", Some("swiglu"), query);
        let buffers = rows_buffers();
        let names = field_names!(Entry {
            symbol,
            query,
            params
        });
        let fields: Vec<Field<Entry>> = vec![
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_encoder_other_v1".into()),
            ),
            (
                "query",
                Class::Launch,
                Box::new(|e| e.query = crate::encoder::kernel::with_capacity(e.query, 32)),
            ),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
        ];
        keys_follow_launches(&base, names, fields, launches, |e| {
            let op = Sm120::new()
                .with_encoders([e.clone()])
                .encoder_rows(&query, Need::rows(8))?;
            let launched = [1, 8].map(|rows| {
                op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{:?}", e.params), launched))
        });
    }
}
