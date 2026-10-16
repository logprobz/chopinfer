//! Entry symbols derived from (variant id, parameters): `chop_enc_`, the id
//! without its version (`.`, `-` and `:` as `_`), the kernel's role inside
//! the variant if it runs more than one, every template parameter of the
//! entry body, the first 12 hex digits of the SHA-256 of the kernel's
//! identity, then the variant's version. The identity is the full variant
//! id, the role, and the entry's code (header, body, ABI and launch
//! bounds). The readable words are for people; the digest keeps ids that
//! read alike (`foo-bar` and `foo.bar`) apart. The symbol without its digest
//! is its stem (`Entry::stem`). Equal stages in two declarations get one symbol, and
//! two bodies never share one. Runtime values the kernel reads at launch
//! (capacity, eps) are not part of the code and not of the symbol.

use kernel_core::contract::{DType, GeluMode, GemmQuery, MultimodalQuery, RotaryStyle};
use sha2::{Digest as _, Sha256};

use super::kernel;

/// The symbol of `query` run by variant `variant` in `role`.
pub fn derive(variant: &str, role: Option<&str>, query: MultimodalQuery) -> Result<String, String> {
    let id = encoder_spec::id::parse(variant)?;
    let mut words = vec!["chop_enc".to_owned()];
    words.extend(id.namespace.map(str::to_owned));
    words.push(id.kind.name().to_owned());
    words.push(id.name.replace(['.', '-'], "_"));
    words.extend(role.map(str::to_owned));
    words.extend(tags(query));
    words.push(identity(variant, role, query)?[..12].to_owned());
    words.push(format!("v{}", id.version));
    Ok(words.join("_"))
}

/// The hex SHA-256 of a kernel's identity, each field length-prefixed.
fn identity(variant: &str, role: Option<&str>, query: MultimodalQuery) -> Result<String, String> {
    let fields = [
        variant.to_owned(),
        role.map_or_else(String::new, |role| format!("role {role}")),
        kernel::header(query)
            .map_err(|error| error.0.to_owned())?
            .to_owned(),
        kernel::body(query).map_err(|refusal| refusal.0.to_owned())?,
        format!("{:?}", kernel::params(query)),
        kernel::threads(query).to_string(),
    ];
    let mut hash = Sha256::new();
    for field in fields {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field.as_bytes());
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn gelu(mode: Option<GeluMode>) -> Option<String> {
    mode.map(|mode| match mode {
        GeluMode::Erf => "gelu_erf".into(),
        GeluMode::Tanh => "gelu_tanh".into(),
    })
}

fn element(dtype: DType) -> String {
    format!("{dtype:?}").to_ascii_lowercase()
}

fn gemm(query: GemmQuery, pairs: bool) -> Vec<String> {
    let epilogues = query.epilogues;
    let mut tags = vec![format!("n{}", query.n), format!("k{}", query.k)];
    tags.extend(epilogues.bias.then(|| "bias".into()));
    tags.extend(gelu(epilogues.gelu));
    tags.extend(epilogues.swiglu.map(|_| "swiglu".into()));
    if pairs {
        tags.extend(epilogues.residual_add.then(|| "residual".into()));
        tags.push(element(query.output));
    }
    tags
}

fn heads(q: u32, kv: u32, dim: u32) -> Vec<String> {
    vec![format!("q{q}"), format!("kv{kv}"), format!("d{dim}")]
}

fn rope(q: u32, kv: u32, dim: u32, rotary: u32, style: RotaryStyle) -> Vec<String> {
    let mut tags = heads(q, kv, dim);
    tags.push(format!("r{rotary}"));
    tags.push(format!("{style:?}").to_ascii_lowercase());
    tags
}

fn mask(
    causal: bool,
    left: Option<u32>,
    right: Option<u32>,
    sink: bool,
) -> impl Iterator<Item = String> {
    let full = !causal && left.is_none() && right.is_none();
    [
        full.then(|| "full".to_owned()),
        causal.then(|| "causal".to_owned()),
        left.map(|left| format!("left{left}")),
        right.map(|right| format!("right{right}")),
        sink.then(|| "sink".to_owned()),
    ]
    .into_iter()
    .flatten()
}

/// Every template parameter of the query's body, in body order.
fn tags(query: MultimodalQuery) -> Vec<String> {
    let width = |width: u32| vec![format!("w{width}")];
    match query {
        MultimodalQuery::AudioRopeBf16 { .. } => Vec::new(),
        MultimodalQuery::EmbeddingSumBf16 {
            tables,
            vocab,
            width,
            ..
        }
        | MultimodalQuery::EmbeddingSum {
            tables,
            vocab,
            width,
            ..
        } => vec![
            format!("t{tables}"),
            format!("vocab{vocab}"),
            format!("w{width}"),
        ],
        MultimodalQuery::ResidualAdd { width: w, .. }
        | MultimodalQuery::Bf16ToF32 { width: w, .. }
        | MultimodalQuery::GeluRows { width: w, .. }
        | MultimodalQuery::LayerNorm { width: w, .. }
        | MultimodalQuery::LayerNormF32 { width: w, .. }
        | MultimodalQuery::Permute { width: w, .. }
        | MultimodalQuery::EmbeddingOverride { width: w, .. }
        | MultimodalQuery::RmsNormF32 { width: w, .. }
        | MultimodalQuery::PermuteF32 { width: w, .. }
        | MultimodalQuery::ResidualAddF32 { width: w, .. }
        | MultimodalQuery::GeluRowsF32 { width: w, .. } => width(w),
        MultimodalQuery::SwigluF32 { hidden, .. }
        | MultimodalQuery::SwigluClampedF32 { hidden, .. } => vec![format!("h{hidden}")],
        MultimodalQuery::PadColumnsF32 { input, output, .. } => {
            vec![format!("w{input}"), format!("to{output}")]
        }
        MultimodalQuery::QkRmsNormF32 {
            q_heads,
            kv_heads,
            head_dim,
            ..
        } => heads(q_heads, kv_heads, head_dim),
        MultimodalQuery::SplitQkv {
            q_width, kv_width, ..
        } => vec![format!("q{q_width}"), format!("kv{kv_width}")],
        MultimodalQuery::ContiguousAttention {
            q_heads,
            kv_heads,
            head_dim,
            causal,
            window_left,
            window_right,
            key0_sink,
            ..
        }
        | MultimodalQuery::ContiguousAttentionPairs {
            q_heads,
            kv_heads,
            head_dim,
            causal,
            window_left,
            window_right,
            key0_sink,
            ..
        } => {
            let mut tags = heads(q_heads, kv_heads, head_dim);
            tags.extend(mask(causal, window_left, window_right, key0_sink));
            tags
        }
        MultimodalQuery::ScratchRope {
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            style,
            ..
        }
        | MultimodalQuery::RopeQkvPairs {
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            style,
            ..
        } => rope(q_heads, kv_heads, head_dim, rotary_dim, style),
        MultimodalQuery::Im2col1d {
            channels,
            kernel,
            stride,
            pad_left,
            pad_right,
            input,
            ..
        } => vec![
            format!("c{channels}"),
            format!("k{kernel}"),
            format!("s{stride}"),
            format!("p{pad_left}_{pad_right}"),
            element(input),
        ],
        MultimodalQuery::Im2col1dF32 {
            channels,
            kernel,
            stride,
            pad_left,
            pad_right,
            ..
        } => vec![
            format!("c{channels}"),
            format!("k{kernel}"),
            format!("s{stride}"),
            format!("p{pad_left}_{pad_right}"),
        ],
        MultimodalQuery::InterpolatedPositionAddF32 {
            width, table_rows, ..
        } => vec![format!("w{width}"), format!("table{table_rows}")],
        MultimodalQuery::Gemm(query) | MultimodalQuery::GemmBf16(query) => gemm(query, false),
        MultimodalQuery::GemmPairs(query) => gemm(query, true),
    }
}
