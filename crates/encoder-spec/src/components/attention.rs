//! `attention`: self-attention over variable-length sequences, a
//! transformer slot.

use serde::{Deserialize, Serialize};

use super::{Context, Typed, Variant, f32_positive, mul};
use crate::policy::{self, Precision};

/// Which keys a query row sees inside its sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Mask {
    /// Every key.
    Full,
    /// Keys at or before the query.
    Causal,
    /// Keys from `query - left` to `query + right`, inclusive.
    Window([u32; 2]),
    /// Keys from `query - n` to the query, inclusive.
    CausalWindow(u32),
}

/// A learned per-head logit that joins every softmax denominator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sink {
    #[default]
    None,
    /// One F32 logit per query head with a zero value, beside key 0.
    Key0,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionParams {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Whether the fused q/k/v projection adds a bias.
    pub qkv_bias: bool,
    /// Whether the output projection adds a bias.
    pub proj_bias: bool,
    pub mask: Mask,
    #[serde(default, skip_serializing_if = "is_none")]
    pub sink: Sink,
    /// The score scale, passed to the kernel as F32. Absent means
    /// `(head_dim as f32).powf(-0.5)`; written only when it differs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// `attention.varlen.qk-rms.pairs.v1` only, and required there: the
    /// epsilon of the per-head q and k RMSNorms, rounded to F32.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_norm_eps: Option<f64>,
}

fn is_none(sink: &Sink) -> bool {
    *sink == Sink::None
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Attention {
    /// BF16 q, k, v and output; F32 softmax.
    #[serde(rename = "attention.varlen.bf16.v1")]
    VarlenBf16(AttentionParams),
    /// q, k, v as BF16 hi+lo pairs: scores qhi khi + qhi klo + qlo khi (q lo
    /// k lo dropped), and P x V with every BF16 probability term (two, three
    /// below 128 keys) times both halves of v, each key tile accumulated
    /// from zero before it joins the row; F32 output
    /// (kernels-sm120 `cuda/attention/varlen_pairs.cuh`).
    #[serde(rename = "attention.varlen.pairs.v1")]
    VarlenPairs(AttentionParams),
    /// As `attention.varlen.pairs.v1`, after an RMSNorm of every q and k
    /// head over its `head_dim` F32 values (correctly rounded rsqrt, scaled
    /// by the `q_norm` or `k_norm` weight), between the q/k/v projection and
    /// the rotary.
    #[serde(rename = "attention.varlen.qk-rms.pairs.v1")]
    VarlenQkRmsPairs(AttentionParams),
}

impl Typed for Attention {
    fn policies(&self) -> &'static [Precision] {
        match self {
            Attention::VarlenBf16(_) => policy::LEGACY,
            Attention::VarlenPairs(_) | Attention::VarlenQkRmsPairs(_) => policy::PAIRS,
        }
    }
}

impl Attention {
    pub fn params(&self) -> &AttentionParams {
        match self {
            Attention::VarlenBf16(params)
            | Attention::VarlenPairs(params)
            | Attention::VarlenQkRmsPairs(params) => params,
        }
    }

    /// Whether q and k take BF16 hi+lo pair rows.
    pub fn pairs(&self) -> bool {
        matches!(
            self,
            Attention::VarlenPairs(_) | Attention::VarlenQkRmsPairs(_)
        )
    }

    /// The F32 epsilon of the per-head q and k RMSNorms, if the variant has
    /// them.
    pub fn qk_norm_eps(&self) -> Result<Option<f32>, String> {
        match (self, self.params().qk_norm_eps) {
            (Attention::VarlenQkRmsPairs(_), Some(eps)) => {
                f32_positive(&format!("`{}`: qk_norm_eps", self.id()), eps).map(Some)
            }
            (Attention::VarlenQkRmsPairs(_), None) => {
                Err(format!("`{}` needs `qk_norm_eps`", self.id()))
            }
            (_, Some(_)) => Err(format!(
                "`{}` has no q and k norms; `qk_norm_eps` belongs to \
                 attention.varlen.qk-rms.pairs.v1",
                self.id()
            )),
            (_, None) => Ok(None),
        }
    }

    /// The fused q/k/v projection's output columns.
    pub fn qkv_width(&self) -> Result<u64, String> {
        let params = self.params();
        let heads = u64::from(params.q_heads) + 2 * u64::from(params.kv_heads);
        mul(heads, u64::from(params.head_dim))
    }

    /// The attention output's columns, the output projection's input.
    pub fn attention_width(&self) -> Result<u64, String> {
        let params = self.params();
        mul(u64::from(params.q_heads), u64::from(params.head_dim))
    }

    /// The F32 scale the kernel receives.
    pub fn scale(&self) -> Result<f32, String> {
        let params = self.params();
        match params.scale {
            Some(scale) => f32_positive(&format!("`{}`: scale", self.id()), scale),
            None => Ok((params.head_dim as f32).powf(-0.5)),
        }
    }

    pub fn check(&self, context: &Context) -> Result<(), String> {
        context.admit(&self.id(), Typed::policies(self))?;
        let params = self.params();
        let fail = |why: String| Err(format!("`{}`: {why}", self.id()));
        if params.q_heads == 0 || params.kv_heads == 0 || params.head_dim == 0 {
            return fail("heads and head_dim are positive".into());
        }
        if params.q_heads % params.kv_heads != 0 {
            return fail(format!(
                "{} query heads do not share {} key/value heads evenly",
                params.q_heads, params.kv_heads
            ));
        }
        if params.head_dim % 2 != 0 {
            return fail(format!("head_dim {} is odd", params.head_dim));
        }
        for width in [self.qkv_width(), self.attention_width()] {
            if width.map_or(true, |width| width > u64::from(u32::MAX)) {
                return fail("the q/k/v width does not fit 32 bits".into());
            }
        }
        let scale = self.scale()?;
        if params.scale.is_some() && scale == (params.head_dim as f32).powf(-0.5) {
            return fail(format!(
                "scale {scale} is the default for head_dim {}; remove it",
                params.head_dim
            ));
        }
        self.qk_norm_eps()?;
        check_mask(params.mask).or_else(fail)
    }
}

pub fn check_mask(mask: Mask) -> Result<(), String> {
    match mask {
        Mask::Window([0, 0]) | Mask::CausalWindow(0) => {
            Err("a window sees no key but the query; use a wider window".into())
        }
        _ => Ok(()),
    }
}

/// An attention change for a set of blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<Mask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sink: Option<Sink>,
}
