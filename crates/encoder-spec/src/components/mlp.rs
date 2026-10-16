//! `mlp`: a transformer block's feed-forward slot, two GEMMs whose linear
//! variants the precision policy picks.

use serde::{Deserialize, Serialize};

use super::{Context, Gelu, Typed, Variant, f32_positive, mul};
use crate::policy::{self, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeluParams {
    pub hidden: u32,
    /// The width the GEMMs run, zero-padded past `hidden`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_to: Option<u32>,
    pub gelu: Gelu,
    pub bias: bool,
    /// `bf16-legacy.v1` only: the up projection rounds to BF16 before GELU.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub round_linear: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwigluParams {
    pub hidden: u32,
    pub bias: bool,
    /// `bf16-legacy.v1` only: the gate/up projection rounds to BF16 before
    /// SiLU and the product.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub round_linear: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClampedSwigluParams {
    pub hidden: u32,
    pub bias: bool,
    /// Rounded to F32: the gate is clamped to at most `limit`, the up half
    /// to `[-limit, limit]`.
    pub limit: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Mlp {
    /// `down(gelu(up(x)))`.
    #[serde(rename = "mlp.gelu.v1")]
    Gelu(GeluParams),
    /// `down(silu(gate) * up)`, with gate the first half of one
    /// `[2 * hidden, width]` projection.
    #[serde(rename = "mlp.swiglu.v1")]
    Swiglu(SwigluParams),
    /// `down(silu(min(gate, limit)) * clamp(up, -limit, limit))`, with gate
    /// and up the halves of one `[2 * hidden, width]` projection (biases
    /// added before the clamps); the row pass runs in F32.
    #[serde(rename = "mlp.swiglu-clamped.v1")]
    SwigluClamped(ClampedSwigluParams),
}

impl Typed for Mlp {
    fn policies(&self) -> &'static [Precision] {
        match self {
            Mlp::Gelu(_) | Mlp::Swiglu(_) => policy::ANY,
            Mlp::SwigluClamped(_) => policy::PAIRS,
        }
    }
}

impl Mlp {
    pub fn bias(&self) -> bool {
        match self {
            Mlp::Gelu(params) => params.bias,
            Mlp::Swiglu(params) => params.bias,
            Mlp::SwigluClamped(params) => params.bias,
        }
    }

    /// The columns of the hidden rows the down projection reads (the gate
    /// and up halves fused before them for a SwiGLU), with any padding.
    pub fn inner(&self) -> u64 {
        match self {
            Mlp::Gelu(params) => u64::from(params.pad_to.unwrap_or(params.hidden)),
            Mlp::Swiglu(params) => u64::from(params.hidden),
            Mlp::SwigluClamped(params) => u64::from(params.hidden),
        }
    }

    /// Whether the up projection writes fused gate and up halves.
    pub fn gated(&self) -> bool {
        !matches!(self, Mlp::Gelu(_))
    }

    /// The roles over rows of `width`.
    pub fn roles(&self, width: u32) -> Result<Vec<Role>, String> {
        let width = u64::from(width);
        let inner = self.inner();
        let (up, rows) = match self.gated() {
            false => ("up", inner),
            true => ("gate_up", mul(2, inner)?),
        };
        let mut roles = vec![
            Role::new(format!("{up}.weight"), vec![rows, width]),
            Role::new("down.weight", vec![width, inner]),
        ];
        if self.bias() {
            roles.extend([
                Role::new(format!("{up}.bias"), vec![rows]),
                Role::new("down.bias", vec![width]),
            ]);
        }
        Ok(roles)
    }

    pub fn check(&self, context: &Context) -> Result<(), String> {
        context.admit(&self.id(), Typed::policies(self))?;
        let (hidden, pad_to, round_linear) = match self {
            Mlp::Gelu(params) => (params.hidden, params.pad_to, params.round_linear),
            Mlp::Swiglu(params) => (params.hidden, None, params.round_linear),
            Mlp::SwigluClamped(params) => {
                f32_positive(&format!("`{}`: limit", self.id()), params.limit)?;
                (params.hidden, None, false)
            }
        };
        if hidden == 0 {
            return Err(format!("`{}`: hidden is zero", self.id()));
        }
        if pad_to.is_some_and(|pad_to| pad_to <= hidden) {
            return Err(format!(
                "`{}`: pad_to does not exceed hidden {hidden}",
                self.id()
            ));
        }
        if round_linear && context.precision != Precision::Bf16Legacy {
            return Err(format!(
                "`{}`: round_linear is allowed only under bf16-legacy.v1",
                self.id()
            ));
        }
        Ok(())
    }
}
