//! `norm`: a per-row normalization with a learned scale (and shift).

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Typed, Variant, f32_positive};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormParams {
    /// Rounded to F32 where the kernel adds it.
    pub eps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Norm {
    /// LayerNorm with weight and bias over BF16 rows.
    #[serde(rename = "norm.layer.v1")]
    Layer(NormParams),
    /// LayerNorm with weight and bias over F32 rows, two-pass, correctly
    /// rounded rsqrt.
    #[serde(rename = "norm.layer-f32.v1")]
    LayerF32(NormParams),
    /// RMSNorm scaled by the plain weight over BF16 rows.
    #[serde(rename = "norm.rms.v1")]
    Rms(NormParams),
    /// RMSNorm scaled by the plain weight over F32 rows, correctly rounded
    /// rsqrt.
    #[serde(rename = "norm.rms-f32.v1")]
    RmsF32(NormParams),
}

impl Norm {
    pub fn params(&self) -> &NormParams {
        match self {
            Norm::Layer(params)
            | Norm::LayerF32(params)
            | Norm::Rms(params)
            | Norm::RmsF32(params) => params,
        }
    }

    /// Whether the norm has a learned shift.
    pub fn has_bias(&self) -> bool {
        matches!(self, Norm::Layer(_) | Norm::LayerF32(_))
    }

    /// The roles of this norm's weights over rows of `width`, named
    /// `<slot>.weight` and `<slot>.bias`, or `weight` and `bias` for a stage.
    pub fn slot_roles(&self, slot: &str, width: u32) -> Vec<Role> {
        let dot = if slot.is_empty() { "" } else { "." };
        let mut roles = vec![Role::new(
            format!("{slot}{dot}weight"),
            vec![u64::from(width)],
        )];
        if self.has_bias() {
            roles.push(Role::new(
                format!("{slot}{dot}bias"),
                vec![u64::from(width)],
            ));
        }
        roles
    }

    /// Checks parameters and policy for a norm anywhere in a declaration.
    pub fn check(&self, context: &Context) -> Result<(), String> {
        context.admit(&self.id(), Typed::policies(self))?;
        f32_positive(&format!("`{}`: eps", self.id()), self.params().eps)?;
        Ok(())
    }
}

impl Typed for Norm {
    fn policies(&self) -> &'static [Precision] {
        match self {
            Norm::Layer(_) | Norm::Rms(_) => policy::LEGACY,
            Norm::LayerF32(_) | Norm::RmsF32(_) => policy::PAIRS,
        }
    }
}

impl Component for Norm {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        let width = input.width()?;
        self.check(context)?;
        Ok(input.with(Data::Rows {
            width,
            element: context.rows_element(),
        }))
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        Ok(self.slot_roles("", input.width()?))
    }
}
