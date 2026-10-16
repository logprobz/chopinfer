//! Portable source/effective precision and supplied implementation identities.

use super::{
    ActivationSpec, ArithmeticSpec, DecodeOrder, OperationKind, OperationSpec, OperationUse,
    PrecisionError, WeightLayout,
};
use crate::contract::WeightFormat;
use std::{fmt, str::FromStr};

pub const ACTIVATION_POLICY_ENV: &str = "CHOPINFER_ACTIVATION_POLICY";
pub const MAX_PRECISION_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;

pub struct Binding<H> {
    pub identity: ImplementationIdentity,
    pub handle: H,
}

/// A supplied implementation domain. Binding checks the complete operation
/// and returns a retained handle without compilation or materialization.
pub trait ImplementationProvider {
    type Handle;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Self::Handle>, String>;
}

/// An authenticated component inside the canonical bundle. Decoders bound the
/// input bytes before deserialization and call validate before using bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct PrecisionDocument {
    pub version: u32,
    pub operations: Vec<PrecisionSelection>,
}

impl PrecisionDocument {
    pub fn validate(&self) -> Result<(), PrecisionError> {
        if self.version != super::PRECISION_VERSION
            || self.operations.is_empty()
            || self.operations.len() > 4096
        {
            return Err(PrecisionError::new(
                "precision component",
                "version/count",
                "unsupported version or operation count",
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        for operation in &self.operations {
            operation.validate()?;
            if !ids.insert(&operation.source.id) {
                return Err(PrecisionError::new(
                    &operation.source.id,
                    "operation",
                    "duplicate precision record",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(rename_all = "lowercase"))]
pub enum ActivationPolicy {
    #[default]
    Auto,
    Checkpoint,
    Bf16,
}

impl fmt::Display for ActivationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Checkpoint => "checkpoint",
            Self::Bf16 => "bf16",
        })
    }
}

impl FromStr for ActivationPolicy {
    type Err = PrecisionError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "checkpoint" => Ok(Self::Checkpoint),
            "bf16" => Ok(Self::Bf16),
            _ => Err(PrecisionError::new(
                "activation policy",
                "value",
                format!("activation policy {value:?}: expected auto, checkpoint or bf16"),
            )),
        }
    }
}

impl ActivationPolicy {
    /// Pass the environment explicitly so library users need not mutate process
    /// state and an explicit argument can override even an invalid default.
    pub fn from_precedence(
        explicit: Option<&str>,
        environment: Option<&str>,
    ) -> Result<Self, PrecisionError> {
        explicit.or(environment).unwrap_or("auto").parse()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum WeightReuse {
    PlainFp8Bf16V1,
    PlainMxBf16V1,
    PlainNvfp4Bf16V1,
}

impl WeightReuse {
    /// The named weight-reuse policies permit these distinct BF16 arithmetic
    /// contracts. A supplied implementation must cover every use of one choice;
    /// preparation records that choice, including its rounding boundaries.
    pub fn bf16_decode_orders(self, source: &OperationSpec) -> Vec<DecodeOrder> {
        let mut orders = if source.kind == OperationKind::Dense {
            vec![DecodeOrder::ScaleThenCode, DecodeOrder::CodeThenScale]
        } else {
            vec![DecodeOrder::CodeThenScale, DecodeOrder::ScaleThenCode]
        };
        match source.weights[0].format {
            WeightFormat::Nvfp4 => orders.insert(0, DecodeOrder::ExactLocalThenGlobal),
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => {
                orders.insert(0, DecodeOrder::ExactScaledOperand);
            }
            _ => {}
        }
        if self == Self::PlainFp8Bf16V1
            && matches!(
                source.weights[0].format,
                WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
            )
        {
            orders.push(DecodeOrder::BlockScaleAfterMma);
        }
        orders
    }

    pub fn for_format(format: WeightFormat) -> Option<Self> {
        match format {
            WeightFormat::Fp8PerTensor
            | WeightFormat::Fp8PerChannel
            | WeightFormat::Fp8Block128x128(_)
            | WeightFormat::Fp8Rows64x128(_) => Some(Self::PlainFp8Bf16V1),
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => Some(Self::PlainMxBf16V1),
            WeightFormat::Nvfp4 => Some(Self::PlainNvfp4Bf16V1),
            WeightFormat::Bf16 => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct ImplementationIdentity {
    pub implementation: String,
    pub dependency: String,
    pub weight_layouts: Vec<WeightLayout>,
    pub activations: Vec<ActivationSpec>,
    pub arithmetic: ArithmeticSpec,
    /// Each quantizer, gather and layout copy has its own supplied entry.
    pub stages: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct BoundUse {
    pub usage: OperationUse,
    pub implementation: ImplementationIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct PrecisionSelection {
    pub source: OperationSpec,
    pub effective: OperationSpec,
    pub policy: ActivationPolicy,
    pub reuse: Option<WeightReuse>,
    pub fallback_reasons: Vec<String>,
    pub bindings: Vec<BoundUse>,
}

impl PrecisionSelection {
    pub fn validate(&self) -> Result<(), PrecisionError> {
        self.source.validate()?;
        self.effective.validate()?;
        let fail = |reason| PrecisionError::new(&self.source.id, "prepared precision", reason);
        let mut expected = self.source.clone();
        if let Some(rule) = self.reuse {
            if Some(rule) != WeightReuse::for_format(expected.weights[0].format)
                || expected.activations.iter().all(ActivationSpec::is_bf16)
                || self.policy == ActivationPolicy::Checkpoint
                || expected.weights.iter().any(|w| !w.transforms.is_empty())
                || expected
                    .activations
                    .iter()
                    .any(|a| !a.transforms.is_empty())
            {
                return Err(fail("invalid weight-reuse declaration"));
            }
            expected.activations.fill(ActivationSpec::bf16());
            expected.arithmetic.weight_operand = crate::contract::DType::Bf16;
            if !rule
                .bf16_decode_orders(&self.source)
                .contains(&self.effective.arithmetic.decode_order)
            {
                return Err(fail(
                    "weight-reuse arithmetic does not implement a declared BF16 scale policy",
                ));
            }
            expected.arithmetic.decode_order = self.effective.arithmetic.decode_order;
        }
        // The publisher's binding supplies the algorithm ID. The source ID
        // remains provenance; every declared arithmetic field still agrees.
        expected.arithmetic.numerical_contract =
            self.effective.arithmetic.numerical_contract.clone();
        for (weight, effective) in expected.weights.iter_mut().zip(&self.effective.weights) {
            weight.layout = effective.layout.clone();
        }
        if expected != self.effective || self.bindings.len() != expected.uses.len() {
            return Err(fail(
                "source and effective contracts or use coverage disagree",
            ));
        }
        let layouts: Vec<_> = self
            .effective
            .weights
            .iter()
            .map(|w| w.layout.clone())
            .collect();
        for (binding, usage) in self.bindings.iter().zip(&expected.uses) {
            let implementation = &binding.implementation;
            if binding.usage != *usage
                || implementation.activations != self.effective.activations
                || implementation.arithmetic != self.effective.arithmetic
                || implementation.weight_layouts != layouts
                || implementation.stages.is_empty()
                || implementation.stages.len() > 32
            {
                return Err(fail(
                    "implementation contracts or phase/capacity coverage disagree",
                ));
            }
            for name in std::iter::once(&implementation.implementation)
                .chain(std::iter::once(&implementation.dependency))
                .chain(implementation.stages.iter())
            {
                super::validate_name(name)?;
            }
        }
        if self.fallback_reasons.len() > 256 || self.fallback_reasons.iter().any(|s| s.len() > 4096)
        {
            return Err(fail("fallback report exceeds its bound"));
        }
        self.require_policy(self.policy)
    }

    /// Constrain an authenticated package without rerunning selection or changing
    /// its fixed layouts. Auto retains the prepared decision.
    pub fn require_policy(&self, policy: ActivationPolicy) -> Result<(), PrecisionError> {
        let legal = match policy {
            ActivationPolicy::Auto => true,
            ActivationPolicy::Checkpoint => {
                self.reuse.is_none() && self.source.activations == self.effective.activations
            }
            ActivationPolicy::Bf16 => {
                self.effective
                    .activations
                    .iter()
                    .all(ActivationSpec::is_bf16)
                    && self.effective.epilogues.output_quantization.is_none()
            }
        };
        if legal {
            Ok(())
        } else {
            Err(PrecisionError::new(
                &self.source.id,
                "activation policy",
                format!(
                    "authenticated package cannot satisfy {policy}; prepare the requested policy"
                ),
            ))
        }
    }
}
