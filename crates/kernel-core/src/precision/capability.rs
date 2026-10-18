//! Portable domains exported by typed implementation builders. A checkpoint
//! compiler binds these authenticated descriptions without a device toolchain.

use super::execution::{Binding, ImplementationIdentity, ImplementationProvider};
use super::{
    ActivationSpec, ArithmeticSpec, Calibration, CodeEncoding, DynamicScale, MatrixGeometry,
    NibbleOrder, OperationKind, OperationSpec, OperationUse, ScaleConvention, ScaleScope,
    WeightLayout,
};
use crate::contract::{ActivationFormat, DType, Epilogues, ScaleMode, Target, WeightFormat};
use std::collections::BTreeSet;

pub const MAX_CAPABILITY_BYTES: usize = 16 * 1024 * 1024;
pub const CAPABILITY_SECTION: &str = ".chopinfer.precision";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub enum ActivationCapability {
    Bf16,
    Fp8Group128(ScaleMode),
    Mxfp8,
    Nvfp4StaticScalar,
    Nvfp4TensorAmax,
    Mxfp8Ceil { amax_floor_bits: u32 },
    F32,
}

impl ActivationCapability {
    pub fn format(self) -> ActivationFormat {
        match self {
            Self::Bf16 => ActivationFormat::Bf16,
            Self::F32 => ActivationFormat::F32,
            Self::Fp8Group128(mode) => ActivationFormat::Fp8Group128(mode),
            Self::Mxfp8 | Self::Mxfp8Ceil { .. } => ActivationFormat::Mxfp8,
            Self::Nvfp4StaticScalar | Self::Nvfp4TensorAmax => ActivationFormat::Nvfp4,
        }
    }

    fn require(self, activation: &ActivationSpec) -> Result<(), String> {
        let storage = if self == Self::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        if activation.format != self.format()
            || activation.storage != storage
            || !activation.transforms.is_empty()
        {
            return Err(format!(
                "requested {:?} activation operands with {:?} storage and {} transforms; supplied composition requires {:?} operands, {} storage and no transforms",
                activation.format,
                activation.storage,
                activation.transforms.len(),
                self.format(),
                if storage == DType::F32 { "F32" } else { "BF16" }
            ));
        }
        if self == Self::F32 {
            return if activation.is_f32() {
                Ok(())
            } else {
                Err("F32 binding cannot quantize or round activation operands".into())
            };
        }
        if self == Self::Bf16 {
            return if activation.is_bf16() {
                Ok(())
            } else {
                Err("BF16 binding cannot quantize activation operands".into())
            };
        }
        let calibration = &activation
            .quantizer
            .as_ref()
            .ok_or("missing activation quantizer contract")?
            .calibration;
        let fits = match (self, calibration) {
            (
                Self::Mxfp8Ceil { amax_floor_bits },
                Calibration::Dynamic(DynamicScale::CeilPowerOfTwo {
                    amax_floor_bits: actual,
                }),
            ) => amax_floor_bits == *actual,
            (Self::Fp8Group128(mode), Calibration::Dynamic(DynamicScale::GroupAmax(actual))) => {
                mode == *actual
            }
            (Self::Mxfp8, Calibration::Dynamic(DynamicScale::MxPowerOfTwo))
            | (Self::Nvfp4TensorAmax, Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax)) => true,
            (Self::Nvfp4StaticScalar, Calibration::Static(scale)) => {
                scale.tensor.dtype == DType::F32
                    && scale.tensor.elements().map_err(|e| e.to_string())? == 1
                    && matches!(scale.scope, ScaleScope::Tensor | ScaleScope::Projection)
                    && scale.groups.is_empty()
                    && scale.convention == ScaleConvention::Multiply
            }
            _ => false,
        };
        if fits {
            Ok(())
        } else {
            Err("activation calibration requires another scope or quantizer; scalar NVFP4 compositions cannot execute per-expert input scales".into())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum GlobalScaleCapability {
    Absent,
    /// One scalar source reference, shared by fused projections and all experts.
    SharedScalar,
    /// Packed globals preserve each expert and projection independently.
    ExpertProjection,
}

/// A finite supplied implementation domain, not an operation graph. Shape is
/// the physical matrix per operand after model-owned slicing and padding.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct OperationCapability {
    pub kind: OperationKind,
    pub target: Target,
    pub usage: OperationUse,
    pub geometry: MatrixGeometry,
    pub operands: u8,
    pub weight: WeightFormat,
    pub activation: Option<ActivationCapability>,
    pub epilogues: Epilogues,
    pub layout: WeightLayout,
    /// Accepted source scale dtypes. Exact widening is part of packing, not
    /// a change of the checkpoint's source scale representation.
    pub local_scale_dtypes: Vec<DType>,
    pub globals: GlobalScaleCapability,
    pub shared_local_scale: bool,
    pub arithmetic: ArithmeticSpec,
    pub implementation: String,
    pub stages: Vec<String>,
}

impl OperationCapability {
    pub fn validate(&self) -> Result<(), String> {
        if self.operands == 0
            || self.operands > 2
            || self.geometry.rows == 0
            || self.geometry.k == 0
            || self.usage.capacity == 0
            || self
                .geometry
                .rows
                .checked_mul(self.geometry.k)
                .is_none_or(|n| n > super::MAX_TENSOR_BYTES)
            || self.stages.is_empty()
            || self.stages.len() > 32
            || self.local_scale_dtypes.len() > 3
            || (self.kind == OperationKind::Lookup) != self.activation.is_none()
            || self.weight == WeightFormat::Nvfp4 && self.globals == GlobalScaleCapability::Absent
            || self.weight != WeightFormat::Nvfp4 && self.globals != GlobalScaleCapability::Absent
            || matches!(self.weight, WeightFormat::Bf16 | WeightFormat::F32)
                && !self.local_scale_dtypes.is_empty()
            || !matches!(self.weight, WeightFormat::Bf16 | WeightFormat::F32)
                && self.local_scale_dtypes.is_empty()
            || self.arithmetic.accumulator != DType::F32
        {
            return Err("invalid supplied operation capability".into());
        }
        if self.activation == Some(ActivationCapability::F32) {
            if self.kind != OperationKind::Dense
                || self.operands != 1
                || !matches!(self.weight, WeightFormat::Bf16 | WeightFormat::F32)
                || self.arithmetic.weight_operand != DType::F32
                || self.arithmetic.output != DType::F32
                || self.arithmetic.decode_order != super::DecodeOrder::ExactF32Operands
                || self.epilogues != Epilogues::default()
            {
                return Err("F32 capability must preserve exact F32 inputs and F32 output".into());
            }
        } else if self.weight == WeightFormat::F32
            || self.arithmetic.decode_order == super::DecodeOrder::ExactF32Operands
        {
            return Err("raw F32 weights and exact F32 arithmetic require F32 activations".into());
        }
        if self.operands == 2 && self.epilogues.swiglu.is_none() {
            return Err("two supplied operands require a declared projection fusion".into());
        }
        for name in [
            &self.implementation,
            &self.arithmetic.numerical_contract,
            &self.layout.id,
        ]
        .into_iter()
        .chain(&self.stages)
        {
            super::validate_name(name).map_err(|e| e.to_string())?;
        }
        let unique: BTreeSet<_> = self.stages.iter().collect();
        if unique.len() != self.stages.len() || self.layout.version != 1 {
            return Err("duplicate stage or unsupported supplied layout version".into());
        }
        Ok(())
    }

    fn require(&self, operation: &OperationSpec) -> Result<(), String> {
        if let Some(activation) = self.activation {
            for input in &operation.activations {
                activation.require(input)?;
            }
            if operation.activations.windows(2).any(|p| p[0] != p[1]) {
                return Err("supplied fused projection requires identical gate/up activation recipes and calibration".into());
            }
        }
        let first = &operation.weights[0];
        for weight in &operation.weights {
            if weight.layout != WeightLayout::row_major() && weight.layout != self.layout {
                return Err(format!(
                    "tensor {}: unsupported physical layout {}",
                    weight.codes.tensor, weight.layout.id
                ));
            }
            if weight
                .local_scale
                .as_ref()
                .is_some_and(|s| !self.local_scale_dtypes.contains(&s.tensor.dtype))
            {
                return Err(format!(
                    "tensor {}: supplied scale dtype differs from {:?}",
                    weight.codes.tensor, self.local_scale_dtypes
                ));
            }
            if self.shared_local_scale && weight.local_scale != first.local_scale {
                return Err("fused per-tensor FP8 requires one shared scale; distinct projection scales need separate projections".into());
            }
            if self.globals == GlobalScaleCapability::SharedScalar
                && (weight.global_scale != first.global_scale
                    || weight.global_scale.as_ref().is_none_or(|s| {
                        s.tensor.dtype != DType::F32
                            || s.tensor.elements().ok() != Some(1)
                            || s.scope == ScaleScope::Expert
                    }))
            {
                return Err("NVFP4 execution requires proven bank-uniform weight globals and a shared gate/up activation recipe; per-expert globals support an ExpertProjection BF16 binding".into());
            }
        }
        let mut expected = operation.arithmetic.clone();
        expected.numerical_contract = self.arithmetic.numerical_contract.clone();
        if expected != self.arithmetic {
            return Err(format!(
                "arithmetic requires {:?} weight operands, {:?} accumulation and {:?} decode order with {:?} output",
                self.arithmetic.weight_operand,
                self.arithmetic.accumulator,
                self.arithmetic.decode_order,
                self.arithmetic.output
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct CapabilityDocument {
    pub version: u32,
    pub operations: Vec<OperationCapability>,
}

impl CapabilityDocument {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1 || self.operations.is_empty() || self.operations.len() > 8192 {
            return Err("invalid supplied capability version or count".into());
        }
        for operation in &self.operations {
            operation.validate()?;
        }
        Ok(())
    }
    pub fn entries(&self) -> BTreeSet<String> {
        self.operations
            .iter()
            .flat_map(|o| o.stages.iter().cloned())
            .collect()
    }
}

/// `dependency` and `available` must come from the publisher's authenticated,
/// audited artifact. The handle is an index into that retained inventory.
pub struct CapabilityInventory {
    document: CapabilityDocument,
    available: BTreeSet<String>,
    dependency: String,
}

impl CapabilityInventory {
    pub fn new(
        document: CapabilityDocument,
        available: BTreeSet<String>,
        dependency: String,
    ) -> Result<Self, String> {
        document.validate()?;
        if available.len() > 32768
            || dependency.len() != 64
            || !dependency
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err("invalid supplied implementation inventory or native digest".into());
        }
        Ok(Self {
            document,
            available,
            dependency,
        })
    }
    pub fn document(&self) -> &CapabilityDocument {
        &self.document
    }

    /// Whether a supplied implementation could bind `operation` at all: a
    /// valid operation whose weights use normalized even-low nibble packing,
    /// no required transforms, multiplying scales and group origins on group
    /// boundaries. `bind_matching` refuses an operation that fails this
    /// before it looks for a capability.
    pub fn admits(&self, operation: &OperationSpec) -> Result<(), String> {
        operation.validate().map_err(|e| e.to_string())?;
        for weight in &operation.weights {
            if weight.encoding == CodeEncoding::E2M1(NibbleOrder::EvenHigh) {
                return Err(format!(
                    "tensor {}: supplied kernels require normalized even-low nibble packing",
                    weight.codes.tensor
                ));
            }
            if !weight.transforms.is_empty()
                || weight
                    .local_scale
                    .iter()
                    .chain(&weight.global_scale)
                    .any(|s| {
                        s.convention != ScaleConvention::Multiply
                            || s.groups.iter().any(|g| g.origin % u64::from(g.size) != 0)
                    })
            {
                return Err(format!(
                    "tensor {}: required transform, scale convention or group origin is unsupported by supplied implementations",
                    weight.codes.tensor
                ));
            }
        }
        Ok(())
    }

    /// Narrow a supplied domain to the caller's operation composition, for
    /// example a projection port of an existing multi-output fusion. The
    /// predicate cannot add capabilities, entries or dependency identities.
    /// Returned handles still index this inventory, not a copied subset.
    pub fn bind_matching(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
        accepts: impl Fn(&OperationCapability) -> bool,
    ) -> Result<Binding<usize>, String> {
        self.admits(operation)?;
        let geometry = operation.matrix_geometry().map_err(|e| e.to_string())?;
        let mut reasons = BTreeSet::new();
        for (index, capability) in self
            .document
            .operations
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.usage == usage
                    && accepts(c)
                    && c.target == operation.target
                    && c.kind == operation.kind
                    && c.geometry == geometry
                    && usize::from(c.operands) == operation.weights.len()
                    && c.weight == operation.weights[0].format
                    && c.epilogues == operation.epilogues
            })
        {
            if let Err(reason) = capability.require(operation) {
                reasons.insert(reason);
                continue;
            }
            let missing: Vec<_> = capability
                .stages
                .iter()
                .filter(|s| !self.available.contains(*s))
                .map(String::as_str)
                .collect();
            if !missing.is_empty() {
                reasons.insert(format!(
                    "missing supplied compiled entries: {}",
                    missing.join(", ")
                ));
                continue;
            }
            return Ok(Binding {
                identity: ImplementationIdentity {
                    implementation: capability.implementation.clone(),
                    dependency: self.dependency.clone(),
                    weight_layouts: vec![capability.layout.clone(); operation.weights.len()],
                    activations: operation.activations.clone(),
                    arithmetic: capability.arithmetic.clone(),
                    stages: capability.stages.clone(),
                },
                handle: index,
            });
        }
        if reasons.is_empty() {
            reasons.insert(format!(
                "no supplied implementation for {:?} {:?} N={} K={} operands={} on {:?} at this phase/capacity/graph use with {:?}",
                operation.kind, operation.weights[0].format, geometry.rows, geometry.k,
                operation.weights.len(), operation.target, operation.epilogues
            ));
        }
        Err(reasons.into_iter().take(8).collect::<Vec<_>>().join("; "))
    }
}

impl ImplementationProvider for CapabilityInventory {
    type Handle = usize;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<usize>, String> {
        self.bind_matching(operation, usage, |_| true)
    }
}
