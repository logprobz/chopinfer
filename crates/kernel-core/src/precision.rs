//! Portable checkpoint precision descriptions. These describe data and arithmetic,
//! not kernel availability. A consumer must bind the complete operation contract.

pub mod buffers;
pub mod capability;
#[cfg(feature = "precision-serde")]
pub mod component;
pub mod execution;
pub mod f32_gemm;
pub mod format;
pub mod layout;
pub mod packed;
pub mod quantizer;
pub mod scale_expand;
pub mod scale_offset;

use crate::contract::{ActivationFormat, DType, Epilogues, ScaleMode, Target, WeightFormat};
use std::fmt;

pub const PRECISION_VERSION: u32 = 1;
pub const MAX_TENSOR_RANK: usize = 5;
pub const MAX_TENSOR_BYTES: u64 = 1 << 40;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrecisionError {
    pub tensor: String,
    pub field: &'static str,
    pub reason: String,
}

impl PrecisionError {
    pub fn new(tensor: &str, field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            tensor: tensor.into(),
            field,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for PrecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor {}: {}: {}", self.tensor, self.field, self.reason)
    }
}

impl std::error::Error for PrecisionError {}

/// A named tensor span. The owner authenticates the source or component that
/// contains it; offsets are relative to that owner's bounded byte view.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct TensorSpan {
    pub tensor: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub byte_length: u64,
}

impl TensorSpan {
    pub fn elements(&self) -> Result<u64, PrecisionError> {
        shape_elements(&self.tensor, &self.shape)
    }

    pub fn validate(&self) -> Result<(), PrecisionError> {
        validate_name(&self.tensor)?;
        let bits = dtype_bits(self.dtype);
        let size = self
            .elements()?
            .checked_mul(bits)
            .and_then(|v| v.checked_add(7))
            .map(|v| v / 8);
        if size != Some(self.byte_length)
            || self.byte_length > MAX_TENSOR_BYTES
            || self.offset.checked_add(self.byte_length).is_none()
        {
            return Err(PrecisionError::new(
                &self.tensor,
                "span",
                "shape, dtype or byte bounds disagree",
            ));
        }
        Ok(())
    }
}

pub fn dtype_bits(dtype: DType) -> u64 {
    match dtype {
        DType::Bf16 | DType::F16 => 16,
        DType::F32 | DType::I32 => 32,
        DType::I64 => 64,
        DType::Fp4E2M1 => 4,
        DType::Fp6E2M3 | DType::Fp6E3M2 => 6,
        _ => 8,
    }
}

fn validate_name(name: &str) -> Result<(), PrecisionError> {
    if name.is_empty() || name.len() > 1024 || name.chars().any(char::is_control) {
        return Err(PrecisionError::new(
            name,
            "name",
            "expected 1..=1024 non-control bytes",
        ));
    }
    Ok(())
}

pub fn shape_elements(tensor: &str, shape: &[u64]) -> Result<u64, PrecisionError> {
    if shape.len() > MAX_TENSOR_RANK || shape.contains(&0) {
        return Err(PrecisionError::new(
            tensor,
            "shape",
            "invalid rank or zero dimension",
        ));
    }
    shape.iter().try_fold(1_u64, |n, &v| {
        n.checked_mul(v)
            .filter(|&size| size <= MAX_TENSOR_BYTES * 2)
            .ok_or_else(|| {
                PrecisionError::new(tensor, "shape", "element count exceeds the supported bound")
            })
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum NibbleOrder {
    EvenLow,
    EvenHigh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum CodeEncoding {
    Bf16LittleEndian,
    E4M3,
    E2M1(NibbleOrder),
    F32LittleEndian,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ScaleConvention {
    Multiply,
    Reciprocal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ScaleScope {
    Tensor,
    Channel,
    Block,
    Expert,
    Projection,
    Table,
}

/// Group origin is in logical source coordinates. Slicing a tensor must not
/// silently restart its source scale grid at a rank boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct GroupAxis {
    pub axis: u8,
    pub size: u32,
    pub origin: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct ScaleSpec {
    pub tensor: TensorSpan,
    pub scope: ScaleScope,
    pub convention: ScaleConvention,
    pub groups: Vec<GroupAxis>,
}

impl ScaleSpec {
    pub fn validate(&self, logical_shape: &[u64]) -> Result<(), PrecisionError> {
        shape_elements(&self.tensor.tensor, logical_shape)?;
        self.tensor.validate()?;
        if !matches!(
            self.tensor.dtype,
            DType::Bf16 | DType::F32 | DType::Fp8E4M3 | DType::Ue4M3 | DType::Ue8M0
        ) {
            return Err(PrecisionError::new(
                &self.tensor.tensor,
                "scale dtype",
                "unsupported scale encoding",
            ));
        }
        let mut axes = 0_u8;
        for group in &self.groups {
            if usize::from(group.axis) >= logical_shape.len()
                || group.size == 0
                || axes & (1 << group.axis) != 0
                || group
                    .origin
                    .checked_add(logical_shape[usize::from(group.axis)])
                    .is_none()
            {
                return Err(PrecisionError::new(
                    &self.tensor.tensor,
                    "grouping",
                    "invalid or repeated axis, size or origin",
                ));
            }
            axes |= 1 << group.axis;
        }
        if matches!(
            self.scope,
            ScaleScope::Tensor | ScaleScope::Projection | ScaleScope::Table
        ) && (!self.groups.is_empty() || self.tensor.elements()? != 1)
        {
            return Err(PrecisionError::new(
                &self.tensor.tensor,
                "scope",
                "a scalar scale requires exactly one element and no grouping",
            ));
        }
        Ok(())
    }
}

/// Importers retain ordered execution requirements even when the current
/// implementation cannot execute them. Policy overrides cannot discard them.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum RequiredTransform {
    InputScale(ScaleSpec),
    Rotation {
        identity: String,
        tensor: TensorSpan,
    },
    Unsupported {
        field: String,
        value: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct WeightLayout {
    pub id: String,
    pub version: u32,
}

impl WeightLayout {
    pub fn row_major() -> Self {
        Self {
            id: "row-major".into(),
            version: 1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct WeightSpec {
    pub codes: TensorSpan,
    pub logical_shape: Vec<u64>,
    pub format: WeightFormat,
    pub encoding: CodeEncoding,
    pub layout: WeightLayout,
    pub local_scale: Option<ScaleSpec>,
    pub global_scale: Option<ScaleSpec>,
    pub transforms: Vec<RequiredTransform>,
}

impl WeightSpec {
    pub fn validate(&self) -> Result<(), PrecisionError> {
        self.codes.validate()?;
        shape_elements(&self.codes.tensor, &self.logical_shape)?;
        if !(2..=3).contains(&self.logical_shape.len()) {
            return Err(PrecisionError::new(
                &self.codes.tensor,
                "logical shape",
                "expected a matrix or a uniform expert bank",
            ));
        }
        validate_name(&self.layout.id)?;
        if self.layout.version == 0 {
            return Err(PrecisionError::new(
                &self.codes.tensor,
                "layout",
                "layout version must be nonzero",
            ));
        }
        let (encoding, dtype, local, global) = match self.format {
            WeightFormat::Bf16 => (CodeEncoding::Bf16LittleEndian, DType::Bf16, false, false),
            WeightFormat::F32 => (CodeEncoding::F32LittleEndian, DType::F32, false, false),
            WeightFormat::Nvfp4 => (
                CodeEncoding::E2M1(NibbleOrder::EvenLow),
                DType::U8,
                true,
                true,
            ),
            WeightFormat::Mxfp4 => (
                CodeEncoding::E2M1(NibbleOrder::EvenLow),
                DType::U8,
                true,
                false,
            ),
            _ => (CodeEncoding::E4M3, DType::Fp8E4M3, true, false),
        };
        // Both nibble orders are describable; packing must normalize them.
        let encoding_matches = self.encoding == encoding
            || matches!(
                (self.encoding, encoding),
                (CodeEncoding::E2M1(_), CodeEncoding::E2M1(_))
            );
        if !encoding_matches
            || self.codes.dtype != dtype
            || self.local_scale.is_some() != local
            || self.global_scale.is_some() != global
        {
            return Err(PrecisionError::new(
                &self.codes.tensor,
                "encoding",
                "code dtype or scale companions disagree with weight format",
            ));
        }
        if self.layout == WeightLayout::row_major() {
            let mut shape = self.logical_shape.clone();
            if matches!(self.encoding, CodeEncoding::E2M1(_)) {
                let k = shape.last_mut().unwrap();
                if !k.is_multiple_of(2) {
                    return Err(PrecisionError::new(
                        &self.codes.tensor,
                        "packing",
                        "E2M1 rows require an even logical K",
                    ));
                }
                *k /= 2;
            }
            if self.codes.shape != shape {
                return Err(PrecisionError::new(
                    &self.codes.tensor,
                    "packing",
                    "stored shape disagrees with logical row-major shape",
                ));
            }
        }
        for scale in self.local_scale.iter().chain(self.global_scale.iter()) {
            scale.validate(&self.logical_shape)?;
        }
        self.validate_scale_grid()?;
        Ok(())
    }

    fn validate_scale_grid(&self) -> Result<(), PrecisionError> {
        let Some(local) = &self.local_scale else {
            return Ok(());
        };
        let fail = |reason| PrecisionError::new(&self.codes.tensor, "scale grid", reason);
        let rank = self.logical_shape.len();
        let (rows, k) = match self.format {
            WeightFormat::Bf16 | WeightFormat::F32 => unreachable!(),
            WeightFormat::Fp8PerTensor => (0, 0),
            WeightFormat::Fp8PerChannel => (1, 0),
            WeightFormat::Fp8Block128x128(_) => (128, 128),
            WeightFormat::Fp8Rows64x128(_) => (64, 128),
            WeightFormat::Fp8Block32x32 => (32, 32),
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => (1, 32),
            WeightFormat::Nvfp4 => (1, 16),
        };
        let expected_dtype = match self.format {
            WeightFormat::Nvfp4 => matches!(local.tensor.dtype, DType::Fp8E4M3 | DType::Ue4M3),
            WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => {
                local.tensor.dtype == DType::Ue8M0
            }
            WeightFormat::Fp8PerTensor => matches!(local.tensor.dtype, DType::F32 | DType::Bf16),
            // BF16 export scales widen exactly during preparation. Their
            // source encoding stays distinct from the kernel's F32 buffer.
            _ => matches!(local.tensor.dtype, DType::F32 | DType::Bf16),
        };
        if !expected_dtype {
            return Err(fail("scale dtype disagrees with weight encoding"));
        }
        if rows == 0 {
            if local.tensor.elements()? != 1
                || !local.groups.is_empty()
                || !matches!(
                    local.scope,
                    ScaleScope::Tensor | ScaleScope::Projection | ScaleScope::Table
                )
            {
                return Err(fail("per-tensor encoding requires a scalar scale"));
            }
        } else {
            let axes: Vec<_> = [(rank - 2, rows), (rank - 1, k)]
                .into_iter()
                .filter(|(_, size)| *size != 0)
                .collect();
            if local.groups.len() != axes.len()
                || axes.iter().any(|&(axis, size)| {
                    !local
                        .groups
                        .iter()
                        .any(|g| usize::from(g.axis) == axis && g.size == size)
                })
                || local.scope
                    != if k == 0 {
                        ScaleScope::Channel
                    } else {
                        ScaleScope::Block
                    }
            {
                return Err(fail("scale grouping disagrees with weight encoding"));
            }
            let mut shape = self.logical_shape.clone();
            if k == 0 {
                *shape.last_mut().unwrap() = 1;
            }
            for group in &local.groups {
                let dimension = &mut shape[usize::from(group.axis)];
                *dimension = (group.origin + *dimension - 1) / u64::from(group.size)
                    - group.origin / u64::from(group.size)
                    + 1;
            }
            if local.tensor.elements()? != shape_elements(&self.codes.tensor, &shape)? {
                return Err(fail(
                    "scale element count disagrees with grouping and logical shape",
                ));
            }
        }
        if let Some(global) = &self.global_scale {
            let count = global.tensor.elements()?;
            let scoped = match global.scope {
                ScaleScope::Tensor | ScaleScope::Projection | ScaleScope::Table => count == 1,
                ScaleScope::Expert => rank == 3 && count == self.logical_shape[0],
                _ => false,
            };
            if global.tensor.dtype != DType::F32 || !global.groups.is_empty() || !scoped {
                return Err(fail(
                    "NVFP4 globals require F32 scalars or one F32 scalar per expert and projection",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Rounding {
    NearestEven,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Clipping {
    FiniteSaturate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub enum DynamicScale {
    GroupAmax(ScaleMode),
    /// OCP MX floor rule. This is distinct from the ceiling rule below.
    MxPowerOfTwo,
    Nvfp4TensorAmax,
    /// Round inputs to BF16 before amax. Use the exponent-bit ceiling of
    /// RN32(max(amax, floor) * RN32(1 / code_max)), stored as UE8M0.
    CeilPowerOfTwo {
        amax_floor_bits: u32,
    },
}

impl DynamicScale {
    pub const MXFP8_CEIL: Self = Self::CeilPowerOfTwo {
        amax_floor_bits: 1e-4_f32.to_bits(),
    };

    pub fn validate(self) -> Result<(), PrecisionError> {
        if let Self::CeilPowerOfTwo { amax_floor_bits } = self {
            let floor = f32::from_bits(amax_floor_bits);
            // The exponent-bit rule operates on normal positive ratios.
            // E4M3 has the largest code range among these MX formats.
            if !floor.is_finite() || floor * (1.0 / 448.0) < f32::MIN_POSITIVE {
                return Err(PrecisionError::new(
                    "quantizer",
                    "amax floor",
                    "ceil scale requires a finite positive floor with a normal F32 ratio",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Calibration {
    Dynamic(DynamicScale),
    /// This is the exported multiplicative global, not an amax. Do not
    /// divide it by the code range again.
    Static(ScaleSpec),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct QuantizerSpec {
    pub calibration: Calibration,
    pub groups: Vec<GroupAxis>,
    pub rounding: Rounding,
    pub clipping: Clipping,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct ActivationSpec {
    pub format: ActivationFormat,
    pub storage: DType,
    pub compute: DType,
    pub quantizer: Option<QuantizerSpec>,
    pub transforms: Vec<RequiredTransform>,
}

impl ActivationSpec {
    /// Unquantized F32 input. Implementations must preserve all input bits.
    pub fn f32() -> Self {
        Self {
            format: ActivationFormat::F32,
            storage: DType::F32,
            compute: DType::F32,
            quantizer: None,
            transforms: Vec::new(),
        }
    }

    /// Trained E4M3/32 quantization with BF16 input rounding and a ceil scale.
    pub fn mxfp8_ceil(amax_floor_bits: u32) -> Self {
        Self {
            format: ActivationFormat::Mxfp8,
            storage: DType::Bf16,
            compute: DType::Fp8E4M3,
            quantizer: Some(QuantizerSpec {
                calibration: Calibration::Dynamic(DynamicScale::CeilPowerOfTwo { amax_floor_bits }),
                groups: vec![GroupAxis {
                    axis: 1,
                    size: 32,
                    origin: 0,
                }],
                rounding: Rounding::NearestEven,
                clipping: Clipping::FiniteSaturate,
            }),
            transforms: Vec::new(),
        }
    }

    pub fn bf16() -> Self {
        Self {
            format: ActivationFormat::Bf16,
            storage: DType::Bf16,
            compute: DType::Bf16,
            quantizer: None,
            transforms: Vec::new(),
        }
    }

    pub fn is_bf16(&self) -> bool {
        self.format == ActivationFormat::Bf16
            && self.storage == DType::Bf16
            && self.compute == DType::Bf16
            && self.quantizer.is_none()
    }

    pub fn is_f32(&self) -> bool {
        self.format == ActivationFormat::F32
            && self.storage == DType::F32
            && self.compute == DType::F32
            && self.quantizer.is_none()
    }

    pub fn validate(&self, tensor: &str, shape: &[u64]) -> Result<(), PrecisionError> {
        shape_elements(tensor, shape)?;
        let expected = match self.format {
            ActivationFormat::Bf16 => DType::Bf16,
            ActivationFormat::F32 => DType::F32,
            ActivationFormat::Fp8Group128(_) | ActivationFormat::Mxfp8 => DType::Fp8E4M3,
            ActivationFormat::Nvfp4 => DType::Fp4E2M1,
        };
        if self.compute != expected
            || (self.storage != expected && self.storage != DType::Bf16)
            || (matches!(self.format, ActivationFormat::Bf16 | ActivationFormat::F32))
                != self.quantizer.is_none()
            || (self.format == ActivationFormat::F32
                && (self.storage != DType::F32 || !self.transforms.is_empty()))
        {
            return Err(PrecisionError::new(
                tensor,
                "activation",
                "storage, compute and quantizer contracts disagree",
            ));
        }
        if let Some(quantizer) = &self.quantizer {
            if let Calibration::Dynamic(rule) = quantizer.calibration {
                rule.validate()?;
            }
            if let Calibration::Static(scale) = &quantizer.calibration {
                scale.validate(shape)?;
            }
            let size = match self.format {
                ActivationFormat::Fp8Group128(_) => 128,
                ActivationFormat::Mxfp8 => 32,
                ActivationFormat::Nvfp4 => 16,
                ActivationFormat::Bf16 | ActivationFormat::F32 => unreachable!(),
            };
            if quantizer.groups.len() != 1
                || quantizer.groups[0].size != size
                || usize::from(quantizer.groups[0].axis) + 1 != shape.len()
                || quantizer.groups[0].origin != 0
            {
                return Err(PrecisionError::new(
                    tensor,
                    "activation grouping",
                    "expected complete row groups on the final axis",
                ));
            }
            let supported = matches!(
                (self.format, &quantizer.calibration),
                (
                    ActivationFormat::Fp8Group128(_),
                    Calibration::Dynamic(DynamicScale::GroupAmax(_))
                ) | (
                    ActivationFormat::Mxfp8,
                    Calibration::Dynamic(
                        DynamicScale::MxPowerOfTwo | DynamicScale::CeilPowerOfTwo { .. }
                    )
                ) | (
                    ActivationFormat::Nvfp4,
                    Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax)
                ) | (ActivationFormat::Nvfp4, Calibration::Static(_))
            );
            if !supported
                || matches!((self.format, &quantizer.calibration),
                (ActivationFormat::Fp8Group128(a), Calibration::Dynamic(DynamicScale::GroupAmax(b))) if a != *b)
            {
                return Err(PrecisionError::new(
                    tensor,
                    "activation calibration",
                    "calibration does not describe this quantizer",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DecodeOrder {
    /// RN32(code * RN32(local * global)), then BF16 RNE (dense engine).
    ScaleThenCode,
    /// RN32(RN32(code * local) * global), then BF16 RNE (grouped engine).
    CodeThenScale,
    /// Exact BF16(code * local); apply the NVFP4 F32 global once after the
    /// complete F32 reduction, before residuals, SwiGLU or route weighting.
    ExactLocalThenGlobal,
    /// Exact BF16(code * power-of-two scale). Reject MXFP8 and MXFP4 payload values
    /// outside the exactly representable BF16 domain, including underflow.
    ExactScaledOperand,
    /// NVFP4 lookup: round the exact code * local * global directly to the
    /// final BF16 output. There is no F32 or BF16 weight intermediate.
    ExactWeightToBf16,
    /// Convert FP8 codes exactly to BF16, accumulate one K block, then apply
    /// its weight scale in F32. There is no scaled BF16 weight intermediate.
    BlockScaleAfterMma,
    /// Accumulate raw FP8 operands per K block, then multiply by
    /// RN32(activation_scale * weight_scale) before the outer F32 sum.
    ScaleProductAfterBlockMma,
    /// Accumulate raw FP8 operands per K block, then multiply by the weight
    /// scale and activation scale in that order before the outer F32 sum.
    SuccessiveScalesAfterBlockMma,
    /// The native block-scaled MMA consumes the code and local scale operands.
    NativeBlockScaledMma,
    /// Native block-scaled MMA followed by the F32 product of the tensor-global
    /// weight and activation scales, before the output epilogue.
    NativeBlockScaledMmaThenGlobals,
    /// F32 activations and exactly widened BF16 or raw F32 weights. No
    /// quantizer or reduced-precision operand rounding is permitted.
    ExactF32Operands,
}

impl DecodeOrder {
    pub fn scales_before_bf16_mma(self) -> bool {
        matches!(
            self,
            Self::ScaleThenCode | Self::CodeThenScale | Self::ExactWeightToBf16
        )
    }

    pub fn has_bf16_weight_operand(self) -> bool {
        self.scales_before_bf16_mma()
            || matches!(self, Self::ExactLocalThenGlobal | Self::ExactScaledOperand)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct ArithmeticSpec {
    pub weight_operand: DType,
    pub accumulator: DType,
    pub output: DType,
    pub decode_order: DecodeOrder,
    pub rounding: Rounding,
    pub numerical_contract: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum OperationKind {
    Dense,
    Grouped {
        experts: u32,
        top_k: u32,
    },
    Lookup,
    /// `groups` independent dense matrices of the operation's geometry, each
    /// over its own K-wide activation slice, without routing or a sum across
    /// groups: the grouped fold GEMM contract (`mxfp8_gemm`), which no dense
    /// or expert-bank capability admits.
    GroupedDense {
        groups: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    Verify,
}

/// Graph is a scheduling property, not another precision recipe. A graph
/// bucket can use a prefill implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct OperationUse {
    pub phase: ExecutionPhase,
    pub capacity: u32,
    pub graph: bool,
}

/// Rank-local matrix geometry after the model's declared slicing and padding.
/// Source weights keep their original logical shape and encoding. The owning
/// placement plan authenticates exact row/K selections and zero padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct MatrixGeometry {
    pub rows: u64,
    pub k: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct OperationSpec {
    pub id: String,
    pub kind: OperationKind,
    pub target: Target,
    pub weights: Vec<WeightSpec>,
    #[cfg_attr(
        feature = "precision-serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub geometry: Option<MatrixGeometry>,
    /// One input recipe per weight operand. Fused gate/up can retain different
    /// calibrations. Lookup consumes integer IDs and leaves this empty.
    pub activations: Vec<ActivationSpec>,
    pub arithmetic: ArithmeticSpec,
    pub epilogues: Epilogues,
    pub uses: Vec<OperationUse>,
}

impl OperationSpec {
    pub fn matrix_geometry(&self) -> Result<MatrixGeometry, PrecisionError> {
        let first = self
            .weights
            .first()
            .ok_or_else(|| PrecisionError::new(&self.id, "geometry", "missing weight operand"))?;
        if first.logical_shape.len() < 2 {
            return Err(PrecisionError::new(
                &self.id,
                "geometry",
                "weight is not a matrix",
            ));
        }
        let shape = self.geometry.unwrap_or(MatrixGeometry {
            rows: first.logical_shape[first.logical_shape.len() - 2],
            k: *first.logical_shape.last().unwrap(),
        });
        shape_elements(&self.id, &[shape.rows, shape.k])?;
        Ok(shape)
    }
    pub fn validate(&self) -> Result<(), PrecisionError> {
        validate_name(&self.id)?;
        if self.weights.is_empty()
            || self.weights.len() > 2
            || self.uses.is_empty()
            || self.uses.len() > 256
            || self.uses.iter().any(|u| u.capacity == 0)
            || self
                .uses
                .iter()
                .enumerate()
                .any(|(i, u)| self.uses[..i].contains(u))
        {
            return Err(PrecisionError::new(
                &self.id,
                "operation",
                "invalid operands or phase/capacity coverage",
            ));
        }
        for weight in &self.weights {
            weight.validate()?;
        }
        let geometry = self.matrix_geometry()?;
        let first = &self.weights[0];
        let expected_inputs = if self.kind == OperationKind::Lookup {
            0
        } else {
            self.weights.len()
        };
        if self.activations.len() != expected_inputs {
            return Err(PrecisionError::new(
                &self.id,
                "activation",
                "lookup uses integer IDs; projections require an activation recipe",
            ));
        }
        for activation in &self.activations {
            activation.validate(&self.id, &[1, geometry.k])?;
        }
        if let OperationKind::Grouped { experts, top_k } = self.kind {
            if experts == 0
                || top_k == 0
                || top_k > experts
                || self.weights.iter().any(|w| {
                    w.logical_shape.len() != 3
                        || if self.geometry.is_some() {
                            w.logical_shape[0] < u64::from(experts)
                        } else {
                            w.logical_shape[0] != u64::from(experts)
                        }
                })
            {
                return Err(PrecisionError::new(
                    &self.id,
                    "experts",
                    "bank shape or routing geometry disagrees",
                ));
            }
        } else if let OperationKind::GroupedDense { groups } = self.kind
            && (groups < 2
                || self.geometry.is_none()
                || self.weights.iter().any(|w| w.logical_shape.len() != 2))
        {
            return Err(PrecisionError::new(
                &self.id,
                "groups",
                "grouped dense needs two or more groups of one stacked matrix at a declared rank geometry",
            ));
        } else if self.weights.iter().any(|w| w.logical_shape.len() != 2) {
            return Err(PrecisionError::new(
                &self.id,
                "shape",
                "dense projections and lookups require matrices",
            ));
        }
        if self
            .weights
            .iter()
            .any(|w| w.logical_shape != first.logical_shape || w.format != first.format)
        {
            return Err(PrecisionError::new(
                &self.id,
                "fusion",
                "fused operands require matching geometry and encoding",
            ));
        }
        if self.arithmetic.accumulator != DType::F32
            || !matches!(self.arithmetic.output, DType::Bf16 | DType::F32)
        {
            return Err(PrecisionError::new(
                &self.id,
                "arithmetic",
                "unsupported accumulator or output precision",
            ));
        }
        if self
            .activations
            .iter()
            .any(|a| a.format == ActivationFormat::F32)
        {
            if self.kind != OperationKind::Dense
                || self.weights.len() != 1
                || !matches!(first.format, WeightFormat::Bf16 | WeightFormat::F32)
                || self.activations.iter().any(|a| !a.is_f32())
                || self.arithmetic.weight_operand != DType::F32
                || self.arithmetic.output != DType::F32
                || self.arithmetic.decode_order != DecodeOrder::ExactF32Operands
                || self.epilogues != Epilogues::default()
            {
                return Err(PrecisionError::new(
                    &self.id,
                    "F32 arithmetic",
                    "F32 inputs require a dense unquantized BF16 or F32 matrix, exact F32 operands and F32 output without epilogues",
                ));
            }
        } else if first.format == WeightFormat::F32
            || self.arithmetic.decode_order == DecodeOrder::ExactF32Operands
        {
            return Err(PrecisionError::new(
                &self.id,
                "F32 arithmetic",
                "raw F32 weights and exact F32 arithmetic require F32 activations",
            ));
        }
        validate_name(&self.arithmetic.numerical_contract)?;
        Ok(())
    }
}
