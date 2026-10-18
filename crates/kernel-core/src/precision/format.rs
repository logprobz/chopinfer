//! Format-owned arithmetic, companion bindings, physical layouts and lossless slices.
//! Kernel availability remains the capability provider's responsibility.

use super::execution::BlockScaledExecution;
use super::packed::{MatrixSlice, MatrixView, PackedMatrix};
use super::{DecodeOrder, NibbleOrder, PrecisionError, ScaleScope, WeightLayout, WeightSpec};
use crate::contract::{ActivationFormat, DType, WeightFormat};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(rename_all = "snake_case"))]
pub enum FormatId {
    Bf16,
    Fp8PerTensor,
    Fp8PerChannel,
    Fp8Block128,
    Fp8Rows64,
    Mxfp8,
    Mxfp4,
    Nvfp4,
    Nvfp4WeightOnly,
    Fp8Block32,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelFamily {
    Dense,
    GroupedExpert,
    Row,
    Lookup,
    LayoutCopy,
}

/// Block FP8 has two existing A16 recipes. The declaration must retain the
/// operation's recipe rather than infer it from the checkpoint encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(rename_all = "snake_case"))]
pub enum A16Recipe {
    ScaledWeight,
    ScaleAfterBlockMma,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(rename_all = "snake_case"))]
pub enum SliceAxis {
    Rows,
    K,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddingTile {
    pub rows: usize,
    pub k: usize,
}

impl PaddingTile {
    pub const NONE: Self = Self { rows: 1, k: 1 };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SliceRequest {
    pub axis: SliceAxis,
    pub range: Range<usize>,
    pub padding: PaddingTile,
    pub nibble_order: NibbleOrder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompanionRole {
    LocalScale,
    GlobalScale,
}

/// Imported and launch dtypes differ only for BF16 scales widened to F32.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompanionBinding {
    pub role: CompanionRole,
    pub tensor: String,
    pub source_dtype: DType,
    pub launch_dtype: DType,
    pub scope: ScaleScope,
    pub convention: super::ScaleConvention,
    pub groups: Vec<super::GroupAxis>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayoutChoice {
    pub weight: WeightLayout,
    pub codes: &'static str,
    pub local_scales: Option<&'static str>,
    pub global_scales: Option<&'static str>,
}

/// Existing aliases and legacy fragment transforms are explicit variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutVariant {
    Standard,
    AlternateBlockScaleNames,
    Mxfp4Strips { fused_gate_up: bool },
}

pub trait FormatPlugin: Sync {
    /// Geometry required by every phase of the standard family implementation.
    /// Legacy implementations supply their own geometry through a provider.
    fn padding_tile(&self, family: KernelFamily) -> PaddingTile {
        if self.id() == FormatId::Bf16 && family == KernelFamily::Dense {
            PaddingTile { rows: 64, k: 1 }
        } else {
            PaddingTile::NONE
        }
    }

    /// Local scale group sizes. Zero means the complete tensor or K axis.
    fn local_scale_groups(&self) -> (u64, u64) {
        match self.id() {
            FormatId::Bf16 | FormatId::F32 | FormatId::Fp8PerTensor => (0, 0),
            FormatId::Fp8PerChannel => (1, 0),
            FormatId::Fp8Block128 => (128, 128),
            FormatId::Fp8Rows64 => (64, 128),
            FormatId::Fp8Block32 => (32, 32),
            FormatId::Mxfp8 | FormatId::Mxfp4 => (1, 32),
            FormatId::Nvfp4 | FormatId::Nvfp4WeightOnly => (1, 16),
        }
    }
    fn id(&self) -> FormatId;
    fn accepts(&self, format: WeightFormat) -> bool;
    fn decode_order(
        &self,
        family: KernelFamily,
        activation: ActivationFormat,
        recipe: A16Recipe,
    ) -> Result<DecodeOrder, PrecisionError>;

    /// Select an explicit instruction path for MX operands. Legacy callers
    /// keep `decode_order`; new bindings declare and check this choice.
    fn decode_order_for_execution(
        &self,
        family: KernelFamily,
        activation: ActivationFormat,
        execution: BlockScaledExecution,
    ) -> Result<DecodeOrder, PrecisionError> {
        if !matches!(
            self.id(),
            FormatId::Fp8Block32 | FormatId::Mxfp8 | FormatId::Mxfp4
        ) || !matches!(family, KernelFamily::Dense | KernelFamily::GroupedExpert)
            || activation != ActivationFormat::Mxfp8
        {
            return Err(error(
                "execution claim",
                "explicit block-scaled MMA requires MX weights and MXFP8 activations",
            ));
        }
        Ok(execution.decode_order())
    }

    /// The encoding of the resident operand this plug-in packs from an
    /// `imported` weight for `family`. A dense 32x32 UE8M0 operand is stored
    /// as its lossless per-row expansion ([`Self::repack_mxfp8_rows`]), the
    /// MXFP8 rows the fold GEMM reads; every other operand keeps its encoding.
    fn resident_format(&self, imported: WeightFormat, family: KernelFamily) -> WeightFormat {
        if self.id() == FormatId::Fp8Block32
            && imported == WeightFormat::Fp8Block32x32
            && family == KernelFamily::Dense
        {
            WeightFormat::Mxfp8
        } else {
            imported
        }
    }

    /// Repeat each 32x32 block scale on its rows, preserving every code and
    /// source K origin. No quantization or scaled-weight rounding takes place.
    fn repack_mxfp8_rows(&self, matrix: MatrixView<'_>) -> Result<PackedMatrix, PrecisionError> {
        if self.id() != FormatId::Fp8Block32 || matrix.format != WeightFormat::Fp8Block32x32 {
            return Err(error(
                "repack",
                "row scale expansion requires E4M3/UE8M0 32x32 blocks",
            ));
        }
        matrix.validate()?;
        let (_, columns) = matrix.scale_shape()?;
        let count = matrix
            .rows
            .checked_mul(columns)
            .filter(|&n| n as u64 <= super::MAX_TENSOR_BYTES)
            .ok_or_else(|| error("repack", "expanded scale grid exceeds the supported bound"))?;
        let source = matrix
            .local
            .ok_or_else(|| error("repack", "missing block scales"))?;
        let mut scales = Vec::with_capacity(count);
        for row in 0..matrix.rows {
            let start = matrix.scale_index(row, 0)?;
            scales.extend_from_slice(&source.bytes[start..start + columns]);
        }
        Ok(PackedMatrix {
            format: WeightFormat::Mxfp8,
            rows: matrix.rows,
            k: matrix.k,
            row_origin: matrix.row_origin,
            k_origin: matrix.k_origin,
            logical_rows: matrix.rows,
            logical_k: matrix.k,
            nibble_order: matrix.nibble_order,
            codes: matrix.codes.to_vec(),
            scale_dtype: DType::Ue8M0,
            scales,
            global: None,
        })
    }
    fn layout(
        &self,
        family: KernelFamily,
        global_scope: Option<ScaleScope>,
    ) -> Result<LayoutChoice, PrecisionError>;

    fn layout_for_weight(
        &self,
        weight: &WeightSpec,
        family: KernelFamily,
        variant: LayoutVariant,
    ) -> Result<LayoutChoice, PrecisionError> {
        let companions = self.companions(weight)?;
        let mut layout = self.layout_variant(
            family,
            weight.global_scale.as_ref().map(|s| s.scope),
            variant,
        )?;
        if companions
            .iter()
            .any(|b| b.role == CompanionRole::LocalScale && b.launch_dtype == DType::Bf16)
        {
            layout.local_scales = Some("bf16_row_major_v1");
        }
        Ok(layout)
    }

    fn companions(&self, weight: &WeightSpec) -> Result<Vec<CompanionBinding>, PrecisionError> {
        weight.validate()?;
        if !self.accepts(weight.format) {
            return Err(error("format", "plug-in disagrees with imported weight"));
        }
        Ok([
            (CompanionRole::LocalScale, weight.local_scale.as_ref()),
            (CompanionRole::GlobalScale, weight.global_scale.as_ref()),
        ]
        .into_iter()
        .filter_map(|(role, scale)| {
            scale.map(|s| CompanionBinding {
                role,
                tensor: s.tensor.tensor.clone(),
                source_dtype: s.tensor.dtype,
                launch_dtype: if s.tensor.dtype == DType::Bf16
                    && !matches!(weight.format, WeightFormat::Fp8PerTensor)
                {
                    DType::F32
                } else {
                    s.tensor.dtype
                },
                scope: s.scope,
                convention: s.convention,
                groups: s.groups.clone(),
            })
        })
        .collect())
    }

    fn layout_variant(
        &self,
        family: KernelFamily,
        global_scope: Option<ScaleScope>,
        variant: LayoutVariant,
    ) -> Result<LayoutChoice, PrecisionError> {
        let mut choice = self.layout(family, global_scope)?;
        match variant {
            LayoutVariant::Standard => {}
            LayoutVariant::AlternateBlockScaleNames => {
                choice.local_scales = Some(match self.id() {
                    FormatId::Fp8Block128 => "fp8_f32_block_scale_128x128_v1",
                    FormatId::Fp8Rows64 => "fp8_f32_row_scale_64x128_v1",
                    _ => return Err(error("layout", "block scale alias requires block FP8")),
                });
            }
            LayoutVariant::Mxfp4Strips { fused_gate_up } => {
                if self.id() != FormatId::Mxfp4 || family != KernelFamily::GroupedExpert {
                    return Err(error("layout", "MXFP4 strips require grouped MXFP4"));
                }
                choice.weight.id = "mxfp4-moe-strips".into();
                choice.codes = if fused_gate_up {
                    "mxfp4_gate_up_moe_fragments_16x64_v1"
                } else {
                    "mxfp4_moe_fragments_16x64_v1"
                };
                choice.local_scales = Some(if fused_gate_up {
                    "mxfp4_gate_up_moe_fragment_scales_16x64_v2"
                } else {
                    "mxfp4_moe_fragment_scales_16x64_v2"
                });
            }
        }
        Ok(choice)
    }

    /// Slice legality is exactly the shared repacker's legality, including
    /// retained source scale origins. Partial groups need not start at zero.
    fn slice(
        &self,
        matrix: MatrixView<'_>,
        request: &SliceRequest,
    ) -> Result<PackedMatrix, PrecisionError> {
        if !self.accepts(matrix.format) {
            return Err(error("format", "plug-in disagrees with imported matrix"));
        }
        let limit = match request.axis {
            SliceAxis::Rows => matrix.rows,
            SliceAxis::K => matrix.k,
        };
        if request.range.is_empty() || request.range.end > limit {
            return Err(error(
                "slice",
                "range is empty or exceeds the source matrix",
            ));
        }
        let (rows, k_start, k) = match request.axis {
            SliceAxis::Rows => (request.range.clone().collect(), 0, matrix.k),
            SliceAxis::K => (
                (0..matrix.rows).collect(),
                request.range.start,
                request.range.len(),
            ),
        };
        let rows: Vec<usize> = rows;
        let pad = |n: usize, tile: usize| {
            if tile == 0 {
                return Err(error("padding", "padding tile must be nonzero"));
            }
            n.checked_add(tile - 1)
                .map(|v| v / tile * tile)
                .ok_or_else(|| error("padding", "padded dimension overflow"))
        };
        matrix.repack(&MatrixSlice {
            padded_rows: pad(rows.len(), request.padding.rows)?,
            padded_k: pad(k, request.padding.k)?,
            rows,
            k_start,
            k,
            nibble_order: request.nibble_order,
        })
    }
}

fn error(field: &'static str, reason: &str) -> PrecisionError {
    PrecisionError::new("format plug-in", field, reason)
}

/// One immutable plug-in instance for each stored format and the W4A16 recipe.
#[derive(Clone, Copy, Debug)]
pub struct Plugin(FormatId);

impl FormatPlugin for Plugin {
    fn id(&self) -> FormatId {
        self.0
    }

    fn accepts(&self, format: WeightFormat) -> bool {
        matches!(
            (self.0, format),
            (FormatId::Bf16, WeightFormat::Bf16)
                | (FormatId::F32, WeightFormat::F32)
                | (FormatId::Fp8PerTensor, WeightFormat::Fp8PerTensor)
                | (FormatId::Fp8PerChannel, WeightFormat::Fp8PerChannel)
                | (FormatId::Fp8Block128, WeightFormat::Fp8Block128x128(_))
                | (FormatId::Fp8Rows64, WeightFormat::Fp8Rows64x128(_))
                | (FormatId::Fp8Block32, WeightFormat::Fp8Block32x32)
                | (FormatId::Mxfp8, WeightFormat::Mxfp8)
                | (FormatId::Mxfp4, WeightFormat::Mxfp4)
                | (
                    FormatId::Nvfp4 | FormatId::Nvfp4WeightOnly,
                    WeightFormat::Nvfp4
                )
        )
    }

    fn decode_order(
        &self,
        family: KernelFamily,
        activation: ActivationFormat,
        recipe: A16Recipe,
    ) -> Result<DecodeOrder, PrecisionError> {
        use DecodeOrder::*;
        use FormatId::*;
        if activation == ActivationFormat::F32 {
            return if family == KernelFamily::Dense && matches!(self.0, Bf16 | F32) {
                Ok(ExactF32Operands)
            } else {
                Err(error(
                    "activation",
                    "F32 inputs require dense unquantized BF16 or F32 weights",
                ))
            };
        }
        if self.0 == F32 {
            return Err(error(
                "activation",
                "raw F32 weights require F32 activations",
            ));
        }
        if self.0 == Nvfp4WeightOnly && activation != ActivationFormat::Bf16 {
            return Err(error(
                "activation",
                "weight-only NVFP4 requires BF16 activations",
            ));
        }
        if family == KernelFamily::Lookup && activation != ActivationFormat::Bf16 {
            return Err(error("activation", "lookup has no activation quantizer"));
        }
        let compatible = match activation {
            ActivationFormat::Bf16 => true,
            ActivationFormat::Fp8Group128(_) => matches!(self.0, Fp8Block128 | Fp8Rows64),
            ActivationFormat::Mxfp8 => matches!(self.0, Fp8Block32 | Mxfp8 | Mxfp4),
            ActivationFormat::Nvfp4 => self.0 == Nvfp4,
            ActivationFormat::F32 => unreachable!("handled above"),
        };
        if !compatible {
            return Err(error(
                "activation",
                "weight and activation formats have no shared recipe",
            ));
        }
        let grouped = family == KernelFamily::GroupedExpert;
        Ok(match activation {
            ActivationFormat::Bf16 => match self.0 {
                Nvfp4 | Nvfp4WeightOnly if family == KernelFamily::Lookup => ExactWeightToBf16,
                Nvfp4 | Nvfp4WeightOnly => ExactLocalThenGlobal,
                Fp8Block32 | Mxfp8 | Mxfp4 => ExactScaledOperand,
                Fp8Block128 | Fp8Rows64 if recipe == A16Recipe::ScaleAfterBlockMma => {
                    BlockScaleAfterMma
                }
                _ if grouped || family == KernelFamily::Lookup => CodeThenScale,
                _ => ScaleThenCode,
            },
            ActivationFormat::Fp8Group128(_) if grouped => SuccessiveScalesAfterBlockMma,
            ActivationFormat::Fp8Group128(_) => ScaleProductAfterBlockMma,
            ActivationFormat::Mxfp8 => NativeBlockScaledMma,
            ActivationFormat::Nvfp4 => NativeBlockScaledMmaThenGlobals,
            ActivationFormat::F32 => unreachable!("handled above"),
        })
    }

    fn layout(
        &self,
        family: KernelFamily,
        global_scope: Option<ScaleScope>,
    ) -> Result<LayoutChoice, PrecisionError> {
        use FormatId::*;
        let grouped = family == KernelFamily::GroupedExpert;
        let fragment = matches!(family, KernelFamily::Dense | KernelFamily::GroupedExpert);
        let (id, codes, local) = match self.0 {
            F32 => ("row-major", "f32_row_major_v1", None),
            Bf16 if fragment => ("bf16-fragment", "bf16_mma_fragments_16x16_v1", None),
            Bf16 => ("row-major", "bf16_row_major_v1", None),
            Fp8Block128 if grouped => (
                "moe-packed-Absent",
                "fp8_e4m3_row_major_v1",
                Some("f32_block_scale_128x128_v1"),
            ),
            Fp8Block128 | Fp8Rows64 if fragment => (
                "fp8-fragment",
                "fp8_e4m3_mma_fragments_16x32_v1",
                Some(if self.0 == Fp8Rows64 {
                    "fp8_f32_row_scale_64x128_v1"
                } else {
                    "f32_block_scale_128x128_v1"
                }),
            ),
            Fp8Block128 | Fp8Rows64 => (
                "row-major",
                "fp8_e4m3_row_major_v1",
                Some(if self.0 == Fp8Rows64 {
                    "fp8_f32_row_scale_64x128_v1"
                } else {
                    "f32_block_scale_128x128_v1"
                }),
            ),
            Fp8PerTensor | Fp8PerChannel => (
                "row-major",
                "fp8_e4m3_row_major_v1",
                Some("f32_row_major_v1"),
            ),
            // A dense operand is resident as MXFP8 rows (`resident_format`).
            Fp8Block32 if family == KernelFamily::Dense => (
                "row-major",
                "mxfp8_e4m3_rows_v1",
                Some("mxfp8_ue8m0_scale_rows_v1"),
            ),
            Fp8Block32 => (
                "row-major",
                "fp8_e4m3_row_major_v1",
                Some("fp8_ue8m0_block_scale_32x32_v1"),
            ),
            Mxfp8 => (
                if grouped {
                    "moe-packed-Absent"
                } else {
                    "row-major"
                },
                "mxfp8_e4m3_rows_v1",
                Some(if grouped {
                    "mxfp8_ue8m0_scale_tiles_16x8_v1"
                } else {
                    "mxfp8_ue8m0_scale_rows_v1"
                }),
            ),
            Mxfp4 => (
                if grouped {
                    "moe-packed-Absent"
                } else {
                    "row-major"
                },
                "mxfp4_e2m1_rows_v1",
                Some(if grouped {
                    "mxfp4_ue8m0_scale_tiles_16x8_v1"
                } else {
                    "mxfp4_ue8m0_scale_rows_v1"
                }),
            ),
            Nvfp4 | Nvfp4WeightOnly => (
                if grouped {
                    if global_scope == Some(ScaleScope::Expert) {
                        "moe-packed-ExpertProjection"
                    } else {
                        "moe-packed-Scalar"
                    }
                } else {
                    "row-major"
                },
                "nvfp4_e2m1_rows_v1",
                Some(if grouped {
                    "nvfp4_e4m3_scale_tiles_16x8_v1"
                } else {
                    "nvfp4_e4m3_scale_rows_v1"
                }),
            ),
        };
        let nv = matches!(self.0, Nvfp4 | Nvfp4WeightOnly);
        if nv != global_scope.is_some() {
            return Err(error("global scale", "global scope disagrees with format"));
        }
        if nv
            && !matches!(
                global_scope,
                Some(
                    ScaleScope::Tensor
                        | ScaleScope::Projection
                        | ScaleScope::Table
                        | ScaleScope::Expert
                )
            )
        {
            return Err(error("global scale", "unsupported global scope"));
        }
        Ok(LayoutChoice {
            weight: WeightLayout {
                id: id.into(),
                version: 1,
            },
            codes,
            local_scales: local,
            global_scales: nv.then_some("f32_row_major_v1"),
        })
    }
}

pub mod registry {
    use super::*;

    pub static PLUGINS: [Plugin; 11] = [
        Plugin(FormatId::Bf16),
        Plugin(FormatId::Fp8PerTensor),
        Plugin(FormatId::Fp8PerChannel),
        Plugin(FormatId::Fp8Block128),
        Plugin(FormatId::Fp8Rows64),
        Plugin(FormatId::Mxfp8),
        Plugin(FormatId::Mxfp4),
        Plugin(FormatId::Nvfp4),
        Plugin(FormatId::Nvfp4WeightOnly),
        Plugin(FormatId::Fp8Block32),
        Plugin(FormatId::F32),
    ];

    fn require_plugin(
        plugin: Option<&'static Plugin>,
        field: &'static str,
        identifier: impl std::fmt::Debug,
    ) -> Result<&'static dyn FormatPlugin, PrecisionError> {
        plugin.map(|p| p as &dyn FormatPlugin).ok_or_else(|| {
            PrecisionError::new(
                "format registry",
                field,
                format!("no plug-in registered for {identifier:?}"),
            )
        })
    }

    pub fn by_id(id: FormatId) -> Result<&'static dyn FormatPlugin, PrecisionError> {
        require_plugin(PLUGINS.iter().find(|p| p.id() == id), "format id", id)
    }

    pub fn by_format(format: WeightFormat) -> Result<&'static dyn FormatPlugin, PrecisionError> {
        require_plugin(
            PLUGINS.iter().find(|p| p.accepts(format)),
            "weight format",
            format,
        )
    }

    /// Stored NVFP4 resolves to the unrestricted plug-in. A W4A16 declaration
    /// selects Nvfp4WeightOnly explicitly, since storage cannot distinguish it.
    pub fn by_weight(weight: &WeightSpec) -> Result<&'static dyn FormatPlugin, PrecisionError> {
        weight.validate()?;
        by_format(weight.format)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn missing_plugins_name_the_refused_identifier() {
            for field in ["format id", "weight format"] {
                let Err(error) = require_plugin(None, field, "unregistered") else {
                    panic!("missing plug-in must be refused");
                };
                assert_eq!(error.tensor, "format registry");
                assert_eq!(error.field, field);
                assert_eq!(error.reason, "no plug-in registered for \"unregistered\"");
            }
        }
    }
}

#[cfg(test)]
mod tests;
