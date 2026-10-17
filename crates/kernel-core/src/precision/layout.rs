//! Persistent packed sections shared by importers and executable view admission.
//! Size validation does not allocate or impose a materializer's working-set limit.
use super::{PrecisionError, WeightLayout};
use crate::contract::{DType, SwigluColumns, WeightFormat};
use crate::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct PackingPlan {
    pub layout: WeightLayout,
    pub format: WeightFormat,
    pub experts: u32,
    pub projections: u32,
    /// Physical rows in each projection, including declared zero padding.
    pub rows: u32,
    pub k: u32,
    pub columns: SwigluColumns,
    pub local_scale_dtype: Option<DType>,
}

/// Persistent sections of one implementation's weight contract. Models place
/// these sections in their existing rank images without duplicating tile-size
/// or scale-padding rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct SectionBytes {
    pub codes: u64,
    pub local_scales: u64,
    pub global_scales: u64,
}

impl SectionBytes {
    pub fn total(self) -> Result<u64, PrecisionError> {
        self.codes
            .checked_add(self.local_scales)
            .and_then(|n| n.checked_add(self.global_scales))
            .ok_or_else(|| error("resident size overflow"))
    }
}

fn error(reason: impl std::fmt::Display) -> PrecisionError {
    PrecisionError::new("packed operand", "layout", reason.to_string())
}

impl PackingPlan {
    pub fn grouped(&self) -> bool {
        self.layout.id.starts_with("moe-packed-")
    }
    pub fn per_expert_global(&self) -> bool {
        self.layout.id == "moe-packed-ExpertProjection"
    }
    pub fn fragment(&self) -> Option<FragmentWeightFormat> {
        match self.layout.id.as_str() {
            "bf16-fragment" => Some(FragmentWeightFormat::Bf16),
            "fp8-fragment" => Some(FragmentWeightFormat::Fp8E4M3),
            _ => None,
        }
    }
    pub fn code_bytes(&self) -> u64 {
        match self.format {
            WeightFormat::Bf16 => u64::from(self.k) * 2,
            WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => u64::from(self.k) / 2,
            _ => u64::from(self.k),
        }
    }
    pub fn groups(&self) -> (u32, u32) {
        match self.format {
            WeightFormat::Bf16 | WeightFormat::Fp8PerTensor => (u32::MAX, u32::MAX),
            WeightFormat::Fp8PerChannel => (1, u32::MAX),
            WeightFormat::Fp8Block128x128(_) => (128, 128),
            WeightFormat::Fp8Rows64x128(_) => (64, 128),
            WeightFormat::Fp8Block32x32 => (32, 32),
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => (1, 32),
            WeightFormat::Nvfp4 => (1, 16),
        }
    }
    pub fn scale_width(&self) -> usize {
        match self.local_scale_dtype {
            Some(DType::F32) => 4,
            Some(DType::Bf16) => 2,
            Some(_) => 1,
            None => 0,
        }
    }
    pub fn total_rows(&self) -> u64 {
        u64::from(self.experts) * u64::from(self.projections) * u64::from(self.rows)
    }
    pub fn scale_shape(&self) -> (usize, usize) {
        if self.format == WeightFormat::Fp8PerTensor {
            return (1, 1);
        }
        let (rn, rk) = self.groups();
        (
            (u64::from(self.experts)
                * (u64::from(self.rows) * u64::from(self.projections)).div_ceil(u64::from(rn)))
                as usize,
            self.k.div_ceil(rk) as usize,
        )
    }
    pub fn tiled_scales(&self) -> bool {
        self.grouped()
            && matches!(
                self.format,
                WeightFormat::Mxfp4 | WeightFormat::Mxfp8 | WeightFormat::Nvfp4
            )
    }
    pub fn validate(&self) -> Result<(), PrecisionError> {
        if self.layout.version != 1
            || self.experts == 0
            || self.rows == 0
            || self.k == 0
            || !(1..=2).contains(&self.projections)
            || u64::from(self.rows) * u64::from(self.projections) > u64::from(u32::MAX)
        {
            return Err(error("invalid version, bank shape or fusion count"));
        }
        let layout_ok = match self.layout.id.as_str() {
            "row-major" => self.experts == 1,
            "bf16-fragment" => self.format == WeightFormat::Bf16,
            "fp8-fragment" => matches!(
                self.format,
                WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
            ),
            "moe-packed-Scalar" | "moe-packed-ExpertProjection" => {
                self.format == WeightFormat::Nvfp4
            }
            "moe-packed-Absent" => matches!(
                self.format,
                WeightFormat::Mxfp4 | WeightFormat::Mxfp8 | WeightFormat::Fp8Block128x128(_)
            ),
            _ => false,
        };
        if !layout_ok {
            return Err(error(format!(
                "unsupported {:?} physical layout {}",
                self.format, self.layout.id
            )));
        }
        let dtype_ok = match self.format {
            WeightFormat::Bf16 => self.local_scale_dtype.is_none(),
            WeightFormat::Nvfp4 => {
                matches!(self.local_scale_dtype, Some(DType::Fp8E4M3 | DType::Ue4M3))
            }
            WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => {
                self.local_scale_dtype == Some(DType::Ue8M0)
            }
            WeightFormat::Fp8PerTensor => {
                matches!(self.local_scale_dtype, Some(DType::F32 | DType::Bf16))
            }
            _ => self.local_scale_dtype == Some(DType::F32),
        };
        if !dtype_ok {
            return Err(error("physical scale dtype does not match the encoding"));
        }
        let (rn, rk) = self.groups();
        if matches!(self.format, WeightFormat::Nvfp4 | WeightFormat::Mxfp4)
            && !self.k.is_multiple_of(2)
            || self.grouped() && (!self.rows.is_multiple_of(16) || !self.k.is_multiple_of(rk))
            || self.projections == 2
                && rn != u32::MAX
                && rn != 1
                && (self.columns == SwigluColumns::Interleaved || !self.rows.is_multiple_of(rn))
        {
            return Err(error(
                "fusion, packed row or scale groups require aligned physical geometry",
            ));
        }
        if let Some(format) = self.fragment() {
            FragmentWeightLayout {
                rows: (self.rows * self.projections) as usize,
                k: self.k as usize,
                format,
            }
            .packed_bytes()
            .map_err(error)?;
        }
        self.resident_bytes()?;
        Ok(())
    }
    pub fn resident_bytes(&self) -> Result<u64, PrecisionError> {
        self.section_bytes()?.total()
    }

    pub fn section_bytes(&self) -> Result<SectionBytes, PrecisionError> {
        if self.experts == 0
            || self.rows == 0
            || self.k == 0
            || self.projections == 0
            || u64::from(self.rows) * u64::from(self.projections) > u64::from(u32::MAX)
        {
            return Err(error("invalid packed geometry"));
        }
        let codes = if let Some(format) = self.fragment() {
            (FragmentWeightLayout {
                rows: (self.rows * self.projections) as usize,
                k: self.k as usize,
                format,
            }
            .packed_bytes()
            .map_err(error)? as u64)
                .checked_mul(u64::from(self.experts))
                .ok_or_else(|| error("fragment size overflow"))?
        } else {
            self.total_rows()
                .checked_mul(self.code_bytes())
                .ok_or_else(|| error("packed size overflow"))?
        };
        let (rows, columns) = self.scale_shape();
        let scales = if self.tiled_scales() {
            (rows as u64)
                .div_ceil(16)
                .checked_mul((columns as u64).div_ceil(8))
                .and_then(|n| n.checked_mul(128))
                .ok_or_else(|| error("scale tile size overflow"))?
        } else {
            (rows as u64)
                .checked_mul(columns as u64)
                .and_then(|n| n.checked_mul(self.scale_width() as u64))
                .ok_or_else(|| error("scale size overflow"))?
        };
        let globals = if self.format != WeightFormat::Nvfp4 {
            0
        } else if self.per_expert_global() {
            u64::from(self.experts) * u64::from(self.projections) * 4
        } else {
            4
        };
        let sections = SectionBytes {
            codes,
            local_scales: scales,
            global_scales: globals,
        };
        sections.total()?;
        Ok(sections)
    }
}
