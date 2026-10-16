//! Exact physical packing selected by prepared implementations. Models choose
//! matrix slices, rank ownership and fusion order; this module never calibrates.

use crate::CompilerError;
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::NibbleOrder;
use kernel_core::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
use kernel_core::transform::fragment::FragmentWeightLayout;
use kernel_core::transform::{ScaleTileLayout, Transform};

const MAX_PACKED_OPERAND_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub use kernel_core::precision::layout::{PackingPlan, SectionBytes};

#[derive(Clone, Debug)]
pub struct PackedWeights {
    pub plan: PackingPlan,
    pub codes: Vec<u8>,
    pub local_scales: Vec<u8>,
    pub global_scales: Vec<u8>,
}

fn error(reason: impl std::fmt::Display) -> CompilerError {
    CompilerError::validation(format!("precision packing: {reason}"))
}

/// Feed one source matrix at a time, retaining only packed codes and scales.
/// `finish` applies the shared fragment/scale transforms without BF16 expansion.
pub struct PackedWeightBuilder {
    plan: PackingPlan,
    codes: Vec<u8>,
    scales: Vec<u8>,
    globals: Vec<u8>,
    written: Vec<bool>,
    scalar_local: Option<Vec<u8>>,
    scalar_global: Option<Vec<u8>>,
}

impl PackedWeightBuilder {
    pub fn new(plan: PackingPlan) -> Result<Self, CompilerError> {
        plan.validate()?;
        if plan.resident_bytes()? > MAX_PACKED_OPERAND_BYTES {
            return Err(error(
                "packed operand exceeds the 2 GiB materialization bound",
            ));
        }
        let (rows, columns) = plan.scale_shape();
        let result = Self {
            codes: vec![0; (plan.total_rows() * plan.code_bytes()) as usize],
            scales: vec![
                if matches!(plan.format, WeightFormat::Mxfp4 | WeightFormat::Mxfp8) {
                    127
                } else {
                    0
                };
                rows * columns * plan.scale_width()
            ],
            globals: vec![
                0;
                if plan.format == WeightFormat::Nvfp4 {
                    if plan.per_expert_global() {
                        (plan.experts * plan.projections * 4) as usize
                    } else {
                        4
                    }
                } else {
                    0
                }
            ],
            written: vec![false; (u64::from(plan.experts) * u64::from(plan.projections)) as usize],
            scalar_local: None,
            scalar_global: None,
            plan,
        };
        Ok(result)
    }
    pub fn write_matrix(
        &mut self,
        expert: u32,
        projection: u32,
        source: MatrixView<'_>,
        slice: &MatrixSlice,
    ) -> Result<(), CompilerError> {
        let context = |reason| {
            error(format!(
                "tensor {}, expert {expert}, projection {projection}: {reason}",
                source.tensor
            ))
        };
        if expert >= self.plan.experts || projection >= self.plan.projections {
            return Err(context("bank coordinate is out of bounds"));
        }
        let index = (expert * self.plan.projections + projection) as usize;
        if self.written[index] {
            return Err(context("matrix was already supplied"));
        }
        if source.format != self.plan.format
            || slice.padded_rows != self.plan.rows as usize
            || slice.padded_k != self.plan.k as usize
            || slice.nibble_order != NibbleOrder::EvenLow
        {
            return Err(context(
                "encoding, slice or padding differs from the prepared physical contract",
            ));
        }
        let packed = source.repack(slice).map_err(error)?;
        let (rn, rk) = self.plan.groups();
        if (rn != u32::MAX && packed.row_origin % rn as usize != 0)
            || (rk != u32::MAX && packed.k_origin % rk as usize != 0)
        {
            return Err(context(
                "implementation requires an aligned source scale-group origin",
            ));
        }
        let row_bytes = self.plan.code_bytes() as usize;
        let base_row = (u64::from(expert)
            * u64::from(self.plan.projections)
            * u64::from(self.plan.rows)) as usize;
        let target_row = |r: usize| {
            base_row
                + match self.plan.columns {
                    SwigluColumns::ConcatenatedHalves => {
                        projection as usize * self.plan.rows as usize + r
                    }
                    SwigluColumns::Interleaved => {
                        r * self.plan.projections as usize + projection as usize
                    }
                }
        };
        let mut scales = packed.scales;
        if let Some(dtype) = self.plan.local_scale_dtype {
            if packed.scale_dtype == DType::Bf16 && dtype == DType::F32 {
                scales = scales
                    .chunks_exact(2)
                    .flat_map(|s| [0, 0, s[0], s[1]])
                    .collect();
            } else if packed.scale_dtype != dtype
                && !(self.plan.format == WeightFormat::Nvfp4
                    && matches!(
                        (packed.scale_dtype, dtype),
                        (DType::Ue4M3, DType::Fp8E4M3) | (DType::Fp8E4M3, DType::Ue4M3)
                    ))
            {
                return Err(context("scale conversion would change stored values"));
            }
        }
        if self.plan.format == WeightFormat::Fp8PerTensor {
            if self.scalar_local.as_ref().is_some_and(|old| old != &scales) {
                return Err(context("fused per-tensor scales are not bitwise equal"));
            }
            self.scalar_local = Some(scales.clone());
        }
        if let Some(global) = &packed.global {
            if !self.plan.per_expert_global()
                && self.scalar_global.as_ref().is_some_and(|old| old != global)
            {
                return Err(context(
                    "scalar-global implementation requires bitwise equal weight globals across every expert and projection",
                ));
            }
            self.scalar_global = Some(global.clone());
            let start = if self.plan.per_expert_global() {
                index * 4
            } else {
                0
            };
            self.globals[start..start + 4].copy_from_slice(global);
        }
        for row in 0..self.plan.rows as usize {
            let target = target_row(row) * row_bytes;
            self.codes[target..target + row_bytes]
                .copy_from_slice(&packed.codes[row * row_bytes..(row + 1) * row_bytes]);
        }
        if self.plan.format == WeightFormat::Fp8PerTensor {
            self.scales.copy_from_slice(&scales);
        } else if self.plan.local_scale_dtype.is_some() {
            let width = self.plan.k.div_ceil(rk) as usize * self.plan.scale_width();
            let rows = self.plan.rows.div_ceil(rn) as usize;
            let scale_rows_per_expert =
                (self.plan.rows * self.plan.projections).div_ceil(rn) as usize;
            for row in 0..rows {
                let target = if rn == 1 {
                    target_row(row)
                } else {
                    expert as usize * scale_rows_per_expert + projection as usize * rows + row
                };
                self.scales[target * width..(target + 1) * width]
                    .copy_from_slice(&scales[row * width..(row + 1) * width]);
            }
        }
        self.written[index] = true;
        Ok(())
    }
    pub fn finish(self) -> Result<PackedWeights, CompilerError> {
        if self.written.contains(&false) {
            return Err(error("not every expert/projection matrix was supplied"));
        }
        let codes = if let Some(format) = self.plan.fragment() {
            let layout = FragmentWeightLayout {
                rows: (self.plan.rows * self.plan.projections) as usize,
                k: self.plan.k as usize,
                format,
            };
            let bytes = layout.rows * layout.k * layout.element_bytes();
            let mut out = Vec::new();
            for expert in self.codes.chunks_exact(bytes) {
                out.extend(layout.pack(expert).map_err(error)?.bytes);
            }
            out
        } else {
            self.codes
        };
        let local_scales = if self.plan.tiled_scales() {
            let (rows, k_blocks) = self.plan.scale_shape();
            ScaleTileLayout {
                rows,
                k_blocks,
                tile_rows: 16,
                k_blocks_per_tile: 8,
            }
            .pack(&self.scales)
            .map_err(error)?
            .bytes
        } else {
            self.scales
        };
        if codes.len() as u64 + local_scales.len() as u64 + self.globals.len() as u64
            != self.plan.resident_bytes()?
        {
            return Err(error("packed footprint differs from the prepared contract"));
        }
        Ok(PackedWeights {
            plan: self.plan,
            codes,
            local_scales,
            global_scales: self.globals,
        })
    }
}

/// Coalescing never takes a maximum or synthesizes calibration. Source callers
/// retain every contributing tensor's provenance even when the bytes coalesce.
pub fn uniform_f32_global<'a>(
    values: impl IntoIterator<Item = ScaleBytes<'a>>,
) -> Result<[u8; 4], CompilerError> {
    let mut selected = None;
    for scale in values {
        if scale.dtype != DType::F32 {
            return Err(error("bank-uniform global requires F32 source scalars"));
        }
        scale.validate(1, true).map_err(error)?;
        let bytes: [u8; 4] = scale.bytes.try_into().expect("validated scalar");
        if selected.is_some_and(|old| old != bytes) {
            return Err(error("source globals are not bitwise bank-uniform"));
        }
        selected = Some(bytes);
    }
    selected.ok_or_else(|| error("no source globals to coalesce"))
}
