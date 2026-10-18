//! Validated, borrowed checkpoint matrices. Loading and slicing preserve codes
//! and scales; decoding here is for reference arithmetic, not persistent images.

use super::{DecodeOrder, NibbleOrder, PrecisionError};
use crate::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f32, f32_to_bf16};
use crate::contract::{DType, WeightFormat};

#[derive(Clone, Copy, Debug)]
pub struct ScaleBytes<'a> {
    pub dtype: DType,
    pub bytes: &'a [u8],
}

impl ScaleBytes<'_> {
    pub fn element_bytes(self) -> Result<usize, &'static str> {
        match self.dtype {
            DType::F32 => Ok(4),
            DType::Bf16 => Ok(2),
            DType::Fp8E4M3 | DType::Ue4M3 | DType::Ue8M0 => Ok(1),
            _ => Err("unsupported scale dtype"),
        }
    }

    pub fn value(self, index: usize) -> Result<f32, &'static str> {
        let size = self.element_bytes()?;
        let start = index.checked_mul(size).ok_or("scale offset overflow")?;
        let bytes = self
            .bytes
            .get(start..start.checked_add(size).ok_or("scale offset overflow")?)
            .ok_or("scale index out of bounds")?;
        let value = match self.dtype {
            DType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
            DType::Bf16 => bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())),
            DType::Fp8E4M3 | DType::Ue4M3 => {
                if bytes[0] & 0x80 != 0 || bytes[0] == 0x7f {
                    return Err("E4M3 scale must have a clear sign bit and a finite code");
                }
                SmallFloat::Ue4M3.decode_f32(bytes[0])
            }
            DType::Ue8M0 => {
                if bytes[0] == 0xff {
                    return Err("UE8M0 scale 0xff is not finite");
                }
                decode_ue8m0_f32(bytes[0])
            }
            _ => unreachable!(),
        };
        if !value.is_finite() || value.is_sign_negative() {
            return Err("scale must be finite with a clear sign bit");
        }
        Ok(value)
    }

    pub fn validate(self, count: usize, positive: bool) -> Result<(), &'static str> {
        let size = self.element_bytes()?;
        if count.checked_mul(size) != Some(self.bytes.len()) {
            return Err("scale byte count disagrees with its grid");
        }
        for i in 0..count {
            if self.value(i)? == 0.0 && positive {
                return Err("global scale must be positive");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a> {
    pub tensor: &'a str,
    pub format: WeightFormat,
    pub rows: usize,
    pub k: usize,
    /// Source coordinates of this view's first column and row. They define
    /// scale-group boundaries after slicing, independently of rank-local K.
    pub row_origin: usize,
    pub k_origin: usize,
    pub nibble_order: NibbleOrder,
    pub codes: &'a [u8],
    pub local: Option<ScaleBytes<'a>>,
    pub global: Option<ScaleBytes<'a>>,
}

fn span_groups(origin: usize, length: usize, group: usize) -> Option<usize> {
    origin
        .checked_add(length.checked_sub(1)?)?
        .checked_div(group)?
        .checked_sub(origin / group)?
        .checked_add(1)
}

impl MatrixView<'_> {
    fn error(self, field: &'static str, reason: impl Into<String>) -> PrecisionError {
        PrecisionError::new(self.tensor, field, reason)
    }

    pub fn code_bits(self) -> usize {
        match self.format {
            WeightFormat::Bf16 => 16,
            WeightFormat::F32 => 32,
            WeightFormat::Nvfp4 | WeightFormat::Mxfp4 => 4,
            _ => 8,
        }
    }

    pub fn groups(self) -> (usize, usize) {
        match self.format {
            WeightFormat::Bf16 | WeightFormat::F32 | WeightFormat::Fp8PerTensor => {
                (usize::MAX, usize::MAX)
            }
            WeightFormat::Fp8PerChannel => (1, usize::MAX),
            WeightFormat::Fp8Block128x128(_) => (128, 128),
            WeightFormat::Fp8Rows64x128(_) => (64, 128),
            WeightFormat::Fp8Block32x32 => (32, 32),
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => (1, 32),
            WeightFormat::Nvfp4 => (1, 16),
        }
    }

    pub fn scale_shape(self) -> Result<(usize, usize), PrecisionError> {
        let (r, k) = self.groups();
        let rows = span_groups(self.row_origin, self.rows, r);
        let cols = span_groups(self.k_origin, self.k, k);
        rows.zip(cols)
            .ok_or_else(|| self.error("scale grid", "shape or origin overflow"))
    }

    /// Validate once before reading or packing. Scale validation checks raw
    /// bits, including negative zero, rather than only comparing decoded floats.
    pub fn validate(self) -> Result<(), PrecisionError> {
        if self.rows == 0 || self.k == 0 || (self.code_bits() == 4 && !self.k.is_multiple_of(2)) {
            return Err(self.error("shape", "empty matrix or odd packed row width"));
        }
        let expected = self
            .rows
            .checked_mul(self.k)
            .and_then(|v| v.checked_mul(self.code_bits()))
            .map(|v| v / 8);
        if expected != Some(self.codes.len()) {
            return Err(self.error("codes", "byte count disagrees with logical shape"));
        }
        let (sr, sk) = self.scale_shape()?;
        let scales = sr
            .checked_mul(sk)
            .ok_or_else(|| self.error("scale grid", "element count overflow"))?;
        match (self.format, self.local, self.global) {
            (WeightFormat::Bf16 | WeightFormat::F32, None, None) => {}
            (WeightFormat::Bf16, _, _) => {
                return Err(self.error("scales", "BF16 weights have no quantization scales"));
            }
            (WeightFormat::F32, _, _) => {
                return Err(self.error("scales", "F32 weights have no quantization scales"));
            }
            (WeightFormat::Nvfp4, Some(local), Some(global)) => {
                if !matches!(local.dtype, DType::Fp8E4M3 | DType::Ue4M3)
                    || global.dtype != DType::F32
                {
                    return Err(
                        self.error("scales", "NVFP4 requires E4M3 local and F32 global scales")
                    );
                }
                local
                    .validate(scales, false)
                    .map_err(|e| self.error("local scale", e))?;
                global
                    .validate(1, true)
                    .map_err(|e| self.error("global scale", e))?;
            }
            (WeightFormat::Nvfp4, _, _) => {
                return Err(self.error("scales", "NVFP4 requires local and global companions"));
            }
            (format, Some(local), None) => {
                let dtype_ok = match format {
                    WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => {
                        local.dtype == DType::Ue8M0
                    }
                    WeightFormat::Fp8PerTensor => matches!(local.dtype, DType::Bf16 | DType::F32),
                    _ => matches!(local.dtype, DType::F32 | DType::Bf16),
                };
                if !dtype_ok {
                    return Err(
                        self.error("local scale", "scale dtype disagrees with weight format")
                    );
                }
                local
                    .validate(scales, false)
                    .map_err(|e| self.error("local scale", e))?;
            }
            _ => return Err(self.error("scales", "unexpected or missing scale companions")),
        }
        // All E2M1 codes are finite. Check the other payloads without expanding.
        if self.format == WeightFormat::Bf16 {
            if self
                .codes
                .chunks_exact(2)
                .any(|v| !bf16_to_f32(u16::from_le_bytes(v.try_into().unwrap())).is_finite())
            {
                return Err(self.error("codes", "non-finite BF16 weight"));
            }
        } else if self.format == WeightFormat::F32 {
            if self
                .codes
                .chunks_exact(4)
                .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
            {
                return Err(self.error("codes", "non-finite F32 weight"));
            }
        } else if self.code_bits() == 8 && self.codes.iter().any(|c| c & 0x7f == 0x7f) {
            return Err(self.error("codes", "non-finite E4M3 weight"));
        }
        Ok(())
    }

    fn raw(self, row: usize, col: usize) -> Result<f32, PrecisionError> {
        if row >= self.rows || col >= self.k {
            return Err(self.error("index", "weight index out of bounds"));
        }
        let index = row
            .checked_mul(self.k)
            .and_then(|v| v.checked_add(col))
            .ok_or_else(|| self.error("index", "weight offset overflow"))?;
        let missing = || self.error("codes", "weight byte index out of bounds");
        Ok(match self.code_bits() {
            4 => {
                let byte = *self.codes.get(index / 2).ok_or_else(missing)?;
                let low = index.is_multiple_of(2) == (self.nibble_order == NibbleOrder::EvenLow);
                SmallFloat::E2M1.decode_f32(if low { byte & 15 } else { byte >> 4 })
            }
            8 => SmallFloat::E4M3.decode_f32(*self.codes.get(index).ok_or_else(missing)?),
            32 => {
                let start = index.checked_mul(4).ok_or_else(missing)?;
                let bytes = self
                    .codes
                    .get(start..start.checked_add(4).ok_or_else(missing)?)
                    .ok_or_else(missing)?;
                f32::from_le_bytes(bytes.try_into().unwrap())
            }
            _ => {
                let start = index.checked_mul(2).ok_or_else(missing)?;
                let bytes = self
                    .codes
                    .get(start..start.checked_add(2).ok_or_else(missing)?)
                    .ok_or_else(missing)?;
                bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()))
            }
        })
    }

    pub fn scale_index(self, row: usize, col: usize) -> Result<usize, PrecisionError> {
        if row >= self.rows || col >= self.k {
            return Err(self.error("index", "scale coordinates out of bounds"));
        }
        let (rg, kg) = self.groups();
        let (_, columns) = self.scale_shape()?;
        let r = self
            .row_origin
            .checked_add(row)
            .map(|v| v / rg - self.row_origin / rg);
        let k = self
            .k_origin
            .checked_add(col)
            .map(|v| v / kg - self.k_origin / kg);
        r.zip(k)
            .and_then(|(r, k)| r.checked_mul(columns)?.checked_add(k))
            .ok_or_else(|| self.error("index", "scale offset overflow"))
    }

    /// BF16 decoding for paths that scale weights before MMA. Engines that
    /// apply block scales after accumulation retain their existing oracles.
    /// Explicit F32 steps preserve the selected scale application order.
    pub fn bf16_operand(
        self,
        row: usize,
        col: usize,
        order: DecodeOrder,
    ) -> Result<u16, PrecisionError> {
        if self.format == WeightFormat::F32 {
            return Err(self.error("operand", "raw F32 weights cannot be narrowed to BF16"));
        }
        let code = self.raw(row, col)?;
        let local = match self.local {
            Some(scale) => scale
                .value(self.scale_index(row, col)?)
                .map_err(|e| self.error("local scale", e))?,
            None if self.format == WeightFormat::Bf16 => 1.0,
            None => return Err(self.error("local scale", "missing companion")),
        };
        let global = match self.global {
            Some(scale) => scale.value(0).map_err(|e| self.error("global scale", e))?,
            None if self.format != WeightFormat::Nvfp4 => 1.0,
            None => return Err(self.error("global scale", "missing companion")),
        };
        let value = match order {
            DecodeOrder::ExactWeightToBf16 if self.format == WeightFormat::Nvfp4 => {
                let operand = crate::codec::f64_to_bf16(
                    f64::from(code) * f64::from(local) * f64::from(global),
                );
                if !bf16_to_f32(operand).is_finite() {
                    return Err(self.error("operand", "lookup output exceeds BF16 range"));
                }
                return Ok(operand);
            }
            DecodeOrder::ScaleThenCode => code * (local * global),
            DecodeOrder::CodeThenScale => (code * local) * global,
            DecodeOrder::ExactLocalThenGlobal if self.format == WeightFormat::Nvfp4 => code * local,
            DecodeOrder::ExactScaledOperand
                if matches!(
                    self.format,
                    WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp8 | WeightFormat::Mxfp4
                ) =>
            {
                code * local
            }
            _ => {
                return Err(self.error(
                    "decode order",
                    "this arithmetic has no scaled BF16 weight operand before MMA",
                ));
            }
        };
        let operand = f32_to_bf16(value);
        if matches!(
            order,
            DecodeOrder::ExactLocalThenGlobal | DecodeOrder::ExactScaledOperand
        ) && f64::from(bf16_to_f32(operand)) != f64::from(code) * f64::from(local)
        {
            return Err(self.error(
                "operand",
                format!("weight [{row},{col}] is not exactly representable in BF16"),
            ));
        }
        if !value.is_finite() || !bf16_to_f32(operand).is_finite() {
            return Err(self.error(
                "operand",
                format!("decoded weight [{row},{col}] exceeds BF16 range"),
            ));
        }
        Ok(operand)
    }

    /// Oracle weight in output units. The NVFP4 global product is evaluated
    /// exactly in F64, without the obsolete scaled BF16 intermediate.
    pub fn oracle_weight(
        self,
        row: usize,
        col: usize,
        order: DecodeOrder,
    ) -> Result<f64, PrecisionError> {
        if order == DecodeOrder::ExactF32Operands {
            return self.f32_operand(row, col).map(f64::from);
        }
        let operand = f64::from(bf16_to_f32(self.bf16_operand(row, col, order)?));
        if order == DecodeOrder::ExactLocalThenGlobal {
            let global = self
                .global
                .ok_or_else(|| self.error("global scale", "missing companion"))?
                .value(0)
                .map_err(|e| self.error("global scale", e))?;
            Ok(operand * f64::from(global))
        } else {
            Ok(operand)
        }
    }

    /// Exact widening of an unquantized BF16 weight or unchanged raw F32.
    pub fn f32_operand(self, row: usize, col: usize) -> Result<f32, PrecisionError> {
        if !matches!(self.format, WeightFormat::Bf16 | WeightFormat::F32)
            || self.local.is_some()
            || self.global.is_some()
        {
            return Err(self.error(
                "operand",
                "exact F32 operands require unquantized BF16 or F32 weights",
            ));
        }
        let value = self.raw(row, col)?;
        if !value.is_finite() {
            return Err(self.error("operand", "F32 weight operand must be finite"));
        }
        Ok(value)
    }

    /// Check the operand conversion bound for engines that scale each weight
    /// before BF16 MMA. Ordinary scales need only a scale-grid scan; extreme
    /// scales check actual codes so a valid zero group is not rejected by a
    /// worst-case nonzero code. No expanded weight buffer is allocated.
    pub fn validate_bf16_operands(self, order: DecodeOrder) -> Result<(), PrecisionError> {
        if self.format == WeightFormat::F32 {
            return Err(self.error("operand", "raw F32 weights cannot be narrowed to BF16"));
        }
        if !order.has_bf16_weight_operand() {
            return Err(self.error(
                "decode order",
                "this arithmetic has no scaled BF16 weight operand before MMA",
            ));
        }
        self.validate()?;
        if (matches!(
            order,
            DecodeOrder::ExactLocalThenGlobal | DecodeOrder::ExactWeightToBf16
        ) && self.format != WeightFormat::Nvfp4)
            || (order == DecodeOrder::ExactScaledOperand
                && !matches!(
                    self.format,
                    WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp8 | WeightFormat::Mxfp4
                ))
        {
            return Err(self.error(
                "decode order",
                "exact operand order disagrees with weight format",
            ));
        }
        let Some(local) = self.local else {
            return Ok(());
        };
        let global = self
            .global
            .map(|g| g.value(0))
            .transpose()
            .map_err(|e| self.error("global scale", e))?
            .unwrap_or(1.0);
        let maximum = if self.code_bits() == 4 {
            6.0_f32
        } else {
            448.0
        };
        let (rows, columns) = self.scale_shape()?;
        let mut risky = Vec::with_capacity(rows * columns);
        for i in 0..rows * columns {
            let scale = local.value(i).map_err(|e| self.error("local scale", e))?;
            let value = match order {
                DecodeOrder::ScaleThenCode => maximum * (scale * global),
                DecodeOrder::CodeThenScale | DecodeOrder::ExactWeightToBf16 => {
                    (maximum * scale) * global
                }
                DecodeOrder::ExactLocalThenGlobal | DecodeOrder::ExactScaledOperand => {
                    maximum * scale
                }
                _ => unreachable!("checked pre-MMA scale order"),
            };
            risky.push(
                !value.is_finite()
                    || !bf16_to_f32(f32_to_bf16(value)).is_finite()
                    || (order == DecodeOrder::ExactScaledOperand
                        && matches!(
                            self.format,
                            WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp8
                        )
                        && scale < 2_f32.powi(-124)),
            );
        }
        if risky.contains(&true) {
            for row in 0..self.rows {
                for col in 0..self.k {
                    if risky[self.scale_index(row, col)?] {
                        self.bf16_operand(row, col, order)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Lossless row selection and column slicing for supported encodings.
    /// Whole source scale groups are retained, including a partial first group.
    /// Physical padding gets zero codes; no calibration or requantization runs.
    pub fn repack(self, slice: &MatrixSlice) -> Result<PackedMatrix, PrecisionError> {
        self.validate()?;
        if slice.rows.is_empty()
            || slice.rows.iter().any(|&r| r >= self.rows)
            || slice.k == 0
            || slice
                .k_start
                .checked_add(slice.k)
                .is_none_or(|k| k > self.k)
            || slice.padded_rows < slice.rows.len()
            || slice.padded_k < slice.k
            || (self.code_bits() == 4 && !slice.padded_k.is_multiple_of(2))
        {
            return Err(self.error("repacking", "slice or padding exceeds matrix bounds"));
        }
        let k_origin = self
            .k_origin
            .checked_add(slice.k_start)
            .ok_or_else(|| self.error("repacking", "origin overflow"))?;
        let row_origin = if self.groups().0 == usize::MAX {
            0
        } else {
            self.row_origin
                .checked_add(slice.rows[0])
                .ok_or_else(|| self.error("repacking", "row origin overflow"))?
        };
        let columns = span_groups(k_origin, slice.padded_k, self.groups().1)
            .ok_or_else(|| self.error("repacking", "scale grid overflow"))?;
        let scale_rows = span_groups(row_origin, slice.padded_rows, self.groups().0)
            .ok_or_else(|| self.error("repacking", "scale row overflow"))?;
        let bytes = slice
            .padded_rows
            .checked_mul(slice.padded_k)
            .and_then(|v| v.checked_mul(self.code_bits()))
            .map(|v| v / 8)
            .filter(|&v| v as u64 <= super::MAX_TENSOR_BYTES)
            .ok_or_else(|| self.error("repacking", "packed size overflow"))?;
        let scale_count = scale_rows
            .checked_mul(columns)
            .filter(|&v| v as u64 <= super::MAX_TENSOR_BYTES)
            .ok_or_else(|| self.error("repacking", "scale size overflow"))?;
        let scale_width = self
            .local
            .map(|s| s.element_bytes())
            .transpose()
            .map_err(|e| self.error("repacking", e))?
            .unwrap_or(0);
        let scale_bytes = scale_count
            .checked_mul(scale_width)
            .filter(|&n| n as u64 <= super::MAX_TENSOR_BYTES)
            .ok_or_else(|| self.error("repacking", "scale byte size overflow"))?;
        let mut codes = vec![0; bytes];
        let mut scales = vec![
            if matches!(
                self.format,
                WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp8 | WeightFormat::Mxfp4
            ) {
                127
            } else {
                0
            };
            scale_bytes
        ];
        let mut scale_sources = vec![None; if self.local.is_some() { scale_rows } else { 0 }];
        for (out_row, &row) in slice.rows.iter().enumerate() {
            if self.code_bits() >= 8 || slice.k_start.is_multiple_of(2) && slice.k.is_multiple_of(2)
            {
                let src = (row * self.k + slice.k_start) * self.code_bits() / 8;
                let dst = out_row * slice.padded_k * self.code_bits() / 8;
                let length = slice.k * self.code_bits() / 8;
                codes[dst..dst + length].copy_from_slice(&self.codes[src..src + length]);
                if self.code_bits() == 4 && self.nibble_order != slice.nibble_order {
                    for byte in &mut codes[dst..dst + length] {
                        *byte = byte.rotate_left(4);
                    }
                }
            } else {
                for col in 0..slice.k {
                    let src = row * self.k + slice.k_start + col;
                    let dst = out_row * slice.padded_k + col;
                    if self.code_bits() == 4 {
                        let low =
                            src.is_multiple_of(2) == (self.nibble_order == NibbleOrder::EvenLow);
                        let code = if low {
                            self.codes[src / 2] & 15
                        } else {
                            self.codes[src / 2] >> 4
                        };
                        let shift = if dst.is_multiple_of(2)
                            == (slice.nibble_order == NibbleOrder::EvenLow)
                        {
                            0
                        } else {
                            4
                        };
                        codes[dst / 2] |= code << shift;
                    }
                }
            }
            if let Some(local) = self.local {
                let target_row =
                    (row_origin + out_row) / self.groups().0 - row_origin / self.groups().0;
                let start = self.scale_index(row, slice.k_start)?;
                if scale_sources[target_row].is_some_and(|prior| prior != start) {
                    return Err(self.error("repacking", "row selection combines different source scale groups; retain whole groups or use a supported remap"));
                }
                scale_sources[target_row] = Some(start);
                let length = span_groups(k_origin, slice.k, self.groups().1).unwrap();
                let dst = target_row * columns * scale_width;
                scales[dst..dst + length * scale_width].copy_from_slice(
                    &local.bytes[start * scale_width..(start + length) * scale_width],
                );
            }
        }
        Ok(PackedMatrix {
            format: self.format,
            rows: slice.padded_rows,
            k: slice.padded_k,
            k_origin,
            row_origin,
            logical_rows: slice.rows.len(),
            logical_k: slice.k,
            nibble_order: slice.nibble_order,
            codes,
            scale_dtype: self.local.map_or(DType::F32, |s| s.dtype),
            scales,
            global: self.global.map(|g| g.bytes.to_vec()),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixSlice {
    pub rows: Vec<usize>,
    pub k_start: usize,
    pub k: usize,
    pub padded_rows: usize,
    pub padded_k: usize,
    pub nibble_order: NibbleOrder,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedMatrix {
    pub format: WeightFormat,
    pub rows: usize,
    pub k: usize,
    pub k_origin: usize,
    pub row_origin: usize,
    pub logical_rows: usize,
    pub logical_k: usize,
    pub nibble_order: NibbleOrder,
    pub codes: Vec<u8>,
    pub scale_dtype: DType,
    pub scales: Vec<u8>,
    pub global: Option<Vec<u8>>,
}

impl PackedMatrix {
    pub fn view<'a>(&'a self, tensor: &'a str) -> MatrixView<'a> {
        MatrixView {
            tensor,
            format: self.format,
            rows: self.rows,
            k: self.k,
            row_origin: self.row_origin,
            k_origin: self.k_origin,
            nibble_order: self.nibble_order,
            codes: &self.codes,
            local: (!matches!(self.format, WeightFormat::Bf16 | WeightFormat::F32)).then_some(
                ScaleBytes {
                    dtype: self.scale_dtype,
                    bytes: &self.scales,
                },
            ),
            global: self.global.as_ref().map(|bytes| ScaleBytes {
                dtype: DType::F32,
                bytes,
            }),
        }
    }
}
