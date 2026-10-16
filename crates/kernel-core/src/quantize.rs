//! Bit-exact CPU quantizers. Input layouts are row-major and K-contiguous.
//! All scale grids use row-major order. Zero groups use scale 1 and zero codes,
//! except E4M3 local scales, which store a zero scale.

use crate::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f32, decode_ue8m0_f64};
use crate::contract::{Refusal, ScaleMode};

/// Storage of block scales.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleFormat {
    F32,
    /// BF16 values, stored widened (exactly) in `Scales::F32`.
    Bf16,
    /// `2^(code - 127)`.
    Ue8M0,
    /// Unsigned E4M3 code times the tensor's F32 global scale.
    E4M3,
}

/// A block-scaled element format over a row-major [rows, k] matrix: one scale
/// per block of `block = (rows, cols)` values. With `chunk_rows`, the block
/// grid restarts every `chunk_rows` matrix rows, as in a checkpoint that
/// stores one scale grid per tensor-parallel chunk, so a chunk's partial last
/// block never shares a scale with the next chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantFormat {
    pub element: SmallFloat,
    pub block: (usize, usize),
    pub scale: ScaleFormat,
    pub chunk_rows: Option<usize>,
}

impl QuantFormat {
    pub fn validate(self) -> Result<(), Refusal> {
        if self.element == SmallFloat::Ue4M3 {
            return Err(Refusal("UE4M3 is a scale format, not an element format"));
        }
        if self.block.0 == 0 || self.block.1 == 0 || self.chunk_rows == Some(0) {
            return Err(Refusal("quantization blocks and chunks must be nonempty"));
        }
        Ok(())
    }

    /// E2M1 codes pack two per byte; even flat indices use the low nibble.
    fn packed(self) -> bool {
        self.element == SmallFloat::E2M1
    }

    pub fn code_bytes(self, elements: usize) -> usize {
        if self.packed() {
            elements.div_ceil(2)
        } else {
            elements
        }
    }

    pub fn scale_rows(self, rows: usize) -> usize {
        match self.chunk_rows {
            None => rows.div_ceil(self.block.0),
            Some(chunk) => rows.div_ceil(chunk) * chunk.div_ceil(self.block.0),
        }
    }

    pub fn scale_cols(self, k: usize) -> usize {
        k.div_ceil(self.block.1)
    }

    /// Scale-grid row of a matrix row, counted from its chunk's origin.
    pub fn scale_row(self, row: usize) -> usize {
        match self.chunk_rows {
            None => row / self.block.0,
            Some(chunk) => row / chunk * chunk.div_ceil(self.block.0) + row % chunk / self.block.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scales {
    F32(Vec<f32>),
    Ue8M0(Vec<u8>),
    Ue4M3(Vec<u8>),
}

impl Scales {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Ue8M0(codes) | Self::Ue4M3(codes) => codes.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    pub rows: usize,
    pub k: usize,
    pub format: QuantFormat,
    /// One code per element, or packed E2M1 (even flat index in the low nibble).
    pub codes: Vec<u8>,
    pub scales: Scales,
    /// E4M3 scales only. Dequantization multiplies by local and global scales.
    pub global_scale: f32,
}

impl Quantized {
    /// Checked assembly of stored codes and scales, such as a checkpoint
    /// tensor, for decoding.
    pub fn from_parts(
        rows: usize,
        k: usize,
        format: QuantFormat,
        codes: Vec<u8>,
        scales: Scales,
        global_scale: f32,
    ) -> Result<Self, Refusal> {
        format.validate()?;
        let storage = match (&scales, format.scale) {
            (Scales::F32(_), ScaleFormat::F32) | (Scales::Ue8M0(_), ScaleFormat::Ue8M0) => {
                global_scale == 1.0
            }
            (Scales::F32(values), ScaleFormat::Bf16) => {
                global_scale == 1.0
                    && values
                        .iter()
                        .all(|v| bf16_to_f32((v.to_bits() >> 16) as u16).to_bits() == v.to_bits())
            }
            (Scales::Ue4M3(_), ScaleFormat::E4M3) => global_scale.is_finite() && global_scale > 0.0,
            _ => false,
        };
        if !storage
            || rows == 0
            || k == 0
            || codes.len() != format.code_bytes(rows * k)
            || scales.len() != format.scale_rows(rows) * format.scale_cols(k)
        {
            return Err(Refusal("quantized codes or scales do not match the format"));
        }
        Ok(Self {
            rows,
            k,
            format,
            codes,
            scales,
            global_scale,
        })
    }

    pub fn scale_cols(&self) -> usize {
        self.format.scale_cols(self.k)
    }
    pub fn scale_rows(&self) -> usize {
        self.format.scale_rows(self.rows)
    }

    /// The exact value: element times block scale (times the global scale,
    /// for E4M3 scales) in F64.
    pub fn decode(&self, row: usize, col: usize) -> f64 {
        self.decode_with(row, col, ScaleProduct::F64)
    }

    pub fn decode_with(&self, row: usize, col: usize, product: ScaleProduct) -> f64 {
        let index = row * self.k + col;
        let code = if self.format.packed() {
            let byte = self.codes[index / 2];
            if index.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            }
        } else {
            self.codes[index]
        };
        let scale_index =
            self.format.scale_row(row) * self.scale_cols() + col / self.format.block.1;
        if product == ScaleProduct::F32 {
            let local = match &self.scales {
                Scales::F32(values) => values[scale_index],
                Scales::Ue8M0(values) => decode_ue8m0_f32(values[scale_index]),
                Scales::Ue4M3(values) => SmallFloat::Ue4M3.decode_f32(values[scale_index]),
            };
            let value = f64::from(self.format.element.decode_f32(code) * local);
            return match self.scales {
                Scales::Ue4M3(_) => value * f64::from(self.global_scale),
                _ => value,
            };
        }
        let scale = match &self.scales {
            Scales::F32(values) => f64::from(values[scale_index]),
            Scales::Ue8M0(values) => decode_ue8m0_f64(values[scale_index]),
            Scales::Ue4M3(values) => {
                SmallFloat::Ue4M3.decode_f64(values[scale_index]) * f64::from(self.global_scale)
            }
        };
        self.format.element.decode_f64(code) * scale
    }

    pub fn decoded(&self) -> Vec<f64> {
        self.decoded_with(ScaleProduct::F64)
    }

    pub fn decoded_with(&self, product: ScaleProduct) -> Vec<f64> {
        (0..self.rows)
            .flat_map(|r| (0..self.k).map(move |c| self.decode_with(r, c, product)))
            .collect()
    }
}

/// Where a decode forms the element times its block scale. `F64` is exact
/// (an E4M3 local scale times the global scale first). `F32` rounds that
/// product to F32, so it overflows to infinity or rounds below the normal
/// range as an F32 kernel product does; a global scale then multiplies in
/// F64.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleProduct {
    F64,
    F32,
}

/// How a block's amax becomes the F32 ratio `amax / max`, `max` being the
/// element's largest finite value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AmaxRatio {
    Divide,
    /// `amax * (1 / max)`, with the reciprocal rounded to F32 first.
    MultiplyReciprocal,
}

/// How a block's amax becomes its scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleRule {
    /// The ratio itself.
    Ratio(AmaxRatio),
    /// `2^ceil(log2(ratio))` from F32 `log2` and `powf`. Just above a power
    /// of two, the F32 logarithm can round to an integer and give that power.
    PowfCeilLog2(AmaxRatio),
    /// The smallest power of two at or above the ratio, from its exponent
    /// bits. A ratio below 2^-126 (subnormal, or zero when a nonzero amax
    /// underflows in F32) takes the minimum normal scale 2^-126.
    CeilPowerOfTwo(AmaxRatio),
    /// OCP MX v1.0: `2^(floor(log2(amax)) - floor(log2(max)))` with the F32
    /// logarithm. This deliberately may saturate large elements in a block.
    OcpFloor,
    /// NVIDIA NVFP4: local scale `RNE_E4M3((amax / max) / global)`, applied
    /// as `local * global`. Elements divide by that product, all in F32.
    LocalOverGlobal { global: f32 },
}

/// A quantization emulation: the format, its scale rule and an optional
/// floor raised under every block's amax before the rule. Except under
/// `LocalOverGlobal`, elements are `RNE(f32(x / scale))` with the division in
/// f64 by the stored scale. Implemented storage per rule: F32 for `Ratio`,
/// `PowfCeilLog2` and `CeilPowerOfTwo`; UE8M0 (clamped to 2^-127..2^127) for
/// `OcpFloor` and `CeilPowerOfTwo`; E4M3 for `LocalOverGlobal`. Any other
/// pair, including BF16 storage, is refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantizer {
    pub format: QuantFormat,
    pub rule: ScaleRule,
    pub amax_floor: Option<f32>,
}

impl Quantizer {
    /// E4M3 with F32 scales over `block`, as a `ScaleMode` selects.
    pub fn fp8_blocks(block: (usize, usize), mode: ScaleMode) -> Self {
        Self {
            format: QuantFormat {
                element: SmallFloat::E4M3,
                block,
                scale: ScaleFormat::F32,
                chunk_rows: None,
            },
            rule: match mode {
                ScaleMode::ExactF32 => ScaleRule::Ratio(AmaxRatio::Divide),
                ScaleMode::PowerOfTwoF32 => ScaleRule::PowfCeilLog2(AmaxRatio::Divide),
            },
            amax_floor: None,
        }
    }

    /// OCP MX: 1 x 32 blocks with UE8M0 scales under the floor rule.
    pub fn mx(element: SmallFloat) -> Self {
        Self {
            format: QuantFormat {
                element,
                block: (1, 32),
                scale: ScaleFormat::Ue8M0,
                chunk_rows: None,
            },
            rule: ScaleRule::OcpFloor,
            amax_floor: None,
        }
    }

    /// NVFP4: E2M1 over 1 x 16 blocks with E4M3 local scales.
    pub fn nvfp4(global: f32) -> Self {
        Self {
            format: QuantFormat {
                element: SmallFloat::E2M1,
                block: (1, 16),
                scale: ScaleFormat::E4M3,
                chunk_rows: None,
            },
            rule: ScaleRule::LocalOverGlobal { global },
            amax_floor: None,
        }
    }
}

fn check_input(
    values: &[f32],
    rows: usize,
    k: usize,
    block_rows: usize,
    block_k: usize,
) -> Result<(), Refusal> {
    if rows == 0
        || k == 0
        || block_rows == 0
        || block_k == 0
        || values.len() != rows.checked_mul(k).ok_or(Refusal("matrix size overflow"))?
    {
        return Err(Refusal("invalid quantizer matrix shape"));
    }
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Refusal("quantizer input must be finite"));
    }
    Ok(())
}

fn amax_ratio(amax: f32, max: f32, ratio: AmaxRatio) -> f32 {
    match ratio {
        AmaxRatio::Divide => amax / max,
        AmaxRatio::MultiplyReciprocal => amax * (1.0 / max),
    }
}

/// `ceil(log2(value))` of a nonnegative finite F32 from its exponent bits,
/// never below -126 (the minimum normal exponent).
fn ceil_log2(value: f32) -> i32 {
    if value < f32::MIN_POSITIVE {
        return -126;
    }
    let bits = value.to_bits();
    ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0)
}

/// Quantize `values` [rows, k]. Partial edge blocks are allowed.
pub fn quantize(
    values: &[f32],
    rows: usize,
    k: usize,
    quantizer: Quantizer,
) -> Result<Quantized, Refusal> {
    let Quantizer {
        format,
        rule,
        amax_floor,
    } = quantizer;
    format.validate()?;
    check_input(values, rows, k, format.block.0, format.block.1)?;
    let global = match (format.scale, rule) {
        (
            ScaleFormat::F32,
            ScaleRule::Ratio(_) | ScaleRule::PowfCeilLog2(_) | ScaleRule::CeilPowerOfTwo(_),
        )
        | (ScaleFormat::Ue8M0, ScaleRule::OcpFloor | ScaleRule::CeilPowerOfTwo(_)) => 1.0,
        (ScaleFormat::E4M3, ScaleRule::LocalOverGlobal { global }) => {
            if !global.is_finite() || global <= 0.0 {
                return Err(Refusal("E4M3 global scale outside F32 range"));
            }
            global
        }
        _ => {
            return Err(Refusal(
                "quantizer scale rule is not implemented for this scale storage",
            ));
        }
    };
    if amax_floor.is_some_and(|floor| !floor.is_finite() || floor <= 0.0) {
        return Err(Refusal("quantizer amax floor must be positive and finite"));
    }
    let max = format.element.max_finite() as f32;
    let scale_cols = format.scale_cols(k);
    let mut maxima = vec![0.0_f32; format.scale_rows(rows) * scale_cols];
    for row in 0..rows {
        let base = format.scale_row(row) * scale_cols;
        for col in 0..k {
            let cell = &mut maxima[base + col / format.block.1];
            *cell = cell.max(values[row * k + col].abs());
        }
    }
    if let Some(floor) = amax_floor {
        for amax in &mut maxima {
            *amax = amax.max(floor);
        }
    }
    let scales = match format.scale {
        ScaleFormat::E4M3 => Scales::Ue4M3(
            maxima
                .iter()
                .map(|&amax| {
                    if amax == 0.0 {
                        0
                    } else {
                        let ratio = amax_ratio(amax, max, AmaxRatio::Divide);
                        SmallFloat::Ue4M3.encode_f32(ratio / global)
                    }
                })
                .collect(),
        ),
        ScaleFormat::Ue8M0 => Scales::Ue8M0(
            maxima
                .iter()
                .map(|&amax| {
                    let exponent = match rule {
                        _ if amax == 0.0 => 0,
                        ScaleRule::CeilPowerOfTwo(ratio) => ceil_log2(amax_ratio(amax, max, ratio)),
                        ScaleRule::OcpFloor => {
                            amax.log2().floor() as i32 - max.log2().floor() as i32
                        }
                        _ => unreachable!("scale rule and storage were checked above"),
                    };
                    (exponent + 127).clamp(0, 254) as u8
                })
                .collect(),
        ),
        ScaleFormat::F32 | ScaleFormat::Bf16 => Scales::F32(
            maxima
                .iter()
                .map(|&amax| match rule {
                    _ if amax == 0.0 => 1.0,
                    ScaleRule::Ratio(ratio) => amax_ratio(amax, max, ratio),
                    ScaleRule::PowfCeilLog2(ratio) => {
                        2_f32.powf(amax_ratio(amax, max, ratio).log2().ceil())
                    }
                    ScaleRule::CeilPowerOfTwo(ratio) => {
                        let exponent = ceil_log2(amax_ratio(amax, max, ratio));
                        f32::from_bits(((exponent + 127) as u32) << 23)
                    }
                    _ => unreachable!("scale rule and storage were checked above"),
                })
                .collect(),
        ),
    };
    let mut codes = vec![0_u8; format.code_bytes(values.len())];
    for row in 0..rows {
        let base = format.scale_row(row) * scale_cols;
        for col in 0..k {
            let index = row * k + col;
            let cell = base + col / format.block.1;
            let value = values[index];
            let code = match &scales {
                Scales::F32(scales) => format
                    .element
                    .encode_f32((f64::from(value) / f64::from(scales[cell])) as f32),
                Scales::Ue8M0(codes) => format
                    .element
                    .encode_f32((f64::from(value) / decode_ue8m0_f64(codes[cell])) as f32),
                Scales::Ue4M3(_) if maxima[cell] == 0.0 => 0,
                Scales::Ue4M3(codes) => {
                    let local = SmallFloat::Ue4M3.decode_f32(codes[cell]);
                    format.element.encode_f32(value / (local * global))
                }
            };
            if !format.packed() {
                codes[index] = code;
            } else if index.is_multiple_of(2) {
                codes[index / 2] = code;
            } else {
                codes[index / 2] |= code << 4;
            }
        }
    }
    Ok(Quantized {
        rows,
        k,
        format,
        codes,
        scales,
        global_scale: global,
    })
}

pub fn mxfp8(values: &[f32], rows: usize, k: usize) -> Result<Quantized, Refusal> {
    check_input(values, rows, k, 1, 32)?;
    if !k.is_multiple_of(32) {
        return Err(Refusal("MX K must be divisible by 32"));
    }
    quantize(values, rows, k, Quantizer::mx(SmallFloat::E4M3))
}

pub fn mxfp4(values: &[f32], rows: usize, k: usize) -> Result<Quantized, Refusal> {
    check_input(values, rows, k, 1, 32)?;
    if !k.is_multiple_of(32) {
        return Err(Refusal("MX K must be divisible by 32"));
    }
    quantize(values, rows, k, Quantizer::mx(SmallFloat::E2M1))
}

/// NVIDIA NVFP4 recipe: global_scale = global_amax/(448*6); each 16-K
/// local scale is RNE_UE4M3((local_amax/6)/global_scale). Elements are
/// RNE_E2M1(x/(decoded_local_scale*global_scale)), saturated to +/-6.
/// Dequantization multiplies by both scales. Zero tensors use global scale 1.
/// Scale arithmetic before each encoding uses F32 operations.
pub fn nvfp4(values: &[f32], rows: usize, k: usize) -> Result<Quantized, Refusal> {
    check_input(values, rows, k, 1, 16)?;
    if !k.is_multiple_of(16) {
        return Err(Refusal("NVFP4 K must be divisible by 16"));
    }
    let global_amax = values.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let global_scale = if global_amax == 0.0 {
        1.0
    } else {
        global_amax / 2688.0
    };
    nvfp4_with_global(values, rows, k, global_scale)
}

/// Reference the calibrated execution recipe over supplied input values. This
/// consumes the exported global scale; it neither recalibrates nor substitutes
/// a tensor maximum. The dynamic recipe above uses the same exact block math.
pub fn nvfp4_with_global(
    values: &[f32],
    rows: usize,
    k: usize,
    global_scale: f32,
) -> Result<Quantized, Refusal> {
    check_input(values, rows, k, 1, 16)?;
    if !k.is_multiple_of(16) {
        return Err(Refusal("NVFP4 K must be divisible by 16"));
    }
    if !global_scale.is_finite() || global_scale <= 0.0 {
        return Err(Refusal("NVFP4 global scale outside F32 range"));
    }
    quantize(values, rows, k, Quantizer::nvfp4(global_scale))
}

/// Quantize E4M3 with F32 scales over an arbitrary rectangular block grid.
/// The 128x128 weight and 1x128 activation variants use this same formula.
/// PowerOfTwoF32 has a distinct contract ID from exact F32 scaling.
pub fn fp8_blocks(
    values: &[f32],
    rows: usize,
    k: usize,
    block_rows: usize,
    block_k: usize,
    mode: ScaleMode,
) -> Result<Quantized, Refusal> {
    check_input(values, rows, k, block_rows, block_k)?;
    quantize(
        values,
        rows,
        k,
        Quantizer::fp8_blocks((block_rows, block_k), mode),
    )
}

/// One F32 scale for the complete weight matrix.
pub fn fp8_weight_per_tensor(
    values: &[f32],
    rows: usize,
    k: usize,
    mode: ScaleMode,
) -> Result<Quantized, Refusal> {
    fp8_blocks(values, rows, k, rows, k, mode)
}

/// One F32 scale per weight row (output channel).
pub fn fp8_weight_per_channel(
    values: &[f32],
    rows: usize,
    k: usize,
    mode: ScaleMode,
) -> Result<Quantized, Refusal> {
    fp8_blocks(values, rows, k, 1, k, mode)
}

pub fn fp8_weight_128x128(
    values: &[f32],
    rows: usize,
    k: usize,
    mode: ScaleMode,
) -> Result<Quantized, Refusal> {
    fp8_blocks(values, rows, k, 128, 128, mode)
}

pub fn fp8_activation_1x128(
    values: &[f32],
    rows: usize,
    k: usize,
    mode: ScaleMode,
) -> Result<Quantized, Refusal> {
    fp8_blocks(values, rows, k, 1, 128, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_nvfp4_retains_the_supplied_global() {
        let values = (0..64)
            .map(|i| (i as f32 - 31.0) / 32.0)
            .collect::<Vec<_>>();
        let dynamic = nvfp4(&values, 2, 32).unwrap();
        assert_eq!(
            dynamic,
            nvfp4_with_global(&values, 2, 32, dynamic.global_scale).unwrap()
        );
        let calibrated = nvfp4_with_global(&values, 2, 32, 1.0 / 1024.0).unwrap();
        assert_eq!(
            calibrated.global_scale.to_bits(),
            (1.0_f32 / 1024.0).to_bits()
        );
        assert_ne!(calibrated.scales, dynamic.scales);
        // The stored calibration also survives a zero input batch.
        let zero = nvfp4_with_global(&[0.0; 32], 1, 32, 0.25).unwrap();
        assert_eq!(zero.global_scale, 0.25);
        assert_eq!(zero.codes, vec![0; 16]);
        assert_eq!(zero.scales, Scales::Ue4M3(vec![0; 2]));
        for invalid in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(nvfp4_with_global(&values, 2, 32, invalid).is_err());
        }
    }
    #[test]
    fn mx_and_nvfp4_zero_and_order() {
        for q in [
            mxfp8(&[0.0; 32], 1, 32).unwrap(),
            mxfp4(&[0.0; 32], 1, 32).unwrap(),
            nvfp4(&[0.0; 32], 1, 32).unwrap(),
        ] {
            assert!(q.decoded().iter().all(|x| *x == 0.0));
        }
        let values = (0_usize..32)
            .map(|v| if v.is_multiple_of(2) { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let q = mxfp4(&values, 1, 32).unwrap();
        let scale = match &q.scales {
            Scales::Ue8M0(v) => decode_ue8m0_f64(v[0]) as f32,
            _ => unreachable!(),
        };
        assert_eq!(q.codes[0] & 15, SmallFloat::E2M1.encode_f32(1.0 / scale));
        assert_eq!(q.codes[0] >> 4, SmallFloat::E2M1.encode_f32(-1.0 / scale));
        assert_eq!(
            q.decoded(),
            values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
        );
    }
    #[test]
    fn fp8_scale_modes_are_distinct() {
        let mut values = vec![0.0; 128];
        values[0] = 500.0;
        let exact = fp8_activation_1x128(&values, 1, 128, ScaleMode::ExactF32).unwrap();
        let pow2 = fp8_activation_1x128(&values, 1, 128, ScaleMode::PowerOfTwoF32).unwrap();
        assert_ne!(exact.scales, pow2.scales);
        assert_eq!(exact.decode(0, 1), 0.0);
    }

    #[test]
    fn fixed_block_scale_examples() {
        let mut fp8 = [0.0_f32; 32];
        fp8[0] = 256.0;
        let mx = mxfp8(&fp8, 1, 32).unwrap();
        assert_eq!(mx.scales, Scales::Ue8M0(vec![127]));
        assert_eq!(mx.codes[0], SmallFloat::E4M3.encode_f32(256.0));

        let mut fp4 = [0.0_f32; 32];
        fp4[0] = 6.0;
        let mx = mxfp4(&fp4, 1, 32).unwrap();
        assert_eq!(mx.scales, Scales::Ue8M0(vec![127]));
        assert_eq!(mx.codes[0] & 15, 7);

        let nv = nvfp4(&fp4, 1, 32).unwrap();
        assert_eq!(nv.global_scale, 1.0 / 448.0);
        assert_eq!(nv.scales, Scales::Ue4M3(vec![0x7e, 0]));
        assert_eq!(nv.codes[0] & 15, 7);
        assert!((nv.decode(0, 0) - 6.0).abs() < 1e-6);

        let mut block = vec![0.0; 128 * 128];
        block[0] = 448.0;
        let q = fp8_weight_128x128(&block, 128, 128, ScaleMode::ExactF32).unwrap();
        assert_eq!(q.scales, Scales::F32(vec![1.0]));
        assert_eq!(q.codes[0], 0x7e);
    }
}
