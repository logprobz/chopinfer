//! Dense and uniform grouped E4M3/UE8M0 projections with 128-K folds.

use crate::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use crate::contract::{
    DType, NumericalContract, NumericalContractId, Refusal, Validate, WeightFormat,
};
use crate::oracle::OracleOutput;
use crate::precision::execution::BlockScaledExecution;
use crate::precision::packed::{MatrixView, ScaleBytes};
use crate::precision::{DecodeOrder, NibbleOrder};
use crate::quantize::{Quantized, ScaleFormat, Scales};

pub const FOLD_K: u32 = 128;

/// Shared row-status bit for non-finite or invalid quantized input.
pub const NON_FINITE_INPUT: u32 = 1;

pub(crate) const OPERAND_REFUSAL: Refusal =
    Refusal("MXFP8 fold operands must be finite and exact in BF16");

/// Static weights are admitted on the host before their packed bytes can
/// be uploaded. Native execution and exact decoding share this domain.
pub fn admit_operand(matrix: MatrixView<'_>) -> Result<(), Refusal> {
    if matrix.format != WeightFormat::Mxfp8 {
        return Err(Refusal(
            "MXFP8 fold weights require a lossless per-row repack",
        ));
    }
    matrix
        .validate_bf16_operands(DecodeOrder::ExactScaledOperand)
        .map_err(|_| OPERAND_REFUSAL)
}

/// Each output is a sum over the exact decoded operands. Accumulate at most
/// 128 K from zero in F32, then add each partial to the outer F32 accumulator
/// in increasing K order. The last partial may contain fewer than 128 values;
/// padding it with zeros must not read past logical K. TF32 is not permitted.
///
/// Weights are `[groups, n, k]`, with row scales `[groups, n, k / 32]`.
/// Activation group `g` begins at `row * activation_row_stride +
/// g * activation_group_stride` (in elements), including its own 1x32 scale
/// groups. Outputs are contiguous `[row, groups, n]`. One group is dense GEMM;
/// multiple groups have independent activations, without routing or a sum
/// across groups. Both native and exact-decoded BF16 paths enforce finite,
/// BF16-exact scaled operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Mxfp8FoldGemmQuery {
    pub m_capacity: u32,
    pub n: u32,
    pub k: u32,
    pub groups: u32,
    pub activation_row_stride: u32,
    pub activation_group_stride: u32,
    pub output: DType,
    pub execution: BlockScaledExecution,
}

impl Validate for Mxfp8FoldGemmQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.m_capacity == 0
            || self.n == 0
            || self.k == 0
            || self.groups == 0
            || !self.k.is_multiple_of(32)
            || !self.activation_row_stride.is_multiple_of(32)
            || !self.activation_group_stride.is_multiple_of(32)
            || self.activation_group_stride < self.k
            || !matches!(self.output, DType::Bf16 | DType::F32)
        {
            return Err(Refusal(
                "MXFP8 folds require nonzero dimensions, complete 32-K groups and BF16 or F32 output",
            ));
        }
        let span = (self.groups - 1)
            .checked_mul(self.activation_group_stride)
            .and_then(|v| v.checked_add(self.k));
        if span.is_none_or(|v| v > self.activation_row_stride) {
            return Err(Refusal(
                "grouped MXFP8 activation offsets exceed the row stride",
            ));
        }
        let bounded = |dims: &[u32]| {
            dims.iter()
                .try_fold(4_u64, |n, &d| n.checked_mul(u64::from(d)))
                .is_some_and(|bytes| bytes <= crate::precision::MAX_TENSOR_BYTES)
        };
        if !bounded(&[self.m_capacity, self.activation_row_stride])
            || !bounded(&[self.groups, self.n, self.k])
            || !bounded(&[self.m_capacity, self.groups, self.n])
        {
            return Err(Refusal(
                "MXFP8 fold operand size exceeds the supported bound",
            ));
        }
        Ok(())
    }
}

impl NumericalContract for Mxfp8FoldGemmQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let execution = match self.execution {
            BlockScaledExecution::NativeBlockScaledMma => "native-block-scaled",
            BlockScaledExecution::ExactDecodedBf16Mma => "exact-decoded-bf16",
        };
        Ok(NumericalContractId::new(format!(
            "gemm.w-mxfp8.a-mxfp8.acc-f32.fold128.{execution}.v1"
        )))
    }
}

/// Validate the exact decoded domain, including signed zeros, NaN codes,
/// UE8M0 byte 255 and scaled values outside the finite BF16 range.
pub fn exact_bf16_values(matrix: &Quantized) -> Result<Vec<f64>, Refusal> {
    let Scales::Ue8M0(scales) = &matrix.scales else {
        return Err(Refusal("MXFP8 fold operands require UE8M0 scales"));
    };
    if matrix.format.element != SmallFloat::E4M3
        || matrix.format.scale != ScaleFormat::Ue8M0
        || matrix.format.block != (1, 32)
        || matrix.format.chunk_rows.is_some()
        || matrix.global_scale != 1.0
        || !matrix.k.is_multiple_of(32)
    {
        return Err(Refusal(
            "MXFP8 fold operands require per-row E4M3/32 without a global scale",
        ));
    }
    let view = MatrixView {
        tensor: "MXFP8 fold operand",
        format: WeightFormat::Mxfp8,
        rows: matrix.rows,
        k: matrix.k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &matrix.codes,
        local: Some(ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: scales,
        }),
        global: None,
    };
    admit_operand(view)?;
    Ok(matrix.decoded())
}

/// F64 oracle over the exact operands, before final output rounding. `rows`
/// is a nonzero live row count, no larger than the prepared capacity.
pub fn oracle(
    query: Mxfp8FoldGemmQuery,
    rows: usize,
    activations: &Quantized,
    weights: &Quantized,
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let (n, k, groups) = (query.n as usize, query.k as usize, query.groups as usize);
    if rows == 0
        || rows > query.m_capacity as usize
        || activations.rows != rows
        || activations.k != query.activation_row_stride as usize
        || weights.rows != groups * n
        || weights.k != k
    {
        return Err(Refusal(
            "MXFP8 fold oracle operands disagree with the query",
        ));
    }
    let a = exact_bf16_values(activations)?;
    let b = exact_bf16_values(weights)?;
    let mut values = Vec::with_capacity(rows * groups * n);
    let mut magnitudes = Vec::with_capacity(values.capacity());
    for row in 0..rows {
        for group in 0..groups {
            let start = row * activations.k + group * query.activation_group_stride as usize;
            for col in 0..n {
                let w = (group * n + col) * k;
                let (sum, magnitude) =
                    crate::gated_mlp_oracle::compensated_dot(&a[start..start + k], &b[w..w + k])?;
                values.push(sum);
                magnitudes.push(magnitude);
            }
        }
    }
    OracleOutput::new(values, magnitudes, k)
}

/// CPU witness of the fold boundaries for exactly representable BF16 inputs.
/// It does not claim the internal order of an MMA instruction or GPU evidence.
pub fn folded_dot(a: &[f32], b: &[f32]) -> Result<f32, Refusal> {
    if a.is_empty()
        || a.len() != b.len()
        || !a.len().is_multiple_of(32)
        || a.iter()
            .chain(b)
            .any(|&v| !v.is_finite() || bf16_to_f32(f32_to_bf16(v)).to_bits() != v.to_bits())
    {
        return Err(Refusal(
            "folded dot requires equal, complete groups of finite BF16 operands",
        ));
    }
    let mut sum = 0.0_f32;
    for (a, b) in a.chunks(FOLD_K as usize).zip(b.chunks(FOLD_K as usize)) {
        let mut partial = 0.0_f32;
        for (&a, &b) in a.iter().zip(b) {
            partial = a.mul_add(b, partial);
        }
        sum += partial;
    }
    Ok(sum)
}

crate::scratch_record! {
    /// E4M3 codes, UE8M0 scales and output rows in the query's layouts. The
    /// I32 device live count bounds all reads and writes of token rows.
    /// The caller initializes one U32 status per row. Launches preserve its
    /// bits and OR `NON_FINITE_INPUT` on an invalid activation code or scale.
    /// A flagged row writes zeros across every group and output column.
    /// `status_row_offset` addresses this launch's first row within the
    /// caller's status allocation; dead rows and other status words stay intact.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Mxfp8FoldGemmBuffers {
        pub activations: value,
        pub activation_scales: value,
        pub weights: value,
        pub weight_scales: value,
        pub output: value,
        pub live_rows: value,
        pub status: value,
        pub status_row_offset: scalar(u64),
    }
}

/// Buffers for a fold with persistent packed weights and self-resetting tickets.
/// `first` and optional `second` retain the ordinary fold's admission, status
/// and output contract. `packed_weights` are initialized from immutable weights
/// once, before capture, and have 16-byte aligned addresses. They may be
/// shared only after all initializers finish. `state` is U32-aligned,
/// initialized at the same time, and belongs to
/// this prepared operation until it is destroyed. Neither state nor workspace
/// may overlap another operation between a stream fork and its join. A failed
/// or interrupted launch requires reinitialization before the state is reused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mxfp8FoldStreamBuffers<B> {
    pub first: Mxfp8FoldGemmBuffers<B>,
    pub second: Option<Mxfp8FoldGemmBuffers<B>>,
    pub packed_weights: [B; 2],
    pub state: B,
}
