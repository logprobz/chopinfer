//! RMSNorm requirements owned by an operation's caller, without model tags.

use crate::contract::{DType, OutputQuantization, Refusal, RmsNormKind, RowQuery, Validate};

/// Layout of the input, output and optional residual rows. Weights are one
/// contiguous width-sized vector except in [`Self::StridedHeads`].
/// Alignment applies to every vector buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormLayout {
    ContiguousRows {
        alignment_bytes: u32,
    },
    StridedRows {
        stride_elements: u64,
        alignment_bytes: u32,
    },
    /// Rows of per-head norms (kind `QkPerHead`, `width / head_dim` heads):
    /// a row's heads are contiguous, input rows start `input_stride`
    /// elements apart and output rows `output_stride` apart, and the weight
    /// is one head-sized vector every head shares. It has no residual rows.
    StridedHeads {
        input_stride: u64,
        output_stride: u64,
        alignment_bytes: u32,
    },
}

impl RmsNormLayout {
    /// The elements from one input row's start to the next and from one
    /// output row's start to the next, for rows of `width` values.
    pub fn strides(&self, width: u32) -> (u64, u64) {
        match *self {
            Self::ContiguousRows { .. } => (u64::from(width), u64::from(width)),
            Self::StridedRows {
                stride_elements, ..
            } => (stride_elements, stride_elements),
            Self::StridedHeads {
                input_stride,
                output_stride,
                ..
            } => (input_stride, output_stride),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RmsNormReduction {
    Ordered,
    WarpTree { threads: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SquareAccumulation {
    FusedMultiplyAdd,
    SeparateNearestEven,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormReciprocal {
    ApproximateRsqrt,
    SqrtThenDivide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormRounding {
    NearestEven,
    TowardZero,
}

/// Reduction and rounding are part of compatibility, including the rounding
/// of a fused residual before computing the variance. A requirement pins them
/// only where its caller needs that exact arithmetic (a reference order, a
/// bitwise claim across backends); otherwise it leaves them to the backend.
/// A bitwise claim across backends needs more than these fields where they
/// leave the order open: `TREE_256` names no element-to-thread assignment or
/// tree shape, so two backends that both compute it need not agree bit for
/// bit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormNumerics {
    pub accumulator: DType,
    pub reduction: RmsNormReduction,
    pub square_accumulation: SquareAccumulation,
    pub reciprocal: RmsNormReciprocal,
    pub normalized_bf16: bool,
    pub residual_rounding: RmsNormRounding,
    pub output_rounding: RmsNormRounding,
}

impl RmsNormNumerics {
    /// A 256-thread warp tree of fused square accumulations and an
    /// approximate reciprocal square root. It does not pin which elements
    /// each thread accumulates or the shape of the tree, so pinning it alone
    /// does not make two backends bitwise equal.
    pub const TREE_256: Self = Self {
        accumulator: DType::F32,
        reduction: RmsNormReduction::WarpTree { threads: 256 },
        square_accumulation: SquareAccumulation::FusedMultiplyAdd,
        reciprocal: RmsNormReciprocal::ApproximateRsqrt,
        normalized_bf16: false,
        residual_rounding: RmsNormRounding::NearestEven,
        output_rounding: RmsNormRounding::NearestEven,
    };

    /// HF's order: squares rounded separately and summed in index order, a
    /// square root then a division, and the normalized row rounded to BF16
    /// before the weight.
    pub const HF_ORDERED: Self = Self {
        reduction: RmsNormReduction::Ordered,
        square_accumulation: SquareAccumulation::SeparateNearestEven,
        reciprocal: RmsNormReciprocal::SqrtThenDivide,
        normalized_bf16: true,
        ..Self::TREE_256
    };

    /// [`Self::HF_ORDERED`] with the sum of squares in a fixed 256-lane F32
    /// tree, a function of the width only: lane `l` adds the rounded
    /// squares of columns `8 l + 2048 i + j` (`i` ascending, then `j` = 0 to
    /// 7, below the width) from +0; each warp of 32 lanes adds its partials
    /// pairwise at lane distances 16, 8, 4, 2 and 1 (the lower lane's value
    /// first), and the eight warp sums the same way at distances 4, 2 and 1.
    pub const HF_TREE_256: Self = Self {
        reduction: RmsNormReduction::WarpTree { threads: 256 },
        ..Self::HF_ORDERED
    };
}

/// A stateless row operation. A fused residual is added in F32 and rounded
/// to the input format before normalization; its rounded rows are also
/// written to a distinct residual output. Rows beyond the live count are
/// untouched. No scale tensor, persistent state or implicit layout conversion
/// exists in this contract.
///
/// Every field but `numerics` is math. `numerics` is `None` when the caller
/// leaves the reduction order, square accumulation, reciprocal and rounding
/// points to the backend: the backend chooses them, names its choice in
/// `OpInfo::implementation`, and is judged by the oracle gate of
/// [`crate::oracle::rmsnorm`]. `Some` pins them; a backend that computes other
/// arithmetic refuses. The numerical contract id follows the query, so it is
/// the same either way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormRequirement {
    pub width: u32,
    pub epsilon_bits: u32,
    pub kind: RmsNormKind,
    pub input: DType,
    pub weight: DType,
    /// Output storage. Plain F32 output applies the weight in F32 and has
    /// no BF16 boundary before or after the weight. BF16 rounds the final
    /// product; the HF kind additionally rounds before the weight.
    pub output: DType,
    pub layout: RmsNormLayout,
    pub fused_residual: bool,
    pub output_quantization: Option<OutputQuantization>,
    pub numerics: Option<RmsNormNumerics>,
}

impl RmsNormRequirement {
    /// Explicit BF16 convenience composition. The caller chooses the norm
    /// kind and pins the numerical rules independently.
    pub fn bf16(
        width: u32,
        epsilon: f32,
        kind: RmsNormKind,
        fused_residual: bool,
        numerics: RmsNormNumerics,
    ) -> Self {
        Self {
            width,
            epsilon_bits: epsilon.to_bits(),
            kind,
            input: DType::Bf16,
            weight: DType::Bf16,
            output: DType::Bf16,
            layout: RmsNormLayout::ContiguousRows {
                alignment_bytes: 16,
            },
            fused_residual,
            output_quantization: None,
            numerics: Some(numerics),
        }
    }

    /// The BF16 composition of [`Self::bf16`] with backend-chosen numerics.
    pub fn bf16_backend_chosen(
        width: u32,
        epsilon: f32,
        kind: RmsNormKind,
        fused_residual: bool,
    ) -> Self {
        Self {
            numerics: None,
            ..Self::bf16(
                width,
                epsilon,
                kind,
                fused_residual,
                RmsNormNumerics::TREE_256,
            )
        }
    }

    /// The values of the weight vector: a head under
    /// [`RmsNormLayout::StridedHeads`], otherwise the width.
    pub fn weight_width(&self) -> u32 {
        match (self.layout, self.kind) {
            (RmsNormLayout::StridedHeads { .. }, RmsNormKind::QkPerHead { head_dim }) => head_dim,
            _ => self.width,
        }
    }

    /// The oracle reference ([`crate::oracle::rmsnorm`], whose gate applies)
    /// of the first `rows` rows of `input`, read in this requirement's
    /// layout, as compact rows of `width` values, in the order its numerics
    /// name ([`crate::oracle::rmsnorm_numerics`]: the HF kind's tree or
    /// serial sum). Each output row belongs at its row times the layout's
    /// output stride. Without a fused residual.
    pub fn oracle(
        &self,
        input: &[f64],
        weight: &[f64],
        rows: usize,
    ) -> Result<crate::oracle::OracleOutput, Refusal> {
        self.validate()?;
        if self.fused_residual {
            return Err(Refusal("the RMSNorm oracle reads no residual rows"));
        }
        let width = self.width as usize;
        let (stride, _) = self.layout.strides(self.width);
        let stride =
            usize::try_from(stride).map_err(|_| Refusal("RMSNorm stride exceeds the host"))?;
        if rows == 0
            || weight.len() != self.weight_width() as usize
            || rows
                .checked_sub(1)
                .and_then(|last| last.checked_mul(stride))
                .and_then(|start| start.checked_add(width))
                .is_none_or(|end| end > input.len())
        {
            return Err(Refusal("RMSNorm oracle rows or weight mismatch"));
        }
        let compact = (0..rows)
            .flat_map(|row| &input[row * stride..row * stride + width])
            .copied()
            .collect::<Vec<_>>();
        let weights = weight
            .iter()
            .copied()
            .cycle()
            .take(width)
            .collect::<Vec<_>>();
        crate::oracle::rmsnorm_numerics(
            &compact,
            width,
            &weights,
            f64::from(f32::from_bits(self.epsilon_bits)),
            self.kind,
            None,
            self.numerics,
        )
    }

    pub fn query(&self) -> RowQuery {
        RowQuery::RmsNorm {
            width: self.width,
            eps: f32::from_bits(self.epsilon_bits),
            kind: self.kind,
            fused_residual: self.fused_residual,
            output_quantization: self.output_quantization,
        }
    }
}

impl Validate for RmsNormRequirement {
    fn validate(&self) -> Result<(), Refusal> {
        self.query().validate()?;
        if let RmsNormLayout::StridedHeads {
            input_stride,
            output_stride,
            alignment_bytes,
        } = self.layout
        {
            if !matches!(self.kind, RmsNormKind::QkPerHead { .. }) {
                return Err(Refusal("strided heads normalize each head"));
            }
            if self.fused_residual {
                return Err(Refusal("strided heads carry no residual rows"));
            }
            let width = u64::from(self.width);
            if input_stride < width || output_stride < width {
                return Err(Refusal("strided head rows overlap"));
            }
            if !alignment_bytes.is_power_of_two() {
                return Err(Refusal("RMSNorm alignment is a power of two"));
            }
        }
        Ok(())
    }
}

crate::scratch_record! {
    /// An RMSNorm's buffers by role: input rows `x`, the width-sized
    /// `weight`, output rows `y` and the device count of live rows. A fused
    /// residual supplies its input rows and the rows of the rounded sum it
    /// writes; that output may alias its input, but not `y`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RmsNormBuffers {
        pub x: value,
        pub weight: value,
        pub residual: optional(FusedResidual),
        pub y: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A fused residual's input rows and the rounded sum's output rows. An
    /// RMSNorm and the fused all-reduces share it; each op says whether the
    /// output may alias the input.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct FusedResidual {
        pub input: value,
        pub output: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{bf16_to_f32, f32_to_bf16};
    use crate::contract::NumericalContract;
    use crate::oracle;

    fn id(requirement: RmsNormRequirement) -> String {
        requirement
            .query()
            .numerical_contract_id()
            .unwrap()
            .to_string()
    }

    /// The presets keep their contract ids, and leaving the numerics to the
    /// backend keeps the same ids: the id names the math, not the arithmetic.
    #[test]
    fn contract_ids_follow_the_kind_alone() {
        for (kind, numerics, expected) in [
            (
                RmsNormKind::Plain,
                RmsNormNumerics::TREE_256,
                "row.rmsnorm-plain.v1",
            ),
            (
                RmsNormKind::GemmaOnePlusWeight,
                RmsNormNumerics::TREE_256,
                "row.rmsnorm-gemma.v1",
            ),
            (
                RmsNormKind::QkPerHead { head_dim: 128 },
                RmsNormNumerics::TREE_256,
                "row.rmsnorm-qk.v1",
            ),
            (
                RmsNormKind::HfBf16Intermediate,
                RmsNormNumerics::HF_ORDERED,
                "row.rmsnorm-hf-bf16.v1",
            ),
        ] {
            for fused in [false, true] {
                let pinned = RmsNormRequirement::bf16(4096, 1e-6, kind, fused, numerics);
                let chosen = RmsNormRequirement::bf16_backend_chosen(4096, 1e-6, kind, fused);
                assert_eq!(pinned.numerics, Some(numerics));
                assert_eq!(chosen.numerics, None);
                assert_eq!(
                    chosen,
                    RmsNormRequirement {
                        numerics: None,
                        ..pinned
                    }
                );
                assert_eq!(chosen.query(), pinned.query());
                assert_eq!(id(pinned), expected);
                assert_eq!(id(chosen), expected);
            }
        }
    }

    fn rows(width: usize, rows: usize, seed: u32) -> Vec<u16> {
        (0..width * rows)
            .map(|i| {
                let v = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 20;
                f32_to_bf16((v % 257) as f32 / 64.0 - 2.0)
            })
            .collect()
    }

    fn widen(bits: &[u16]) -> Vec<f64> {
        bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
    }

    /// A 256-lane tree of fused square accumulations over vectors of eight,
    /// an F32 reciprocal square root one ulp off, and BF16 rounding: an
    /// arithmetic other than the reference's, which the plain kind's gate
    /// admits.
    fn tree_norm(x: &[u16], weight: &[u16], eps: f32) -> Vec<f64> {
        let width = weight.len();
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(width) {
            let mut lanes = [0.0_f32; 256];
            for (i, &v) in row.iter().enumerate() {
                let v = bf16_to_f32(v);
                let lane = &mut lanes[i / 8 % 256];
                *lane = v.mul_add(v, *lane);
            }
            let mut offset = 128;
            while offset > 0 {
                for lane in 0..offset {
                    lanes[lane] += lanes[lane + offset];
                }
                offset /= 2;
            }
            let mean = lanes[0] / width as f32 + eps;
            let inv = f32::from_bits((1.0 / mean.sqrt()).to_bits() + 1);
            for (&v, &w) in row.iter().zip(weight) {
                let y = bf16_to_f32(v) * inv * bf16_to_f32(w);
                out.push(f64::from(bf16_to_f32(f32_to_bf16(y))));
            }
        }
        out
    }

    fn words(values: &[f64]) -> Vec<u16> {
        values.iter().map(|&v| f32_to_bf16(v as f32)).collect()
    }

    /// The gate of a backend-chosen norm is the oracle's: any F32 order
    /// within the F64 bound for the plain kind, HF's BF16 words alone for
    /// the HF kind.
    #[test]
    fn the_oracle_gate_admits_other_orders_except_for_the_hf_kind() {
        let eps = 1e-6_f32;
        for width in [1024, 4096] {
            let (x, weight) = (rows(width, 3, 7), rows(width, 1, 11));
            let reference = oracle::rmsnorm(
                &widen(&x),
                width,
                &widen(&weight),
                f64::from(eps),
                RmsNormKind::Plain,
                None,
            )
            .unwrap();
            let tree = tree_norm(&x, &weight, eps);
            oracle::check_numeric(&tree, &reference, DType::Bf16).unwrap();

            let hf = oracle::rmsnorm(
                &widen(&x),
                width,
                &widen(&weight),
                f64::from(eps),
                RmsNormKind::HfBf16Intermediate,
                None,
            )
            .unwrap();
            oracle::check_exact(&words(&hf.values), &words(&hf.values)).unwrap();
            // One BF16 step off one output, either way, fails the HF gate.
            let index = hf.values.iter().position(|&v| v != 0.0).unwrap();
            for step in [1, -1] {
                let mut off = words(&hf.values);
                off[index] = off[index].wrapping_add_signed(step);
                assert_eq!(oracle::check_exact(&off, &words(&hf.values)), Err(index));
            }
        }
    }

    fn strided_heads(heads: u32, input_stride: u64, output_stride: u64) -> RmsNormRequirement {
        RmsNormRequirement {
            width: heads * 128,
            layout: RmsNormLayout::StridedHeads {
                input_stride,
                output_stride,
                alignment_bytes: 16,
            },
            ..RmsNormRequirement::bf16_backend_chosen(
                heads * 128,
                1e-6,
                RmsNormKind::QkPerHead { head_dim: 128 },
                false,
            )
        }
    }

    /// Strided heads state math alone: per-head norms of rows that do not
    /// overlap, without residual rows, at a power-of-two alignment. The
    /// query, and so the contract id, is the per-head kind's.
    #[test]
    fn strided_heads_validate_the_math_alone() {
        let ok = strided_heads(8, 8 * 128 * 3, 8 * 128);
        assert_eq!(ok.validate(), Ok(()));
        assert_eq!(ok.weight_width(), 128);
        assert_eq!(ok.layout.strides(ok.width), (8 * 128 * 3, 8 * 128));
        assert_eq!(id(ok), "row.rmsnorm-qk.v1");
        // Any alignment and stride a backend might not serve is still math.
        let odd = RmsNormRequirement {
            layout: RmsNormLayout::StridedHeads {
                input_stride: u64::from(u32::MAX) + 1,
                output_stride: 8 * 128,
                alignment_bytes: 1,
            },
            ..ok
        };
        assert_eq!(odd.validate(), Ok(()));
        let refused = |requirement: RmsNormRequirement, rule: &'static str| {
            assert_eq!(
                requirement.validate(),
                Err(Refusal(rule)),
                "{requirement:?}"
            );
        };
        refused(
            RmsNormRequirement {
                kind: RmsNormKind::Plain,
                ..ok
            },
            "strided heads normalize each head",
        );
        refused(
            RmsNormRequirement {
                fused_residual: true,
                ..ok
            },
            "strided heads carry no residual rows",
        );
        refused(
            strided_heads(8, 8 * 128 - 1, 8 * 128),
            "strided head rows overlap",
        );
        refused(
            strided_heads(8, 8 * 128, 8 * 128 - 1),
            "strided head rows overlap",
        );
        refused(
            RmsNormRequirement {
                layout: RmsNormLayout::StridedHeads {
                    input_stride: 8 * 128,
                    output_stride: 8 * 128,
                    alignment_bytes: 24,
                },
                ..ok
            },
            "RMSNorm alignment is a power of two",
        );
        // Heads that do not divide the width are the query's refusal.
        assert!(
            RmsNormRequirement {
                width: 8 * 128 + 8,
                ..strided_heads(8, 8 * 128 * 3, 8 * 128 * 3)
            }
            .validate()
            .is_err()
        );
        // The other layouts keep their weight width and strides.
        let contiguous =
            RmsNormRequirement::bf16_backend_chosen(4096, 1e-6, RmsNormKind::Plain, false);
        assert_eq!(contiguous.weight_width(), 4096);
        assert_eq!(contiguous.layout.strides(4096), (4096, 4096));
    }

    /// A one-element-per-thread head norm: 128 rounded squares summed by a
    /// shuffle-down tree in each of four warps and again over the warp
    /// sums, a reciprocal square root, and the output rounded to BF16 once.
    fn head_norm(x: &[u16], weight: &[u16], eps: f32) -> Vec<f64> {
        let tree = |mut lanes: [f32; 32]| {
            let mut offset = 16;
            while offset > 0 {
                for lane in 0..32 - offset {
                    lanes[lane] += lanes[lane + offset];
                }
                offset /= 2;
            }
            lanes[0]
        };
        x.chunks_exact(128)
            .flat_map(|head| {
                let mut warps = [0.0_f32; 32];
                for (warp, lanes) in head.chunks_exact(32).enumerate() {
                    warps[warp] = tree(std::array::from_fn(|lane| {
                        let v = bf16_to_f32(lanes[lane]);
                        v * v
                    }));
                }
                let inv = 1.0 / (tree(warps) * (1.0 / 128.0) + eps).sqrt();
                head.iter()
                    .zip(weight)
                    .map(move |(&v, &w)| {
                        let y = bf16_to_f32(v) * inv * bf16_to_f32(w);
                        f64::from(bf16_to_f32(f32_to_bf16(y)))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The oracle reads strided heads at the input stride and applies the
    /// head's weight to every head, as the per-head kind over the gathered
    /// rows with the weight repeated does; a one-element-per-thread head
    /// norm passes its gate.
    #[test]
    fn the_strided_heads_oracle_reads_rows_by_stride() {
        let eps = 1e-6_f32;
        for (heads, stride, count) in [(4, 4 * 128 * 3, 3), (1, 7 * 128, 5), (8, 8 * 128, 2)] {
            let requirement = strided_heads(heads, stride, u64::from(heads) * 128);
            let width = heads as usize * 128;
            let stride = stride as usize;
            let input = rows(stride, count, 13);
            let weight = rows(128, 1, 17);
            let reference = requirement
                .oracle(&widen(&input), &widen(&weight), count)
                .unwrap();
            let gathered = (0..count)
                .flat_map(|row| &input[row * stride..row * stride + width])
                .copied()
                .collect::<Vec<_>>();
            let repeated = weight.repeat(heads as usize);
            let direct = oracle::rmsnorm(
                &widen(&gathered),
                width,
                &widen(&repeated),
                f64::from(eps),
                requirement.kind,
                None,
            )
            .unwrap();
            assert_eq!(reference, direct);
            oracle::check_numeric(&head_norm(&gathered, &weight, eps), &reference, DType::Bf16)
                .unwrap();
            // Rows past the input, a weight of the row's width and a fused
            // residual are refused.
            assert!(
                requirement
                    .oracle(
                        &widen(&input[..stride * (count - 1) + width - 1]),
                        &widen(&weight),
                        count
                    )
                    .is_err()
            );
            if heads > 1 {
                assert!(
                    requirement
                        .oracle(&widen(&input), &widen(&repeated), count)
                        .is_err()
                );
            }
        }
        let contiguous =
            RmsNormRequirement::bf16_backend_chosen(1024, eps, RmsNormKind::Plain, false);
        let (x, weight) = (rows(1024, 3, 7), rows(1024, 1, 11));
        assert_eq!(
            contiguous.oracle(&widen(&x), &widen(&weight), 3).unwrap(),
            oracle::rmsnorm(
                &widen(&x),
                1024,
                &widen(&weight),
                f64::from(eps),
                RmsNormKind::Plain,
                None
            )
            .unwrap()
        );
        assert!(
            RmsNormRequirement {
                fused_residual: true,
                ..contiguous
            }
            .oracle(&widen(&x), &widen(&weight), 3)
            .is_err()
        );
    }

    /// The HF kind's exact reference follows the requirement's numerics. On
    /// a row whose 256-lane tree and serial sums of squares differ (1.0078125
    /// then 4095 values of 2^-13), the outputs differ by one BF16 step at
    /// column 1: the tree reference is the tree order's HF norm and refuses
    /// the serial bits, and the serial reference the other way round.
    #[test]
    fn the_hf_reference_follows_the_sum_order_of_its_numerics() {
        let mut x = vec![f32_to_bf16(2.0_f32.powi(-13)); 4096];
        x[0] = 0x3f81;
        let weight = vec![f32_to_bf16(1.0); 4096];
        let eps = 1e-6_f32;
        let requirement = |numerics| {
            RmsNormRequirement::bf16(4096, eps, RmsNormKind::HfBf16Intermediate, false, numerics)
        };
        let reference = |requirement: RmsNormRequirement| {
            words(
                &requirement
                    .oracle(&widen(&x), &widen(&weight), 1)
                    .unwrap()
                    .values,
            )
        };
        let tree = reference(requirement(RmsNormNumerics::HF_TREE_256));
        let ordered = reference(requirement(RmsNormNumerics::HF_ORDERED));
        // HF's norm of the row from a given sum of squares.
        let hf_norm = |sum: f32| {
            let inv = 1.0 / (sum / 4096.0 + eps).sqrt();
            x.iter()
                .zip(&weight)
                .map(|(&v, &w)| {
                    let n = bf16_to_f32(f32_to_bf16(bf16_to_f32(v) * inv));
                    f32_to_bf16(n * bf16_to_f32(w))
                })
                .collect::<Vec<_>>()
        };
        let squares = x
            .iter()
            .map(|&v| {
                let v = bf16_to_f32(v);
                v * v
            })
            .collect::<Vec<_>>();
        let serial = squares.iter().fold(0.0_f32, |sum, &square| sum + square);
        let tree_sum = oracle::hf_tree_256_sum(&squares);
        assert_ne!(tree_sum, serial);
        assert_eq!(tree, hf_norm(tree_sum));
        assert_eq!(ordered, hf_norm(serial));
        assert_eq!(tree[1].abs_diff(ordered[1]), 1);
        assert_eq!(oracle::check_exact(&ordered, &tree), Err(1));
        assert_eq!(oracle::check_exact(&tree, &ordered), Err(1));
        // Left to the backend, the HF kind keeps HF's serial order, the
        // direct oracle's.
        let chosen = RmsNormRequirement::bf16_backend_chosen(
            4096,
            eps,
            RmsNormKind::HfBf16Intermediate,
            false,
        );
        assert_eq!(reference(chosen), ordered);
        // Numerics without an exact HF reference are refused.
        let approximate = requirement(RmsNormNumerics::TREE_256);
        assert!(approximate.oracle(&widen(&x), &widen(&weight), 1).is_err());
        // The other kinds' F64 reference does not depend on the order.
        let plain = |numerics| RmsNormRequirement {
            kind: RmsNormKind::Plain,
            ..requirement(numerics)
        };
        assert_eq!(
            plain(RmsNormNumerics::TREE_256)
                .oracle(&widen(&x), &widen(&weight), 1)
                .unwrap(),
            plain(RmsNormNumerics::HF_TREE_256)
                .oracle(&widen(&x), &widen(&weight), 1)
                .unwrap()
        );
    }

    /// At a power of two the step below is half an ulp of the reference
    /// away: `check_numeric` admits it, the HF gate does not.
    #[test]
    fn the_hf_gate_refuses_the_step_below_a_power_of_two() {
        let hf = oracle::rmsnorm(
            &[1.0; 8],
            8,
            &[1.0; 8],
            f64::from(1e-6_f32),
            RmsNormKind::HfBf16Intermediate,
            None,
        )
        .unwrap();
        assert_eq!(hf.values, [1.0; 8]);
        let mut below = hf.values.clone();
        below[0] = 0.99609375;
        assert_eq!(oracle::numeric_bound(&hf, 0, DType::Bf16), 1.0 - below[0]);
        assert!(oracle::check_numeric(&below, &hf, DType::Bf16).is_ok());
        assert_eq!(
            oracle::check_exact(&words(&below), &words(&hf.values)),
            Err(0)
        );
    }
}
