//! Sinkhorn-mixed hyper-connections: the residual is `streams` (N) BF16
//! copies `X_i` of width H per token. Before each sublayer a learned mixer
//! turns the token's streams into a collapse weight `pre`, a placement
//! weight `post` and an N x N doubly stochastic stream mixer `comb`; the
//! sublayer reads the `pre`-weighted sum of the streams, and its output y
//! returns as `X'_j = post_j * y + sum_i comb[i][j] * X_i`. This is a
//! different op from the sigmoid-mean hyper-connection
//! (`contract::SigmoidMeanHyperConnectionQuery`).
//!
//! Per sublayer the parameters are `fn` `[(2 + N) * N, N * H]` (BF16 or
//! F32), `base` `[(2 + N) * N]` F32 and `scale` `[3]` F32.
//!
//! Every field of [`SinkhornMix`] is math the model states. A query's
//! `capacity` bounds the rows of one launch and is not math. The order of
//! the mix projection's F32 sums is the backend's choice, judged by the
//! oracle's accumulation gate; the rest of each query's arithmetic is
//! pinned.

use crate::collective::AllReduceBuffers;
use crate::contract::{
    DType, NumericalContract, NumericalContractId, Refusal, Validate, valid_eps,
};
use crate::rmsnorm::RmsNormReduction;

/// The order of the Sinkhorn normalizations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SinkhornOrder {
    /// `comb = softmax_j(C) + eps` (the row softmax stands in for the first
    /// row normalization), then iteration 1 is one column normalization and
    /// each of iterations 2..=n is a row normalization followed by a column
    /// normalization. With n = 20 that is one column normalization plus 19
    /// (row, column) rounds.
    RowSoftmaxThenColumnFirst,
}

/// Where the mixer runs relative to the sublayers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HyperConnectionPlacement {
    /// A mixer around every sublayer (attention and MLP each).
    PerSublayer,
    /// One mixer per layer computed a sublayer late in a single pass.
    /// Declared; refused.
    LaggedSinglePass,
}

/// How the final stream collapse before the model head works.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StreamHeadMode {
    /// `bf16(sum_i f32(X_i) / N)`, the sum in stream order.
    MeanOfStreams,
    /// A learned sigmoid-weighted collapse. Declared; refused.
    SigmoidWeighted,
}

/// The mixer's parameters (4 streams, width 4096, 20 iterations, eps 1e-6,
/// norm eps 1e-5 for the served model).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SinkhornMix {
    pub streams: u32,
    pub width: u32,
    pub iterations: u32,
    /// F32 bits of the eps added in `pre`, `comb` and every normalization.
    pub eps_bits: u32,
    /// F32 bits of the unweighted RMS norm's eps.
    pub norm_eps_bits: u32,
    pub order: SinkhornOrder,
    pub placement: HyperConnectionPlacement,
    /// Storage of `fn`: BF16 or F32.
    pub fn_dtype: DType,
}

impl SinkhornMix {
    /// Mixer outputs per token: `(2 + N) * N`.
    pub fn mixes(&self) -> u32 {
        (2 + self.streams) * self.streams
    }

    /// The projection's K: `N * H`.
    pub fn flat_width(&self) -> u32 {
        self.streams * self.width
    }
}

/// Checks the math only: the widths, iteration counts and stream counts a
/// backend's kernels hold are its own limits. The projection's K `N * H`
/// and its mixes `(2 + N) * N` must fit u32 ([`SinkhornMix::flat_width`],
/// [`SinkhornMix::mixes`]).
impl Validate for SinkhornMix {
    fn validate(&self) -> Result<(), Refusal> {
        let representable = self.streams.checked_mul(self.width).is_some()
            && self
                .streams
                .checked_add(2)
                .and_then(|n| n.checked_mul(self.streams))
                .is_some();
        if self.streams == 0
            || self.width == 0
            || !representable
            || self.iterations == 0
            || !valid_eps(self.eps_bits)
            || !valid_eps(self.norm_eps_bits)
            || !matches!(self.fn_dtype, DType::Bf16 | DType::F32)
        {
            return Err(Refusal(
                "invalid Sinkhorn mixer streams, width, iterations, eps or fn dtype",
            ));
        }
        if self.placement == HyperConnectionPlacement::LaggedSinglePass {
            return Err(Refusal(
                "lagged single-pass hyper-connection placement is declared but not supported",
            ));
        }
        Ok(())
    }
}

/// Sinkhorn hyper-connection contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SinkhornHyperConnectionQuery {
    /// Per token, from X `[rows, N, H]` BF16, all F32 with IEEE FMA:
    ///
    /// 1. `rstd = rsqrt(sum_k x_k^2 / (N * H) + norm_eps)` over the flat
    ///    `x = f32(X)` (no weight), correctly rounded.
    /// 2. `logit_m = rstd * sum_k fn[m][k] * x_k` for every mix m: the
    ///    projection runs over the raw x, then one multiply by rstd (equal
    ///    to projecting the normalized x in exact arithmetic). The order of
    ///    its F32 sums is the backend's.
    /// 3. `pre_i = sigmoid(fma(logit_i, scale[0], base[i])) + eps`,
    ///    `post_i = 2 * sigmoid(fma(logit_{N+i}, scale[1], base[N+i]))` and
    ///    `C[i][j] = fma(logit_{2N+iN+j}, scale[2], base[2N+iN+j])`, with
    ///    `sigmoid(v) = 1 / (1 + exp(-v))`.
    /// 4. Sinkhorn per `SinkhornOrder`: a row normalization is `C[i][j] /=
    ///    sum_j C[i][j] + eps`, a column normalization `C[i][j] /= sum_i
    ///    C[i][j] + eps`, sums in index order; the softmax subtracts the
    ///    row maximum.
    /// 5. `collapsed = bf16(sum_i pre_i * f32(X_i))`, FMA from zero in
    ///    stream order.
    ///
    /// Outputs: collapsed `[rows, H]` BF16, `post` `[rows, N]` and `comb`
    /// `[rows, N, N]` (`comb[i][j]` at `i * N + j`), each rounded once to
    /// BF16 as the reference rounds them before use. `pre` is not stored.
    MixCollapse { mix: SinkhornMix, capacity: u32 },
    /// `MixCollapse`, then the sublayer's input RMSNorm of the collapsed row
    /// x (`RmsNormKind::HfBf16Intermediate` with a BF16 weight w, the
    /// numerics of `RmsNormNumerics::HF_TREE_256`), per token:
    ///
    /// 1. `MixCollapse`'s outputs, as that variant states them.
    /// 2. `s` = the squares `fl(x_k * x_k)` of the stored BF16 collapsed row
    ///    added in a fixed 256-lane F32 tree: lane `l` adds the squares of
    ///    columns `8 l + 2048 i + j` (`i` ascending, then `j` = 0 to 7,
    ///    below H) from +0; each warp of 32 lanes adds its partials pairwise
    ///    at lane distances 16, 8, 4, 2 and 1 (lane m adds lane m + d), and
    ///    the eight warp sums the same way at distances 4, 2 and 1.
    /// 3. `inv = 1 / sqrtf(s / H + eps)`, with the division by H, the add,
    ///    the square root and the last division each correctly rounded in
    ///    F32, and `normed_k = bf16(bf16(x_k * inv) * w_k)`.
    ///
    /// Writes `MixCollapse`'s outputs and the normed row `[rows, H]` BF16.
    /// The normed row is an exact function of the collapsed row and w.
    MixCollapseNorm {
        mix: SinkhornMix,
        rmsnorm_eps_bits: u32,
        capacity: u32,
    },
    /// `X'_j = bf16(bf16(post_j * f32(y)) + bf16(sum_i comb[i][j] *
    /// f32(X_i)))` per element, with y the sublayer output `[rows, H]` BF16,
    /// `post` and `comb` the BF16 values `MixCollapse` wrote, products in
    /// F32 and the stream sum F32 FMA from zero in stream order. Every
    /// rounding is the reference's BF16 mixing.
    Expand {
        streams: u32,
        width: u32,
        capacity: u32,
    },
    /// The collapse before the final norm and model head, `[rows, H]` BF16.
    Head {
        streams: u32,
        width: u32,
        mode: StreamHeadMode,
        capacity: u32,
    },
    /// The fused comm epilogue, per token:
    ///
    /// 1. `y = bf16(sum_r p_r)`: the ranks' F32 sublayer partials `[rows, H]`
    ///    reduced as the unfused `CommQuery::AllReduce` of `dtype`: with
    ///    F32, added in F32 from zero in rank order and the sum rounded once
    ///    to BF16; with BF16, each rank's partial rounded once to BF16
    ///    before that sum (the BF16-partials all-reduce, whose output is y).
    /// 2. `Expand` of y into X'.
    /// 3. The next sublayer's `MixCollapse` of X'.
    /// 4. Its input RMSNorm of the collapsed row x
    ///    (`RmsNormKind::HfBf16Intermediate` with a BF16 weight w):
    ///    `s = sum_k fl(x_k * x_k)` in F32 in the order `norm` names,
    ///    `inv = 1 / sqrtf(s / H + eps)` (a correctly rounded square root,
    ///    then a correctly rounded division), and
    ///    `bf16(bf16(x_k * inv) * w_k)`. `norm` is math the model states:
    ///    `RmsNormReduction::Ordered` adds the squares from zero in column
    ///    order (`RmsNormNumerics::HF_ORDERED`);
    ///    `RmsNormReduction::WarpTree { threads: 256 }` adds them in
    ///    `MixCollapseNorm`'s 256-lane tree (`RmsNormNumerics::HF_TREE_256`).
    ///    No other tree is a contract.
    ///
    /// Writes y, X', the next `post` and `comb`, the collapsed row and the
    /// normed row, bit-identical to the unfused sequence (whose norm sums
    /// in `norm`'s order). The widths a backend fuses are its own limit.
    AllReduceExpandMixNorm {
        ranks: u32,
        dtype: DType,
        mix: SinkhornMix,
        rmsnorm_eps_bits: u32,
        norm: RmsNormReduction,
        capacity: u32,
    },
}

impl Validate for SinkhornHyperConnectionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        match *self {
            Self::MixCollapse { mix, capacity } => {
                mix.validate()?;
                if capacity == 0 {
                    return Err(Refusal("zero hyper-connection capacity"));
                }
            }
            Self::MixCollapseNorm {
                mix,
                rmsnorm_eps_bits,
                capacity,
            } => {
                mix.validate()?;
                if !valid_eps(rmsnorm_eps_bits) || capacity == 0 {
                    return Err(Refusal(
                        "invalid hyper-connection mix and norm eps or capacity",
                    ));
                }
            }
            Self::Expand {
                streams,
                width,
                capacity,
            } => {
                if streams == 0 || width == 0 || capacity == 0 {
                    return Err(Refusal("invalid hyper-connection expand geometry"));
                }
            }
            Self::Head {
                streams,
                width,
                mode,
                capacity,
            } => {
                if streams == 0 || width == 0 || capacity == 0 {
                    return Err(Refusal("invalid hyper-connection head geometry"));
                }
                if mode == StreamHeadMode::SigmoidWeighted {
                    return Err(Refusal(
                        "a sigmoid-weighted stream head is declared but not supported",
                    ));
                }
            }
            Self::AllReduceExpandMixNorm {
                ranks,
                dtype,
                mix,
                rmsnorm_eps_bits,
                norm,
                capacity,
            } => {
                mix.validate()?;
                if ranks == 0
                    || !matches!(dtype, DType::Bf16 | DType::F32)
                    || !valid_eps(rmsnorm_eps_bits)
                    || capacity == 0
                {
                    return Err(Refusal(
                        "invalid fused hyper-connection all-reduce ranks, dtype, eps or capacity",
                    ));
                }
                if !matches!(
                    norm,
                    RmsNormReduction::Ordered | RmsNormReduction::WarpTree { threads: 256 }
                ) {
                    return Err(Refusal(
                        "the fused hyper-connection norm sums in column order or the 256-lane tree",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl NumericalContract for SinkhornHyperConnectionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::MixCollapse { .. } => "row.hc-sinkhorn-mix-collapse.v1",
            Self::MixCollapseNorm { .. } => "row.hc-sinkhorn-mix-collapse-rmsnorm-tree256.v1",
            Self::Expand { .. } => "row.hc-sinkhorn-expand-bf16.v1",
            Self::Head { .. } => "row.hc-stream-mean.v1",
            Self::AllReduceExpandMixNorm {
                norm: RmsNormReduction::Ordered,
                ..
            } => "comm.all-reduce-hc-sinkhorn-expand-mix-rmsnorm.v1",
            Self::AllReduceExpandMixNorm { .. } => {
                "comm.all-reduce-hc-sinkhorn-expand-mix-rmsnorm-tree256.v1"
            }
        };
        Ok(NumericalContractId::new(name.to_owned()))
    }
}

crate::scratch_record! {
    /// A Sinkhorn mixer's learned parameters for one sublayer, row-major:
    /// `fn_` `[(2 + N) * N, N * H]` of the mixer's `fn_dtype`, F32 `base`
    /// `[(2 + N) * N]` and F32 `scale` `[3]`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MixerWeights {
        pub fn_: value,
        pub base: value,
        pub scale: value,
    }
}

crate::scratch_record! {
    /// Scratch a mixer may use. The op lists the regions it uses and their
    /// least bytes in its info (`scratch.partials`, `scratch.tickets`), and
    /// the host passes those. `partials` is rewritten by every launch
    /// before it is read; `tickets` is zeroed once before the first launch,
    /// and every launch leaves it zeroed.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MixScratch {
        pub partials: option,
        pub tickets: option,
    }
}

crate::scratch_record! {
    /// A mix-collapse's buffers (`SinkhornHyperConnectionQuery::MixCollapse`):
    /// the BF16 streams `x` `[rows, N * H]` and the sublayer's `mixer`
    /// weights it reads, and the BF16 `collapsed` `[rows, H]`, `post`
    /// `[rows, N]` and `comb` `[rows, N * N]` it writes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MixCollapseBuffers {
        pub x: value,
        pub mixer: record(MixerWeights),
        pub collapsed: value,
        pub post: value,
        pub comb: value,
        pub scratch: record(MixScratch),
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A mix-collapse and input norm's buffers
    /// (`SinkhornHyperConnectionQuery::MixCollapseNorm`): a mix-collapse's
    /// buffers, the BF16 input norm weight (`norm_weight`, `[H]`) it reads
    /// and the BF16 `normed` rows `[rows, H]` it writes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MixCollapseNormBuffers {
        pub x: value,
        pub mixer: record(MixerWeights),
        pub norm_weight: value,
        pub collapsed: value,
        pub post: value,
        pub comb: value,
        pub normed: value,
        pub scratch: record(MixScratch),
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// An expand's buffers (`SinkhornHyperConnectionQuery::Expand`): it
    /// reads the BF16 streams `x` `[rows, N * H]`, the BF16 sublayer output
    /// `y` `[rows, H]` and the `post` and `comb` the mix-collapse wrote, and
    /// writes the BF16 expanded streams `output` of `x`'s shape.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct StreamExpandBuffers {
        pub x: value,
        pub y: value,
        pub post: value,
        pub comb: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A stream head's buffers (`SinkhornHyperConnectionQuery::Head`): the
    /// BF16 streams `x` `[rows, N * H]` and the BF16 collapsed `output`
    /// `[rows, H]`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct StreamHeadBuffers {
        pub x: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// The fused comm epilogue's buffers on one rank
    /// (`SinkhornHyperConnectionQuery::AllReduceExpandMixNorm`): the
    /// all-reduce of the sublayer's F32 partials (`reduce.input`) into this
    /// rank's F32 sums (`reduce.output`) or, with BF16 partials, its rounded
    /// sums (`reduce.output`, BF16), with every rank's views and state;
    /// the BF16 streams the sublayer read (`streams`); the rounded sum it
    /// writes (`reduced`, `[rows, H]`) and the `expanded` streams; the next
    /// sublayer's `mixer` weights and the `collapsed` rows its mix-collapse
    /// writes; and the BF16 input norm weight (`norm_weight`, `[H]`) and the
    /// `normed` rows it writes.
    ///
    /// `post` and `comb` are read and written: they hold the ending
    /// sublayer's mix-collapse outputs, which the expand reads, and the op
    /// overwrites them with the next sublayer's.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ReduceExpandMixNormBuffers {
        pub reduce: record(AllReduceBuffers),
        pub streams: value,
        pub reduced: value,
        pub expanded: value,
        pub mixer: record(MixerWeights),
        pub post: value,
        pub comb: value,
        pub collapsed: value,
        pub norm_weight: value,
        pub normed: value,
        pub scratch: record(MixScratch),
        pub live_rows: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mix(width: u32, eps: f32) -> SinkhornMix {
        SinkhornMix {
            streams: 4,
            width,
            iterations: 20,
            eps_bits: eps.to_bits(),
            norm_eps_bits: 1e-5_f32.to_bits(),
            order: SinkhornOrder::RowSoftmaxThenColumnFirst,
            placement: HyperConnectionPlacement::PerSublayer,
            fn_dtype: DType::Bf16,
        }
    }

    #[test]
    fn served_and_oracle_only_mixers_validate() {
        let served = mix(4096, 1e-6);
        served.validate().unwrap();
        assert_eq!(served.mixes(), 24);
        assert_eq!(served.flat_width(), 16_384);
        let wide = SinkhornMix {
            fn_dtype: DType::F32,
            ..mix(5120, 1e-20)
        };
        wide.validate().unwrap();
        for mix in [served, wide] {
            SinkhornHyperConnectionQuery::MixCollapse {
                mix,
                capacity: 8192,
            }
            .numerical_contract_id()
            .unwrap();
        }
        SinkhornHyperConnectionQuery::Expand {
            streams: 4,
            width: 4096,
            capacity: 8192,
        }
        .validate()
        .unwrap();
        SinkhornHyperConnectionQuery::Head {
            streams: 4,
            width: 4096,
            mode: StreamHeadMode::MeanOfStreams,
            capacity: 8192,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn mix_collapse_norm_checks_its_mixer_eps_and_capacity() {
        let fused = |mix, eps: f32, capacity| SinkhornHyperConnectionQuery::MixCollapseNorm {
            mix,
            rmsnorm_eps_bits: eps.to_bits(),
            capacity,
        };
        for width in [4096, 5120] {
            let id = fused(mix(width, 1e-6), 1e-5, 8192)
                .numerical_contract_id()
                .unwrap();
            assert_eq!(
                id,
                NumericalContractId::new(
                    "row.hc-sinkhorn-mix-collapse-rmsnorm-tree256.v1".to_owned()
                )
            );
        }
        assert!(fused(mix(4096, 1e-6), 0.0, 8).validate().is_err());
        assert!(fused(mix(4096, 1e-6), f32::NAN, 8).validate().is_err());
        assert!(fused(mix(4096, 1e-6), 1e-5, 0).validate().is_err());
        assert!(fused(mix(0, 1e-6), 1e-5, 8).validate().is_err());
    }

    #[test]
    fn the_fused_norm_order_is_part_of_the_contract() {
        let fused = |norm| SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
            ranks: 4,
            dtype: DType::F32,
            mix: mix(4096, 1e-6),
            rmsnorm_eps_bits: 1e-5_f32.to_bits(),
            norm,
            capacity: 256,
        };
        let id = |norm| fused(norm).numerical_contract_id().unwrap();
        assert_eq!(
            id(RmsNormReduction::Ordered),
            NumericalContractId::new("comm.all-reduce-hc-sinkhorn-expand-mix-rmsnorm.v1".into())
        );
        assert_eq!(
            id(RmsNormReduction::WarpTree { threads: 256 }),
            NumericalContractId::new(
                "comm.all-reduce-hc-sinkhorn-expand-mix-rmsnorm-tree256.v1".into()
            )
        );
        assert_ne!(
            fused(RmsNormReduction::Ordered),
            fused(RmsNormReduction::WarpTree { threads: 256 })
        );
        // A tree of other lanes names no order.
        for threads in [0, 32, 128, 512] {
            assert!(
                fused(RmsNormReduction::WarpTree { threads })
                    .validate()
                    .is_err()
            );
        }
    }

    #[test]
    fn fused_epilogue_is_rank_and_width_generic() {
        for ranks in 1..=8 {
            for width in [4096, 5120] {
                SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
                    ranks,
                    dtype: DType::F32,
                    mix: mix(width, 1e-6),
                    rmsnorm_eps_bits: 1e-5_f32.to_bits(),
                    norm: RmsNormReduction::Ordered,
                    capacity: 256,
                }
                .numerical_contract_id()
                .unwrap();
            }
        }
    }

    #[test]
    fn validation_checks_math_not_kernel_limits() {
        // Stream counts, widths, iteration counts and rank counts past any
        // one backend's kernels are valid math.
        for (streams, width, iterations) in [(4, 4128, 20), (1, 3, 1), (16, 32_768, 100)] {
            SinkhornMix {
                streams,
                iterations,
                ..mix(width, 1e-6)
            }
            .validate()
            .unwrap();
        }
        SinkhornHyperConnectionQuery::Expand {
            streams: 12,
            width: 4097,
            capacity: 1,
        }
        .validate()
        .unwrap();
        SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
            ranks: 16,
            dtype: DType::F32,
            mix: mix(4096, 1e-6),
            rmsnorm_eps_bits: 1e-5_f32.to_bits(),
            norm: RmsNormReduction::Ordered,
            capacity: 1,
        }
        .validate()
        .unwrap();
        // No streams, no width, no iterations, sizes past u32 and no ranks
        // are not.
        for (streams, width, iterations) in [(0, 4096, 20), (4, 0, 20), (4, 4096, 0)] {
            assert!(
                SinkhornMix {
                    streams,
                    iterations,
                    ..mix(width, 1e-6)
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SinkhornMix {
                streams: 1 << 16,
                ..mix(1 << 16, 1e-6)
            }
            .validate()
            .is_err()
        );
        // Expand and head have no mixer matrix and size their rows in u64,
        // so stream rows past u32 elements are valid.
        for (streams, width) in [(8, 1 << 29), (70_000, 1)] {
            SinkhornHyperConnectionQuery::Expand {
                streams,
                width,
                capacity: 1,
            }
            .validate()
            .unwrap();
            SinkhornHyperConnectionQuery::Head {
                streams,
                width,
                mode: StreamHeadMode::MeanOfStreams,
                capacity: 1,
            }
            .validate()
            .unwrap();
        }
        for (streams, width) in [(0, 4096), (4, 0)] {
            assert!(
                SinkhornHyperConnectionQuery::Expand {
                    streams,
                    width,
                    capacity: 1,
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
                ranks: 0,
                dtype: DType::F32,
                mix: mix(4096, 1e-6),
                rmsnorm_eps_bits: 1e-5_f32.to_bits(),
                norm: RmsNormReduction::Ordered,
                capacity: 1,
            }
            .validate()
            .is_err()
        );
        // The reduce sums F32 or BF16 partials, as `CommQuery::AllReduce`.
        for (dtype, valid) in [(DType::Bf16, true), (DType::Fp8E4M3, false)] {
            let query = SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
                ranks: 4,
                dtype,
                mix: mix(4096, 1e-6),
                rmsnorm_eps_bits: 1e-5_f32.to_bits(),
                norm: RmsNormReduction::Ordered,
                capacity: 1,
            };
            assert_eq!(query.validate().is_ok(), valid, "{dtype:?}");
        }
    }

    #[test]
    fn declared_variants_refuse_by_name() {
        let lagged = SinkhornMix {
            placement: HyperConnectionPlacement::LaggedSinglePass,
            ..mix(4096, 1e-6)
        };
        assert_eq!(
            lagged.validate(),
            Err(Refusal(
                "lagged single-pass hyper-connection placement is declared but not supported"
            ))
        );
        assert_eq!(
            SinkhornHyperConnectionQuery::Head {
                streams: 4,
                width: 4096,
                mode: StreamHeadMode::SigmoidWeighted,
                capacity: 1,
            }
            .validate(),
            Err(Refusal(
                "a sigmoid-weighted stream head is declared but not supported"
            ))
        );
        assert!(
            SinkhornMix {
                iterations: 0,
                ..mix(4096, 1e-6)
            }
            .validate()
            .is_err()
        );
    }
}
