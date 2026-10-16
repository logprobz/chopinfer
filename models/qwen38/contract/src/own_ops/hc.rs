//! The fused hyper-connection up mix (`cuda/gemm/qwen38_hc.cuh` in the
//! bundle): the bottleneck of the down projection, its up projection to
//! every stream and the stream mix, in one launch per decode bucket.

use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Refusal, SigmoidMeanHyperConnectionQuery, Validate};
use kernel_core::hyper_connection_oracle::{bottleneck, mix};
use kernel_core::oracle::{GemmEpilogue, GemmShape, gemm};

use super::Bounded;
use crate::contract::Shape;

/// What the up mix reads beside its weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpMixInput {
    /// The rank-wide down projection; the op forms the bottleneck
    /// `silu(down / streams)` and rounds it to BF16.
    Down,
    /// The BF16 bottleneck an earlier launch formed.
    Bottleneck,
}

/// `out[r][c] = mean_s(sigmoid(up[r][s][c]) * normed[r][s][c])` with
/// `up = bottleneck x W_up^T` over `rank` columns, rounded to BF16, and the
/// bottleneck rounded to BF16: the unfused chain's rounding points.
/// `W_up` is `[streams * width][rank]`. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcUpMix {
    pub width: u32,
    pub streams: u32,
    pub rank: u32,
    pub input: UpMixInput,
}

impl HcUpMix {
    pub fn of(shape: Shape, input: UpMixInput) -> HcUpMix {
        HcUpMix {
            width: shape.hidden(),
            streams: shape.streams(),
            rank: shape.hc_rank(),
            input,
        }
    }

    fn wide(self) -> usize {
        self.width as usize * self.streams as usize
    }
}

impl Validate for HcUpMix {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 || self.streams == 0 || self.rank == 0 {
            return Err(Refusal("an up mix needs streams, width and a rank"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An up mix's buffers: the up weight in the layout the backend names
    /// for `weight` (`OpInfo::weight_layouts`), the down projection or
    /// bottleneck rows, the normalized X rows, the mixed output rows and the
    /// device `u32` count of live rows. Every view starts at 16-byte
    /// alignment.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct HcUpMixBuffers {
        pub weight: value,
        pub input: value,
        pub normed: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// Half a BF16 ulp of `value`, doubled: the most one rounding to BF16 of a
/// value near it moves it, with room for the value's own error.
fn rounding(value: f64) -> f64 {
    value.abs() * 2_f64.powi(-8) + 2_f64.powi(-133)
}

/// The F64 chain over `rows` rows of `input` (`[rows][rank]`), `weight`
/// (`[streams * width][rank]`, unpacked) and `normed`
/// (`[rows][streams][width]`). Each element's bound covers the two BF16
/// roundings inside the chain (the bottleneck's for [`UpMixInput::Down`],
/// the up projection's), carried through the sigmoid (at most 1/4
/// Lipschitz), the F32 accumulations (`2^-22 * K * S`) and the output's
/// rounding.
pub fn hc_up_mix(
    op: &HcUpMix,
    weight: &[u16],
    input: &[u16],
    normed: &[u16],
    rows: u32,
) -> Result<Bounded, Refusal> {
    op.validate()?;
    let (rank, wide, streams) = (op.rank as usize, op.wide(), op.streams as usize);
    let rows = rows as usize;
    let widen = |bits: &[u16], len: usize| -> Result<Vec<f64>, Refusal> {
        Ok(bits
            .get(..len)
            .ok_or(Refusal("an up mix buffer is too small"))?
            .iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect())
    };
    let input = widen(input, rows * rank)?;
    let weight = widen(weight, wide * rank)?;
    let normed = widen(normed, rows * wide)?;
    let (activation, activation_error) = match op.input {
        UpMixInput::Bottleneck => (input, vec![0.0; rows * rank]),
        UpMixInput::Down => {
            let formed = bottleneck(
                SigmoidMeanHyperConnectionQuery::Bottleneck {
                    streams: op.streams,
                    rank: op.rank,
                    capacity: rows as u32,
                },
                &input,
            )?;
            let error = formed.values.iter().map(|&value| rounding(value)).collect();
            (formed.values, error)
        }
    };
    let up = gemm(
        &activation,
        &weight,
        GemmShape {
            m: rows,
            n: wide,
            k: rank,
            experts: 1,
        },
        &[],
        GemmEpilogue {
            residual: None,
            swiglu: None,
        },
    )?;
    // The up projection's error: the bottleneck's carried through the
    // weights, the accumulation and the BF16 rounding.
    let mut up_error = vec![0.0; rows * wide];
    for row in 0..rows {
        for column in 0..wide {
            let index = row * wide + column;
            let carried = (0..rank)
                .map(|k| weight[column * rank + k].abs() * activation_error[row * rank + k])
                .sum::<f64>();
            up_error[index] = carried
                + kernel_core::oracle::accumulation_bound(rank, up.sum_abs_products[index])
                + rounding(up.values[index]);
        }
    }
    let mixed = mix(
        SigmoidMeanHyperConnectionQuery::Mix {
            streams: op.streams,
            width: op.width,
            capacity: rows as u32,
        },
        &up.values,
        &normed,
    )?;
    let width = op.width as usize;
    let bounds = (0..rows * width)
        .map(|index| {
            let (row, column) = (index / width, index % width);
            let carried = (0..streams)
                .map(|stream| {
                    let at = row * wide + stream * width + column;
                    // The sigmoid's own F32 error is a few ulp of at most 1.
                    (up_error[at] / 4.0 + 2_f64.powi(-20)) * normed[at].abs()
                })
                .sum::<f64>()
                / streams as f64;
            carried
                + kernel_core::oracle::accumulation_bound(streams, mixed.sum_abs_products[index])
                + rounding(mixed.values[index])
        })
        .collect();
    Ok(Bounded {
        values: mixed.values,
        bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernels' chain in F32 at their rounding points stays within the
    /// bounds of the F64 chain.
    #[test]
    fn the_rounded_f32_chain_is_within_the_bounds() {
        let op = HcUpMix {
            width: 4,
            streams: 2,
            rank: 3,
            input: UpMixInput::Down,
        };
        let bf = kernel_core::codec::f32_to_bf16;
        let f = |bits: u16| bf16_to_f32(bits);
        let weight = (0..8 * 3)
            .map(|v| bf(((v * 7) % 11) as f32 / 8.0 - 0.6))
            .collect::<Vec<_>>();
        let down = (0..2 * 3)
            .map(|v| bf(v as f32 / 3.0 - 0.9))
            .collect::<Vec<_>>();
        let normed = (0..2 * 8)
            .map(|v| bf(((v * 5) % 13) as f32 / 6.0 - 1.0))
            .collect::<Vec<_>>();
        let reference = hc_up_mix(&op, &weight, &down, &normed, 2).unwrap();
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let mut emulated = Vec::new();
        for row in 0..2 {
            let b = (0..3)
                .map(|k| {
                    let v = f(down[row * 3 + k]) / 2.0;
                    f(bf(v * sigmoid(v)))
                })
                .collect::<Vec<_>>();
            let up = (0..8)
                .map(|n| {
                    let sum = (0..3).fold(0_f32, |sum, k| f(weight[n * 3 + k]).mul_add(b[k], sum));
                    f(bf(sum))
                })
                .collect::<Vec<_>>();
            for column in 0..4 {
                let sum = (0..2).fold(0_f32, |sum, stream| {
                    let at = stream * 4 + column;
                    sum + sigmoid(up[at]) * f(normed[row * 8 + at])
                });
                emulated.push(bf(sum / 2.0));
            }
        }
        reference.check_bf16(&emulated).unwrap();
        // A wrong stream's mix is outside them.
        let mut wrong = emulated.clone();
        wrong[1] = bf(f(wrong[1]) + 0.25);
        assert!(reference.check_bf16(&wrong).is_err());
        // The precomputed form reads the bottleneck as it is.
        let formed = HcUpMix {
            input: UpMixInput::Bottleneck,
            ..op
        };
        let b = down
            .iter()
            .map(|&d| {
                let v = f(d) / 2.0;
                bf(v * sigmoid(v))
            })
            .collect::<Vec<_>>();
        hc_up_mix(&formed, &weight, &b, &normed, 2)
            .unwrap()
            .check_bf16(&emulated)
            .unwrap();
    }
}
