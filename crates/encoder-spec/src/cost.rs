//! Cost model: an estimate of each stage's arithmetic and memory traffic in
//! one run, for scheduling. It counts the declared math, not the kernels:
//! - FLOPs are two per multiply-add of every GEMM (convolutions as their
//!   im2col GEMM), four per query-key pair a mask lets attention see (scores
//!   and the weighted sum over values, `head_dim` wide per head), and
//!   `NORM_FLOPS` per value of norms and position taps; residual adds,
//!   rotary and casts are not counted;
//! - bytes are each stage's input and output rows read or written once (a
//!   transformer block's intermediates written and read once), plus its
//!   weights read once per run, at the elements the declaration states.
//!
//! A model's registered variant states its own estimate (`Component::cost`).
//!
//! So attention grows with the square of its sequence lengths (less under a
//! window), everything else linearly in rows. The `f32-pairs.v1` policy runs
//! several tensor-core products per declared one (two per GEMM product,
//! three per score, four to six per weighted value under
//! `attention.varlen.pairs.v1`); the estimate does not scale for them, and
//! device constants (rates per variant) are measured, not modeled here.

use crate::components::attention::Mask;
use crate::components::{Boundary, Conv1d, Merge, StageOp, Transformer};
use crate::geometry::{Geometry, Run};
use crate::{Declaration, Error};

/// One stage's estimated work in one run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cost {
    pub flops: u64,
    pub bytes: u64,
}

impl std::ops::Add for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost {
            flops: self.flops + other.flops,
            bytes: self.bytes + other.bytes,
        }
    }
}

/// `sum over i < n of min(n - 1, i + right)`.
fn sum_min(n: u64, right: u64) -> u64 {
    let below = n.saturating_sub(right);
    below * below.saturating_sub(1) / 2 + below * right + (n - below) * (n - 1)
}

/// `sum over i < n of max(0, i - left)`.
fn sum_max(n: u64, left: u64) -> u64 {
    let over = n.saturating_sub(left + 1);
    over * (over + 1) / 2
}

/// The query-key pairs `mask` lets a sequence of `n` rows see.
pub fn visible_pairs(mask: Mask, n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    match mask {
        Mask::Full => n * n,
        Mask::Causal => n * (n + 1) / 2,
        Mask::Window([left, right]) => {
            sum_min(n, u64::from(right)) + n - sum_max(n, u64::from(left))
        }
        Mask::CausalWindow(left) => n * (n + 1) / 2 - sum_max(n, u64::from(left)),
    }
}

/// Per-value FLOPs of a norm (mean, variance, scale, shift).
const NORM_FLOPS: u64 = 8;

/// One transformer stage over `rows` rows in `sequences`, weights aside.
fn transformer(transformer: &Transformer, rows: u64, sequences: &[u64], element: u64) -> Cost {
    let p = transformer.params();
    let a = p.attention.params();
    let width = u64::from(p.width);
    let qkv = p.attention.qkv_width().unwrap_or(0);
    let heads = p.attention.attention_width().unwrap_or(0);
    let down = p.mlp.inner();
    let up = if p.mlp.gated() { 2 * down } else { down };
    // The per-head q and k norms' values, if the attention has them.
    let qk_norms = match p.attention.qk_norm_eps() {
        Ok(Some(_)) => (u64::from(a.q_heads) + u64::from(a.kv_heads)) * u64::from(a.head_dim),
        _ => 0,
    };
    let lengths: Vec<u64> = sequences.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let mut cost = Cost::default();
    for block in 0..p.blocks {
        let (mask, _) = transformer.attention_at(block);
        let pairs: u64 = lengths.iter().map(|&n| visible_pairs(mask, n)).sum();
        let gemms = rows * (width * qkv + heads * width + width * up + down * width);
        cost.flops += 2 * gemms
            + 4 * u64::from(a.q_heads) * u64::from(a.head_dim) * pairs
            + 2 * NORM_FLOPS * rows * width
            + NORM_FLOPS * rows * qk_norms;
        // Each intermediate (two norm outputs, qkv, attention output, the
        // MLP's inner rows) written and read once; the residual stream read
        // by two norms and read and written by two adds.
        let rows_moved = rows * (2 * (2 * width + qkv + heads + down) + 6 * width);
        cost.bytes += rows_moved * element;
    }
    cost
}

impl Declaration {
    /// The estimated cost of each stage of `run` (a run of this
    /// declaration's `geometry`), in the run's stage order.
    pub fn cost(&self, run: &Run) -> Result<Vec<Cost>, Error> {
        let steps = self.steps()?;
        let context = self.context();
        let mut out = Vec::with_capacity(run.steps.len());
        for step in &run.steps {
            let at = &steps[step.stage];
            let fail = |error: String| Error(format!("{}: {error}", at.at));
            let (input, output): (&Boundary, &Boundary) = (&at.input, &steps[step.stage + 1].input);
            let [rows, written] = step.rows;
            let moved = rows * input.data.row_bytes() + written * output.data.row_bytes();
            let weights: u64 = at
                .component
                .roles(&context, input)
                .map_err(fail)?
                .iter()
                .map(|role| {
                    let blocks = match (&role.blocks, at.component.blocks()) {
                        (Some(blocks), _) => blocks.len() as u64,
                        (None, Some(blocks)) => u64::from(blocks),
                        (None, None) => 1,
                    };
                    role.shape.iter().product::<u64>() * role.element.bytes() * blocks
                })
                .sum();
            let gemm = |k: u64, n: u64| 2 * written * k * n;
            let cost = match &self.stages[step.stage].op {
                StageOp::Linear(linear) => Cost {
                    flops: gemm(
                        u64::from(input.width().map_err(fail)?),
                        u64::from(linear.params().out),
                    ),
                    bytes: moved + weights,
                },
                StageOp::Conv1d(Conv1d::Im2col(params)) => {
                    let taps = u64::from(params.kernel) * u64::from(input.width().map_err(fail)?);
                    Cost {
                        flops: gemm(taps, u64::from(params.out)),
                        bytes: moved + weights + written * taps * context.rows_element().bytes(),
                    }
                }
                StageOp::Norm(_) | StageOp::Position(_) => Cost {
                    flops: NORM_FLOPS * rows * u64::from(input.width().map_err(fail)?),
                    bytes: moved + weights,
                },
                StageOp::Transformer(t) => {
                    let sequences = step.sequences.as_deref().unwrap_or(&[]);
                    let element = context.rows_element().bytes();
                    let cost = transformer(t, rows, sequences, element);
                    Cost {
                        flops: cost.flops,
                        bytes: cost.bytes + weights,
                    }
                }
                StageOp::Projector(projector) => {
                    let p = projector.params();
                    let width = u64::from(input.width().map_err(fail)?);
                    let inner = p.mlp.inner();
                    let up = if p.mlp.gated() { 2 * inner } else { inner };
                    // The norm's rows, the up projection's and the hidden
                    // rows written and read once each.
                    let element = context.rows_element().bytes();
                    Cost {
                        flops: NORM_FLOPS * rows * width + gemm(width, up) + gemm(inner, width),
                        bytes: moved + weights + 2 * rows * (width + up + inner) * element,
                    }
                }
                StageOp::Merge(Merge::Consecutive(_)) => Cost::default(),
                StageOp::Registered(_) => {
                    let own = at.component.cost(step.rows, input, output).map_err(fail)?;
                    let own = own.unwrap_or(Cost {
                        flops: 0,
                        bytes: moved,
                    });
                    Cost {
                        flops: own.flops,
                        bytes: own.bytes + weights,
                    }
                }
                _ => Cost {
                    flops: 0,
                    bytes: moved + weights,
                },
            };
            out.push(cost);
        }
        Ok(out)
    }

    /// The estimated cost of every run of `geometry`, summed: what one item
    /// costs to encode.
    pub fn item_cost(&self, geometry: &Geometry) -> Result<Cost, Error> {
        let mut total = Cost::default();
        for run in &geometry.runs {
            for cost in self.cost(run)? {
                total = total + cost;
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_pairs_count_each_mask() {
        for n in 0..40 {
            let brute = |sees: &dyn Fn(i64, i64) -> bool| {
                let n = n as i64;
                (0..n)
                    .flat_map(|q| (0..n).map(move |k| (q, k)))
                    .filter(|&(q, k)| sees(q, k))
                    .count() as u64
            };
            assert_eq!(visible_pairs(Mask::Full, n), brute(&|_, _| true));
            assert_eq!(visible_pairs(Mask::Causal, n), brute(&|q, k| k <= q));
            for (left, right) in [(0, 3), (2, 2), (5, 0), (64, 64)] {
                assert_eq!(
                    visible_pairs(Mask::Window([left, right]), n),
                    brute(&|q, k| k >= q - i64::from(left) && k <= q + i64::from(right)),
                    "window {left} {right} over {n}"
                );
            }
            for left in [1, 3, 128] {
                assert_eq!(
                    visible_pairs(Mask::CausalWindow(left), n),
                    brute(&|q, k| k <= q && k >= q - i64::from(left)),
                    "causal window {left} over {n}"
                );
            }
        }
    }
}
