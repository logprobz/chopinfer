//! DSpark's Markov correction and diagnostic confidence projection.
//!
//! The checkpoint stores all three head operands as BF16: the Markov
//! embedding, its vocabulary projection and the confidence weight. The
//! reference promotes them to F32 for projection and keeps F32 outputs.
//! BF16 values are decoded exactly; no activation quantization or TF32 is
//! permitted. Collapsed hidden activations remain F32 through confidence;
//! that path has no trained quantizer. F32 reduction chains contain at most
//! 128 products, summed in increasing K order, before adding the Markov
//! correction to base logits.

use kernel_core::backend::Need;
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::Refusal;
use kernel_core::oracle::{OracleOutput, compensated_sum_with_magnitude};

pub const MARKOV_CONTRACT: &str = "deepseek41.markov-step.bf16.acc-f32-chain128.add-f32.v1";
pub const CONFIDENCE_CONTRACT: &str =
    "deepseek41.confidence.hidden-f32.markov-bf16.acc-f32-chain128.out-f32.v1";

/// One position of every live draft block, on one vocabulary shard:
/// `output[r,v] = base[r,v] + sum_k embedding[r,k] * weight[v,k]`.
///
/// The embedding is the previously sampled token's Markov embedding,
/// already gathered across vocabulary shards. The projection completes
/// before the F32 add, matching `logits[:, position].add_(logits_bias)`.
/// Sampling and its RNG are separate; the next Markov step consumes that
/// sample. All five positions run even when fewer proposals are emitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MarkovStep {
    pub rank: u32,
    pub vocab_rows: u32,
    /// F32 elements between block items in base/output, at least vocab_rows.
    /// A view of one position in `[item][5][vocab]` has stride `5 * vocab`.
    pub logits_stride: u32,
}

impl MarkovStep {
    pub fn validate(&self, need: Need) -> Result<(), Refusal> {
        if self.rank == 0
            || self.vocab_rows == 0
            || self.logits_stride < self.vocab_rows
            || need.capacity == 0
        {
            return Err(Refusal("invalid DSpark Markov dimensions or row stride"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// BF16 embedding `[rows, rank]` and weight `[vocab_rows, rank]`;
    /// F32 base and output rows spaced by `logits_stride` elements. Output
    /// may alias base exactly; partial overlaps and aliases with other roles
    /// are forbidden. Rows beyond the device U32 live_rows and gaps between
    /// vocabulary shards' rows remain untouched. Scalar elements are aligned
    /// to their size. Views begin at the selected draft position.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MarkovStepBuffers {
        pub embedding: value,
        pub weight: value,
        pub base: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// One confidence score per live block item:
/// `sum_i hidden[r,i] * weight[i] + sum_k markov[r,k] * weight[hidden+k]`.
///
/// `hidden` is the final stage's collapsed row before its vocabulary-head
/// RMSNorm, as in the reference. `markov` is the predecessor embedding used
/// for this position's correction, not the newly sampled token's embedding.
/// The reduction traverses the concatenated operands without an extra
/// rounding at their boundary. Scores are diagnostic only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfidenceHead {
    pub hidden: u32,
    pub markov_rank: u32,
    pub hidden_stride: u32,
    pub score_stride: u32,
}

impl ConfidenceHead {
    pub fn validate(&self, need: Need) -> Result<(), Refusal> {
        if self.hidden == 0
            || self.markov_rank == 0
            || self.hidden_stride < self.hidden
            || self.score_stride == 0
            || self.hidden.checked_add(self.markov_rank).is_none()
            || need.capacity == 0
        {
            return Err(Refusal(
                "invalid DSpark confidence dimensions or row strides",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// F32 hidden rows spaced by hidden_stride, packed BF16 predecessor
    /// embeddings `[rows, markov_rank]`, BF16 weight `[hidden+markov_rank]`,
    /// F32 scores spaced by score_stride and the device U32 live_rows.
    /// No output aliases an input. Rows beyond live_rows and stride gaps
    /// remain untouched. Each view begins at the selected draft position.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ConfidenceHeadBuffers {
        pub hidden: value,
        pub markov: value,
        pub weight: value,
        pub scores: value,
        pub live_rows: value,
    }
}

fn finite_bf16(values: &[u16]) -> Result<(), Refusal> {
    if values.iter().any(|&value| !bf16_to_f32(value).is_finite()) {
        return Err(Refusal("DSpark head operands must be finite BF16"));
    }
    Ok(())
}

/// F64 oracle of one item's local vocabulary logits. The bound is the
/// shared F32 accumulation bound with K=rank+1 and S including the base
/// logit. The extra term covers the separate final F32 add.
pub fn markov_reference(
    op: &MarkovStep,
    embedding: &[u16],
    weight: &[u16],
    base: &[f32],
) -> Result<OracleOutput, Refusal> {
    op.validate(Need::rows(1))?;
    let (rank, vocab) = (op.rank as usize, op.vocab_rows as usize);
    if embedding.len() != rank
        || weight.len() != rank * vocab
        || base.len() != vocab
        || base.iter().any(|value| !value.is_finite())
    {
        return Err(Refusal("DSpark Markov oracle operand shape or value"));
    }
    finite_bf16(embedding)?;
    finite_bf16(weight)?;
    let mut output = OracleOutput {
        values: Vec::with_capacity(vocab),
        sum_abs_products: Vec::with_capacity(vocab),
        reduction_lengths: vec![rank + 1; vocab],
    };
    for (&base, weights) in base.iter().zip(weight.chunks_exact(rank)) {
        let (sum, magnitude) = compensated_sum_with_magnitude(
            std::iter::once(f64::from(base)).chain(
                embedding
                    .iter()
                    .zip(weights)
                    .map(|(&x, &w)| f64::from(bf16_to_f32(x)) * f64::from(bf16_to_f32(w))),
            ),
        );
        output.values.push(sum);
        output.sum_abs_products.push(magnitude);
    }
    Ok(output)
}

/// F64 oracle of one confidence score over F32 hidden values concatenated
/// with exactly decoded BF16 Markov values, weighted by decoded BF16.
/// K=hidden+markov_rank; there is no activation or output storage rounding.
pub fn confidence_reference(
    op: &ConfidenceHead,
    hidden: &[f32],
    markov: &[u16],
    weight: &[u16],
) -> Result<OracleOutput, Refusal> {
    op.validate(Need::rows(1))?;
    let width = (op.hidden + op.markov_rank) as usize;
    if hidden.len() != op.hidden as usize
        || markov.len() != op.markov_rank as usize
        || weight.len() != width
    {
        return Err(Refusal("DSpark confidence oracle operand shape"));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(Refusal(
            "DSpark confidence hidden values must be finite F32",
        ));
    }
    finite_bf16(markov)?;
    finite_bf16(weight)?;
    let (sum, magnitude) = compensated_sum_with_magnitude(
        hidden
            .iter()
            .copied()
            .chain(markov.iter().copied().map(bf16_to_f32))
            .zip(weight)
            .map(|(x, &w)| f64::from(x) * f64::from(bf16_to_f32(w))),
    );
    Ok(OracleOutput {
        values: vec![sum],
        sum_abs_products: vec![magnitude],
        reduction_lengths: vec![width],
    })
}

#[cfg(test)]
mod tests;
