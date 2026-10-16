//! Precision and parallel policies.

use serde::{Deserialize, Serialize};

/// Every rounding point of every component. Policies are versioned like
/// components: a policy's rules never change once a committed declaration
/// selects it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Precision {
    /// The default for every new declaration: F32 payload, F32 rows and
    /// residual stream between ops, activation operands of tensor-core
    /// products as BF16 hi+lo pairs, two-pass F32 norms, host tables in the
    /// reference's F32 math, and one BF16 rounding at the rows the language
    /// model reads.
    #[serde(rename = "f32-pairs.v1")]
    F32Pairs,
    /// Retained only to report a clear refusal for retired declarations.
    #[serde(rename = "bf16-legacy.v1")]
    Bf16Legacy,
}

pub const ANY: &[Precision] = &[Precision::F32Pairs];
pub const PAIRS: &[Precision] = &[Precision::F32Pairs];
pub const LEGACY: &[Precision] = &[];

impl Precision {
    pub fn validate(self) -> Result<(), String> {
        match self {
            Self::F32Pairs => Ok(()),
            Self::Bf16Legacy => Err("precision bf16-legacy.v1 is retired; use f32-pairs.v1".into()),
        }
    }
}

/// How the ranks share one step's encoder work.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParallelPolicy {
    /// Every rank runs every record. The default, and the reference the
    /// other policies must equal bit for bit.
    #[serde(rename = "replicated")]
    Replicated,
    /// Ranks take disjoint records and copy their merged rows to the others.
    #[serde(rename = "record-parallel")]
    RecordParallel,
    /// One record's rows split across ranks, with K and V gathered per
    /// attention block.
    #[serde(rename = "sequence-split")]
    SequenceSplit,
}

/// What a component needs of the rows around the ones it writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelClass {
    /// Each output row reads only its own input row.
    RowLocal,
    /// Each output row reads a few neighbouring input rows (convolutions).
    RowHalo,
    /// Each output row reads its whole attention sequence.
    SequenceCoupled,
    /// Each output row reads one whole merge group.
    GroupCoupled,
    /// Rows move across the sequence (reorders).
    Permuting,
}
