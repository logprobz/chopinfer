//! The exactness DeepSeek V4.1 relies on, as each op asks it at create
//! (`Need::exact`): a backend that cannot keep an ask refuses the op, and
//! create with it, never a step.
//!
//! - Prefix exactness: a prompt row keeps its bits in every chunk split on
//!   absolute multiples of `PREFILL_GRANULE` (whole pages, so no split
//!   cuts a compression group or a candidate block).
//! - AR equals speculation: a verify row has the bits of its row decoded
//!   alone, at every verify step the instance serves (`verify_exact`). A
//!   verify op asks one-row bits up to the rows it serves; decode ops ask
//!   nothing, as the one-row reference they are.
//!
//! Ops that serve every phase ask both.

use kernel_core::backend::Need;
use kernel_core::properties::{Chunk, Invariance, Rows};
use model_deepseek41_contract::serving::PREFILL_GRANULE;

use crate::program::Phase;

/// The kind of phase an op is prepared for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Prefill,
    Decode,
    Verify,
}

impl From<Phase> for Kind {
    fn from(phase: Phase) -> Kind {
        match phase {
            Phase::Prefill { .. } => Kind::Prefill,
            Phase::Decode(_) => Kind::Decode,
            Phase::Verify { .. } => Kind::Verify,
        }
    }
}

/// What a prefill op needs: its rows kept over chunk splits on the
/// granule.
pub(crate) const PREFILL: Invariance = Invariance {
    chunk: Chunk::Granule(PREFILL_GRANULE),
    ..Invariance::NONE
};

/// What a verify op of up to `rows` rows needs: each row's one-row bits.
pub(crate) const fn verify(rows: u32) -> Invariance {
    Invariance {
        rows: Rows::OneRow { max: rows },
        ..Invariance::NONE
    }
}

/// What an op serving every phase needs: [`PREFILL`] and the one-row bits
/// of verify launches of up to `rows` rows.
pub(crate) const fn every_phase(rows: u32) -> Invariance {
    Invariance {
        rows: Rows::OneRow { max: rows },
        ..PREFILL
    }
}

/// What an op of `phase`'s kind needs, prepared for `capacity` rows.
#[allow(dead_code, reason = "the per-phase families ask it")]
pub(crate) fn of_phase(phase: impl Into<Kind>, capacity: u32) -> Invariance {
    match phase.into() {
        Kind::Prefill => PREFILL,
        Kind::Decode => Invariance::NONE,
        Kind::Verify => verify(capacity),
    }
}

/// `Need::rows(capacity)` with `exact`.
pub(crate) fn need(capacity: u32, exact: Invariance) -> Need {
    Need::rows(capacity).with_exact(exact)
}
