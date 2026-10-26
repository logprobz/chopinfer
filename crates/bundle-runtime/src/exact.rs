//! Exactness requests with model-owned prefill granules and row capacities.

use kernel_core::backend::Need;
use kernel_core::contract::{AttentionPhase, RecurrentPhase};
use kernel_core::precision::ExecutionPhase;
use kernel_core::properties::{Chunk, Invariance, Rows};

/// The kind of phase an op is prepared for, however its family names it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Prefill,
    Decode,
    Verify,
}

impl From<AttentionPhase> for Kind {
    fn from(phase: AttentionPhase) -> Kind {
        match phase {
            AttentionPhase::Prefill => Kind::Prefill,
            AttentionPhase::Decode => Kind::Decode,
            AttentionPhase::Verify => Kind::Verify,
        }
    }
}

impl From<RecurrentPhase> for Kind {
    fn from(phase: RecurrentPhase) -> Kind {
        match phase {
            RecurrentPhase::Prefill => Kind::Prefill,
            RecurrentPhase::Decode => Kind::Decode,
            RecurrentPhase::Verify { .. } => Kind::Verify,
        }
    }
}

impl From<ExecutionPhase> for Kind {
    fn from(phase: ExecutionPhase) -> Kind {
        match phase {
            ExecutionPhase::Prefill => Kind::Prefill,
            ExecutionPhase::Decode => Kind::Decode,
            ExecutionPhase::Verify => Kind::Verify,
        }
    }
}

/// Rows keep their bits across splits on the model's prefill granule.
pub const fn prefill(granule: u32) -> Invariance {
    Invariance {
        chunk: Chunk::Granule(granule),
        ..Invariance::NONE
    }
}

/// Each verify row has its one-row bits through `rows` rows.
pub const fn verify(rows: u32) -> Invariance {
    Invariance {
        rows: Rows::OneRow { max: rows },
        ..Invariance::NONE
    }
}

/// Prefill chunk invariance and the one-row bits of verify launches.
pub const fn every_phase(prefill: Invariance, rows: u32) -> Invariance {
    Invariance {
        rows: Rows::OneRow { max: rows },
        ..prefill
    }
}

/// Exactness for a phase prepared for `capacity` rows.
pub fn of_phase(prefill: Invariance, phase: impl Into<Kind>, capacity: u32) -> Invariance {
    match phase.into() {
        Kind::Prefill => prefill,
        Kind::Decode => Invariance::NONE,
        Kind::Verify => verify(capacity),
    }
}

/// A row-capacity request with the stated exactness.
pub fn need(capacity: u32, exact: Invariance) -> Need {
    Need::rows(capacity).with_exact(exact)
}
