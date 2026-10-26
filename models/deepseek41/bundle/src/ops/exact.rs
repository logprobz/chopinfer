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

use bundle_runtime::exact;
pub(crate) use bundle_runtime::exact::{Kind, need, verify};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::serving::PREFILL_GRANULE;

use crate::program::Phase;

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
pub(crate) const PREFILL: Invariance = exact::prefill(PREFILL_GRANULE);

/// What an op serving every phase needs: [`PREFILL`] and the one-row bits
/// of verify launches of up to `rows` rows.
pub(crate) const fn every_phase(rows: u32) -> Invariance {
    exact::every_phase(PREFILL, rows)
}

/// What an op of `phase`'s kind needs, prepared for `capacity` rows.
pub(crate) fn of_phase(phase: impl Into<Kind>, capacity: u32) -> Invariance {
    exact::of_phase(PREFILL, phase, capacity)
}
