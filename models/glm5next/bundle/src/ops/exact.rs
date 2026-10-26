//! The exactness GLM 5.3 Flash relies on, as each op asks it at create
//! (`Need::exact`): a backend that cannot keep an ask refuses the op, and
//! create with it, never a step.
//!
//! - Prefix exactness: a prompt row keeps its bits in every chunk split on
//!   absolute multiples of `PREFILL_GRANULE` (the scheduler ends chunks
//!   there, `contract::PREFILL_GRANULE`), and an op that writes per-slot
//!   state in prefill (the KDA convolution and delta rule, the indexer's
//!   key write) also leaves it at in-step anchors, which sit on those
//!   multiples (`ANCHOR_ITEMS` per step). Paged latent KV is captured by
//!   page, so its write asks no anchors.
//! - AR equals MTP: a verify row has the bits of its row decoded alone, at
//!   every verify step the instance serves (`verify_exact`, `determinism`).
//!   A verify op asks one-row bits up to the rows it serves; decode ops ask
//!   nothing, as the one-row reference they are.
//!
//! Ops that serve every phase ask both. Ops whose declarations claim less
//! than GLM relies on ask nothing; GLM-F's HANDOFF lists them. The drafter's
//! own ops only choose drafts, which the target's verify accepts or
//! rejects, so they ask nothing.

use bundle_runtime::exact;
pub(crate) use bundle_runtime::exact::{Kind, need, verify};
use kernel_core::properties::{Anchors, Chunk, Invariance};
use model_glm5next_contract::contract::PREFILL_GRANULE;

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

/// `exact` with the anchors a prefill op that writes per-slot state needs:
/// its state at anchors on the granule's multiples. Other phases' needs
/// keep none.
pub(crate) fn with_state(exact: Invariance) -> Invariance {
    match exact.chunk {
        Chunk::None => exact,
        _ => exact.with_anchors(Anchors::Aligned {
            granule: PREFILL_GRANULE,
        }),
    }
}

/// What an op serving every phase needs: [`PREFILL`] and the one-row bits
/// of verify launches of up to `rows` rows.
pub(crate) const fn every_phase(rows: u32) -> Invariance {
    exact::every_phase(PREFILL, rows)
}

/// What an op of `phase`'s kind needs, prepared for `capacity` rows.
pub(crate) fn of_phase(phase: impl Into<Kind>, capacity: u32) -> Invariance {
    exact::of_phase(PREFILL, phase, capacity)
}

/// The most rows a verify launch carries: the widest decode or verify step
/// (`buffers::widest_step_rows`).
pub(crate) fn verify_rows() -> u32 {
    crate::buffers::widest_step_rows()
}
