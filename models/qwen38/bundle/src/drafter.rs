//! The MTP drafter's bookkeeping over the shared rules
//! (`bundle_runtime::drafter`): which drafter rows each step runs,
//! where their inputs come from, and what commit keeps.
//!
//! Drafter row `p` of a request reads the target's final X of token `p` and
//! the token at `p + 1`, sits at position `p`, and predicts token `p + 2`
//! (vLLM's `Qwen4ExpMultiTokenPredictor`). Its K and V live in the
//! request's pages beside the target's, and it reads only K, V, block keys
//! and ring entries of earlier positions. A slot keeps the target X of its
//! tokens from the frontier on in `RETAINED_ROWS` retained rows; the hidden
//! row the shared rules gather is that X, and their gather is `mtp_gather`.
//! The row at position 0 embeds its token like every other row.

pub use bundle_runtime::drafter::{DraftError, DraftRow, Pending, Retain, RunItem, Source, Token};

use crate::contract::{DRAFT_ITEMS, MAX_VERIFY_ROWS, PREFILL_TOKENS};
use crate::scratch::{DRAFTER_ROWS, RETAINED_ROWS};

/// The MTP drafter's constants.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Mtp;

impl bundle_runtime::drafter::Params for Mtp {
    const SLOTS: u32 = DRAFT_ITEMS;
    const MAX_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
    const RETAINED_ROWS: u32 = RETAINED_ROWS;
    const DRAFTER_ROWS: u64 = DRAFTER_ROWS;
}

pub type Drafter = bundle_runtime::drafter::Drafter<Mtp>;
pub type Plan = bundle_runtime::drafter::Plan<Mtp>;
pub type Run = bundle_runtime::drafter::Run<Mtp>;

/// A full prefill chunk after a slot's retained rows fits the drafter: the
/// retained rows run before the target, and the chunk's rows minus one
/// after it.
const _: () =
    assert!(RETAINED_ROWS as u64 <= DRAFTER_ROWS && PREFILL_TOKENS as u64 - 1 <= DRAFTER_ROWS);
