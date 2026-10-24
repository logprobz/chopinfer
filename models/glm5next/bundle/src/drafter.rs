//! The MTP drafter's bookkeeping over the shared rules
//! (`bundle_runtime::drafter`): which drafter rows each step runs,
//! where their inputs come from, and what commit keeps.
//!
//! Drafter row `p` of a request reads the target's hidden state `h_p` and
//! the token at `p + 1`, sits at position `p`, and predicts token `p + 2`
//! (`eh_proj(cat[enorm(embed(t_{p+1})), hnorm(h_p)])`, vLLM's and SGLang's
//! GLM MTP). Its latent KV lives in the request's pages as a twelfth sparse
//! layer, and it reads only latent KV and indexer keys of earlier positions.
//! A drafting verify item's catch-up rows run in the extend pass, which runs
//! the drafter's indexer; each further draft is one chained row of a draft
//! step, which reuses the extend pass's selection
//! (`index_share_for_mtp_iteration`).

pub mod copies;

use model_glm5next_contract::contract::{
    DRAFT_ITEMS, MAX_VERIFY_ROWS, PAGE_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS, Shape,
};

/// Hidden rows of each slot the drafter has not consumed yet, by position
/// modulo this many: a verify item's accepted rows, or a decode item's rows
/// until the drafter catches up.
pub const RETAINED_ROWS: u32 = MAX_VERIFY_ROWS;
/// Rows one drafter run takes: a whole prefill chunk, or the retained rows
/// of every item.
pub const DRAFTER_ROWS: u64 = PREFILL_TOKENS as u64;

/// The token of a drafter row at position 0. vLLM's GLM MTP zeroes that
/// row's embedding before `enorm` (`fused_eh_norm`, `pos == 0`); an id in
/// no rank's vocabulary shard gathers zeros on every rank, so the
/// all-reduced embedding and its norm are zero.
pub const NO_TOKEN: u32 = u32::MAX;

/// The MTP drafter's constants.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Mtp;

impl bundle_runtime::drafter::Params for Mtp {
    const SLOTS: u32 = DRAFT_ITEMS;
    const MAX_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
    const RETAINED_ROWS: u32 = RETAINED_ROWS;
    const DRAFTER_ROWS: u64 = DRAFTER_ROWS;

    fn position_zero_token(_: u32) -> u32 {
        NO_TOKEN
    }
}

pub type Drafter = bundle_runtime::drafter::Drafter<Mtp>;
pub type Plan = bundle_runtime::drafter::Plan<Mtp>;
pub type Run = bundle_runtime::drafter::Run<Mtp>;

/// The drafter frontier a prefix snapshot taken at `anchor` inside a
/// prompt-ending item records: the indexer pool boundary before it. A
/// chunk that ended at the anchor would leave the frontier one row short
/// of it, with the drafter layer's incomplete pool in its indexer tail;
/// the step that keeps the anchor ran the drafter past it and kept no such
/// tail. From the pool boundary the continuation reads no tail: it runs the
/// pool's rows again from the retained rows the step copied into the
/// anchor's spare and its own tokens, and rewrites their drafter KV, in the
/// page it copies (`with_last_page_rewrites`), to the same bits. The
/// anchor is a page multiple, so the frontier stays in the snapshot's last
/// page and other requests may restore it.
pub fn anchor_frontier(shape: Shape, anchor: u64) -> u64 {
    anchor - u64::from(shape.profile.index_pool)
}

/// A full prefill chunk after a slot's retained rows fits the drafter: the
/// retained rows run before the target, and the chunk's rows minus one
/// after it.
const _: () =
    assert!(RETAINED_ROWS as u64 <= DRAFTER_ROWS && PREFILL_TOKENS as u64 - 1 <= DRAFTER_ROWS);

/// An anchor sits on a page boundary and a pool boundary, and the pool
/// before it fits the retained rows (`anchor_frontier`).
const _: () = assert!(
    PREFILL_GRANULE as u64 % PAGE_TOKENS == 0
        && Shape::TP4.profile.index_pool <= RETAINED_ROWS
        && PREFILL_GRANULE % Shape::TP4.profile.index_pool == 0
);

#[cfg(test)]
mod tests {
    use bundle_runtime::drafter::Params;

    use super::*;

    /// A request's first drafter row, at position 0, embeds no token (as
    /// vLLM's MTP zeroes its embedding).
    #[test]
    fn the_row_at_position_zero_embeds_no_token() {
        assert_eq!(Mtp::position_zero_token(11), NO_TOKEN);
    }

    /// An anchor snapshot's frontier is the pool boundary before the
    /// anchor, inside its last page and the retained rows.
    #[test]
    fn an_anchor_frontier_is_the_pool_before_it() {
        let shape = Shape::TP4;
        assert_eq!(anchor_frontier(shape, 3008), 3004);
        for anchor in [64, 3008, 8192] {
            let frontier = anchor_frontier(shape, anchor);
            assert_eq!(frontier % u64::from(shape.profile.index_pool), 0);
            assert!(frontier >= (anchor - 1) / PAGE_TOKENS * PAGE_TOKENS);
            assert!(anchor - frontier <= u64::from(RETAINED_ROWS));
        }
    }
}
