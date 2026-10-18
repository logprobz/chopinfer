//! Compact candidate storage for the Position v3 indexer.
//!
//! Block IDs replace the dense mask without changing score arithmetic,
//! block maxima, newest-block pinning or ties on original positions. A row
//! stores sorted unique nonnegative IDs, followed by -1 padding. Slot s
//! addresses `ids[s / block_size] * block_size + s % block_size`.
//! Visibility and page holes still exclude positions before operand reads.
//!
//! The producer copies a caller-owned U64 row tag and the visible compressed
//! count alongside each list. Tags identify a request incarnation and token
//! row: they must differ whenever retained storage is reused for another
//! row, including another chunk. All Reindex layers use that same tag. A
//! consumer compares the stored tag and visibility against its current row.
//! Tags are opaque equality tokens, not pointers or counters used by CUDA.
//!
//! Each launch overwrites status: 0 is success, 1 rejects the row and emits
//! an all--1 selection. Failure includes Position v3 non-finite arithmetic,
//! a failed source row, stale provenance, out-of-range, unsorted or repeated
//! IDs, nonterminal padding, and absence of the newest visible block. Empty
//! visibility requires an empty list. An invalid list never authorizes an
//! out-of-range operand read. The caller rejects any failed step before
//! accepting its results. The original dense Position v3 ABI is unchanged.

use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};
use crate::pooled_indexer::{
    CandidateBlocksQuery, IndexerCandidates, PositionIndexerQuery, PositionKeyPages,
};

/// Position v3 with a compact block list instead of a dense candidate mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PositionBlockIndexerQuery {
    /// All-mode geometry; the block list supplies the candidate restriction.
    pub indexer: PositionIndexerQuery,
    pub block_size: u32,
    pub top_k_blocks: u32,
}

impl PositionBlockIndexerQuery {
    pub fn filter(&self) -> CandidateBlocksQuery {
        CandidateBlocksQuery {
            block_size: self.block_size,
            top_k_blocks: self.top_k_blocks,
            capacity: self.indexer.capacity,
        }
    }

    pub fn score_slots(&self) -> Result<u32, Refusal> {
        self.block_size
            .checked_mul(self.top_k_blocks)
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or(Refusal("compact position score width overflows"))
    }
}

impl Validate for PositionBlockIndexerQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.indexer.validate()?;
        self.filter().validate()?;
        self.score_slots()?;
        if self.indexer.candidates != IndexerCandidates::All {
            return Err(Refusal(
                "compact position indexer cannot also bind a dense mask",
            ));
        }
        Ok(())
    }
}

impl NumericalContract for PositionBlockIndexerQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(
            "sparse.position-indexer-mxfp4-block-ids-f32-topk-lower-index.v3".to_owned(),
        ))
    }
}

crate::scratch_record! {
    /// Retained candidate record. Per row: I32 IDs `[top_k_blocks]`, U64 tag,
    /// I32 visible count and I32 source status. The scalar width is the
    /// original position capacity, never the compact score-slot count.
    /// Every plane is contiguous and independently sliceable by row.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CandidateBlockIds {
        pub ids: value,
        pub row_tags: value,
        pub visible_positions: value,
        pub status: value,
        pub position_capacity: scalar(u32),
    }
}

crate::scratch_record! {
    /// Produce IDs from the same own-F32 scores as CandidateBlocksBuffers.
    /// Input: F32 scores `[rows][position_capacity]`, I32 visible counts,
    /// U64 current row tags, I32 source status, and U32 device live count.
    /// Write every retained plane for each live row. A failed source emits
    /// an empty list with status 1. NaN or +inf in a visible input score,
    /// or a visible count outside 0..=position_capacity, also sets status 1
    /// and writes empty IDs. Future scores are not read. No data error traps.
    /// Workspace is one F32 block max per
    /// physical block and row. No dense candidate mask is materialized.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CandidateBlockIdsBuffers {
        pub scores: value,
        pub visible_positions: value,
        pub row_tags: value,
        pub source_status: value,
        pub live_rows: value,
        pub output: record(CandidateBlockIds),
    }
}

crate::scratch_record! {
    /// Direct score/select. q, weights, key pages, positions and row_batch
    /// have the dense Position v3 layouts. `row_tags` is current U64 `[rows]`,
    /// stored separately from the retained tag plane.
    /// F32 scores are `[rows][top_k_blocks * block_size]`; -inf marks padding,
    /// future positions and page holes. I32 status is `[rows]`, and output
    /// is I32 `[rows][max_top_k]`, containing original compressed positions
    /// ascending then -1. The retained record is read-only across Reindex
    /// layers; status and output belong to this invocation. No workspace.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PositionBlockSelectBuffers {
        pub q: value,
        pub head_weights: value,
        pub keys: record(PositionKeyPages),
        pub positions: value,
        pub row_batch: value,
        pub row_tags: value,
        pub live_rows: value,
        pub candidates: record(CandidateBlockIds),
        pub scores: value,
        pub status: value,
        pub output: value,
    }
}
