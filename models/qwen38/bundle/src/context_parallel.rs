//! Context-parallel groups of 2 ranks at TP4: each rank's buffers and
//! peer-visible comm state, and the switch that keeps create refusing a
//! size above 1 until the lowering passes its GPU check.
//!
//! Ranks {0, 1} hold KV head 0 and ranks {2, 3} KV head 1, so the groups of
//! two consecutive ranks are the replica sets, which at size 1 store the
//! same K, V and block keys twice. A group splits each sequence's KV pages
//! round-robin (`runtime_kv::ContextParallelPagesV1`): physical page j lives
//! on group rank `j % 2` at the local page `table[j / 2]`, so the step's
//! block table is every rank's own table of local pages. Heads stay split
//! over the ranks; the raw-key ring, linear and PLE state stay per slot on
//! every rank. Per full-attention layer (`lower/context_parallel.rs`):
//!
//! 1. each appended position is localized to its owner
//!    (`localize_positions2`) and only the owner writes the row's K and V;
//!    every rank writes its Q rows;
//! 2. the indexer key write takes the group rank and size: both ranks write
//!    the ring, and only a block's owner stores its key;
//! 3. each rank selects over its own block keys (`select_per_rank2`), the
//!    group all-gathers the (score, block) pairs and every rank merges them
//!    into the ascending block list QSA's selection writes
//!    (`topk_merge_blocks2`);
//! 4. the group all-gathers the queries `[2][rows][6][256]`, and each rank
//!    runs the selected partials of both sources' heads over its own pages,
//!    writing destination-major blocks;
//! 5. one all-gather per owner o, targeting o
//!    (`group::owner_gather_arguments`), moves each member's block o into
//!    o's compact receive buffer `[2][block]`; each rank merges its blocks
//!    (`partial_merge2`) into the buffer the output gate reads;
//! 6. a group barrier (an all-gather of one word) ends the piece.
//!
//! Prefill runs these in the selection's pieces of `SCORE_ROWS` rows. Group
//! launches carry the rows of a power-of-two bucket ([`bucket`]): an
//! all-gather's flag offsets follow its tile count, so each bucket keeps
//! its own gather state. A gather overwrites a peer's output only once the
//! peer has acknowledged the previous call of the same state, which it
//! does on entering the next one, after reading that output. Outputs that
//! several states write are ordered as GLM's groups order them: each
//! bucket gathers its pairs into its own output (a later piece's selection
//! may enter its gather while a peer still merges the previous piece's
//! pairs); the received blocks and the gathered queries are one buffer for
//! every bucket, which the barrier that ends each piece protects. The
//! barrier's single word has one tile at every bucket, so one state serves
//! every piece and layer.

pub use bundle_runtime::context_parallel::{ContextParallel, GatherProtocol, RankState};
pub use model_qwen38_contract::context_parallel::{
    MAX_BUCKET, bucket, bucket_index, buckets, dims,
};

/// Whether create serves a context-parallel size above 1: on since the
/// lowering passed its GPU gate (plan step 45, qual.dcp2.json's full cohort,
/// 2026-10-07). Turning it off makes create refuse sizes above 1 again.
pub const SERVED: bool = true;

#[cfg(test)]
thread_local! {
    /// Lets this thread's fake-host tests create context-parallel instances.
    static SERVED_IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether create accepts context-parallel size `size`.
pub fn serves(size: u32) -> bool {
    #[cfg(test)]
    if SERVED_IN_TESTS.with(std::cell::Cell::get) {
        return true;
    }
    size == 1 || SERVED
}

/// Serves context parallelism on this thread's instances, for tests.
#[cfg(test)]
pub(crate) fn serve_in_tests(served: bool) {
    SERVED_IN_TESTS.with(|cell| cell.set(served));
}

// The largest bucket is a prefill selection piece.
const _: () = assert!(MAX_BUCKET == crate::contract::SCORE_ROWS);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_every_group_launch() {
        assert_eq!(buckets().count(), 12);
        assert_eq!(bucket_index(1).unwrap(), 0);
        assert_eq!(bucket_index(24).unwrap(), 5);
        assert_eq!(bucket_index(MAX_BUCKET).unwrap(), 11);
        assert!(bucket_index(MAX_BUCKET + 1).is_err());
        assert!(bucket_index(0).is_err());
        // The largest decode and verify graph fits below the largest piece.
        use crate::contract::{DECODE_BATCH, MAX_VERIFY_ROWS};
        assert!(bucket_index(DECODE_BATCH * MAX_VERIFY_ROWS).is_ok());
    }

    /// One rank's footprint: two send and two received blocks of 2048
    /// rows, the gathered queries, and one gather state per bucket.
    #[test]
    fn rank_state_follows_the_contract_bytes() {
        use model_qwen38_contract::context_parallel::rank_bytes;
        use runtime_bundle_abi::DeviceBufferSpanV1;
        let shape = crate::contract::Shape::TP4;
        let bytes = rank_bytes(shape, 2).unwrap();
        let mut lengths = Vec::new();
        let state = RankState::allocate(&bytes, |length| {
            lengths.push(length);
            Ok::<_, ()>(DeviceBufferSpanV1::new(
                runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, lengths.len() as u32),
                0,
                length,
            ))
        })
        .unwrap();
        assert_eq!(lengths.iter().sum::<u64>(), bytes.total());
        let block = dims(shape).block_floats(MAX_BUCKET) * 4;
        assert_eq!(state.send.length, 2 * block);
        assert_eq!(state.receive.length, 2 * block);
        assert_eq!(
            state.queries.length,
            2 * u64::from(MAX_BUCKET) * 6 * 256 * 2
        );
        assert_eq!(state.zeroed().len(), 2 * (12 + 12 + 12 + 1) + 1);
        assert_eq!(state.zeroed().last(), Some(&state.barrier_input));
        assert!(
            rank_bytes(
                crate::contract::Shape::of(crate::contract::Ranks::TP2).unwrap(),
                2
            )
            .is_err()
        );
    }
}
