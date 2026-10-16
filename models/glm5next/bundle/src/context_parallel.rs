//! Context-parallel groups of 2 or 4 ranks: each rank's buffers and
//! peer-visible comm state, and the switch under which create serves a size
//! above 1.
//!
//! A group of `size` consecutive ranks splits each sequence's KV pages
//! round-robin (`runtime_kv::ContextParallelPagesV1`): physical page j lives
//! on group rank `j % size` at the local page `table[j / size]`, so the
//! step's block table is every rank's own table of local pages. Heads stay
//! split over the ranks. Per sparse layer (`lower/context_parallel.rs`):
//!
//! 1. each appended position is localized to its owner (`localize_positions`)
//!    and only the owner writes the latent row; the indexer key write takes
//!    the group rank and size;
//! 2. each rank selects over its own pools (`select_per_rank`), the group
//!    all-gathers the (score, pool) pairs and every rank merges them into
//!    the global key list (`topk_merge`);
//! 3. each rank localizes the list to its own pages (`localize_keys`);
//! 4. the group all-gathers the absorbed queries `[size][rows][heads]
//!    [latent]`, and each rank runs the heads-16 partial attention once per
//!    source block over its own keys, writing destination-major blocks;
//! 5. after every partial attention, one exchange
//!    (`phased::exchange_arguments`) moves each member's block o into group
//!    rank o's compact receive buffer `[size][block]`, source-major; each
//!    rank merges its blocks (`partial_merge`) into `AttentionOut`;
//! 6. a group barrier (an all-gather of one word) ends the piece; the value
//!    projection and the all-reduce follow unchanged.
//!
//! The pair and query all-gathers and the exchange run the phased group
//! gather (`kernels_sm120::context_parallel::phased`), which sends to one
//! peer at a time; the barrier runs the shared all-gather
//! (`group::gather_arguments`). Group launches carry the rows of a
//! power-of-two bucket ([`bucket`]), and each bucket keeps its own state per
//! call site (the shared all-gather's flag offsets follow its tile count,
//! as the one-shot all-reduce keeps one per row count; the phased gather
//! uses the first `2 * size` flag words of the same buffers). A gather
//! overwrites a peer's output only once the peer has acknowledged the
//! previous call of the same state, which it does on entering the next
//! one, after reading that output. The shared all-gather's first call of a
//! state has nothing to wait for (the phased gather's waits for the peer to
//! enter it), and either way an output that several states write needs
//! another ordering:
//!
//! - the gathered pairs: an eager prefill's indexer pieces select one after
//!   another with no other group launch between them, so a rank could enter
//!   a later piece's gather (another bucket) while a peer still merges the
//!   previous piece's pairs. Each bucket gathers into its own output.
//! - the received blocks: one buffer for every bucket, written only by the
//!   exchanges. A member enters the next piece's exchange only after the
//!   barrier that ends this piece, which completes only once every member
//!   has entered it, after its merge read its blocks. The last piece of a
//!   phase is followed by the sublayer's all-reduce as well.
//! - the gathered queries: one buffer for every bucket. Every partial
//!   attention precedes the exchange, which completes on a member only once
//!   every source has sent to it, after that source's partial attention
//!   read the queries; the barrier then orders the next piece's query
//!   gather after every member's reads.
//!
//! The barrier's single word has one tile at every bucket, so one state
//! serves every piece and layer.

use model_glm5next_contract::context_parallel::Dims;
use model_glm5next_contract::contract::Shape;

use crate::step::PIECE_ROWS;

/// Whether create serves a context-parallel size above 1: on since the
/// group's layer matched size 1 on four GPUs
/// (`crates/kernel-lab/tests/glm5next_context_parallel_chain.rs`) and the
/// served sizes' bring-up matched the reference.
pub const SERVED: bool = true;

/// Whether create accepts context-parallel size `size`; the topology's
/// declared sizes are checked when the record is read.
pub fn serves(size: u32) -> bool {
    size == 1 || SERVED
}

pub use bundle_runtime::context_parallel::{ContextParallel, GatherProtocol, RankState};
pub use model_glm5next_contract::context_parallel::{MAX_BUCKET, bucket, bucket_index, buckets};

// The largest bucket is an eager prefill's indexer piece.
const _: () = assert!(PIECE_ROWS == MAX_BUCKET);

/// Floats of one partial block of `rows` rows: `m` and `l` `[rows, heads]`,
/// then `o` `[rows, heads, latent]`.
pub fn block_floats(shape: Shape, rows: u32) -> u64 {
    Dims::of(shape).group().block_floats(rows)
}

/// 32-bit words of one rank's absorbed queries of `rows` rows.
pub fn query_words(shape: Shape, rows: u32) -> u64 {
    Dims::of(shape).group().query_words(rows)
}

/// Words of one rank's (score, pool) pairs of `rows` rows.
pub fn pair_words(shape: Shape, rows: u32) -> u64 {
    Dims::of(shape).group().pair_words(rows)
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::DeviceBufferSpanV1;

    use super::*;

    #[test]
    fn buckets_cover_every_group_launch() {
        assert_eq!(buckets().count(), 12);
        assert_eq!(bucket_index(1).unwrap(), 0);
        assert_eq!(bucket_index(24).unwrap(), 5);
        assert_eq!(bucket_index(PIECE_ROWS).unwrap(), 11);
        assert!(bucket_index(PIECE_ROWS + 1).is_err());
        assert!(bucket_index(0).is_err());
        // The largest verify bucket fits below the largest piece.
        use model_glm5next_contract::contract::{DECODE_BATCH, MAX_VERIFY_ROWS};
        assert!(bucket_index(DECODE_BATCH * MAX_VERIFY_ROWS).is_ok());
    }

    /// One rank's footprint at the largest piece: `size` received blocks of
    /// 2048 rows.
    #[test]
    fn rank_state_sizes_follow_the_bucket() {
        let shape = Shape::TP4;
        assert_eq!(block_floats(shape, 1), 16 * 514);
        assert_eq!(query_words(shape, 1), 4096);
        assert_eq!(pair_words(shape, 1), 1024);
        for size in [2, 4] {
            let mut bytes = Vec::new();
            let rank_bytes =
                model_glm5next_contract::context_parallel::rank_bytes(Dims::of(shape), size)
                    .unwrap();
            let state = RankState::allocate(&rank_bytes, |length| {
                bytes.push(length);
                Ok::<_, ()>(DeviceBufferSpanV1::new(
                    runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(
                        1,
                        bytes.len() as u32,
                    ),
                    0,
                    length,
                ))
            })
            .unwrap();
            let block = block_floats(shape, MAX_BUCKET) * 4;
            assert_eq!(state.send.length, u64::from(size) * block);
            assert_eq!(state.receive.length, u64::from(size) * block);
            assert_eq!(
                state.queries.length,
                u64::from(size) * u64::from(MAX_BUCKET) * 16 * 512 * 2
            );
            // Bucket 2048's query gather: 2048 tiles of flags per source
            // and per acknowledgment.
            assert_eq!(
                state.query_gathers[11].flags.length,
                2 * u64::from(size) * 2048 * 4
            );
            assert_eq!(state.zeroed().len(), 2 * (12 + 12 + 12 + 1) + 1);
            assert_eq!(state.zeroed().last(), Some(&state.barrier_input));
        }
    }
}
