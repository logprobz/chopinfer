//! Opt-in helpers for context-parallel groups: the row buckets a group
//! launch carries, the per-rank sizes of what a group exchanges, and each
//! rank's group buffers with their sizes and allocation order.
//!
//! A group of consecutive ranks gathers each member's queries and (score,
//! index) pairs, exchanges the partial blocks each member computed over its
//! own pages, and meets at a barrier (an all-gather of one word). Every
//! all-gather keeps its own state per bucket. Which kernels run, the
//! selection and attention the group lowers, and the model's own per-rank
//! buffer (`local`) stay with the bundle.

use runtime_bundle_abi::DeviceBufferSpanV1;

/// Words each member contributes to the group barrier: an all-gather of
/// one word, which completes on a member only once every member has
/// entered it.
pub const BARRIER_WORDS: u64 = 1;

/// The output, arrival flag and epoch bytes of one group all-gather state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherBytes {
    pub output_bytes: u64,
    pub flag_bytes: u64,
    pub epoch_bytes: u64,
}

/// The rows a group launch carries for `rows` rows: the power of two at or
/// above them.
pub const fn bucket(rows: u32) -> u32 {
    rows.next_power_of_two()
}

/// Every bucket up to `max_bucket` (a power of two), smallest first;
/// bucket `1 << i` is index `i`.
pub fn buckets(max_bucket: u32) -> impl Iterator<Item = u32> {
    (0..=max_bucket.ilog2()).map(|power| 1 << power)
}

/// The index of `rows`' bucket among [`buckets`] up to `max_bucket`.
pub fn bucket_index(rows: u32, max_bucket: u32) -> Result<usize, String> {
    let bucket = bucket(rows);
    if rows == 0 || bucket > max_bucket {
        return Err(format!("no context-parallel bucket holds {rows} rows"));
    }
    Ok(bucket.ilog2() as usize)
}

/// The per-rank sizes the group buffers follow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupDims {
    /// Query heads per rank.
    pub heads: u32,
    /// The width of a head's query and partial output.
    pub width: u32,
    /// Entries one selection keeps per row.
    pub kept: u32,
    /// Rows of the localized positions.
    pub scratch_rows: u32,
}

impl GroupDims {
    /// Floats of one partial block of `rows` rows: `m` and `l` `[rows,
    /// heads]`, then `o` `[rows, heads, width]`.
    pub const fn block_floats(self, rows: u32) -> u64 {
        rows as u64 * self.heads as u64 * (self.width as u64 + 2)
    }

    /// 32-bit words of one rank's BF16 queries of `rows` rows.
    pub const fn query_words(self, rows: u32) -> u64 {
        rows as u64 * self.heads as u64 * self.width as u64 / 2
    }

    /// Words of one rank's (score, index) pairs of `rows` rows.
    pub const fn pair_words(self, rows: u32) -> u64 {
        rows as u64 * self.kept as u64 * 2
    }
}

/// Bytes of one rank's group buffers, in [`RankState::allocate`]'s order:
/// its localized positions (scratch rows), the model's own buffer, its
/// pairs and destination-major send blocks at the largest bucket, the
/// gathered queries (one output for every bucket), its compact received
/// blocks (one per member), the gathered pairs of each bucket and the
/// barrier's word and output, then each bucket's query, pair and partial
/// gather flags and epochs and the barrier's.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankBytes {
    pub positions: u64,
    /// The model's own per-rank buffer.
    pub local: u64,
    pub pairs: u64,
    pub send: u64,
    pub queries: u64,
    pub receive: u64,
    pub gathered_pairs: Vec<u64>,
    pub barrier_input: u64,
    pub barrier_output: u64,
    pub query_gathers: Vec<GatherBytes>,
    pub pair_gathers: Vec<GatherBytes>,
    /// The partial blocks' gather states: one per bucket.
    pub partial_gathers: Vec<GatherBytes>,
    pub barrier: GatherBytes,
}

impl RankBytes {
    /// The bytes of groups of `size` over buckets up to `max_bucket`, with
    /// `local` bytes of the model's own buffer; `gather` sizes one group
    /// all-gather of (members, words per member) as the backend runs it.
    pub fn new<E>(
        dims: GroupDims,
        size: u32,
        max_bucket: u32,
        local: u64,
        gather: impl Fn(u32, u64) -> Result<GatherBytes, E>,
    ) -> Result<Self, E> {
        let gathers = |words: fn(GroupDims, u32) -> u64| {
            buckets(max_bucket)
                .map(|rows| gather(size, words(dims, rows)))
                .collect::<Result<Vec<_>, _>>()
        };
        let query_gathers = gathers(GroupDims::query_words)?;
        let pair_gathers = gathers(GroupDims::pair_words)?;
        let partial_gathers = gathers(GroupDims::block_floats)?;
        let barrier = gather(size, BARRIER_WORDS)?;
        Ok(Self {
            positions: u64::from(dims.scratch_rows) * 4,
            local,
            pairs: dims.pair_words(max_bucket) * 4,
            send: u64::from(size) * dims.block_floats(max_bucket) * 4,
            queries: query_gathers[query_gathers.len() - 1].output_bytes,
            receive: partial_gathers[partial_gathers.len() - 1].output_bytes,
            gathered_pairs: pair_gathers
                .iter()
                .map(|bytes| bytes.output_bytes)
                .collect(),
            barrier_input: BARRIER_WORDS * 4,
            barrier_output: barrier.output_bytes,
            query_gathers,
            pair_gathers,
            partial_gathers,
            barrier,
        })
    }

    /// Every buffer's bytes.
    pub fn total(&self) -> u64 {
        let protocols = self
            .query_gathers
            .iter()
            .chain(&self.pair_gathers)
            .chain(&self.partial_gathers)
            .chain([&self.barrier])
            .map(|bytes| bytes.flag_bytes + bytes.epoch_bytes)
            .sum::<u64>();
        self.positions
            + self.local
            + self.pairs
            + self.send
            + self.queries
            + self.receive
            + self.gathered_pairs.iter().sum::<u64>()
            + self.barrier_input
            + self.barrier_output
            + protocols
    }
}

/// One group all-gather state of one bucket on one rank.
#[derive(Clone, Copy, Debug)]
pub struct GatherProtocol {
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

/// One rank's group buffers. The rank's own: its localized positions, the
/// model's own buffer, its pairs, the destination-major partial blocks it
/// sends and the barrier's word. Peer-visible, which the group writes: the
/// gathered queries, the gathered pairs of each bucket, the compact
/// received blocks, the barrier's output, and the all-gathers' (per bucket)
/// and the barrier's flags and epochs.
#[derive(Clone, Debug)]
pub struct RankState {
    pub positions: DeviceBufferSpanV1,
    pub local: DeviceBufferSpanV1,
    pub pairs: DeviceBufferSpanV1,
    pub send: DeviceBufferSpanV1,
    pub queries: DeviceBufferSpanV1,
    /// Per bucket, as [`buckets`].
    pub gathered_pairs: Vec<DeviceBufferSpanV1>,
    pub receive: DeviceBufferSpanV1,
    pub barrier_input: DeviceBufferSpanV1,
    pub barrier_output: DeviceBufferSpanV1,
    pub query_gathers: Vec<GatherProtocol>,
    pub pair_gathers: Vec<GatherProtocol>,
    /// Per bucket: the partial blocks' gather state.
    pub partial_gathers: Vec<GatherProtocol>,
    pub barrier: GatherProtocol,
}

impl RankState {
    /// Allocates one rank's buffers of `bytes` through `allocate`, after
    /// peer access is granted, in [`RankBytes`]' order.
    pub fn allocate<E>(
        bytes: &RankBytes,
        mut allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, E>,
    ) -> Result<Self, E> {
        let positions = allocate(bytes.positions)?;
        let local = allocate(bytes.local)?;
        let pairs = allocate(bytes.pairs)?;
        let send = allocate(bytes.send)?;
        let queries = allocate(bytes.queries)?;
        let receive = allocate(bytes.receive)?;
        let gathered_pairs = bytes
            .gathered_pairs
            .iter()
            .map(|&length| allocate(length))
            .collect::<Result<Vec<_>, _>>()?;
        let barrier_input = allocate(bytes.barrier_input)?;
        let barrier_output = allocate(bytes.barrier_output)?;
        let mut protocols = |gathers: &[GatherBytes]| -> Result<Vec<_>, E> {
            gathers
                .iter()
                .map(|gather| {
                    Ok(GatherProtocol {
                        flags: allocate(gather.flag_bytes)?,
                        epoch: allocate(gather.epoch_bytes)?,
                    })
                })
                .collect()
        };
        let query_gathers = protocols(&bytes.query_gathers)?;
        let pair_gathers = protocols(&bytes.pair_gathers)?;
        let partial_gathers = protocols(&bytes.partial_gathers)?;
        let barrier = protocols(&[bytes.barrier])?.remove(0);
        Ok(Self {
            positions,
            local,
            pairs,
            send,
            queries,
            gathered_pairs,
            receive,
            barrier_input,
            barrier_output,
            query_gathers,
            pair_gathers,
            partial_gathers,
            barrier,
        })
    }

    fn protocols(&self) -> impl Iterator<Item = &GatherProtocol> {
        self.query_gathers
            .iter()
            .chain(&self.pair_gathers)
            .chain(&self.partial_gathers)
            .chain([&self.barrier])
    }

    /// The spans that must start at zero: every protocol's flags and
    /// epochs, and the barrier's word, which no launch writes.
    pub fn zeroed(&self) -> Vec<DeviceBufferSpanV1> {
        self.protocols()
            .flat_map(|protocol| [protocol.flags, protocol.epoch])
            .chain([self.barrier_input])
            .collect()
    }

    /// Every span, for bounds checks.
    pub fn spans(&self) -> Vec<DeviceBufferSpanV1> {
        [
            self.positions,
            self.local,
            self.pairs,
            self.send,
            self.queries,
            self.receive,
            self.barrier_output,
        ]
        .into_iter()
        .chain(self.gathered_pairs.iter().copied())
        .chain(self.zeroed())
        .collect()
    }
}

/// An instance's context-parallel groups: the size and every rank's state,
/// in topology rank order.
#[derive(Clone, Debug)]
pub struct ContextParallel {
    pub size: u32,
    pub ranks: Vec<RankState>,
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;

    #[test]
    fn buckets_are_the_powers_of_two_up_to_the_largest() {
        assert_eq!(buckets(2048).count(), 12);
        assert_eq!(bucket_index(1, 2048).unwrap(), 0);
        assert_eq!(bucket_index(24, 2048).unwrap(), 5);
        assert_eq!(bucket_index(2048, 2048).unwrap(), 11);
        assert!(bucket_index(2049, 2048).is_err());
        assert!(bucket_index(0, 2048).is_err());
    }

    /// The allocation follows the bytes in order, and the zeroed spans are
    /// every protocol's then the barrier's word.
    #[test]
    fn a_rank_allocates_its_bytes_in_order() {
        let dims = GroupDims {
            heads: 2,
            width: 64,
            kept: 4,
            scratch_rows: 16,
        };
        let gather = |size: u32, words: u64| {
            Ok::<_, ()>(GatherBytes {
                output_bytes: u64::from(size) * words * 4,
                flag_bytes: 8,
                epoch_bytes: 4,
            })
        };
        let bytes = RankBytes::new(dims, 2, 16, 100, gather).unwrap();
        assert_eq!(bytes.send, 2 * 16 * 2 * 66 * 4);
        let mut lengths = Vec::new();
        let state = RankState::allocate(&bytes, |length| {
            lengths.push(length);
            let handle = BundleDeviceBufferHandleV1::from_parts(1, lengths.len() as u32);
            Ok::<_, ()>(DeviceBufferSpanV1::new(handle, 0, length))
        })
        .unwrap();
        assert_eq!(lengths.iter().sum::<u64>(), bytes.total());
        assert_eq!(lengths[..2], [16 * 4, 100]);
        assert_eq!(state.local.length, 100);
        assert_eq!(state.zeroed().len(), 2 * (5 + 5 + 5 + 1) + 1);
        assert_eq!(state.zeroed().last(), Some(&state.barrier_input));
        assert_eq!(state.spans().len(), 7 + 5 + state.zeroed().len());
    }
}
