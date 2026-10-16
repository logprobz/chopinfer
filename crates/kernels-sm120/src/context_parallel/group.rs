//! Group-scoped launches of the shared all-gather
//! (`chop_comm_allgather_f32_v1`, `cuda/comm/comm.cuh`) and row exchange
//! (`chop_comm_exchange_rows_f32_v1`, `cuda/comm/push.cuh`) for a
//! context-parallel group.
//!
//! The comm kernels never see topology ranks: a launch takes a rank, a rank
//! count and pointer arrays in slot order. A group of `size` consecutive
//! topology ranks therefore runs the shared entry unchanged with rank = the
//! group rank, ranks = the group size and only the group's outputs and flags,
//! in group-rank order. Each group keeps its own state for each call site
//! (output span, flags and epoch per rank): the flag layout depends on the
//! tile count, so two gathers of different sizes never share flags, and two
//! groups never share them either.

use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};
use kernel_core::row_exchange::{
    EXCHANGE_ROWS_EPOCH_BYTES, EXCHANGE_ROWS_FLAG_BYTES, ExchangeRowsCall, exchange_rows_arguments,
    exchange_rows_plan,
};

/// The shared all-gather entry a group launches.
pub const ALLGATHER_SYMBOL: &str = "chop_comm_allgather_f32_v1";

/// Elements one all-gather CTA moves (the entry's tile).
const TILE: u64 = 4096;
const THREADS: u32 = 256;

/// A topology rank's place in its context-parallel group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupRank {
    /// The group's index: groups are consecutive topology ranks.
    pub group: u32,
    /// The rank within its group, the comm launch's `rank`.
    pub rank: u32,
    /// Ranks per group, the comm launch's `ranks`.
    pub size: u32,
}

impl GroupRank {
    /// Topology rank `rank` of `ranks` in groups of `size`.
    pub fn of(rank: u32, ranks: u32, size: u32) -> Result<Self, Refusal> {
        if size == 0
            || size as usize > MAX_COMM_RANKS
            || ranks == 0
            || !ranks.is_multiple_of(size)
            || rank >= ranks
        {
            return Err(Refusal(
                "context-parallel group does not divide the topology ranks",
            ));
        }
        Ok(Self {
            group: rank / size,
            rank: rank % size,
            size,
        })
    }

    /// The group's topology ranks in group-rank order: the slot order of
    /// every pointer array the group passes.
    pub fn members(self) -> impl Iterator<Item = u32> {
        let first = self.group * self.size;
        first..first + self.size
    }

    /// The group's entries of a per-topology-rank table, in slot order.
    pub fn select<T: Copy>(self, per_rank: &[T]) -> Result<Vec<T>, Refusal> {
        self.members()
            .map(|rank| per_rank.get(rank as usize).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or(Refusal("a per-rank table misses a group member"))
    }
}

/// Per-rank bytes of one group all-gather state of `elements_per_rank` F32
/// words: the gathered output (every rank's shard, rank-major), the flags
/// and the epoch cells. Flags and epoch start zeroed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherBuffers {
    pub output_bytes: u64,
    pub flag_bytes: u64,
    pub epoch_bytes: u64,
}

fn tiles(elements_per_rank: u64) -> Result<u64, Refusal> {
    if elements_per_rank == 0 {
        return Err(Refusal("an all-gather moves at least one element"));
    }
    Ok(elements_per_rank.div_ceil(TILE))
}

pub fn gather_buffers(size: u32, elements_per_rank: u64) -> Result<GatherBuffers, Refusal> {
    let tiles = tiles(elements_per_rank)?;
    let size = u64::from(size);
    Ok(GatherBuffers {
        output_bytes: size
            .checked_mul(elements_per_rank)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or(Refusal("all-gather output overflows u64"))?,
        // Arrival flags of every source and the acknowledgements of every
        // destination, per tile.
        flag_bytes: 2 * size * tiles * 4,
        epoch_bytes: tiles * 4,
    })
}

/// The launch of a gather of `elements_per_rank` words: one CTA per tile.
pub fn gather_plan(elements_per_rank: u64) -> Result<LaunchPlan, Refusal> {
    let plan = LaunchPlan {
        grid: [
            u32::try_from(tiles(elements_per_rank)?)
                .map_err(|_| Refusal("all-gather tiles exceed the grid"))?,
            1,
            1,
        ],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// One topology rank's state of one group gather call site, by device
/// address: its gathered output span, flags and epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherState {
    pub output: u64,
    pub flags: u64,
    pub epoch: u64,
}

/// The shared all-gather's arguments for topology rank `rank`: its `input`
/// shard of `elements_per_rank` words lands at slot `group rank` of every
/// group member's output, which then holds the group's shards rank-major.
/// `states` lists every topology rank's state of this call site.
pub fn gather_arguments(
    rank: GroupRank,
    states: &[GatherState],
    input: u64,
    elements_per_rank: u64,
) -> Result<ArgumentList, Refusal> {
    // Every member receives every shard.
    arguments(rank, states, input, elements_per_rank, rank.size)
}

/// The shared all-gather's arguments delivering to group rank `owner`
/// only: `input` lands at slot `group rank` of `owner`'s output, compact
/// `[size][elements_per_rank]` source-major, and every other output stays
/// untouched. The entry's `target`. Only the owner waits for every source;
/// a sender waits only for the owner's acknowledgment of the state's
/// previous call, so the members must run the same owner sequence on one
/// state, and an output's reuse needs its own ordering once the state is
/// cold.
pub fn owner_gather_arguments(
    rank: GroupRank,
    states: &[GatherState],
    input: u64,
    elements_per_rank: u64,
    owner: u32,
) -> Result<ArgumentList, Refusal> {
    if owner >= rank.size {
        return Err(Refusal("an owner gather targets a member of its group"));
    }
    arguments(rank, states, input, elements_per_rank, owner)
}

fn arguments(
    rank: GroupRank,
    states: &[GatherState],
    input: u64,
    elements_per_rank: u64,
    target: u32,
) -> Result<ArgumentList, Refusal> {
    tiles(elements_per_rank)?;
    let members = rank.select(states)?;
    let own = members[rank.rank as usize];
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(input),
        Argument::DevicePointerArray(members.iter().map(|state| state.output).collect()),
        Argument::DevicePointerArray(members.iter().map(|state| state.flags).collect()),
        Argument::DeviceBuffer(own.epoch),
        Argument::U64(elements_per_rank),
        Argument::U32(rank.rank),
        Argument::U32(rank.size),
        Argument::U32(target),
    ]);
    args.validate()?;
    Ok(args)
}

/// Words of a gathered BF16 tensor of `elements` values: the F32 entry
/// moves 32-bit words, two BF16 values each, bit for bit.
pub fn bf16_words(elements: u64) -> Result<u64, Refusal> {
    if elements == 0 || !elements.is_multiple_of(2) {
        return Err(Refusal("a BF16 gather moves whole 32-bit words"));
    }
    Ok(elements / 2)
}

/// The shared row exchange a group launches for its partials.
pub const EXCHANGE_SYMBOL: &str = "chop_comm_exchange_rows_f32_v1";

/// Per-rank bytes of one group exchange of blocks of `block_floats` F32
/// words: the destination-major send buffer `[size][block]`, the receive
/// buffer and the exchange's flags and epoch cells, which start zeroed.
///
/// The shared entry stores source q's block for owner o at
/// `outputs[o] + (o * size + q) * block` and is only given allocation bases,
/// so each receive buffer spans `size * size` blocks; owner o's blocks are
/// the `size` blocks from [`exchange_receive_offset`] (the merge's view), and
/// the rest of the buffer is never written.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExchangeBuffers {
    pub send_bytes: u64,
    pub receive_bytes: u64,
    pub flag_bytes: u64,
    pub epoch_bytes: u64,
}

pub fn exchange_buffers(size: u32, block_floats: u64) -> Result<ExchangeBuffers, Refusal> {
    if size == 0 || size as usize > MAX_COMM_RANKS || block_floats == 0 {
        return Err(Refusal("invalid group exchange geometry"));
    }
    let send = u64::from(size)
        .checked_mul(block_floats)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or(Refusal("group exchange buffer overflows u64"))?;
    Ok(ExchangeBuffers {
        send_bytes: send,
        receive_bytes: send
            .checked_mul(u64::from(size))
            .ok_or(Refusal("group exchange buffer overflows u64"))?,
        flag_bytes: EXCHANGE_ROWS_FLAG_BYTES,
        epoch_bytes: EXCHANGE_ROWS_EPOCH_BYTES,
    })
}

/// The byte offset in group rank `owner`'s receive buffer where its blocks
/// start, source-major: `owner * size * block` floats. The merge reads the
/// `size * block` floats from there (`MergeLayout::exchanged`).
pub fn exchange_receive_offset(owner: u32, size: u32, block_floats: u64) -> u64 {
    u64::from(owner) * u64::from(size) * block_floats * 4
}

/// The exchange's launch: one cluster of eight CTAs.
pub fn exchange_plan() -> LaunchPlan {
    exchange_rows_plan()
}

/// One topology rank's state of one group exchange call site, by device
/// address: its receive buffer (sized by [`exchange_buffers`]), flags and
/// epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExchangeState {
    pub receive: u64,
    pub flags: u64,
    pub epoch: u64,
}

/// The shared row exchange's arguments for topology rank `rank`: row d of
/// its destination-major `send` buffer (`[size][block_floats]`) goes to
/// group rank d, at `receive[d] + exchange_receive_offset(d) + q * block`
/// for source group rank q. Every pointer is an allocation base.
pub fn exchange_arguments(
    rank: GroupRank,
    states: &[ExchangeState],
    send: u64,
    block_floats: u64,
) -> Result<ArgumentList, Refusal> {
    let members = rank.select(states)?;
    exchange_rows_arguments(ExchangeRowsCall {
        input: send,
        outputs: members.iter().map(|state| state.receive).collect(),
        flags: members.iter().map(|state| state.flags).collect(),
        epoch: members[rank.rank as usize].epoch,
        rows: rank.size,
        shard: block_floats,
        rank: rank.rank,
        ranks: rank.size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_are_consecutive_ranks() {
        let ranks = (0..4)
            .map(|rank| GroupRank::of(rank, 4, 2).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            ranks.iter().map(|r| (r.group, r.rank)).collect::<Vec<_>>(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        assert_eq!(ranks[3].members().collect::<Vec<_>>(), [2, 3]);
        assert_eq!(GroupRank::of(3, 4, 4).unwrap().rank, 3);
        for (rank, ranks, size) in [(0, 4, 3), (4, 4, 2), (0, 4, 0), (0, 16, 16)] {
            assert!(GroupRank::of(rank, ranks, size).is_err());
        }
        assert!(ranks[3].select(&[1, 2, 3]).is_err());
    }

    #[test]
    fn a_group_passes_only_its_members_in_group_order() {
        let states = (0..4)
            .map(|rank| GatherState {
                output: 0x1000 * (rank + 1),
                flags: 0x10 * (rank + 1),
                epoch: rank + 100,
            })
            .collect::<Vec<_>>();
        let rank = GroupRank::of(3, 4, 2).unwrap();
        let args = gather_arguments(rank, &states, 0xbeef, 8192).unwrap();
        assert_eq!(
            args.0,
            [
                Argument::DeviceBuffer(0xbeef),
                Argument::DevicePointerArray(vec![0x3000, 0x4000]),
                Argument::DevicePointerArray(vec![0x30, 0x40]),
                Argument::DeviceBuffer(103),
                Argument::U64(8192),
                Argument::U32(1),
                Argument::U32(2),
                Argument::U32(2),
            ]
        );
        // The owner gather keeps the members and names group rank 0 only.
        let owner = owner_gather_arguments(rank, &states, 0xbeef, 8192, 0).unwrap();
        assert_eq!(owner.0[..7], args.0[..7]);
        assert_eq!(owner.0[7], Argument::U32(0));
        assert!(owner_gather_arguments(rank, &states, 0xbeef, 8192, 2).is_err());
        assert!(gather_arguments(rank, &states, 0xbeef, 0).is_err());
    }

    #[test]
    fn buffers_and_plans_follow_the_tiles() {
        // 32 decode rows of 64 gathered heads of (512 + 2) floats.
        let elements = 32 * 64 * 514;
        let tiles = u64::div_ceil(elements, 4096);
        assert_eq!(
            gather_buffers(4, elements).unwrap(),
            GatherBuffers {
                output_bytes: 4 * elements * 4,
                flag_bytes: 2 * 4 * tiles * 4,
                epoch_bytes: tiles * 4,
            }
        );
        assert_eq!(gather_plan(elements).unwrap().grid, [tiles as u32, 1, 1]);
        assert_eq!(gather_plan(1).unwrap().grid, [1, 1, 1]);
        assert!(gather_plan(0).is_err());
        assert_eq!(bf16_words(32 * 16 * 512).unwrap(), 32 * 16 * 256);
        assert!(bf16_words(3).is_err());
    }

    /// Where the exchange entry stores (push.cuh `rows::exchange`): source
    /// q's row d at `outputs[d] + (d * ranks + q) * shard` floats. With the
    /// allocation bases, every block lands inside its owner's receive buffer,
    /// source-major from the owner's receive offset.
    #[test]
    fn exchanged_blocks_land_source_major_at_each_owner() {
        let block = 1000_u64;
        for (ranks, size) in [(4_u32, 2_u32), (4, 4), (2, 2)] {
            let buffers = exchange_buffers(size, block).unwrap();
            let states = (0..u64::from(ranks))
                .map(|rank| ExchangeState {
                    receive: 0x10_0000_0000 * (rank + 1),
                    flags: 0x100 * (rank + 1),
                    epoch: 0x200 * (rank + 1),
                })
                .collect::<Vec<_>>();
            for topology in 0..ranks {
                let rank = GroupRank::of(topology, ranks, size).unwrap();
                let args = exchange_arguments(rank, &states, 0xabc, block).unwrap();
                let Argument::DevicePointerArray(outputs) = &args.0[1] else {
                    panic!("outputs are a pointer array");
                };
                assert_eq!(args.0[4], Argument::U32(size));
                assert_eq!(args.0[5], Argument::U64(block));
                let q = u64::from(rank.rank);
                for (d, member) in rank.members().enumerate() {
                    let receive = states[member as usize].receive;
                    assert_eq!(outputs[d], receive, "an allocation base");
                    let store = outputs[d] + (d as u64 * u64::from(size) + q) * block * 4;
                    let view = receive + exchange_receive_offset(d as u32, size, block);
                    assert_eq!(store, view + q * block * 4);
                    assert!(store + block * 4 <= receive + buffers.receive_bytes);
                }
            }
        }
        assert_eq!(
            exchange_buffers(4, block).unwrap(),
            ExchangeBuffers {
                send_bytes: 4 * block * 4,
                receive_bytes: 16 * block * 4,
                flag_bytes: EXCHANGE_ROWS_FLAG_BYTES,
                epoch_bytes: EXCHANGE_ROWS_EPOCH_BYTES,
            }
        );
        assert_eq!(exchange_receive_offset(3, 4, block), 12 * block * 4);
        assert!(exchange_buffers(0, block).is_err());
        assert_eq!(exchange_plan().grid, [8, 1, 1]);
    }
}
