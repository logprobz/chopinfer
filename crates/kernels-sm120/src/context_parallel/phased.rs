//! The phased group gather (`chop_comm_group_phased_f32_v1`,
//! `cuda/context_parallel/phased_gather.cuh`): one cluster of eight CTAs on
//! each member sends one block of 32-bit words to every member of a
//! context-parallel group, one destination at a time, so no GPU writes to
//! two peers at once. The all-gather of one shard
//! ([`gather_arguments`]) leaves every member's output as the shared
//! all-gather's (`group::gather_arguments`) does, and the exchange of a
//! destination-major buffer ([`exchange_arguments`]) leaves each owner's
//! output as the owner gathers (`group::owner_gather_arguments`, owners
//! 0..size) do together: the same bits at the same places.
//!
//! A state (one per call site, as for the shared all-gather) is the
//! members' outputs, flags and epochs. The protocol needs [`flag_bytes`] of
//! zeroed flags and [`EPOCH_BYTES`] of zeroed epoch per member, which every
//! `group::gather_buffers` state of the same group size holds, so a call
//! site can switch transports on the same buffers.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};

use super::group::{GatherState, GroupRank};

/// The entry. Its `chop_comm_` prefix marks it as a collective for hosts
/// that queue eager collectives in rank lockstep.
pub const SYMBOL: &str = "chop_comm_group_phased_f32_v1";

/// CTAs of the sending cluster, the whole grid.
const CTAS: u32 = 8;
const THREADS: u32 = 256;

/// Bytes of one member's flags at group size `size`: a data flag per source
/// and an acknowledgement per receiver.
pub fn flag_bytes(size: u32) -> u64 {
    2 * u64::from(size) * 4
}

/// Bytes of one member's epoch.
pub const EPOCH_BYTES: u64 = 4;

/// Refuses a member's state buffers that cannot hold the protocol at group
/// size `size`.
pub fn check_state(size: u32, flags: u64, epoch: u64) -> Result<(), Refusal> {
    if size == 0 || size as usize > MAX_COMM_RANKS {
        return Err(Refusal("a phased group gather needs 1 to 8 members"));
    }
    if flags < flag_bytes(size) || epoch < EPOCH_BYTES {
        return Err(Refusal(
            "a phased group gather state holds too few flag or epoch bytes",
        ));
    }
    Ok(())
}

/// The launch: one cluster of eight CTAs.
pub fn plan() -> LaunchPlan {
    LaunchPlan {
        grid: [CTAS, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [CTAS, 1, 1],
        pdl: true,
    }
}

/// The all-gather of `words` words of `input`: every member's shard lands
/// at slot `group rank` of every member's output, which then holds the
/// group's shards rank-major. `states` lists every topology rank's state of
/// this call site.
pub fn gather_arguments(
    rank: GroupRank,
    states: &[GatherState],
    input: u64,
    words: u64,
) -> Result<ArgumentList, Refusal> {
    arguments(rank, states, input, 0, words)
}

/// The exchange of the destination-major `send` buffer `[size][words]`:
/// block d goes to group rank d, at slot `group rank` of its output, so
/// owner d's output holds block d of every member, source-major.
pub fn exchange_arguments(
    rank: GroupRank,
    states: &[GatherState],
    send: u64,
    words: u64,
) -> Result<ArgumentList, Refusal> {
    arguments(rank, states, send, words, words)
}

fn arguments(
    rank: GroupRank,
    states: &[GatherState],
    input: u64,
    input_stride: u64,
    words: u64,
) -> Result<ArgumentList, Refusal> {
    if words == 0 {
        return Err(Refusal("a phased group gather moves at least one word"));
    }
    let members = rank.select(states)?;
    let own = members[rank.rank as usize];
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(input),
        Argument::U64(input_stride),
        Argument::DevicePointerArray(members.iter().map(|state| state.output).collect()),
        Argument::DevicePointerArray(members.iter().map(|state| state.flags).collect()),
        Argument::DeviceBuffer(own.epoch),
        Argument::U64(words),
        Argument::U32(rank.rank),
        Argument::U32(rank.size),
    ]);
    args.validate()?;
    Ok(args)
}

/// The launch ABI.
pub fn params() -> Vec<EntryParam> {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let array = |name: &str, cxx_type: &str| EntryParam::PointerArray {
        name: name.into(),
        slots: MAX_COMM_RANKS as u8,
        cxx_type: cxx_type.into(),
    };
    vec![
        pointer("input", "const uint32_t"),
        EntryParam::U64 {
            name: "input_stride".into(),
        },
        array("outputs", "uint32_t"),
        array("flags", "unsigned int"),
        pointer("epoch", "unsigned int"),
        EntryParam::U64 {
            name: "words".into(),
        },
        EntryParam::U32 {
            name: "rank".into(),
        },
        EntryParam::U32 {
            name: "ranks".into(),
        },
    ]
}

pub fn entry_spec() -> EntrySpec {
    EntrySpec::new(
        SYMBOL.into(),
        params(),
        "chop::context_parallel::phased::gather(input, input_stride, outputs, flags, epoch, \
         words, rank, ranks);"
            .into(),
        THREADS,
        1,
    )
}

/// The entry's cubin, named `name`.
pub fn cubin_spec(name: String) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"context_parallel/phased_gather.cuh\"".into()],
        defines: vec![],
        entries: vec![entry_spec()],
    }
}

#[cfg(test)]
mod tests {
    use super::super::group::{self, gather_buffers};
    use super::*;

    fn states() -> Vec<GatherState> {
        (0..4)
            .map(|rank| GatherState {
                output: 0x1000 * (rank + 1),
                flags: 0x10 * (rank + 1),
                epoch: rank + 100,
            })
            .collect()
    }

    /// The gather passes the shared all-gather's members, epoch, words and
    /// ranks; the exchange differs only in its input stride.
    #[test]
    fn arguments_follow_the_shared_gather() {
        let rank = GroupRank::of(3, 4, 2).unwrap();
        let shared = group::gather_arguments(rank, &states(), 0xbeef, 8192).unwrap();
        let gather = gather_arguments(rank, &states(), 0xbeef, 8192).unwrap();
        assert_eq!(gather.0[0], shared.0[0]);
        assert_eq!(gather.0[1], Argument::U64(0));
        assert_eq!(gather.0[2..], shared.0[1..7]);
        let exchange = exchange_arguments(rank, &states(), 0xbeef, 8192).unwrap();
        assert_eq!(exchange.0[1], Argument::U64(8192));
        assert_eq!(exchange.0[2..], gather.0[2..]);
        assert!(gather_arguments(rank, &states(), 0xbeef, 0).is_err());
        assert!(gather_arguments(rank, &states()[..3], 0xbeef, 1).is_err());
    }

    /// Every shared all-gather state of a group size holds the protocol,
    /// down to one word.
    #[test]
    fn shared_gather_states_hold_the_protocol() {
        for size in [2, 4] {
            for words in [1, 4096, 4097, 2048 * 8224] {
                let buffers = gather_buffers(size, words).unwrap();
                check_state(size, buffers.flag_bytes, buffers.epoch_bytes).unwrap();
            }
            assert!(check_state(size, flag_bytes(size) - 4, EPOCH_BYTES).is_err());
            assert!(check_state(size, flag_bytes(size), 0).is_err());
        }
        assert!(check_state(9, 1 << 20, 4).is_err());
    }

    #[test]
    fn one_cluster_launches() {
        let plan = plan();
        plan.validate().unwrap();
        assert_eq!(
            (plan.grid, plan.cluster, plan.block),
            ([8, 1, 1], [8, 1, 1], [256, 1, 1])
        );
        let spec = cubin_spec("comm_group_phased_test_sm120".into());
        assert_eq!(spec.entries.len(), 1);
        assert_eq!(spec.entries[0].symbol, SYMBOL);
        assert!(SYMBOL.starts_with("chop_comm_"));
    }
}
