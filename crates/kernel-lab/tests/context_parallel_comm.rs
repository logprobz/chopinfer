//! The shared F32 all-gather launched per context-parallel group
//! (`kernels_sm120::context_parallel::group`): every rank passes its group
//! rank, the group size and only its group's outputs and flags. With four
//! GPUs, two groups of two run in one launch, each with its own state, and
//! each rank's output holds exactly its group's shards rank-major; then one
//! group of every rank runs.
//!
//! Each state takes six calls: two eager, two replays of one captured graph
//! and two eager calls behind a PDL producer that holds every input at NaN
//! poison until after its tail. Before every call the outputs are poisoned
//! and the inputs hold that call's own shards, while the flags and epochs
//! carry over, so a call that writes nothing, writes a stale shard or reads
//! its input before its dependency wait fails.
//!
//! The row exchange of partial blocks runs the same six calls per state:
//! each rank sends row d of its destination-major buffer to group rank d,
//! and every rank receives its group's blocks source-major in the view at
//! `group::exchange_receive_offset` of a receive buffer of `size * size`
//! blocks, whose other bytes stay poisoned (the entry is given allocation
//! bases only).
//!
//! The owner gathers (`group::owner_gather_arguments`) run the same six
//! calls per state: each call launches the shared all-gather once per owner
//! o in order 0..size, each member passing its send block o and naming o
//! as the target, then the group barrier (an all-gather of one word). Each
//! owner's compact receive buffer of `size` blocks must hold its group's
//! blocks source-major, every byte written, with partials distinct per
//! call, topology source and owner (some pairs empty: `m = -inf`, `l = 0`).
//! The same send buffers then go through the row exchange, and the partial
//! merge (`partial_merge{size}`, heads 16 of 512) of each rank's heads over
//! the owner-gathered blocks equals, bit for bit, the merge over the
//! exchange's view.
//!
//! Run with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release --test context_parallel_comm`.

#[path = "../context_parallel_shapes.rs"]
mod shapes;

use std::collections::BTreeMap;

use kernel_core::contract::{Argument, ArgumentList, DType};
use kernel_lab::launch::{self, Bindings, CollectiveModules};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use kernels_sm120::context_parallel::group::{
    ALLGATHER_SYMBOL, EXCHANGE_SYMBOL, ExchangeState, GatherState, GroupRank, exchange_arguments,
    exchange_buffers, exchange_plan, exchange_receive_offset, gather_arguments, gather_buffers,
    gather_plan, owner_gather_arguments,
};
use kernels_sm120::context_parallel::{Kind, MergeLayout};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument};

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// Rank `rank`'s shard at call `call`: finite, and distinct per call, rank
/// and element (below 2^20 elements).
fn shard(call: usize, rank: usize, elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|i| f32::from_bits(0x2000_0000 + (((call * 8 + rank) as u32) << 20) + i as u32))
        .collect()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn zeroed(session: &Session, bytes: u64) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes as usize)?;
    buffer
        .copy_from_bytes(&vec![0; bytes as usize], &session.stream)
        .map_err(err)?;
    Ok(buffer)
}

/// The PDL producer probe on one rank, with one staged operand: it
/// overwrites `input` with NaN poison, triggers its dependents, and after
/// its tail copies `fresh` (the call's shard) into `input`.
struct Producer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    live: DeviceBuffer,
    descriptor: DeviceBuffer,
    fresh: DeviceBuffer,
}

impl Producer {
    fn new(session: &Session, input: &DeviceBuffer, elements: usize) -> LabResult<Producer> {
        let module = session.module("probe_pdl_prefetch_sm120")?;
        let fresh = session.allocate(elements * 4)?;
        let mut descriptor = Vec::new();
        for buffer in [&fresh, input] {
            descriptor.extend_from_slice(
                &buffer
                    .device_address_at(0, elements * 4)
                    .map_err(err)?
                    .to_ne_bytes(),
            );
        }
        descriptor.extend_from_slice(&(elements as u32).to_ne_bytes());
        // Bit 1: floating poison; bit 0 clear: `fresh` is not overwritten.
        descriptor.extend_from_slice(&2_u32.to_ne_bytes());
        let staged = session.allocate(descriptor.len())?;
        staged
            .copy_from_bytes(&descriptor, &session.stream)
            .map_err(err)?;
        Ok(Producer {
            function: module
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction: zeroed(session, 16)?,
            checksum: session.allocate(128 * 4)?,
            live: zeroed(session, 16)?,
            descriptor: staged,
            fresh,
        })
    }

    fn launch(&self, session: &Session) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (1, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(&self.descriptor),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(&self.live),
                    KernelArgument::U32(0),
                    KernelArgument::U64(15_000),
                ],
            )
            .map_err(err)
    }
}

fn run_groups(sessions: &[Session], size: usize, elements: usize) -> LabResult<()> {
    let ranks = sessions.len();
    let buffers = gather_buffers(size as u32, elements as u64).map_err(|e| e.0.to_string())?;
    let plan = gather_plan(elements as u64).map_err(|e| e.0.to_string())?;
    let inputs = sessions
        .iter()
        .map(|session| session.allocate(elements * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let outputs = sessions
        .iter()
        .map(|session| session.allocate(buffers.output_bytes as usize))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| zeroed(session, buffers.flag_bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let epochs = sessions
        .iter()
        .map(|session| zeroed(session, buffers.epoch_bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let producers = sessions
        .iter()
        .zip(&inputs)
        .map(|(session, input)| Producer::new(session, input, elements))
        .collect::<LabResult<Vec<_>>>()?;
    // Handles stand for device addresses; slot order comes from the group.
    let states = (0..ranks as u64)
        .map(|rank| GatherState {
            output: rank + 1,
            flags: rank + 11,
            epoch: rank + 21,
        })
        .collect::<Vec<_>>();
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let gather = |only: Option<usize>| {
        launch::launch_collective(sessions, &modules, ALLGATHER_SYMBOL, plan, only, |rank| {
            let group = GroupRank::of(rank as u32, ranks as u32, size as u32)
                .map_err(|e| e.0.to_string())?;
            let mut bindings: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
            bindings.insert(100, &inputs[rank]);
            for peer in 0..ranks {
                bindings.insert(states[peer].output, &outputs[peer]);
                bindings.insert(states[peer].flags, &flags[peer]);
                bindings.insert(states[peer].epoch, &epochs[peer]);
            }
            Ok(launch::RankCall {
                args: gather_arguments(group, &states, 100, elements as u64)
                    .map_err(|e| e.0.to_string())?,
                bindings: Bindings { buffers: bindings },
            })
        })
    };
    // Poisons every output and stages call `call`'s shards in `targets`
    // (the inputs, or the producers' fresh copies), with every rank idle.
    let prepare = |call: usize, targets: &[&DeviceBuffer]| -> LabResult<()> {
        poison::synchronize(sessions)?;
        for (rank, session) in sessions.iter().enumerate() {
            targets[rank]
                .copy_from_bytes(&bytes(&shard(call, rank, elements)), &session.stream)
                .map_err(err)?;
            poison::poison(session, &[&outputs[rank]])?;
        }
        poison::synchronize(sessions)
    };
    let check = |phase: &str, call: usize| -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(err)?;
        }
        for (rank, session) in sessions.iter().enumerate() {
            let label = format!("size {size} {phase} call {call} rank {rank}");
            let written = poison::read(session, &outputs[rank])?;
            poison::check(
                &label,
                &written,
                &WriteMap::of(DType::F32).byte_prefix(buffers.output_bytes as usize)?,
                Fill::Pattern(&[POISON]),
            )?;
            let group = GroupRank::of(rank as u32, ranks as u32, size as u32)
                .map_err(|e| e.0.to_string())?;
            let expected = group
                .members()
                .flat_map(|member| shard(call, member as usize, elements))
                .collect::<Vec<_>>();
            let actual = session.download_f32(&outputs[rank], size * elements)?;
            if actual
                .iter()
                .zip(&expected)
                .any(|(a, e)| a.to_bits() != e.to_bits())
            {
                return Err(format!("{label}: the gathered shards differ"));
            }
        }
        Ok(())
    };
    let inputs_ref = inputs.iter().collect::<Vec<_>>();
    for call in 0..2 {
        prepare(call, &inputs_ref)?;
        gather(None)?;
        check("eager", call)?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| gather(Some(rank)))?;
    for call in 2..4 {
        prepare(call, &inputs_ref)?;
        graphs.launch(sessions)?;
        check("graph replay", call)?;
    }
    let fresh = producers.iter().map(|p| &p.fresh).collect::<Vec<_>>();
    for call in 4..6 {
        prepare(call, &fresh)?;
        for (producer, session) in producers.iter().zip(sessions) {
            producer.launch(session)?;
        }
        gather(None)?;
        check("PDL", call)?;
    }
    Ok(())
}

/// Block `d` of topology rank `source`'s send buffer at call `call`:
/// finite, and distinct per call, topology source, destination and element
/// (below 2^20), so a block from another group's rank of the same group
/// rank differs.
fn block(call: usize, source: usize, destination: usize, floats: usize) -> Vec<f32> {
    let tag = ((call * 64 + source * 8 + destination) as u32) << 20;
    (0..floats)
        .map(|i| f32::from_bits(0x2000_0000 + tag + i as u32))
        .collect()
}

/// The launch values of resolved arguments: addresses pass as 64-bit
/// scalars, pointer arrays as themselves.
fn kernel_arguments(list: &ArgumentList) -> Vec<KernelArgument<'_>> {
    list.0
        .iter()
        .map(|argument| match argument {
            Argument::DeviceBuffer(address) | Argument::U64(address) => {
                KernelArgument::U64(*address)
            }
            Argument::U32(value) => KernelArgument::U32(*value),
            Argument::DevicePointerArray(addresses) => {
                KernelArgument::DevicePointerArray(addresses)
            }
            other => unreachable!("the exchange takes no {other:?}"),
        })
        .collect()
}

fn run_exchange(sessions: &[Session], size: usize, floats: usize) -> LabResult<()> {
    let ranks = sessions.len();
    let buffers = exchange_buffers(size as u32, floats as u64).map_err(|e| e.0.to_string())?;
    let config = launch::config(exchange_plan())?;
    let send = sessions
        .iter()
        .map(|session| session.allocate(buffers.send_bytes as usize))
        .collect::<LabResult<Vec<_>>>()?;
    let receive = sessions
        .iter()
        .map(|session| session.allocate(buffers.receive_bytes as usize))
        .collect::<LabResult<Vec<_>>>()?;
    let flags = sessions
        .iter()
        .map(|session| zeroed(session, buffers.flag_bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let epochs = sessions
        .iter()
        .map(|session| zeroed(session, buffers.epoch_bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let producers = sessions
        .iter()
        .zip(&send)
        .map(|(session, input)| Producer::new(session, input, size * floats))
        .collect::<LabResult<Vec<_>>>()?;
    let address = |buffer: &DeviceBuffer| -> LabResult<u64> {
        buffer.device_address_at(0, buffer.byte_len()).map_err(err)
    };
    let states = (0..ranks)
        .map(|rank| {
            Ok(ExchangeState {
                receive: address(&receive[rank])?,
                flags: address(&flags[rank])?,
                epoch: address(&epochs[rank])?,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let groups = (0..ranks)
        .map(|rank| {
            GroupRank::of(rank as u32, ranks as u32, size as u32).map_err(|e| e.0.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let arguments = (0..ranks)
        .map(|rank| {
            exchange_arguments(groups[rank], &states, address(&send[rank])?, floats as u64)
                .map_err(|e| e.0.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let function = modules.function(EXCHANGE_SYMBOL)?;
    let exchange = |only: Option<usize>| {
        launch::submit_collective(sessions, &function, config, only, |rank| {
            Ok(kernel_arguments(&arguments[rank]))
        })
    };
    // Poisons every receive buffer and stages call `call`'s send blocks in
    // `targets` (the send buffers, or the producers' fresh copies).
    let prepare = |call: usize, targets: &[&DeviceBuffer]| -> LabResult<()> {
        poison::synchronize(sessions)?;
        for (rank, session) in sessions.iter().enumerate() {
            let staged = (0..size)
                .flat_map(|destination| block(call, rank, destination, floats))
                .collect::<Vec<_>>();
            targets[rank]
                .copy_from_bytes(&bytes(&staged), &session.stream)
                .map_err(err)?;
            poison::poison(session, &[&receive[rank]])?;
        }
        poison::synchronize(sessions)
    };
    let check = |phase: &str, call: usize| -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(err)?;
        }
        for (rank, session) in sessions.iter().enumerate() {
            let label = format!("exchange size {size} {phase} call {call} rank {rank}");
            let owner = groups[rank].rank as usize;
            let view = exchange_receive_offset(owner as u32, size as u32, floats as u64) as usize;
            let written = poison::read(session, &receive[rank])?;
            poison::check(
                &label,
                &written,
                &WriteMap::of(DType::F32)
                    .byte_ranges(std::iter::once(view..view + size * floats * 4))?,
                Fill::Pattern(&[POISON]),
            )?;
            let expected = groups[rank]
                .members()
                .flat_map(|source| block(call, source as usize, owner, floats))
                .collect::<Vec<_>>();
            let actual = written[view..view + size * floats * 4]
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>();
            if actual
                .iter()
                .zip(&expected)
                .any(|(a, e)| a.to_bits() != e.to_bits())
            {
                return Err(format!("{label}: the received blocks differ"));
            }
        }
        Ok(())
    };
    let send_ref = send.iter().collect::<Vec<_>>();
    for call in 0..2 {
        prepare(call, &send_ref)?;
        exchange(None)?;
        check("eager", call)?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| exchange(Some(rank)))?;
    for call in 2..4 {
        prepare(call, &send_ref)?;
        graphs.launch(sessions)?;
        check("graph replay", call)?;
    }
    let fresh = producers.iter().map(|p| &p.fresh).collect::<Vec<_>>();
    for call in 4..6 {
        prepare(call, &fresh)?;
        for (producer, session) in producers.iter().zip(sessions) {
            producer.launch(session)?;
        }
        exchange(None)?;
        check("PDL", call)?;
    }
    Ok(())
}

/// Heads and latent width of the merged partials (`shapes::GEOMETRY`).
const HEADS: usize = 16;
const WIDTH: usize = 512;

/// The partials topology rank `source` sends owner `owner` at call `call`,
/// `rows` rows of `HEADS` heads: `m`, `l`, then `o`, distinct per call,
/// source and owner. Every fifth pair of a source is empty (`m = -inf`,
/// `l = 0`, `o = 0`), as a rank with no key of the row writes it.
fn partial_block(call: usize, source: usize, owner: usize, rows: usize) -> Vec<f32> {
    let pairs = rows * HEADS;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ ((call * 64 + source * 8 + owner) as u64) << 40;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let empty = |pair: usize| (pair + source).is_multiple_of(5);
    let mut out = Vec::with_capacity(pairs * (WIDTH + 2));
    out.extend((0..pairs).map(|pair| {
        if empty(pair) {
            f32::NEG_INFINITY
        } else {
            4.0 * next()
        }
    }));
    let l = (0..pairs)
        .map(|pair| {
            if empty(pair) {
                0.0
            } else {
                0.5 + 2.0 * next().abs()
            }
        })
        .collect::<Vec<_>>();
    out.extend(&l);
    for (pair, &l) in l.iter().enumerate() {
        out.extend((0..WIDTH).map(|_| if empty(pair) { 0.0 } else { l * next() }));
    }
    out
}

fn address(buffer: &DeviceBuffer) -> LabResult<u64> {
    buffer.device_address_at(0, buffer.byte_len()).map_err(err)
}

/// Every topology rank's state of one gather call site: its output, flags
/// and epoch.
struct Site {
    outputs: Vec<DeviceBuffer>,
    flags: Vec<DeviceBuffer>,
    epochs: Vec<DeviceBuffer>,
}

impl Site {
    fn new(sessions: &[Session], size: usize, words: usize) -> LabResult<Site> {
        let bytes = gather_buffers(size as u32, words as u64).map_err(|e| e.0.to_string())?;
        Ok(Site {
            outputs: sessions
                .iter()
                .map(|session| session.allocate(bytes.output_bytes as usize))
                .collect::<LabResult<_>>()?,
            flags: sessions
                .iter()
                .map(|session| zeroed(session, bytes.flag_bytes))
                .collect::<LabResult<_>>()?,
            epochs: sessions
                .iter()
                .map(|session| zeroed(session, bytes.epoch_bytes))
                .collect::<LabResult<_>>()?,
        })
    }

    fn states(&self) -> LabResult<Vec<GatherState>> {
        (0..self.outputs.len())
            .map(|rank| {
                Ok(GatherState {
                    output: address(&self.outputs[rank])?,
                    flags: address(&self.flags[rank])?,
                    epoch: address(&self.epochs[rank])?,
                })
            })
            .collect()
    }
}

fn run_owner_gathers(sessions: &[Session], size: usize, rows: usize) -> LabResult<()> {
    let ranks = sessions.len();
    let floats = rows * HEADS * (WIDTH + 2);
    let groups = (0..ranks)
        .map(|rank| {
            GroupRank::of(rank as u32, ranks as u32, size as u32).map_err(|e| e.0.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let send = sessions
        .iter()
        .map(|session| session.allocate(size * floats * 4))
        .collect::<LabResult<Vec<_>>>()?;
    let partials = Site::new(sessions, size, floats)?;
    let barrier = Site::new(sessions, size, 1)?;
    let barrier_inputs = sessions
        .iter()
        .map(|session| zeroed(session, 4))
        .collect::<LabResult<Vec<_>>>()?;
    let producers = sessions
        .iter()
        .zip(&send)
        .map(|(session, input)| Producer::new(session, input, size * floats))
        .collect::<LabResult<Vec<_>>>()?;
    let partial_states = partials.states()?;
    let barrier_states = barrier.states()?;
    // Per rank: each owner's gather, then the barrier.
    let mut calls = Vec::new();
    for rank in 0..ranks {
        let mut rank_calls = Vec::new();
        for owner in 0..size {
            let input = send[rank]
                .device_address_at(owner * floats * 4, floats * 4)
                .map_err(err)?;
            rank_calls.push(
                owner_gather_arguments(
                    groups[rank],
                    &partial_states,
                    input,
                    floats as u64,
                    owner as u32,
                )
                .map_err(|e| e.0.to_string())?,
            );
        }
        rank_calls.push(
            gather_arguments(
                groups[rank],
                &barrier_states,
                address(&barrier_inputs[rank])?,
                1,
            )
            .map_err(|e| e.0.to_string())?,
        );
        calls.push(rank_calls);
    }
    let gather_config = launch::config(gather_plan(floats as u64).map_err(|e| e.0.to_string())?)?;
    let barrier_config = launch::config(gather_plan(1).map_err(|e| e.0.to_string())?)?;
    let modules = CollectiveModules::load(sessions, "comm_sm120")?;
    let function = modules.function(ALLGATHER_SYMBOL)?;
    let step = |only: Option<usize>| -> LabResult<()> {
        for index in 0..=size {
            let config = if index < size {
                gather_config
            } else {
                barrier_config
            };
            launch::submit_collective(sessions, &function, config, only, |rank| {
                Ok(kernel_arguments(&calls[rank][index]))
            })?;
        }
        Ok(())
    };
    // The row exchange of the same send buffers, for the merge comparison.
    let exchange = exchange_buffers(size as u32, floats as u64).map_err(|e| e.0.to_string())?;
    let exchanged = sessions
        .iter()
        .map(|session| session.allocate(exchange.receive_bytes as usize))
        .collect::<LabResult<Vec<_>>>()?;
    let exchange_flags = sessions
        .iter()
        .map(|session| zeroed(session, exchange.flag_bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let exchange_epochs = sessions
        .iter()
        .map(|session| zeroed(session, exchange.epoch_bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let exchange_states = (0..ranks)
        .map(|rank| {
            Ok(ExchangeState {
                receive: address(&exchanged[rank])?,
                flags: address(&exchange_flags[rank])?,
                epoch: address(&exchange_epochs[rank])?,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let exchange_calls = (0..ranks)
        .map(|rank| {
            exchange_arguments(
                groups[rank],
                &exchange_states,
                address(&send[rank])?,
                floats as u64,
            )
            .map_err(|e| e.0.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let exchange_function = modules.function(EXCHANGE_SYMBOL)?;
    let exchange_config = launch::config(exchange_plan())?;
    // The merge of each rank's heads.
    let merge = shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.kind == Kind::PartialMerge { ranks: size as u32 })
        .ok_or("no lab partial merge entry")?;
    let layout = MergeLayout::exchanged(rows as u32, HEADS as u32);
    merge
        .check_merge(rows as u32, layout)
        .map_err(|e| e.0.to_string())?;
    let merge_config = launch::config(merge.plan(rows as u32).map_err(|e| e.0.to_string())?)?;
    let merges = sessions
        .iter()
        .map(|session| {
            session
                .module(shapes::CUBIN)?
                .function(&merge.symbol())
                .map_err(err)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let live = sessions
        .iter()
        .map(|session| {
            let buffer = session.allocate(4)?;
            buffer
                .copy_from_bytes(&(rows as u32).to_ne_bytes(), &session.stream)
                .map_err(err)?;
            Ok(buffer)
        })
        .collect::<LabResult<Vec<_>>>()?;
    // The merge of every rank's heads, over its owner-gathered blocks or
    // over the exchange's view.
    let merged = |via_exchange: bool| -> LabResult<Vec<Vec<u8>>> {
        sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| {
                let out = session.allocate(rows * HEADS * WIDTH * 2)?;
                poison::poison(session, &[&out])?;
                let owner = groups[rank].rank as usize;
                let base = if !via_exchange {
                    address(&partials.outputs[rank])?
                } else {
                    exchanged[rank]
                        .device_address_at(
                            exchange_receive_offset(owner as u32, size as u32, floats as u64)
                                as usize,
                            size * floats * 4,
                        )
                        .map_err(err)?
                };
                merges[rank]
                    .launch(
                        &session.stream,
                        merge_config,
                        &[
                            KernelArgument::U64(base),
                            KernelArgument::Buffer(&live[rank]),
                            KernelArgument::Buffer(&out),
                            KernelArgument::U32(layout.rank_rows),
                            KernelArgument::U32(layout.input_heads),
                            KernelArgument::U32(layout.head_offset),
                        ],
                    )
                    .map_err(err)?;
                session.stream.synchronize().map_err(err)?;
                poison::read(session, &out)
            })
            .collect()
    };
    let prepare = |call: usize, targets: &[&DeviceBuffer]| -> LabResult<()> {
        poison::synchronize(sessions)?;
        for (rank, session) in sessions.iter().enumerate() {
            let staged = (0..size)
                .flat_map(|owner| partial_block(call, rank, owner, rows))
                .collect::<Vec<_>>();
            targets[rank]
                .copy_from_bytes(&bytes(&staged), &session.stream)
                .map_err(err)?;
            poison::poison(session, &[&partials.outputs[rank], &barrier.outputs[rank]])?;
        }
        poison::synchronize(sessions)
    };
    let check = |phase: &str, call: usize| -> LabResult<()> {
        for session in sessions {
            session.stream.synchronize().map_err(err)?;
        }
        for (rank, session) in sessions.iter().enumerate() {
            let label = format!("owner gathers size {size} {phase} call {call} rank {rank}");
            let owner = groups[rank].rank as usize;
            let written = poison::read(session, &partials.outputs[rank])?;
            poison::check(
                &label,
                &written,
                &WriteMap::of(DType::F32).byte_prefix(written.len())?,
                Fill::Pattern(&[POISON]),
            )?;
            let expected = groups[rank]
                .members()
                .flat_map(|source| partial_block(call, source as usize, owner, rows))
                .collect::<Vec<_>>();
            if bytes(&expected) != written {
                return Err(format!("{label}: the received blocks differ"));
            }
            // Every member's barrier word arrived.
            let words = poison::read(session, &barrier.outputs[rank])?;
            poison::check(
                &label,
                &words,
                &WriteMap::of(DType::F32).byte_prefix(words.len())?,
                Fill::Pattern(&[POISON]),
            )?;
        }
        // The same partials through the row exchange merge identically.
        poison::synchronize(sessions)?;
        launch::submit_collective(
            sessions,
            &exchange_function,
            exchange_config,
            None,
            |rank| Ok(kernel_arguments(&exchange_calls[rank])),
        )?;
        poison::synchronize(sessions)?;
        let owned = merged(false)?;
        let via_exchange = merged(true)?;
        for (rank, (owned, via_exchange)) in owned.iter().zip(&via_exchange).enumerate() {
            let label = format!("merge size {size} {phase} call {call} rank {rank}");
            poison::check(
                &label,
                owned,
                &WriteMap::of(DType::Bf16).byte_prefix(owned.len())?,
                Fill::Pattern(&[POISON]),
            )?;
            if owned != via_exchange {
                return Err(format!(
                    "{label}: the owner-gathered merge differs from the exchanged one"
                ));
            }
        }
        Ok(())
    };
    let send_ref = send.iter().collect::<Vec<_>>();
    for call in 0..2 {
        prepare(call, &send_ref)?;
        step(None)?;
        check("eager", call)?;
    }
    let graphs = RankReplay::new(sessions, |rank, _| step(Some(rank)))?;
    for call in 2..4 {
        prepare(call, &send_ref)?;
        graphs.launch(sessions)?;
        check("graph replay", call)?;
    }
    let fresh = producers.iter().map(|p| &p.fresh).collect::<Vec<_>>();
    for call in 4..6 {
        prepare(call, &fresh)?;
        for (producer, session) in producers.iter().zip(sessions) {
            producer.launch(session)?;
        }
        step(None)?;
        check("PDL", call)?;
    }
    Ok(())
}

/// Each owner's gathers deliver its group's partial blocks compact and
/// source-major, behind the group barrier, eager, from a graph and behind a
/// PDL producer, and the merge over them equals the merge over the row
/// exchange's view.
#[test]
fn owner_gathers_deliver_each_owner_its_blocks() {
    gpu_test_multi(512 << 20, |sessions| {
        let ranks = sessions.len();
        // A 32-row decode bucket and a single row.
        for rows in [32, 1] {
            if ranks >= 4 && ranks.is_multiple_of(2) {
                run_owner_gathers(sessions, 2, rows)?;
            }
            if ranks == 2 || ranks == 4 {
                run_owner_gathers(sessions, ranks, rows)?;
            }
        }
        Ok(())
    })
    .expect("owner gathers deliver each owner its blocks");
}

#[test]
fn group_exchanges_deliver_each_owner_its_blocks() {
    gpu_test_multi(256 << 20, |sessions| {
        let ranks = sessions.len();
        // 32 decode rows of 16 heads of (512 + 2) floats, an unaligned block
        // and a single float.
        for floats in [32 * 16 * 514, 1001, 1] {
            if ranks >= 4 && ranks.is_multiple_of(2) {
                run_exchange(sessions, 2, floats)?;
            }
            if ranks <= 8 {
                run_exchange(sessions, ranks, floats)?;
            }
        }
        Ok(())
    })
    .expect("group exchanges deliver each owner its blocks");
}

#[test]
fn group_all_gathers_keep_to_their_groups() {
    gpu_test_multi(256 << 20, |sessions| {
        let ranks = sessions.len();
        // 32 decode rows of 16 heads of BF16 latent queries (as words), and
        // 32 rows of a group's 32 heads of partials.
        for elements in [32 * 16 * 256, 32 * 32 * 514, 1] {
            if ranks >= 4 && ranks.is_multiple_of(2) {
                run_groups(sessions, 2, elements)?;
            }
            if ranks <= 8 {
                run_groups(sessions, ranks, elements)?;
            }
        }
        Ok(())
    })
    .expect("group all-gathers keep to their groups");
}
