//! The phased group gather (`kernels_sm120::context_parallel::phased`)
//! against the transports it replaces: its all-gather of one shard equals
//! the shared all-gather (`group::gather_arguments`) bit for bit, and its
//! exchange of a destination-major buffer equals the owner gathers
//! (`group::owner_gather_arguments`, owners 0..size, then the one-word
//! group barrier) bit for bit, in two groups of two and one group of every
//! rank.
//!
//! Each case takes six calls on one state: two eager, two replays of one
//! captured graph and two eager calls behind a PDL producer that holds
//! every input at NaN poison until after its tail. Before every call the
//! outputs are poisoned and the inputs hold that call's own words, while the
//! flags and epochs carry over, so a call that writes nothing, writes a
//! stale block, writes outside its slots or reads its input before its
//! dependency wait fails. The phased state's flags and epoch are the shared
//! all-gather's buffers of the same size (`group::gather_buffers`), as the
//! bundles allocate them. The eager calls also run the replaced transport
//! on the same inputs and compare the outputs byte for byte.
//!
//! `phased_transfers_at_served_sizes` times both transports at the served
//! latent shape's block sizes (`context_parallel_shapes.rs`: 2048-row
//! prefill pieces and 32-row decode buckets) and writes receipts; it
//! asserts nothing about speed.
//!
//! Run with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release --test context_parallel_phased`.

use std::collections::BTreeMap;

use kernel_core::contract::{Argument, ArgumentList, DType};
use kernel_lab::families::context_parallel_phased::CUBIN;
use kernel_lab::launch::{self, CollectiveFunction, CollectiveModules};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi, time_multi_graph};
use kernels_sm120::context_parallel::group::{
    ALLGATHER_SYMBOL, GatherState, GroupRank, gather_arguments, gather_buffers, gather_plan,
    owner_gather_arguments,
};
use kernels_sm120::context_parallel::phased;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument};
use serde_json::json;

/// Words of the group barrier: an all-gather of one word.
const BARRIER_WORDS: u64 = 1;

fn err(e: impl ToString) -> String {
    e.to_string()
}

fn refusal(e: kernel_core::contract::Refusal) -> String {
    e.0.to_string()
}

/// The words topology rank `source` sends group rank `destination` at call
/// `call`: finite F32 bit patterns, distinct per call, source, destination
/// and word (below 2^20 words).
fn words_of(call: usize, source: usize, destination: usize, words: usize) -> Vec<u32> {
    let tag = ((call * 64 + source * 8 + destination) as u32) << 20;
    (0..words).map(|i| 0x2000_0000 + tag + i as u32).collect()
}

fn bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn zeroed(session: &Session, bytes: u64) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes as usize)?;
    buffer
        .copy_from_bytes(&vec![0; bytes as usize], &session.stream)
        .map_err(err)?;
    Ok(buffer)
}

fn address(buffer: &DeviceBuffer) -> LabResult<u64> {
    buffer.device_address_at(0, buffer.byte_len()).map_err(err)
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
            other => unreachable!("a gather takes no {other:?}"),
        })
        .collect()
}

/// The PDL producer probe on one rank, with one staged operand: it
/// overwrites `input` with NaN poison, triggers its dependents, and after
/// its tail copies `fresh` (the call's words) into `input`.
struct Producer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    live: DeviceBuffer,
    descriptor: DeviceBuffer,
    fresh: DeviceBuffer,
}

impl Producer {
    fn new(session: &Session, input: &DeviceBuffer, words: usize) -> LabResult<Producer> {
        let module = session.module("probe_pdl_prefetch_sm120")?;
        let fresh = session.allocate(words * 4)?;
        let mut descriptor = Vec::new();
        for buffer in [&fresh, input] {
            descriptor.extend_from_slice(
                &buffer
                    .device_address_at(0, words * 4)
                    .map_err(err)?
                    .to_ne_bytes(),
            );
        }
        descriptor.extend_from_slice(&(words as u32).to_ne_bytes());
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

/// Every topology rank's state of one call site: its output (`size`
/// blocks of `words`), and flags and epoch of the shared all-gather's sizes
/// for `words` words.
struct Site {
    outputs: Vec<DeviceBuffer>,
    flags: Vec<DeviceBuffer>,
    epochs: Vec<DeviceBuffer>,
}

impl Site {
    fn new(sessions: &[Session], size: usize, words: usize) -> LabResult<Site> {
        let buffers = gather_buffers(size as u32, words as u64).map_err(refusal)?;
        phased::check_state(size as u32, buffers.flag_bytes, buffers.epoch_bytes)
            .map_err(refusal)?;
        Ok(Site {
            outputs: sessions
                .iter()
                .map(|session| session.allocate(buffers.output_bytes as usize))
                .collect::<LabResult<_>>()?,
            flags: sessions
                .iter()
                .map(|session| zeroed(session, buffers.flag_bytes))
                .collect::<LabResult<_>>()?,
            epochs: sessions
                .iter()
                .map(|session| zeroed(session, buffers.epoch_bytes))
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

/// How a case moves its words.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    /// One shard to every member; the reference is the shared all-gather.
    Gather,
    /// Block d to group rank d; the reference is the owner gathers.
    Exchange,
}

/// The input words of topology rank `rank` at call `call`: one shard, or
/// `size` destination-major blocks.
fn staged(mode: Mode, call: usize, rank: usize, size: usize, words: usize) -> Vec<u32> {
    match mode {
        Mode::Gather => words_of(call, rank, 0, words),
        Mode::Exchange => (0..size)
            .flat_map(|destination| words_of(call, rank, destination, words))
            .collect(),
    }
}

/// What topology rank `rank`'s output holds after call `call`: slot q holds
/// group rank q's shard (gather) or its block for this rank (exchange).
fn expected(mode: Mode, call: usize, group: GroupRank, words: usize) -> Vec<u32> {
    group
        .members()
        .flat_map(|source| match mode {
            Mode::Gather => words_of(call, source as usize, 0, words),
            Mode::Exchange => words_of(call, source as usize, group.rank as usize, words),
        })
        .collect()
}

/// The phased launches and the replaced transport's on the same inputs,
/// for every topology rank.
struct Case<'a> {
    sessions: &'a [Session],
    mode: Mode,
    size: usize,
    words: usize,
    groups: Vec<GroupRank>,
    inputs: Vec<DeviceBuffer>,
    phased: Site,
    shared: Site,
    barrier: Site,
    barrier_inputs: Vec<DeviceBuffer>,
    phased_function: CollectiveFunction,
    shared_function: CollectiveFunction,
    _modules: [CollectiveModules; 2],
}

impl<'a> Case<'a> {
    fn new(sessions: &'a [Session], mode: Mode, size: usize, words: usize) -> LabResult<Self> {
        let ranks = sessions.len();
        let groups = (0..ranks)
            .map(|rank| GroupRank::of(rank as u32, ranks as u32, size as u32).map_err(refusal))
            .collect::<LabResult<Vec<_>>>()?;
        let blocks = if mode == Mode::Exchange { size } else { 1 };
        let phased_modules = CollectiveModules::load(sessions, CUBIN)?;
        let shared_modules = CollectiveModules::load(sessions, "comm_sm120")?;
        Ok(Self {
            sessions,
            mode,
            size,
            words,
            groups,
            inputs: sessions
                .iter()
                .map(|session| session.allocate(blocks * words * 4))
                .collect::<LabResult<_>>()?,
            phased: Site::new(sessions, size, words)?,
            shared: Site::new(sessions, size, words)?,
            barrier: Site::new(sessions, size, 1)?,
            barrier_inputs: sessions
                .iter()
                .map(|session| zeroed(session, 4))
                .collect::<LabResult<_>>()?,
            phased_function: phased_modules.function(phased::SYMBOL)?,
            shared_function: shared_modules.function(ALLGATHER_SYMBOL)?,
            _modules: [phased_modules, shared_modules],
        })
    }

    /// The phased call, on every rank or on `only`.
    fn phased(&self, only: Option<usize>) -> LabResult<()> {
        let states = self.phased.states()?;
        let arguments = (0..self.sessions.len())
            .map(|rank| {
                let input = address(&self.inputs[rank])?;
                let words = self.words as u64;
                match self.mode {
                    Mode::Gather => {
                        phased::gather_arguments(self.groups[rank], &states, input, words)
                    }
                    Mode::Exchange => {
                        phased::exchange_arguments(self.groups[rank], &states, input, words)
                    }
                }
                .map_err(refusal)
            })
            .collect::<LabResult<Vec<_>>>()?;
        launch::submit_collective(
            self.sessions,
            &self.phased_function,
            launch::config(phased::plan())?,
            only,
            |rank| Ok(kernel_arguments(&arguments[rank])),
        )
    }

    /// The replaced transport on every rank or on `only`: the shared
    /// all-gather, or the owner gathers then the group barrier.
    fn shared(&self, only: Option<usize>) -> LabResult<()> {
        let states = self.shared.states()?;
        let words = self.words as u64;
        let config = launch::config(gather_plan(words).map_err(refusal)?)?;
        let calls = match self.mode {
            Mode::Gather => vec![None],
            Mode::Exchange => (0..self.size as u32).map(Some).collect(),
        };
        for owner in calls {
            let arguments = (0..self.sessions.len())
                .map(|rank| {
                    let group = self.groups[rank];
                    let input = address(&self.inputs[rank])?;
                    match owner {
                        None => gather_arguments(group, &states, input, words),
                        Some(owner) => owner_gather_arguments(
                            group,
                            &states,
                            input + u64::from(owner) * words * 4,
                            words,
                            owner,
                        ),
                    }
                    .map_err(refusal)
                })
                .collect::<LabResult<Vec<_>>>()?;
            launch::submit_collective(
                self.sessions,
                &self.shared_function,
                config,
                only,
                |rank| Ok(kernel_arguments(&arguments[rank])),
            )?;
        }
        if self.mode == Mode::Exchange {
            let states = self.barrier.states()?;
            let arguments = (0..self.sessions.len())
                .map(|rank| {
                    gather_arguments(
                        self.groups[rank],
                        &states,
                        address(&self.barrier_inputs[rank])?,
                        BARRIER_WORDS,
                    )
                    .map_err(refusal)
                })
                .collect::<LabResult<Vec<_>>>()?;
            launch::submit_collective(
                self.sessions,
                &self.shared_function,
                launch::config(gather_plan(BARRIER_WORDS).map_err(refusal)?)?,
                only,
                |rank| Ok(kernel_arguments(&arguments[rank])),
            )?;
        }
        Ok(())
    }

    /// Poisons both transports' outputs and stages call `call`'s words in
    /// `targets` (the inputs, or the producers' fresh copies), with every
    /// rank idle.
    fn prepare(&self, call: usize, targets: &[&DeviceBuffer]) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            targets[rank]
                .copy_from_bytes(
                    &bytes(&staged(self.mode, call, rank, self.size, self.words)),
                    &session.stream,
                )
                .map_err(err)?;
            poison::poison(
                session,
                &[&self.phased.outputs[rank], &self.shared.outputs[rank]],
            )?;
        }
        poison::synchronize(self.sessions)
    }

    /// Every output word of the phased transport (and of the replaced one,
    /// when it ran) is the expected word, bit for bit.
    fn check(&self, phase: &str, call: usize, shared_ran: bool) -> LabResult<()> {
        for session in self.sessions {
            session.stream.synchronize().map_err(err)?;
        }
        let label = |rank: usize| {
            format!(
                "{:?} size {} words {} {phase} call {call} rank {rank}",
                self.mode, self.size, self.words
            )
        };
        let output_bytes = self.size * self.words * 4;
        for (rank, session) in self.sessions.iter().enumerate() {
            let want = bytes(&expected(self.mode, call, self.groups[rank], self.words));
            let written = poison::read(session, &self.phased.outputs[rank])?;
            poison::check(
                &label(rank),
                &written,
                &WriteMap::of(DType::F32).byte_prefix(output_bytes)?,
                Fill::Pattern(&[POISON]),
            )?;
            if written[..output_bytes] != want[..] {
                return Err(format!("{}: the phased output differs", label(rank)));
            }
            if shared_ran {
                let shared = poison::read(session, &self.shared.outputs[rank])?;
                if shared[..output_bytes] != written[..output_bytes] {
                    return Err(format!(
                        "{}: the phased output differs from the replaced transport's",
                        label(rank)
                    ));
                }
            }
        }
        Ok(())
    }

    fn run(&self) -> LabResult<()> {
        let blocks = if self.mode == Mode::Exchange {
            self.size
        } else {
            1
        };
        let producers = self
            .sessions
            .iter()
            .zip(&self.inputs)
            .map(|(session, input)| Producer::new(session, input, blocks * self.words))
            .collect::<LabResult<Vec<_>>>()?;
        let inputs = self.inputs.iter().collect::<Vec<_>>();
        for call in 0..2 {
            self.prepare(call, &inputs)?;
            self.phased(None)?;
            self.shared(None)?;
            self.check("eager", call, true)?;
        }
        let graphs = RankReplay::new(self.sessions, |rank, _| self.phased(Some(rank)))?;
        for call in 2..4 {
            self.prepare(call, &inputs)?;
            graphs.launch(self.sessions)?;
            self.check("graph replay", call, false)?;
        }
        let fresh = producers.iter().map(|p| &p.fresh).collect::<Vec<_>>();
        for call in 4..6 {
            self.prepare(call, &fresh)?;
            for (producer, session) in producers.iter().zip(self.sessions) {
                producer.launch(session)?;
            }
            self.phased(None)?;
            self.check("PDL", call, false)?;
        }
        Ok(())
    }
}

/// The group sizes a run covers: two groups of two (at four or more even
/// ranks) and one group of every rank (up to eight).
fn sizes(ranks: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    if ranks >= 4 && ranks.is_multiple_of(2) {
        sizes.push(2);
    }
    if ranks <= 8 {
        sizes.push(ranks);
    }
    sizes
}

/// Words per rank at the served latent shape: absorbed queries (16 heads
/// of 512 BF16) and (score, pool) pairs (512 kept pools) per row, and
/// partial blocks (16 heads of 512 + 2 floats) per row.
const QUERY_WORDS: usize = 16 * 512 / 2;
const PAIR_WORDS: usize = 512 * 2;
const BLOCK_WORDS: usize = 16 * 514;

#[test]
fn phased_gathers_equal_the_shared_all_gather() {
    gpu_test_multi(1 << 30, |sessions| {
        // One word, an unaligned count, one decode bucket's queries and a
        // 2048-row piece's pairs.
        for words in [1, 1001, 32 * QUERY_WORDS, 2048 * PAIR_WORDS] {
            for size in sizes(sessions.len()) {
                Case::new(sessions, Mode::Gather, size, words)?.run()?;
            }
        }
        Ok(())
    })
    .expect("phased gathers equal the shared all-gather");
}

#[test]
fn phased_exchanges_equal_the_owner_gathers() {
    gpu_test_multi(1 << 30, |sessions| {
        // An unaligned block, and the partial blocks of 1, 32 and 256 rows.
        for words in [1001, BLOCK_WORDS, 32 * BLOCK_WORDS, 256 * BLOCK_WORDS] {
            for size in sizes(sessions.len()) {
                Case::new(sessions, Mode::Exchange, size, words)?.run()?;
            }
        }
        Ok(())
    })
    .expect("phased exchanges equal the owner gathers");
}

/// Median microseconds per call of `calls` captured calls of `work` on
/// every rank (the critical path over ranks).
fn time(
    sessions: &[Session],
    calls: usize,
    work: impl Fn(usize) -> LabResult<()>,
) -> LabResult<f32> {
    let graphs = sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for _ in 0..calls {
                    work(rank)?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let timing = time_multi_graph(sessions, &graphs, calls, 2, 5)?;
    Ok(timing.critical_path.median_ms * 1000.0 / calls as f32)
}

#[test]
fn phased_transfers_at_served_sizes() {
    gpu_test_multi(3 << 30, |sessions| {
        let ranks = sessions.len();
        let cases = [
            ("queries, 2048 rows", Mode::Gather, 2048 * QUERY_WORDS, 4),
            ("pairs, 2048 rows", Mode::Gather, 2048 * PAIR_WORDS, 4),
            ("partials, 2048 rows", Mode::Exchange, 2048 * BLOCK_WORDS, 4),
            ("queries, 32 rows", Mode::Gather, 32 * QUERY_WORDS, 32),
            ("pairs, 32 rows", Mode::Gather, 32 * PAIR_WORDS, 32),
            ("partials, 32 rows", Mode::Exchange, 32 * BLOCK_WORDS, 32),
        ];
        for size in sizes(ranks) {
            for (label, mode, words, calls) in cases {
                let case = Case::new(sessions, mode, size, words)?;
                let inputs = case.inputs.iter().collect::<Vec<_>>();
                case.prepare(0, &inputs)?;
                let phased_us = time(sessions, calls, |rank| case.phased(Some(rank)))?;
                let shared_us = time(sessions, calls, |rank| case.shared(Some(rank)))?;
                println!(
                    "size {size} {label}: {words} words per rank, phased {phased_us:.1} us, \
                     replaced {shared_us:.1} us"
                );
                for (rank, session) in sessions.iter().enumerate() {
                    session.receipt(
                        "context_parallel_phased",
                        BTreeMap::from([
                            ("case".into(), json!(label)),
                            ("size".into(), json!(size)),
                            ("rank".into(), json!(rank)),
                            ("words".into(), json!(words)),
                            ("phased_us_per_call".into(), json!(phased_us)),
                            ("replaced_us_per_call".into(), json!(shared_us)),
                        ]),
                    )?;
                }
            }
        }
        Ok(())
    })
    .expect("phased transfers time at served sizes");
}
